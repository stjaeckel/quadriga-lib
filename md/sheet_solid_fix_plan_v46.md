# Sheet / Solid — Fix Plan v46

**Scope:** the thin-sheet composite and the traced-solid slab series in `ray_mesh_interact.cpp`,
covering both public functions in that file, the three per-ray state words they share with the
tracer loop, and the direction they hand it.

**Status:** the test suite is green, 1297934 assertions in 1074 test cases, in well under a minute on an idle machine.
`src/ray_mesh_interact.cpp` is 5016 lines. Every item below is either
measured, with the diagnostic table that measures it named, or explicitly marked as having no fixture
yet. `test.txt` in the repository root is the baseline a change is diffed against; it was
generated after the last source and fixture change.
The first step of any work on this plan is to regenerate it with
`./build_linux/test_bin > test.txt` and to check that its last line reports 1297934 assertions
in 1074 test cases, before anything is changed. The build directory has to be current: a
`cmake --build build_linux --parallel` that recompiles test objects changes those totals.

`html_docu/quadriga_lib_material_model.md` is out of scope for the work in this plan for now. It is
current only for the mass-law term, the bonded pair, the ports of a mass-law face and the buffer
word's flag and code as they were before the material index became 14 bits and before the code
carried an equivalent thickness; issue 9 lists what is stale in it. Do not edit it as a side effect of another item.

This document stands on its own. Section 1 is the set of settled conventions a change has to work
within, section 2 ranks what is open, section 3 states each open item, section 4 records what is
closed and must not be reopened, and section 5 records limitations that are accepted rather than
open. Section 6 gives the execution order and section 7 the acceptance criteria.

---

## 1. Where things stand

Two functions in an anonymous namespace carry the physics, and each has one job.

**`qd_stack_resolve`, the solver,** builds the composite of a sheet or of an ordered stack of
sheets outright. A sheet has zero world-space thickness, so there is nothing for the tracer to
re-emit into and the internal series has to be summed in one event. It takes an ordered list of
layers between two half-spaces and returns absolute transmit and reflect coefficients on both
ports. Every sheet event of both public functions goes through it, with one layer or with two.

**`qd_slab_resolve`, the series of a traced solid,** adjusts a coefficient the tracer has already
written for a real traversed solid. It returns the per-channel series factor
`S = 1/(1 − rn·rf·φ²)` and nothing else, because the port the caller holds already carries the
Fresnel coefficient of its own interface. Where a mirror of that series is a sheet or an ordered
stack, it asks the solver for the layer's reflection.

`ray_state_update` re-resolves every sheet crossing when medium tracking is on, so in any run with
state its composite is the one that reaches the output and whatever `ray_mesh_interact` produced for
that crossing is discarded. `ray_mesh_interact`'s own composite serves callers without state and
agrees with the redo wherever the function can know what the redo knows: for an isolated sheet, a
glued entry and exit, an ordered stack in all four orientations of its two normals, a corner with
the sheet as the first face, with the lumped columns and under a closed outer pair. What it
cannot know without state is in §5.

Where sections 4 and 5 describe how an item was closed, they name the code as it was then:
"produce mode" and "the core" are today's solver, which was two pieces of code with one physics,
"the matrix" and "the scattering form" were two forms of it, and "correction mode" is
`qd_slab_resolve`.

### The solver

Per medium the admittance is `z = sqrt(ε/μ)` and the index is `n = sqrt(ε·μ)`. Every cosine comes
from the transverse invariant `ε μ (1 − c²)`, which is conserved across every boundary and is fixed
by the incidence cosine in the near medium. Per polarization the tilted admittance is

    η_TE = z·c        η_TM = z/c

which are the groupings `Material::interact_with` is written in: its TE reflection is
`(η₁−η₂)/(η₁+η₂)` exactly, and dividing its TM reflection through by `c₁c₂` gives
`(η₂−η₁)/(η₂+η₁)`. So the library's TM reflection is minus the admittance-form one and its TM
transmission carries an extra `c₁/c₂`.

**One recursion.** Each boundary `a | b` is a two-port in raw Fresnel amplitudes,
`(r, t_ab, t_ba, r_ba = −r)`, in the form `Material::interact_with` writes them, and each
coherent layer is the one-way factor `φ = exp(−j k₀ n c d)`. The stack is evaluated by recursion
from the far end,

    G_j = r_j + t_j·tb_j·G_{j+1}·φ² / (1 + r_j·G_{j+1}·φ²)      T_j = t_j·φ·T_{j+1} / (1 + r_j·G_{j+1}·φ²)

which is the characteristic matrix of the stack, `[cos δ, (i/η) sin δ; i η sin δ, cos δ]` per
layer, in another basis: at `tf = 0`, `B = W_a⁻¹W_b` and `W·P·W⁻¹` is the layer matrix with
`W = [[1, 1], [η, −η]]`. It needs no inverse of a closed boundary (`t = 0` at `tf = −1`) or of an
opaque layer, so it carries no loss cap: an opaque layer transmits at its own level, down to
underflow. At one layer it is the Airy pair,

    r = (r01 + r12 φ²)/(1 + r01 r12 φ²)      t = t01 t12 φ/(1 + r01 r12 φ²)

**What propagates is amplitude, not flux.** Tangential E and H are continuous across every boundary
while normal flux decays inside a lossy layer, so the recursion runs in amplitudes and the conversion
to per-channel normal flux is applied once, at the outer pair, on the transmit port. A closed mass-law
layer cuts the chain, carries no flux, and is crossed by leaks that are flux statements; what the
conversion is then is in *A closed mass-law layer*.

**Nothing is clamped.** A reflection above one seen from a lossy medium is the Fresnel value: the
incident and the reflected wave interfere there, and the net flux towards the face is still
positive. The sum with such a mirror is the exact one, and seen from a lossless medium the two
ports of a wall without `tf` together read at most one. A denominator near zero is not clamped
either: a lossless sheet at a transmission resonance has a small denominator and a small
numerator, and the quotient is finite. Where a denominator is exactly zero the solver declines:
exactly at the critical angle of a lossless sheet, a single point (§5).

**What the solver carries.** `tf` at every boundary, by the two media that form it: the
transformer at the face of a coherent layer, the pad where nothing propagates behind the face,
the pair rule at an open mass-law face (*The lumped material columns of a sheet*). The interface
gains as a product on the transmit port, one per layer entered plus the far medium's where it is
newly entered. On the reflect port a layer's gain is owed by what entered that layer and came
back, once, which a second recursion carries beside `G`,

    Ge_j = r_j + sqrt(g_{j+1})·t_j·tb_j·Ge_{j+1}·φ² / (1 + r_j·G_{j+1}·φ²)

with `g_{j+1}` the interface gain of the layer behind boundary `j`. Its denominator is the one
of `G`, so the series, and with it the transmit port, never reads a gain; putting the gain on the
forward crossing instead makes every internal re-entry pay again and moves the transmit port, by
0.005 to 0.034 in amplitude on two 6 mm layers with 3 dB and 2 dB. With every gain at one `Ge` is
`G` bit for bit. A permittivity resonance is part of its medium's permittivity, at every face and
in the layer phase (*The lumped material columns of a sheet*). The `alpha` column is dB per meter of path, a
real one-way factor on `φ` over the path `d / |Re c|`; where the wave does not advance in the
layer, `Re c = 0`, the layer is opaque to it (issue 3 has what that means for a tunnelling sheet).

**A mass-law layer** is not made coherent. It ends the coherent segment in front of it as
a half-space of its own admittance, which is what its `eta` and `mu` are calibrated for; contributes
one real amplitude factor, `qd_mass_amp`, the loss over the path `d / |Re c|` under a saturation
cap of 400 dB and the mass term; and starts the next
segment as its incidence half-space. Nothing returns across it, so the reflect port is that of the
first segment alone and reads neither `m` nor anything behind the layer. A mass-law layer displaces no host thickness. Mass-law layers with no coherent layer between them are bonded and
pay as one plate; see the mass-law convention below. The transmit port of such a stack is invariant
under layer order where no face of the mass-law layer is closed, at 1.2e-16. With a rigid mass-law
layer, whose faces are closed and pass a `tf` leak, it is invariant by the rule of *A closed
mass-law layer* below: exactly where the layers next to it are lossless, and to the flux splitting
of §5 where one of them is lossy.

**A closed far side.** The solver is told where near | far is past its critical angle,
`far_closed`. Only a sheet glued to a solid reaches it: an ordered stack has one medium on both
sides and is never closed. The far face of a coherent sheet is then the pad of a closed face,
which hands back what it passes of the incident flux; no near-to-far conversion is applied; and
the displaced medium is taken out at the incidence cosine. A mass-law sheet there passes the two
crossings a traced solid of its material would (*A sheet under total reflection of its outer
pair*).

**A wave held in a mass-law medium** takes the chain of held crossings instead of the recursion
(*A wave held undeviated inside a mass-law medium*).

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
- Under total reflection there is no forward Fresnel port; what crosses is the `tf` leak of a
  closed face, which in front of a lossless medium is the flat coefficient `sqrt(1 − R')`.
- Reflection stays within one medium, where the factor is 1, and is untouched.
- Consecutive crossings that share a medium cancel their obliquity factors exactly, but only if both
  are evaluated at the same complex cosine in that medium. That condition holds across a traced
  solid, across a sheet reached from inside a medium, and across the solver, which applies the factor
  once at the outer pair only.

### The incidence cosine

`abs_cos_theta` is a real angle and is what a face hit supplies. A sheet reached from inside a traced
medium has no real incidence cosine: the propagation cosine there is complex and the tracer's
direction carries only its real projection. `SHEET_RESOLVE` hands the solver the recovered value
as its `ci` argument where the recovery ran, and the traced cosine where it declined, and the
solver anchors the transverse invariant in it. Every cosine of the composite follows from that
one: the cosine in each layer, in the far medium and in the displaced medium, the flux
conversion, and the path `d / |Re c|` of the `alpha` column and of a mass-law layer's loss.
Everything is analytic in the cosine, so a real value reproduces the angle path.

The mass term does not read the recovered cosine. The surface-mass surrogate
and its angle factor are lumped quantities the material table is calibrated on, not wave quantities,
and `medium_gain` takes them as real by construction, which is the same reasoning that excludes a
mass-law slab from the exact in-medium branch in `SCALE_BY_MEDIUM`. The angle factor is the cosine
the wave has in air, which the solver takes as its argument `mass_cos`; see the mass-law convention. The two interface amplitudes of a mass-law layer are wave quantities and do carry
the recovered cosine; where the sheet is closed they are the pad of its near face and its reverse
behind the far face (*A closed mass-law layer*).

`qd_slab_resolve` takes the exact in-slab cosine of the traced solid as `ct_slab_exact` where the
recovery ran and the real traced angle otherwise, and evaluates both mirrors and the propagation
factor at it.

### The glued-sheet flux conversion

The solver applies the near-to-far obliquity factor once, on the transmit port, so
a glued sheet reports the same energy as the equivalent traced entry. The gate is
`!mat_far.same_as(mat_near)`, so only the glued cases reach it: `sheet_case 2` (entry,
`air | sheet | solid`) and `sheet_case 3` (exit, `solid | sheet | air`). An isolated sheet, a
collapsed duplicate, an ordered stack in one medium and a sheet embedded in a medium all have
`near == far` and take the factor as exactly 1. A closed mass-law sheet does not take it either,
glued or not: its two leaks are flux statements already (*A closed mass-law layer*).

The composite is the Airy pair, or the matrix pair for more than one layer, in raw amplitudes, times
that conversion once, divided by the propagation of the medium the stack displaces. The numerator of
the conversion is evaluated at the cosine in the far medium, written from the near-side sine directly
rather than by chaining through the layers. Snell is transitive, so the layers drop out of it and the
far cosine is the same quantity a bare near|far crossing would have used. The denominator is the
incidence-side normal flux `Re(z_n c_n)` per channel, and it has to be the same real number the
crossing that entered the near medium put in its numerator, or the pair does not telescope. The
reflect port turns around in the near medium, where the ratio is 1, and carries no conversion.

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
`EXACT_IN_SLAB_COSINE` inverts the map itself: Newton on
`η²u = t²·Re(sqrt(1 − ε_r u))²` with `u = sin²θᵢ`, starting from the real-index guess and run until
the step is below rounding, `|Δu| ≤ 1e-15·u`, with at least three steps and at most fifty. It reaches
the exact complex cosine to machine precision. Three steps are enough wherever real-index Snell has a
solution near the root. In a lossy medium rarer than the one the wave came from it has none past
`sqrt|ε_r|·sin θ = 1`, and towards an invariant above one the start is far as well; there the
solve takes more. Permeability is carried: the two ends of `ε_r` are `ε·μ`
products, so `η²` is the same `snell²` that `qd_refract` used.

It declines, returning zero, when: the wave was not bent at the face behind it, which is read off
the traced and the geometric cosine differing by more than 1e-6 or, where they coincide, off the
bent flag of the current-medium word (see the two tracer modes and the state words); the current
medium is not a real material; the previous face is not parallel; or the
entry and slab media are matched. Where it declines, every consumer keeps the real projection and
nothing moves.

**The hit code's total-reflection bit does not gate it.** That bit is `ray_mesh_interact`'s verdict
on the face ahead of the ray at the geometric angle. With the undeviated tracer a ray inside a solid
travels at its external angle, so the bit is set from `sqrt|ε|·sin θ_ext ≥ 1` on a face the wave
crosses: from 53.4° for `eps 1.50 − 0.40j`, from 25.8° for `eps 5.24 − 0.63j`, from 19.4° for
`eps 9.0 − 1.20j`, as codes 33 for a bare exit and 101 for a glued one. What the recovery inverts is
the bend at the face behind the ray, which the bit says nothing about. A wave that was not bent
there, a leak, holds its geometric direction and is declined by the first gate. `ray_state_update`
re-derives the verdict it reports at the corrected angle, in `REPLACE_BY_VBS` and in
`SHEET_RESOLVE`; apart from that the bit of the hit code is only copied into the resolved type,
where it stands on the events those two do not touch, an entry from air among them.

**The first gate asks whether the wave was bent, and the cosines are only one way to know.**
With the undeviated tracer a bent wave travels at a direction that differs from its segment, and
the difference says so. With the refracting tracer the segment is the bent direction and the
cosines coincide, as they do for a leak; there the event that entered the medium recorded what it
did in the bent flag. See the two tracer modes below.

**Behind a sheet the entry medium is not named, and does not have to be.** `mtl_prev` names the
medium the current leg was entered from, except after a sheet event, where `SHEET_EVENT` wrote the
sheet into it. The sheet is not the entry medium: Snell is transitive and it drops out, and the
medium before it is one level further back than the state reaches. With `p = ε₁ sin²θ₁` the equation
the recovery solves is

    |p| / |ε₂| = t² · Re(sqrt(1 − p/ε₂))²

and for a lossless entry medium `p` is real and `ε₁` appears nowhere else, so the recovered cosine
is the same for every lossless medium the wave could have come from. Where `mtl_prev` holds a sheet
the solve therefore runs with a unit entry permittivity, where the unknown is the transverse
invariant itself, and without the bound `sin² ≤ 1`, which belongs to air and not to the invariant.
That is exact for a glued entry from air, for the far side of a sheet inside a host, and for the
second sheet of a wall lined on both faces.

A lossy medium in front of the host does not change this. A ray starts in air, the transverse
wavenumber is conserved at every parallel face, and it therefore stays real however lossy the
layers in between are; only the normal component becomes complex. The cosine in the host is
`sqrt(1 − sin²θ_air / ε_host)` and whatever lies in front drops out of it, so the real-invariant
solve is the right one behind any stack of parallel layers.

**Behind a solid there are two cases, and one flag tells them apart.** What the recovery has to
return is the cosine the face behind the ray was evaluated at, and a shared face between two
solids is evaluated in one of two ways.

*At the recovered cosine of the first solid.* The wave there came from air across parallel faces,
its invariant `p` is real, and the face was evaluated at `sqrt(1 − p/ε₁)`, a complex cosine if the
solid is lossy. The wave in the host then has the same real invariant and the solve runs unnamed,
exactly as behind a sheet. Naming the solid would assume a real *angle* in it, which that wave
does not have: the argument under the root is then rotated by `arg ε₁`, an error of first order
in the first solid's loss tangent, worth 3.5 dB at 85° for `eps 5.24 − 0.63j` in front of
`eps 1.50 − 0.40j` and a spurious total reflection behind a lossless host from 84.6°.

*At a real angle in the first solid.* That is what the face gets wherever the recovery declined
on the first leg: the wave was launched inside that solid, the solid's own faces are not parallel,
or its total-reflection gate tripped. A real angle in a lossy medium is a complex invariant,
`ε₁ sin²θ₁`, the cosine in the host is `sqrt(1 − (ε₁/ε₂) sin²θ₁)`, and the solve with the solid
named, for a real `u = sin²θ₁` in `[0, 1]`, is the inverse of what was applied. An unnamed solve
there is off by the mirror image of the error above: −1.0 dB at 75° and a killed ray at 85° of
external incidence behind a lossy wedge, −1.35 dB for a wave launched at 25° in the solid.

The transition that crosses the face knows which it did and says so in bit 14 of the
previous-medium word it writes, `prev_real_inv` (see the state words). Set, the solve is unnamed;
clear, it keeps the name. A lossless solid never sets it: for a lossless entry medium a real angle
and a real invariant are one statement, and the named form is kept there so that nothing moves.

**The direction written at a transition is the one a single refraction from air writes.** The
traced direction in a lossy solid is a projection: `qd_refract` normalized a vector of length
`N = sqrt(|1 − c²| + Re(c)²)`, which is 1 only for a real cosine, so the tangential part of that
direction is short by `N`. Refracting it onward with the plain Snell ratio hands the shortfall to
the next medium, where the recovery, which inverts the single-step map, then reads an invariant
that is too small. `REPLACE_BY_VBS` therefore multiplies the Snell ratio it hands `qd_refract` by
`N`, formed from the recovered cosine, wherever it sets the flag above. The direction in the next
medium is then `sqrt(p/|ε₂|)` against `Re(c₂)`, independent of what lies in front, and the unnamed
solve is exact at any depth. Without it the residual was second order in the first solid's loss
tangent but amplified near grazing by `1/(1 − p)`: +0.14 dB at 85° for `eps 5.24 − 0.63j` in
front, +3.1 dB for `eps 1.50 − 0.40j` in front.

**The direction written at an exit into air is the direction the wave has there.** Across parallel
faces the transverse invariant `p` of a wave that started in air is real and conserved, so behind
any stack of layers the direction in air has tangential part `sqrt(p)` and normal part
`sqrt(1 − p)`: parallel to the direction the ray came in at. Two writers produce it, both with the
transition's construction, `qd_refract` on the direction the tracer holds with the Snell ratio
times `N` and the far cosine of the outer pair at the recovered cosine.

- *A glued exit*, in `SHEET_RESOLVE`, where the near medium is lossy and its cosine was recovered
  for a real invariant. The total-reflection verdict is formed at that cosine. Where near and far
  are one medium (an isolated or embedded sheet, an ordered stack, a merged host) the event hands
  the direction on unchanged.
- *A bare exit*, in `REPLACE_BY_VBS`, wherever a refracted direction exists and the face is
  open. A resolved ray takes it as any other ray does.

Where the recovery declines - faces known not parallel, a wave launched inside the medium -
both writers apply the real-angle map to the direction the wave has: the plain Snell ratio and
the far cosine at the traced angle. For a lossless medium that is the wave; for a lossy one it
is the accepted approximation of §5. A
leg with no refracted direction keeps what `ray_mesh_interact` wrote.

**A closed face hands on the direction of the leak.** What crosses a face that is closed at the
cosine the wave has there is a `tf` leak, and a leak carries one thing of the wave in front of
it: the transverse wavenumber. It has no Snell direction. Where that wavenumber has a direction
in air, an invariant below one, that is the direction handed on, tangential part `sqrt(p)`,
normal part `sqrt(1 − p)`: for a wave that came from air through parallel faces the direction
it arrived in, whatever the layers in front made of it, and the same at every frequency of a
run. Where it has none, an invariant of one or more, the face hands on the direction the wave
has in the medium it leaves. One rule for a bare exit, a transition into another solid and the
outer pair of a sheet: a glued entry from air hands on its incoming direction, which is the
direction in air, and a glued exit is closed only above one. `REPLACE_BY_VBS` writes it with
the construction of an exit into air, `EXIT_DIRECTION` with air as the far medium, at both of
its closed-face sites: the general branch, and the early return of a bent leg between two
lossless media without the mass column. With types 2 and 5 the segment and the ray tube follow
(the relaunch geometry). So a layer on a leaking wall hands the wall the same direction as a
traced solid and as a sheet: the wall is met at the incidence angle, its mass term is formed at
the incidence cosine, and a mass-law wall is left in the incidence direction. The step at an
invariant of one is §5. What the real-angle map makes of a lossy host, for the record: the
tangential part is short by `N`, and the far cosine is `Re sqrt(1 − p·e^{−jδ}/N²)` with
`δ = arg ε`, whose imaginary part `p·sin δ/N²` does not vanish at grazing, so the normal part
floors near `sqrt(p·sin δ/2)` and no ray leaves `eps 1.50 − 0.40j` steeper than about 70°.

Consumers, all on that one gate, with either tracer: both boundaries of the Airy denominator, the slab propagation factor
inside `qd_slab_resolve`, the interface coefficient and the total-reflection verdict in
`REPLACE_BY_VBS`, the in-medium traversal factor in `SCALE_BY_MEDIUM`, the incidence cosine that
`SHEET_RESOLVE` passes to the solver, and the total-reflection verdict of a
sheet's outer pair.

**A resolved ray is evaluated as any other.** A ray whose series a reflection collapsed is on
its way back to a face of the same medium, and what `ray_mesh_interact` wrote for that face was
formed at the angle of the segment. `REPLACE_BY_VBS` has no branch for it: the coefficient, the
verdict, the direction and, on a transition into another solid, the real-invariant flag are
formed as for any ray. The one thing a resolved ray skips is the series, and the callers do
that, `APPLY_TRANSITION` and `SHEET_EVENT`. A reflection off a sheet flags
the ray resolved where the host's series was summed at that event, so this is the path of an
echo from a far-side coating whose series was collapsed; where the series was declined the echo
is an ordinary ray.

**The verdict of a sheet's outer pair and of a bare face are one rule.** Both are formed at the
recovered cosine where the near medium has one and at the traced angle otherwise. For a lossy
near medium the real-angle test reads `|ε|·sin²θ_t = p/N²`, the invariant short by the length
the projection lost, and would find the pair open on a closed wave for an invariant between 1
and `N²`: up to about 1.045 for `eps 1.50 − 0.40j`.

### The total-reflection verdict

`Material::interact_with` declares total reflection where `sqrt|ε₁μ₁ / ε₂μ₂|·sin θ ≥ 1` **and the
second medium is lossless**, `Im(ε₂μ₂) = 0` with the resonance in `ε₂`. For two lossless media that
is the critical angle. A lossy second medium has none: it always admits a transmitted wave, with a
complex cosine, `Re(z₂c₂)` is positive at every angle in both channels, and `flux_amp` closes no
port on it. So on a lossy second medium there is no verdict to give, at any angle:

- the transmit port is the Fresnel crossing in normal flux, `1 − R_X` per channel for a lossless
  first medium, with `tf` as the transformer of the face (*The lumped material columns in
  the solver*);
- the reflect port is the Fresnel reflection, at its own magnitude;
- the direction is the one `qd_refract` writes, the real projection of the complex cosine, which
  is the map the recovery inverts;
- the hit code carries no total-reflection bit, the refracting tracer bends the segment and the
  leg entered carries the bent flag;
- the ray tube of `ray_mesh_interact` is not closed by a vertex ray: its own per-vertex test,
  `eta·sin θ_V ≥ 1`, is skipped where the far medium is lossy;
- the outer pair of a sheet in front of a lossy medium is open, so the composite with its flux
  conversion stands and its far face is a transformer;
- a lossy sheet has no verdict of its own either: it is a wave medium at every angle, under a
  closed outer pair as anywhere else.

The gate is decided on the medium and never on a value. A conductivity or a resonance is non-zero
at every frequency, so a lossy medium has one verdict for every frequency of a run, and the bent
flag, which is the base frequency's, describes every frequency's wave there.

**What stays a critical angle.** Two lossless media, at a real angle and at a recovered cosine.
A rigid body of the acoustic mapping is lossless, `ε = (Z_air / Z)² ≪ 1`, and stays closed from
0.003°: transmission into it exists only as the `tf` leak, flat and undeviated. A lossy first
medium in front of a lossless second one keeps the test as it was: at a recovered cosine the
transverse term is real and the test is the evanescence of the far wave; at a real angle it is
the magnitude statement of §5. The undeviated rule of a mass-law medium does not read the
verdict; its gates are its own.

**The two spellings are kept.** The real-angle form and the complex form of the test are still
written separately (§5); the gate is one condition in front of both.

**A wave past that angle needs the recovery to converge.** The angle at which the test used to
trip is where real-index Snell stops having a solution, which is the start of the recovery's
solve. See the in-medium angle.

### The decaying root

A refraction root `sqrt(1 − (ε₁/ε₂) sin²)` whose argument is real and negative has to be the
decaying one, `−j·sqrt|·|`. For two lossless media at a real angle the argument is exactly real,
the principal root of a negative real with a `+0.0` imaginary part is the growing wave, and the
sign of the zero is forced. Formed from a recovered complex cosine the same argument is real only
up to rounding, of either sign, and a test for an exact zero leaves the branch to that sign.
`qd_evanescent` is the test at all ten sites that force the root: the argument is negative and
its imaginary part is within 1e-9 of the real part, relative. An exactly real argument takes the
branch it always took. A far medium with real loss has a negative imaginary part many orders
above the bound and takes the principal root, which is the decaying one.

### The two tracer modes

`ray_progress` drives both public functions in one of two modes, selected by `refraction_mode`,
which is documented as `true` by default. It passes the hit code and `path_dirN` of
`ray_mesh_interact` straight into `ray_state_update`, the direction per frequency and the state
words once.

*The undeviated tracer*, types 1 and 4. The segment continues along the incoming direction, so
the geometry is frequency-independent, and the physical direction is carried separately in
`path_dirN`. Inside a solid the two differ, which is what `use_vbs` detects, and the whole of the
exact machinery of this section is built on that difference: the VBS relocation, the recovery,
the exit port at the recovered cosine, the inhomogeneous traversal, and the direction an exit
writes. In this mode `ray_state_update` is the only writer of a physical direction after the
entry, and it never moves the geometry: the segment stays the incoming one.

*The refracting tracer*, types 2 and 5. `ray_mesh_interact` bends the segment itself, at the base
frequency, so inside a solid the traced and the geometric cosine coincide and the direction
`ray_mesh_interact` writes is the trajectory. That the leg follows a bend is then a fact of the
polyline and not of any cosine, and it is carried in the state: the *bent* flag, bit 14 of the
current-medium word, written by the event that enters a medium with type 2 or 5 where
`ray_mesh_interact` did refract, which its total-reflection bit in the hit code says. On a leg
that carries the flag every consumer of the recovered cosine runs exactly as for the undeviated
tracer, with the VBS on the FBS: the recovery, the exit port and the transition at the recovered
cosine, the series, the sheet composite, the real-invariant flag a transition writes, and the
direction an exit writes. The traversal takes the exact form on every pass, the reflect pass and
a resolved ray included, as it does for any wave that was bent. A resolved ray's exit is
evaluated at the recovered cosine as well, with either tracer.

Two things follow from "the VBS is the FBS" and are kept apart from "the wave was bent". The
type code's VBS bit stays clear on a bent leg, so a bent leg and an undeviated one differ in no
type code. And between two lossless media without the mass column the coefficient
`ray_mesh_interact` wrote from the segment is this face at this angle, and is left as it stands;
only the exit direction is written, from the incoming direction, because the segment carries the
relaunch anchoring of the event before it.

A leak is not a bend. A wave that enters under total reflection holds its direction with either
tracer, at the entry and at every later face of a mass-law body (see the wave held undeviated
below), the flag stays clear, and the leg is treated as it always was: on one wall the rigid
bodies of the acoustic mapping are identical, bitwise, with both tracers, and behind an air gap
to the last digit (§5).

With this tracer the segment out of a lossy medium is not the one `ray_mesh_interact` built.
That function bends at a real angle in the medium the ray leaves, and out of a lossy medium
that is not the direction the wave has. `ray_state_update` therefore takes the relaunch geometry
as optional in/out arguments and puts it on the direction it writes; the next subsection has the
rule. What is left of the base-frequency bend is this: at any frequency of a run other than the
base one the direction inside a medium differs from the base-frequency segment, so the VBS
construction runs there with the VBS off the FBS. The wave of that frequency leaves the medium
beside the committed point, and the coefficient is its field at the committed point (the
propagation convention below): one path for every frequency, and no geometry per frequency.

Every chain fixture written before the recovery-gate case hands `ray_state_update` a synthetic
hit code and a hand-set direction, in the shape of the undeviated tracer. A statement about a
traced run needs a row through `ray_triangle_intersect` and `ray_mesh_interact`.

### The relaunch geometry

`ray_mesh_interact` builds the segment a ray is relaunched on: `origN`, a few float ulp past the
point it leaves at, and `destN`, both along its own outgoing direction; and, for a beam, the
three vertex-ray directions `tridirN`, each refracted on its own. With types 1 and 4 that
direction is the incoming one. With types 2 and 5 it is the Snell bend at a real angle in the
medium the ray leaves, which is the wave wherever that medium is lossless.

**Where one medium lies on both sides there is no bend.** An isolated sheet, a collapsed
duplicate and an ordered stack of two sheets have near equal to far. The transverse wavenumber
and the medium behind the event are what they were in front of it, so the ray goes on along the
line it arrived on: the spine, each vertex ray of a tube and `ray_mesh_interact`'s own
`path_dirN`, with every type, whatever the layers are, whichever comes first, however the two
normals are turned and at every frequency. For a sheet and a duplicate that comes out of the
near | far interface call, whose Snell ratio is one. An ordered stack makes the same call: `ray_mesh_interact`
takes air on both of its sides, as for a single sheet, whatever the two normals are. The
segment of a stack with types 2 and 5 is then the segment of types 1 and 4, to rounding. A stack relaunches from the SBS and aims from the FBS along one
direction, which is one line, so it carries no anchoring of its own (§5). Inside a host the ray
continues the segment the stack was reached on, and `ray_state_update` hands on the direction
it was handed; the two agree behind the stack as well as they did in front of it. The basis of
the own EM coefficient is that of the outgoing direction, so with type 2 it is type 1's.

**Out of a lossy medium the segment is the direction `ray_state_update` writes.** The function
takes `origN`, `destN`, the incoming `tridir` and `tridirN` as optional trailing arguments,
Cartesian only, and corrects them in place at exactly the three places where it writes the
direction of a wave whose cosine in the medium it leaves was recovered for a real transverse
invariant: a bare exit into air (`EXIT_DIRECTION` with `real_inv`, which a resolved ray
passes through as well), a transition into another solid (`face_real_inv`), and a
glued exit. A fourth place is a closed face behind a leg with a refracted direction, where the
direction written is the leak's direction in air: there the geometry follows out of a lossless
medium as well, because the segment `ray_mesh_interact` built under total reflection is the
direction of the wave in the layer and the mass term of the wall behind reads the segment.
Only with types 2 and 5. The gate is decided on the media and the state, never on
a value, so a lossless leg at an open face, a leak that entered from air, a leg whose recovery
declines, a closed face above an invariant of one, the reflect pass and the
undeviated tracer keep their geometry bit for bit, and a caller that passes none of the
arguments gets the geometry it always got (§5 has what that costs behind a closed face).

**The line.** `origN` and `destN` are both put on one line through the point the ray leaves at,
along the written direction: the SBS of a co-located pair, a glued exit and a transition, and
the FBS otherwise. The offset keeps its size, `qd_relaunch_offset`, shared by both functions,
and the length is the one `ray_mesh_interact` gave the segment.

**The tube.** Each vertex ray is a ray of its own, so it gets the spine's construction from its
own direction in the medium: its recovered cosine, the Snell ratio with the length the
projection lost, the far cosine of the pair, which for a leak is the pair of the medium and
air. Behind a slab entered from air it then leaves
parallel to the direction it was launched in, and the fan of a beam behind a lossy wall is the
fan behind a lossless one. The footprint on the face, `trivecN`, is where the vertex rays
arrived and does not move. All or nothing per ray: if one of the three cannot be formed, the
ray keeps the geometry `ray_mesh_interact` wrote, spine included.

**Why the tube has to turn with the spine.** `subdivide_rays` builds the destination of each
sub-beam from the sum of its three vertex directions, not from the parent's segment, and for a
ray in air `ray_progress` then sets every direction slice to that new segment. After a split in
air the tube is the only carrier of direction.

**Who passes the geometry.** `ray_progress` holds all four arrays after `ray_mesh_interact` and
hands them to `ray_state_update` on the base-frequency transmit call only. The geometry is
shared by every frequency of a run and is the base frequency's; the function cannot know which
call of a band it is, so that is the caller's convention. In air every frequency has the same
direction, the one the ray arrived in, so the polyline in air is right for all of them.

**One type-code bit follows the geometry.** The VBS bit of the resolved type says that the
direction the wave has differs from the segment. On the leg behind a lossy first solid it did,
by 0.4° to 0.9°, and the exit of that leg carried resolved type 3; with the geometry handed in
the two coincide and it carries 1, which is what §1 says of every bent leg. No other bit, no
state word and no coefficient of a single wall depends on whether the geometry is passed.

### The propagation convention

**One leg, one form, whichever event closes it.** `SCALE_BY_MEDIUM` asks one question, whether
the wave was bent at the face behind it (`use_vbs || bent`, §"The in-medium angle"), and the
transmit pass, the reflect pass and a resolved ray all get the same answer.

*Bent, with a recovered cosine:* the exact form, `exp(−j k₀ (n·cs·d + q·s))`, with `d` the normal
thickness and `s` the transverse displacement, both pure geometry off the direction the wave has,
and `q² = n²(1 − cs²)` the transverse wavenumber Snell conserves from the entry medium.
Conductivity lives inside `n`; only the alpha column is drawn from `medium_gain`, through
`excess_gain`. For a real index the two terms recombine into `n·dist_refract` exactly.

*Bent, without one* - a mass-law solid the wave is refracted into, or a leg whose recovery
declines: the ITU-R P.2040 rate along the refracted path for the magnitude, and the phase of the
real index. There is no inhomogeneous wave to be exact about. What is left of that
hybrid is in §5.

*Not bent* - a leak, or a leg with no refracted direction: the walk-off form below.

**Which point the phase belongs to: the one the polyline leaves at.** The tracer commits one
polyline for every frequency of a run, and the path length, the delay and the free-space phase
of a path are taken from it. A coefficient therefore owes the field at the point where that
polyline leaves the medium. The wave itself leaves beside that point wherever the polyline is
not the refracted ray: with the undeviated tracer always, and with the refracting tracer at
every frequency but the base one. For a plane wave the field at the polyline's exit point is the
stack's transfer on the thickness axis times `exp(−j k₀ sin θ·Δy_polyline)`, whatever the
polyline is, so the two points differ by the transverse wavenumber times the lateral distance
between them, in phase only.

`SCALE_BY_MEDIUM` holds both ledgers and both cosines and forms that without state. Where
`use_vbs` is set, the VBS being off the FBS:

- the exact form is multiplied by `exp(−j k₀ Re(q)·(s_polyline − s))`, with
  `s_polyline = dist_geo·sin θ_i` the lateral walk of the polyline. `Re(q)` and not `q`, so that
  it is a phase also where the invariant is complex (a wave launched inside a lossy solid, a
  wedge) and every magnitude stays the one of the refracted ray;
- the form without a recovered cosine takes its phase as `k₀·n_re·dist_geo·(dir · segment)`, the
  wave vector along the direction the wave has times the polyline, which needs no parallel faces.

Where `use_vbs` is clear the two points are one and nothing is applied: a bent leg of the
refracting tracer at the base frequency, and every leg that was not bent. Nothing of the geometry
is moved for it at any frequency, and no state is read.

That is the convention of the sheet composite as well, and it always was: dividing the displaced
host out on the thickness axis, `exp(−j k₀ n_d·c_disp·t_disp)`, is the insertion factor at a
fixed lateral position. A sheet in air, a glued sheet and a sheet inside a host therefore sit on
the same point as the host legs around them, and the displaced-host term carries no transverse
part and must not get one.

For the undeviated tracer the result is the insertion phase of a slab,
`k₀ d (n cos θ_t − cos θ_i)`, on the straight line.

**The walk-off form.** On the branch `SCALE_BY_MEDIUM` takes for a wave that was not bent,
the applied argument is `−k₀ (n_re·dist_refract·ct_t² − dist_geo·ct_i²)`. Grouping each term into a
normal thickness, `d_t = dist_refract·ct_t` and `d_i = dist_geo·ct_i`, that is
`−k₀ (n·d_t·ct_t − d_i·ct_i)`, the tilted-slab excess phase `k₀ d (n cos θ_t − cos θ_i)` referenced to
the emergent wavefront rather than to the ray path. A wave held undeviated has no `cos θ_t` of its
own, so the two cosines coincide and the expression collapses to `k₀ d cos θ_i (n − 1)`, which is the
`cos²` written in a path length. Both terms sit on the thickness axis, which is why the free-space
reference carries a cosine as well. It is referenced to the polyline's point by construction: the
polyline of such a wave is the wave. A rigid body of the acoustic mapping is entered as a leak and
takes this form with either tracer, bit for bit the same.

### Where the layer of a sheet lies

A sheet has no thickness in the mesh. Its composite stands for a layer of thickness `thk`, the
tracer charges the path up to the mesh plane as the surrounding medium, and a reflected ray turns
around at that plane. So the composite has to say where the layer lies about the plane, and both
ports have to say the same. A layer whose front face is `u` in front of the plane has, referenced
to the plane and with `φ_d(x)` the one-way factor of the displaced medium over a normal length
`x`,

    S11 = r / φ_d(u)²        S22 = r / φ_d(thk − u)²        S21 = t / φ_d(thk)

The transmit port takes the displaced medium out over `thk`, which is `S21` for any `u`. The
reflect port owes the round trip through the part of the layer in front of the plane:

| sheet | the layer lies | the reflect port carries |
| --- | --- | --- |
| glued entry, `air \| sheet \| solid` | in the near medium, in front of the solid's face: `u = thk` | `1/φ_d(thk)²` |
| one medium on both sides: isolated, inside a host, an ordered stack | about the mesh plane, which is its mid-plane: `u = thk/2` from either side | `1/φ_d(thk)` |
| glued exit, `solid \| sheet \| air` | in the air beyond the solid's face: `u = 0` | nothing |
| mass-law sheet | nowhere: it displaces nothing and has no in-layer phase | nothing |

`φ_d` is the factor the transmit port already forms, at the same cosine and under the same
cap, so the three rows are that factor squared, once and not at all. For air it is
a phase, `exp(+2j k₀ cos θ·thk)` for a coating: 14° for 6 mm at 1 GHz and 10°, 99° for 50 mm of
mineral wool at 1 kHz and 20°. In a lossy host it is also a magnitude, +0.18 dB to +0.24 dB for
6 mm inside `eps 1.50 − 0.40j` at 10° to 60°: the host that the legs charged on either side of the
plane and the wave did not travel.

**No more than the host on its side.** Taking a medium back out is a division, by its loss as
much as its phase, so each part of the layer takes at most the host that lies on its side of
the mesh plane, as a normal thickness: with `u_f` and `u_b` the parts in front and behind,
each the smaller of its share of `thk` and the host there,

    S11 = r / φ_d(u_f)²        S22 = r / φ_d(u_b)²        S21 = t / φ_d(u_f + u_b)

That is the layer between what is left of the host on either side, a wall that exists, on every
port: a sheet or a stack that fits its host is not capped at all, up to the whole host, and one
that does not fit keeps its thickness and the wall comes out longer than the mesh. A wall of
lossless media then returns what it received, summed over its paths, whatever the thickness is:
1.00003 at most over 72 walls. In front the host is the leg the ray is on, the accumulated leg
times the traced cosine. Behind it is the gap along the segment to the second face of the hit,
or to the destination, times the cosine of the segment, so both tracers see one host. Where the
second face is co-located - an ordered stack, a collapsed duplicate - the hit says nothing of
what lies behind, and the leg in front is taken for both sides (§5 has the one geometry that
gets wrong). A mirror of the host's series is capped by the leg the series runs over, which is
the host in front of it. Air costs nothing to displace and is not capped, and a mass-law layer
displaces nothing.
**Why the mid-plane.** An event sees the two media, the sheet and the angle, and nothing else.
"Behind the plane as the ray first met it" is path history, which no state word holds, and it is
not one wall: a ray from the other side would find the layer in the other half. Put on by a
fixture, it breaks the equality of the transmission from the two sides of a wall with a sheet
off its middle by up to 0.13 dB. "On the side the normal points to" is readable from the hit
code but not reliable: a sheet has no inside, exporters write double-sided duplicates, and a
collapsed pair has both normals. The mid-plane needs neither, and it is the only placement for
which `S11 = S22`.

**What it is not.** The polyline, the path length and the delay stay at the mesh plane; the
factor is the field a plane wave has there, per frequency, and no geometry moves.

**Sites.** `qd_stack_resolve` forms both factors from the displaced medium and takes the host on
either side as `host_near` and `host_far`. `SHEET_RESOLVE` hands it the two and selects the
port; `ray_mesh_interact`'s own composite displaces air in every case it resolves and hands it
none, so the two public functions agree. Where `qd_slab_resolve` reads a sheet as a mirror of
the host's series it asks the solver, names the displacement side and hands it the leg of the
series: a lining lies beyond the host's face and its mirror is referenced to that face as it
stands; a sheet inside the host has the host on both sides and the solver centres it, ahead of
the ray and behind it. An ordered stack is centred over its coherent thickness.

**What follows.** A wall coated on its near side, a wall lined on both faces, a sheet in front of
a wall across an air gap and a wall with a sheet inside it are each one wall on both ports,
whether a series is summed or followed. A coating on a wall and the same layer meshed as a
traced solid in front of that wall reflect with one phase. `air | host 50 | sheet 6 mm | host 50
| air` as meshed is host 47 | sheet 6 | host 47. A sheet of air on a wall is the bare wall,
phase included.

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

**The angle is the one the wave has in air.** A plate is driven at the trace wavenumber, which the
transverse invariant `p` fixes, and the angle enters its transmission only through the wave
admittances of the media on its two faces, symmetrically. The term is the equal-fluid form with `X₀`
calibrated on a plate in air, so its cosine is the cosine of air at that trace wavenumber,
`sqrt(1 − p)`: for a wall standing in air the incidence cosine, whatever refracting layer lies in
front of the plate. What that layer does to the admittance is carried by the two interface
amplitudes and the flux conversion, which are wave quantities. One rule for every statement of one
wall: a mass-law sheet in air, a lining on the far face of a traced solid, a mass-law sheet inside
a host, a stack with a mass-law layer in air or inside a host, the same wall met from its other
side, and a traced mass-law wall behind a layer, which a closed face hands the direction the
invariant has in air (the in-medium angle).

For a plate with a cover on both faces this is the thin-cover view: the plate pays what the bare
plate pays, and the cover acts through its own loss and its own faces. The exact plate inside an
unbounded host would read the host's admittance, which a term calibrated in air cannot express;
the convention is the one under which a cover shrinking to nothing leaves the bare plate, and under
which the term does not step where a cover ends.

`SHEET_RESOLVE` forms the cosine once and hands it to the solver. The gate is
decided on media and state: the near medium is not air, it carries no mass column, the leg is
known to be bent (`refracted`), and the sheet or a layer of the stack carries `m > 0`. The cosine
is the far cosine of near | air, at the recovered cosine where there is one and at the traced angle
where the recovery declines. Everywhere else the term keeps the traced cosine, which reached from
air, and from inside a mass-law medium that holds the wave undeviated, is the incidence cosine.
Where the invariant has no direction in air, one or more, the term keeps the cosine the wave has in
the near medium, as the direction a closed face hands on does; that choice is taken on a value, per
frequency, and selects that frequency's coefficient and nothing else. What the rule leaves is §5:
the step at an invariant of one, a ray launched inside a host, and a lossy wedge.

Callers supply the two pieces as follows. A sheet passes the normal thickness and the cosine
the wave has in air, so the quotient `thk / cos` is never formed. A traced solid passes neither and gets
`dist_geo·cos θ` and `cos θ` by default, whose product is the `dist_geo·cos²θ` the term was written in
before; across a slab traversal `dist_geo` is the normal thickness over the cosine, so the surrogate
is the normal thickness and the two callers form the same argument. The public
`quadriga_lib::medium_gain` entry point, and `calc_diffraction_gain` through it, supply no angle and
therefore read `A` unchanged.

`m > 0` also excludes a slab from correction-mode resolution entirely, from the exact in-medium
propagation branch in `SCALE_BY_MEDIUM`, and from the solver: the eta/mu of such a material are
calibrated for surface impedance only, so the in-slab index is not physical and the internal series
must not be resummed.

### The lumped material columns of a sheet

Four columns act on a sheet composite outside the wave problem. Each has a fixture and is exact
against its reference; what follows is the convention each fixture pins.

**`att` and the coincidence term**, through `interface_gain`. A penetration loss is owed once per
medium entered, which is what `Material::interact_with` charges for a single transmission-class
crossing. The solver forms `ig = sqrt(G_sheet · (far_is_new ? G_far : 1))` with
`far_is_new = !far_is_air && !far.same_as(near)`, on the transmit port only:

| stack | owed |
| --- | --- |
| isolated, `air \| sheet \| air` | `G_sheet` |
| glued entry, `air \| sheet \| wall` | `G_sheet · G_wall` — the wall is entered here and nowhere else |
| embedded, `host \| sheet \| host` | `G_sheet` — the host was entered earlier and paid then |
| glued exit, `host \| sheet \| air` | `G_sheet` |

The reflect port carries the sheet's gain once, as `sqrt(G_sheet)` in amplitude, on the part of
it that entered the sheet and came back out, and nothing on the part that turned around at the
near face. It never carries the far medium's: nothing entered it. That is what a traced solid
does with the same echo, whose entry pays the gain inside `interact_with` and whose back
reflection, series and exit into air pay nothing more, so a sheet and a traced solid of one
material and thickness are one wall on both ports. The series itself pays nothing in either. In
a stack each layer's gain is on what came back out of that layer (the solver, above: `Ge`). The mass-law
route does not change: its reflect port is the near interface, and nothing entered. `att` is an
empirical level with no statement of where the energy goes; the rule is consistency between the
two representations and nothing more. For a sheet inside a host the traced counterpart pays the
host's gain again on re-entry and the sheet does not, by the table above, so that equality is
for a sheet in air and for a coating.

**`tf`** moves power between the two ports of a boundary. There are three rules, and which one a
face takes is decided on the verdict of the face and on the mass column, never on a mode and
never on the transverse invariant: the transformer where a wave is on both sides, the pad where
a wave is on one side only, the pair rule on the reflectance where a mass-law medium has none.

*The transformer*, through `Material::tf_face`, at every face that has a wave on both sides: the
face of a coherent sheet or layer - one without the mass column - in the solver; a bare boundary, the face of a traced solid, in `Material::interact_with` wherever its
total-reflection verdict is open; and the mirror `fresnel_r` of `qd_slab_resolve` forms for such a
face. Per channel at every site. A wave has two faces that interfere, so the rule has to say
what a face does to an amplitude. Between two lossless media that carry a wave the power
statement fixes that, and it is one statement: the impedance step of the face is moved towards
its neighbour (`tf > 0`) or away from it (`tf < 0`). That is an ideal transformer of real ratio
`k` in front of the far side of the face: the tangential field is scaled by `k` and its partner
by `1/k`, so the normal flux is continuous whatever the two media are - lossy, or with an
evanescent wave in one of them - and the face creates nothing and loses nothing. With
`u = (1 − r)/(1 + r)` the admittance ratio of the two sides, `x0 = |u|` and
`rm = (1 − x0)/(1 + x0)` the real reflection of that modulus,

    Rm = the pair rule on rm²        s = sqrt(Rm) at the sign of rm        nu = ((1 − s)/(1 + s)) / x0
    r' = (1 − nu·u)/(1 + nu·u)       t' = t·sqrt(nu)·(1 + r')/(1 + r), the same factor on both crossings

`tf_face` writes this without `nu`, so that `tf = −1` needs no infinity. The library's TE and
TM reflections differ in sign and the form is the same for both. Where `u` is real - two
lossless media below any critical angle - this is the pair rule on the reflectance, digit for
digit. A face neither of whose media carries `tf` is not touched and is the raw Fresnel face,
which is what the characteristic matrix carries. A face whose raw reflection is exactly `+1` or
`−1` is left raw. At a match of the two sides, `x0` within 1e-9 of one, a forced
reflection has no side to move away from, and the sign is taken from the media: the side that
forces less sees it negative, so the face is one face from its two sides.

*The pad of a closed face*, through `Material::closed_face`, at every face that has a wave on
one side only: nothing propagates behind it, so the far side is a reactance, reflects
everything and carries no flux, and a transformer has nothing to hand a flux to. The leak is a
matched resistive pad in front of that reactance, formed on the modulus of the immittance of the
side the wave is on, as the transformer forms its ratio on a modulus. With `y` that immittance
(`z·c` for TE and the scalar type, `conj(z)·c` for TM, the form the reflection is written in),
`b` the far side over `|y|` read as a reactance, `s² = R'` the pair rule below on a reflectance
of one and `nu = (1 − s)/(1 + s)`,

    load = |y|·(nu + b)/(1 + nu·b)        r' = (y − load)/(y + load)        passed = |1 + r'|²·Re(load) per unit forward amplitude

A pad is a real element: the face creates nothing whatever is in front of it, a lossy medium or
a sheet the wave is evanescent in, and what it passes is what it absorbs. It is driven by the
field at the face, as a leaking wall is driven by the pressure on it, and not by the forward
wave alone, which in a lossy medium carries only part of the flux and as an evanescent wave
none. In front of a lossless medium that carries a wave it is `sqrt(R')` on the Fresnel
reflection and `1 − R'` of the flux, flat and undeviated; where the far wave goes to grazing it
is the limit of the transformer, so a face has no step at its critical angle beyond the
square-root cusp of the far cosine. A pair without `tf`, and one at `tf = −1`, is left raw and
passes exactly nothing. Only the reactance of the far side is read, so a verdict that is forced
closes the face as well. The phase of the leak is real in EM and that of the raw transmission
in scalar mode. Five sites: *a closed bare boundary* in `Material::interact_with`, total
reflection by its verdict, in the scalar and the EM types alike; *the far face of a coherent
sheet under a closed outer pair*, on both ports; *the mirror of `qd_slab_resolve` at a closed
face*; *a closed mass-law layer of the solver seen from a coherent side*; and *the two faces of a
closed mass-law sheet in the solver* (*A closed mass-law layer*). The reflection of a closed
face is the value the pad gives, above one in magnitude where the Fresnel value seen from a
lossy medium is above one; nothing caps it, at a sheet, at a mirror or at a bare boundary, in
the scalar and the EM types alike. A mass-law medium has no wave inside: where it is
the side the ray is on at a closed bare boundary, its immittance is taken real.

*The pair rule on the reflectance*, through `Material::apply_tf_pair`. Per boundary and per
channel, with `R0` a reflectance clamped into `[0, 1]`, which is the one clamp left in the
file and acts only at an open mass-law face next to a lossy medium (issue 6):

    R_leak = R0·(1 − tf_A⁺)·(1 − tf_B⁺)        R' = R_leak + (1 − R_leak)·max(tf_A⁻, tf_B⁻)

with `tf⁺ = max(tf, 0)` and `tf⁻ = max(−tf, 0)` of the two media `A` and `B` that form the boundary.
A boundary is owed by the pair, symmetrically. On a reflectance of one it is the leak of the
pad. On the Fresnel reflectance it is the rule of a face that is a statement about power and
not about a wave: *the faces of a mass-law sheet and of a mass-law layer in the solver that are
not closed*, with `r' = sqrt(R')·exp(j·arg r)` and `t' = t·sqrt((1 − R')/(1 − R0))`, and a
closed mass-law layer seen from another mass-law layer, with the flat leak.

What the transformer gives, measured through the intersector on sheets of 1, 6 and 25 mm, eight
permittivities from `eps 0.31` to `eps 5.24 − 5j`, `tf` 0.5, 0.9 and −0.5 on the sheet or 0.5 on
the host, 0° to 85°, scalar, TE and TM:

- *Nothing is created.* In air, inside the lossless `eps 2.50`, and as a lining on its near and
  its far face and on the near face of `eps 2.50 − 0.80j`, the two ports of the sheet event
  together read at most one on every row; the same through an ordered stack in the solver. A
  lossless sheet between lossless media reads one to 1e-6 on every row, tunnelling included.
- *One expression.* Nothing is decided on the transverse invariant, so a sheet has no step where
  it becomes rarer than the invariant: neighbours 0.02° apart are within 0.0028 dB on the
  transmit port in air, inside a lossless and a lossy host, as a lining and on a rigid wall.
- *The end points.* `tf = −1` closes the transmit port exactly, for every sheet. `tf = +1`
  closes the reflect port exactly where the two sides of a face are in phase; where they are
  not, a real ratio cannot match the face and something is left (§5).
- *A tunnelling sheet.* A face with an evanescent wave behind it reflects everything, with `tf`
  or without; `tf` changes the coupling of the two faces through the layer. S 6 mm of `eps 0.31`
  inside `eps 2.50` at 45° passes −0.041 dB without `tf`, −0.023 dB with `tf` 0.5 and −0.263 dB
  with `tf` −0.5, the two ports adding to one each time; with a host loss of 0.8, −0.345 dB at
  `tf` 0.5. `tf` ±1e-9 sits on the sheet without `tf` to 1.2e-7 dB.

And at a bare boundary, measured as the two ports of a traced solid in air, each a sum over
every path, seven permittivities from `eps 0.31 − 0.20j` to the lossless `eps 5.24`, `tf` 0.5,
0.9 and −0.5, 0° to 85°, 1, 6 and 50 mm, types 4, 5, 1 and 2, with the series summed and with
every bounce followed:

- *Nothing is created, in any channel.* The two ports together read at most 1.00026 on every
  row, and a lossless solid at least 0.99992. One face seen from air passes what it does not
  reflect, in VV, in HH and in scalar mode, to 1e-9.
- *A sheet is its traced solid.* A 6 mm layer with `tf` in air as a solid and as a sheet is one
  wall on both ports, within 1.7e-3 dB, scalar and EM, lossless and lossy. A sheet of air on a
  wall with `tf` is the bare wall, lossless or lossy, to 1e-15 in the complex plane.
- *A summed series and a followed one are one wall*, because the mirror is the face.
- *The acoustic main road is where it was.* A rigid wall of the acoustic mapping is closed from
  0.003° and takes the pair rule: its first face reflects `1 − tf` and passes `tf`, at `tf` 0.1,
  0.3 and 0.9, with and without the mass column, to 2.2e-16.

What it costs is in §5: the reflectance of a face whose two sides are not in phase is not
`(1 − tf)·R0`; a forced face steps where the admittance ratio has modulus one; a forced
reflection appears in TM at the Brewster angle; and a rigid wall with a loss column, which is
not a valid material, steps from `1 − tf` to one.

**No column but `m > 0` says that a permittivity is an impedance mapping.** A rigid sheet of
the acoustic mapping without the mass column is a wave medium like any other and tunnels, with
or without `tf`; with the column it is a closed mass-law layer and without `tf` passes nothing.
A rigid sheet carries the mass column (§5). A rigid traced solid is closed by its verdict with
or without the column.

**Under a closed outer pair** the far face of a coherent sheet has no wave behind it and is the
pad of a closed face; the near face is the transformer it is everywhere else. Both ports are
the one composite: the reflect port with the pad's reflection and the transmit
port as on any other sheet, the near crossing, the one-way factor and the series, times what
the pad passes of the incident flux in place of the far crossing and the near-to-far
conversion. The two ports of a sheet on a closed far side together read at most one, a
lossless sheet one; a sheet rarer than the invariant tunnels and decays with its thickness,
lossless or lossy alike.

The sites, then: every boundary of the solver, each paired with its own neighbour, by the
transformer where the layer is coherent and the far side carries a wave, by the pad where it
does not, and by the pair rule where the layer carries the mass column and is not closed;
`Material::interact_with` for a bare boundary and `fresnel_r` of `qd_slab_resolve`, slab paired
with neighbour, by the transformer where the face is open and by the pad where it is closed.
Where `fresnel_r` resolves a bounding sheet or stack as layers, the solver's composite already
carries the `tf` of every face of it and nothing is redistributed a second time.

A forced reflection at a matched boundary is still a reflection: an admittance ratio of one with
`tf < 0` gives `R' = |tf|`. So a sheet taken to the permittivity of a `tf < 0` wall does not
become that wall, it becomes two mirrors, and a lossless wall built that way - coated, or as two
solids of one permittivity - still passes and returns what it received, to 2e-3. A sheet of air
on a `tf`-bearing wall is the bare `tf`-bearing wall exactly, for a lossless and for a lossy
wall.

**`resF`, `resQ`, `resS`**, a Lorentz pole `resS·resF² / (resF² − f² + j (resF / resQ) f)`, all in
GHz, that `Material::eta()` adds to the permittivity. It is part of the medium: every admittance,
cosine, Snell ratio and verdict, every layer phase and traversal, the displaced medium,
`medium_gain` and `refractive_index` read `ε + res`. A resonant medium is a dispersive dielectric
and nothing else: a lossy pole is the same medium as a conductivity of that size, bit for bit on
every path of a sheet and of a traced solid, and a pole that takes `ε` to 1 makes a sheet of air.
A pole has a non-zero imaginary part at every frequency, so a resonant medium is lossy for every
verdict of §1 at every frequency of a run; `resQ` 1e300 still leaves 2e-300. `medium_gain` forms
its rate as `8.686 k₀ |Im n|` from the complex index, which holds where a strong pole takes
`Re ε` below zero. A negative `resS` is a gain medium (§5). What a resonance does on a material
with the mass column is issue 1.

**`alpha`**, through `excess_gain`. A real one-way amplitude factor `10^(−α·thk / (20·|Re c|))` on
`φ`, so it enters the Airy numerator once and the denominator twice. The solver carries it in the layer
phase, as above.

**`m`** takes the incoherent mass-law route of the section above. Of the four, only the interface
gains still apply to it, together with whatever `tf` does to its two interface amplitudes. Its reflect
port is its near interface and nothing else, so a mass-law sheet named as a bounding medium in
`qd_slab_resolve` is a half-space by its own convention, not by a fallback, and `fresnel_r` does not
hand it to the solver.

### A closed mass-law layer

A mass-law layer has no wave inside by convention, and where the wave its permittivity would give
it is evanescent it has no forward port on either face. That is the whole of a rigid body of the
acoustic mapping, `ε = (Z_air / Z)² ≪ 1`, from 0.003° on.

**The verdict belongs to the layer.** A layer or sheet with `m > 0` is *closed* where it is
lossless, `Im(ε μ) = 0` with the resonance in `ε`, and the transverse invariant is at or above
its `ε μ`. That is the test `Material::interact_with` applies to a wave that meets the medium, asked
of the layer and not of a face: both faces are then closed together. Asked of the far face from
inside the layer, the same function would judge the medium behind it and find a rigid layer open
towards air. The Fresnel reflectance does not carry the verdict. For a rigid layer the tilted
admittance of the scalar and TE channel is finite and purely imaginary, `−j sin θ`, so against a
lossless neighbour `|r| = 1` to rounding, and against a lossy one it is below one although nothing
propagates: `1 − R0` is 6.4e-2 for `eps 2.50 − 0.80j` at 10° and 8.8e-5 for a loss of 1e-3. That
remainder is the interference of the incident and the reflected wave in the lossy medium, not a
forward port. In the TM channel the face reads one to 5e-8 either way.

**What a closed layer passes.** A closed layer has no wave inside and carries a leak as a flux.
Seen from a side that carries a wave, a coherent layer or a medium without the mass column, its
face is the pad of a closed face (*The lumped material columns of a sheet*): on the way in
the flux the pad passes, on the way out the reverse, the amplitude that flux launches in the
medium behind, `passed / |y|` in amplitude per unit root flux. The two are one face read from
its two sides. Where neither medium of a face carries `tf` it passes exactly nothing. Seen from
another mass-law layer or sheet the face keeps the flat leak, `1 − R'` with the reflectance
taken as one, a statement about the normal flux of that side.

**Where the conversion goes.** The layer carries no flux of its own, so the one conversion at
the outer pair has nothing to convert through:

- a pad face hands over a flux directly: the transmit port is what the near face passes of the
  incident flux, what that launches behind the far face, the mass term and the interface gains,
  and the near-to-far conversion is not applied;
- a flat face between mass-law layers is put on flux by the ratio of the normal flux on its two
  sides.

A mass-law layer that is not closed is not touched: its conversions telescope into the one at
the outer pair.

**What follows, measured.** `C` is 10 mm of `eps 2.50 − jε″`, `M` a rigid sheet 22 mm with `tf`
0.2, levels without the mass term, scalar, at 10°:

| ε″ of C | stack C \| M | stack M \| C | M side over C side |
| --- | --- | --- | --- |
| 0.8 | −13.57 dB | −13.57 dB | 0.000000 dB |
| 0.1 | −12.63 dB | −12.63 dB | 0.000000 dB |
| 1e-3 | −12.48 dB | −12.48 dB | 0.000000 dB |
| 1e-6 | −12.48 dB | −12.48 dB | 0.000000 dB |
| 0 | −12.48 dB | −12.48 dB | 0.000000 dB |

The laminate with `C` a traced solid and `M` its lining reads the same to 2.2e-4 dB from either
side. The two sides of one wall are one transmission whether `C` is lossless or lossy, in the
scalar, the TE and the TM channel: −13.20 dB and −13.32 dB at 30° and 60° for ε″ 0.8 from both
sides, and the same for the rare lossy `eps 0.31 − 0.20j`, where the forward-flux form the
faces had before read up to 13.49 dB apart. The sheet event of a lining on the near face of a
lossless host is its two leaks, −13.98 dB for `tf` 0.2; next to `eps 2.50 − 0.80j` it reads
−14.15, −14.41 and −14.76 dB at 10°, 30° and 60°, and what the plate passes into a lossy host
and out of it mirror each other about the two leaks, to 1e-6 dB. A rigid layer without `tf`
passes exactly nothing, as a sheet, a lining and a layer of a stack alike. A rigid layer
stacked with a mass-law layer that is not rigid is one wall in both orders. Both tracers and
the types 1 and 2 read alike. `ray_mesh_interact`'s own composite forms the same rule, being
the same function.

**What it does not cover.** A near medium that holds the wave undeviated, a rigid traced wall:
a mass-law sheet behind it is a later face of that wall and takes the rule of *A sheet behind a
held wave* below, not this one. A coherent sheet, without the mass
column, that is rarer than the invariant: it tunnels, and its faces are transformers, or the
pad where its far side is closed (*The lumped material columns of a sheet*). The
verdict is taken on a value, per frequency, and selects that frequency's coefficient; it reaches
no state word and no direction.

### A sheet under total reflection of its outer pair

`SHEET_RESOLVE` re-derives the total-reflection verdict from the outer pair, near | far, at the
cosine the wave has in the near medium: the recovered cosine where there is one, the traced
angle otherwise. That verdict says one thing only: no Fresnel wave propagates in the far medium. It
says nothing about the sheet, and the sheet stays in the result.

**Both ports are the composite.** The wave enters the sheet, turns around at the far face and
comes back, and none of that needs a forward port in the far medium. The near face is the
transformer of *The lumped material columns of a sheet*, as on any other composite; the
far face is the pad of a closed face, whose reflection the reflect port and the series carry,
and whose passed flux is the transmit port. The reference plane and the echo
gain are those of any other sheet. The displaced host is taken out at the incidence cosine,
because the far side carries no wave and its Snell cosine is evanescent and would amplify.
With no `tf` anywhere the reflect port is the matrix of the layer on its closed backing, at
unit magnitude for a lossless sheet, to 9.9e-16 for a rare sheet on a rigid wall, and the
transmit port is exactly zero.

Measured through the intersector, S 10 mm of `eps 0.31 − jε″` with `tf` 0.5 on a rigid wall
with `tf` 0.3 at 60°, scalar: −3.03 dB at ε″ 0.2, −2.80 dB at 1e-2 and −2.79 dB at 1e-3, at
1e-6 and at zero; at zero loss −2.13, −2.79, −12.13 and −24.06 dB at 1, 10, 100 and 200 mm. On
a wall without `tf` the same sheet at 60° reads −4.31 dB at ε″ 0.2 and −4.07 dB at 1e-6 and at
zero, and reflects −2.48 and −2.16 dB. Over 6,552 rows of a sheet on a rigid wall and of a
coating on `eps 5.24` met from inside - eight to ten permittivities, `tf` 0.5, 0.9, −0.5 and
none, 1, 6 and 25 mm, seven angles, scalar, TE and TM - the two ports together read at most
one, and each sits on the reference of the rule at 1.4e-14.

**A mass-law sheet** has no wave inside and is not a composite. `qd_slab_resolve` forms its
transmit port, when told the far side is closed, as the two crossings a traced solid of the
same material would produce, with the mass term between them:

    near | sheet    Material::interact_with's transmission, with the sheet's interface gain
    sheet | far     Material::interact_with's transmission, with the far medium's interface gain

Each is evaluated at the cosine the wave has there. *Entered as a leak*, near | sheet itself
past its critical angle, which is every rigid sheet: the wave kept the incidence cosine and
meets the far face with it, as a wave held undeviated; see the next subsection. *Entered by an
ordinary crossing*: the wave is at the Snell cosine, where the far face is closed whenever the
outer pair is, and the verdict is forced rather than re-derived from a complex cosine. It
carries no series at any angle, and its reflect port is its near face.

`ray_mesh_interact`'s own composite does the same, for a spine and for a tube edge past the critical
angle.

**The mass column separates a plate from a layer.** A coherent sheet is a wave medium at every
angle and under every outer pair: denser than the invariant it propagates, rarer it tunnels and
decays with its thickness, and it has no verdict of its own, lossless or lossy. With the mass
column a sheet rarer than the invariant is a closed plate and passes the leaks of its two
faces at every thickness: the lossless `eps 0.31` with `tf` 0.5 and `m` 20 on a rigid wall with
`tf` 0.3 passes 0.325031 and reflects 0.500000 at 1, 10 and 100 mm. A porous absorber on a
rigid wall is a coherent sheet, and its leak into a wall with `tf` is continuous in the angle,
in its loss and in its thickness.

**Why this is the main road in acoustics.** Rigid materials are mapped to `ε = (Z_air / Z)² ≪ 1`:
concrete is 2.4e-9, wood 2.2e-8. The air | wall critical angle is then about 0.003°, so every sheet
on a rigid wall is under total reflection of its outer pair at every practical angle. Transmission
into a rigid body exists only as the `tf > 0` leak, `m`, `att` and the coincidence term are the
isolation on the way through, and the reflectance of a face seen from the room is one minus its leak.
In EM the same branch is a coated solid reached from inside at a steep angle.

### A wave held undeviated inside a mass-law medium

A material with `m > 0` is calibrated for surface impedance only, so its `ε·μ` is not a propagation
index, and the critical angle of a face seen from inside it is formed from exactly that index. A
`tf` leak keeps its direction, so the reverse of a leak that crossed a face at an angle is the
crossing from the other side at the same angle. Judged from inside, that crossing is total
reflection past the critical angle and passes only the flat leak; judged from the other side it
never is. Wood (`ε = 2.1609e-8`) on concrete (`ε = 2.401e-9`) has a Snell ratio of 3 and differed by
4.5 dB at 30° between the two directions for that reason.

`Material::interact_with` therefore takes an `undeviated` flag. Where it is set, the medium the wave
is in carries `m > 0`, and the face leads towards the lower index, the transmit port is what the
other direction charges at the same cosine, with the interface gain of the medium entered, and the
verdict handed back on that port is the undeviated one, so no Snell bend is applied. Three gates,
each for a reason:

- *`m > 0`*, a material column, never a mode. A rigid material with `tf` and `m = 0` keeps the
  ordinary verdict; see §5.
- *Towards the lower index only.* The other direction has no critical angle and is the ordinary
  crossing, untouched and bit-identical.
- *Undeviated only.* A wave the tracer refracted into the medium arrives at a Snell angle, where the
  crossing from inside is already the reverse of its entry.

Who sets the flag: `ray_mesh_interact` always, on both passes, because its incidence angle is the
geometric one and `ray_state_update` replaces its result wherever a refracted direction exists;
`ray_state_update` wherever no refracted direction was supplied or the traced and geometric cosines
coincide, unless the leg carries the bent flag; the two crossings of a mass-law sheet under a closed outer pair, for the far face of a sheet that was entered as a leak; and the held branch of a sheet or a stack behind a held wave, for every face of it (below). No state is
carried between iterations for it: whether a wave is undeviated is read off the direction the
tracer already holds.

**The direction is held at every face, towards either index.** A mass-law body is a lumped
plate with no wave inside, and a face that is charged the same from both sides at one angle is
a face the wave has that angle on both sides of: a crossing without a bend. Trace matching says
the same from outside, a wave through a plate between two half-spaces of one medium leaves
parallel to the direction it arrived in. Towards the lower index the undeviated verdict above
suppresses the bend. Towards the higher index, which every exit into air is, the crossing is
the ordinary one and its Snell ratio would be formed from two permittivities that are impedance
mappings; there `interact_with` hands back the geometry of no bend instead, a Snell ratio of
one and the incidence cosine, on the two gates *`m > 0`* and *undeviated* and for a
transmission. Every caller then reproduces the direction it came in with through the code it
already has: the segment and the ray tube of types 2 and 5, `path_dirN` of every type, the
direction a glued exit writes. The coefficient, the verdict, the hit code and the resolved type
of such a crossing are the ordinary ones and did not move. So sound transmitted through a wall,
a lined wall or a layered wall continues in the direction it arrived in, the later faces of a
layered wall are met at the incidence angle, and what stands behind the wall is met where and
how the incoming ray would have met it.

A leg entered out of a mass-law medium is not a bent one, and `APPLY_TRANSITION` does not write
the bent flag for it whatever the hit code says: the wave was held there.

**A sheet behind a held wave is a later face of the same wall.** A mass-law sheet on the far
face of a mass-law body that holds the wave, or inside it, is met as the same body meshed as a
traced solid would be met: the wave is a leak, it keeps the incidence cosine, and the event is

    near | sheet    Material::interact_with with the wave held, at the incidence cosine
    the sheet       its mass term
    sheet | far     Material::interact_with with the wave held, at the incidence cosine

each crossing with the interface gain of the medium it enters, the far medium's only where it
is newly entered. That is the chain of a mass-law sheet entered as a leak (*A sheet under
total reflection of its outer pair*), taken without a closed outer pair. The reflect port is
the held reflection of the near face, the other direction's with its sign reversed, so the two
ports of that face add up as they do at a bare face. The Snell cosine of near | sheet is formed
from two impedance mappings and is not used: at it, past the critical angle of the pair, both
faces are flat leaks, which are flux statements, and the near-to-far conversion of the
composite, `1 / Re(z cos θ)` with `z` a mapping of 1.5e-4, multiplied them into a gain.

Three gates, decided on the media and on the direction the
tracer holds, never on a value: the near medium carries `m > 0`; the wave in it is undeviated
(`!refracted`, which `SHEET_RESOLVE` hands the solver, and which `ray_mesh_interact` sets
always); and every layer carries `m > 0` while **at least one face** of the run
near | layers | far leads towards the lower index. One gate serves a single sheet and an ordered
stack: the solver then returns the chain of held crossings, one per face, with each layer's mass
term under the bonded-plate rule, and the held reflection of the first face. A sheet or a stack
inside a wall has the wall on both sides, so unless all permittivities are equal some face leads
down and the gate is met; a lining meets it wherever either of its faces leads down, which a
mass-law lining that is not rigid does at its far face towards air. Where no face leads down - a
wood lining on a concrete wall, both steps upward - the sheet stays on the composite at the Snell
cosine; what that leaves is issue 4. With a layer that carries no mass column the sheet or stack
stays on the recursion, which is issue 2.

Measured, levels without the mass term, scalar, W wood `eps 2.1609e-8` `tf` 0.2 as a wall of
100 mm, K concrete `eps 2.401e-9` `tf` 0.1 as a lining; W | K is closed from 19.5°:

| W wall, K lining | 10° | 19.4° | 19.6° | 30° | 60° | 75° |
| --- | --- | --- | --- | --- | --- | --- |
| the wall met first | −17.864 | −17.920 | −17.922 | −18.037 | −18.899 | −19.994 |
| two traced solids, W then K | −17.864 | −17.920 | −17.922 | −18.037 | −18.899 | −19.994 |
| the lining met first | −17.861 | −17.918 | −17.920 | −18.035 | −18.898 | −19.994 |

The two sides differ by the flat first faces of the two bodies (§5), 0.0025 dB at most. A rigid
sheet inside a rigid wall is on its three solids, two rigid sheets inside one on their four
solids in all six orders, and the reflect port on the face of the solid twin, each exactly;
types 1 and 2 per channel alike. Both tracers read alike. `ray_mesh_interact`'s own glued exit
forms the same rule, being the same function. Inside the cone in which a rigid wall admits a
wave, below 0.0084° for W, the leg is refracted, not held, and the closed-layer rule applies as
before; the step at the edge of the cone is the one of §5 and did not move.

**The reflect port follows the same rule**, under the same three gates. It is the reflection the
other direction sees at the same cosine with its sign reversed, which is Stokes' relation for one
face read from its two sides. It stays in the medium, so it carries no interface gain, and the
verdict handed back for a reflection is the ordinary magnitude test, not the undeviated one, so no
type code moved when it landed. For a lossless pair the two ports of the face add to one, in scalar
mode and in EM, where it is asserted on the TE/TM mean. The port is bookkeeping, as the transmit port is: a mass-law
plate is a lumped element with no wave inside, and the impedance step of the face is formed from
two permittivities that were each calibrated against air. What the rule guarantees is that nothing
is created and that one face is one face. Before it, wood (`tf` 0.2) to concrete (`tf` 0.1)
reflected 0.72 and transmitted 0.785 at 30°, a sum of 1.50.

One consequence to know. A mass-law solid with permittivity above 1, exited with no refracted
direction, takes the rule as well and reports the undeviated bit on the transmit pass, at normal
incidence too; its gain at normal incidence is unchanged.

### Bonded mass-law bodies are one plate

Two mass-law bodies within the co-location distance have no air gap, so they move with one velocity
and their transfer impedances are in series. What adds is the impedance ratio `X₀ = 10^(A/20)` that
`medium_gain` already forms per body, not the term in dB:

    TL = 20·log10(X₀₁ + X₀₂)

and the pair is one body in every other respect: the single-body angle shape is applied to the
summed `X₀` at the real incidence cosine, so both end points and the 20 dB per decade carry over.
A 35 dB panel on a 53 dB wall is 54.03 dB where the terms added in dB gave 88. A decoupled leaf is
meshed with a real gap and reaches the tracer as separate events with air between, where adding is
right, so co-location is the discriminator and no column is needed.

Three conventions complete the rule.

- *The zero sits on the sum.* Each body contributes its `X₀ = ((f/fRef)·M)^(m/20)` unclipped, below
  an argument of one included, and the plate is zero below a summed `X₀` of one. A body below its
  own knee therefore still adds its small mass, and the pair has no step where it crosses the knee.
- *`m = 0` is absent*, a gate on the column. A lining without the column (a carpet, an absorber)
  acts through its other columns and leaves the wall's term bit-identical. A layer with `m = 0`
  between two mass-law bodies separates them, so their terms add in dB, which is what a resilient
  underlay does. A traced solid whose leg is below the 1.5 mm mass floor is absent as well.
- *The charging telescopes.* The first body crossed pays its own term, by the unchanged code path,
  and each later one pays `TL(X_run + X₀) − TL(X_run)` with `X_run` the summed `X₀` of the bonded
  bodies crossed before it on this traversal. No event carries a factor above one, the product over
  the plate is the plate's term in either direction, and a ray that ends inside the second body has
  paid the first. Through a 35 dB panel then a 53 dB wall the events pay 35 and 19.03 dB; the other
  way 53 and 1.03 dB.

`Material::medium_gain` takes `X_run` as its `mass_bonded` argument; `Material::mass_X0` and
`Material::mass_TL` are the two pieces. How each of the three pairings supplies it:

| pair | first body | second body learns `X_run` from |
| --- | --- | --- |
| sheet \| sheet | first layer, in the solver | the solver's own run over adjacent mass-law layers; one event |
| sheet \| solid, entry | the sheet, at its crossing | `mtl_prev` holds the sheet, whose `thk` is a column; the buffer flag says it was crossed |
| sheet \| solid, exit | the solid, closing its leg in `SHEET_EVENT` | the sheet is handed the leg's `X₀` in the same event (`qd_slab_resolve`'s `mass_bonded`) |
| solid \| solid | the first solid, at the shared face | `mtl_prev` names the material, the buffer word carries the thickness code |
| wall \| sheet \| wall, a sheet inside a wall | the first half, then the sheet, both at the crossing | `mtl_prev` holds the sheet with its inside flag, whose `thk` is a column; the buffer word carries the first half's thickness code, read with the current material |
| anything deeper | as above | the same two words: the thickness code is an equivalent thickness that holds the bodies behind the one the words name (*Depth*, below) |

The shared face keeps its own crossing in all of them, with its impedance step, the pair's `tf`
and the undeviated rule. `att` and the coincidence term are interface gains, owed once per medium
entered, and still add in dB across a bonded pair; see issue 5.

**Depth.** The two words name one body behind the ray, or two where that body is a sheet inside
the current medium. What lies further back travels inside the thickness code. Bodies of one slope
`m` have `X₀ = ((f/fRef)·d)^(m/20)` with one power of the frequency, so their summed `X₀` is that
of a single body at every frequency, of the thickness

    d_eq = ( Σ (dᵢ · fRef_R / fRefᵢ)^p )^(1/p)        p = m / 20

in the material `R` the code is read with. No frequency enters it, which is what a word shared by
every frequency of a run may hold. `MASS_FOLD` forms it, from the leg that is closed, what
`mtl_prev` and the code already name, and the first layer of an ordered stack, at the three sites
that write a code: the transition out of a mass-law solid, the sheet crossed inside a mass-law
medium, and the ordered stack crossed there. `MASS_RUN` reads the code as it always did. With one
slope the plate is therefore exact at any depth, from either side, to the quantization of the
code: three and four solids, a lining on two solids, a lined wall holding a sheet, a solid in
front of such a wall, two sheets inside a wall, and bodies of unequal `fRef`.

*Mixed slopes.* A body whose slope is not the reading material's has no such thickness and is
left out: it is treated as not crossed, and the leg pays more than the plate would. A sheet
in `mtl_prev` is named by its column whatever its slope, so only bodies behind the named one are
affected. A stack of mixed slopes is exact from the side on which the bodies that have to be
folded agree and pays more from the other; the sizes are in §5. The decision whether to fold is
taken on the `m` column of two materials and never on a value.

*A sheet with its host's permittivity.* A sheet that is no boundary is merged into the host leg
(*A sheet in the previous-medium word*), except where host and sheet both carry the mass column:
a mass-law host has no bounce series for the merge to protect, and a leg carried through would
pay its own term without the sheet in its plate. Such a sheet cuts the leg like any other sheet
inside a wall, and the two halves and the sheet are three bonded bodies.

**Fitted slopes.** With `m = 20` and one material, `X₀` is linear in thickness and the rule is the
summed thickness exactly. For a fitted slope other than 20 it is the series-impedance value, which
differs from the table's own thickness extrapolation by `0.301·(20 − m)` dB for two equal halves
(+2.41 dB at `m = 12`, −3.01 dB at `m = 30`). The +6 dB per doubling is the physical one. It is
reachable wherever one material is crossed as two bodies: a sheet on a solid of its own material,
two sheets differing in `thk`, and a wall cut in two by a mass-law sheet inside it, whatever that
sheet's permittivity. Two solids of one material merge into one leg, and two identical co-located
sheets collapse to one.

**A bounce inside a mass-law body is followed**, as before, and pays per traversal: a leg that
begins at a reflection starts a new traversal with `X_run = 0`, so the way back through a pair pays
the plate once more, the body the bounce is in its own term and the other the increment. Where the
mass term is large the bounce is worth nothing (0.000 dB on the transmitted level of 20 cm of
concrete from 125 Hz up). Where it is small it carries real energy in the model: a 22 mm wood panel
as a traced solid, with faces that reflect 0.9 from inside, gains +0.09 dB at 1 kHz, +1.7 dB at
500 Hz and +7.2 dB below its knee at 398 Hz. That regime is the model's weakest - a panel below
its knee is physically almost transparent while the model charges two `tf` leaks - and dropping the
bounce would move the result further from the physics.

### The state words

Three signed `short` words per ray. The material index is 14 bits, 16383 materials, and bits 14
and 15 of each word are flags. The state must not grow: a production run carries on the order of
150 million rays.

| word | bits 0 to 13 | bit 14 | bit 15 |
| --- | --- | --- | --- |
| `mtl_ind_current` | the medium the ray is in, 0 = air | *bent*: the leg follows a bend the refracting tracer traced; zero in air | *resolved*: a slab reflection was collapsed into the series and the ray is on its return path |
| `mtl_ind_prev` | the medium the current leg was entered from; after any sheet event the sheet crossed or reflected off, a glued exit into air included | with a sheet in the word, *inside*: the sheet lies inside the current medium. With a solid in the word, *real invariant*: the shared face was evaluated at the recovered cosine of a wave whose transverse invariant is real | *known non-parallel*: the entry and exit faces are not a slab pair |
| `mtl_ind_buffer`, bit 15 clear | a deferred medium from nested or overlapping geometry; with bit 14 set, the *stack code*: the layer of an ordered stack behind the ray that the previous-medium word does not name | *followed*, with a zero index: the leg began at a reflection that left the bounce series to the tracer | |
| `mtl_ind_buffer`, bit 15 set | a thickness code in bits 0 to 14 | part of the code | *mass-crossed*: the leg was entered by crossing the bonded mass-law body in `mtl_prev` |

All three words are decoded in one commented block at the top of the ray loop in
`ray_state_update`, and nowhere else in that function. `ray_state_update`, `ray_progress` and
`ray_commit` reject more than 16383 materials, and `calc_diffraction_gain` a material index above
that.

**The bent flag** says that the leg the ray is on was entered by a refraction the tracer followed.
It is written where a medium is entered with type 2 or 5 and the hit code carries no
total-reflection bit: an entry from air, a glued entry and a transition out of a solid without
the mass column into another solid. A transition out of a mass-law solid never writes it. It is
carried while the ray stays in that medium - a back reflection, with or without the resolved
flag, a sheet inside the medium on both passes, a transparent transition, a merged host and a
leg with no hit - and it is zero in air. Types 1 and 4 never set it, so every word the
undeviated tracer writes is what it was. It is decided on a value, the total-reflection verdict
at the base frequency, where the other flags are decided on media: it describes the polyline,
which is itself bent at the base frequency and shared by every frequency of a run. A ray
launched inside a medium and a caller without state have no flag and are treated as before.

**The flag has to stay zero while the ray is in air.** `ray_subdivide_flag` and the subdivision
loop in `ray_progress` compare the current word whole, unmasked, to decide whether a ray is
outside a medium. `ray_commit` and `calc_diffraction_gain` mask the word to its 14-bit index.

**No spare bit is left** in any of the three words. The buffer word without its flag bit had
one, bit 14, and the followed mark took it. The code points are another matter: bit 14 of the
buffer word with a non-zero index beside it was written by no site until the stack code took
it, and nothing else of that kind is left - bit 14 with bit 15 belongs to the thickness code.

**The followed mark** says that the leg the ray is on began at a reflection inside the medium
that did not sum the bounce series: buffer word `0x4000`, bit 14 set, bit 15 clear, index zero.
A marked leg forms no series, at a bare face, at a transition and at a sheet; the convention is
*One verdict per body* below. Both reflection writers set it, the bare-face branch of the
reflect pass and `SHEET_EVENT`, exactly where they did not set the resolved flag; so a ray
reflected inside a medium is resolved or followed, never neither. It is carried where a leg
carries on - no hit, a transparent same-material transition, a merged host - and gone wherever
a leg begins by a transmission. It shares the word with the deferred medium: a nested object
met on a marked leg stores its medium there and replaces the mark, and that leg judges its
series for itself again, against whatever the previous-medium word holds (§5). It is the base
frequency's, as the resolved flag is: it is the same verdict, seen from the other side. Nothing
reads it but the three series gates. The input range check of `ray_state_update` masks the
buffer word to 14 bits unless its flag bit is set.

**The stack code** is the same bit with an index beside it, `0x4000 | layer`, and is not a
mark: the leg began by crossing an ordered stack whose last layer is coherent, the
previous-medium word names that last layer, and the index is the other one. The mirror behind
the ray is then both layers, backed by the current medium, and the leg judges its series
against them like any other leg: summed where `ρ` reaches the threshold, followed where it does
not. `SHEET_EVENT` writes it on the crossing, decided on the media; it is carried where a leg
carries on and cleared by every reflection, which writes zero or the mark as after any other
leg; a deferred medium that has to be stored replaces it, as it replaces the mark (§5). Only
the host's series reads it. A stack whose last layer is mass-law writes the thickness code
there, as before, and is a half-space from behind.

**The inside flag** qualifies a sheet in the previous-medium word. Set, the sheet has the current
medium on both sides - a metal layer inside a wall - and what is behind it is that medium. Clear,
the sheet is a lining and what is behind it is air. `SHEET_EVENT` writes it wherever the two sides
of the event are one medium, on a crossing and on a reflection, and it travels with the word.

**The real-invariant flag** is the same bit, read where the word names a solid; which of the two
meanings applies follows from what the word holds, and every reader of the inside flag tests for
a sheet first. `APPLY_TRANSITION` sets it where all of these hold: the transition leads into a
medium, not into air; `REPLACE_BY_VBS` evaluated the face at a recovered cosine; that cosine was
solved for a real invariant (unnamed, or with a lossless entry medium named); and the solid the
ray leaves is lossy. It is decided on the media and the state words and never on a computed
value, because the words are shared across the frequencies of a run. It travels with the word,
through a back reflection and a transparent transition, and is overwritten by the next sheet
event or transition. Only the recovery reads it. Every leg that existed before the flag did
keeps its word bit for bit: a lossless solid, a face evaluated at a real angle and an exit into
air all leave it clear.

**The thickness code** is a logarithmic code of a normal thickness,
`d = 10^(−4 + 6·(k − 1)/32766)` m for `k = 1 … 32767`: 0.1 mm to 100 m in steps of 0.044 %, which
is 0.0038 dB on a mass term at `m = 20`. The thickness is formed from geometry and material
columns and from no frequency, so one code serves every frequency of a multi-frequency run, where
the state words are shared and only the refracted distance and direction are per frequency.
`MASS_CODE` writes it, from the thickness `MASS_FOLD` hands it, and `MASS_RUN` reads it. What the
thickness belongs to follows from the previous-medium word:

| `mtl_prev` holds | the code is | read with |
| --- | --- | --- |
| a traced solid | the normal thickness of that solid's leg, enlarged to the equivalent thickness that also holds every bonded body of its slope crossed before it and named by the words at that face | the previous material |
| a sheet with the inside flag | the same for the current medium's leg in front of the sheet, with the first layer of an ordered stack; zero if that medium carries no mass column | the current material |
| a sheet without it | zero; the sheet's own thickness is a column | |

For two bodies the equivalent thickness is the leg's own, bit for bit. The code is quantized again
at every bonded face it is carried across.

The flag and code are written where a mass-law solid is crossed into another medium
(`APPLY_TRANSITION`) and where a mass-law sheet is crossed into a medium (`SHEET_EVENT`); kept
where a leg carries on - no hit, a transparent same-material transition, a merged host; and
cleared wherever a leg begins any other way, every reflection included; after a reflection the
word is zero where the series was summed and the followed mark where it was not. So a set flag
always means the neighbour was crossed on this traversal, which is what lets a bounce be told
from a first pass without history. Every reader of the deferred medium sees zero while the flag is set, the input
range check exempts a flagged word, and a deferred medium that has to be stored replaces the code,
in which case that leg pays its own term in full (issue 8).

### A sheet in the previous-medium word

`SHEET_EVENT` writes the sheet into `mtl_prev` on every transmission that restarts a leg - a glued
entry, a sheet inside a medium, an ordered stack (its last layer), a glued exit - and on a
reflection off a sheet from inside a medium. An internal back reflection at a bare face, a
transparent transition, a merged host and a leg with no hit carry the word on, which is how a
bounce still holds the sheet it entered by. A sheet has no volume and encloses nothing, so the word
is then a boundary and never a medium. What each reader makes of it:

| reader | reads the word as | with a sheet in it |
| --- | --- | --- |
| the host's series, `SLAB_AIRY_FACTOR`, on a leg that carries no followed mark | the mirror behind the ray | a layer of the sheet's thickness, backed by the current medium under the inside flag and by air without it; with the stack code in the buffer word, both layers of the stack; a mass-law sheet a half-space |
| `MASS_RUN` | the bonded body crossed, under the buffer flag | the sheet, whose thickness is a column, plus under the inside flag the part of the current medium in front of it |
| the recovery | the entry medium | none named: it solves for the real transverse invariant; see the in-medium angle. A solid in the word is named unless its real-invariant flag is set |
| a glued exit, `sheet_case 3` | nothing: beyond its sheet is air, whatever the word holds | not read |

**Behind a sheet is air or the host, and nothing else is stored.** A lined body nested inside
another medium is not a use case in EM or in acoustics, and a sheet between two solids is not a
geometry the intersector reports, so the medium behind a sheet in this word is always one of two
and one flag names it. No medium index is carried for it.

**A glued exit exits into air.** A glued exit, `sheet_case 3`, is two faces hit together, the
solid's exit face and a sheet, and what lies beyond the sheet is not in the hit. It is air,
whatever the state words hold: for the composite, the outer-pair total-reflection verdict, the
continuation direction, the backing of the sheet in the host's series and the medium the ray
continues in. It is what `ray_mesh_interact`'s own composite assumes for every glued exit. The
previous-medium word is not read for it. That word names what the host was entered from, or the
sheet it was entered through, and either is behind the ray: it is the mirror behind the ray in the
host's series and nothing else. A wall of two solid layers with a coating on its far side,
`air | A | B | coating | air`, meets the coating with `A` in the word, with or without the
real-invariant flag; a wall lined on both faces and a bounce returning to the glued sheet it
entered by meet it with a sheet there; all of them exit into air. The medium a host was entered
from lies beyond its exit sheet only for a lined body nested in that medium, which is not a use
case. Every glued exit leaves `mtl_prev` holding the sheet it crossed, `mtl_current` zero and the
buffer word zero, with resolved type 71; its reflect pass leaves `(sheet, host, 0)` with the resolved flag on the host where the series was summed and `(sheet, host, followed)` where it was not, and type 207. Nothing reads `mtl_prev` while the ray is in air.

In the same words, the second sheet of a lined wall is `(prev = S1, cur = host, buffer zero or the
bare mass flag)` and a bounce returning to its entry sheet is `(prev = S, cur = host)` with the
resolved flag and a zero buffer word where the reflection collapsed the series, and with the
followed mark in the buffer word where it did not. With one lining on both faces the first leg
and the bounce differ only in those flags, and they do not have to be told apart for the far
medium.

**A sheet between two solids is not a valid geometry.** The intersector reports two faces and has no
triple-hit detection, so `solid | sheet | solid` does not reach this file as one event. If it occurs
anyway the ray continues in air, and the next exit event corrects the current medium from the face
it leaves through, with the fix-flag set. A sheet inside one solid is valid and common, a metal
structure inside a wall, and so is a glued entry, a coating on the room side.

**A resolved ray gets no series at a sheet.** A reflection that collapsed the host's bounce series
flags the ray as resolved, and the close-out of that leg skips the series at a sheet event as
`APPLY_TRANSITION` skips it at a bare face. The sheet's own composite and the host's traversal are
still applied. A transmission through a sheet starts a new leg and clears the flag.

**A ray reflected off a sheet is resolved exactly where the host's series was summed there.** A
host leg that closes at a sheet either has its bounce series summed, `ρ ≥ eps`, or leaves the
bounces to the tracer. Summed, the reflected ray carries the whole series, is flagged resolved,
and its next reflection is killed, as after a resolved back reflection at a bare face. Declined,
the reflected ray is followed bounce by bounce and carries the followed mark, so the events
that close its legs form no series. The sheet it reflected off is in the previous-medium word,
which is the mirror then behind it; with the mark set nothing reads it as one. The type code of
the event does not tell the two apart: bit 2 is part of the thin-sheet marker and is set either
way; the resolved flag of the current-medium word does.

**How the host's series treats the sheet at a sheet event.** The event is the sheet composite
`host | sheet | far` times the host's series `1/(1 − r_ahead·r_behind·φ²)`. `r_ahead` is the sheet
resolved by the solver as a layer of its own thickness, seen from the host, with the far medium
behind it; `r_behind` is whatever the host was entered through, a half-space for a bare face and
for a sheet a layer backed by air or, under the inside flag, by the host. A lining lies beyond
the host's face and is referenced to it as it stands; a sheet inside the host is centred on its
face, so its mirror carries the one-way factor of the host over the sheet's thickness, ahead of
the ray and behind it (*Where the layer of a sheet lies*). That factorization is
the whole stack and not an approximation of it for a sheet on a face: a coated wall and a wall
lined on both faces sit on the characteristic matrix of the stack end to end. A wall with a sheet inside it is two
series, each half taking the sheet as a layer backed by a half-space of the host, and what the
other half's outer face returns through the sheet arrives as further paths; see the next
subsection. At an
ordered stack the mirror ahead is both layers, backed by the host: the hit names them. The
mirror behind a ray that has crossed a stack is both layers as well: the previous-medium word
names the last one and the stack code in the buffer word the other (the state words), so each
half of the host sums its series against the whole stack, and a wall with a stack inside it
costs the paths of a wall with one sheet inside it. A stack whose last layer is mass-law is a
half-space from behind by the mass-law convention.

### A layered wall is a sum of paths

A traced body is resolved on its own. Its series is `1/(1 − r_behind·r_ahead·φ²)` over its own
thickness, with whatever bounds it taken as it stands at that face: a neighbouring solid as a
half-space, a sheet as a layer of its own thickness backed by air or by the host. Nothing a
neighbouring solid's far side returns is in that series. It is not dropped: it arrives as rays.

**The decomposition is exact.** For `air | A | B | air` splitting off the last layer gives

    T = T(air | A | B half-space) · φ_B · t(B | air) / (1 − R_behind · r(B | air) · φ_B²)

with `R_behind` the reflection of the layer `A` backed by air seen from `B`. Writing
`R_behind = r(B | A) + X`, with `X` what comes back through `A` from its front face, the series
expands around the half-space mirror into a geometric sequence in `X`, and each term is a path
the tracer follows: the ray reflected at `B`'s far face, its series collapsed, crosses back into
`A`, which starts a new leg and clears the resolved flag; it is reflected at `A`'s front face,
`A`'s series collapsed; it crosses into `B` again and leaves with `B`'s series once more. The
reflections that would repeat a collapsed series are the ones a resolved ray is killed at, so
there is exactly one path per order, 3 + 4n events for order n. The reflected field is the same
sum from the other end. A wall with a sheet inside it is the same statement with the sheet
crossed in place of the shared face.

**So the first path through a layered wall is not the wall, and the sum is.** With every series
resolved the straight path is off the characteristic matrix of the whole wall by exactly what
the layer mirror would add, in dB at 10°, 20°, 40°, 60°, 75° and 85° for `A` 50 mm in front of
`B` 100 mm:

| `A` in front of `B` | 10° | 20° | 40° | 60° | 75° | 85° |
| --- | --- | --- | --- | --- | --- | --- |
| `eps 5.24` / `eps 1.50 − 0.40j` | +0.12 | +0.16 | +0.30 | +0.31 | −0.00 | −0.51 |
| `eps 5.24 − 0.63j` / `eps 1.50 − 0.40j` | +0.08 | +0.11 | +0.23 | +0.26 | +0.02 | −0.37 |
| `eps 5.24` / `eps 1.50` | +0.30 | +0.35 | +0.46 | +0.14 | −1.44 | −4.88 |
| `eps 5.24 − 0.63j` / `eps 1.50` | +0.22 | +0.26 | +0.37 | +0.15 | −1.01 | −3.62 |

and for a 6 mm `eps 5.24 − 0.63j` sheet between two 50 mm halves −0.06, −0.08, −0.13, −0.12,
+0.24 and +0.93 dB in the host `eps 1.50 − 0.40j` and −0.07, −0.07, −0.05, +0.22, +1.06 and
+2.38 dB in `eps 1.50`. The next path is 15 dB to 35 dB below the first for a lossy second
layer and 3 dB to 31 dB below for a lossless one, the small figures towards grazing. Summed, the
paths are on the matrix of the whole wall: through the intersector their limit is within
6.5e-4 dB and 0.024°, and through `ray_progress` with 20 interactions the five transmitted and
six reflected paths of a ray sum to within 0.0016 dB.

**What a run needs for it.** Each order costs four interactions, and the paths have to survive
`min_gain_dB`. A lossless second layer near grazing converges slowly, 3.0 dB per path at 85° for
`eps 5.24` in front of `eps 1.50`: +1.37 dB with 20 interactions, +0.04 dB with 60. The later
paths leave the wall later and laterally displaced, as the bounces of a single slab do where its
series is re-emitted.

**Resolved or followed, never neither.** A series is summed at the event that closes the leg, or
its bounces are the tracer's. Both are exact, and they are exact together in one wall: a bare
wall, a wall coated on either side, two solids and a wall with a sheet inside it give the wall
whether every series is summed, every one declined, or the threshold falls between them. The
followed sum of a coated wall and the summed one are both on the matrix.

### One verdict per body

Whether a body's bounce series is summed or followed is one verdict. It belongs to the cavity -
two mirrors, a thickness, an angle - and not to the leg a ray happens to be on, and it is taken
once: by the first event that closes a leg in the body, which is the one event at which the
state names both mirrors, the face ahead from the hit and the one behind from the
previous-medium word. `ρ = |r_ahead·r_behind|·|φ|²` is symmetric in the two, so with the true
mirrors every later leg in that body would return the same verdict. Measured: the threshold at
which the first leg stops summing equals the true pair's `ρ` of the later leg to four digits, on
384 rows.

**For every leg.** The verdict travels in the state. Summed, the reflected ray is resolved: the
event that closes its next leg forms no series and its next reflection is killed. Declined, the
reflected ray carries the followed mark (see the state words): the event that closes its next
leg forms no series either, and its next reflection is followed and marked again. A
transmission into another body, or through a sheet, starts a new leg and a new verdict, with a
word that is true again. So no event ever forms a series against a mirror the state does not
hold. That matters because after a back reflection the mirror behind the ray is the face it was
reflected off, which no word holds: a bare-face reflection does not write the previous-medium
word, and must not, because the recovery reads that word as the medium the leg was entered from
and a reflection does not change the wave in the medium. One word cannot be both. Inheriting
needs neither.

Inheriting "followed" cannot produce a wrong power. Following a bounce is exact whatever the
mirrors are; the one error mode is summing against a wrong mirror, and a marked leg does not
sum. Where a reflected ray meets a face that is not its body's other slab face - a side wall, a
face that is coated in part - the bounces are followed where a fresh verdict might have summed,
which costs rays and nothing else.

**For every frequency.** The traced paths of a multi-frequency run are the base frequency's: a
reflected ray is resolved or followed as that frequency decided. A frequency that formed its
own verdict on the first leg would count its series twice where it sums and the base follows,
and not at all where it declines and the base sums. `ray_state_update` therefore takes the
verdict as an optional per-event byte, `verdict_inN`, and reports its own in `verdict_outN`;
where a verdict is handed in, the series is summed exactly where it says and `eps` is not read.
`ray_progress` holds one such vector for the duration of a call, written by the base-frequency
call of each pass and read by the calls of every other frequency. Summing a weak series is
exact, so a frequency whose own `ρ` is small loses nothing by it; a series near a pole is
summed like any other, with or without a verdict: the factor is large, the port it multiplies
is small, and the product is the finite closed form. The later legs need no argument: the
resolved flag and the followed mark are state, and every frequency reads them.

**The mirror behind a solid is not handed to the solver as a layer, and the thickness code is not
written for it.** The buffer word could carry the first solid's thickness to the second one's
series at no cost in state, and the first path through two solids would then be the wall. It
was measured and not taken, for three reasons. The further paths would then count the coupling
a second time unless the returning ray's reflection is killed, which needs a marker and a rule
for a wall with one series summed and the other followed. The reflected field cannot be put on
one path at all: the first solid's reflect port would need the second one's thickness before
that solid is crossed. And three solids deep the state names one body behind the ray and not
what backs it, and a layer backed by air is then usually worse than the half-space: on the third
solid's series at 85°, +6.68 dB against +1.81 dB for `eps 5.24 | 1.50 | 5.24` and −2.59 dB
against −1.15 dB for a lossy triple. The thickness code stays what it is, the surface mass of a
mass-law body.

**Which wall a sheet inside a host is.** The sheet has no thickness in the mesh, its composite
takes its thickness of host out of the transit, half of it on either side of its face, and both
host series run over the meshed legs against a mirror that says so. Summed over the paths,
`air | host 50 | sheet 6 mm | host 50 | air` is host 47 | sheet 6 | host 47 on both ports: the
limit of the paths through the intersector is within 1.7e-4 dB and 0.005° of it, and the sums
through `ray_progress` read 0.0000 dB and 0.001°.

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
| Walk-off traversal factor against `diag_medium_walk`, 6 angles, phase | 0.01 deg | 6e-15 deg |
| Walk-off traversal factor, magnitude on the same rows | 1e-12 | 0.000e+00 |
| Walk-off form separated from the raw-path form, 5 angles past normal | > 1.0 deg | 3.157 to 26.174 deg |
| Generalized single-layer builders against `diag_cm_stack`, `μ ≠ 1`, 4 stacks × 2 modes × 7 angles, real and complex cosine | 1e-14 | 6.75e-16 |
| Magnetic sheet, isolated and glued entry, both channels, complex | 1e-9 | 1.33e-15 |
| `ε↔μ` swap on a sheet in air exchanges the TE and TM ports | 1e-9 | 5.55e-16 |
| The same swap leaves the power-averaged gain invariant | 1e-12 | 7.77e-16 |
| The two ports are separated enough for the exchange to be a statement, 4 angles | > 1e-3 | 0.0188 to 0.2995 |
| Impedance-matched dense sheet, `ε = μ = 4`: reflect port at normal incidence | 1e-15 | 0.000000000 |
| The same, transmit port magnitude and phase against `−k₀·thk·(n−1)` | 1e-9, 1e-3 deg | exact |
| The same, reflection independent of polarization at every angle | 1e-14 | 0.00e+00 |
| Traced `air \| slab \| air` chain through a magnetic solid against the generalized analytic slab, 3 solids × 5 angles | 1e-6 dB | 0.000000 dB |
| The traced cosine of a magnetic solid against the map the recovery inverts | 1e-9 | 1.11e-16 |
| `ε↔μ` swap on a traced solid leaves the traced cosine unmoved | 1e-12 | 0.000e+00 |
| The same swap exchanges the two channels of the analytic slab | 1e-14 | 4.44e-16 |
| The two channels are separated enough for that exchange to be a statement, 40° to 75° | > 1e-2 dB | 0.024 to 0.930 dB |
| Matched solid, `ε = μ = 2`: channels equal at every angle, and unit transmittance at normal incidence | 1e-14, 1e-12 | 0.00e+00, exact |
| Ordered stack with one magnetic layer against `diag_cm_stack`, both ports, both orders, scalar and EM | 1e-9 | 5.66e-16 |
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
| Lumped interface gain: isolated, glued entry, embedded and glued exit, each gain owed paid once, 4 stacks × up to 6 angles | 1e-9 | 6.1e-16 |
| The sheet's gain as one factor on the transmit port, against the same sheet without the column | 1e-14 | exact |
| An embedded sheet against the same host without its `att` column | bit-identical | bit-identical |
| The reflect port with the sheet's `att` against the reference whose echo carries `sqrt(G)`, complex; and against the same sheet without the column | 1e-9; > 1e-3 relative | within bound; told apart |
| `tf` sweep on a sheet, 6 values × 3 stacks, both ports, EM and scalar | 1e-9 | 3.2e-15 |
| Energy of a lossless sheet per channel under `tf`, 6 values × 4 angles | 1e-12 | 8.9e-16 |
| `tf = +1` closes the reflect port, `tf = −1` the transmit port | exactly 0 | exactly 0 |
| Resonance on the sheet paths: resonant sheet, resonant far solid, resonant host, both ports, EM and scalar, against the reference with the pole in the permittivity | 1e-9 | 2.3e-15 |
| The embedded row of it in power, with the decay of its 100 mm resonant host leg written from the medium, −3.70 dB | 1e-9 | 2.8e-16 |
| The same rows against a reference with the resonance dropped | > 1e-3 | 0.024 to 0.29 |
| Resonance-matched sheet, `ε` 4 with a pole of −3: both reflections, and the transmit port against 1 | 1e-12 | 7.8e-16 |
| A resonant medium against its twin without a pole - a lossy pole against the same conductivity, lossless poles against `ε` 1, 2 and 5.24 - as a sheet, a traced solid summed and followed, a resonant sheet inside a host and a sheet inside a resonant host, every path, types 4, 5, 1, 2 | 1e-9 | 0, bit for bit |
| The matched sheet and its traced solid in air, 6 and 37.5 mm, every path | 1e-12 | 1.6e-14 |
| Both ports together on walls with a pole: lossless ones within the plain wall's residual; lossy ones, the strong pole that takes `ε` 2.5 to −1.04 − 0.26j included, at most one | 2.5e-4; 1 | 1.4e-4; 0.98 |
| A resonant sheet against its traced solid, both ports, and the followed series against the summed one, beside the plain twins that give the tracer's own residual | 0.02 dB, 0.05°; 0.06 dB, 0.1° | 0.013 dB, 0.04°; 0.053 dB, 0.085° |
| A resonant solid through `ray_progress` on the matrix of `ε + pole(f)` at every frequency of a band, either frequency the base, both tracers, float | 1e-3 dB, 0.01° | 1e-4 dB, 0.001° |
| `refractive_index` against `Re √(ε + res)`; `medium_gain` against the conductivity twin, or against `8.686 k₀ \|Im n\|` where `Re(ε + res) < 0` | 1e-12; 1e-4 dB | within bound |
| Ordered stack with `att` on both layers, with a resonant layer, and with an `alpha` layer, against `diag_cm_stack`, both orders, both ports; the reflect port of the `att` stack against `diag_sc_stack` with a gain per layer | 1e-9 | 4.5e-16 |
| The same `att` stack: the transmit gain over its twin without the column is the product of the gains, the two taking two forms of one product | 1e-12 | at the rounding floor |
| The same: the gain is worth something on the reflect port | > 1e-3 | holds |
| `alpha`: `6 mm at 100 dB/m \| 6 mm at 0` as a stack against one 12 mm sheet at 50 dB/m, both ports, to 85° | 1e-9 | 4.6e-16 |
| Ordered stack whose second layer carries `tf` or `m`, against the scattering reference, both orders, both ports, scalar and EM | 1e-9 | 4.6e-16 |
| Host layer bounded by a mass-law sheet, against the half-space series | 1e-9 | 4.0e-16 |
| The same with a sheet carrying `tf = 0.25`, against the layer series with `tf` | 1e-9 | 3.8e-16 |
| The same with a sheet carrying `alpha = 1e-6 dB/m`, against the layer series | 1e-9 | 2.7e-10, which is the alpha itself |
| Magnetic sheet through `ray_mesh_interact`'s own composite against the generalized analytic slab, transmit | 1e-6 dB | 0.000000 dB |
| The same, both ports per channel and labelled, all six thin-sheet materials × 8 angles | 1e-9 | 1.6e-15 |
| `ε↔μ` swap through `ray_mesh_interact` exchanges VV and HH, both ports, in a real incidence plane | 1e-9 | 1.4e-15 |
| `att`, `tf ±0.5` and a resonance on a sheet through `ray_mesh_interact`, both ports per channel | 1e-9 | 1.7e-15 |
| Neighbour `tf` by the pair rule, glued entry and glued exit, both ports, EM and scalar, 9 `tf` pairs × 4 angles | 1e-9 | 1.2e-15 |
| A sheet of its host's permittivity on a `tf`-bearing wall against the bare wall, lossless and lossy: complex in scalar mode and per channel in EM | 1e-9 | 1.0e-15 |
| A coated wall's `tf` on the transmit gain against the reference: −0.055082, −0.067695 dB at `tf +0.5`, −1.631402, −1.638190 dB at `tf −0.5`, at 0° and 45° | 1e-6 dB | exact |
| Scattering-form reference against `diag_cm_stack` at `tf = 0`, two layers with a resonance and `alpha` | 1e-13 | 6.0e-16 |
| Scattering-form reference against `diag_sheet_composite` with `tf` per medium, one layer | 1e-14 | 4.0e-16 |
| Ordered stack with `tf` on one or both layers, 5 values, both orders, both ports, scalar and EM | 1e-9 | 1.3e-15 |
| Energy of a lossless stack per channel under `tf` | 1e-12 | 1.1e-15 |
| Transmit port of a `tf`-bearing stack invariant under layer order, reflect port not | 1e-14, > 1e-3 | 1.0e-15 |
| `tf = −1` on a layer closes the transmit port, `tf = +1` on every layer the reflect port | exactly 0 | exactly 0 |
| `tf`: two 6 mm sheets of one permittivity and one `tf > 0` as a stack against one 12 mm sheet, both ports, to 85° | 1e-9 | 4.7e-16 |
| Host layer bounded by a `tf`-bearing sheet, and a `tf`-bearing host bounded by a sheet, against the layer series with `tf` by the pair rule | 1e-9 | 5.6e-16 |
| The same with `tf = 1e-9` against the `tf`-free layer series | 1e-8 | 1.0e-10 |
| Mass-law reference at one layer against a single mass-law sheet through the library | 1e-9 | 1.2e-16 |
| Ordered stack with a mass-law layer, with `tf` on it, and with two mass-law layers, both orders, both ports, scalar | 1e-9 | 6.7e-16 |
| Transmit port of a mass-law stack invariant under layer order, mass-law layers of permittivity 1.5 and above, no closed face | 1e-14 | 1.2e-16 |
| Mass term as one real factor: `m 20` against `m 10` on the same layer | 1e-9 dB | exact |
| Reflect port of a mass-law stack independent of `m` | 1e-15 | exactly 0 |
| An air-like coherent sheet in front of a mass-law sheet against that sheet alone, both ports | 1e-12 | 1.2e-16 |
| A lossless-permittivity mass-law stack reflects and transmits no more than it received | ≤ 1 | 0.416 at most |
| Reflect port of a lossy sheet on a rigid wall, with and without a leaking wall, scalar, to 70° | 1e-9 | 7.4e-16 |
| Reflect port of a coated solid past its critical angle, TE and scalar, to 80° | 1e-9 | 1.9e-15 |
| The same, TM, where the far face reflects above one | 1e-9 | within bound |
| Mass-law panel on a rigid wall, both with `tf` and `att`: transmit in power and complex, reflect, entry and exit | 1e-9 | within bound |
| `tf` leak through a coated solid past its critical angle with its internal series, EM and scalar, power and complex | 1e-9 | 4.5e-16 |
| The same, scalar: `\|R\|² + \|T\|²` of the lossless stack, with no reference | 1e-12 | 1.000000000000 |
| Glued entry against glued exit, lossless solid, scalar and EM gain, 0° to 80° | 1e-9 | 1.6e-15 |
| The same, lossy solid `eps 1.50 − 0.40j`, against the closed form `sec²(arg η)` per channel | 1e-9 | 2.2e-15 |
| The same, the residual itself: TE and EM gain to 80°; 0.073946 dB at normal incidence | 0.50 dB, 0.25 dB | 0.462 dB, 0.200 dB |
| Leak reciprocity, a mass-law panel on a rigid wall as a sheet, entry against exit with the wall's interface gain removed, to 70° | 0.02 dB | 0.0102 dB |
| The same with the recorded flat-first-face term removed | 1e-9 dB | 2.3e-12 dB |
| The same face as two traced solids through `ray_mesh_interact`, both directions | 1e-9 dB | 0.000000 dB |
| A wave refracted into a mass-law solid against the same solid without the column | 1e-12 | exact |
| Closed outer pair with no `tf`, sheet rarer than its host, EM and scalar, and the acoustic counterpart | exactly 0 | exactly 0 |
| Coherent lossless sheet rarer than the invariant on a rigid wall: the tunnelling sheet on the pad, power and complex, 40° to 75° | 1e-9 | 3.7e-16 |
| Leak into a rigid wall through a lossy absorber without `tf`, with its series, complex | 1e-9 | 1.2e-16 |
| Step from the open to the closed far face at the critical angle of the outer pair, ±1e-5°, scalar, TE, TM | 0.01 dB, 0.5 dB | −0.0024, −0.0024, −0.0231 dB |
| Scalar closed face: five porous rows of the acoustic table as a box on concrete against the sheet, summed and followed | at most what it received; 0.02 dB, 0.2° | largest −0.06 dB; 0.0001 dB, 0.001° |
| Closed face: both ports of a sheet under a closed outer pair, 6,552 rows, scalar, TE, TM | sum below 1 + 1e-6; each port on the reference at 1e-9 | 0 rows above; 1.4e-14 |
| A rare sheet losing its loss under a closed outer pair, ε″ 1e-6 against 0, 1 to 200 mm, both ports | 1e-3 dB | 0.0000 dB |
| The sheet against its traced solid on a rigid wall with `tf`, lossy, `tf` 0.5, −0.5 and none, summed and followed | 0.02 dB | 0.0045 dB |
| A lossy layer with a closed plate from its two sides, stack and traced laminate, scalar, VV, HH | 1e-6 dB, 1e-3 dB | 0.000000 dB |
| Wood panel on concrete across the critical angle of the outer pair, 0.0028° | 0.05 dB | 0.022 dB |
| Bonded mass, sheet on sheet of one material (8 mm \| 12 mm) against one 20 mm sheet, complex, both orders, to 75° | 1e-9 | 3.1e-16 |
| The same, reflect port against the single sheet | 1e-9 | 1.1e-16 |
| Bonded mass, sheet on sheet of two materials (`m` 20 and 30), mass paid against the plate, both orders | 1e-9 dB | 2.1e-14 dB |
| Bonded mass, sheet on a solid of its own material against that solid with the sheet's mass folded into `fRef`, complex, entry and exit | 1e-9 | 1.7e-13 |
| Bonded mass, wood panel on concrete with `tf` and `att`, mass paid, entry and exit, and the two against each other | 1e-9 dB | 1.4e-14 dB |
| An `m = 0` lining on concrete: the concrete pays its own term, entry and exit | 1e-9 dB | 3.6e-14 dB |
| Bonded mass, two traced solids, wood and concrete, both directions, through the thickness code | 0.01 dB | 4.3e-14 dB (100 mm is a code point) |
| The same, wood 60 mm \| wood 40 mm against one 100 mm solid, both orders | 0.01 dB | 7.1e-4 dB |
| The first body crossed pays its own term, every solid \| solid row | 1e-9 dB | exact |
| The buffer word carries its flag after a bonded shared face and no mass flag after a reflection, where it is the followed mark | exact | exact |
| A bounce in the second body: the reflection closes the first traversal at the plate, the way back pays the plate once more, the body it starts in its own term | 0.01 dB, 1e-9 dB | within bound, exact |
| Ports of a mass-law face from inside, lossless wood \| concrete through `ray_mesh_interact`, 0° to 70°: reflect plus transmit, scalar | 1e-12 | 1.1e-16 |
| The same: reflectance equal to the other direction's, and the sign reversed | 1e-12 | 1.1e-16 |
| The same: reflect plus transmit on the EM gain | 1e-12 | exact |
| Wall lined on both faces, second event, both ports against the composite with air beyond, 3 linings × 3 angles, scalar | 1e-9 | 1.2e-15 |
| The same against the same exit entered through a bare face, scalar and all eight EM entries, and the continuation direction | bit-identical | bit-identical |
| State words after a glued exit, `(sheet, 0, 0)` and type 71; after the reflect pass `(sheet, host, followed)` with the series declined and type 207 | exact | exact |
| Host series between two sheets, both as layers backed by air, both ports | 1e-9 | 3.8e-16 |
| A resolved ray at a glued exit, an embedded sheet and an ordered stack, lossy and lossless host: the resolving run against the declined one | bit-identical | bit-identical |
| Solid between two mass-law sheets of its own material against the solid carrying all three masses, complex, two events | 1e-9 | 1.0e-13 |
| Wood panel \| concrete \| rigid board, mass paid against the plate of three | 1e-9 dB | 3.6e-14 dB |
| A bounce returning to the glued sheet: mass paid forward and back against the plate; the return event against the bare-entered exit | 1e-9 dB, 1e-12 | 1.4e-14 dB, identical |
| Traced chain entered through a glued sheet, `air \| S \| host \| air` and `air \| S \| host \| S \| air`, lossless and lossy host, against the characteristic matrix of the whole stack, 10° to 85° | 1e-6 dB | 0.000000 dB |
| A leg behind a sheet against the same leg with a clean previous-medium word, bare exit, glued exit and embedded sheet, lossy host, to 85° | 1e-12 | identical |
| The recovered cosine independent of the lossless entry medium, named `eps 5.24` against a sheet in the word, invariant 0.2 to 1.45 | 1e-12 | 4.1e-16 |
| Host series on the leg behind a sheet inside the host, the sheet a layer backed by the host, 2 sheets × 2 lengths × 4 angles, both ports | 1e-9 | 3.3e-16 |
| The same rows: the host-backed and the air-backed series far enough apart to be told | > 1e-3 | 7.8e-3 to 1.1e-1 |
| The coating control on the same sweep: its sheet stays backed by air | 1e-9 | 4.7e-16 |
| Bonded mass, a sheet inside a wall, three bodies against the plate through the thickness code, 4 stacks × 4 angles | 0.01 dB | 1.6e-3 dB |
| The plate at depth, one slope, through the intersector: three and four solids, a lining on two solids, a lined wall holding a sheet, a solid in front of a wall holding a sheet, two sheets inside a wall, unequal `fRef`, a merged host at `m` 20 and 30; both directions, both tracers, far above and near the knee, 0° to 75° | 0.01 dB | 3.0e-3 dB |
| The same for mixed slopes from the side on which the folded bodies agree, and for a deferred medium and a clamped code from the side that needs neither | 0.01 dB | within bound |
| A bounce on those stacks: the forward traversal and the way back, each against the plate of the bodies it crossed | 0.01 dB | 1.5e-3 dB |
| Every state word of those chains at 0.25 GHz and 4 GHz equal to the one 1 GHz writes, 8 stacks × 2 directions × 2 angles | exact | exact |
| Through `ray_progress`, float: six stacks, both directions, both tracers, alone and in four bands, the strongest path against the plate; a frequency of a band against the same frequency alone | 0.01 dB; 2e-3 dB | 1.5e-3 dB; within bound |
| The same: the crossing pays the first half and the sheet | 1e-9 dB | within bound |
| A sheet without the mass column inside a wall separates the halves | 1e-9 dB | exact |
| The material index: 16383 accepted with and without the resolved flag, 16384 materials rejected, bit 14 of the previous-medium word not part of the index | exact | exact |
| `ray_progress` and `ray_commit` accept 16383 materials and reject 16384 | exact | exact |
| Host behind a solid across a bare face, `air \| first \| host \| air`, single pass at the exact cosines, 9 pairs × 6 angles to 85°, lossless and lossy first solid | 1e-6 dB | 0.000000 dB |
| The same rows: the last event with the words the transition wrote against a clean previous-medium word; no ray killed | 1e-9 dB | 0.000000 dB |
| The same rows: the traced cosine in the host against one refraction from air | 1e-9 | equal to nine digits |
| The same rows: bit 14 of the word the transition wrote set exactly where the first solid is lossy | exact | exact |
| The same chain in EM, TE and TM separately, 4 pairs × 4 angles | 1e-6 dB | 0.000000 dB |
| Three solids, `air \| A \| B \| C \| air`, 4 triples × 4 angles: the chain, the second transition alone, and the traced cosine in `C` | 1e-6 dB, 1e-6 dB, 1e-9 | 0.000000 dB |
| A sheet inside a host that was entered from a lossy solid: the sheet event on the composite at the exact cosine, both ports, against its clean-word twin, and the transmitted chain end to end | 1e-6 dB, 1e-9 dB, 1e-6 dB | 0.000000 dB |
| A face evaluated at a real angle in a lossy solid, wedge and launched wave: the word carries no flag and the next event sits on the cosine the face was evaluated at, 16 rows | exact, 1e-6 dB | 0.000000 dB |
| Wall of two solid layers with a far coating: the glued exit with the first layer in the previous-medium word, with and without bit 14, both ports against the composite with air beyond, 3 first layers × 3 angles, scalar | 1e-9 | 7.1e-16 |
| The same against the same exit with a clean previous-medium word, scalar and all eight EM entries, and the continuation direction | bit-identical | bit-identical |
| The same: state words `(sheet, 0, 0)` and type 71 after the exit; `(sheet, host, followed)` with the series declined and type 207 after the reflect pass | exact | exact |
| The same: the host's series with the coating a layer backed by air and the first layer behind as a half-space, both ports | 1e-9 | 3.5e-16 |
| The same rows: the air-backed and the first-layer-backed series far enough apart to be told | > 1e-3 | 5.6e-2 to 1.5e-1 |
| Chain `air \| A \| B \| coating \| air` with `path_dir`, series declined, both ports against the single pass through `A` and `B` times the composite at the exact cosine, 4 pairs × 5 angles to 85° | 1e-6 dB | 0.000000 dB |
| The same rows: the glued exit fed the words the transition wrote against a clean previous-medium word, gain and direction | 1e-9 dB, 1e-9 | 0.000000 dB, identical |
| The same chain with the second solid's series resolved, against the characteristic matrix of {`B`, coating} between the half-spaces `A` and air, | 1e-6 dB | 0.000000 dB |
| The next wall after a glued exit behind a solid, reflect, enter and leave, against the same events with clean state words | bit-identical | bit-identical |
| The hit code `ray_mesh_interact` writes for an exit met at the external angle: 33 and 101 exactly where `sqrt\|ε\|·sin θ_ext ≥ 1`, 1 and 69 elsewhere, 4 materials × 6 angles | exact | exact |
| Bare exit with the code a traced run carries, chain against the analytic slab, 4 materials × 6 angles to 85° | 1e-6 dB | 0.000000 dB |
| The same event with the bit and without, transmit pass, reflect pass and EM, bare and glued | bit-identical | bit-identical |
| Glued exit with the code a traced run carries, both ports against the composite at the exact cosine, lossy and lossless host, to 85° | 1e-6 dB | 0.000000 dB |
| Total reflection at a recovered cosine, invariant 1.10 to 1.45 in steps of 0.025: phase step between neighbours | < 45° | 3.1° to 5.8° |
| The same rows: the dense solid named against a sheet in the previous-medium word | 1e-9 relative | identical |
| The traced run through the intersector, undeviated tracer, `air \| slab \| air` against the analytic slab, lossy and lossless, to 85° | 1e-3 dB | 2.8e-5 dB |
| Direction after a glued exit against the incoming one, bare-entered, behind a sheet inside the host and behind an ordered stack inside the host, lossy and lossless host, to 85°, both components | 1e-9 | 1.1e-15 |
| The same rows: the sheet and the stack event hand the direction on | 1e-12 | exactly 0 |
| Direction after a bare exit through the intersector, undeviated tracer, straight through and for a resolved ray leaving by the front face, lossy and lossless, to 85° | 1e-9 | 7.6e-16 |
| Exits that keep their direction: no direction supplied and faces known not parallel, against the real-angle map, lossy and lossless host | 1e-9 | 1.1e-16 |
| The same: a closed outer pair hands on the incoming direction | 1e-12 | 1.1e-16 |
| A leak written undeviated: into a rigid body, wood to concrete as two traced solids, a glued entry under a closed outer pair, an isolated mass-law panel | 1e-9 deg | within bound |
| Refracting tracer through the intersector, `air \| slab \| air`, 4 materials × 6 angles to 85°: power against the analytic slab of the thickness crossed, scalar and both EM channels | 1e-3 dB | 0.000000 dB |
| The same rows: phase at the polyline's exit point | 0.05° | 0.009° |
| The same rows: refracting over undeviated tracer, each over the slab it crossed and at the point its own polyline leaves at, power and phase | 1e-3 dB, 0.05° | 0.000000 dB, 0.009° |
| The bent flag: set after a refracting entry with types 2 and 5, never with type 4, zero after the exit; carried through a sheet inside the host and a back reflection; written again by a transition | exact | exact |
| The state words of the two tracers differ in that one bit and nowhere else, sheets on and in the host and two solids, 13 stacks × 6 angles | exact | exact |
| The type code at the exit of a bent leg carries no VBS bit | exact | exact |
| Refracting over undeviated tracer, each at its polyline's exit point: glued entry, glued exit, lined, a sheet inside the host and two solids across a bare face, lossy and lossless, power and phase | 2e-3 dB, 0.05° | 1.1e-3 dB, 0.030° |
| The same for two solids across a bare face, single pass, 5 pairs, power | 2e-3 dB | 5.4e-4 dB |
| Back reflection with the refracting tracer, single bounce and with the series collapsed, against `t01 r10 t10 φ²` and its series at the polyline's exit point, lossy and lossless, power and phase | 1e-3 dB, 0.05° | 4e-6 dB, 0.006° |
| Direction written at the exit of a bent leg against the incoming one, every row above, lossy and lossless | 1e-9 | within bound |
| A leak with the refracting tracer, three rigid stacks: every event against the undeviated tracer's, coefficient, type code and all three words | bit-identical | bit-identical |
| A second frequency of a run, geometry and words at 1 GHz, coefficient and direction at 0.5 and 2 GHz, refracting tracer: power and phase against the slab at the polyline's exit point, and the exit direction | 1e-3 dB, 0.05°, 1e-9 | 4.6e-5 dB, 0.005°, within bound |
| Trajectory through `ray_progress`, float: the segment behind one wall against the launched direction, 4 materials × 6 angles, both tracers | 1e-6 | 3.0e-7 |
| The same: a second wall 1.9 m behind, the lateral run between the walls against the undeviated tracer's, 3 first walls × 6 angles | 2e-4 m | 7.6e-5 m |
| The same: the segment inside the second wall against the direction the undeviated tracer holds there, and the segment behind it | 1e-6 | 6.0e-8, 3.9e-7 |
| Trajectory, two solids across a bare face, 6 pairs × 6 angles: the segment in the second solid against `path_dir` and against the same solid entered from air; the segment behind the wall | 1e-6 | 1.2e-7, 3.9e-7 |
| Trajectory, glued entry, glued exit, lined and a sheet inside a lossy host, to 85° | 1e-6 | 6.7e-8 |
| The same stacks on a lossless host, which nothing moves | 1e-5 | 1.6e-6 |
| Trajectory, a ray leaving by the front face after a bounce, resolved and not, against the mirror of the launched direction; an internal reflection launches along `path_dir` | 1e-6 | 6.7e-8 |
| The ray tube behind one wall: each vertex direction against the one it was launched in, 3 materials × 3 angles | 1e-6 | 4.6e-7 |
| The ray tube into a second solid: each vertex direction against the same vertex entering that solid from air, 4 pairs × 3 angles | 1e-6 | 6.0e-8 |
| A beam split in air behind a lossy wall: the four sub-beams against the undeviated tracer's, segment and `path_dir` | 1e-6 | 1.2e-7 |
| Chain through the intersector with `origN` and `destN` handed in, double: the segment against `path_dirN` on every event that leaves a lossy medium, bare, sheets and two solids, types 5 and 2 | 1e-9 | 3e-16 |
| The same rows: the direction behind the wall, the chain against the analytic slab and against the undeviated tracer | 1e-9, 1e-3 dB, 2e-3 dB | as without the geometry |
| The same rows: state words with and without the geometry; type codes but for the VBS bit, which is clear behind a lossy solid | exact | exact |
| The same, one wall: every coefficient with and without the geometry | bit-identical | bit-identical |
| Phase at the polyline's exit point, `air \| 100 mm \| air` through the intersector, series resolved, both tracers, 3 materials × 6 angles to 85°, at 1 GHz and at 0.5 and 2 GHz run as frequencies of a band | 1e-3 dB, 0.05° | 0.000000 dB, 0.015° |
| Back reflection through the intersector, both tracers, lossy and lossless, 1 GHz and 2 GHz of a band, series declined and collapsed, to 85°: the whole path and each leg at the polyline's point | 1e-3 dB, 0.05° | 0.000000 dB, 0.012° |
| Leg 1 of a back reflection whose series the undeviated tracer collapsed, against `r10 φ S` | 1e-3 dB, 0.05° | 0.000000 dB, 0.002° |
| A sheet inside a host through the intersector, both tracers, host series declined, against the single pass through the host and the sheet's Airy pair at the polyline's point | 1e-3 dB, 0.05° | 0.000000 dB, 0.006° |
| An acoustic absorber as a traced solid, mineral wool 50 mm, 250 Hz to 4 kHz with 1 kHz as base, both tracers, 20° to 75°: through, and on a rigid wall enter, reflect, leave | 1e-3 dB, 0.05° | 0.000000 dB, 0.016° |
| Two solids across a bare face, the two tracers each at its polyline's point, lossy and lossless pair, gaps of 1, 10 and 50 µm | 0.05° | 0.017° |
| The same rows: the phase left between the tracers against what follows from the geometry no ledger holds | 0.005° | 0.0002° |
| The traversal of a mass-law solid alone, 100 mm over 50 mm, refracted into (`eps 1.50`, `m 20`) and entered as a leak (wood), both tracers, at the polyline's point | 0.05° | 0.0000° |
| The same: a leak with the refracting tracer against the undeviated one, coefficient, type code and current word | bit-identical | bit-identical |
| A ray reflected off a far-side coating and leaving by the front face, through the intersector, both tracers, lossy and lossless host, to 85°, against `t01 R t10 φ²` at the polyline's point | 1e-3 dB, 0.05° | 0.000000 dB, 0.006° |
| A resolved ray returning across a bare face into a first solid, `air \| A \| B \| air`, every series resolved, 4 pairs × 5 angles to 75°: the undeviated over the refracting tracer, each at its polyline's point | 2e-3 dB, 0.05° | 9.4e-4 dB, 0.0024° |
| A wedge through the intersector, exit face tilted by 10°, lossy and lossless host, both tracers: a bare and a glued exit write one direction | 1e-9 | within bound |
| The same: the entry event sets the non-parallel flag itself | exact | exact |
| The verdict at a bare and at a glued exit of a lossy host entered from a dense lossless solid, invariant 0.50 to 1.45: closed exactly from an invariant of one | exact | exact |
| The same rows: below one both exits against the reference from the invariant; above one both pass nothing and the glued exit hands on the direction the wave has | 1e-6 dB, exactly 0, 1e-12 | within bound |
| The same glued exit with `tf` 0.5 on the coating: the leak crosses from an invariant of one on, with no hole between neighbours | > −10 dB, 0.5 dB | −5.86 dB to −6.25 dB, 0.26 dB |
| Reflect port of a coated lossy solid, invariant 0.50 to 1.45, past its critical angle from 1.02: scalar and TE against the composite at the exact cosine | 1e-9 | below 1e-15 |
| The same, TM | 1e-9 | 1.7e-16 |
| The same event to 1.45 against the reference at the invariant itself, the recovery run until converged | 1e-12 | 4.4e-16 |
| A porous sheet, `eps 0.31 − 0.20j`, 10 mm in air: both ports against the analytic slab, 9 angles to 85° across 37.40°, both tracers | 1e-9 | within bound |
| The same sheet on a rigid wall, with and without `tf` on either: transmit port against the chain of an ordinary crossing with its series, reflect port against the composite, `ray_mesh_interact`'s own composite against the redo, 54 rows | 1e-9, 1e-9, 1e-12 | 2.0e-15 on the transmit port |
| A dense lossy sheet glued on the porous medium as a half-space: both ports against the analytic layer, the direction against `qd_refract`'s map, hit code 71, the bent flag with the refracting tracer | 1e-9, 1e-6 deg, exact | within bound |
| The porous medium as a traced solid 50 mm in air, entry: both ports against the half-space, hit code 3, direction, segment, bent flag, and the exit direction equal to the incidence one | 1e-9, 1e-6 deg | within bound |
| The same solid, every path against the analytic slab, both tracers, series summed and followed, to 85° | 5e-3 dB, 0.05° | 7.4e-4 dB, 0.0011° |
| The same solid on a rigid wall with `tf` 0.3 against the sheet of its thickness: the leak into the wall, series summed | 5e-3 dB | 6e-4 dB |
| `ray_mesh_interact` with a ray tube on `air \| porous`, spine at −3°, −0.3°, +0.3°, +3° about 37.40°: hit code 3 and the gain of the spine alone | exact, 1e-12 | exact |
| EM, `air \| porous` and `eps 5.24 \| eps 1.50 − 0.40j` launched in the first: transmit and reflect power per channel against the half-space, to 85° and 80° | 1e-9 | within bound |
| EM, `eps 5.24 \| 6 mm eps 1.50 − 0.40j \| air` past the critical angle of its outer pair, `tf` 0 and 0.5 on the coating: transmit power per channel against the chain with its series | 1e-9 | within bound |
| Three lossy solids rarer than air, `0.31 − 0.20j`, `0.50 − 0.60j`, `0.31 − 0.60j`, 50 mm, to 85°: exit direction equal to the incidence one, every path against the analytic slab | 1e-4 deg, 5e-3 dB | 0 in six digits, 1.3e-3 dB |
| A lossless rigid wall with `tf` 0.3, with and without its mass law, 10° to 75°: hit code 35, leak 0.3, reflectance 0.7, direction held | 1e-9 | within bound |
| Through `ray_progress`, the porous solid in air at 0.5, 1 and 2 GHz and in four bands whose frequencies used to fall on different sides of the angle: every row against the analytic slab; every band row against the same frequency alone, also on and as a sheet on a leaking wall | 5e-3 dB, 2e-3 dB | 2e-4 dB against the slab |
| A leak through rigid walls through the intersector, 8 stacks (two bonded solids in both orders, a glued entry, a glued exit, and a bare wall, a lined wall, a panel and a wood wall behind a wall across 0.5 m of air) × 2 mass levels × 3 angles to 75°, both tracers: the direction every event writes, the direction it is handed and the segment it is met on are the incidence one | 1e-9 rad | within bound |
| The same: no leg carries the bent flag, and the refracting tracer's hit codes, resolved types and current-medium words are the undeviated tracer's | exact | exact |
| The same: every event on the same run with the fixture doing the holding | 1e-9 dB | within bound |
| The same, the four walls without an air gap: every gain of the refracting tracer is the undeviated tracer's | exact | exact |
| The same, both tracers ending at one place | 1e-4 m | within bound |
| Five rigid stacks at 30° in the direction case: every event undeviated | 1e-9° | exact |
| Concrete in front of wood in the refracting-tracer case: words, type codes and coefficients of the two tracers | bitwise | bitwise |
| A bare exit and a transition into a solid, closed at the corrected angle, invariant 1.01 to 1.45, both tracers, with and without `tf`: the direction written is the one the wave has, and the verdict is closed exactly from an invariant of one | 1e-12, exact | within bound |
| Two solids across a bare face through the intersector: each series, the resolving run over the declined one, against its own body between half-spaces, 4 pairs × 6 angles, both tracers | 1e-9 | 8e-15 |
| The same rows: the half-space mirror and the layer mirror far enough apart to be told | > 1e-3 | 1.7e-2 to 1.97 |
| The same rows: the first path with the layer mirror put in for the half-space, against the matrix of the whole wall | 1e-3 dB | 0.000000 dB |
| The factorized form against the matrix of the whole wall, two solids and a sheet inside a wall, reference against reference | 1e-12 | 7.0e-16 |
| The paths a tracer follows through two solids, every series resolved, three orders: a geometric sequence, and its limit on the matrix of the whole wall at the polyline's point | 1e-4, 1e-3 dB, 0.05° | 2.2e-5, 6.5e-4 dB, 0.024° |
| The same, refracting tracer, lossless second layer at 85°, inside the relaunch offsets of §5 | 1e-2 dB | 4.9e-3 dB |
| Through `ray_progress`, two solids, every series resolved, one ray until none is left: the summed transmitted and reflected paths against the matrix of the whole wall, 20 and 60 interactions; at least five paths behind the wall | 5e-3 dB | 1.6e-3 dB |
| The same, lossless second layer at 75° and 85° with 60 interactions | 0.05 dB | 0.038 dB |
| The current-medium word after a reflection off a sheet: resolved exactly where the host's series was summed, far coating and sheet inside the host, 2 hosts × 2 tracers × 6 angles, `eps` 1.5 and 0 | exact | exact |
| A wall coated on its far side through the intersector, every series declined: the transmitted paths a geometric sequence and their limit on the matrix of the wall | 1e-4, 1e-3 dB | 2.9e-6, 2.3e-4 dB |
| The same, reflected: the limit against the front-face term, in dB, and its phase | 2e-4, 1e-3 dB (5e-3 dB where the wall reflects 10 dB less than its front face), 0.05° | 8.6e-5, 1.3e-3 dB, 0.014° |
| The same wall with every series summed, one path, on the matrix | 1e-3 dB | 0.000000 dB |
| Through `ray_progress`, a wall coated on its far side at `thin_slab_threshold` 0.15 and 1.0: the summed paths against the matrix, transmitted and reflected | 5e-3 dB | within bound |
| The same with every series declined: every bounce a path | ≥ 9 paths | 9 to 30 |
| Through `ray_progress`, a sheet inside a wall, threshold 0.15 against every series resolved, to 60°, with a gain floor of −100 dB | 0.02 dB | 0.0082 dB |
| A leg that begins at a declined reflection forms no series: the event that closes it at `eps` 0 against the same at `eps` 1.5, six legs (far and near coating, both halves around a sheet inside a wall, both of two solids), 2 hosts × 2 tracers × 16 angles | bit-identical | bit-identical, 384 rows |
| The same rows: the buffer word is the followed mark after every declined reflection, at a bare face and at a sheet, and the ray is not resolved | exact | exact, 512 reflections |
| The same rows: a reflection at a bare face leaves the previous-medium word bit for bit, so the recovery reads what it read before | exact | exact, 384 reflections |
| Through `ray_progress`, the default threshold against every series summed, 20 interactions, no gain floor: bare wall, far coating, near coating, two solids (lossy and lossless second layer), 2 hosts × 2 tracers × 16 angles from 10° to 85° | 5e-3 dB | 3.4e-3 dB |
| The same for a sheet inside a wall, which follows 511 paths where it declines | 1e-2 dB | 8.2e-3 dB |
| A band of two through `ray_progress`, bare wall `eps 1.50 − 0.40j`, 1 GHz with 0.5, 0.25 and 4 GHz in either order, 10 angles from 60° to 85°, both tracers: each frequency's summed paths against the matrix at its frequency | 5e-3 dB | 1e-4 dB, 120 rows, 40 of them with the two frequencies on different sides of the threshold |
| A verdict handed in, the event that closes the first leg, both passes, bare wall, far coating and two solids, 2 hosts × 2 tracers × 5 angles: "sum" against the event at `eps` 0 and "do not sum" against the event at `eps` 1.5, coefficient and all three words; the verdict handed out | bit-identical, exact | bit-identical, exact, 120 rows |
| The glued entry of `ray_mesh_interact` against the redo of `ray_state_update`, both ports, lossy and lossless host, and under a closed outer pair (wool on a rigid wall, reflect port) | 1e-12 | 2.4e-16 |
| A wall coated on its near side from `ray_mesh_interact` alone, no state, lossless host, refracting tracer: the limits of the reflected and the transmitted paths against the matrix with the coating in front of the mesh plane | 5e-3 dB, 0.05° | 1.1e-3 dB, 0.009° |
| A wall coated on its near side and a wall lined on both faces through the intersector, every path, `eps` 0 and 0.15, 2 hosts × 2 tracers × 6 angles to 85°: the reflected sum against that matrix | 5e-3 dB, 0.05° | 2e-3 dB, 0.008° |
| The same rows at 85° on the lossy host: the distance from the wall over the path off the first coating | 5e-3 | 3.6e-6 |
| The same walls through `ray_progress`, 0.3, 1, 3 and 6 GHz, thresholds 0 and 0.15, both tracers, 10° to 70°: the reflected sum; the transmitted sum | 5e-3 dB (1e-2 dB at 6 GHz), 0.05°; 5e-3 dB | within bound, 8.9e-3 dB at 6 GHz; 1.1e-3 dB |
| The same: where the wall reflects 10 dB less than the path off its first coating, the distance from the wall over that path | 5e-3 | 4.3e-4 on the one such row, lined, lossless host, 3 GHz, 30° |
| Mineral wool 50 mm on a rigid wall, 250 Hz to 4 kHz, 20° to 75°: the sheet's reflect port against the layer with its front face 50 mm in front of the mesh plane, its mirror at the Fresnel value, above one where that is above one, complex | 1e-9 | within bound |
| The same: the sheet over the summed paths of the wool as a traced solid in front of the wall, | 5e-3 dB, 0.05° | 4.0e-3 dB, 0.014° |
| A sheet in front of a wall across 30 mm of air, every path, through the intersector at 1 GHz: both sums against the wall whose layer is centred on the sheet's face | 5e-3 dB, 0.05° | within bound |
| The same rows: the walls whose layer lies behind the face and in front of it are told apart | > 1e-2 | holds |
| The same through `ray_progress`, 0.3 to 6 GHz, 110 interactions and a floor of −200 dB | 5e-3 dB, 0.05° | 4e-4 dB, 0.011° |
| A sheet inside a wall, host 50 \| sheet 6 mm \| host 50 as meshed, every path, `eps` 1.5, 0 and 0.15, 2 hosts × 2 tracers × 3 angles to 60°: both sums against host 47 \| sheet \| host 47 | 1e-2 dB, 0.05° | 3.9e-3 dB, 0.012° |
| A wall with the sheet off its middle against its mirror image: the transmitted sums from the two sides | 1e-3 dB, 0.01° | 4e-4 dB, 0.002° |
| A sheet against a traced solid of one material and thickness, 20 mm, lossy and lossless, `att` 0, 3 and 10 dB, 0° to 85°: the sheet's reflect port against the Airy reflection whose echo carries `sqrt(G)`, at the mid-plane, complex | 1e-9 | within bound |
| The same: the summed reflection of the solid against that form; the sheet over the solid; the transmit ports | 1e-3 dB; 1e-3 dB and 0.05°; 5e-3 dB | 7e-5 dB; within bound; 3.2e-3 dB at 85°, the relaunch offset |
| A layer 50 mm on a leaking wall as a traced solid against the same layer as a sheet, porous `0.31 − 0.20j`, lossless `1.50` and lossy `1.50 − 0.40j`, wall rigid with `tf` and `m`, rigid with `tf` only, and `eps 0.20` with `tf`, both tracers, 10° to 85°: the direction in the wall and behind it | 1e-6° | 1.3e-13° |
| The same rows: the wall \| air event; on the rigid walls the segment in and behind the wall | 1e-4 dB, 1e-6° | 4.6e-6 dB, 1.3e-13° |
| The same, wall with `m`: the direction behind the wall on the incidence angle; the mass term against the sheet's and the bare wall's | 1e-6°, 5e-3 dB | 1.3e-13°, 4.6e-6 dB |
| A host entered from a denser solid in front of the leaking wall, invariant 0.50 to 1.45: the direction of the invariant in air below one, with tracer 5 the segment and, behind the wall with `m`, the direction; above one the direction the wave has | 1e-6°, 1e-9° | within bound |
| The ray tube at a closed face, type 5, three layers × three angles: the spine and each vertex ray continue in the direction they were launched in | 1e-6° | within bound |
| Porous 50 mm on the rigid wall with `tf` 0.3 and `m` 30 against the sheet, 10° to 85°, both tracers: the wall's exit with its mass term, the direction in the wall and behind it | 5e-3 dB, 1e-6° | 4.6e-6 dB, within bound |
| Through `ray_progress`, the porous layer on the wall with and without `m`, as a solid, as a sheet and the wall bare, 0.5, 1 and 2 GHz and four bands, 30° to 80°, both tracers: the power sum of each frequency of a band against the same frequency alone | 2e-3 dB | 1e-4 dB |
| The same, wall with `m`: the direction and the segment of the strongest path behind the wall on the incidence angle, float | 1e-3° | within bound |
| A corner with a sheet as the first face, the second face a sheet or a solid, front- and back-hit, both passes: the redo against the sheet alone, coefficient, resolved type and words; `ray_mesh_interact`'s own coefficient against the sheet alone | bitwise; 1e-12 | bitwise, 24 rows; exactly 0 |
| The same with the sheet as the second face: both functions against the first face alone | bitwise | bitwise |
| The same: co-location, same-direction and corner bits of the hit code cleared, the ray relaunched behind the face that was dropped | exact | exact |
| Displacement of a sheet inside 10 mm of host, thk / L 0.1 to 3, lossy and lossless sheet, both tracers, 20° to 85°: the transmit port against `diag_cm_stack` capped per side, the reflect port against it over twice the part in front | 1e-9 | within bound |
| The same rows: the host's series on the leg behind the sheet against the layer mirror capped by that leg | 1e-9 | within bound |
| The same rows, a sheet that does not fit its host included: finite, the wall passes less than it received with the series declined and summed, no mirror above one | < 0 dB, < 1 | within bound |
| An ordered stack inside a host, 13 stacks × 2 tracers × 2 angles, 10 mm and 100 mm of host, the stack centred, near the front and near the back: both ports against `diag_cm_stack` capped by the leg in front on both sides; a stack that fits is not capped; one that does not fit in front is finite and passes less than it received | 1e-9; exact; < 0 dB | within bound |
| One wall: a sheet or an ordered stack of lossless media inside a lossless host, 9 geometries of which 4 do not fit, both tracers, 20° and 60°, every path: both ports together | 0.999 to 1.0001 | 1.00003 at most |
| The same walls where the layers fit: both ports against the characteristic matrix of the wall; the paths behind the wall | 5e-3 dB, 1e-2 dB; < 40 | 1e-4 dB, 1.7e-3 dB; 6 at most |
| A lossless sheet off the middle of a lossy host that it does not fit, 90 mm to 140 mm in 100 mm, every path: both ports together | < 1 | 0.855 at most |
| An opaque layer, swept from 100 dB to 2000 dB one way in a two-layer stack, both orders: reflect port against the reference | 1e-12 | within bound |
| The same: transmit port against the reference at every loss, with no floor | 1e-9 relative to 400 dB, 1e-6 dB above | within bound |
| The same layer as a lining on the far face of a traced solid: the host's series against the layer mirror and against the half-space of the layer's material | 1e-9 | 4.1e-14 |
| A sheet without the mass column, of its mass-law host's permittivity, inside 100 mm of a rigid wall, both tracers, 0° to 60°: mass paid against one body of the thickness crossed; the wall against the wall without the sheet; the state words carried through | 1e-3 dB, 1e-3 dB, exact | 1.4e-4 dB, 1.4e-4 dB, exact |
| The same with a sheet of another permittivity: the halves separate and pay in dB | > 10 dB | 35.9 dB to 41.9 dB |
| Float against double, the same chain event by event, six stacks (sheet in air, glued entry, glued exit, sheet inside, ordered stack, bare slab) × types 4, 5, 1, 2 × two paths × 0° to 80°, every series summed: every xprmat entry; hit code, resolved type and words | 1e-4 relative, exact | 3.1e-6, exact |
| `ray_mesh_interact`'s own glued entry against the redo, from air: a mass-law panel on concrete with `att`, a lossless sheet rarer than air on a rigid wall, a sheet and a solid with `att`, `tf` and a resonance; types 4, 5, 3, 1, 0, to 70° | 1e-12 | 4.4e-16 |
| The same rows, scalar: against `diag_leak_series` under a closed pair entered by an ordinary crossing, and against `diag_sheet_composite` with its lumped columns | 1e-9 | 4.8e-16, 1.4e-16 |
| `ray_mesh_interact`'s own glued exit against the redo, EM, a ray launched in a lossless solid, below and past the critical angle of the outer pair: every magnitude, and one host-leg factor for VV and HH | 1e-12 | 6.7e-16 |
| A ray tube with one edge past the critical angle of the outer pair, refracting tracer: the redo against the undeviated tracer's event without a tube, coefficient, resolved type and direction; `ray_mesh_interact`'s own coefficient against the redo | bitwise, 1e-9°; 1e-12 | bitwise; within bound |
| A stack of a coherent and a mass-law layer against the same laminate meshed as a traced solid with the mass-law layer as a lining, both sides, both tracers, 10° to 60°, two mass levels: the reflect port at the stack's plane, and the phase of the transmit port | 5e-3 dB, 0.05° | 7e-4 dB, 0.011° |
| The same: the transmit port, from the side of the solid and from the side of the lining | 5e-3 dB | 1.5e-4 dB |
| The cosine of a mass term: a mass-law sheet 22 mm on the far face of a host, inside it, and in a stack inside it in both orders, hosts `2.50 − 0.80j`, `2.50` and the porous `0.31 − 0.20j`, two mass levels, both tracers, 10° to 60°, through the intersector: the mass paid against the term at the incidence cosine, and against the same sheets standing in air | 1e-6 dB | 0.000000 dB, 144 rows |
| The same rows: the direction written behind the wall on the incidence one; for a single sheet the last hit point on that of the same host without the sheet | 1e-4°, 1e-9 m | 1e-14°, 4e-16 m |
| The mass paid by the laminate from the side of the solid and from the side of the lining, and by the stack in both orders, lossy and lossless solid, two mass levels, both tracers | 1e-6 dB | 0.000000 dB |
| A mass-law lining on the tilted exit face of a wedge, the entry carrying the non-parallel flag, lossy and lossless host, both tracers: the mass paid against the term at `Re sqrt(1 − ε (1 − c²))` | 1e-6 dB | within bound |
| A host entered from the lossless `eps 5.24` solid, invariant 0.50 to 1.45, lining and sheet inside: the mass paid at `sqrt(1 − p)` below one and at the cosine in the host from one on | 1e-6 dB | within bound |
| A ray inside a lossless host with no state, glued exit: the mass paid at the cosine of the segment; `ray_mesh_interact`'s own glued exit likewise | 1e-9 dB | exact |
| The followed mark across the shared face of two abutting blocks of one material, one and two indices, lossy and lossless, both tracers: the mark after the crossing, the last event at `eps` 0 against 1.5, the path against the wall meshed as one block | exact, bitwise, 2e-3 dB and 0.05° | exact, bitwise, 9.4e-4 dB and 0.011° |
| The followed mark at a nested object: the object's medium in the buffer word, a zero word after it is left, and the twin without the object forming no series | exact, bitwise | exact, bitwise |
| The same: the leg then forms a series of its own, the recorded limitation | > 1e-3 | +0.08 dB and −0.09 dB lossy, +0.02 dB and −0.54 dB lossless, at `eps` 0 |
| A resolved ray leaving a wall into a body that overlaps it: resolved flag kept at the body's face, the body's medium entered, the event at `eps` 0 against 1.5; the level against the same crossing with the body ending at the wall's face; the phase at normal incidence | exact, bitwise, 2e-3 dB, 0.05° | exact, bitwise, 2.9e-4 dB, 0.003° |
| A verdict handed in on a leg flagged not parallel, both tracers, both passes: verdict 1 against no verdict at `eps` 0, verdict 0 against no verdict at `eps` 0.15, coefficient, words and type; the verdict handed out | bitwise, exact | bitwise, exact |
| A series near a pole, 20 mm of lossless `eps 1e4` at 80°, both passes: with a verdict handed in and without, against the exact Airy series; the two runs against each other | 1e-6 relative, bitwise | within bound, bitwise |
| A stack of 10 mm `eps 2.50 − jε″` and a rigid mass-law layer with `tf` 0.2, both orders, and the laminate with the coherent layer a traced solid, both sides, without the mass term, ε″ 0.8, 0.1, 1e-3, 1e-6 and 0, 10° to 60°, types 4, 5, 1 and 2, both EM channels: no more than received | < 0 dB | −11.92 dB at most, 360 levels |
| The same: a loss of 1e-6 against none | 1e-3 dB | equal to four decimals |
| The same: one side over the other against `sec²(arg η)` of the coherent layer per channel, the stack; the traced laminate | 1e-6 dB; 1e-5 dB | 7.6e-15 dB; 1.6e-6 dB |
| The same: the stack against the traced laminate, each side; the mass paid by a 26.8 dB layer against the term at the incidence cosine | 5e-3 dB; 1e-6 dB | 3.7e-4 dB; 7.1e-15 dB |
| The same with `tf` 0.3 on the coherent layer: below one, and the two sides on the splitting; with no `tf` on the rigid layer: nothing passes | < 0 dB, 1e-6 dB; < −120 dB | within bound |
| The sheet event of a rigid lining on the near face of five hosts, lossy, lossless and porous, both tracers: the two leaks, from the redo and from `ray_mesh_interact` alone | 1e-9 dB | within bound |
| The same lining on the far face against the same sheet inside the host; below its two leaks; `ray_mesh_interact`'s own glued exit below one; without `tf` nothing passes, on either face and inside | 1e-6 dB; < 0 dB; < −120 dB | within bound |
| A rigid and a non-rigid mass-law layer as a stack in air, `eps 1.5` and `2.5`, with and without `tf`, both orders | 1e-9 dB | 0.000000 dB |
| Twins of the closed layer: a rigid sheet in air passes `tf²`; a stack whose mass-law layer has a permittivity above one in both orders | 1e-12; 1e-14 | within bound |
| A rigid lining on a rigid traced wall towards the lower index, 3 pairs × 13 angles, both tracers: the level without mass | < 0 dB | −15.33 dB at most |
| The same: the lining against the same body as a traced solid; types 1 and 2 per channel | 1e-3 dB | exact |
| The same: the wall from its two sides, to the flat first faces | 0.02 dB | 0.0025 dB |
| The same: across the critical angle of the pair, 0.2° apart | 0.01 dB | 0.002 dB |
| A rigid sheet inside a rigid wall against three solids, 3 pairs × 8 angles, both tracers | 1e-3 dB | exact |
| The reflect port a held wave sees at a lining, a sheet inside and a stack, against the face of the solid twin, complex, scalar and both EM channels | 1e-4 | 3.3e-16 |
| Two rigid sheets inside a rigid wall, six orders × 7 angles, both tracers: against four solids, and against the other order | 1e-3 dB | exact |
| The mass paid by a lined rigid wall from its two sides | 0.01 dB | 1.4e-14 dB |
| An ordered stack in air: segment, tube and own `path_dirN` on the incoming direction, 7 pairs × 4 orientations of the normals × 4 types × 3 angles | 1e-9 | 2.2e-16 |
| An ordered stack inside a lossless and a lossy host, bare-entered: segment on the direction written, the next wall met where a single sheet lets it be met | 1e-9 | 3.3e-16, 1.3e-15 m |
| The same behind a glued entry: on the line of the segment the stack was reached on | 1e-9 | 1.1e-16 |
| A chain through a stack in a host and a wall behind it, refracting against undeviated tracer | 1e-3 dB | 2.6e-4 dB |
| An ordered stack through `ray_progress`, float, a band of two frequencies: segment, direction, the wall behind | 1e-6 | 1.2e-7 |
| A beam split in air behind a stack against the undeviated tracer's, segments and vertex rays | 1e-6 | identical |
| The own EM coefficient of a stack on a ray out of the horizontal plane, type 2 against type 1 | 1e-12 | exact |
| A rare sheet with `tf`, 6 mm `eps 0.31` inside `eps 2.50 − jε″`, ε″ 0.8, 0.1, 1e-3, 1e-6 and 0, `tf` 0.5, 0.1 and −0.5 on the sheet and 0.5 on the host, 45° to 75°, types 4, 5, 1 and 2, both EM channels: each port; the two together in the lossless host | < 0 dB; ≤ 1 + 1e-6 | −0.02 dB at most; 1.000000 (1.024 at most in a lossy host, recorded) |
| The same: a loss of 1e-6 against none, both ports | 1e-3 dB | equal to four decimals |
| The same: `tf` ±1e-9 against no `tf`, TE, lossy host; the tunnelled wave kept for `tf` up to ±0.1 | 1e-6 dB; > −3 dB | within bound |
| The same across the critical angle of the sheet, 33.80° against 33.87°, and the two ports together | 0.5 dB, ≤ 1 | 0.0015 dB at ε″ 0.8 and 0.0001 dB at 1e-3 and 0, within bound |
| The same sheet with a loss of its own, 0.2 to 0, inside a lossy and a lossless host and in air: each port; 1e-6 against none | < 0 dB, 1e-3 dB | −0.01 dB at most, equal to four decimals |
| The same sheet as a lining on the near and on the far face of a host: both ports of both; the lining tunnels; `ray_mesh_interact`'s own entry on the redo; a lossless lined wall from its two sides | < 0 dB, > −30 dB, 1e-9 dB, 1e-6 dB | −0.23 dB to −10.9 dB, equal, equal to four decimals |
| The same sheet in an ordered stack with a coherent layer, both orders, in air and inside the host: the level, and one order against the other | < 0 dB, 1e-9 dB | below 0 dB, 6.4e-15 dB |
| A rare sheet with `tf` on a rigid wall, sheet loss 0.2 to 0: the reflect port; 1e-6 against none | ≤ 0 dB, 1e-3 dB | 0.00 dB at zero loss, −1.73 dB at most with loss |
| Through `ray_progress`, float, both tracers, scalar and EM, a wall with the rare sheet in it, `tf` on the sheet or on the host: the path behind the wall is there, carries no more than was launched - per channel with `tf` on the sheet, on the TE/TM mean with `tf` on the traced host, an assertion that predates the per-channel bare face -, and a loss of 1e-6 sits on none | < 0 dB, 1e-2 dB | −0.07 dB at most in HH, mean 0.94 with `tf` on the host, 1e-4 dB |
| Reflect port of a porous sheet with `tf` on a rigid wall, 10° to 85°, against the composite with a transformer on its near face and the pair rule on its closed far face | 1e-9 | 1.2e-15 |
| A coherent sheet with `tf`, 1, 6 and 25 mm, eight permittivities, `tf` 0.5, 0.9, −0.5 on the sheet and 0.5 on the host, 0° to 85°, scalar, TE, TM: both ports together in air, inside `eps 2.50`, as a lining on its near and far face and on the near face of `eps 2.50 − 0.80j` | ≤ 1 + 1e-6 | 1.000000 at most |
| The same: a lossless sheet between lossless media, both ports together | 1 to 1e-6 | within bound |
| The same: both powers against the reference of the rule written in tilted admittances | 1e-9 | 5.7e-15 |
| The same inside `eps 2.50 − 0.80j`: each port (the sum is recorded, §5) | < 1 + 1e-6 | within bound; sum 1.233 at most |
| A lossless sheet with `tf` below its critical angle against the pair rule on the reflectance, complex, both ports | 1e-9 | 1.9e-15 |
| A sheet without `tf`, lossy or lossless, rare or dense, in air and on a lossy host, against `diag_cm_stack`, complex, both ports | 1e-9 | 1.9e-15 |
| Neighbours 0.02° apart about the angle at which a lossy sheet becomes rarer than the invariant, in air, inside a lossless and a lossy host, as a lining and on a rigid wall with `tf`: the transmit port; the reflect port in amplitude against the same pair with no `tf` | 0.01 dB; 1e-3 | 0.0024 dB; 2.6e-4 |
| `tf = −1` on a sheet, 8 permittivities, in air and inside a host: the transmit port. `tf = +1` on a lossless sheet denser than its surroundings: the reflect port | power < 1e-24 | exactly 0 |
| An ordered stack of two 6 mm sheets with `tf` on one, five pairs, both orders, in air: both ports together; the lossless pairs | ≤ 1 + 1e-6; 1 to 1e-6 | 1.000000 |
| Through `ray_progress`, float, both tracers, scalar and EM, every path summed: a coherent sheet with `tf` alone in air, on 10 mm of `eps 2.50` and inside 10 mm + 12 mm of it, both ports together per channel; a lossless wall | ≤ 1.002; ≥ 0.995 | 1.000019 at most; 0.999984 at least |
| The same: the transmitted sum 0.02° either side of the angle at which the sheet becomes rarer than the invariant | 0.01 dB | 0.0042 dB |

| A lossless rare sheet without `tf` inside a host, both ports against `diag_cm_stack`, TE, TM and scalar, both tracers, host loss 0.8 to 0 | 1e-9 | 1.1e-15 transmit, 4.1e-14 reflect |
| The same alone in air across its critical angle, and as a lining on the near and the far face; `ray_mesh_interact`'s own composite against the redo | 1e-9; 1e-12 | 8.8e-15; 4.8e-15 |
| `tf` of 1e-9 and −1e-9 against none on those rows, both ports | 1e-6 dB | 1.2e-7 dB |
| One rare sheet against the ordered stack of its two halves, transmit | 1e-9 dB lossless, 2e-3 dB lossy | 4.1e-15 dB |
| A rare sheet without `tf` on a rigid wall without `tf`: the transmit port; the reflect port against the matrix | exactly 0; 1e-9 | 0; 9.9e-16 |
| A rigid sheet without `tf`: with the mass column closed; without it on the matrix | 1e-12; 1e-9 | 2.2e-16; 2.7e-15 |
| Through `ray_progress`, float, both tracers, scalar and EM: the rare sheet without `tf` alone in air against the lossless slab; inside a wall against the same wall with `tf` ±1e-9 | 1e-3 dB | on it in every printed digit |
| One traced solid with `tf` in air, both ports as sums over every path, series summed and every bounce followed, scalar and per channel in EM: together at most one; a lossless solid loses nothing | 1.0005; 0.999 | 1.00026; 0.99992 |
| One face air \| solid with `tf`, per channel and scalar: transmitted and reflected power together | 1e-9 | exact in nine printed digits |
| A 6 mm layer with `tf` in air as a traced solid against the same layer as a sheet, both ports, scalar and EM | 2e-3 dB | 1.2e-3 dB transmit, 1.7e-3 dB reflect |
| A sheet of air on a wall with `tf` against the bare wall, lossless and lossy, both ports | 1e-9 | 5.6e-16 |
| A rigid wall with `tf`, with and without the mass column, `tf` 0.1, 0.3, 0.9: the first face reflects `1 − tf` and passes `tf` | 1e-12 | 2.2e-16 |
| A rigid wall with a loss column: entry and exit together are finite and at most 0 dB | 0 dB | −6.5 dB at most |
| A lossless wall with a forced reflection at an exactly matched face: coated, and as two solids | 2e-3 | 1.6e-4; 1.5e-3 |
| Through `ray_progress`, float, both tracers, scalar and EM: a traced solid with `tf` alone, behind a coating and as the second of two solids, every path summed; solids denser than air asserted on all three walls | 1.002; lossless 0.995 | 1.00003 alone, 1.00015 coated, 1.00023 second solid |
| A solid 6 mm with a 6 mm coating on its near face, its far face or both, solids rarer than air and lossy, every series summed, every path, 45° to 85°, scalar, TE, TM: both ports against the characteristic matrix of the wall; the two together | 5e-3 dB, 2e-2 dB; 1.0005 | within bound, 0.0072 dB; 1.0000 |
| A coating on the lossless eps 5.24 met from inside, past the critical angle, no `tf`, 0.2 to 6 mm, four coatings: the reflect port against the characteristic matrix, TE and TM | 1e-9 | within bound |
| The same layer as a traced solid, every path: the power it returns; the coating over the solid, TE and TM | 1.0005; 0.03 dB | within bound; 0.017 dB |
| A lossless sheet of eps 5.24 in air at a transmission resonance, `tf` 0, −0.9 and −0.99, 30° to 89°: both ports together; the transmit port | 1 to 1e-9; 0 dB to 1e-6 | within bound |
| An ordered stack in air through `ray_mesh_interact` alone, four orientations of the two normals, types 4, 3, 1, 0, to 60°: the own coefficient against the redo; no total-reflection bit in the hit code | 1e-9; exact | within bound; exact |
| An ordered stack inside a host, 20 mm \| stack \| 30 mm, two hosts, 10° to 60°: the host's series in front of the stack against the series whose mirror is both layers; the two mirrors told apart | 1e-6; > 1e-4 | within bound |
| The same: the buffer word behind the stack is the stack code with the first layer; the host's series behind the stack against the series whose mirror is both layers, and against the last layer alone; the paths behind the wall | exact; 1e-6, > 1e-4; 8 at most | exact; 3e-16, 0.019 to 0.058; 6 |
| The same, every path: both ports against the characteristic matrix of the four-layer wall | 5e-3 dB, 1e-2 dB | within bound |
| Through `ray_progress`, host 50 mm \| 6 mm + 6 mm stack \| host 50 mm, lossy and lossless host, both tracers, 20°, 60° and 85°, every series summed, 20 interactions, a gain floor of −140 dB: the paths behind the wall and the rays launched; both sums against the matrix of the wall | 6, 40; 2e-2 dB, 1e-2 dB | 5, 32; 3e-4 dB but 8.9e-3 dB for the lossless host at 85°, 1.4e-3 dB |
| The face eps 5.24 \| eps 5.24 with `tf` −0.5 from its two sides through `ray_mesh_interact` alone, 161 angles: the two reflections each other's negative; each `\|tf\|` in power | all; 1e-9 | 161 of 161; within bound |
| A mass-law lining that is not rigid on a rigid wall, met from the wall, against its solid twin, 10° to 75°, types 4 and 1 | 1e-3 dB | 0.0000 dB |
| `ray_mesh_interact`'s own mass-law lining with types 2 and 5: the mass paid against the term at the cosine in air | 1e-9 dB | within bound |

Two further properties were established offline and have no fixture. The real-angle path through
`Material::interact_with` is bit-identical to its pre-change form over 140544 comparisons across four
materials, six `tf`/`att` cases, six interaction types and 244 angles. The entry and exit obliquity
factors across a traced solid multiply to 1.000000000000 in both channels at every angle tested,
which is the telescoping property the chain fixture measures end to end.

### Diagnostic infrastructure

`QD_TEST_VERBOSE` in each test file switches the tables on. It is currently defined in
`test_ray_mesh_interact.cpp`, `test_ray_state_update.cpp` and `test_ray_progress.cpp`, the last
for the trajectory case, the path-sum cases, the reflect-port case, the lossy-verdict case and the stack case. Every diagnostic in the first two files carries at least one
numeric bound, so the assertions run either way and no case is a no-op with the tables off.

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
traversal forms, defined together; `diag_medium_vbs` takes the cosine of the segment as a trailing
argument and then carries the phase to the polyline's point, without it the phase is the
refracted ray's), `diag_traced_ct` and `diag_ct_exact` (the `qd_refract` map and the
value it projects), `diag_r_slabside`, `diag_medium_gain`, and `diag_quad_x` / `diag_ray_at` for the
mesh calls.

`diag_leak_series`, returning `DiagLeak`, is the reference for the transmit port of a coherent sheet
whose outer pair is closed: the near crossing of the composite, a transformer in raw amplitude,
the one-way factor, the internal series and what the pad of the closed far face passes of the
incident flux, per channel, with the series factor returned separately so a row can say what
the series alone is worth. It holds for a sheet denser and rarer than the invariant. It assumes
the displaced medium is air, which holds on every row that calls it.

`diag_closed_face` is the reference of the pad of §1 for one face and one channel: the load
`|y|·(nu + b)/(1 + nu·b)` in front of the reactance behind the face, the reflection rescaled in
place and the amplitude passed returned in flux. It is written in the form the library
evaluates; the closed-face case carries a second reference in tilted admittances that shares
nothing with either.

`diag_sheet_composite`, returning `DiagSheet`, is the produce-mode reference: the full three-medium
Airy pair, both ports, per channel, with the flux conversion on the transmit port only and the
displaced-host propagation divided out. Its reflect port carries the convention of §1: the round
trip through the part of the layer in front of the mesh plane, by the same three cases, and
`sqrt(g_slab)` on the echo. `diag_sheet_composite_cx`, `diag_cm_stack` and `diag_sc_stack` carry
the same reference plane; `qd_slab_ref_lumped` in `test_ray_mesh_interact.cpp` carries the echo
gain and is in powers, where the plane does not show. It takes the three permittivities, the normal thickness, a
real incidence cosine and the displacement side. `diag_sheet_composite_cx` is the same form written
at a complex incidence cosine, for a sheet reached from inside a traced medium; it reproduces the
real-cosine builder exactly, which is asserted on every row of the sheet-in-a-medium case and again
with permeability live.

Both carry `(ε, μ)`. The three permeabilities trail the argument list with defaults of 1, so every
dielectric call site reads as it did before; `diag_cm_stack` interleaves its pairs instead, because
its μ is per layer.

`diag_sheet_composite` additionally takes a trailing `DiagLump`, which carries the lumped columns of
§1: `g_slab` and `g_far`, the interface power gains; `tf`, `tf_near` and `tf_far`, the transmission
factor of the sheet and of its two neighbours; and `res_near`, `res_slab`, `res_far`, permittivity
resonances as increments, carried as part of the permittivity. `g_far` is deliberately not gated inside the builder: the row states what
the far medium is owed, so the library's `far_is_new` gate is measured against a statement of the
physics rather than against a copy of itself. `tf` is expressible there because the Airy form is
written in the per-boundary `r` and `t` that `tf` rescales; `diag_tf_pair` is the pair rule on a
reflectance and `diag_tf_port` applies it to one boundary and one channel, returning zero for the
transmission factor of a boundary with no forward port. At its defaults nothing in the struct
executes. `diag_sheet_composite_cx` does not take it: every lumped-column row in the suite is reached
at a real incidence cosine.

`diag_slab_p_pol`, `diag_slab_p_avg`, `diag_traced_ct` and `diag_ct_exact` carry a trailing `μ` the
same way, and so do `qd_slab_ref` and `qd_slab_ref_R` in `test_ray_mesh_interact.cpp`; the latter
also returns the two reflect channels separately through optional pointers. `qd_slab_ref_lumped`,
returning `QdLumpRef`, is the analytic `air | sheet | air` slab with `att`, `tf` and a resonance on
both ports, kept separate so the plain oracle stays the plain Airy pair; at its defaults it is
asserted to reproduce both.

`diag_tf_face` is the reference of the transformer of §1 for one boundary and one channel,
written from the raw reflection as a Fresnel face of the admittance ratio `nu·u` with the factor
`sqrt(nu)·(1 + r')/(1 + r)` on both crossings, which is not the form the library evaluates.
`diag_sheet_composite` takes it on both faces of its sheet, and `diag_closed_face` - the pad -
on a far face with a lossless far medium past the critical angle of the outer pair. `diag_sc_stack` takes it at every boundary neither of whose layers carries the mass
column and `diag_tf_port` at the others. `diag_leak_series` runs its series on the near face
`diag_tf_face` gives and the far face `diag_closed_face` gives. `qd_slab_ref_lumped` in `test_ray_mesh_interact.cpp` writes the same rule
for its symmetric sheet as a sheet of admittance scaled by `nu`. `diag_tf_port_rare`, an earlier
form of the rule for a tunnelling face, has no caller left and can go.

`DiagLayer` carries a per-layer resonance `res` and a per-layer `alpha` in dB/m after its thickness,
then `tf`, `m`, `fRef` and the interface power gain `g`, which only `diag_sc_stack` reads; `g` acts
on its reflect port alone, through the second recursion of §1, and the transmit gain is left to
the caller as in `diag_cm_stack`. `diag_cm_stack` takes the two half-space
resonances as trailing arguments. `tf` has no entry in `diag_cm_stack` and cannot have one: a
boundary that has been through `tf` no longer satisfies `1 + r = t`, so it is not a boundary a
characteristic matrix describes. Interface gains are an end quantity there and are left to the
caller.

`diag_sc_stack` is the scattering-form reference: an ordered stack with a `tf` per medium, half-spaces
included, and the mass-law column, written as the far-end recursion of §1 and sharing no code with
the library or with `diag_cm_stack`. It reproduces `diag_cm_stack` at `tf = 0` and
`diag_sheet_composite` with `tf` at one layer, both asserted. It takes the real incidence cosine for
the mass term as its last argument and displaces the coherent thickness only. Mass-law layers with
no coherent layer between them pay as one plate there, written its own way: a lone layer keeps
`diag_mass_TL`, a later one pays the plate's term less what was paid. `diag_mass_TL` is the mass
term of one body in dB from the column definition.

`diag_bonded_TL`, over a list of `DiagMass` (`m`, `fRef`, normal thickness), is the plate rule from
the column definitions: unclipped `X₀` per body, `m = 0` absent, the single-body shape on the sum.
At one body it reproduces `diag_mass_TL`, asserted at 1e-12. The bonded-mass case uses two
instruments. Where both bodies are one material with `m = 20` the pair is compared with one body of
the summed mass through the library, which does not use the rule. Everywhere else the mass paid is
isolated as a ratio against the same bodies with their mass argument taken to nothing (`fRef` 1e9),
which leaves every face, `tf` and interface gain in place. A two-event chain is run by feeding the
second event the three state words the first one wrote, buffer word included.

`LumpMtl`, `lump_palette`, `lump_eps` and `lump_res` build a local palette that carries `att`, `tf`,
the three resonance columns, `alpha`, `m` and `fRef`, so `diag_palette` and `diag_eps` stay plain and
every older residual stays where it is. `fRef` is there because the mass term is a hard zero below an
argument `(f / fRef)·thk` of one, which at the palette's default of 1 GHz every millimeter-scale
sheet is; a row that wants a mass term worth anything sets it. `lump_layer` turns a palette row into
a `DiagLayer`. `diag_res` is the Lorentz pole written from the column definition.

Three instruments came with the sheet-in-the-previous-medium-word work and are worth reusing. *The
bare-entered twin*: the same event with a clean previous-medium word, compared bitwise, which needs
no reference and isolates everything that reads that word. *The series as a ratio*: the run that
resolves (`eps` 0) over the run that declines (`eps` 1.5), which is the series factor outright and
is compared with the factorized form whose mirrors come from `diag_cm_stack`. *The chain through a
glued entry*: `ray_state_update`'s own glued-entry event, tilted so the segment and the reported
angle agree, supplies the entry coefficient and the direction the tracer then holds; the next event
is run along that direction with the state words the entry wrote, and the product is compared in
power with `diag_cm_stack` over the whole stack. Its lossless-host rows have nothing to recover and
are the control on the chain and on the reference. Comparing two runs that should be identical is
done on the fields, bitwise; a complex ratio of two identical numbers is not exactly one.

Three instruments came with the state-word work. *The single pass at the exact cosines*:
`air | first | host | air` across bare faces with every series declined, the entry from
`ray_mesh_interact`, the transition and the exit from `ray_state_update`, against
`|t01 t12 t20|²` times the two normal-thickness decays written from the media alone. It is the
reference of *Measurement - a host entered from a lossy solid across a bare face*, which runs the
last event twice, with the words the transition wrote and with a clean previous-medium word; the
case kept its name and asserts every row. *The
backing as two references*: the series ratio against the factorized form with the sheet behind
the ray backed by the host and by air, with an assertion that the two differ, so a row says which
backing the code used. *The mass paid across a chain*: each event against its twin with the mass
argument taken to nothing, summed over the events.

Three more came with the recovery behind a lossy solid, all in *Recovery behind a lossy solid -
EM, a second transition and a sheet inside the host* and its companion *... a face that was
evaluated at a real angle keeps the solid named*. *The chain builders*: the lambdas `entry`
(`ray_mesh_interact`'s own entry, returning the VV and HH coefficients and the traced cosine and
sine) and `event` (one `ray_state_update` event closing a leg of given normal thickness along a
given direction, with `path_dirN` read back), from which a chain of any length is written event
by event; in that frame VV is TE and HH is TM, which the lossless rows confirm. *The clean-word
twin on a recovered leg*: a clean previous-medium word names air, which is a real invariant too,
so on a leg behind a flagged solid the twin has to be identical, and on a leg behind an unflagged
lossy solid its difference is what an unnamed solve would cost; the real-angle case records that
column. *The real-angle face*: a transition with the non-parallel flag set on its incoming word
makes the first leg's recovery decline through the library's own code, and the direction a wave
launched inside the solid would hold is written by the fixture from `qd_refract`'s map at the
real angle. A reflect pass at a sheet closes the host leg as the transmit pass does, so both
reference legs are the normal-thickness decay.

Three came with the glued exit of a layered wall, all in *Layered wall - a glued exit behind a
bare-entered second layer exits into air*. *The matrix between half-spaces*: two traced solids
with the second one's series resolved sit on `diag_cm_stack` with the first solid as the near
half-space, at its exact cosine, and {second solid, sheet} as the layers, times the single pass
through the first solid. That is what one path is exact for; the matrix of the whole wall is the sum of the paths (the layered wall above). *The next wall*: three events on a second wall, reflect, enter and
leave, fed the words an exit wrote, against the same events with clean words, bitwise. It is the
instrument for "is the ray where the state says it is", which no coefficient on the first wall
can show. *Two recorded columns*, neither asserted there: `out_deg`, the direction the ray leaves
the glued exit in, which reads the external angle on every row and is asserted in the direction
case; and `d_wall`, both series resolved against the matrix of `air | A | B | coating | air`, which is
the first path of a layered wall against the wall and is what the further paths carry; its path
sum has no fixture.

Four came with the recovery gate and the exit direction. *The code a traced run carries*:
`exit_code` in *Recovery gate - the hit code a traced run carries* asks `ray_mesh_interact` for
the code of an exit met along the external direction, bare or with a co-located sheet, so a
hand-built event can be fed the code a traced run would give it. *The bit twin*: the same event
with that code and with the bit cleared, compared on the fields, bitwise, for the transmit pass,
the reflect pass and EM. *The traced run*: the lambdas `traced` and `onward` in *Direction - a
ray leaves a parallel slab parallel to how it arrived* run one event through
`ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update` and hand its state words,
ledgers and direction to the next, on a mesh built face by face with `face(x, flip)`; nothing in
such a chain is synthetic. The relaunch offset moves a leg by a few float ulp, so a traced chain
sits 2e-5 dB off a hand-built one and is bounded at 1e-3 dB, not 1e-6. The first segment has to
reach past the far face: a segment that ends inside the solid preloads the ledger and the next
one adds the leg again. *The direction split*: `d_t` and `d_n`, the tangential and the normal
component of the written unit vector minus those of the incoming one, with `map_t` and `map_n`,
what the real-angle map makes of the traced direction before normalization.

Five came with the refracting tracer, all in *Refracting tracer - a traced bend is known from the
state and the leg is exact*. *The chain runner*: the lambda `run` takes a mesh built by `build`
from a list of faces, a list of interaction types, one repeated until the ray is out or one per
event, and returns the chain coefficients, the in-wall polyline length, the lateral walk between
the first and the last hit point, the written direction, the segment's direction and every
event. *The absolute transfer*: the chain coefficient times `exp(−j k₀ L_in)`, the free-space
reference over the in-wall polyline put back. Each coefficient carries that reference over its
own tracer's polyline, so only absolute transfers can be compared. *The point reference*: the
absolute transfer times `exp(+j k₀ sin θ Δy)`, with `Δy` the polyline's lateral walk, against
the plane-wave transfer of the stack, is the field at the polyline's exit point. Two tracers
are compared there, each with its own `Δy`, and are then one number. *The thickness crossed*:
`ray_mesh_interact` relaunches a few float ulp of the coordinate magnitude past the face, along
the outgoing direction, so a traced leg is short by the normal part of that, 1e-6 m with the
undeviated tracer and up to 1.06e-5 m with the refracting one at 85°, and the series of a dense
slab turns it into 4e-3 dB. The reference slab is given `L − origN_x` of the entry event and
both tracers then sit at 0.000000 dB; rows without a reference compare two polylines that are
each short by their own offsets and are bounded at 2e-3 dB for it. *A band*: with a base
frequency given, `run` takes the geometry, the state words and the geometric ledger from the
base-frequency event and the coefficient, the direction and the refracted ledger from the event
at the other frequency, which is how `ray_progress` runs every frequency but the first.

Six came with the relaunch geometry. In *Refracting tracer - a traced bend is known from the
state and the leg is exact*: *the switch* `relaunch`, a captured flag that makes `traced` hand
`origN` and `destN` to `ray_state_update`; it is off for every table but the last, so the older
rows keep the geometry `ray_mesh_interact` wrote and stay where they are. *`seg_err`*, the unit
segment of an event against its `path_dirN`, largest component. *The twin without the
geometry*: the same chain run with the switch off, compared on words, type codes and, for one
wall, coefficients, which is the instrument for "nothing else moved".

In *ray_progress - refracting tracer: the segment out of a lossy medium is the direction the
wave has*, in `test_ray_progress.cpp`, float and scalar at 1 GHz: *`trace`* runs a ray or a beam
through a wall built face by face and returns the launch configuration after every call, with
the type codes and the hit point of each ray. *The raw column* prints `orig` and `dest` with
every digit of a float for the rows a change must not move, so a diff of the output shows a
moved lossless, undeviated or reflected ray where no assertion would. *The reference that needs
no formula*: a segment or a vertex direction in a second solid against the same solid entered
from air; a split beam against the undeviated tracer's, whose tube is the launched one; the
lateral run between two walls against the undeviated tracer's, the two differing only in where
they leave the first wall. A split in air is forced through `subdiv_flag_in`.

Six came with the phase reference, all in *Phase reference - a traversal carries the phase at
the point the polyline leaves at*, which copies the chain runner. *The two points*: every row
carries the phase at the polyline's exit point, asserted, and at the point where the refracted
ray of that frequency leaves, recorded, with both lateral walks in mm; the ray's is the normal
thickness crossed times the tangent of the direction the entry event wrote. *A band as
`ray_progress` runs it*: the base-frequency event is handed `origN` and `destN` on a transmit
pass, the event at the other frequency is not. *The legs apart*: on a back reflection each
event against its own reference, `r10 φ [S]` for the leg the reflect pass closes and `t10 φ` for
the leg the exit closes, each over the polyline between its two hit points, which says which
form an event applied. *The sheet's four points*: for a sheet inside a host, the polyline's, the
host ray's with the sheet taken as host, the ray's that is also refracted in the sheet, and the
host ray's over the host thickness less the sheet; the last is where a transverse term in the
displaced-host division would put the phase, and is recorded so that the term is not proposed
again. *The traversal as a ratio of two thicknesses*: the absolute transfer through 100 mm over
that through 50 mm isolates the traversal of a body whose face phases have no simple reference,
a leak among them. *What no ledger holds*: per event the polyline length and lateral walk
between hit points against the length and lateral walk the ledger was given, relaunch point to
hit point; the difference is the relaunch offset and, across a co-located pair, the gap, and it
accounts for the phase between the two tracers through two solids to 0.0002°.

Five came with the exits and verdicts, all in *Exits and verdicts - every exit at the cosine
the wave has there*, which copies the chain runner once more. *The far-side coating*: enter,
reflect off a coating on the far face, leave by the front face, with the host series declined;
the ray is resolved all the same, so the row reaches a resolved exit without `eps` 0, and it
has a reference, `t01 R t10 φ²` with `R` the reflect port of host | coating | air. *The tracer
twin on five events*: `air | A | B | air` with every series resolved, enter, cross, reflect,
cross back, leave, the undeviated over the refracting tracer, each at its polyline's point;
no reference, and none is needed for "the two tracers are one wave". *The wedge through the
intersector*: `tilt` turns an exit face about a point on the ray, so the entry event sets the
non-parallel flag through the library's own wedge test; angles are taken against the exit
normal, with `map` and `rmi` the real-angle map of the direction the wave has and of the
segment, which tells which of the two an exit used. *The invariant sweep*: a lossy host entered
from a lossless dense solid, named in the previous-medium word, with the direction
`qd_refract`'s map gives for an invariant `p`, which reaches `p > 1`; `exit_code` supplies the
hit code of the segment, and `ref_of` writes every cosine from `p` directly, with the decaying
root in air above one, bare face or coated, TE or TM. These are hand-built events: no wall
standing in air reaches an invariant above one. *The solve written twice*: `qd_refract`'s map
inverted in the fixture with three Newton steps and with twelve, which shows what three steps
used to leave; the event itself is asserted on the reference at the invariant.

One trap in the phase-reference case: a 2 m segment ends inside a 200 mm slab at 85°, and the next event then
adds the leg again; the thicker body of a ratio is 100 mm for that reason. Another: a pair of
faces 100 µm apart is not one co-located event at 85° (§5).

One trap in the hand-built chain fixtures: they prime the geometric ledger with the normal
thickness and hand in a geometric angle, which do not form a triangle, so the lateral walk of
their "polyline" is not a physical one. Their references repeat the code's statement with the
fixture's own values. The convention itself is pinned by the cases through the intersector.

One trap there: the angle between two float unit vectors must not be taken with `acos` of their
dot product, which has a noise floor of 0.014°; the case uses `atan2` of the cross product.

*Leak direction - a wave held in a mass-law body keeps its direction and a closed face hands on
the wave's*, the last case of `test_ray_state_update.cpp`, has two tables. *Rigid walls*: `chain`
runs a stack through `ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update` and
records per event the direction handed in, the segment, the gain, the direction written, the
codes and the current word. Its `hold` argument makes the fixture do the holding: after an
event that leaves a mass-law medium it hands on the incoming direction, relaunches along it for
type 5 and clears the bent flag, at every such face (`hold`) or only at an exit into air
(`hold_air`). With the code as it stands `now` and `hold` are one run, which is the assertion;
`hold_air` is the option that was not taken and is recorded, and it shows what bending at a
shared face alone costs. Two palettes, `heavy` and `light`, put the bodies far above and near
the knee of the mass term. *The closed face*: a ray launched inside the lossless `eps 5.24`
solid with seeded state words, into the lossy host with an invariant `p`, then a bare exit, a
coated exit or a transition into `RARE`, a tenth palette material of `eps 1`; two events
through the intersector, the second read in full.

One trap in that case, and in any chain that relaunches by hand: the relaunched segment has to
cross every face that is left. A segment that ends inside a body makes `ray_state_update`
preload the ledger for a ray that ends there, and a fixture that then relaunches a fresh
segment adds the leg again; a 2 m segment across a 0.5 m air gap at 75° ended 18 mm inside the
second wall and put 2.1 dB on its exit. `chain` scales the segment to a fixed extent along the
normal for that reason. Stack names must not contain a comma, since the tables are CSV.

One trap in the recovery-gate case's last table: an event with no `path_dir` takes the reported angle as the
in-medium one, so a bare exit out of `eps 9.0` at 20° is past its critical angle, 19.5°. The rows
stop at 10° for that reason.

One trap in `make1`: its default normals are the antiparallel pair of a slab, entry face and exit
face. A media-to-media event takes its plane normal from the second face, and where the
continuation direction it writes is read, that normal has to be reversed to oppose the ray, as the
front face of the next solid does. Left as it is, the event refracts the wrong way and a control
row fails by tenths of a dB to several dB.

Seven came with the layered wall as a sum of paths. In *Measurement - the mirror behind a second
solid and behind a sheet inside a wall* and *Sheet reflection - a declined series is followed
and not dropped*, both in `test_ray_state_update.cpp`, which copy the chain runner once more:
*`run` with a type and an `eps` per event*, so that one event of a chain resolves and the others
decline, which makes the series of any one body a ratio of two runs, and so that a path with
reflections in it is written as its sequence of types (`seq_of`, `seqT`, `seqR`). *`at_point`*,
the chain coefficient as the field at the polyline's exit point, which is what makes paths that
leave at different places addable. *The reference written in the fixture*: `cm`, a TE
characteristic matrix in the admittance `b = sqrt(eps − sin²)` with `rint`, `tint` and `phi`,
ten lines that share nothing with the library or with `diag_cm_stack`, used wherever a mirror
has to be a layer seen from inside a medium. *The limit of a geometric sequence*: the first two
paths give the ratio, the third checks it, and `F0 / (1 − F1/F0)` is compared with the wall, so
a slowly converging row needs no more than three paths. *The mirror as two references*: the
series against the half-space and the layer form with an assertion that the two differ. *The
word after a reflection*: the resolved flag against whether the event's coefficient moved
between `eps` 0 and 1.5. And the state words of a whole path printed as `prev.cur.buf` per
event.

In `test_ray_progress.cpp`, *Measurement - ray_progress follows the coupling between the series
of a layered wall* and *ray_progress - a declined series at a sheet is followed and not dropped*:
*`trace` returning a `Sum`*, one ray launched and `ray_progress` called until no ray is left,
every path that ends in air collected in the call after its exit (its segment count equals the
call count), behind the wall into `T` and in front into `R`, each as
`coefficient · exp(−j k₀ L_in) · exp(+j k₀ sin θ·Δy)` from the path's own coordinates and
`xpr_coeff`, with the shortest path kept apart. It takes the threshold, the interaction limit
and the gain floor, which are the three things the sum depends on. *The four walls*: the summed
paths of a sheet inside a host against 50 | 6 | 50 with the sheet's thickness of host divided
out, 47 | 6 | 47, 44 | 6 | 50 and 50 | 6 | 44, which is how it was found which wall the code
describes. `thin_slab_threshold` is limited to 1, which declines every series.

One trap there: with the series declined on two coupled bodies the paths multiply, 511 behind
two solids after 20 interactions at 10°, so a deeper run needs a gain floor, and a floor of
−70 dB moves the sum by 0.01 dB to 0.2 dB.

Six came with the verdict. In *Measurement - the mirror behind the ray after a back reflection*
in `test_ray_state_update.cpp`, which copies the chain runner: *the leg table*, one row per leg
that begins at a back reflection at a bare face, with `ρ` of the true pair and of the pair the
previous-medium word names from the reference, the threshold at which the first leg in that
body stops summing found by *bisection on `eps`* (30 steps on one event of a chain whose other
events decline), and a window column that is 1 where the true pair declines and the word's pair
would resolve at 0.15; the columns `|S|`, `d_true`, `d_word` and `rho_lib` describe the series
that event forms, which is none: `|S|` is 1 and `rho_lib` reads −1. *The walker*, `walk`: one
ray, both passes at every event, every ray followed until it is in air, dead, 20 events old or
below a floor, summed at the polyline's point; it runs the library as it is, or with `eps` 1.5
on every leg that began at a declined reflection, or with the mirror written into the
previous-medium word by the fixture, and counts the events on such a leg that summed a series.
With the mark in place the three agree; they are kept as the instrument for any later change to
who forms a series. *The verdict rows*: `traced` takes `verdict_inN` and `verdict_outN`. In
*Measurement - ray_progress and the mirror behind a back reflection* in
`test_ray_progress.cpp`: *`trace` over a band*, a `Sum` per frequency from `xpr_coeff(f)`, with
the shortest reflected path kept apart; *the gain-floor table* for a sheet inside a wall, both
regimes at floors of −100 to −160 dB against the run with every series summed and no floor;
and *the near-side coating over frequency*, the reflected sum against the matrix with the
coating in front of the mesh plane.

Seven came with the reflect port of a sheet, in *Sheet reflect port - where the layer lies and
what the echo pays* in `test_ray_state_update.cpp`, which copies the chain runner and the walker,
and *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer*.
*The plane a sum is referenced to*: a reflected sum times `exp(−2j k₀ cos θ·x)` with `x` the
plane the polyline entered at is the plane-wave reflection referenced to `x = 0`, which is what
makes a sheet on a wall and a traced solid in front of that wall one number. *The walker through
air*: `walk` follows a ray in air between two bodies like any other and collects it only
outside the structure, with the polyline length counted over every segment between the first
and the last hit; `trace` in the `ray_progress` case does the same. *The three walls*: a sheet in
front of a wall with its layer behind the face, centred and in front, each referenced to the
face, with an assertion that the other two are told apart. *`on_wall`*: a reflected sum against
its wall in dB and degrees, or, where the wall reflects 10 dB less than the path off its first
face, as its distance from the wall over that first path.
*The stateless chain*: `traced` with its `stateless` switch stops after `ray_mesh_interact`, and
the lambda `stateless` multiplies those coefficients along a sequence of types with the host legs
as `exp(−j k₀ n L)`, which is what a caller without state has; three paths give the limit.
*`xpr0`*: what `ray_mesh_interact` wrote, kept beside what the redo made of it. *The sheet
against its solid*: one material as a sheet on one face and as two faces a thickness apart, the
solid walked with its series summed and with every bounce followed.

Two traps there. A cavity in air is followed bounce by bounce and converges slowly: a sheet
30 mm in front of a lossless wall at 70° and 3 GHz is 0.059 dB short with 60 interactions and a
floor of −120 dB, and 618300 paths are behind the wall at 6 GHz with 110 and −200 dB. And a
traced solid of 20 mm is short by the relaunch offset, 3.2e-3 dB on its transmit port at 85°, so
a sheet and its solid meet at 1e-3 dB and not at the 1e-9 of a closed form.

One thing to know about fixtures that call the library one ray at a time, many thousand times:
the OpenMP thread team costs twenty times the work. Both measurement cases hold a scoped guard
that sets one thread and restores the count; without it the first takes 3.5 minutes where it
takes 11 seconds.

One thing to know when reading residuals: the last digit of a reference-against-reference table can
move when unrelated code in the same translation unit changes. The complex-cosine column of the
characteristic-matrix-against-builders case moved from 1.166e-15 to 9.437e-16 worst when `DiagLump`
was added, while the builders themselves compare bit-identical before and after in a standalone
translation unit at `-O0` and `-O3`. The likely cause is compile-time evaluation shifting with
inlining; that is not confirmed. Bounds on such tables are 1e-14 for that reason and should not be
tightened.

`diag_cm_stack`, with `DiagLayer`, is the N-layer characteristic-matrix reference, returning the same
`DiagSheet`. It takes `(ε, μ)` pairs interleaved, and needed no change when the two single-layer builders
acquired theirs. At
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

Where a fixture already isolates the quantity by construction, the ratio is the wrong instrument
instead, because it discards the absolute level. The walk-off case is the example: with no `path_dir`
supplied `REPLACE_BY_VBS` fast-returns without writing a coefficient and the correction-mode series
declines, so `out / in` is the traversal factor outright and its magnitude is worth asserting.

**The lossy-verdict cases.** *Lossy verdict - a lossy second medium has no critical angle* in
`test_ray_state_update.cpp` drives one ray at a time through `ray_triangle_intersect`,
`ray_mesh_interact` and `ray_state_update` with the lambdas `traced`, `build`, `onward`, `chain`
(one path, a sequence of interaction types) and `walk` (every path), and the scalar reference
`cm`, which shares nothing with the library. Its sections: A a porous sheet in air; B on a rigid
wall; C the outer pair on a lossy half-space; D the traced solid, per event and every path; E the
traced solid on a leaking wall against the sheet; F the ray tube; G and H the EM counterparts,
the latter launched inside a dense lossless solid with a hand-built state; I the recovery in
three rarer lossy media; J a rigid wall with and without a loss column and a mass law; K the
direction a traced absorber and a sheet hand a mass-law wall. J's rows with a loss column
are recorded and not asserted: they are §5. K is asserted.
*ray_progress - a lossy medium has no critical angle and a band takes one path* in
`test_ray_progress.cpp` does the same through the tracer, with `trace` returning one `Sum` per
frequency of the band. Both use the one-thread guard: a fixture that calls the library one ray
at a time spends its time in the thread team otherwise.

**The closed-face cases.** *Leak direction - a closed face hands the wall behind a traced
layer the direction the wave has in air*, the last case of `test_ray_state_update.cpp`, with
the one-thread guard. Its lambda `run` builds one layer on one wall three ways - a traced solid
(entry, layer | wall, wall | air), the layer as a sheet (glued entry, wall | air) and the wall
bare - and `chain` relaunches each event on the segment the one before wrote, with `origN` and
`destN` handed to `ray_state_update`. Tables: A the directions and segments; B the levels, with
*the mass term as a ratio* (the wall against its twin whose `fRef` is 1e9, which leaves the
`m > 0` gate, every face and `tf` in place) and *the series as a ratio* (`eps` 0 over 1.5); C
*the wall | air event with what it is handed changed one thing at a time* (the incidence
direction with and without the segment relaunched along it, no direction, a clean
previous-medium word), which is the instrument for "which input does this event read"; D a
host entered from the lossless `eps 5.24` solid the ray is launched in, in front of the wall,
invariant 0.50 to 1.45; E the ray tube, with `tridir` and `tridirN` handed in; S the solid
against the sheet, worst over the angles, with every digit. The series columns and the
direction behind a wall with `tf` and no `m` are recorded. *ray_progress - the wall behind a
traced layer is met at the incidence angle at every frequency of a band* in
`test_ray_progress.cpp`: `trace` returns per frequency the power sum behind the wall and, for
the strongest path, its power, the direction it is handed on with (`path_dir_prev` of that
frequency), its segment and its lateral walk; the level against the sheet is recorded there,
the layer's series being followed at the default threshold.

**The plate-depth cases.** *Bonded mass - the plate at any depth where the slopes agree*, the
last case of `test_ray_state_update.cpp`, and *ray_progress - the bonded plate at any depth where
the slopes agree*, the last of `test_ray_progress.cpp`, both with the one-thread guard. *The
token builder*: a stack along x is a list of `O` / `C` (open and close a solid), `S` (a sheet),
`N` / `n` (an object nested in the solid the ray is in) and `G(d)` (advance); faces with no
advance between them are co-located and one event, `reversed` gives the stack from the other
side, and `bodies` derives the bodies in crossing order with the event that pays each. *The twin
palette*: every `fRef` times 1e12, which takes every mass argument to nothing and keeps the ratio
of `fRef` between bodies, so the thickness codes, and with them every state word, are the run's;
a twin that sets `fRef` to one value for all bodies writes other codes wherever `fRef` differs
and is the wrong instrument for a chain three bodies deep. *The mass paid per event* is the twin
over the run, and *the increment the plate takes at that event* is printed beside it, with the
three words each event wrote. *`rest`*, the chain on the twin palette, is everything but the
mass and is what a change to the mass rule must not move. Each stack carries a two-bit `exact`
flag, one bit per direction: set, the rows are asserted on the plate; clear, they are recorded.
Tables: A every stack, S the range of paid minus plate per stack, B a bounce (forward to an
event, reflected there, back out), C each frequency alone with its words compared to 1 GHz. The
tracer case takes the strongest path behind the stack against the same path on the twin, per
frequency of a band.

**The branch cases.** Nine cases at the end of `test_ray_state_update.cpp`, named *Unentered
branches - ...*, each for a branch that had a guard and no row. They share the namespace `ub`
instead of copying the chain runner once more: `ub::traced<T>` runs one event through
`ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update` at dtype `T`, with everything
handed in and returned in double, and takes an `ub::Opt` (`eps`, the relaunch switch, a verdict to
hand in, a ray tube); the event it returns holds what `ray_mesh_interact` wrote (`xpr0`, `dir0`)
beside what the redo made of it, both hit points, the three words, the ledgers, both type codes
and the verdict handed out. `ub::face`, `ub::tilted` (a face through a point, turned about z, for a
corner) and `ub::build` make the mesh; `ub::onward` relaunches on the segment an event wrote;
`ub::chain` runs a sequence of types with an `eps` per event and `ub::chain_t<float>` the same at
float; `ub::at_point` is the field of a path at the point its polyline ends at; `ub::same_bits`
compares two matrices bitwise; `ub::OneThread` is the one-thread guard. Instruments that came with
them: *the thin-sheet ratio*, the event over the same event with a 1 mm sheet, which isolates a
composite inside a host without a reference for the host leg; *the host of an event*, formed in
the fixture as normal thicknesses: the leg in front from the relaunch point to the sheet's
face, the gap behind from the two hit points; *the loss sweep*, a 1 mm layer of `eps 1 − jE` with `E` set
for a one-way loss in dB; *the pole*, a lossless slab of `eps 1e4` whose frequency is tuned from
the thickness the leg crossed and detuned by a round-trip phase; *the seeded state*, an event
struct with the current-medium word, a direction and zero ledgers set by hand, for a ray launched
inside a solid; and *the laminate twice*, a stack against the same layers meshed as a traced solid
with a lining, the reflected paths referenced to the stack's plane.

Traps met there. A case name must not contain a comma. `arma::accu(arma::square(...))` links
against a BLAS wrapper the test binary does not have; sum by hand. `arma::abs` of an xprmat is the
magnitude of its real and imaginary rows, not of the complex entries; use `cpx`. `diag_leak_series`
applies where the outer pair is closed and the sheet was entered by an ordinary crossing, which at
exactly normal incidence on a rigid wall it is not. A single event of two different polylines
cannot be compared in phase: with the refracting tracer `ray_mesh_interact` bends the segment at
the face of a deferred body (§5), and each coefficient is the field at its own polyline's point.

**The mass-cosine case.** *Mass cosine - a mass-law sheet behind a refracting layer forms its term
at the cosine the wave has in air*, the last case of `test_ray_state_update.cpp`, on the `ub`
helpers, with the one-thread guard. *The mass paid* is the chain on a twin palette whose mass-law
`fRef` is multiplied by 1e12 over the chain itself; every face, `tf` and `m > 0` gate stays in
place, so the ratio is the mass term and nothing else, and nothing about the faces of the sheet
enters it. Tables: A the lining, the sheet inside and the two stack orders inside
a host, with `rest` (the chain on the twin, what a change to the mass term must not move), the
direction and the segment behind the wall and the last hit point against the same host without the
sheet; B the laminate from its two sides and the stack in both orders, with the recorded column
`d_sides`, the whole transmission from one side over the other, which reads the flux splitting of
the coherent layer and is asserted in the closed-plate case; C a wedge built with `ub::tilted`; D the invariant sweep with a seeded state, launched in
the lossless `eps 5.24` solid; E a ray with no state and `ray_mesh_interact`'s own result (`xpr0`).
The segment and hit-point columns of the stack rows of table A are recorded; they read rounding,
a stack keeping the line it was reached on, which the stack-geometry case asserts.

One trap there: `diag_ray_at(theta)` starts at x = −1, so the ray crosses x = 0 at
y = 0.37 + tan θ and not at 0.37. A tilted face has to be turned about that point, or at 30° and
60° the ray meets the wedge past its apex.

**The closed-plate case.** *Closed plate - a mass-law layer the wave is evanescent in passes its
two leaks and nothing else*, the last case of `test_ray_state_update.cpp`, on the `ub` helpers, with
the one-thread guard. *The level without the mass term*: the chain on a palette whose mass-law
`fRef` is multiplied by 1e12, read directly and not as a ratio, which is what shows a face that
passes too much. *The loss sweep to zero*: the coherent layer at ε″ 0.8, 0.1, 1e-3, 1e-6 and 0,
with the last two compared, which is the instrument for a branch that is taken on a value and
not on the medium. *The splitting beside the two sides*: `split`, `sec²(arg η)` of the
lossy layer per channel from its permittivity and the incidence angle, recorded beside one side
over the other, which is asserted at zero. *The event of a lining three ways*:
on the near face, on the far face and inside the host, with what `ray_mesh_interact` wrote for the
event (`xpr0`) beside the redo. *The twin table with every digit*, table E: the complex
coefficient of the rows a change to the closed layer must not move, printed with 17 digits, so a
diff of the output shows a moved bit where no assertion would. Tables: A the stack in both orders
and the traced laminate from both sides; B `tf` on the neighbour and none on the layer; C the
sheet event; D a rigid with a non-rigid mass-law layer; E the twins; F1 a mass-law sheet of
`eps 0.50` across its critical angle, recorded, with the level bounded below one; F2 a rigid
lining on a rigid traced wall, recorded, whose wall-first rows are asserted in the held-lining case
below (the header of F2 still says the lining is entered at the Snell cosine, which it no longer
is towards the lower index); F3 a coherent sheet with `tf` rarer than its
lossy host, recorded: it reads the transformer of §1 and is asserted in the rare-sheet
case below, and its header still names the item it was recorded for. A row that passes nothing ends its chain at that event, so table B
reads a short chain as a zero.

One trap there: Catch2 prints the expansion of a failed comparison of two long doubles on three
lines, so a script that counts failures by the first line of the expansion undercounts.

**The rare-sheet cases.** *Rare sheet - a coherent sheet with tf that is rarer than the invariant
passes no more than it received*, the last case of `test_ray_state_update.cpp`, on the `ub`
helpers, with the one-thread guard, and the case of the same name with the prefix
`ray_progress -`, the last of `test_ray_progress.cpp`, float, both tracers, scalar and EM, one
ray with no reflection followed. *The two ports of one event*: `inside` runs the sheet event
inside a host twice, with a transmission and with a reflection as its last type, so a row
carries `T`, `R` and `T + R`; the levels are those of the event with its host leg, because the
composite alone divides out the sheet's thickness of host and reads above 0 dB in a lossy one
(§5). *The loss sweep to zero twice*: of the host, and of the sheet's own loss, each with 1e-6
compared with zero. *`tf` to zero*: ±1e-9 against none, asserted at every loss
and in both channels. *The critical angle of the sheet as a
pair of neighbours*, 33.80° and 33.87°. *The lining from its two sides*, which needs no
reference where the host is lossless. *The transmit port under a closed outer pair with every
digit*, table E, beside the reflect port of the same event. Tables: A inside a host, A2 `tf` to
zero, A3 the critical angle, B a loss of its own, C the lining, D the solver
in both orders, E under a closed outer pair, F twins with every digit - a sheet denser than
the invariant with `tf`, a rare sheet without `tf`, a rare sheet below its critical angle, a
mass-law sheet, a rigid mass-law sheet and a sheet under a closed outer pair, whose rows keep
the label `leak chain`.

The headers and comments of these two cases still describe the rule they were written for, a
pair rule continued past the critical angle; their assertions are bounds and equalities that
the transformer meets, and table A asserts the sum of the two ports for the lossless host only.

**The sheet-tf cases.** *Sheet tf - the face of a coherent sheet with tf creates nothing and has
no gate on the invariant*, the last case of `test_ray_state_update.cpp`, on the `ub` helpers,
with the one-thread guard, and the case of the same name with the prefix `ray_progress -`, the
last of `test_ray_progress.cpp`. *`ports`*: both ports of one sheet event in four
configurations, alone in air, inside a host, glued on its near face and glued on its far face,
with the host leg where there is one. *The reference of the rule in powers*: `kz_of`, `face_ref`
and `sheet_ref`, a sheet between two half-spaces in tilted admittances - `kz` for TE and the
scalar type, `eps / kz` for TM - with a transformer of ratio `sqrt(nu)` at each face, which
shares nothing with the library or with the other builders; it returns false within 1e-6 of the
tie of a forced face (§5). *The summary row*: table A prints, per configuration, sheet, `tf` and
channel, the worst sum with its angle and thickness, the count of rows above one and the worst
distance from the reference, 480 rows for 16,000 events. *The step against the twin without
`tf`*: the reflect port of two neighbouring angles in amplitude, less the same difference with
no `tf` anywhere, which removes the slope the matrix has. Tables: A both ports; B a lossless
sheet below its critical angle on the pair rule; C a sheet without `tf` on the matrix; D the
step at the gate; D2 recorded, the tie of a forced face; E the end points; G recorded, both
ports on a rigid wall with `tf`, which the closed-face case asserts; S the ordered stack; H twins with every digit, 26 families -
faces in phase, faces not in phase, no `tf`, mass-law, under a closed outer pair, stacks. In the
tracer case: *`trace` returning both sums per channel*, one ray, `ray_progress` called until no
ray is left with every series summed and 60 interactions, VV and HH; A the two ports, B the
step.

Traps met there. The EM reflect port at exactly normal incidence is not on the references'
basis, with or without `tf`: tables B and C start at 5°. `T + R` of an event is not what a face
conserves next to a lossy medium, so inside a lossy host only each port can be asserted. The
critical angle of a lossless sheet is a kink of the reflect port with `tf` and without. The two
halves of a wall with a sheet inside it are coupled by paths that converge slowly for a lossless
wall towards grazing: 1.0076 after 20 interactions at 73°, 1.00002 after 60. And `eps 0.90 −
0.20j` in air has an admittance ratio of modulus exactly one at 60° in TE, where a reference
and the library take a forced face to opposite sides on rounding.

Three instruments from the analysis are not in the suite and are quick to rebuild. *The generic
probe*: a case that reads one sheet event per line from a file named in an environment variable
- configuration, type, angle, the sheet's and the host's columns - and prints both ports with
every digit, so that a new question costs no recompile; with the same lines for a traced solid
of the sheet's material, both ports as sums of two paths with the series summed. *The face
model*: a few lines of script for one sheet between two half-spaces in tilted admittances with
the face rule as a function, which reproduced the library's two ports to four digits on every
row checked and scans a hundred thousand rows in seconds; a candidate rule is tried there first
and confirmed through the probe. *The thickness sweep*, 1 mm to 200 mm.

**The flat-branch cases.** *Flat branch - a coherent sheet rarer than the invariant without tf
tunnels on the characteristic matrix*, the last case of `test_ray_state_update.cpp`, on the `ub`
helpers, with the one-thread guard, and *ray_progress - a coherent sheet rarer than the
invariant without tf tunnels*, the last of `test_ray_progress.cpp`, float, both tracers, scalar
and EM. *The matrix as the reference of the tunnelled wave*: `diag_cm_stack` for
near | sheet | far, at the complex cosine `sqrt(1 − sin²θ / ε_host)` where the near medium is a
host. *The ratio against a sheet that is never rare*: inside a host and at a glued exit the
event carries its host leg, so the event over the same event with a 1 mm sheet of `eps 3.0` is
compared with the ratio of the two matrices, as a relative distance in the complex plane; the
leg cancels and the comparison holds at 1e-9 for both tracers. In air and at a glued entry the
event is the composite and is compared with the matrix directly. *`pick`*, the port and
channel of a `DiagSheet`; scalar rows hand the matrix its scalar flag and read the TE channel.
*The sum of the two ports* is asserted for a lossless host only: in a lossy one the matrix
itself exceeds one (§5). *`tf` to zero on every geometry*: inside, in air, entry and exit,
±1e-9 against none in dB on both ports. *One sheet against its halves*: 2 mm + 4 mm of the same
permittivity as two materials, which the redo resolves as an ordered stack through the solver;
the two faces are 10 µm apart in the mesh, so in a lossy host the bound is 2e-3 dB. *The
rigid sheet twice*: with `m` 20 and `fRef` 1e9 and with no column. Tables: A inside a host; B
in air; C the lining; D `tf` to zero; E the halves; F under a closed outer pair without `tf`;
G the rigid sheet; H twins with every digit, twenty-two row families - sheets denser than the
invariant, lossy, below their critical angle, with `tf`, mass-law and under a closed outer
pair, beside the rows of the case - so that a diff of the output shows which rows a change
to a face reaches. In the tracer case: *the lossless slab written from its two Fresnel faces*,
a lambda of eight lines with the decaying root, since the reference builders of the other file
are not visible there; and *the `tf` twin as the reference of a wall*, the same wall with `tf`
±1e-9 on the sheet, which needs no formula.

One trap met while writing it: a sheet has to fit its host. Each half of it takes at most the
host on its side, so a 22 mm sheet 10 mm inside a host is capped in front and sits off the
matrix of the meshed wall on both ports; the rigid sheet stands in 100 mm + 100 mm for that reason.

**The held-lining case.** *Held lining - a mass-law sheet behind a wave held in a mass-law wall
is the same wall as its traced solid*, the last case of `test_ray_state_update.cpp`, on the `ub`
helpers, with the one-thread guard. *The solid twin as the reference*: every sheet geometry is
built a second time with the sheet meshed as a traced solid of its thickness, which needs no
formula and is what a rule for a lumped body has to reproduce; the builders are the lambdas
`lined` (a wall of 100 mm with a body on one face: lining met last, lining met first, the two
as solids in either order), `inside` (X 50 | Y | X 50, sheet or three solids) and `stacked`
(X 50 | Y Z | X 50, an ordered stack or four solids), over a list of `Body` (name, sheet
material, solid material, thickness). *`level`* is a chain coefficient, zero where the chain
ended early. *The reflect port against the twin's face*: enter and reflect, sheet against
solid, compared in the complex plane because both turn around at one plane. *The angle pairs
across a critical angle*: 19.4° and 19.6°, 29.9° and 30.1°, 41.7° and 41.9° straddle the
critical angles of W | K, T | K and W | T, and neighbours closer than 0.5° are asserted
continuous. Tables: A the lining towards the lower index, three pairs; B types 1 and 2 per
channel; C a sheet inside; D the reflect port; E two sheets inside, every order; F the mass
paid, with every digit; G twins that must not move, with every digit - towards the higher
index, a wall without the mass column, a lining met first, a mass-law lining that is not rigid
and the solid twins; H recorded, with a bound of 0 dB - what the rule does not reach, issues 2
and 6 and the lining without `tf` of §5. The palette carries three rigid bodies, W, K and T
(`eps 9.604e-9`, `tf` 0.15), so that a stack has three different mappings.

One trap met while measuring for it: a mass-paid ratio is meaningless on a row whose level is
at the rounding of `1 − R0`, a lining without `tf`; it read 43 dB there.

**The stack-geometry cases.** *Stack geometry - an ordered stack leaves the ray on the line it
arrived on*, the last case of `test_ray_state_update.cpp`, on the `ub` helpers, with the
one-thread guard, and *ray_progress - an ordered stack leaves the ray on the line it arrived
on*, the last of `test_ray_progress.cpp`, float, scalar, a band of 1 and 2 GHz. *The four
orientations of the two normals*: `front|back` (outward, what an exporter writes),
`front|front`, `back|back` and `back|front`, as the `flip` pair of `ub::build`; any statement
about a stack has to be made on all four: the order of the layers is read from them. *The single-sheet twin*: the same mesh with the first layer alone in
place of the stack, which leaves the ray where it was and needs no formula. *The undeviated
tracer as the reference of the refracting one*: for a stack the two segments are one, and the
column `bits` records that they are bit for bit. *`fan`*: three vertex rays about the spine,
handed to `ub::traced` through `ub::Opt`, each compared with the direction it arrived in. *The
ray out of the horizontal plane*: the basis of an EM coefficient does not show on a ray in the
plane z = 0, where TE and TM are the two basis vectors whatever the azimuth is; table C2 runs
from `(−1, 0.37, 0.2)` to `(1, 0.37 + 2 tan θ, 1.5)`. *`sig`*, a weighted sum of a matrix
printed with 17 digits, which shows a moved bit of a coefficient in one column. Tables of the
first case: A one event in air with a tube, 7 pairs, 4 orientations, types 2, 5, 1 and 4; B the
stack in the middle of 22 mm of a lossless and a lossy host, bare-entered and behind a glued
sheet, with a wall 0.5 m behind, five events; C1 what must not move with every digit, a single
sheet, a duplicate, a glued entry, a bare entry, and the stack with types 1, 4, 0 and 3; C2 the
stack event itself. Of the second: A the transmitted path in air and in both hosts against the
single-sheet twin and the undeviated tracer, B a beam split in air behind the stack, C the raw
column. The palette of the first case: C1, C2 and L are coherent sheets without `tf` and
without the mass column; R, N and Q are a rigid mass-law sheet, a mass-law sheet that is not
rigid and a rigid sheet with `tf` and no mass column, so types 1 and 2 on those rows are not EM
materials.

Two traps there. Behind a glued entry the exit of the host can carry resolved type 3 where a
bare-entered one carries 1 (§5), so the exit behind a stack is compared with the twin's and
not with a constant. And a table whose fields contain `|` must not be pasted side by side with
`|` as the delimiter.

**The bare-tf cases.** *Bare tf - a bare face with tf is a transformer where it is open and a
flat leak where it is closed*, the last case of `test_ray_state_update.cpp`, with the one-thread
guard, and *ray_progress - a traced solid with tf creates nothing*, the last of
`test_ray_progress.cpp`, float, both tracers, scalar and EM. *The walker*: `bf::walk`, in a
namespace in front of the case, sends one ray through `ray_triangle_intersect`,
`ray_mesh_interact` and `ray_state_update`, runs both passes at every event, relaunches each
event on the segment the one before wrote, and sums every path that is back in air at its
polyline's point, in front of the wall into `R` and behind it into `T`; with `eps` 0 every
series is summed, with 1.5 every bounce is followed, and `inc` is the sum of the path powers,
which no interference enters. The lambda `ports` runs it on a mesh and `event` returns one
coefficient of one event. Tables: A the summary of 3024 walls, worst sum with its row and the
count above one, per solid, `tf`, type and mode; B one face per channel, with the Brewster
angle of `eps 5.24`; C the solid against the sheet; D a sheet of air against the bare wall; E
the rigid wall, with every digit; F the rigid wall with a loss column, which has to fail
gracefully; G the forced reflection at an exact match; H recorded, the wall about the tie of a
forced face; T twins with every digit, solids without `tf`. In the tracer case: `trace` as in
the sheet-tf case, with three walls - the solid alone, behind a 6 mm coating, and behind 20 mm
of a lossless solid.

Traps met there. *The followed mode runs away on two solids*: the tree branches at every event,
and a depth of 60 does not return; at 13 events the sums are not converged and are no
reference. Two solids are summed through `ray_progress`, which is valid: lossless rows read
1.0000. *A followed series can diverge without a defect*: in a lossy layer rarer than air
between strong reflectors the exact round trip `|r1 r2 phi²|` exceeds one, 1.19 for 6 mm of
`eps 0.31 − 0.20j` behind a coating at 85°, in the formula and in the library alike, and only
the summed row means anything (§5). *A coherent sum over the paths of a `tf` leak is not an
energy statement* (§5); `inc` is. *The exact tie of a forced face*: `eps 0.90 − 0.20j` in air
at 60° in the scalar and TE channel, where rounding decides the side; table C runs at 65°.
*The EM reflect port at exactly normal incidence* has its own basis, as in the sheet-tf case.

Instruments from the analysis that are not in the suite. *The generic walker probe*: the same
walker in a case that reads one wall per line from a file named in an environment variable -
type, angle, depth, the materials and the faces - so that a scan of ten thousand walls costs
twenty seconds on one thread and no recompile. *The marker with a key*: a counter in
`Material::interact_with` keyed by mode, port, verdict, which side is lossy, the mass column and
the sign of `tf`, dumped per test case by the listener below, gives the table of which cases
reach a branch in one run. *Variants as real code*: each candidate written as a patched copy of
the source and compiled, so that the line count and the scan come from the same text. *The
tf-gated twin*: the candidate applied only where a `tf` is present, diffed against the
candidate applied everywhere, isolates what a boundary without `tf` moves.

**The scalar-closed-face case.** *Scalar closed face - a lossy box on a rigid wall reflects no
more than it received and is its sheet*, in `test_ray_state_update.cpp`, with the one-thread
guard. Five porous rows of `data/materials_ac.csv` at 1 kHz at the thicknesses of the modelling
guidelines, on concrete, no `tf` and no mass column. Tables: A the reflect port of the box over
every path with `bf::walk`, series summed and every bounce followed, against the same layer as a
sheet, types 4 and 5, with type 1 (VV and HH) as the control; B one path, enter the box and
reflect off the wall, scalar against the VV channel, with the Fresnel reflection written from
the two media beside it, which is the cleanest statement of a magnitude rule at a face; C twins
with every digit - a lossless box, the lossy box in VV and HH, the bare wall, the lossy box's
front face.

**The closed-face case.** *Closed face - tf at a closed face is a pad and the two ports of a
sheet on it add up*, the last case of `test_ray_state_update.cpp`, on the `ub` helpers, with the
one-thread guard. *`cf::walk`*, in a namespace in front of the case: `bf::walk` with a sink
medium, a ray being collected where it is in air or has entered that medium, which makes a wall
behind a layer a half-space and gives both ports of a sheet on a wall and of its traced solid
from one walker; and with a ray launched inside a medium by a seeded state, whose first event
forms no series there. *`ports`*: both ports of one sheet event on a rigid wall, or of a coating
met from inside a host the ray is launched in. *`pad_ref`*: the reference of the rule in powers,
in tilted admittances - `kz` for TE and the scalar type, `eps / kz` for TM - with a transformer
at the near face and, at the far face, the transformer where the far side carries a wave and the
pad where it does not; it returns false at the tie of a forced near face. *The loss sweep with
1e-6 as the reference row.* *The critical angle of the outer pair in two halves*: across it the
far cosine has a square-root cusp and two neighbours 0.02° apart differ by 0.1 to 1.2 dB on the
exact rule, so the event is asserted on the reference on both sides and the reference's two
limits at an invariant of 1 ± 1e-10 are asserted equal. Tables: A both ports, a summary row per
sheet, `tf` and channel; C the loss sweep, 1 to 200 mm; D the critical angle of the sheet as a
pair of neighbours; D2 that of the outer pair; E the sheet against its traced solid; F a lossy
layer with a closed plate as a stack and as a laminate from both sides; G the end points - no
`tf`, `tf` −1 and the plate with the mass column, asserted on itself across thickness; H twins
with every digit.

Three traps met there. A number written by hand for the plate, `0.5·0.65`, was wrong in the
fifth digit; the far face of a plate entered as a leak is what the other direction charges and
not a flat leak. With a ray launched inside a host only the followed rows of a sheet event mean
anything: the summed mode forms the host's series on the seeded leg, and the coherent sum of a
traced solid launched inside a host is not validated. And a sum of path powers is not an energy
statement next to a lossy medium: a single path out of a lossy medium is not bounded by one.

Instruments from the analysis that are not in the suite, kept in `build_linux/scratch/i43`. *The
walker probe with a sink and a start medium*, one wall per line from a file named in an
environment variable, 24,000 lines in ten seconds on one thread. *The formula model of a sheet
on a closed far side*, `model.py`, whose rule for the code as it was reproduced the library on
13,440 values to the nine printed digits. *The variant generator*, each candidate rule a
patched copy of the source, so that line counts and measurements come from one text. *The
column reader*, `colread.py`: every moved table read column by column, with the count of rows
and the largest move per column and its row, which is how a table is read in the unit it
reports. *The key reader*, `rows.py`: two runs of one table paired row by row.

**The one-solver case.** *One solver - a sheet and a stack are one sum and no mirror is
clamped*, the last case of `test_ray_state_update.cpp`, on the `ub` helpers with `bf::walk` and
`cf::walk`, with the one-thread guard. Tables: A a solid with a coating on either face or both,
every series summed, against `diag_cm_stack`; B a coating on the lossless `eps 5.24` met from
inside (*the seeded leg*: `prev` 0, `cur` 1, a direction and ledgers of 0.1) against the matrix,
and the same layer as a traced solid; E a lossless sheet at a transmission resonance,
`thk = N·π / (k₀ sqrt(ε − sin²))`; F the own coefficient of an ordered stack in the four
orientations; G a stack inside a host - the series as a ratio against both mirrors, the buffer
word, and every path against four layers on `diag_cm_stack`; H a forced matched face from its
two sides, the same ray on a mesh with the two materials swapped, through `ray_mesh_interact`
alone. Traps met: at 89.9° `diag_ray_at` misses the face, so grazing rows stop at 89°; a wall of
two cavities walked with every bounce followed never returns, so table G walks with the series
summed and a depth of 26.

Instruments from that analysis that are not in the suite, kept in `build_linux/scratch/i44` with
a `NOTES.md` of every measured figure. *The switch copy*, `mkwi.py`: the source with a counter per
site and one switch per candidate change behind a bitmask in an environment variable, so one
object serves every variant of the suite and of the walker. *The pair probe*, `wi.py` with
`libwi.so`: the copy as a shared library called from Python, one layer or N layers through the
solver and one face through `Material::interact_with`, materials handed across as the 23 doubles
of `Material`; half a million values in two seconds with no recompile. *The variant generator*,
`mku.py`: a change written as real code with one option per decision, so line counts and suite
runs come from one text. *The walker with a permeability per material, a bent flag on the seeded
leg and a summed-only switch*, `probe.cpp`. Two readings to know: a ray launched inside a host
with a seeded state and type 4 is a leg the state does not know to be bent, and its summed rows
are the instrument's - type 5 with the bent flag on the seed gives summed equal to followed; and
a followed row above one whose dropped power is not zero is a series cut at the depth limit.
**The budget cases.** *Unentered branches - the displacement budget caps what a sheet takes back
out of its host*, among the branch cases: the single sheet and the ordered stack inside a host,
each event over the same event with a 1 mm sheet against the same ratio of `diag_cm_stack`,
which is called twice, with the sum of the two parts for the transmit port and with twice the
part in front for the reflect port; the stack rows carry `fits` and `cap`. *Budget - a sheet or
a stack inside a host is one wall on every port whether it is capped or not*, the last case of
`test_ray_state_update.cpp`, with the one-thread guard: the lambda `walls` builds a sheet or a
stack of a given thickness at a given depth in a host, walks every path with `bf::walk` and the
series summed, and compares the two ports together with one, and where the layers fit with
`diag_cm_stack` of air | host | layers | host | air. Families: `lossless`, which is the
instrument for a placement that is not one wall - any inconsistency between the two ports and
the mirror shows as a sum off one, where a lossy wall hides it; `offcentre`, a lossless sheet
in a lossy host that it does not fit; `long`, recorded, the stack thicker than its whole host
of §5. *ray_progress - a wall with a stack inside it is on its matrix with the paths of a
single sheet*, the last case of `test_ray_progress.cpp`: one ray, the paths and the rays
launched counted over all calls, the TE matrix of the wall written in the case.

Instruments from that analysis that are not in the suite, kept in `build_linux/scratch/i45`
with a `NOTES.md` of every measured figure. *The event probe*, `probe.cpp`: one sheet or stack
inside a host per line of a file, both ports of the event against the uncapped reference, both
host series against their mirrors, and every path against the wall, per channel. *The tracer
probe*, `rp.cpp`: a program of its own, without the test framework, that sends one ray through
`ray_progress` until none is left and prints the paths behind and in front, the rays launched,
the run time and both sums against a matrix, for a threshold, an interaction limit and a gain
floor per line; each line is run in its own process under a timeout, because a wall of two
cavities with every series declined launches millions of rays near grazing. *What an event
sees*: a print in the copy of the leg, the accumulated leg, the distance between the two hit
points, the distance to the destination and both cosines; through `ray_progress` the
destination of a stack event is the rest of the segment, 297 m, and the hit count is 3. Two
readings to know: a copy with switches differed from the baseline in 9 residual rows at 2e-15
with every switch off, where a plain copy reproduced it exactly, so the plain copy is checked
first and the switch copy is diffed against its own off-run; and the reflect-port factor
written as the square of the one-way factor instead of the factor over twice the length moves
842 rows at rounding where the form taken moves 414.

**The resonance cases.** *Lumped columns - a permittivity resonance is part of the medium on the
sheet paths*, in `test_ray_state_update.cpp`: the isolated, glued and embedded sheet paths against
`diag_sheet_composite`, the embedded row in power with its host leg, and the matched sheet.
*Resonant medium - a permittivity resonance is part of the medium a wave crosses and travels in*,
the last case of `test_ray_state_update.cpp`, on `bf::walk` with the one-thread guard: the lambda
`walk` builds a sheet in air, a traced solid, or a sheet in the middle of 100 mm of a host;
tables A the matched sheet and its solid, B a medium with a pole against its twin without one, C
both ports together, D a sheet against its traced solid with the plain twins as the tracer's
residual. *ray_progress - a resonant solid is its medium at every frequency of a band*, the last
of `test_ray_progress.cpp`: one ray, a band traced at its first frequency, the TE matrix of
`ε + pole(f)` written in the case. *Resonance - the public index and in-medium gain carry the
permittivity pole*, the last of `test_ray_mesh_interact.cpp`. `diag_sheet_composite`,
`diag_cm_stack`, `diag_sc_stack` and `qd_slab_ref_lumped` carry a resonance as part of the
permittivity, and `diag_res` writes the pole from the column definition.

Instruments from that analysis that are not in the suite, kept in `build_linux/scratch/i46` with
a `NOTES.md` of every measured figure. *The switch copy*, `mkwi.py`: counters per site keyed by
which medium carries a resonance, and switches for readings of the pole, among them `32`, the
verdict of a mass-law medium taken on its permittivity without the pole. *The walker with the
resonance columns and `fRef` per material*, `probe2.cpp` with `runq2.sh`, one wall per line:
`tag itype theta maxdepth start sink nm (a ei thk tf m fRef resF resQ resS)*nm nf (x mat flip)*nf`;
`genW.py` and `anW.py` build and read the window scan of issue 1. *The band probe*, `rp.cpp`: one
ray through `ray_progress` with a band of frequencies, sums per frequency. *The formula model*,
`model.py`, of one layer between two half-spaces of air under three readings of a pole. Traps:
`ray_progress` throws on a `thin_slab_threshold` outside [0, 1], so a followed series is the
walker's; a pole with `resQ` 1e300 still has an imaginary part; a walker sum of 1.005 on a sheet
rarer than its host at 80° is the depth limit, 0.99999 at a depth of 120.

**A geometry the suite has no row for** can be measured before a fixture is written: a scratch
file outside the repository that includes `tests/catch2_tests/test_ray_state_update.cpp` and adds
one case, compiled with the flags of `build_linux/CMakeFiles/test_bin.dir/flags.make` and linked
from `link.txt` in place of that file's object. With a patched library object in front of
`libquadriga.a` the same probe runs on a what-if. **Which existing rows sit in a branch** is shown
by a marker build of the whole suite run with `--durations yes`: the duration line of a case
follows its output, so every marker line is attributed to its case. Counters are attributed
without reading the output by a Catch2 event listener compiled into the relinked binary
(`CATCH_REGISTER_LISTENER`, `testCaseEnded`) that calls a dump function of the marked copy,
which prints and clears a map from a key string to a count; a key that spells out the site,
the face, the channel, the columns present and which media are lossy gives the table of a
branch in one run. A marker printed inline lands in front of the table row it belongs to when
the case runs on one thread, which names the rows.

**Whether a fixture enters its branch** is shown with a marker build: a copy of
`ray_mesh_interact.cpp` outside the repository with a counter at the branch, printed at exit,
compiled and linked in front of `libquadriga.a` as described below, and the case run alone. The
output of such a build is the normal build's, line for line.

**Rows that move without a library change.** Editing a test file can move existing rows of
`test.txt` in their residual columns: 14 to 23 rows by at most 3e-17, reproducibly, with `src/`
untouched, in *sheet inside a host, the host's series on the leg behind it*, *lined on both
faces, the host's series*, *host layer bounded by a sheet* and *layered wall, the host's series
at the glued exit*. The fixtures' shared reference helpers are compiled differently when the
translation unit changes; that reading is inferred and not proven. *Sheet reached from inside
a traced solid* has moved the same way, 24 rows in the four tables at 2.8e-16. Appending the nine
branch cases moved nothing; adding one table to one of them moved 27 rows in those five tables at
2.8e-16. A suite run while another
build is going has moved about 27 rows at 1e-16. Appending the mass-cosine case and editing the laminate
case moved no existing row; appending the closed-plate case moved 27 rows in those five tables at
2.8e-16; appending the held-lining case moved none; appending the two stack-geometry cases moved
27 rows at 2.8e-16; appending the two flat-branch cases moved none. Diff such rows apart from what a source change
moves, and regenerate `test.txt` after a fixture change before the next source change.

**A candidate change can be measured before the source is touched.** Copy
`src/ray_mesh_interact.cpp` out of the repository, patch the copy, compile it with the flags of
`build_linux/CMakeFiles/quadriga_static.dir/flags.make` and relink `test_bin` from
`build_linux/CMakeFiles/test_bin.dir/link.txt` with the patched object in front of
`libquadriga.a`. The whole suite then runs on the what-if and is diffed like any other run.
Options that are to be compared can be put behind an environment variable read once in the
copy, so one object serves every variant and the probe and the suite are run once per value.
With the variable unset such a copy has to reproduce `test.txt` with no moved row, which is
worth checking first. A relinked binary has to be run from the repository root: eleven cases
open files under `tests/data` by a relative path and fail anywhere else. On a machine another
job is loading the suite takes ten times as long and nothing hangs. The fixtures of an
item can be run on the what-if before the source is touched, by relinking the rebuilt test
objects in front of it.

---

## 2. Ranked issues

Ranked by importance. The primary target of the library is EM: a wrong result that a material
without the mass column can reach ranks above any that needs one, and a change made for
acoustics must not move an EM row. Within each group: the size of a wrong result in a valid
geometry that is in use, then what is unmeasured, then conventions and cleanup. Section 6
gives the execution order.

| # | Issue | Measured size | Where |
| :-: | --- | --- | --- |
| 1 | A mass-law panel with a permittivity resonance: a window | a sheet passes +19 to +147 dB at 45° and 75° and its two ports read up to 5e14; its traced solid −32 to +5.6 dB where the panel without the pole passes −18 to −53 dB; sheet and solid 2.2 to 23 dB apart at normal incidence; at every frequency of a band. A first-principles analysis is owed before any design | the verdict of a mass-law medium, `qd_stack_resolve`, `Material::interact_with`, the mass-law convention |
| 2 | A layer without the mass column behind a held wave: a coherent lining on the far face of a rigid wall, the same layer as a traced solid, a stack with such a layer inside a rigid wall | 0.22 to 0.96 dB between the two sides for a dense lining, 0.07, 0.89, 0.62 and 1.73 dB at 10°, 30°, 60° and 75° for a porous one; the traced solid passes nothing from 60°; 3 to 5 dB between a stack and its solids | `qd_stack_resolve`, `Material::interact_with` |
| 3 | The `alpha` column is charged over a path that saturates: on a rigid mass-law sheet, and on a coherent sheet that tunnels | 2.2 dB at 0°, 400 dB from 10° on; 0.012 dB behind a lossy host; the same body as a traced solid 2.23 to 4.40 dB; a tunnelling sheet with any `alpha` passes nothing | `qd_mass_amp`, `qd_stack_resolve` |
| 4 | A mass-law lining behind a held wave whose two faces both lead towards the higher index keeps the Snell cosine | scalar 0.003 to 0.008 dB between the two sides | `qd_stack_resolve` |
| 5 | The interface gains of a bonded pair add in dB | zero while `att` is zero; otherwise the lighter body's level | `interface_gain` call sites |
| 6 | A lossy mass-law sheet and its traced solid take two `tf` rules, and the sheet's rule holds the one clamp left | no fixture; a scratch scan reads at most 0.99 on the sheet and 3.3 to 10.6 dB between sheet and solid, mostly the solid's bounces; 120 clamped calls in five cases, not sized | `qd_stack_resolve` |
| 7 | The undeviated bit is set at normal incidence for a mass-law exit without a refracted direction | a flag, no gain | `Material::interact_with`, `REPLACE_BY_VBS` |
| 8 | A deferred medium displaces the thickness code | +37.5 dB and +23.1 dB from one side, exact from the other; rare in practice; may be dropped | the buffer word, the nested and overlapping branches of the transmit pass |
| 9 | The material-model documentation is partly out of date | documentation only; out of scope for now | `html_docu/quadriga_lib_material_model.md` |

No item here sends a ray to a wrong place or drops a path; every one is a wrong level, phase
or flag. One creates energy in a geometry meant to be in use: issue 1, a rigid mass-law panel
that carries a resonance, which is what the resonance columns were introduced for in acoustics.
Apart from it, a wall without `tf` reads at most one on its two ports seen from a lossless
medium, as a sheet and as a traced solid, summed and followed, with a sheet or a stack inside
it, capped or not, with a resonance in any medium without the mass column, and a face with `tf`
conserves where it is open and where it is closed. No open item is known that a material without
the mass column reaches: every item here needs the mass column or the rigid mapping and is
acoustics.

Issue 1 is first: the largest wrong level, and in the use the column is meant for. It needs a
first-principles analysis and a literature review before any design. Issue 2 is the
largest of the rest in a geometry that is in use, a rigid wall with an absorber behind it; its
reference side, the absorber met first, is settled. It still has a traced solid that passes
nothing. Issue 3 is reached only by a material that carries the `alpha` column on a mass-law
or a tunnelling sheet, which the tables are to allow. Issue 4 is below 0.01 dB and is a
decision more than a repair. Issue 5 is a convention. Issue 6 has no fixture and needs a
mass-law material that is lossy and not rigid. Issue 7 is cleanup. Issue 8 is a large figure
and is ranked low on purpose: the geometry, an object embedded in a mass-law wall that is
bonded to another, is rare in practice, the fix needs the layout of the buffer word reopened,
and it may be dropped if that turns out not to be feasible once the others are done.

---

## 3. The issues

### 3.1 A mass-law panel with a permittivity resonance

**What it is.** In the acoustic mapping a pane of glass is a rigid body: `ε = (Z_air / Z)²`, of
the order of 1e-9, lossless, so its layer is closed from a few thousandths of a degree on (*A
closed mass-law layer*). Its transmission is the `tf` leak of its faces times the
mass term `m` with the plate rule and the interface gains; its reflection is the pad of a closed
face; the coincidence dip of a thin stiff panel is the `coi*` columns on the interface gain. The
resonance columns were introduced for such thin solids - glass sheets and windows - in acoustics.
The shipped documents say otherwise: the material-model document names Helmholtz, membrane and
micro-perforated absorbers for `res*`, the modelling guidelines under
`tools/acoustic_parameter_mapping` list resonant absorbers as not captured and say nothing of a
resonance on glass. No shipped table carries the columns, and no row of the suite builds a
mass-law material with a pole.

**What happens now.** A pole has a non-zero imaginary part at every frequency (§1), so a rigid
body with one is lossy for every verdict: its layer is never closed - `CLOSED` in the solver, the
total-reflection test of `Material::interact_with`, the tube test of `ray_mesh_interact`, the
closed test of `fresnel_r` - at any angle and any frequency, however small the pole. A sheet then
takes the open mass-law face, the pair rule on a Fresnel reflectance that is one to within the
pole, and its transmission is scaled by `(1 − R') / (1 − R0)` with `1 − R0` all but zero: the
quotient of issue 6, by many orders. A traced solid's faces become transformers with an all but
evanescent wave behind them, and its `tf` leak and most of its mass term go.

Measured on 6 mm of a rigid body of `ε` 8.96e-10 with `m` 20 (30 dB at normal incidence at 1 kHz, `fRef`
1.66e-4 GHz), `tf` 0.1 and 0.5, a pole at 1 kHz (0.874 GHz) with `resQ` 5 and `resS` 1e-9 to
1e-2, scalar, types 4 and 5, every path with the walker, at 500 Hz, 1 kHz and 2 kHz. The
permittivity is the glass row of `data/materials_ac.csv`, used as an example of a rigid body and
nothing more: that table and its copy under `tools/acoustic_parameter_mapping` are out of date,
every material in them is to be replaced, and none is a reference.

| | sheet | traced solid | both ports of the sheet |
| --- | --- | --- | --- |
| no pole, 1 kHz, `tf` 0.1, at 0°, 45°, 75° | −49.98, −46.99, −38.31 dB | −49.97, −46.97, −38.19 dB | 0.900 |
| `resS` 1e-9 to 1e-4, 45° and 75°, 1 kHz | +19.1 to +147.1 dB | −15.1 to −24.6 dB, whatever the pole and `tf` | 82 to 5.1e14 |
| `resS` 1e-6 and 1e-4, 45° and 75°, 500 Hz and 2 kHz | +54.2 to +122.3 dB | −32.4 to +5.6 dB | 2.6e5 to 1.7e12 |
| `resS` 1e-2, 45° and 75°, 1 kHz | −15.6 to +7.3 dB | −15.6 to −25.1 dB | 0.69 to 5.9 |
| `resS` 1e-9, normal incidence, 1 kHz | −47.67 dB | −49.86 dB | 0.900 |
| `resS` 1e-6, normal incidence, 2 kHz | −32.45 dB | −55.41 dB | 0.900 |

With `resS` 1e-9 away from `resF` the sheet keeps its value without the pole and the traced
solid does not: −15.0 dB at 45° and 500 Hz with type 4, +4.8 dB with type 5. Making the
resonance part of the medium moved these rows by 0.25 dB at most; the closed verdict, not the
index, is what breaks.

A candidate measured on a what-if, not a fix: the closed verdict taken on the permittivity
without the pole. The sheet returns to its value without the pole at every oblique angle,
whatever the pole: the resonance has no effect there. The traced solid reads −45.8 to −37.9 dB at
45° where the sheet reads −46.99, 1.2 to 9.1 dB apart. The reflection is the `tf` leak alone,
−0.46 dB for `tf` 0.1, because the pad reads only the reactance of the far side: no absorption
peak. Normal incidence is unchanged, so the panel steps between 0° and 0.002°.

**What has to be answered first.** A first-principles analysis and a literature review, before
any design:

1. *Which panels, and which effect.* What physical mechanism a "resonance" of a thin solid is:
   the mass-air-mass resonance of double glazing, commonly given as
   `f₀ = (1/2π)·√(ρ₀c²(m₁′ + m₂′) / (d·m₁′·m₂′))`; the fundamental bending modes of a finite pane;
   the resonance of a panel absorber or membrane in front of a cavity, commonly `f₀ ≈ 60 / √(m′d)`
   Hz; or the coincidence region, which `coi*` already holds. For each: which materials need it -
   single, laminated, double and triple glazing, gypsum board on studs, wood panels, doors - at
   which frequencies, and what it does: a dip in the transmission loss, a peak in the room-side
   absorption, a reflection phase, or several.
2. *Reported values and established theory.* The mass law and its field-incidence form; the
   single- and double-panel models of Sharp and of Cremer; the stiffness-, resonance-, mass- and
   coincidence-controlled regions; measured transmission loss and absorption of glazing (EN 12758,
   ISO 12354, ISO 10140, published tables for 4 mm, 6 mm and 4-16-4 units). Sources to consult:
   Fahy, *Sound and Structural Vibration*; Cremer, Heckl and Petersson, *Structure-Borne Sound*;
   Bies and Hansen, *Engineering Noise Control*; Kuttruff, *Room Acoustics*; Cox and D'Antonio,
   *Acoustic Absorbers and Diffusers*; Sharp (1978). The two formulas above are to be checked
   against them, and every number taken into the plan carries its source.
3. *What the library can express.* Per material: `a` to `d` and `e` to `h` (ε and μ), `att` and
   `attB` (a level per medium entered), `coiF`, `coiQ`, `coiA` (a Lorentzian on that level), `m`
   and `fRef` (the mass term with its angle shape and the plate rule), `tf` (the leak of a closed
   face, the transformer of an open one), `alpha`, `res*` (a pole in ε). For a rigid body today
   the transmission is the leak times the mass term times the gains and the reflection the pad,
   which reads `tf` and the reactance of the far side. The materials of the analysis are built
   from the findings of 1 and 2 - density, thickness, bending stiffness, loss factor, cavity -
   and not taken from a shipped table. Which mechanism of 1 a pole in ε can carry
   at all, which belongs in the mass term - a panel impedance with a resonance,
   `X₀ → jωm′(1 − f₀²/f²) + r` - which in the gains, and which nowhere.
4. *What is physically correct.* A thin panel between two fluids is one impedance `Z_p`: it
   transmits `|2ρc / (Z_p cos θ + 2ρc)|²`, reflects `|Z_p cos θ / (Z_p cos θ + 2ρc)|²` and absorbs
   the rest; the mass law is its mass-controlled limit and a resonance its stiffness or cavity
   term. Whether the library's split - leak, mass term, pad, gains - reproduces that limit, what
   it makes of a resonance, and what a traced solid of the same panel has to do so that sheet and
   solid are one wall.
5. *How to implement it in the current code.* Where each candidate lives: the four verdict sites
   above, `Material::closed_face`, `Material::medium_gain` and `mass_TL` with the plate rule
   (`MASS_RUN`, `qd_mass_amp`), `interface_gain`. Candidates to measure, none chosen: the pole
   ignored on a material with the mass column, with a panel's resonance expressed through `coi*`
   or a documented rule; the pole as a resonance in the panel impedance of the mass term, shared
   by sheet and solid; the closed verdict taken on the permittivity without the pole, with the pad
   reading the resistance of the far side as well as its reactance.

**Constraints.** Decided on the material columns and the media, never on a value; nothing reaches
the state words or the geometry; a material without the mass column does not move; a wall
without `tf` reads at most one; no clamp; a sheet and its traced solid are one wall; one
expression across angle, with no step at the critical angle of a rigid face; a band sees the pole
per frequency. Any change has to be measured on issue 6 as well, which shares the quotient.

**Change.** None yet: the analysis and a decision proposal per question come first.

**Assertions.** A window with the column as a sheet and as a traced solid, at and away from
`resF`: both ports together at most one, sheet on solid at the relaunch bound, the chosen model
against its closed form, no step in angle; the window without the column and every row without
the mass column where it is.

### 3.2 A layer without the mass column behind a held wave

A wave inside a rigid mass-law wall is a leak at the incidence angle (§1). What it carries
into the next medium is the transverse wavenumber of the wave in air. A coherent layer behind
such a wall therefore has the wave air would have given it, at the Snell cosine of air | layer.
The code gives it neither that as a sheet nor that as a traced solid.

*As a sheet*, a lining on the far face or a sheet inside the wall: the coherent route of
the solver enters it at the Snell cosine of wall | sheet, formed from the wall's impedance
mapping, which is one to eight digits. The sheet is crossed along its normal at every angle.

*As a traced solid* behind the wall: the hold of §1 hands back the geometry of no bend towards
the higher index, so the leg in the layer runs at the incidence angle, without the bent flag,
and meets its own far face there, past that face's critical angle from `asin(1/sqrt ε)` on.

*In an ordered stack inside a rigid wall* with a rigid layer: the solver's segments, at the
cosines of the wall's mapping.

**Size.** Measured through the intersector, scalar, W wood 100 mm `eps 2.1609e-8` `tf` 0.2
`m` 20, the layer 10 mm, series summed, levels without the mass term; `−` is a ray that passed
nothing. Table H of the held-lining case holds the lining rows.

| layer | form | 10° | 30° | 60° | 75° |
| --- | --- | --- | --- | --- | --- |
| `eps 2.50 − 0.80j` | lining, wall met first | −13.781 | −13.781 | −13.784 | −13.785 |
| | lining met first | −13.561 | −13.196 | −13.317 | −14.743 |
| | traced solid behind the wall | −14.413 | −16.839 | − | − |
| `eps 0.31 − 0.20j` | lining, wall met first | −16.254 | −16.256 | −16.263 | −16.267 |
| | lining met first | −16.322 | −15.369 | −15.643 | −17.996 |
| | traced solid behind the wall | −16.486 | −16.700 | −15.376 | −14.099 |

The lining met first is a glued entry under a closed outer pair, entered from air at the
incidence angle with its series, on the pad of the wall face, and is the side that describes
the wave. The two sides of the dense lining differ by 0.22, 0.59, 0.47 and 0.96 dB, of the
porous one by 0.07, 0.89, 0.62 and 1.73 dB; a lossless `eps 2.50` lining was 0.32, 0.88, 0.85
and 0.78 dB apart when it was last measured. A sheet inside the wall,
W 50 | sheet | W 50, reads −24.20 dB to −24.22 dB at every angle against −29.25, −32.89, −31.07
and −27.49 dB for the three solids. A stack W 50 | `eps 2.50 − 0.80j` and K | W 50 reads −28.25
to −32.82 dB against −33.30 to −37.17 dB for its solids. Nothing here passes more than it
received.

The face between the wall and a coherent lining carries the wall's `tf` and is a transformer
(§1), formed on an admittance ratio in which the wall's side is an impedance mapping; whether a
face of a mass-law half-space should take the rule of a wave face is part of this item. The
lossless-lining figures, the sheet inside the wall and the stack rows above were measured
before the face of a coherent sheet became a transformer and have not been measured again.

**What is not yet known.** What the correct statement is for each form, and whether one rule
serves all three: the invariant of air, `sin²θ`, as the transverse term of a sheet or a stack
behind a held wave, with the wall's leak on the near face; and for a traced solid the
direction air | layer would have written, which makes that leg a bent one, with the bent flag
and the recovery, where the hold of §1 now says it is not. That touches a settled statement,
the direction of a leak through a mass-law body, on the faces that lead into a medium without
the mass column; the faces between mass-law bodies and into air are not in question. Whether
the reverse, a coherent layer in front of a rigid wall, which is right today, stays bit for
bit. EM has no part in it: it needs a wall with `m > 0`.

**Change.** Analysis first, with a marker for the cosines of all three forms and the reverse
direction as the reference. Then a decision per form, a what-if build, and one rule.

**Assertions.** A coherent lining on a rigid wall on one transmission from its two sides, to
the flat first face and the flux splitting of §5; a traced coherent layer behind a rigid wall
passing what the same layer in front of it passes, at every angle; every row whose wall carries
no mass column bit-identical.

### 3.3 The `alpha` column of a rigid mass-law sheet is charged over a path that saturates

The mass-law route of a sheet charges the `alpha` column and the conductivity loss over the path
`thk / cos`, with the cosine the in-sheet Snell cosine (§1). The index of a mass-law material is
an impedance mapping and not a propagation index, and for a rigid sheet that cosine is
evanescent: its real part is zero and the path is the saturation cap.

**Size.** Measured on a rigid sheet 22 mm with `tf` 0.2, `m` 20 and `alpha` 100 dB/m, 2.2 dB over
its normal thickness, as the sheet without `alpha` over the sheet with it, both tracers alike:

| the sheet | 0° | 10° | 30° | 60° |
| --- | --- | --- | --- | --- |
| in air | 2.20 dB | 400.00 dB | 400.00 dB | 400.00 dB |
| a lining on `eps 2.50` | 2.20 dB | 400.00 dB | 400.00 dB | 400.00 dB |
| a lining on `eps 2.50 − 0.80j` | 2.20 dB | 0.012 dB | 0.004 dB | 0.002 dB |
| the same body as a traced solid in air | 2.20 dB | 2.23 dB | 2.54 dB | 4.40 dB |

The traced solid holds the wave undeviated and charges the path `thk / cos` at the incidence
angle.

**What is not yet known.** Which path a lumped plate owes: the held one, `thk` over the cosine
its mass term takes, is what the traced solid has and what "no wave inside" says, and would make
sheet and solid one body. A mass-law sheet with a permittivity above one, whose Snell cosine is
real, is charged a refracted path today, and its rows would move; the stack case with a
mass-law layer has such rows. Whether the conductivity of a mass-law material is to follow the
same path.

**Where the solver stands on it today.** Two paths exist for one lumped plate. A mass-law
layer of the solver, `qd_mass_amp`, charges `thk` over the in-layer Snell cosine under the
saturation cap. The chain of held crossings charges `thk` over the incidence cosine, which is
the traced solid's path. No row carries the column on either.

**The same path on a coherent sheet that tunnels.** The `alpha` column of a coherent layer is
charged over `thk / |Re c|` as well. A lossless sheet rarer than the transverse invariant has
`Re c = 0`: it tunnels, and with any `alpha` it is opaque. Measured with a scratch probe, not
in a fixture: 1 mm of the lossless `eps 0.31` with `alpha` 100 dB/m, 0.1 dB over its thickness,
passes nothing past its critical angle, in air, inside a host and as a lining. A decision on
the path of a plate should say what a tunnelling layer owes; the normal thickness is the
candidate there too. Behind a lossy host the path cosine is the one of the wave's real
transverse invariant; no row moves on it.

**Change.** A convention decision first, then one path in `qd_mass_amp`, in the held chain and in the layer phase. No
existing row carries `alpha` on a rigid mass-law sheet; no fixture exists.

**Assertions.** A rigid mass-law sheet with `alpha` on the traced solid of its thickness, in air
and as a lining, 0° to 60°; every row without the column bit-identical.

### 3.4 A mass-law lining behind a held wave whose two faces both lead towards the higher index keeps the Snell cosine

The rule for a sheet behind a held wave (§1) is taken where at least one face of the run
wall | sheet | far leads towards the lower index. Where none does - a wood lining on a concrete
wall, both steps upward - the sheet stays on the composite: its faces are evaluated at the
Snell cosine of the two mappings, where the solid twin and the other side evaluate them at the
incidence cosine.

**Size.** Measured through the intersector, levels without the mass term, the wall met first
against the lining met first, in table H of the held-lining case:

| wall, lining | 10° | 30° | 60° | 75° |
| --- | --- | --- | --- | --- |
| K, W, scalar and type 1 VV | +0.003 dB | +0.003 dB | +0.006 dB | +0.008 dB |
| K, W, type 1, HH | +0.002 dB | +0.002 dB | −0.005 dB | −0.019 dB |

Types 1 and 2 on a material with the mass column are not EM in any use; the scalar figures are
what counts. A mass-law lining that is not rigid on a rigid wall meets the gate at its far face
and is on its solid twin, asserted at 1e-3 dB. A sheet or a stack inside a wall is not
affected: one of its faces always leads down.

**What is known.** With the rule taken with no direction gate at all, measured on a what-if
while the gate was narrower than it is now, the rows above were on their solid twin and the two
sides one to the flat first face. It moved rows of the closed-plate, leak-direction,
refracting-tracer, bonded-mass and plate-depth cases by up to 0.007 dB; which of those the
present gate already moved was not measured again. The mass paid does not move.

**Change.** A decision: drop the direction condition from the held gate of the solver, which
is one condition, and restate the rows that move; or leave it and move this entry to §5. The
size is below 0.01 dB in scalar mode either way.

**Assertions**, if taken: the K, W row of table H on its solid twin at 1e-3 dB in both orders,
scalar; the rows that move listed with their size.

### 3.5 The interface gains of a bonded pair add in dB

`att` and the coincidence term are owed once per medium entered (§1), so a panel bonded to a wall
pays both in full, in every one of the three pairings. For a level carried in `att` that is the
defect the mass terms had: the bonded plate is one partition and has one level. Nothing combines
them today, and there is no rule yet for how they should combine. `att` is a level in dB with no
mass behind it, so it has no impedance to put in series, and a coincidence dip belongs to one
panel's bending and is not obviously inherited by the plate.

**Size.** Unmeasured, and zero for a table that leaves `att` at zero and carries the level in the
mass term and `tf`, which is how the acoustic rows in use are built. For a table that carries the
level in `att` it is the whole of the lighter body's level.

**Change.** A convention decision first: whether a bonded pair owes the larger `att`, the mass-law
combination of the two read as impedances, or both in full as now. No source change until that is
decided. Whatever it is, it has to hold on the reflect port as well, where the echo out of a
coherent sheet pays that sheet's gain (§1).

### 3.6 A lossy mass-law sheet and its traced solid take two `tf` rules

A mass-law sheet keeps the pair rule on the reflectance at both faces, with the transmission
scaled by `(1 − R') / (1 − R0)` where the face has a forward port (§1). A traced solid of the
same material takes the transformer at every open face, because a bare boundary has no gate on
the mass column: a gate would have kept 250 rows where they were and a 10.5 dB disagreement
between a sheet and its solids on types 1 and 2 with them. For a lossless mass-law material
the two rules are one, and a rigid one is closed and takes neither. A mass-law material that
is lossy and not rigid - a panel with a conductivity or a resonance and a mass column - is a
sheet on one rule and a solid on the other, and the quotient next to a lossy medium is the
term that created energy at a bare boundary before it became a transformer. A resonance on a
rigid mass-law panel reaches the same quotient from the closed side, by many orders more; that
is issue 1, and a change to either has to be measured on the other.

**Size.** Not measured in a fixture. No row of the suite builds such a pair; the mass-law rows
that exist are lossless or rigid. A scratch scan with the mass argument at nothing, a lossy
mass-law sheet with `tf` on a rigid wall, 1 to 50 mm and 5° to 85°: the sheet's two ports read
at most 0.99 together, and it parts from its traced solid by 3.3 to 10.6 dB on the transmit
and 13 to 35 dB on the reflect port. With the mass term at nothing that is mostly the bounces
a traced mass-law body returns and a sheet does not (§1), so it does not yet size the two
rules. At a closed bare boundary a mass-law medium is given a real immittance (§5), which is
a third statement about such a material.

**The one clamp left stands here.** The pair rule works on a reflectance and clamps it into
`[0, 1]`. At an open mass-law face next to a lossy medium the Fresnel reflectance is above one
in TM, and in scalar mode with a permeability; the clamped face then has no forward port left
to scale, and such a sheet without `tf` passes nothing there. A marker build counts it in five
cases of the suite, 120 calls, all types 1 and 2 on a material with the mass column but two.
Not sized as a level. The transformer at the open faces of a mass-law sheet would remove the
reflectance, the clamp and the quotient together.

**Change.** A measurement first, with the walker of the bare-tf case: both ports of a lossy
mass-law layer with `tf` as a sheet and as a solid, in air and on a wall. Then a decision:
the transformer at the open faces of a mass-law sheet, which removes a rule from two solvers,
or the difference recorded with its size.

**Assertions.** A lossy mass-law layer with `tf` as a sheet against its traced solid at the
relaunch bound, both ports together at most one in scalar mode, or the convention stated;
every lossless and every rigid mass-law row where it is.

### 3.7 The undeviated bit is set at normal incidence for a mass-law exit without a refracted direction

Bit 5 of the resolved type used to mean total reflection. Under the rule of §1 it also marks, on the
transmit pass, a wave held undeviated inside a mass-law medium crossing towards the lower index, and that includes a
mass-law solid with permittivity above 1 exited with no refracted direction, at any angle. With a
refracted direction supplied the bit is clear, except at exactly normal incidence, where the traced
and geometric cosines coincide and the function treats the wave as undeviated. So the resolved type
of one physical crossing reads 1 at 0.1° and 33 at 0°.

The gain is the same either way, and nothing outside `ray_mesh_interact.cpp` was found reading the
bit. It is pinned by *mass-law material is excluded from slab resolution*, which now expects the bit.
Permittivity above 1 together with `m > 0` is an unusual material: softer than air, with a mass law.

**Change.** Decide whether the bit should keep one meaning. If it should, the undeviated verdict
needs its own way out of `interact_with` instead of sharing the total-reflection one.

**With the refracting tracer, a mass-law solid the wave is refracted into.** `ray_mesh_interact`
holds every wave in a mass-law medium undeviated, and with types 2 and 5 its direction is the
segment. Measured on `air | 100 mm eps 1.50 with m 20 | air` at 30°: the leg carries the bent
flag, `ray_state_update` writes the exit direction at 30° and the ordinary crossing with type
code 1, the segment leaves at 24.09°, and the chain is 0.458 dB below the undeviated tracer's,
the mass term being formed at the in-medium cosine. Recorded in the refracting-tracer case. The
segment is not moved onto the written direction either: that is done out of a lossy medium only
(§1), and this one is lossless. Through `ray_progress` the ray leaves at 24.0948° with 30.0000°
in `path_dir`, recorded in the trajectory case. A
mass-law material with a permittivity above 1 is not a material either domain uses.

### 3.8 A deferred medium displaces the thickness code

The buffer word holds either the thickness code under its flag bit or a deferred medium, the
index of an object that nested or overlapping geometry makes the ray enter while it is in
another (§1). Where an object is met on a leg that carries a code, the index replaces the code,
the flag goes with it, and the leg pays its own mass term in full at its far side: the bonded
bodies in front of it are forgotten.

**Size.** Measured in the plate-depth case, tables A and S, at normal incidence and far above
the knee, with every body at `X₀` 100 per 100 mm:

| stack | from the side that meets the object after the bonded face | from the other side |
| --- | --- | --- |
| solid 100 \| wall 300 with an object nested in its middle | +37.50 dB: the wall pays 49.54 dB where the plate takes 12.04 dB | −0.0005 dB |
| lining on a solid 200 that overlaps a second solid over its last 100 | +23.10 dB: the first solid pays 46.02 dB where the plate takes 23.13 dB | −0.0002 dB |

Towards grazing it falls, to +25.8 dB and +11.6 dB at 75°; near the knee it is +3.5 dB to
−1.0 dB. From the other side the object is met before any code exists and the result is the
plate. It is the dB sum of the two bodies, the figure the bonded pair had before it was one
plate, and it is direction-dependent. A plastered wall with a beam or a duct embedded in the
masonry is the picture; only rays that cross the embedded object are affected.

**Why the state cannot hold both.** The code is 15 bits, the index 14, the word 16, and the two
other words have no spare bit. The index cannot simply be left out: without it the exit face of
a nested object, a face of another material met from inside, is the hit that today means the
ray is recorded in the wrong medium, which is corrected with the fix-flag (§5).

**The direction to look in, not yet analysed and not measured.** A code of 14 bits frees one
bit under the mass flag. That bit cannot hold the index, but it could stand in for it: "an
object is pending on this leg and its index is not stored". The exit face of a nested object
would then be known as a face of another material met while the bit is set, which also tells
it apart from the correction above. What has to be answered before anything is written:

- whether every branch that reads the deferred medium can decide on the bit and the face
  material alone: the exit of a nested object, a co-located exit and entry of two nested
  objects, the corner cases, and the reflect pass, which must not reflect off a nested
  object's exit;
- overlapping bodies, where the current body ends first and the ray has to continue in the
  pending one, whose material the bit does not name. The second row of the table is this case
  and may stay as it is;
- what a 14-bit code costs. The step doubles to 0.088 %, 0.0077 dB on a mass term at `m = 20`
  and 0.0115 dB at `m = 30` by the formula; the code is quantized again at every bonded face,
  the worst asserted row is 3.0e-3 dB today against a bound of 0.01 dB, and every existing row
  that reads a code moves in its last digits. The range must not be shortened instead: an
  equivalent thickness of a heavy body in a light material reaches tens of meters (§5);
- whether the bit has a better use. None is known: it exists only on a leg entered across a
  bonded mass-law body, and issue 5, the one open item that lives there, is stateless at pair
  depth and needs a level, not a bit, beyond it.

**Change.** Analysis and a what-if build first (§1, diagnostic infrastructure), with the two
recorded rows of the plate-depth case as the fixture and every moved row counted. Low priority;
to be dropped, and the limitation left in §5 with its measured size, if the branches cannot all
be decided on one bit or if the cost in moved rows is not worth a rare geometry.

**Assertion, if taken.** The nested row on the plate from both sides to the bound the code
then has, with every other row of the plate-depth case where it is.

### 3.9 The material-model documentation is partly out of date

`html_docu/quadriga_lib_material_model.md` was brought up to date only for what the bonded-mass work
touched: the mass-law term and the plate rule (its section 2.3), the two ports of a face seen from
inside a mass-law medium (3.4), bonded linings (6.3) and the buffer word's flag and code (7.2). The
rest of it is **out of scope for now** and is to be left alone until it is taken as its own item,
after the last change that alters what it has to say. Known to be stale there:

- the `tf` owner rule: it still says the single-owner `apply_tf` is retained inside the thin-slab
  factor, where every face is owed by the pair of media that form it; and it does not know that
  `tf` is a transformer at every face with a wave on both sides, the face of a coherent sheet
  and an open bare boundary alike, per channel, while a closed face, a mass-law face and a
  closed far face keep the pair rule on the reflectance;
- the energy partition table of its section 3.4, which predates the per-channel normal-flux
  convention of §1;
- no statement of what a sheet does under total reflection of its outer pair, of a mass-law layer
  in a stack, or of a sheet under a closed outer pair;
- its section 7.9 on stacked slabs, which predates the solver;
- its account of the previous-medium word, which predates the sheet left in it by a glued exit
  into air, air beyond a glued exit behind a sheet, the recovery behind a sheet, the inside
  flag, and the real-invariant flag the same bit carries where the word names a solid;
- its account of the buffer word, which predates the thickness code of the host in front of a
  sheet, the equivalent thickness that code carries for bodies of one slope, and its statement
  of the plate, which stops at two bodies;
- a mass-law sheet of its mass-law host's permittivity, which it does not mention;
- the masks and the public limit on the number of materials: it still gives a 15-bit index and
  32767, where the index is 14 bits and the limit 16383;
- where the layer of a sheet lies about its face, and what its reflect port carries for it and
  for `att`: the document knows neither;
- the cosine of a mass term behind a refracting layer, which it does not state;
- a sheet rarer than the transverse invariant: that it tunnels, with or without `tf`, and that a
  sheet without the mass column is a wave medium, so a rigid sheet has to carry `m`. The
  modelling guidelines under `tools/acoustic_parameter_mapping` mesh a rigid reflector as a box
  and say nothing of a rigid sheet either.

- the permittivity resonance: its sections 2.1, 2.3 and 7.7 say the pole is applied to the
  interface permittivity only and left out of the propagation phase, the bulk loss and
  `refractive_index`; all of them read it now (§1), and `Material::eta_resonance`, which it
  names, is gone. Its section 6 assigns `res*` to absorbers; issue 1 has to say what the columns
  are on a panel. The modelling guidelines and the README under `tools/acoustic_parameter_mapping`
  list resonant absorbers as not captured.
- `src/quadriga_lib_material_helpers.hpp` is included by no file. It holds a second
  `eta_resonance` with the old statement, a second `medium_loss_dB` and the old `tf_apply`, and
  can go.

The API markdown `html_docu/quadriga_lib_api_cpp.md` is generated from the doc blocks in the
sources and was not regenerated after the `ray_state_update` block gained the buffer code and
the bit table of the three words, or after the limits in `ray_progress` and `ray_commit` moved.
The doc block of `ray_state_update` itself does not yet describe the real-invariant flag; the
decode block in the source does. It does describe the followed mark and the two verdict
arguments, and the API markdown was not regenerated for those either. The material-model
document knows neither the mark, nor the stack code beside it, nor the cap per side of a
sheet inside a host, nor that a verdict is taken once per body and by the base
frequency of a run.

**Change.** Bring the document in line with §1, once.

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
mode takes it through `ct_near_exact` and writes all five of its incidence-side terms in it. Worth
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
telescope correctly on their own. Below about 40° the error was almost purely phase; at 20° the two
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

**The walk-off traversal factor.** *Excess-phase walk-off convention* printed its table and asserted
only `isfinite` and `gain >= 0`, so with the tables off it was a no-op and a refactor could have moved
the quantity with the suite staying green. It now asserts the applied factor against
`diag_medium_walk`, called with the fixture's own values so the degeneracies are arguments rather than
part of the expression: phase to 0.01°, measuring 6e-15°, and magnitude to 1e-12, measuring exactly
zero. A one-sided margin asserts that the raw-path convention is excluded by more than 1° on the five
rows past normal incidence, where the separation runs 3.157° to 26.174°. The isolation the case rests
on is that `REPLACE_BY_VBS` fast-returns without writing a coefficient when `use_vbs` is false and no
buffer is set, and that `eps = 1.5` exceeds any reachable `rho`, which is bounded by 1, so the
correction-mode series declines on every row. It supplies no `path_dir`, so it pins the form of a
wave that was not bent and nothing else; a bent wave never takes it.

**Permeability on the sheet, glued and single-layer stack paths.** Measured, and correct as shipped;
no source change was made. Every site that reads μ reads it in one of two forms and never a third.
Through the product `ε·μ`: the index in every propagation factor, every Snell ratio and refraction
root, the total-reflection test, `medium_gain`'s loss rate, `qd_refract`'s `snell`,
`EXACT_IN_SLAB_COSINE`'s `eps_r`, the displaced-host cosine and index, and `SCALE_BY_MEDIUM`, which
never sees μ on its own. Through the quotient `ε/μ`: the admittance in every Fresnel amplitude, the
core's tilted admittances, and `flux_amp`'s `Re(z c)` and `Re(z̄ c)`. So there is one magnetic degree
of freedom, that `n` and `z` stop being the same number, and a single permittivity cannot reach it.

`diag_sheet_composite` and `diag_sheet_composite_cx` were generalized by exactly three substitutions
and reproduce `diag_cm_stack` at `μ ≠ 1` to 6.75e-16; the library reproduces them to 1.33e-15. The
two magnetic-only statements are both asserted: an `ε↔μ` swap on a sheet in air exchanges the TE and
TM ports exactly, at 5.55e-16, while leaving the power-averaged gain invariant, so the statement is
invisible to `gain` and to scalar mode and can only be made per channel; and an impedance-matched
dense sheet, `ε = μ = 4`, is invisible at normal incidence except for its phase, with the reflect
port identically zero, and reflects with no polarization dependence at all at oblique incidence.

One assertion that was planned for this work is withdrawn rather than deferred. A magnetic sheet
cannot split `fx_te` from `fx_tm`: `fx` is a function of the near and far media only, so the sheet
never enters it, and the split requires `Im(z)·Im(c) ≠ 0` in the same medium, which a real incidence
cosine and a lossless far medium forecloses and which a lossy dielectric already produces.

**Permeability on the traced solid and in a multi-layer stack.** Measured, and correct as shipped; no
source change was made. `diag_slab_p_pol`, `diag_traced_ct` and `diag_ct_exact` took the same
substitutions the sheet builders took, and a traced `air | slab | air` chain through a magnetic solid
then sits on the generalized analytic slab at 0.000000 dB over three solids and five angles, with the
traced cosine matching the map the recovery inverts to 1.11e-16.

Two properties are asserted that a dielectric palette cannot reach. The telescoping of the entry and
exit obliquity factors does not read μ at all — the pair cancels as a ratio of one and the same real
number whatever the slab admittance is — so what a magnetic run checks is that both crossings still
read the same `z_s` and the same `c_s`, which is what the recovered cosine exists to supply. And an
`ε↔μ` swap leaves the traced path bit-identical, at 0.000e+00 on every row, because `snell`, the
refraction root, the total-reflection test, `eps_r`, `medium_gain`'s rate and `SCALE_BY_MEDIUM`'s
index are all `ε·μ` products; it moves only the two interface channels, which it exchanges, at
4.44e-16. The chain separates by up to 0.93 dB across that swap while its traced cosine does not move
at all, which is the statement in one line. A matched solid, `ε = μ = 2`, keeps its two channels equal
at every angle and is fully transparent at normal incidence.

An ordered stack with one magnetic layer reproduces `diag_cm_stack` to 5.66e-16 on both ports and in
both orders, with the transmit port order-invariant and the reflect port separating by 0.0153 at
normal incidence, so reciprocity survives layers whose admittance and index are independent.

**Permeability on `ray_mesh_interact`'s own sheet composite.** Measured, and correct as shipped; no
source change was made. `qd_slab_ref` and `qd_slab_ref_R` took the three substitutions — the
admittance `sqrt(ε/μ)`, the Snell ratio as a ratio of `ε·μ` products, the index `sqrt(ε·μ)` — and a
magnetic sheet through `ray_mesh_interact` then sits on the generalized analytic slab at 0.000000 dB
on the transmit port and at 1.6e-15 per channel on both ports. The case now measures both ports on
every row, per channel and labelled, because the min/max channel ratio it used before is blind to
which channel is which. An `ε↔μ` swap exchanges VV and HH at 1.4e-15 on both ports, which is a
statement about this function's polarization basis in a real incidence plane.

The two public functions' sheet call sites differ in more than the material lookup and the basis: the
incidence cosine comes from the face normal here and from `fbs_angle` in the redo, the displacement
budget is the leg just travelled here and a two-sided budget there, and total reflection is handled
by the ray tube here and by a re-derived verdict there. None of the three reads μ and none is engaged
by a sheet in air, so the magnetic row measures the lookup and the basis and nothing else.

**The lumped columns in produce mode.** `att`, `tf` and the resonance each have a fixture, on
the isolated, glued and embedded sheet paths through `ray_state_update` and on an isolated sheet
through `ray_mesh_interact`. All are exact against their references; the conventions they pin are in
§1 and the measurements in the table there.

Three statements stand without a reference. A lossless sheet conserves energy per channel at any
`tf`, at 8.9e-16. The end points `tf = ±1` each close one port exactly. And a sheet whose resonance
takes its permittivity to exactly 1 is a sheet of air: no reflection and a transmission of one with
no phase, at 7.8e-16. That statement replaced one this entry first pinned, a pure phase at the
index without the pole, when the resonance became part of the medium (below).

**`tf` is owed by the pair, at every site.** A boundary between two `tf`-bearing media is owed by both,
in produce mode, in the core, in `fresnel_r` and in `interact_with`; the two rules and which
face takes which are in §1. The figures of this entry were measured while every site took the
pair rule on the reflectance; every face with a wave on both sides is a transformer since
(below), and the limit at the end of the entry holds for a lossless and for a lossy wall. Before, a sheet composite read only the sheet's `tf`, so a wall's column vanished when
the wall was coated: 3.01 dB at `tf = −0.5`. The coated wall now moves by −1.63 dB there and by
−0.06 dB at `tf = +0.5`, which is not what the column is worth on the bare wall and should not be,
since the wall's boundary is sheet | wall and a forced far face adds a series against the sheet's
front face. The bare-wall value is recovered exactly, +0.035719 and −3.010300 dB, by a sheet of the
host's own permittivity. Every row whose neighbour carries no `tf` is bit-identical across the
change. Asserted at 1e-9 over nine `tf` pairs, entry and exit, both ports, scalar and EM, measuring
1.2e-15.

That limit could not be asserted per channel in EM at oblique incidence while the bare boundary
rescaled by the TE/TM mean. It is per channel there now and the limit is asserted complex, in
scalar mode and per channel, at every angle.

**`tf` in a stack.** An ordered stack with a `tf`-bearing layer resolves in the scattering form of §1
instead of killing the ray, and a bounding sheet with `tf` is a layer instead of a half-space, so a
`tf` of 1e-9 on a wallpaper moves the host's series by 1e-10 where it used to move it by 1.56 dB.
Measured against the scattering reference at 1.3e-15, both orders, both ports, scalar and EM; against
produce mode with no reference, two 6 mm sheets of one permittivity and one `tf > 0` against one
12 mm sheet, at 4.7e-16; and by three statements that need no reference, energy per channel of a
lossless stack at 1.1e-15, the transmit port under layer order at 1.0e-15, and the two end points
exactly. A stack carrying neither `tf` nor `m` takes the matrix and every residual of it stayed
where it was.

**A mass-law layer in a stack.** Resolved by coherent segments, as in §1, instead of killing the ray.
Chosen over an incoherent cascade of single-sheet composites because the cascade puts a fictitious
host gap between co-located sheets, replacing the true shared boundary by two false ones, and shows
only the first sheet on the reflect port; the two differ by 0.09 dB to 0.76 dB on the transmit port
and by 0.0015 against 0.159 on a reflect port. There is no exact reference for a mass-law layer
inside a coherent stack, so beyond the convention written a second time, at 6.7e-16, the assertions
are the ones that need none: transmit port invariant under layer order at 1.2e-16, reflect port
independent of `m` exactly, the mass term one real factor exactly, a lossless-permittivity stack
returning no more than it received, and an air-like coherent sheet in front of a mass-law sheet
being that sheet at 1.2e-16.

Two planned assertions are withdrawn rather than weakened. "Bounded by the product of the
single-sheet gains" is false for a coupled stack, where a matching layer exceeds it, and an identity
for a cascade. And "continuous as `m → 0`" towards the coherent stack cannot hold, because the
settled single-sheet route already steps at `m = 0⁺` from the Airy pair to the incoherent route.

A bounding sheet with `m` keeps the half-space, asserted at 1e-9 and bit-identical across the change;
§1 says why that is its own convention.

One statement is restated. "The transmit port is invariant under layer order" was settled on
mass-law layers of permittivity 1.5 and above, where no face is closed, and holds there, at
1.2e-16; those rows have not moved since. For a stack with a rigid mass-law layer, whose faces
are closed, it holds by the rule of the closed layer, further down.

**A sheet under total reflection of its outer pair.** The sheet stays in the result; the convention
is in §1. Before, `SHEET_RESOLVE` wrote the bare near | far boundary there, a mirror on the reflect
pass and the outer pair's `tf` leak on the transmit pass. Measured against what it replaced: a lossy
sheet on a rigid wall reflected `|R|² = 1` where the composite gives 0.45 to 0.95, so a carpet on a
concrete floor absorbed nothing; a wood panel on concrete transmitted −12.0 dB where the chain gives
−29 dB to −37 dB, the panel's interface gain, `tf` and mass term all being absent, and reflected 0.9
into the room where its own face gives 0.8; and a coated EM solid past its critical angle reflected
as a bare mirror where the composite's TE port is −2.1 dB to −3.7 dB for 6 mm of `eps 5.24 − 0.63j`,
the wave being trapped between the near face and the totally reflecting far one and paying the loss
on every bounce. Asserted at 1e-9 on the reflect port, and on the transmit port in power and complex.
What it leaves open is the TM channel inside the clamp of §5.

**`alpha` in the characteristic-matrix core.** The core carries the `alpha` column instead of
declining on it. It is folded into the imaginary part of the layer phase over the path
`thk / |Re c|`, which is the same real one-way factor produce mode puts on `φ`, and it costs no
approximation the single-sheet path does not already make. Measured three ways: against
`diag_cm_stack` carrying it the same way, at 4.5e-16; against produce mode with no reference, a
`6 mm at 100 dB/m | 6 mm at 0` stack of one permittivity against a single 12 mm sheet at 50 dB/m, at
4.6e-16 on both ports to 85°; and as a continuity guard, a bounding sheet with 1e-6 dB/m resolving as
a layer where it used to drop to a half-space and move the host's series by 1.56 dB. Nothing else in
the suite moved when it landed.

**Glued entry against glued exit.** Measured, and correct as shipped; no source change was made.
*Reciprocity - glued entry against glued exit* resolves `air | sheet | solid` from the air and
`solid | sheet | air` from inside the solid along the direction a tracer holds after refracting in
at the same external angle, with the host leg isolated by a path-length ratio. For a lossless solid
the two agree to 1.6e-15 in scalar mode and on the EM gain from 0° to 80°, which is the guard on the
flux conversion: without it they would sit 1.76 dB apart at normal incidence. For a lossy solid they
do not agree, and the residual has a closed form: with `η = z·c` for TE and `z/c` for TM in the
solid, exit over entry is `|η|² / Re(η)² = sec²(arg η)` per channel. It is the flux splitting of §1
and nothing else, asserted against the closed form at 1e-9 and measuring 2.2e-15. For
`eps 1.50 − 0.40j` it is 0.073946 dB at normal incidence, where no cosine enters, 0.096 dB on the EM
gain at 50°, and 0.462 dB on TE and 0.200 dB on the gain at 80°.

**Leak reciprocity at a mass-law face.** A wave held undeviated inside a mass-law medium is charged,
at a face towards the lower index, what the other direction charges; the rule and its gates are in
§1. Before, a wood panel on concrete transmitted 0.20 dB, 4.48 dB, 4.19 dB and 2.99 dB less into the
wall than out of it at 10°, 30°, 45° and 70°, with the wall's interface gain accounted for, and the
same face as two traced solids differed by the same amounts. Now the two traced directions are one
number, at 0.000000 dB, and the sheet differs by 0.010 dB, which is the flat first face of §5 and is
removed to 2.3e-12 dB. It needed no state: the flag is read off the direction the tracer holds. A
wave the tracer refracted into a mass-law solid is asserted untouched. What it left open is issue 7;
the direction towards the higher index is closed further down.

**The leak chain at the cosine the wave has.** The far face of a sheet under a closed outer pair is
judged at the Snell cosine where the wave entered by an ordinary crossing and at the incidence
cosine where it entered as a leak; before, always at the incidence cosine. With a sheet rarer than
its host that found the far face open and wrote a Fresnel crossing into a medium that carries no
propagating wave: `eps 5.24 | 6 mm eps 1.50 | air` transmitted an EM gain of 0.867 at 26° and 0.501
at 32° with no `tf` anywhere, and `air | eps 0.31 | eps 0.05` 0.684 at 13°, where the exact answer
is zero. Both are exactly zero now. A sheet entered as a leak used to charge the evanescent Snell
decay, −0.58 dB at 40° to −1.44 dB at 75° over 10 mm of `eps 0.31`, which a traced solid does not;
it took the undeviated form and sat on its two flat leaks. That stands for a mass-law sheet.
A coherent sheet has no such branch any more: it is a wave medium and tunnels to the pad of
its closed far face (*`tf` at a closed face*, below).

**The series on a closed far face.** A sheet entered by an ordinary crossing sums what its closed
far face reflects, instead of dropping it. For `eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air` that is
worth 2.65 dB at 60° and 4.31 dB at 80°, and the lossless stack now conserves energy in scalar mode
to 1e-12 with no reference. For an absorber without `tf` on a wall with `tf`, which is the common
acoustic shape, the leak into the wall moved by +1.17 dB to +2.88 dB for a dense layer and by
−1.14 dB and −0.27 dB for a porous one. The step from the composite to the chain at the critical
angle of the outer pair closed with it: 1.975 dB before and 0.0024 dB now in scalar mode. The
transmit port of a coherent sheet is since the composite itself, with the series it always
carried on its reflect port (*`tf` at a closed face*, below).

**Bonded mass-law bodies as one plate.** Two mass-law bodies that share a face pay
`20·log10(X₀₁ + X₀₂)` with the single-body angle shape, telescoped along the path; the rule, its
three conventions and how each pairing supplies the running sum are in §1. Before, each body paid
its own term and the terms added in dB, the decoupled double leaf: 13.6 dB over for two sheets of
one wood, 16.5 dB for a wood sheet on a rigid board, 24.4 dB and 25.9 dB for a sheet on a solid,
39.2 dB for two traced solids, all at normal incidence and falling with angle. For acoustic rows
built with the level in the mass term, a 22 mm wood floor on 20 cm of concrete adds 0.25 dB to the
slab at every frequency now, where it added its whole term, 7.8 dB at 1 kHz and 19.8 dB at 4 kHz.
All three pairings are closed in both directions, the two traced solids through a thickness code
in the buffer word under its flag bit, which the state words had room for and which is quantized
to 0.0038 dB. Asserted at 1e-9 for the sheet pairings, in complex where one material makes the
single body a rule-independent reference, and at 0.01 dB for the solids.

One existing table moved when it landed and nothing else did: the stack of two mass-law sheets,
`M | MB`, by −4.81, −4.40, −2.64 and +0.44 dB on its mass term at 0°, 20°, 45° and 70°, in both
orders, its reference moving with it. The last figure is positive because near the knee and near
grazing the plate exceeds the dB sum, the angle shape being quadratic in a small `X₀`. Every row
with one mass-law body, or with an `m = 0` neighbour, is bit-identical across the change, since
the first body crossed runs the unchanged code path.

One planned assertion is restated rather than weakened. "Equal to one body of the summed surface
mass" is a statement independent of the rule only at `m = 20`, where `X₀` is linear in thickness;
it is asserted there, and for other slopes the mass paid is asserted against the rule written a
second time. A solid between two sheets and a bounce returning to the glued sheet it entered by
pay the plate as well, the mass rule needing nothing new for either: the host leg hands the sheet
ahead its own `X₀` plus what it was entered through.

**What a bounce inside a mass-law body pays.** It is followed, as it was, and each traversal pays
the plate once; a leg that begins at a reflection has its buffer word cleared and starts a new
traversal. Dropping the bounce was considered and rejected: it is worth nothing where the mass is
large and carries real energy where it is small, the sizes being in §1. Asserted on wood | concrete
as two solids: the reflection pass at the far face closes the first traversal at the plate, the way
back pays the body it starts in its own term to 1e-9 dB, with a stale previous-medium word in
place, and the plate in total.

**The reflect port of a face seen from inside a mass-law medium.** The other direction's
reflection, sign reversed, under the gates of the transmit port; §1 has the rule and what it does
and does not claim. Before, the two ports of wood | concrete summed to 1.035 at 10°, 1.338 at 19°,
1.505 at 30°, 1.452 at 45° and 1.276 at 70°, and to 1.54 on the EM gain at 30°. They sum to one
now, at 1.1e-16, with the reflectance equal to the other side's and the sign opposite. "One minus
the transmit port" was the alternative and is the same number for a lossless pair; it was not
taken because it has no phase and goes negative for a lossy neighbour.

**A glued exit behind a sheet.** It exits into air; the convention is in §1. Before, the far medium
was read from the previous-medium word, which after a sheet crossing holds that sheet, so a wall
lined on both faces resolved its second sheet against a half-space of the first lining and left the
event with that lining as the current medium. Measured on `eps 1.50 − 0.40j` over 100 mm lined with
6 mm of `eps 5.24 − 0.63j`: the transmit coefficient 0.24 off in the complex plane at normal
incidence and 0.40 at 40°, with the gain within 0.01 dB at normal incidence by coincidence; the
reflect port +0.32 dB and −1.44 dB at 0° and 40° and about 90° off in phase, up to +7.4 dB with
different linings on the two faces; the host's series −0.16 dB to −0.93 dB; and the ray leaving at
10.7° and 20.4° where host to air gives 25.1° and 51.8°, in a medium it was not in, so that the
next wall was entered as a nested solid. A mass-law sandwich and a bounce returning to the glued
sheet transmitted 6.98 dB too much, the leak out of the host being absent. All of it is asserted
now, at 1e-9 against the composite with air beyond and bit-identical against the same exit entered
through a bare face, and end to end: a lossless wall lined on both faces sits on the three-layer
characteristic matrix at 0.000000 dB from 10° to 85°, where it was −0.17 dB to +1.51 dB off. No
existing row moved; the previous-medium word after a glued exit into air changed from the host to
the sheet, which nothing reads.

**A glued exit behind a solid.** It exits into air, as every glued exit does; the convention is
in §1. Before, the far medium was read from the previous-medium word wherever that word named a
medium, so a wall of two solid layers with a coating on its far side, `air | A | B | coating |
air`, resolved `B | coating | A` and left the event with `A` as the current medium. Measured on
`B` = `eps 1.50 − 0.40j` over 100 mm with a 6 mm `eps 5.24 − 0.63j` coating, per event against
the same exit with a clean previous-medium word: the transmit coefficient 0.21 to 0.35 off in the
complex plane for `A` = `eps 5.24`, 0.24 to 0.40 for `eps 5.24 − 0.63j` and 0.35 to 0.51 for
`eps 9.0 − 1.20j` at 0° to 40°; the reflect port 1.18 to 1.74 off, up to +2.75 dB on the EM
gain; the host's series 5.6e-2 to 1.5e-1 off, the coating being backed by `A`; and the ray
leaving at 10.7° where 25.1° is right. End to end with `path_dir`: −0.18 dB, −0.21 dB, −0.08 dB,
+0.83 dB and +3.96 dB on the transmit port at 10°, 40°, 60°, 75° and 85° of external incidence
and +0.75 dB to −6.53 dB on the reflect port, with the ray leaving at 25.2° at 85°.

The state it left was worse than the coefficient. The ray continued in air recorded as inside
`A`, and nothing corrected that: the next wall's front face was a nested pass-through and its
back face a same-medium pass, types 72 and 73, with the gain replaced by one and no fix-flag, so
the wall was transparent and the ray was still in `A` behind it; a reflection off that wall was
the internal reflection `A | air`, a total reflection from 19.5° for `eps 9.0`, with the air
path charged as a leg of `A`, 12.5 dB and 18.2 dB below the clean reflection over 0.5 m for the
two lossy first layers.

All of it is asserted now: bit-identical against the clean-word exit per event, scalar and EM,
with and without the real-invariant flag in the word; at 1e-6 dB on the chain against the single
pass times the composite and, with the second solid's series resolved, against the matrix of
{`B`, coating} between `A` and air; and the next wall bit-identical to a wall met with clean
state words. The change is one line, the far medium of `sheet_case 3`, and no existing row
moved. The mirror behind the ray is still `mtl_prev`, which is right.

One planned assertion is restated rather than weakened. "`air | A | B | coating | air` on the
characteristic matrix of the stack" cannot hold with both series resolved, for a reason that is
not the glued exit: one path through two solids is not the wall, the sum of the paths is (§1). It is asserted in the
two forms the factorization is exact for, and the distance to the whole-wall matrix is a
recorded column.

**The series on a resolved ray at a sheet.** Skipped, as at a bare face. Before, `SHEET_EVENT`
applied the host's series whatever the resolved flag said, so a ray reflected inside a wall and
leaving through a glued sheet, or crossing a sheet inside the wall on its way back, carried the
series twice: 0.03 dB to 0.31 dB over the lossy and the lossless host. Asserted bit-identical
against the run with the series declined, for a glued exit, an embedded sheet and an ordered stack.

**The recovery behind a sheet.** It runs, solving for the real transverse invariant with no entry
medium named; the reasoning is in §1.
Before, it declined wherever the previous-medium word held a sheet, so every leg after a sheet was
built at the real projection while the entry had been evaluated at the exact cosine. On the lossy
host that cost a wall entered through its coating −0.002 dB, −0.025 dB, +0.17 dB, +1.49 dB and
+5.47 dB end to end at 10°, 40°, 60°, 75° and 85°; a wall lined on both faces −0.01 dB, −0.25 dB,
−0.81 dB, −1.04 dB and +1.18 dB; the bare far face alone up to +3.80 dB; and a sheet inside the
host reached behind another one up to −0.49 dB. Both chains are on the characteristic matrix now,
at 0.000000 dB to 75° and inside the clamp of §5 at 85°, and a leg behind a sheet is identical to
the same leg with a clean previous-medium word. That the entry medium does not have to be named is
asserted on its own: a leg entered from a lossless `eps 5.24` solid gives the same result with
that solid in the word and with a sheet there, at 4.1e-16, past the invariant air could launch.

One planned assertion is withdrawn rather than weakened. For an invariant above one the comparison
was first made on the total reflection at the host | air face, where it failed by a sign: the
branch of the evanescent root at a recovered cosine was decided by rounding, which was a defect
of its own, closed below, and not the recovery. It is made on the transition into the dense solid instead.

**The recovery behind a sheet names no medium, also behind a lossy one.** Handing it the medium in
front of the sheet was considered and rejected on a measurement: a solve with a lossy medium
named assumes a real angle in it, and was 3.5 dB off the reference at 85° where the unnamed solve
is exact. A planned assertion, "a leg behind a sheet in a host entered
from a lossy medium at the cosine that medium gives", is withdrawn: it describes the wrong physics.

**The recovery behind a lossy solid.** It solves for the real transverse invariant wherever the
shared face was evaluated at the recovered cosine of the first solid; the two cases and the flag
are in §1. Before, the solve named whatever solid the previous-medium word held and assumed a
real angle in it. For `air | eps 5.24 − 0.63j 50 mm | eps 1.50 − 0.40j 100 mm | air` across bare
faces that was −0.0015 dB, +0.059 dB, +0.647 dB and +3.522 dB end to end at 40°, 60°, 75° and
85°, +2.44 dB on the TM channel at 85°, and a ray killed behind a lossless host from 84.6° by a
total reflection the recovered cosine invented. With `eps 1.50 − 0.40j` in front it was +1.63 dB
at 75° and +5.44 dB at 85°. Per consumer at 85°: the exit port +3.49 dB, the traversal +0.03 dB,
the host's series −0.22 dB (−1.75 dB for a lossless host), and the composite of a sheet inside
the host −0.23 dB on its transmit port and +0.08 dB on its reflect port. All of it is asserted at
1e-6 dB now, scalar and per channel in EM, with the named run identical to the run with a clean
previous-medium word. Entered from air the solve is untouched: the bound `u ≤ 1` it carries there
never binds, because the residual is concave and Newton approaches the root from below, checked
bitwise over 75108 rows offline and by the diff, in which no air-entered row moved.

**The direction a transition writes.** Out of a lossy solid whose cosine was recovered, the Snell
ratio handed to `qd_refract` carries the length `N` the projection in that solid lost (§1), so
the direction in the next medium is the one a single refraction from air writes. Before, the
shortfall was handed on: the recovered invariant in the host read 0.9918 for a true 0.9924 behind
`eps 5.24 − 0.63j` and 0.962 behind `eps 1.50 − 0.40j`, worth +0.011 dB and +0.138 dB at 75° and
85° in the first case and +0.53 dB and +3.06 dB in the second, and it compounded: three solids
deep the chain was +3.34 dB off at 85° and the traced cosine in the third solid 0.017 to 0.037
off. Undoing it at the reader was the alternative and was rejected: it needs the solid named,
which the unnamed solve gave up, it is exact two solids deep only, and the factor depends on
frequency, so it could not have travelled in the state. Asserted to nine digits on the cosine and
at 1e-6 dB on chains of two and three solids. For a lossless solid the factor is not applied and
those rows are bit-identical.

**A face evaluated at a real angle keeps the solid named.** Dropping the name on every leg was
the first design and was measured before it was taken: behind a lossy solid with non-parallel
faces it cost −0.065 dB, −1.01 dB and the ray at 60°, 75° and 85° of external incidence, and for
a wave launched inside the solid −0.13 dB and −1.35 dB at 23° and 25°, on legs the named solve
has exactly right. The real-invariant flag separates the two at no cost in state; its polarity,
set where the cosine *was* recovered, is what leaves every earlier leg and every earlier state
word untouched. Asserted on 16 rows at 1e-6 dB with the flag asserted clear; the case keeps a
recorded column of what the unnamed solve would do there.

Twenty-four rows of one existing table moved when this landed, the lossy rows of the bare-face
measurement case, all onto the reference, and nothing else did.

**The material index is 14 bits.** 16383 materials, with bits 14 and 15 of each state word as
flags; the layout and the one decode block are in §1. Masks, range checks and limits moved together
in `ray_state_update`, `ray_progress`, `ray_commit` and `calc_diffraction_gain`. The thickness code
keeps its 15 bits: cutting it to 14 would have doubled its step to 0.0076 dB and moved the
wood 60 mm | 40 mm row. No existing row and no existing state-word assertion moved.

**A sheet inside a host is backed by the host.** The inside flag, bit 14 of the previous-medium
word, names what is behind the sheet behind the ray; §1 has the rule. Before, that sheet was a layer
backed by air whatever it lay in. Measured on a 6 mm `eps 5.24 − 0.63j` sheet inside 50 mm of
`eps 1.50 − 0.40j`, as the series the leg behind the sheet applied against the host-backed one:
−0.035, −0.076, −0.316 and −0.694 dB at 0°, 20°, 40° and 50°; for a near-matched sheet, which
inside its host is almost no mirror at all, −0.979 dB at 50°. Asserted at 1e-9 over 32 rows,
measuring 3.3e-16, with the coating as the control that stays on air.

**What is behind a sheet needs a flag and no index.** Storing the medium a host was entered from,
so that a lined body nested in another medium would exit into that medium and back its lining
with it, was designed and then dropped: such a body is not a use case in EM or in acoustics, and
with it gone the medium behind a sheet is air or the host. Three planned assertions went with it
and are withdrawn rather than weakened: a wall lined on both faces inside a medium exiting into
that medium, and the two rows of a backing sweep with a solid behind the sheet, which need a sheet
between two solids. Moving the sheet out of the previous-medium word into the buffer word, so
that the previous-medium word always names a medium, was the more readable layout and was rejected
for one reason: a sheet's index and a 15-bit thickness code do not fit one word, and the plate
below needs both.

**The plate through a sheet inside a wall.** A wall in two halves with a mass-law sheet between
them pays as one plate of three bodies. The crossing already charged the first half and the sheet
correctly; the second half knew only the sheet and paid almost its whole term again. It now reads
the first half's thickness from the code in the buffer word, with its own material. Before, at
normal incidence: wood 60 mm | denser wood sheet 8 mm | wood 40 mm paid 52.21 dB against a plate of
40.67 dB; wood 50 | rigid board 12 | wood 50, 46.09 dB against 43.02 dB; concrete 50 | wood 20 |
concrete 50, 76.87 dB against 57.23 dB; concrete 30 | wood 20 | concrete 70, 74.94 dB against
57.73 dB. All four are within 0.0016 dB of the plate now, which is the code's quantization, and
the slopes of sheet and wall need not agree. A sheet without the mass column still separates the
halves, asserted exact. A sheet with the wall's own permittivity takes this path as well where
it carries the mass column; see the next entry.

**The plate at any depth where the slopes agree.** A leg learned what it is bonded to from two
words that name one body behind the ray, or two around a sheet inside the medium, and every
body further back was treated as not crossed. Three solids paid `TL(a+b) + TL(b+c) − TL(b)`,
the same from both sides. Measured through the intersector at normal incidence, far above the
knee, over the plate: three equal solids +2.50 dB, 30 \| 100 \| 200 mm +1.45 dB, a 13 mm layer
bonded between two 100 mm bodies +13.28 dB, four solids +6.02 dB, three solids of `m` 30
+0.40 dB, 15 mm of plaster on either side of a brick wall +0.12 dB, a lining on two solids
+0.59 dB from the lining's side and exact from the other, a lined wall holding a sheet +0.55 dB
and exact, a solid in front of a wall holding a sheet +2.02 dB and +2.98 dB, two sheets inside
a wall +0.50 dB and +0.64 dB, bodies of unequal `fRef` +1.51 dB, and a wall holding a sheet of
its own permittivity +0.42 dB and +0.92 dB at `m` 20, +3.02 dB and +3.35 dB at `m` 30. The
over-charge was flat in the angle above the knee and turned into an under-charge near it, to
−1.09 dB at 75°. The thickness code now carries the equivalent thickness of §1, and a mass-law
sheet in a mass-law host is not merged. All of these are within 3.0e-3 dB of the plate, both
directions, both tracers, far above and near the knee, to 75°, and through `ray_progress` alone
and in a band; a bounce pays the plate once per traversal on them. No frequency enters the
code: the state words of every chain are the same at 0.25, 1 and 4 GHz, asserted.

Nothing else moved when it landed: every existing row of the diagnostic output was identical,
the two-body rows because the equivalent thickness of one body is its own, and the chain
without its mass, `rest`, is unchanged on every row. Three statements were restated with it
and none was reopened: what the thickness code holds (§1, the state words), where a fitted
slope other than 20 is reachable (§1; a wall cut by a sheet inside it had been asserted on the
summed impedances of its halves all along), and what is merged into a host leg. Three options
were measured or weighed and not taken. Matching the thickness of a mixed-slope run at the base
frequency is exact there and wrong elsewhere and makes a result depend on which frequency is
the base. Refusing to fold where the next body has another slope keeps three solids of mixed
slopes equal from both sides by keeping the error on both, and is a gate for a symmetry the
stacks with sheets never had. Keeping a mass-law host merged, with the sheet charged as crossed
first and remembered in the code, keeps the table's term for a host of `m` other than 20 but
needs the code read with the current material where air is in the previous-medium word. What
remains is in §5: mixed slopes, the range of the code, and the deferred medium, which is
issue 8.

**The recovery in a traced run.** The hit code's total-reflection bit no longer makes the recovery
decline; the reasoning is in §1. Before, every exact-cosine consumer fell back to the projection
on a face the wave crosses, from the angle at which the geometric verdict trips, and no fixture
saw it, because every chain handed `ray_state_update` a synthetic code. Measured with the code
`ray_mesh_interact` writes: `air | 100 mm | air` against the analytic slab +0.308, +1.748 and
+5.792 dB at 60°, 75° and 85° for `eps 1.50 − 0.40j`; +0.008, +0.025, +0.671 and +4.357 dB at
40° to 85° for `eps 5.24 − 0.63j`; +0.001 dB to +4.633 dB from 20° for `eps 9.0 − 1.20j`; a
glued exit out of `eps 1.50 − 0.40j` −0.439, −0.162 and +2.659 dB on the transmit port and
−0.319, −1.765 and −3.756 dB on the reflect port. All of it is at 0.000000 dB now, the event
with the bit is bit-identical to the event without it, and the traced run through the
intersector sits on the slab at 2.8e-5 dB. No existing row moved, since none carried the bit.

**The decaying root at a recovered cosine.** `qd_evanescent`, a relative test, at the ten sites
that force the root; §1 has the rule. Before, the reflected phase at a face closed at a
recovered cosine jumped between the two roots from one angle to the next, 8 of 14 steps above
45° and up to 179° on a sweep of the invariant from 1.10 to 1.45, and the same wave with a
dense solid named and with a sheet in the previous-medium word disagreed on 6 of 15 rows. The
sequence it now sits on is the decaying one, checked against the closed form: at an invariant
of 1.1 the two roots differ by 86.68° in the reflected phase and the decaying one has a
reflection magnitude below one. Magnitudes did not move, the reflectance of a closed face being
set to one either way. One existing row moved when it landed, in the last digit: the `30.0`
entry row of *an m = 0 lining on concrete*, its residual from −1.4e-14 dB to zero, stable from
run to run; which of the ten sites it passes was not traced.

**The direction a ray leaves a slab in.** An exit into air writes the direction the wave has
there; the two writers and their gates are in §1. Before, a glued exit out of a lossy host
formed its direction from the real-angle interface call at the traced angle and left
`eps 1.50 − 0.40j` at 9.99°, 19.93°, 39.32°, 56.55°, 64.96° and 67.29° for 10° to 85° of
external incidence, whatever the host was entered through. The split was measured: at 85° the
tangential part is 0.0187 short of `sqrt(p)`, which is the `1/N` of the projection and worth
0.4°, and the normal part is 0.3219 above `sqrt(1 − p)` and worth 17.3°. A bare exit kept what
`ray_mesh_interact` wrote, which with the undeviated tracer is a second bend of the external
direction: 12.28°, 24.76° and 51.93° out of the lossless `eps 1.50` for 10°, 20° and 40°, and
exact above only because the geometric verdict there returns the undeviated direction. Both
are on the incoming direction now, to 1.1e-15, bare-entered, behind a sheet and behind an
ordered stack inside the host, straight through and for a resolved ray. What moved with it:
the `out_deg` column of the layered-wall chain, ten rows, to the external angle, and EM gain
residuals in three tables in the last digit, below 5e-15, the outgoing polarization basis
being built from a direction that differs in the last bit. No coefficient, state word or type
code moved. Read by: a sheet event reached in air, through a glued entry the recovery in the
next wall, and the outgoing basis; a bare entry, a reflection in air and the ledgers do not
read it. Two coated lossy walls in a row at 85° had the second resolved for 67.3°.

One planned assertion is restated rather than weakened. "A leak through a mass-law
solid | solid face written undeviated, as a sheet and as two traced solids" holds towards the
lower index and is asserted there. Towards the higher index it was recorded at the time and
is closed further down.

**The refracting tracer runs the exact machinery.** A leg that follows a traced bend says so in
the bent flag of the current-medium word, and every reader that asked the cosines whether the
wave was bent asks the flag as well; the rule, its writers and what is kept apart from it are in
§1. Before, with types 2 and 5, the default of `ray_progress`, the traced and the geometric
cosine coincided on every leg and every consumer took the branch written for a ray that was not
bent. Measured through the intersector against the undeviated tracer, on `eps 1.50 − 0.40j`
over 100 mm at 40°, 60°, 75° and 85°: `air | slab | air` +0.018, +0.308, +1.748 and +5.792 dB,
with +4.36 dB and +4.63 dB at 85° for `eps 5.24 − 0.63j` and `eps 9.0 − 1.20j` and +4.01 dB on
the TM channel; a glued exit −0.171, −0.330, +0.543 and +4.310 dB; a sheet inside the host
−0.013, +0.193, +1.425 and +5.148 dB; two solids across a bare face up to +3.27 dB and
+5.15 dB; a back reflection −0.072, −0.548, −1.721 and −1.297 dB. Whether a ray took the exact
branch was decided by rounding: in single precision with the scene 100 m from the origin 3 of 17
angles did; behind a glued entry every angle from 60° did, through the relaunch anchoring; and
at every frequency of a run but the base one all of them did, so one run was exact at 2 GHz and
+5.79 dB off at 1 GHz. All of it is asserted now, at 1e-3 dB against the analytic slab and
against the undeviated tracer, and the direction written at the exit is the one the ray arrived
in, to 1e-9, where it was 9.99°, 19.93°, 39.32°, 56.55°, 64.96° and 67.29° for 10° to 85°.

What moved with it. No row of the undeviated tracer and none of its state words. Of the
refracting tracer: the six lossy rows of the last table of the recovery-gate case, onto the
slab, and the bare-exit rows of the direction case, the lossy ones onto parallel and the
lossless ones in the last digit; and the current-medium word inside a medium, which gained the
flag, asserted in four places in `test_ray_progress.cpp` and in the type-5 entry section of
`test_ray_state_update.cpp`.

A stateless rule was the alternative and was rejected on a measurement: "the current medium is
denser than the one it was entered from" covers every single wall in air and leaves a dense
layer in front of a light one where it was, +0.48 dB at 75° and +3.27 dB at 85° for
`eps 5.24 − 0.63j` in front of `eps 1.50 − 0.40j`, and it cannot serve the reflect pass, whose
types are 0 and 3 with either tracer.

**No walk-off on a bent leg.** The `cos²` of the walk-off form stands in for a lateral shift the
tracer did not trace, and with the refracting tracer it was applied to legs whose shift is in
the polyline, lossless ones included. Measured as the phase at the polyline's exit point against
the analytic slab, 100 mm at 1 GHz, 10° to 85°: 0.55°, 2.19°, 8.73°, 19.08°, 27.30° and 30.69°
for the lossless `eps 1.50`, 0.53° to 17.6° for `eps 1.50 − 0.40j`, and twice the lossless
figures on a back reflection, 61.4° at 85°. It is 0.009° now, on every pass. The power of a
lossless row did not move; its phase did, and had to.

**A resolved ray on a bent leg leaves at the recovered cosine.** Its exit is evaluated as any
other; with the series collapsed a back reflection out of `eps 1.50 − 0.40j` was +0.08, +0.82
and +3.85 dB off at 60°, 75° and 85° with everything else in place, and is at 4e-6 dB. The
undeviated tracer's resolved ray is closed further down.

Two planned assertions were restated rather than weakened, and one withdrawn. The bound of the
slab rows is kept at 1e-3 dB by giving the reference the thickness the polyline crossed, which
is short by the relaunch offset; without that a dense slab sat at +0.0014 and −0.0038 dB at 85°
with either statement of the physics. The rows that compare the two tracers on a layered wall
have no reference to put the offset into and are bounded at 2e-3 dB, measuring 1.1e-3 dB on the
lined lossless wall at 85° and below 6e-4 dB elsewhere. And "every rigid stack is the undeviated
tracer's with the flag clear" was withdrawn for concrete in front of wood, where the shared face
bent the ray at the time; it holds since the leak direction was closed, further down, and is
asserted bitwise.

**The segment out of a lossy medium.** With the refracting tracer, the default of
`ray_progress`, the relaunched segment and the ray tube follow the direction `ray_state_update`
writes; the rule, its gate and the line are in §1. Before, `ray_mesh_interact`'s segment at a
real angle in the medium was what the ray travelled on, while `path_dirN` held the right
direction. Measured through `ray_progress` on `air | 100 mm | air`: out of `eps 1.50 − 0.40j`
the segment ran at 9.99°, 19.93°, 39.32°, 56.55°, 64.96° and 67.29° for 10° to 85°, out of
`eps 5.24 − 0.63j` at 75.34° for 85°, and no ray left the first steeper than about 70°. A
second wall 1.9 m behind was met 0.0003 m, 0.0027 m, 0.038 m, 0.41 m, 3.02 m and 17.18 m from
where the wave meets it, at the angle of the segment, and the error was inherited: inside that
wall the ray ran at 46.88° where the wave runs at 51.68°, and left it with 67.29° written and
61.35° travelled. The same held at a glued exit, a lined wall, behind a sheet inside the host
and for a ray leaving by the front face after a bounce. Behind `eps 5.24 − 0.63j` the segment
in a second solid was 0.41° and 0.91° off the direction the transition wrote at 60° and 85°,
0.18° and 0.42° for a lossless second solid, whose own exact exit then left at 82.27° for 85°.

The tube was worse than the spine. Each vertex was refracted by the same real-angle map, which
saturates towards grazing, so a fan launched at 0.5° left `eps 1.50 − 0.40j` at 85° with its
in-plane vertex 0.053° from the spine, every vertex 17.5° to 18.2° off the direction it was
launched in. A beam behind a lossy wall was therefore too narrow to be split where the same
beam behind a lossless wall was, 4 rays against 16 in one measured case, and a split in air
replaced the written direction by the tube's, 65.05° where 75.25° is right.

All of it is asserted now, at 1e-6 per component through `ray_progress` and at 1e-9 through
the intersector chain in double, where the segment sits on the written direction to 3e-16.
Rebuilding the segment in `ray_progress` was the alternative and was rejected: that function
knows neither which events wrote such a direction nor the medium a vertex ray needs, so it
would have had to follow `path_dirN` everywhere, which moves every lossless segment in the last
digit, or use a tolerance, which rounding decides, and it could only have rotated the tube
rigidly, which keeps the collapsed fan.

What moved with it. No row of the baseline. Through `ray_progress`, the geometry of every ray
of the refracting tracer that leaves a lossy medium, and with it everything behind a lossy
wall; the number of rays in beam mode behind such a wall rises, the fan being restored. The
raw `orig` and `dest` of a lossless leg, of the undeviated tracer, of a reflected ray and of
the mass-law solid of issue 7 are identical digit for digit. Where the geometry is handed in,
the exit behind a lossy first solid carries resolved type 1 where it carried 3 (§1).

One planned assertion was withdrawn, two restated. "The chain through two solids on the
undeviated tracer's in phase, to 0.05°" is withdrawn: the two tracers differ by 0.17° at 85°
through two solids whether the geometry is handed in or not, and a lossless pair carries the
same figure, so it never was the segment in the second solid; §5 records it. "Every type code
identical with and without the geometry" holds for every bit but the VBS bit, whose meaning is
the geometry, and is asserted so. And a 1e-6 bound on the lossless rows behind a glued entry
is 1e-5: such a leg carries that entry's anchoring, which this did not touch.

The generated API pages under `html_docu` have to be regenerated for the four arguments.

**The phase of a traversal is the field at the point the polyline leaves at.** The convention,
its two terms and their gate are in §1. Before, the exact form and the path form of a bent wave
carried the phase of the refracted ray, at a point beside the committed path. Measured through
the intersector on `air | 100 mm | air` at 1 GHz as the phase against the analytic slab at the
polyline's exit point: with the undeviated tracer 0.74°, 3.24°, 18.6° and 81.8° at 10° to 60°
for `eps 1.50 − 0.40j`, 0.69° to 76.1° for the lossless `eps 1.50`, 2.1° to 138° for
`eps 5.24 − 0.63j`, more than a turn towards grazing, and the same picture at 0.5 and 2 GHz;
with the refracting tracer zero at the base frequency and 0.07° to 12.4° at half of it, −0.08°
to −22.4° at twice it, twice that on a back reflection. For an acoustic absorber, mineral wool
50 mm with 1 kHz as base, a reflection off the layer on a rigid wall was −13° and −52° off at
2 and 4 kHz for 40° and half a turn at 4 kHz and 75° with the refracting tracer, and 5° to 33°
over 250 Hz to 4 kHz at 40° with the undeviated one. At the refracted ray's point every one of
those rows was within 0.016°, so the form was exact and the point was not the path's. All of it
is asserted now at 0.05°, measuring 0.016° or less, with every power at 0.000000 dB.

No geometry is per frequency, before or after: one polyline, and per frequency the value the
field has where it leaves. A path per frequency was never an option, the later iterations of
`ray_progress` and the storage format both holding one path.

What moved with it. No power of a transmitted leg. Every phase of a leg of the undeviated tracer
that was bent, and of the refracting tracer off the base frequency. The rows that compared the
two tracers as absolute transfers are restated to each tracer at its own point, 210 rows, all
within 0.03°. `named_arg` and `sheet_arg` of the entry-medium table of the recovery-behind-a-sheet
case, four recorded rows, by up to 207°. Three hand-built cases that pin the exact form in
complex took the polyline term into their reference, 68 assertions that had failed by 0.0085 to
0.39 as a complex difference and returned to 1.6e-15. Residual columns of twelve more tables
moved at 1e-15 or in the sign of a printed zero. A bent leg of the refracting tracer at the base
frequency did not move in any printed digit; that it is bit-identical was not asserted
separately.

**One form per leg, on every pass.** A wave that was bent takes the same traversal whether the
transmit pass, the reflect pass or a resolved ray closes its leg; the walk-off form is left to a
wave that was not bent (§1). Before, the reflect pass and the resolved ray of the undeviated
tracer took the walk-off form on a leg that had a recovered cosine, with the path-rate loss, so
a back reflection carried one leg of each form and sat on neither reference: −3.25° at the
ray's point and +3.23° at the polyline's at 20° for `eps 1.50 − 0.40j`. That moved a power, by
decision: the reflect-pass leg through 100 mm of `eps 1.50 − 0.40j` at 1 GHz by 0.000025,
0.00012, 0.0027, 0.0146, 0.0316 and 0.040 dB at 10° to 85°, onto the analytic value; half of
that through 50 mm; up to 0.375 dB for mineral wool at 4 kHz and 75°; and 0.063 to 0.293 dB on
the fifteen recorded rows of the closed-face table of the recovery-gate case, a reflect pass in
a lossy host at an invariant of 1.10 to 1.45, where the wave is close to grazing in the host.
A lossless medium did not move. The references that said "a reflect pass at a sheet closes the
host leg with the path form" are restated to the normal-thickness decay, in the layered-wall
case, the recovery-behind-a-lossy-solid case, the glued-exit table of the recovery-gate case
and the sheet-in-a-medium case, 78 assertions, and sit at their old bounds.

**The path form at the polyline's point.** A mass-law solid the wave is refracted into, and a
leg whose recovery declines, take the phase of the real index along the polyline (§1). Measured
on `eps 1.50` with `m 20` as the traversal over 50 mm, the undeviated tracer was 0.35°, 1.50°,
8.59° and 38.1° off at 10° to 60° and is at 0.0000°. Magnitudes are untouched. A leak into a
rigid body did not move and is asserted bit-identical between the two tracers. A leg whose
recovery declines, a wedge or a wave launched inside a solid, has no fixture for its phase.

**The displaced-host term of the sheet composite is complete.** It divides the displaced host
out on the thickness axis, and that is the insertion factor at a fixed lateral position, the
polyline's convention (§1). A transverse term was proposed for it and is withdrawn on a
measurement: on `air | host 50 | sheet 6 mm | host 50 | air` the chain was within 0.006° of the
single pass at the point of a ray refracted in the host alone, which for the refracting tracer
is the polyline's point, and the proposed term would have put it −0.18° to −9.08° from there,
on no ray either, the ray that is also refracted in the sheet lying at −0.08° to −5.63°. With
the host legs at the polyline's point the chain is on its reference for both tracers, asserted
at 0.05°. No change was made to `qd_slab_resolve`, to the core or to their reference builders.
That is a statement about the transmit port. What the reflect port owes for the same layer is
its own entry below.

**The phase between the two tracers through two solids.** It was 0.17° at 85°, the same for a
lossless pair, and its cause is the polyline between a hit point and the next relaunch point,
which no ledger holds (§5): the relaunch offsets and the gap of the co-located pair, walked at
the external angle by one tracer and at the refracted one by the other. With each tracer at its
polyline's exit point it is 0.017° or less and asserted at 0.05°, and the remainder follows
from that geometry to 0.0002°.

**A resolved ray leaves at the cosine the wave has.** `REPLACE_BY_VBS` used to return early
for a resolved ray, on the grounds that such a ray needs no coefficients for the series, and
with that it also skipped the replacement of what `ray_mesh_interact` had formed at the angle
of the segment and the verdict that goes with it. With the undeviated tracer that angle is the
external one, on a face the wave meets at the refracted one. Measured through the intersector
on `air | 100 mm | air`, enter, reflect at the far face with the series collapsed, leave by
the front face, against `t01 r10 t10 φ² S`: −0.0014, −0.0071 and −0.080 dB at 10°, 20° and 40°
for `eps 1.50 − 0.40j`, with −0.09°, −0.43° and −3.53° of phase, −0.0012, −0.0063 and −0.087 dB
for the lossless `eps 1.50`, and no ray at all from 60°, the face being judged closed from the
geometric critical angle, 53.4° for `eps 1.50` and 25.8° for `eps 5.24 − 0.63j`. A ray
reflected off a coating on the far side showed the same figures with the series declined,
because a reflection off a sheet always flags the ray resolved: with the undeviated tracer
every wall coated on its far side lost its inner echo above that angle, at any `eps`. And a
resolved ray crossing back into a first solid, `air | A | B | air`, was written neither a
coefficient nor a direction: against the refracting tracer −0.02, −0.10 and −0.71 dB at 10°,
20° and 40° for `eps 5.24` in front of `eps 1.50 − 0.40j`, −0.09 and −0.63 dB at 10° and 20°
with the two exchanged, and dead from 60° and from 40°.

The branch is gone, and a resolved ray is evaluated as any other (§1). All of it is asserted:
the back reflection at 1e-3 dB and 0.05° to 85° for both tracers, measuring 0.000000 dB and
0.0044°; the far-side coating at 0.000000 dB and 0.006°; the two solids at 9.4e-4 dB and
0.0024° between the tracers. That moved a power, by decision: every figure above, onto the
reference, and the dead rays back to the full echo. What moved with it in the tables that
existed: the 24 resolved rows of the back-reflection table of the phase-reference case, onto
the reference, and the 12 resolved rows of the bare-exit table of the direction case in their
type code, 1 and 33 becoming 3, the VBS bit set and the total-reflection bit re-derived; their
direction columns did not move. No leak row moved and the trajectory case through
`ray_progress` did not move. The refracting tracer evaluated these exits already.

**A bare and a glued exit write one direction where the recovery declines.** A bare exit out
of a lossy medium whose recovery declined kept what `ray_mesh_interact` wrote, the real-angle
map of the segment, where the glued exit applied that map to the direction the wave has. The
gate that left it there had been drawn while the recovery still declined on the hit code.
Measured through the intersector on a wedge of `eps 1.50 − 0.40j`, exit face tilted by 10°,
undeviated tracer, against the exit normal: the bare exit wrote 0.00°, 12.49°, 38.15° and
65.36° for 10°, 20°, 40° and 60° of external incidence where the glued exit wrote 2.48°,
7.37°, 26.25° and 42.70°. The gate is removed, which took code out, and both write the latter;
asserted at 1e-9 for both tracers, lossy and lossless. No power moved, and no geometry: with
the undeviated tracer the segment is the incoming one. The lossless host and the refracting
tracer wrote one direction before and after, in every printed digit. What the map itself
leaves is in §5.

**The verdict of a sheet's outer pair at the recovered cosine.** `SHEET_RESOLVE` judged total
reflection of near | far with the real-angle call at the traced angle, where a bare face was
judged at the recovered cosine (§1). Measured on `eps 1.50 − 0.40j` entered from a lossless
`eps 5.24` solid: for an invariant of 1.01 to 1.04 the coated exit read open, `p/N²` being
0.971 to 0.997, carried resolved type 71 where 103 is right, and wrote a direction of 67.9° to
68.9° into a medium that carries no wave. The transmit port was zero there all the same, the
flux conversion at the exact cosine being zero, so without `tf` no power was wrong. With `tf`
0.5 on the coating the leak was missing in that window: zero, between −5.51 dB at 0.99 and
−5.89 dB at 1.05. The verdict now comes from the recovered-cosine call wherever the near
medium has a recovered cosine, and the direction block reuses that call. Asserted: closed
exactly from an invariant of one, bare and glued; the direction of the wave handed on; the
leak at −5.86 dB to −5.88 dB through the former window. That is the one power it moved. The
reflect port in the window did not move in any printed digit; its type code went from 207 to
239. No row that existed moved. Reached only by a host entered from a medium denser than what
it exits into, so never by a wall standing in air.

**The reflect port of a coated lossy solid past its critical angle.** Measured, and correct as
shipped; no source change was made. On `eps 1.50 − 0.40j | 6 mm eps 5.24 − 0.63j | air`
entered from a lossless `eps 5.24` solid, scalar and TE sit on the composite at the exact
cosine below 1e-15 up to an invariant of 1.45, asserted at 1e-9; TM is
inside the clamp of §5, 1.2e-2 to 1.8e-2 in amplitude, guarded at 5e-2.

The rows above an invariant of 1.35 used to carry what three Newton steps of the recovery
left, 1.0e-9 at 1.40 and 9.5e-9 at 1.45. The recovery now runs until it has converged and
every row is on the reference, asserted at 1e-12 against the reference at the invariant.

**The direction of a leak through a mass-law body.** A wave held undeviated in a mass-law
medium keeps its direction at every face, towards either index, in the segment and in
`path_dirN`, with either tracer; the rule and where it sits are in §1. Before, only the
total-reflection verdict kept a leak straight: true at the entry and at a face towards the
lower index, false towards the higher index, where `qd_refract` ran with the Snell ratio of two
impedance mappings. Measured through the intersector, wood `ε = 2.1609e-8` and concrete
`ε = 2.401e-9`, at 30°, 60° and 75°: every exit into air left at 0.0014°, 0.0024° and 0.0027°,
along the wall's normal, bare or glued; concrete into wood ran at 9.59°, 16.78° and 18.78°, and
with types 2 and 5 that leg carried the bent flag. With the undeviated tracer that was
`path_dirN` only, which a sheet event in air reads; with the refracting tracer, the default of
`ray_progress`, it was where the ray went.

What it was worth, and what moved, by decision, at 30° / 60° / 75° (`heavy`: mass terms of 34 dB
and 51 dB per 50 mm; `light`: 8 dB and 12 dB):

| stack | undeviated tracer | refracting tracer |
| --- | --- | --- |
| wood \| concrete, panel \| concrete, concrete \| panel | 0 | 0 |
| concrete 50 \| wood 50, heavy | −0.0012 / −0.0046 / −0.0070 dB | −0.0012 / −0.0046 / −0.0069 dB |
| concrete 50 \| wood 50, light | −0.0012 / −0.0047 / −0.0070 dB | +0.063 / +0.481 / +1.517 dB |
| concrete, 0.5 m air, concrete | 0 | −0.0066 / −0.0037 / −0.0019 dB |
| concrete, air, panel \| concrete, heavy | +1.235 / +5.976 / +11.577 dB | the same |
| concrete, air, a panel alone, heavy | +1.225 / +5.966 / +11.568 dB | the same |
| the same two, light | −0.010 dB and −0.020 dB | the same |
| concrete \| wood, air, wood 50, heavy | −0.001 / −0.005 / −0.007 dB | +1.114 / +5.618 / +11.216 dB |
| concrete \| wood, air, wood 50, light | −0.001 / −0.005 / −0.007 dB | +0.929 / +4.346 / +7.678 dB |

The large figures are the mass-law angle factor: a lining or a panel behind a wall was charged
as at normal incidence, `20·log10(cos θ)` too much, with either tracer, because its sheet event
took the direction the first wall had written; and with the refracting tracer a wall behind a
layered wall was met at the bent angle. The two tracers differed by up to 1.52 dB on one light
wall, the mass term taking its cosine from the bent segment. With the refracting tracer the ray
also went elsewhere: behind 100 mm of concrete and 0.5 m of air the next hit was 0.29 m off at
30° and 1.87 m off at 75°, inside the cone in which a rigid material admits a wave, where it was
refracted into the second wall and flagged as bent. All of it is asserted now: every direction
and segment on the incidence angle to 1e-9, no bent flag, every event on the held run, and the
refracting tracer on the undeviated one in every word and type code.

It needed no state and no new gate at a call site: the geometry of no bend comes out of
`interact_with`, and one condition keeps the bent flag off a leg entered out of a mass-law
medium. Coefficients, verdicts, hit codes and resolved types of every face are what they were.
Two options were measured and not taken. Holding the direction at exits into air only leaves
concrete in front of wood leaving at 9.59°, 16.78° and 18.78° and the wall behind it 1.12, 5.63
and 11.23 dB off with the refracting tracer. Handing back the closed verdict in both directions
would have moved the type code of every exit and, through `far_closed`, the composite of a
glued exit, which is settled.

**A closed bare face hands on the direction the wave has.** `REPLACE_BY_VBS` had no branch for
a face it found closed, so a bare exit and a transition kept what `ray_mesh_interact` had
formed from the segment, where a sheet's closed outer pair handed on the direction of the wave.
Measured through the intersector on `eps 1.50 − 0.40j` entered from the lossless `eps 5.24`
solid the ray is launched in, invariant 1.01 to 1.45, the wave at 52.27° to 66.38°: with the
undeviated tracer the bare exit and the transition into a solid of `eps 1` wrote 32.94° to
40.47°, the Snell bend of the segment, and the coated exit the wave's; with the refracting
tracer they wrote 67.93° and 68.94° at 1.01 and 1.04 and the leg behind the transition carried
the bent flag. All three write the direction of the wave now, to 1e-12, both tracers. No power
of the event moved. With `tf` on the far medium the leak, −7.87 dB to −10.66 dB, continues in
that direction. What moved in the tables that existed: the bare-exit direction column of the
verdict table, ten rows, from the segment's 26.04° to 31.74° of that hand-fed row onto the
coated column beside it. The relaunched segment of the refracting tracer is §5. This is the
rule above an invariant of one, which is where it was measured; below one the face hands on
the direction the invariant has in air (§1, and the item on the wall behind a traced layer
below).

What moved with the two together in the tables that existed: the five recorded rows of the leak
table of the direction case and the ten rows of the leak table of the refracting-tracer case,
in their direction, 0.0014° and 9.5941° becoming 30.0000°; two gains in the latter, −8.128378 dB
to −8.129600 dB, the wood exit behind concrete now at the incidence angle; one current word,
`4001` to `0001`; two resolved types, 3 to 1; and the ten rows above. No other row moved, and
no EM row in its last digit.

One planned assertion is restated rather than weakened. "The refracting tracer's gain is the
undeviated tracer's, exactly" holds on one wall and is asserted there. Behind an air gap the
two relaunch from points built differently and the last digit can differ, 2.8e-14 dB worst,
which is printed and recorded (§5).

**A layered wall as a sum of paths.** Measured, and correct as shipped; no source change was
made. The convention, the factorization and the sizes are in §1. Two traced solids that share a
face, and the two halves of a host around a sheet, are each a series between what bounds them,
and the coupling between them is carried by the rays the tracer follows. Each series sits on its
own form at 8e-15 and 3e-16, the factorized form is the matrix of the whole wall at 7e-16
reference against reference, the first path is off the wall by the layer mirror and by nothing
else, at 0.000000 dB once that mirror is put in, and the paths are a geometric sequence whose
limit is the wall, at 6.5e-4 dB and 0.024° through the intersector and at 0.0016 dB through
`ray_progress` with 20 interactions, transmitted and reflected.

Handing the mirror behind a solid to the core as a layer, with the thickness code written for
every solid, was the candidate and was not taken; §1 has the three reasons and the measurement
three solids deep. Its quantization was measured all the same: the code the library writes is
the formula applied to the thickness the leg crossed on 15 rows, the decoded thickness is within
−1.3e-5 to +9.3e-5 of it, which on a round trip through the first solid is 0.04° at 1 GHz and
up to 2.4° at 60 GHz and on the second solid's series 0.005 dB and up to 0.21 dB, and the
relaunch offset the same leg already carries is 5e-6 to 8e-5.

One planned assertion is withdrawn rather than weakened. "A layered wall with every series
resolved on the characteristic matrix of the whole wall" describes something one path cannot
express: the reflect port of the first solid would need the second one's thickness before that
solid is crossed. It is asserted on the sum of the paths, for two solids. For a wall with a
sheet inside it the sum is asserted against host 47 | sheet | host 47 (§1).

**A declined series at a sheet is followed.** A reflection off a sheet flags the ray resolved
only where the host's series was summed at that event; §1 has the rule. Before, the flag was set
whatever the threshold said, and the reflected ray's next reflection was killed, so with the
series declined the bounces of a host that closes at a sheet were neither summed nor followed.
Measured through `ray_progress` on `air | host 100 mm | coating 6 mm | air` at the default
`thin_slab_threshold` of 0.15: one transmitted path where a bare wall gives seven to ten; for
the host `eps 1.50 − 0.40j` −0.09, −0.28, −0.42 and −0.08 dB in transmission at 10°, 40°, 60°
and 75° and up to −0.30 dB in reflection, with the reflected phase up to 7.9° off; for
`eps 1.50` −0.18, −0.35 and −0.31 dB and −0.33, −0.73 and −0.79 dB at 10°, 40° and 60°; with
every series declined +0.78 dB at 85° on the lossy host and +0.62 dB and +2.45 dB at 75° and 85°
on the lossless one. A wall with a sheet inside it was 0.03 dB to 0.72 dB from its own
all-resolved sum. The bare wall and the wall with every series resolved were at 0.0000 dB
throughout, which is what set the two apart.

All of it is asserted now: the word after the reflection, resolved exactly where the series was
summed; through the intersector the transmitted and the reflected paths of the declined wall as
geometric sequences with their limits on the matrix, at 2.3e-4 dB and, against the front-face
term, 8.6e-5; through `ray_progress` both thresholds within 0.0007 dB, and the clamp where the default sums the series, with nine to thirty paths where it is declined.
The followed sum is on the matrix without the clamp of §5: +0.78 dB before and −2e-6 dB now at
85° on the lossy host, where the summed wall carries +0.019 dB of clamp. That moved a power, by
decision: every figure above, onto the wall, and the number of rays behind and in front of a
coated wall with a declined series rises to that of a bare one. What moved in the tables that
existed: the `resolved` column of the 27 declined reflect rows of the lined-wall and the
layered-wall cases, 1 to 0, and with it three state-word assertions, 47 evaluations, restated
from `host + resolved` to `host`. No coefficient and no type code moved.

One planned assertion is restated. The reflected limit in dB holds at
1e-3 except where the wall reflects 10 dB less than its front face, 16 dB and 12 dB at 10° and
20° on the lossy host, where the same relaunch-offset distance reads 1.3e-3 dB; it is asserted
against the front-face term and at 5e-3 dB there.

**One verdict per body: a leg that begins at a declined reflection forms no series.** The
convention is in §1. Before, a ray reflected at a bare face with the series declined was an
ordinary ray, and the event that closed its next leg formed a series from the face ahead and
the previous-medium word, which a bare-face reflection does not write: the word still named
what the leg before the reflection was entered from or through. Measured on six legs - a far
and a near coating, both halves of a host around a sheet, both of two solids - the series sat
on the word's pair at 1e-16 on 384 rows and was 0.03 to 0.8 away from the true one, worth
−0.40 dB to +6.33 dB on a leg behind a coating and +0.37 dB to −5.88 dB on the second of two
solids. It reached a result where the true pair declined and the word's pair resolved at the
threshold; the reverse cannot occur, because a true pair that resolves was summed on the first
leg, and with the word made true by the fixture no followed leg summed a series on 320 rows.
At the default threshold of 0.15, through `ray_progress`, on a 5° grid, both tracers alike:

| wall | angles | `ρ` true / word | transmission, dB | reflection, dB |
| --- | --- | --- | --- | --- |
| far or near coating, host `eps 1.50 − 0.40j` | 70°, 75° | 0.109, 0.145 / 0.160, 0.201 | +0.13, +0.14 | far +0.06, +0.08; near −0.06, +0.09 |
| far or near coating, host `eps 1.50` | 65° | 0.135 / 0.175 | +0.25 | far +0.23; near −0.11 |
| `eps 5.24` in front of `eps 1.50 − 0.40j`, first solid | 10° to 30° | 0.121 to 0.146 / 0.158 to 0.195 | −0.045 to −0.075 | +0.165 to +0.176 |
| `eps 5.24` in front of `eps 1.50`, first solid | 10° to 30° | the same | −0.038 to −0.058 | +0.111 to +0.118 |
| the same, second solid | 55°, 60°, 65° | 0.091 to 0.141 / 0.162 to 0.191 | +0.118, +0.014, −0.194 | −0.194, −0.123, −0.012 |
| a sheet inside a wall; a bare wall | none | | | |

Now a reflection inside a medium that does not sum the series writes the followed mark into
the buffer word, and a marked leg forms none. The previous-medium word is not touched, so the
recovered cosine of a leg is unmoved by a back reflection at a bare face, bit for bit. That
moved a power, by decision: every figure in the table, onto the wall, with the followed paths
in place of the ones a wrongly resolved ray lost, 511 behind two solids at 10° where there were
384. Asserted: the six legs bitwise, the mark and the word after every reflection, and the
five walls through `ray_progress` at the default threshold against every series summed, to
5e-3 dB at every 5° from 10° to 85°, 1e-2 dB for the sheet inside a wall. What moved in the
tables that existed: 18 coefficient rows, all of them rows that recorded this, 16 in the
coupling table of the `ray_progress` measurement and 2 in the declined-series case, which is
asserted on every row now; 83 rows in which the buffer word printed after a declined reflection
reads the mark, with every coefficient identical; and seven word assertions restated from a
zero buffer word to the mark. The table that recorded the defect on the far-coated wall is
removed.

Two candidates were measured and not taken. Writing the mirror into the previous-medium word at
a bare-face reflection gives the same powers, to 5.8e-15 dB on 320 rows, but the recovery reads
that word: a named lossless solid would move by rounding, and an unflagged lossy solid needs
its name and cannot give the word up. And three residuals that had been put down to the stale
word were sorted: two solids at 10° and at 60° were it, to 0.0001 dB and 0.002 dB once
repaired; a sheet inside a wall was not, see the gain floor in §5.

**The verdict across a band.** Every frequency of a run takes the base frequency's verdict on
each series; the convention is in §1. Before, each frequency compared its own `ρ` with the
threshold while the paths were the base frequency's. Measured through `ray_progress` on a bare
100 mm wall of `eps 1.50 − 0.40j` at 1 GHz, undeviated tracer, with a second frequency in the
run and the followed mark already in place: at 0.5 GHz the other frequency was off the matrix
by −0.94 dB and −0.98 dB in transmission and +0.12 dB and +0.14 dB in reflection at 76° and
78°; at 0.25 GHz by −0.54 dB to −0.11 dB and +0.50 dB to +0.57 dB from 65° to 78°; at 4 GHz by
−1.10 dB and −1.26 dB at 80° and 82°; with the two exchanged by up to +0.96 dB and −1.35 dB.
Run alone, each frequency was at 0.0000 dB, and the base frequency of the band on every row. The window
is where the two fall on different sides of the threshold and widens with the band.

`ray_state_update` gained the two optional trailing arguments `verdict_inN` and `verdict_outN`,
and `ray_progress` hands the base frequency's verdict of each pass to the calls of the other
frequencies. A caller that passes neither gets what it always got; the per-ray state did not
grow. All 120 rows of the band table are on the matrix to 1e-4 dB, asserted at 5e-3 dB for
both tracers, and the argument itself is asserted bitwise on 120 rows through the intersector.
No row that existed moved: no multi-frequency fixture had been sitting in such a window.

**The reflect port of a sheet lies where its transmit port puts the layer.** The convention, its
three cases and its sites are in §1. Before, every reflect port was the layer's reflection at its
own front face handed to a ray that turns around at the mesh plane, whatever the transmit port
said about where the layer lies. For a glued entry the transmit port puts the layer in the air in
front, so the first reflection of a coated wall was behind its own echo by `2 k₀ cos θ·thk`;
for a sheet with one medium on both sides each reflect port put the layer behind the plane as
seen from its own side, which is no wall. Measured on 6 mm of `eps 5.24 − 0.63j`, every series
summed or not, both tracers, through the intersector and through `ray_progress`: a wall coated
on its near side reflected −0.62 dB to +0.87 dB and up to 8.4° off the matrix at 1 GHz, up to
1.11 dB at 6 GHz; a wall lined on both faces −2.06 dB to +1.00 dB at 1 GHz, −4.64 dB and
+11.77 dB at 3 GHz where the reflection has a notch, +5.97 dB at 6 GHz; a caller without state
the same, `ray_mesh_interact`'s own composite being the redo's to 2.4e-16. Mineral wool 50 mm
on a rigid wall reflected with the right magnitude and 25°, 99° and 395° behind the same wool
meshed as a traced solid, at 250 Hz, 1 kHz and 4 kHz and 20°. A sheet 30 mm in front of a wall
was on none of the three walls it could be: with its layer in front of the face the
transmitted sum was exact and the reflected one −1.79 dB to +1.45 dB off at 1 GHz, with it
behind the reflected sum was +0.03 dB to −0.66 dB off and the transmitted one up to −1.59 dB at
6 GHz. And a wall with a sheet inside it reflected as host 50 | sheet | host 44 and transmitted
0.05 dB to 0.08 dB above that. All of it is asserted now, on the wall of §1, at 5e-3 dB and
0.05° or the bounds the table in §1 gives.

Three things were measured and not taken. *The layer behind the plane as the ray first met it*:
put on by the fixture as the round trip on the reflections met from behind, it is exact for the
air gap and for a wall with a sheet inside it, and it is not one wall, breaking the equality of
the transmission from the two sides by up to 0.13 dB; it needs path history, which no word holds.
*The layer on the side of the face normal*: readable, and dependent on how a pane was exported.
*The coating in the first `thk` of the meshed solid*, so that the transmit port takes the solid
out: it moves the transmitted sum of every coated wall and divides by the host's loss; not
measured, rejected on the settled displacement side. One expectation was wrong and is recorded:
the transmission through a wall with the sheet off its middle was the same from either side
before the change as well, at 0.0000 dB; the old ports were symmetric, and wrong in one phase.

What moved. No power of a single event in air. The phase of every glued-entry reflection by
`2 k₀ cos θ·thk` and of every reflection off a sheet in air by half of that; inside the lossy
host also its magnitude, +0.18 dB to +0.24 dB for 6 mm. The summed reflection of every
near-coated and lined wall, and both sums of a sheet in front of a wall and of a wall with a
sheet inside it, onto their walls. About 700 rows in 19 tables that existed, every one a near
coating, a sheet inside a host or the `att` stack below, and about 370 rows in 19 more in the
last digit of a residual, 4.5e-16 at most; no table without a sheet. Four references were
restated: a sheet of air on a `tf`-bearing wall is the bare wall with its phase, where the
reference carried a round trip of `thk`; an air-like sheet on a mass-law sheet stands half of
`thk` to either side by layer order; the mirror behind the ray inside a host is centred; and
three measurement tables that printed the repair as a fixture column are referenced to the mesh
plane.

Two planned assertions are restated rather than weakened. "The summed reflected paths on the
matrix to 5e-3 dB and 0.05°" holds except where the wall reflects 10 dB less than its first
coating, one row at 0.021 dB and 0.12° that is 4.3e-4 of that first path, and on the rows that
carry the clamp of §5; both are asserted against the first-face term. And through
`ray_progress` at 6 GHz a lined lossless wall reads 8.9e-3 dB, the relaunch offset the
reference does not know, and is held at 1e-2 dB (§5).

**The echo of a sheet pays the sheet's interface gain.** The convention is in §1. Before, the
reflect port carried no gain at all, so the echo of a sheet was unpaid where the same echo of a
traced solid had paid on entry: 20 mm of `eps 5.24 − 0.63j` in air reflected 0.16 dB to 0.98 dB
more as a sheet than as a solid with 3 dB of `att` and 0.38 dB to 2.28 dB more with 10 dB,
1° to 16° apart, and the lossless counterpart up to 2.63 dB. The traced solid sits on the Airy
reflection whose echo carries `sqrt(G)` at 7e-5 dB, summed and followed, and the sheet on the
same form at 1e-9; the two representations meet at 1e-3 dB, the relaunch offset of the traced
leg. The transmit ports agreed before and after.

In the core the first form tried put a layer's gain on the forward crossing into it. That is
not the transmit port that is settled: every internal re-entry then pays again, and the
transmit port of two 6 mm layers with 3 dB and 2 dB moved by 0.005 to 0.034 in amplitude. It was
withdrawn for the second recursion of §1, which leaves the series and the transmit port
untouched. What moved: the reflect port of every coherent sheet and stack that carries `att`,
the mass-law route excepted, by the figures above for a thick sheet and, on that two-layer
stack, by 0.003 to 0.007 in amplitude with the 3 dB layer in front and by 0.041 to 0.048 with
the 2 dB layer in front. One assertion is restated: "the reflect port with and
without the sheet's `att`, bit-identical" was the old convention pinned, and is now "told
apart, and complex on the reference whose echo pays". "Equal to the equivalent traced solid to
1e-6 dB" describes something a traced leg cannot express and is asserted at 1e-3 dB against the
solid and at 1e-9 against the closed form.

**What a closed face hands the wall behind a traced layer.** The direction of the leak, which
for an invariant below one is its direction in air; the rule is in §1. Before, a closed face
handed on the direction the wave has in the medium it leaves, which was settled on a host
entered from a denser solid, invariant above one, and is unchanged there. Behind a layer
entered from air it handed a leaking wall the direction of the wave in the layer: for the
porous `eps 0.31 − 0.20j` 16.5°, 33.1°, 49.4°, 58.5°, 64.6°, 69.0°, 70.3° and 70.6° at 10°, 20°,
30°, 37°, 45°, 60°, 75° and 85°, for a dense layer an angle below the incidence one. Measured
through the intersector on 50 mm in front of a 200 mm rigid wall with `tf` 0.3 and `m` 30,
the traced solid against the sheet:

| | 10° | 20° | 30° | 37° | 45° | 60° | 75° | 85° |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| porous, tracer 5: mass term, dB | +0.23 | +1.00 | +2.48 | +3.69 | +4.35 | +2.88 | −2.30 | −11.62 |

and worst over the angles 16.5 dB behind lossless `eps 1.50` and 17.0 dB behind
`eps 1.50 − 0.40j`, the wall then left in the layer's direction, 21.5° to 33.3° off. With the
undeviated tracer the mass term was the sheet's to 3e-6 dB, because the term reads the
geometric ledger and the geometric cosine, the segment, and the direction written behind the
wall was 0.0008° to 0.0026°. Why, traced by running the wall | air event again with one input
changed: the direction handed in differed from the segment, so the leg counted as refracted
(`use_vbs`), the undeviated rule did not apply, and the exit was the Snell bend of the wall's
impedance mapping; handed the incidence direction, the same event held it. The
previous-medium word played no part. With the refracting tracer the segment was the layer's
direction, the cosines coincided, the wall held that direction and formed its mass term on it;
handing that tracer the incidence direction without moving the segment made it worse, a
direction along the normal with the mass term unchanged, which is why the segment follows. A
wall of `eps 0.20` with `tf` 0.5, closed from 26.6°, was 1.07 dB to 1.67 dB off on its exit
event. Through `ray_progress` with the default tracer the strongest path behind the solid was
up to 9.45 dB off the sheet's (0.5 GHz, 80°), the power sum of a frequency in a band up to
5.05 dB off the same frequency alone, the mass term being formed on the base frequency's
segment, and every frequency but the base one was written a direction along the normal.

All of it is on the sheet now, for the three layers, the three walls and both tracers: 1.3e-13°
in every direction and 4.6e-6 dB on the mass term and the exit event, a band on each frequency
alone to 1e-4 dB, each vertex ray of a tube in the direction it was launched in. Asserted in
the two closed-face cases (diagnostic infrastructure) and in section K of the lossy-verdict
case. It needed no state, no gate and no change to a type code: one lambda in
`REPLACE_BY_VBS`, used at its two closed-face sites, a relaunch on request in
`EXIT_DIRECTION`, and one test taken out of `RELAUNCH`'s tube loop that repeated its caller's
gate. It moved a power, by decision: every figure above; with the undeviated tracer the exit
event by at most 1.1e-3 dB and the phase of the leg in the wall, which is a leak's walk-off
form now where it was a refracted leg's. What moved in the tables that existed, 92 rows, none
of them asserted: the 36 rows of section E and the 16 of section K of the lossy-verdict case,
and the 40 `PW` rows of *ray_progress - a lossy medium has no critical angle and a band takes
one path*, in their power sums by at most 1.6e-3 dB and in their recorded coherent sums by up
to 2.5 dB transmitted and 6.1 dB reflected; with it each frequency of a band is on the
frequency alone in the coherent sum as well, where it was up to 2.6 dB and 3.6 dB off. No row
of the leak-direction case, the exits-and-verdicts case, the direction case or the trajectory
case moved.

The same rule only out of a lossy layer was measured and not taken: it moves the same rows and
leaves a lossless layer in front of a leaking wall at 16.5 dB and 30.6°.

One reading is corrected with it. A wall without the mass column reads +5.23 dB on its
wall | air event behind the traced solid and +6.56 dB behind the sheet, series summed, at every
angle. That is not the direction, which is worth 0.001 dB of it: the series of the bare wall
is 10.4537 dB, behind the solid 10.4525 dB and behind the sheet 11.7876 dB at 10°. It is the
mirror behind the ray, a half-space behind a solid and a layer backed by air behind a sheet
(*A layered wall is a sum of paths*), and it stands.

**The verdict at a lossy second medium.** The magnitude test is a critical angle only where
the second medium is lossless; the rule is in §1. Before, a lossy medium rarer than its
neighbour closed where the test tripped, 37.40° for `eps 0.31 − 0.20j` in air. Measured on
both sides of that angle, scalar: a porous sheet on a rigid wall with `tf` 0.3 passed −7.10,
−6.43, −5.83 and −6.26 dB at 10°, 20°, 30° and 37.39° and nothing from 37.41°, where the chain
continues from −6.26 dB to −19.55 dB at 85°; with `tf` 0.2 on the sheet it stepped by 6.7 dB.
A dense lossy sheet glued on the porous medium passed nothing where the analytic layer gives
−2.59 dB to −9.74 dB, and wrote the incidence direction where the wave runs at 58.9° to 70.6°.
The porous medium as a traced solid was a mirror from that angle, `|R|² = 1` where the
half-space gives 0.318 to 0.944, so a room lost the absorption of an absorber meshed as a body:
the sum of its paths read +8.35 dB to +0.69 dB above the analytic slab on the reflect side and
nothing on the transmit side. One vertex ray past the angle closed a whole ray tube. In EM both
channels of `air | porous` and of `eps 5.24 | eps 1.50 − 0.40j` were zero, where the half-space
gives −1.66/−0.36 dB to −12.5/−10.2 dB and −2.49/−0.44 dB to −13.6/−11.3 dB; the EM reflect
port kept the Fresnel magnitude, the scalar one did not. A coating of `eps 1.50 − 0.40j` with
`tf` 0.5 on `eps 5.24` stepped by +1.5 dB (TE) and −0.5 dB (TM) at 32.98° and ran up to 4 dB
off. Through `ray_progress`, a frequency that is open alone lost its transmitted path in a band
whose base frequency was closed. All of it is on its reference now, at the bounds of the table
in §1. The sheet in air never read the verdict and did not move. Asserted in *Lossy verdict - a
lossy second medium has no critical angle* and *ray_progress - a lossy medium has no critical
angle and a band takes one path*; the porous rows of the leak-chain table run to 70° where they
stopped at 20°. No row that existed moved with the rule or with the tube test.

Replacing the test by the evanescence of the refraction root's argument everywhere, which
would remove both spellings, was measured and not taken. It gives the same results for a lossy
second medium and also opens a lossy first medium at a real angle: 71 rows moved, 15 assertions
of the recovery-gate case failed on their hit codes (33 to 1, 101 to 69), the stateless
`medium | air` rows of the transmission-convention baseline went from zero to 0.12 to 0.28,
and on a host entered from a denser solid, invariant 1.10 to 1.45, where the face is closed,
`ray_mesh_interact` bent the relaunched segment from 55.3° to 70.7° and the leg behind a
transition took the bent flag. See §5.

**The recovery runs until it has converged.** Newton until `|Δu| ≤ 1e-15·u`, at least three
steps, at most fifty; §1. With three steps the porous solid, once open past 37.40°, was off the
analytic slab by 3e-4 dB at 45°, 0.013 dB at 60°, +0.16 dB at 75° and +2.25 dB at 85° and left
at 79.9° for 85°; below the angle it was converged, exit direction within 9e-6°, in three
media. Now every path is on the slab to 7.4e-4 dB and the exit direction is the incidence one.
What moved in the tables that existed: 81 rows in 22 tables, in residual columns at 3e-15 or
below (7e-14 in *mirror behind, every path*) or in the sign of a printed zero; and the rows
above an invariant of 1.15 of the exits-and-verdicts case, whose residual fell from up to
9.5e-9 to 1e-16, with the ninth digit of nine printed coefficients. Three assertions that
pinned the event to the invariant three steps give were restated against the reference at the
invariant, and the gate that exempted the rows above 1.35 from the 1e-9 bound was removed. The
run-time cost of the longer solve is not measured.

**Fourteen branches that had a guard and no row.** Measured, and correct as shipped with the
state kept; no source change was made. Each is now entered by a case of its own, shown on a
marker build, in the nine *Unentered branches* cases of §1. Six of them were entered before and
asserted nowhere: float, which `ray_progress` runs two million sheet events at; the corner,
which one case entered once and checked for a finite gain; `ray_mesh_interact`'s own composite
under a closed outer pair and with lumped columns, which the redo discards; and the stack of a
mass-law and a coherent layer, asserted against its own statement. What each case found:

- *A corner with a sheet.* The redo is the first face alone, bit for bit, whatever the second
  face is. What `ray_mesh_interact` wrote itself was not, and is since
  (*`ray_mesh_interact`'s own result*, below).
- *The displacement budget.* Both ports and the mirror of the host's series sit on the
  reference with the cap the event has, below it and above it, for a sheet up to three times
  as thick as its host, which stays finite and passes less than it received. The same for an
  ordered stack. What the cap is has changed since; see *A sheet takes no more host than lies
  on each side of it*, below.
- *The core's loss cap.* Reached by an opaque layer, 46.05 Np one way, and by nothing a
  propagating wave does on those rows. The reflect port is the uncapped reference at every
  loss; the transmit port is the reference below the cap and stands on the floor above it; an
  opaque lining as the mirror of a host's series is the half-space of its material.
- *`ray_mesh_interact`'s own glued composite.* The redo's, at 6.7e-16: from air under a closed
  outer pair on both ports, for a sheet entered as a leak and by an ordinary crossing, with
  `att`, `tf` and a resonance on the sheet and on the wall behind it; and for a glued exit of
  a ray launched in a lossless solid, below and past the critical angle. Under a tube edge it
  was not, and is since (*`ray_mesh_interact`'s own result*, below).
- *Float.* Every hit code, resolved type and state word of the double chain, and every
  coefficient within 3.1e-6.
- *The plane of a mixed stack.* The stack and the same laminate meshed as a traced solid with
  the mass-law layer as a lining are one wall on the reflect port from either side and in the
  phase of the transmit port, and since the mass term took the cosine in air on the transmit
  level as well, from either side. The two sides of that wall differed from each other then,
  stack and traced laminate alike, which the closed layer, further down, repaired.
- *The followed mark.* Carried across the shared face of two abutting blocks, so a wall meshed
  as two blocks is the wall meshed as one, to the relaunch offsets. Replaced by the medium of a
  nested object, after which the leg judges for itself; §5.
- *A resolved ray into a deferred medium.* Takes the rewrite like any other ray and forms no
  series.
- *A verdict handed in.* On a leg flagged not parallel it sums where it says. Near a pole the
  denominator is clamped at 1e-2 and the series is exactly 100 in magnitude.
- *A sheet without the mass column of its mass-law host's permittivity.* Merged: the host is
  one body of its whole thickness.

Planned assertions that were withdrawn or restated rather than weakened. "The cap engages from
80 % of the host on" described the refracting tracer only, under a cap that is gone; a sheet
that fits is not capped. "The summed paths of a capped sheet against the wall it
stands for" was replaced by the capped port over the uncapped one, which is the same distance
without a path sum. "The own glued exit on the redo, complex" cannot hold, the redo also
closing the host leg; it is asserted on every magnitude and on one common factor for both
channels. "The laminate's transmit port at 5e-3 dB from the solid's side" did not hold while a
lining behind a solid formed its mass term at the cosine in that solid; it holds now and is
asserted raw. "The overlap crossing on the abutting
one in phase" is asserted at normal incidence, the polylines of the refracting tracer differing
inside the overlap (§5). And "a mirror above one is capped at one" was not reached: the largest
mirror of a sheet inside its host on those rows is 0.864.

What moved in the tables that existed: nothing when the cases landed; 27 rows at 2.8e-16 in
the five tables §1 names when one table was added, with `src/` untouched.

**The cosine of a mass term behind a refracting layer.** It is the cosine the wave has in air; the
rule, its gate and where the cosine comes from are in §1. Before, a mass-law sheet reached from
inside a traced solid formed its term at the cosine the wave has in that solid. Measured through
the intersector on a 26.8 dB sheet (22 mm, `tf` 0.2, `m` 20), as the mass paid above the term at
the incidence cosine, at 10°, 30° and 60°, both tracers alike: +0.083, +0.815 and +4.562 dB behind
`eps 2.50 − 0.80j`, +0.080, +0.790 and +4.447 dB behind `eps 2.50`, and −0.232, −2.470 and
−2.842 dB behind the porous `eps 0.31 − 0.20j`, where the wave bends away from the normal; the
same figures on the far face of the host, for the sheet inside it and for a stack with the
mass-law layer inside it in either order. On a wedge with the exit face tilted by 10°, +0.03,
+0.13 and +1.22 dB. In a host entered from the lossless `eps 5.24` solid, 1.34, 6.35 and 15.07 dB
at an invariant of 0.50, 0.90 and 0.99 for `eps 1.50 − 0.40j` and 2.03, 7.98 and 17.00 dB for
`eps 2.50`. The mass term stepped by those amounts where a cover ends, and the laminate of a
solid and a lining paid differently from its two sides. All of it is on the term at the cosine in
air now, at 0.000000 dB, asserted at 1e-6 dB in the mass-cosine case; the direction behind the
wall and the hit point did not move, and the laminate's raw transmit column is on the stack at
1.5e-4 dB from either side.

It needed no state: one cosine formed in `SHEET_RESOLVE`, and a trailing `mass_cos` on
`qd_slab_resolve`, as the core had. The in-sheet path for `alpha` and the loss, the plate's
running sum, the thickness code and the interface amplitudes read other quantities and did not
move; the tangency gate of the term moved with its cosine. That moved a power, by decision: every
figure above. What moved in the tables that existed: the 24 rows of the laminate table and
nothing else, the 12 rows met from the solid's side in their transmit column, from 0.061 to
4.562 dB onto −0.000154 to +0.000022 dB, and the 12 rows met from the lining's side in the
recorded `echo_dB`, the echo that leaves through the lining from inside the solid, by 0.06 to
4.56 dB. Every row whose mass-law sheet is reached from air or from inside a mass-law medium is
identical, as is every row at an invariant of one or more.

One planned assertion is restated rather than weakened. "The traced laminate equal from its two
sides" holds for the mass paid, to 0.000000 dB, and is asserted so. For the whole transmission
it described something the code did not do then, for a reason that was not the mass term: the
faces of a rigid mass-law layer next to a refracting layer, closed in the next entry. The
two-sided total is a recorded column there and reads the flux splitting now. Three options were weighed and not taken: the cosine in the host for a sheet with
the host on both faces, which is smooth through an invariant of one and steps the term where a
cover ends; a zero cosine from an invariant of one on, which is continuous and lets every ray
that entered a cover through its end pass the plate on its `tf` leak alone; and a real-angle map
in `ray_mesh_interact`'s own composite, which it has since for types 2 and 5.

**The faces of a closed mass-law layer.** A mass-law layer in which the wave is evanescent is
closed on both faces, each passes the flat leak of a bare face, and the leak is a statement
about normal flux; the rule, its two forms and what it does not cover are in §1. Before, both
solvers took the flat leak only where the Fresnel reflectance was within 1e-9 of one, and put
it into a chain of raw amplitudes that was converted to flux once, across the layer. Measured
through the intersector on `C | M`, C 10 mm of `eps 2.50 − jε″`, M a rigid sheet 22 mm with `tf`
0.2, without the mass term, at 10°, the stack and the laminate with C a traced solid alike:

| ε″ of C | C met first, before | M met first, before | now, either side |
| --- | --- | --- | --- |
| 0.8 | +3.18 dB | −11.90 dB | −13.48 dB, −13.37 dB |
| 0.1 | +12.01 dB | −3.06 dB | −12.61 dB |
| 1e-3 | +32.00 dB | +16.92 dB | −12.48 dB |
| 1e-6 | +62.00 dB | +46.92 dB | −12.48 dB |
| 0 | −14.51 dB | −10.45 dB | −12.48 dB |

Three defects were in it. *The gate*: next to a lossy layer the reflectance of the closed face
is 0.936, the face counted as open, and `(1 − R') / (1 − R0)` multiplied a raw `|t|²` of 3.83,
ten dB per decade of the loss with a step at zero loss; a rigid sheet inside
`eps 2.50 − 0.80j` read +3.69, +5.67 and +6.10 dB on its event at 10°, 30° and 60° and +59.9 dB
at a loss of 1e-3. *The flux*: also between lossless media the two sides were
`ε·cos²θ_C / cos²θ` apart, 4.06, 4.77 and 8.45 dB, and next to a lossy layer `tan²θ`, 15.07,
4.77 and 4.77 dB; a rigid lining on lossless `eps 2.50` read −16.01 dB on the far face and
−11.95 dB on the near face where its two leaks are −13.98 dB. *No `tf`*: the scattering form
returned the raw amplitudes of a boundary neither of whose media carries `tf` before any gate,
so a rigid layer without `tf` in a stack transmitted −4.80, +4.07 and +5.89 dB next to the lossy
layer and −3.53, +5.56 and +7.39 dB next to a lossless one, where the same layer as a sheet in
air and as a lining passes nothing; inside a lossy host produce mode passed −8.15, +0.07 and
+3.03 dB. `ray_mesh_interact`'s own glued exit behind the lossy solid, at the real angle of its
segment, read +69.85 dB and +78.09 dB at 10° and 30°. A rigid layer stacked with a mass-law
layer of `eps 1.5` or `2.5` was 1.81 dB to 8.45 dB apart between its orders.

All of it is on the rule now, asserted in the closed-plate case (diagnostic infrastructure):
below one at every loss, −11.92 dB at most over 360 levels; a loss of 1e-6 on a loss of zero;
the two sides on the flux splitting of the coherent layer at 7.6e-15 dB for the stack and
1.6e-6 dB for the traced laminate; the stack on the traced laminate at 3.7e-4 dB; a lining on
the near face on its two leaks at 1e-9 dB from both public functions; nothing through a layer
without `tf`. It needed no state, and no decision that reaches a state word: one lambda and a
flux ratio in `qd_stack_resolve`, one block in the mass-law route of `qd_slab_resolve`. Since
then a closed layer seen from a coherent side is the pad of a closed face, the levels next to
a lossy layer are those of §1, and the two sides are equal exactly and no longer apart by the
flux splitting (*`tf` at a closed face*, below).

It moved a power, by decision: every figure above. What moved in the tables that existed, 214
rows, none of them asserted: the 132 rows of table A of the mass-cosine case in `rest`, by 2.03
to 26.68 dB, which that table's header calls what a change to the mass term must not move and
which a change to the faces does move; the 24 rows of its table B in `d_sides`, onto 0.0000 to
0.2012 dB; 26 rows of its table D in the level of the sheet event; the 24 rows of the laminate
table in the stack's level, 12 of them also in the recorded echo through the lining; and 8 rows
between rigid bodies, the stack `B20 60 | S30 T20 | B20 40` of the plate-depth case at 30° in
`rest`, where one sheet of the stack is closed and the other is not: −31.71 dB forward and
−21.65 dB reverse before, −26.68 dB from both sides now. The mass paid is identical on every
row, 50 cells changing the sign of a printed zero. Every row whose mass-law layer is not closed,
every closed layer between two lossless media that are one medium, and every row that carries
neither `tf` nor the mass column is identical; the 60 twin rows of table E were identical with
every digit across the change.

One planned assertion is restated rather than weakened. "One transmission from the two sides"
holds exactly where the coherent layer is lossless, and is asserted so. Where it is lossy the
two sides differ by `sec²(arg η)` of that layer per channel, which the entry and the exit of a
traced lossy solid carry whatever the sheet does (§5); it is asserted against that closed form
at 1e-6 dB and not against zero.

Three options were measured or weighed and not taken. *The leak weighted by the Fresnel
reflectance*, `R0·(1 − R'(1))`: bounded and continuous as well, 0.29, 0.87 and 1.68 dB lower at
10°, 30° and 60° next to `eps 2.50 − 0.80j`; not taken because the leak chain and a bare face
pass the unweighted leak, and a lining would then differ from the same face reached through
them. *The rule only where the neighbour carries no mass column*: it leaves the 8 plate-depth
rows where they were, 10.06 dB apart between the two directions, and the stack of a rigid and a
non-rigid mass-law layer two-sided. *The magnitude of the tilted admittance in place of its real
part*: by the algebra the stack would be one wall for a lossy layer as well; not measured and
not taken, because it leaves the flux convention of every other crossing and parts the stack
from the traced laminate.

Found beside it: a rigid lining on a rigid traced wall, closed in the next entry, which also
took the stack `B20 60 | S30 T20 | B20 40` off the flux ratio named above and onto the chain
of held crossings; a coherent sheet with `tf` rarer than its lossy host, closed further down; and the step
where a mass-law layer becomes closed, §5.

**A mass-law sheet behind a held wave.** A mass-law sheet or an all-mass-law stack reached by a
wave that is held in a mass-law medium, towards the lower index, is the chain of held crossings
at the incidence cosine, with the held reflection on its reflect port; the rule and its gates
are in §1. Before, the mass-law route entered the sheet at the Snell cosine of two impedance
mappings and applied the near-to-far conversion of wall | air. Measured by a marker on a wood
wall with a concrete lining: at 30° the cosine in the sheet was `−1.118j`, both raw
reflectances one, the two faces the flat leaks 0.28 and 0.1, and the conversion 7855, an event
of +23.42 dB. What it was, levels without the mass term:

| geometry | before | now |
| --- | --- | --- |
| W 100 \| K lining, wall first, 10° / 19.4° / 19.6° / 30° / 60° / 75° | −18.06 / −21.42 / +16.07 / +16.43 / +18.82 / +21.68 dB | −17.86 / −17.92 / −17.92 / −18.04 / −18.90 / −19.99 dB |
| T 100 \| K lining at 30.1°, W 100 \| T lining at 60° | +16.19 dB, +21.16 dB | −18.75 dB, −15.78 dB |
| the same with types 1 and 2, VV and HH alike, 30° / 60° | +16.43 / +18.82 dB | −18.81 / −23.49 dB VV, −17.00 / −14.72 dB HH |
| W 50 \| K sheet \| W 50, 10° / 20° / 30° / 60° | 0.39 / 9.17 / 8.95 / 7.23 dB below three solids | on them |
| W 50 \| K T \| W 50, 30° / 60° | 10.40 / 5.30 dB below four solids, the two orders 3.20 dB apart at 60° | on them, one wall in both orders |
| the reflect port of the lining, 10° / 30° / 60° | −6.59 / −1.43 / −1.43 dB, a mirror of 0.72 past the angle | −7.36 / −6.67 / −4.48 dB, the twin's face |

Below the critical angle of the pair the wall-first event was already off the other side, by
0.20 dB at 10° and 3.50 dB at 19.4°, and stepped by 37.5 dB at the angle. The limitation that
used to be recorded for the stack, a flux ratio formed from the admittances of rigid bodies
whose level had no reference, is gone with it: `B20 60 | S30 T20 | B20 40` read −26.68 dB from
both sides at 30° but −23.48 against −25.51 dB at 60° and −23.72 against −25.24 dB at 75°, and
reads −16.34, −18.54 and −21.70 dB from both sides now.

It needed no state and reached no state word, type code or direction: one block in the
mass-law route of `qd_slab_resolve`, one in `qd_stack_resolve` with a trailing argument that
says the wave is held, and that argument at the one call in `SHEET_RESOLVE`.
`Material::interact_with` is unchanged. Asserted in the held-lining case, 1280 assertions of
which 367 failed before the change.

It moved a power, by decision. In the tables that existed, 218 rows, none asserted: the six
wall-first W | K rows of table F2 of the closed-plate case; 168 rows of table A of the
plate-depth case in `rest`, 7 stacks, by 0.24 to 37.1 dB, among them 36 rows that stood above
0 dB (`lining L20 > B20 60 | S20 | B20 40` reversed at +18.24, +12.29 and +15.15 dB); and 44
rows of the `ray_progress` plate-depth case in the path count `nT`, by 1 to 6. The mass paid is
identical on every row. Table E of the closed-plate case, the leak-reciprocity,
leak-direction, bonded-mass and refracting-tracer cases did not move, and the 78 twin rows of
table G were identical with every digit across the change.

Three options were measured and not taken. *Lifting the exclusion of the closed-layer rule*,
two flat leaks without the conversion: −15.53 dB on the event at 30°, 4.48 dB and 3.62 dB
below the other side at 30° and 60°, the wave being treated as refracted and then turned into
a leak. *The rule towards either index*: it puts a lining towards the higher index on its
solid twin as well, and moves table E of the closed-plate case by up to 0.0047 dB, 36
leak-direction rows by up to 0.007 dB and two refracting-tracer rows by 0.0012 dB; not taken
because the residue is an acoustic one of that size and those rows were to stay (issue 4
holds what is left). *The rule on the transmit port only*: the near face then passes 0.785 and
reflects 0.72 at 30°.

**The geometry of an ordered stack.** Two co-located sheets of different material leave the
ray on the line it arrived on: the spine, the three vertex rays of a tube and
`ray_mesh_interact`'s own `path_dirN`, with every type; the rule and where it sits are in §1.
Before, the geometry came out of the interface call of the legacy path a stack falls through
to, and that call sees a pair of media chosen by the two normals, which is not near | far.
Shown with a marker, through the intersector, double, C1 `eps 5.24 − 0.63j` 6 mm and C2
`eps 1.50 − 0.40j` 6 mm, plain EM materials, the segment with types 2 and 5 at 30° / 60°:

| normals, first \| second | the pair the call saw | C1 \| C2 | C2 \| C1 |
| --- | --- | --- | --- |
| front \| back, outward | second layer into first | 15.73° / 27.98° | 65.38° / 84.84° |
| front \| front | air into first layer | 12.57° / 22.13° | 23.58° / 43.40° |
| back \| back | first layer into air | 30° / 60°, total-reflection bit | 38.15° / 60° |
| back \| front, inward | first layer into second | 65.38° / 84.84° | 15.73° / 27.98° |

The direction `ray_state_update` wrote was right on every row, and it never moved that segment:
near and far being one medium, it hands the direction on and relaunches nothing. Each vertex
ray of a tube was turned by 14.2° to 38.4°. With types 1 and 4 the segment and the tube were
right and the own `path_dirN` was bent, 27.98° for 60°, which a caller without state was
handed. A rigid mass-law sheet in front was not affected in any orientation: air into it is
closed, and as the first medium of a pair it holds the wave. That was not general: a mass-law
sheet that is not rigid bent to 24.09° / 45.00° on two front hits, and a rigid sheet with `tf`
and no mass column left along the normal, 0.004°, on a back hit.

What it was worth, types 2 and 5 alike. A 50 mm `eps 2.50` wall 0.5 m behind a stack in air
was met 0.148 m and 0.600 m off at 30° and 60° behind C1 | C2 and 0.802 m and 4.676 m off
behind C2 | C1, normals outward, entered at 9.87° / 17.26° and 35.10° / 39.04° where 18.43° /
33.21° is right, with +0.15 dB / +1.36 dB and −2.00 dB / −9.92 dB on the chain. Through
`ray_progress`, float, the same figures, and the error was permanent: the bare entry of the
next wall takes the segment, so `path_dir` itself became 15.73°, at every frequency of a band;
the path behind the wall was off by +0.08 dB to −12.88 dB at 1 GHz. A ray reflected back by
the wall crossed the stack from behind in the mirrored branch. Inside 22 mm of lossless
`eps 2.50` the segment behind the stack ran at 9.87° / 17.27° or 35.58° / 75.63° against
18.43° / 33.21° written, the exit did not repair it (15.73° / 28.00° travelled, 30° / 60°
written, resolved type 3), and the chain was up to 3.17 dB off the undeviated tracer's. Inside
lossy `eps 2.50 − 0.80j` the exit put the segment back, and the wall behind was met 0.8 mm to
27 mm off. A beam split in air behind the stack had its four sub-beams at 15.67° to 15.85°
where the undeviated tracer has 29.88° to 30.25°. The reflect pass, a single sheet and a
collapsed duplicate were not affected.

All of it is asserted now, at 1e-9 through the intersector in double, where the worst figure
is 2.2e-16 in air and 1.3e-15 m at the next wall, and at 1e-6 through `ray_progress`, where it
is 1.2e-7 and 6e-4 dB against the undeviated tracer; the split beam is the undeviated
tracer's. The change is one flag and one term in three conditions.

What moved with it. The stack event itself is untouched: hit code, resolved type, the three
words, the redo's coefficient and both ledgers are bit-identical, and with types 1 and 4
nothing moved anywhere. The event behind a stack closes another leg and moved with it: its
coefficient, which is the field at the point the polyline leaves at, and the VBS bit of its
resolved type, 1 where it was 3 (§1). Of the baseline, 20 rows, all of them stacks with type
5: the 18 rows *stack C|M inside* of table A of the mass-cosine case, whose `seg` went from
−5.7°, −17.4° and −37.9° in the lossless host to rounding, `dy` from 0.7 mm to 26 mm to
1.1e-16 m and `rest` by up to 1.4e-4 dB, the mass paid staying where it was; and two rows of
the held-lining case in the sign of a printed zero. The own EM coefficient of a stack with
type 2 moved by decision: its basis follows the outgoing direction, so it is type 1's, where
before an entry differed by 0.056 to 0.233 on a ray out of the horizontal plane. Keeping the
old basis was measured, bit-identical, and not taken: with a straight segment it would have
been the basis of a direction the ray does not travel in.

Two options were measured and not taken. *The near | far interface call for the stack*, as a
sheet and a duplicate get their geometry: the same result to 7e-15, nine lines and a second
interface call per event against one line. *The one flag for a sheet and a duplicate as
well*: one rule for the three cases at the same size, and 17 further baseline rows of single
sheets moved by at most 4e-16; not taken because a single sheet was to stay bit for bit.
Moving the segment in `ray_state_update` instead was not built: it would reach only a caller
that hands the geometry in, and the tube would need a second construction.

One assertion was restated, one withdrawn in part. Behind a glued entry the stack is asserted
on the line of the segment it was reached on, at 1e-9, and on the direction written only at
2e-4: that leg carries the glued entry's anchoring (§5). "The exit behind a stack carries no
VBS bit" holds for a bare-entered host and is withdrawn behind a glued entry, where the
single-sheet twin carries the bit as well (§5); there the exit is asserted on the twin's
type.

What it left then, the own coefficient of a stack and the total-reflection bit of its hit
code, is closed (*`ray_mesh_interact`'s own result*, below); the `no_bend` flag went with it.

**`tf` at the face of a coherent sheet.** It is an ideal transformer of real ratio, in produce
mode and in the core; the rule, what it gives and its sites are in §1 (*The lumped material
columns in produce mode*). Before, both solvers scaled the reflection of a face to the pair
rule's `R'` and its transmission by `(1 − R') / (1 − R0)`, with a second form for a sheet
rarer than the transverse invariant and a gate on the invariant between the two. Three things
created energy. *The quotient*: next to a lossy medium `1 − R0` is not the flux a face passes,
and close to the invariant it is small while the sheet still passes the wave. *The forcing
factor*: `tf < 0` raised a mirror at the phase of a complex reflection, which tuned or detuned
the cavity of the sheet by a phase no medium defines. *The lowered mirror of a tunnelling
face*: a face with an evanescent wave behind it reflects everything, and a rule that lowers it
and keeps its coupling is saved only by the symmetry of two equal faces. Measured through the
intersector, `T + R` of one sheet event, the worst over 1 to 50 mm and 0° to 85°:

| | before | now |
| --- | --- | --- |
| lossy sheet in air, `tf > 0` | 2.44 (`eps 0.31 − 0.20j`, `tf` 0.9, 1 mm, 37°) | 0.9998 |
| lossy sheet in air, `tf < 0` | 1.27 scalar (`0.90 − 0.20j`, 50 mm, 0°), 1.80 TM (`5.24 − 0.63j`, 25 mm, 67°) | 0.9998 |
| lossy sheet inside `eps 2.50` | 2.55 | 0.9993 |
| lossy sheet as a lining of `eps 2.50` | 2.51 | 1.0000 |
| lossless sheet as a lining of `eps 2.50`, `tf` on the sheet / on the host | 1.16 / 1.60 (`eps 0.90`, 1 mm, 73°) | 1.000000 |
| lossless sheet inside a lossless host, past its critical angle | 0.91 | 1.000000 |
| ordered stack in air, lossless | 1.63 | 1.000000 |
| step where a lossy sheet becomes rarer than the invariant, transmit port | −2.44, −2.97, −3.20 dB | 0.0024 dB at most |
| step at the critical angle of a lossless sheet in a lossy host | −0.146 dB | +0.0017 dB |
| every path through `ray_progress`, a sheet with `tf` in air, on a host and inside it | 2.59 | 1.00002 |

Why a transformer and not a better transmission factor: no rule that keeps the reflectance of
every face at the pair rule's value can be passive where a face has an evanescent wave behind
it, and three such forms were measured on a what-if before the choice - the form of the
tunnelling face taken at every face, which closed `tf > 0` between equal media and nothing
else; the pair rule applied to the squared complex reflection, which closed both signs between
equal media and left every lining at 1.05 to 1.60; and the transformer, which left no row
above one wherever the wave comes from a lossless medium. It needed no state and nothing that
reaches a state word: one function, now `Material::tf_face`, called from produce mode's two face lambdas and from `BND`.
The gate on the invariant is gone from both solvers, `apply_tf_pair` lost its clamp argument,
and the file is twelve lines shorter.

It moved powers, by decision. Against the tables that existed, 1480 rows in 36 tables, all with
`tf` on the sheet or on a neighbour: lossless sheets below their critical angle in rounding;
dense lossy sheets with `tf > 0` by up to 0.2 dB (`eps 5.24 − 0.63j`), 0.8 dB (`1.50 − 0.40j`)
and 1.4 dB (`5.24 − 5j`) on the transmit port; lossy sheets with `tf < 0` by many dB, 5.7 dB
in TM at 6 mm for `eps 5.24 − 0.63j`, where a 1 mm sheet that loses 0.6 % per pass used to be
put up to 22 dB off resonance by the phase of its forced mirrors; tunnelling sheets with `tf`
as in §1. Under a closed outer pair the reflect port moved with the near face, 2.8 dB and
3.3 dB for a lossy and a lossless rare sheet with `tf` 0.5 on a rigid wall, and the transmit
port of the lossy one by 0.20 dB through the series of the leak chain. No row of a material
with neither `tf` nor the mass column moved beyond rounding, and no mass-law row.

Asserted in the two sheet-tf cases (diagnostic infrastructure), and in the references of the
older cases, which carry the rule through `diag_tf_face`. Three planned assertions were restated
rather than weakened. *The two ports together inside a lossy host* are recorded: next to a
lossy medium the incident and the reflected wave interfere, the event with its host leg reads
up to 1.233 with a lossless forced sheet, and each port is asserted instead. *`tf = +1` closing
the reflect port* is asserted exactly for a lossless sheet denser than its surroundings and
bounded elsewhere (§5). *Per channel through the tracer* is asserted with `tf` on the sheet;
with `tf` on a traced host the wall of that case is bounded on the TE/TM mean, which the entry
on the bare boundary below has since made a weaker bound than the code meets.

Found beside it: the bare boundary of a traced solid with `tf`, closed below; the two ports
under a closed outer pair, closed further down (*`tf` at a closed face*); and the limitations
recorded in §5.

**A coherent sheet without `tf` that is rarer than the invariant.** It tunnels, on the
characteristic matrix. Produce mode read `1 − R0 < 1e-9` on a face as "no forward port" and
returned the flat leak, which without `tf` is zero: a rule written for a face whose far side
carries nothing, taken by the face of a thin layer. Which faces took it: next to a lossless
medium every one, `1 − R0` being exactly zero or 1.1e-16 to 2.2e-16 in TE, TM and scalar; next
to a lossy medium none in TE, where `1 − R0` is the host's loss, and every one in TM, where the
raw reflectance exceeds one, `1 − R0` −0.178 and −0.093 at 45° and 60° for ε″ 0.8 and −2.1e-4
for 1e-3. S is 6 mm of lossless `eps 0.31`; EM materials, no `tf`, no mass column:

| S without `tf` | before | now, the matrix |
| --- | --- | --- |
| inside `eps 2.50`, 45°, transmit / reflect | nothing / 0.00 dB | −0.041 / −20.27 dB (TE), −0.027 / −22.05 dB (TM) |
| inside `eps 2.50 − 0.80j`, TM | nothing / −0.20 dB | −0.147 / −22.13 dB |
| alone in air, 45° / 60° | nothing | −0.016 / −0.033 dB (TE), −0.020 / −0.152 dB (TM) |
| lining on the near face, ε″ 0.8 to 0, TE | nothing / 0.00 dB | −0.68 to −0.53 dB / −8.39 to −9.36 dB |
| lining on the far face, ε″ 0.8, transmit | nothing | −1.03 dB (TE), −0.51 dB (TM) |
| the same, reflect, TM | −0.51 dB | −15.62 dB |
| against the stack of its halves, in air, 45° | nothing against −0.016 dB | one layer |
| on a rigid wall without `tf`, reflect, 45° | unit magnitude, 0.41 (TE) and 0.85 (TM) off in the complex plane | on the matrix |

The host's series already read the sheet as the tunnelling layer, through the core, to 2e-6 on
both legs, so the series summed against a mirror of 0.097 while the sheet event reflected
1.000; and `tf` was discontinuous at zero, `tf` 1e-9 reading −0.041 dB where none read nothing.

The rule: a face neither of whose media carries `tf` is the raw Fresnel face, on the raw
reflectance, in produce mode as in the core; it needed no state. The figures of this entry
for a sheet or a host with `tf` were measured under the rule the faces had then and are
superseded by the entry above; those for a material with neither `tf` nor the mass column
stand.

It moved a power, by decision. What moved in the tables that existed, 150 rows. Of a material
with neither `tf` nor the mass column, 25: three rows of table A2 of the rare-sheet case,
twenty of its table F and two of table F3 of the closed-plate case, from nothing to the
matrix. Of a sheet or host with `tf`, in TM next to a lossy host, 120: 74 rows of table A of
the rare-sheet case, the transmit port by up to 0.43 dB, the reflect port by up to 0.46 dB,
the sum by up to +0.031 and at most 0.949; 34 rows of the tracer's rare-sheet case, HH by up
to 0.38 dB; twelve more of A2, by up to 0.083 dB. They are the clamp no longer carried: the
reflection and the transmission of a rare face formed on 1.178 where they were formed on one.
Five rows in rounding, at most 5e-15 or the sign of a printed zero. Every sheet denser than
the invariant, every lossy sheet without `tf`, every sheet below its critical angle, every
mass-law sheet, every leak chain and every closed far face is identical in every digit, which
table H of the flat-branch case shows.

Four variants were measured and not taken. *The gate alone*, the clamp kept: one line shorter
as well, 25 rows moved, and TM next to a lossy host off the matrix, 0.083 dB and 0.147 dB on
the transmit port at 45° and 60° for ε″ 0.8 and up to 2.55 dB on the reflect port of a
lining; 114 of the item's assertions fail on it. *A face without `tf` left raw*, as the core
leaves it: on the matrix, six lines longer, and `tf` steps at zero by those 0.083 dB and
0.147 dB. *Raw faces with the clamp dropped throughout produce mode*: 108 assertions fail, an
absorber on a rigid wall up to 0.86 dB off its traced solid. *The raw transmission with the
clamped reflection*: +0.54 dB in TM.

The rigid sheet was decided with it. A rigid sheet of the acoustic mapping without the mass
column and without `tf` passed nothing by the flat branch, which is the right result for a
hard panel, and tunnels now: −0.23, −0.30 and −0.88 dB at 10°, 30° and 60° for 22 mm of
`eps 2.1609e-8` in air. Its tilted admittance is `−j sin θ` and holds nothing of the wall's
impedance. Three ways to keep it closed were weighed: a threshold on `|ε μ|`, a special case
with an arbitrary number that closes a real layer of near-zero permittivity in EM; a gate on
the scalar types, which leaves the defect in scalar mode in full and is per mode; and the mass
column, which costs no code. The last was taken: with `m > 0` the sheet is a closed mass-law
layer, exactly nothing without `tf`. The rigid rows of `data/materials_ac.csv` have no
thickness and are traced solids, the modelling guidelines mesh a rigid reflector as a box,
and the rigid row of the material format document carries `m = 20`; no row of the suite
reaches such a sheet through the redo.

Three planned assertions are restated rather than weakened. The sum of the two ports is
asserted for a lossless host only: the matrix itself reads 1.0011 in TM at ε″ 2.0 (§5). One
sheet against its halves is bounded at 2e-3 dB in a lossy host, the two faces of the stack
being 10 µm apart. And the reflect port under a closed outer pair without `tf` is asserted on
the matrix, where the plan had it "as it is": it kept its magnitude and moved its phase.

Found beside it and recorded in §5: the dip within one ulp of a critical angle, and the sum of
the matrix in a lossy host.

**`tf` at a bare boundary.** A bare face with a wave behind it is the transformer of a coherent
sheet, per channel, and the mirror of correction mode follows it; a closed face keeps the flat
leak. The rule and its sites are in §1. Before, `Material::interact_with` scaled the reflection
to the pair rule's `R'` on the TE/TM mean and the transmission by `(1 − R') / (1 − R0)` on top
of the normal-flux amplitude, and `fresnel_r` kept the pair rule per channel on its mirrors.
Three things were wrong with that. *The quotient*: `1 − R0` is the flux a face passes only next
to a lossless medium; next to a lossy one the incident and the reflected wave interfere and the
face passes `1 − R0 + 2·tan(arg η₁)·Im r`, so the exit of every traced lossy solid with `tf`
scaled an amplitude by a flux statement that leaves that term out. *The mean*: one factor for
both channels conserves on the mean only, and with `tf < 0` it lifts the TE reflectance of an
inner face above one, so a solid whose bounces the tracer follows diverges. *Two rules*: a
layer meshed as a sheet and as a solid parted wherever a face with `tf` has two sides that are
not in phase. Measured as the two ports of a traced solid in air, each a sum over every path,
0° to 85°, 1 to 50 mm, the worst `T + R`:

| | before | now |
| --- | --- | --- |
| `eps 0.31 − 0.20j`, scalar, `tf` 0.1 / 0.5 / 0.9 | 8.5 / 49.2 / 50.5 (1 mm, 85°) | 1.0000 |
| `eps 0.90 − 0.20j`, scalar, `tf` 0.5 / 0.9 / −0.5 | 2.80 / 3.38 / 1.27 | 1.0000 |
| `eps 1.50 − 0.40j`, `2.50 − 0.80j`, `5.24 − 5j`, scalar, `tf` 0.9 | 1.11, 1.05, 1.16 | 1.0000 |
| lossless `eps 5.24`, EM, `tf` 0.9: TM, and the TE/TM mean | 2.41, 1.31 | 1.0000 |
| any solid, EM, `tf` −0.5, bounces followed, TE | 1e15 to 1e102 at 400 events | 1.0003 |
| lossless `eps 2.50`, EM, `tf` −0.5, series summed: TE; least HH | 1.77; 0.12 | 1.0000; 0.9999 |
| one face air \| `eps 5.24`, `tf` 0.5, 75°: TE; TM | 0.792; 1.208 | 1.000; 1.000 |
| the same face, `tf` −0.5, 66.4°: VV; HH | 1.500; 0.500 | 1.000; 1.000 |
| a 6 mm layer as a solid against the sheet, porous, `tf` 0.5, 75° | 11.4 dB transmit, 21.6 dB reflect | 1.7e-3 dB at most |
| a sheet of air on a wall with `tf` against the bare wall, lossy | 0.285 transmit, 1.03 reflect | 5.6e-16 |
| through `ray_progress`, porous solid 6 mm, `tf` 0.5, 75°: alone | 15.1 | below one |
| a lossless wall with a forced reflection at an exact match: coated; two solids | 2.47 to 2.97; 0.77 to 0.91 | 1.0002; 0.9985 to 1.0002 |
| a rigid wall with a loss tangent of 1e-3 to 0.04, entry and exit | +159 to +196 dB | −6.5 to −33.3 dB |

A marker build showed twelve cases of the suite on the quotient with a lossy incidence side,
4764 calls, and no call without `tf` with a raw reflectance of one or more on an open face.

Why the transformer: no rule that keeps the reflectance of a lossy wall at `(1 − tf)·R0` can
conserve, and that was measured. Three variants were written as real code, compiled and run on
10,400 walls. *The transformer where the face is open, the flat leak where it is closed, the
mirrors following*: 41 lines shorter, worst 1.0003 scalar and EM. *The pair rule kept on the
mean and the transmission written in the flux the kept reflection passes*: 11 lines longer,
worst 4.35 scalar and 66 EM, because the entry face is met from both sides at once and a face
that conserves for each side alone does not conserve for both. *The pair rule with its
quotient per channel*: 7 lines shorter, scalar untouched at 50.5. The mirrors have to follow:
with the transformer on the face and the pair rule left on the mirror the summed series read
up to 3.30 where the followed one read 1.0000. A closed face is not touched, by decision: the
acoustic main road is a rigid wall whose reflectance seen from the room is one minus its leak,
and a transformer at a face with an evanescent wave behind it reflects everything.

The change: the two branches of `Material::interact_with` lost the mean, the quotient and the
flat-coefficient branch of a matched face; `qd_tf_face` became the member `Material::tf_face`,
called from the bare boundary, from produce mode, from the core and from `fresnel_r`; `fresnel_r`
has one condition for a closed face. The file is 38 lines shorter. No gate on the transverse
invariant and none on the mass column was introduced, and nothing reaches a state word.

The sign at an exact match was decided with it. A forced reflection, `tf < 0`, at a face whose
two sides have one admittance had the same sign from both sides, which no face has: a lossless
wall coated with its own permittivity read 2.47 to 2.97 and two matched solids 0.77 to 0.91,
where a mismatch of 1e-4 read 1.0000. One condition in `tf_face` takes the sign from the media
(§1). It moved no existing row.

It moved powers, by decision. Against the tables that existed, 743 rows in 48 tables. Of a
material with neither `tf` nor the mass column: eleven rows of scalar reflections in the last
digit, 1e-17, from the spelling of a coefficient; a twin that applies the transformer only
where a `tf` is present differs from the change in exactly those rows. 211 rows in 20 tables
at 1e-12 or less, the scalar rows of the mass-law cases among them. The rest carry `tf`:

- *A gain removed.* The coated solid past the critical angle of its outer pair in EM, table H2
  of the lossy-verdict case, −9.2 and −7.4 dB to −16.7 and −11.8 dB, its two ports from 1.08 and
  1.06 to 0.98 and 0.95. A porous sheet with `tf` 0.2 on a rigid wall, table B of that case,
  −0.02 dB at 10°, −1.4 dB at 60° and −4.65 dB at 85° on the transmit port, the reflect port
  identical, the sum from 1.04 to 1.015 at most; the sheet without `tf` on that wall identical
  in every digit. Table G of the sheet-tf case: all 40 moved rows fell, the largest from 1.204
  to 1.003.
- *Two representations on one rule.* Table S of the closed-face case, the solid against the
  sheet, 5.19e-2 dB to 1.49e-4 dB. Table H of the held-lining case, types 1 and 2 on acoustic
  materials, 10.5 dB between a sheet and its solids to 0.008 dB; table B of that case 4.6 dB,
  the twins staying equal. The per-channel step at the critical angle of the outer pair,
  −0.34 and +0.25 dB to −0.002 and −0.023 dB.
- *A rare sheet losing its loss.* Table E of the rare-sheet case: the lossy rows of a rare sheet
  on a rigid wall fell by 1.4 dB at a loss of 0.2 and by 54 dB at 1e-6 and now conserve; the
  lossless row did not move, so the step between them grew from 1.3 dB to 55 dB; closed below
  (*`tf` at a closed face*).
- *Not an energy statement.* The tracer's lossy-verdict case, a porous solid on a rigid wall
  with `tf` and no mass column, 40 rows, the coherent sums by up to 1.8 dB and together from
  0.96 to 1.16 at 30° and 0.5 GHz, the power sums unchanged at 1.37 (§5, the leak).
- *Smaller.* The verdict window of a coated lossy host, the step at an invariant of one from
  0.29 to 0.34 dB; the neighbour step of a sheet on a rigid wall from 0.0021 to 0.0028 dB;
  mass cosine D 0.74 dB at an invariant of 1.45; the recorded series column of table S on a
  rigid wall without the mass column from 1.1e-3 to 6.3e-2 dB.

The references of four older cases carried the old rule and were adjusted, 71 assertions:
`diag_leak_series` takes the near crossing as the transformer per channel in normal flux, the
mirror of *host layer bounded by a sheet carrying tf* is `diag_tf_face`, and the limit table of
the neighbour-tf case is asserted per channel and for the lossy wall at 1e-9, where it was
recorded at a bound of 0.5.

Asserted in the two bare-tf cases (diagnostic infrastructure). One planned row was withdrawn
rather than weakened: the solid against the sheet at exactly 60° for `eps 0.90 − 0.20j` with
`tf` −0.5 is the tie of a forced face, where rounding decides the side; table C runs at 65°
and table H records the tie. A solid rarer than air behind a coating or behind a solid is
recorded and not asserted there: 1.12 and 14.3 then, a clamped mirror, gone since (*No clamp at
a wave face*, below). The second of two
solids has no validated sum with the walker and is asserted through `ray_progress`.

Found beside it: the step of a rare sheet in its loss, closed below; the clamp on a mirror, closed
since; a lossy mass-law sheet against its solid, issue 6; and in §5 the leak sum, the followed
series that diverges, the spike of a sheet at the exact tie and the Brewster angle.

**The scalar closed face.** A closed bare boundary reflects at the Fresnel magnitude in the
scalar types as in the EM types, capped at one. Before, the scalar branch of
`Material::interact_with` took the Fresnel phase and a magnitude of one, which is the Fresnel
value only in front of a lossless medium. Seen from a lossy medium the magnitude is below one
for the porous absorbers of the acoustic table, 0.012 to 0.85 in power for `eps 0.31 − 0.20j`
on a rigid wall, and a magnitude of one handed that power back: a porous box on a rigid wall,
which is how the modelling guidelines mesh a carpet or a foam panel and which a rigid wall
closes from 0.003°, reflected more than it received with no `tf` anywhere. Measured through the
intersector at 1 kHz on concrete, scalar, the reflect port over every path:

| layer, 60° | as a sheet | traced solid, before | traced solid, now |
| --- | --- | --- | --- |
| `carpet_thin`, 8 mm | −0.47 dB | +5.68 dB summed, +6.70 dB followed | −0.47 dB |
| `carpet_thick`, 20 mm | −1.18 dB | +3.52 dB | −1.18 dB |
| `foam_light_melamine`, 50 mm | −2.03 dB | +0.51 dB | −2.03 dB |

One reflection off the wall inside the box sat above the same path in the VV channel by
exactly the Fresnel magnitude it ignored, 0.77 dB to 18.31 dB. Through `ray_progress` 14 of 40
rows of a porous solid on a rigid wall had a coherent reflected sum above 0 dB, worst +3.07 dB;
none has now, largest −0.86 dB. It had been recorded as reached only with an invariant above
one, and the one fixture that asserted a box against its sheet used mineral wool with a
permeability, whose Fresnel value is above one and is capped, so the cap and the magnitude of
one were the same number there.

The change is one line and no line more. It moved 206 rows in seven tables, all scalar and all
a lossy medium at a closed face: the two tracer tables of a porous solid on a rigid wall
(reflected sum down by up to 5.88 dB, the power behind the wall by at most 0.43 dB), the closed
face inside `eps 1.50 − 0.40j` (−2.44 dB at an invariant of 1.10 to −6.90 dB at 1.45), the
`bare` column of the coated lossy solid (onto its vanishing-coating reference), four mineral-wool
rows where the Fresnel value is below one in two tables, and one corner row. No EM row moved,
and a face in front of a lossless medium is bit for bit. Asserted in the scalar-closed-face
case (diagnostic infrastructure); the phase-reference case takes the wall at the capped Fresnel
coefficient, and the absorber-on-a-rigid-wall table asserts the sheet against the solid on
every row.

**`tf` at a closed face.** It is the pad of a closed face, one function,
`Material::closed_face`, at five sites; the rule and the sites are in §1 (*The lumped material
columns in produce mode*, *A closed mass-law layer*, *A sheet under total reflection of its
outer pair*). Before, a closed face reflected the Fresnel reflection scaled by the pair rule
and passed `1 − R'` of the forward wave's flux. Those are one statement only in front of a
lossless medium that carries a wave. In front of a lossy medium, or of a sheet the wave is
evanescent in, the flux towards the face is carried by the forward and the reflected wave
together, and four things followed. *A gain*: the two ports of a sheet with `tf` on a closed
far side read above one. *Two rules for one face*: the reflect port was the composite and the
transmit port a chain of three events whose first crossing depended on a verdict. *A step in
the loss*: a lossless sheet rarer than the invariant was entered as a leak and charged no
decay, while the same sheet with any loss was charged in proportion to its loss. *A step in the
angle*, at the critical angle of the sheet and, for a lossy sheet, of the outer pair. Measured
through the intersector, the sheet event, 1 to 50 mm and 5° to 85°:

| | before | now |
| --- | --- | --- |
| lossy sheet with `tf` > 0 on a rigid wall, scalar and TE, worst `T + R` | 1.177 (`eps 0.31 − 0.20j`, 1 mm, 37°) | 0.997 |
| the same with `tf` < 0 | 2.380 (1 mm, 45°) | 0.9994 |
| lossy sheet without `tf` on a wall with `tf` 0.3 | 1.102 | 0.9986 |
| lossless sheet rarer than the invariant, `tf` 0.5 / 0.9 | 1.325 / 1.837 | 1.000000 |
| coating on `eps 5.24` met from inside, EM material, lossy, `tf` > 0: TE / TM | 1.147 / 1.128 | 0.998 / 0.996 |
| the same with `tf` < 0: TE / TM | 2.826 / 2.157 | 0.9997 / 0.9993 |
| rows above one of 18,144 | 2,170 | 0 |
| 10 mm `eps 0.31 − jε″`, `tf` 0.5, wall `tf` 0.3, 60°: ε″ 0.2 / 1e-3 / 1e-6 / 0 | −7.51 / −29.72 / −59.71 / −4.88 dB | −3.03 / −2.79 / −2.79 / −2.79 dB |
| the lossless one at 1 / 10 / 100 / 200 mm | −4.88 dB at each | −2.13 / −2.79 / −12.13 / −24.06 dB |
| neighbours 0.02° about the critical angle of that sheet: transmit / reflect | −4.76 / +15.52 dB | −0.0002 / +0.0062 dB |
| lossy layer with `tf` on a rigid wall as a sheet against its traced solid, reflect port | 2.58 dB summed, 3.19 dB followed | 0.0045 dB |
| the traced solid's own two ports, worst | 1.165 | 0.9963 |
| `eps 5.24 \| 6 mm eps 1.50 − 0.40j (tf 0.5) \| air` at 80°, VV / HH transmit | −16.68 / −11.82 dB | −6.38 / −8.72 dB |
| a lossy layer with a closed plate, the two sides: `eps 2.50 − 0.80j` / `eps 0.31 − 0.20j` | 0.04 to 0.20 dB / up to 13.49 dB | 0.000000 dB |

Why a pad on the modulus. The leak of a real wall follows the field on it, and a resistive
element in front of a reactance is passive whatever drives it. Formed on the modulus of the
immittance it is the limit of the transformer of an open face where the far wave goes to
grazing, so the two rules meet at a critical angle; and a rare sheet with a loss of 1e-6 and
one with none are one number. It needed no state and nothing that reaches a state word, and no
gate on the transverse invariant.

Variants written as real code and run on 18,144 rows and on the suite before the choice. *The
forward-flux rule*, the reflection scaled by `sqrt(R')` and the flux that reflection leaves,
`(1 − s)(1 + s·R0)` of the forward wave's: no row above one and the fewest moved rows, but a
lossless rare sheet passes nothing, the sheet steps at its own critical angle from −0.11 dB to
nothing, the leak stays proportional to the loss, and the steps at the critical angle of the
outer pair stay (−0.25 dB on the transmit and +0.11 dB on the reflect port of a lossy sheet in
a lossless host, formula level). *The pad at the far face of a sheet only*: one line, and it
parts the sheet from its traced solid by up to 17 dB in EM, where the two agreed. *The pad at
the sheet, the bare face and the mirror*: nine lines, and it parts a stack of a lossy layer
and a closed plate from the traced laminate by 0.025 dB and 0.28°, asserted at 5e-3 dB. *All
five sites, with the reverse on the way out of a plate*: taken, 19 lines.

It moved powers, by decision. 1,203 rows in 31 tables beyond rounding, read column by column
before the source was touched. Sheets close to or past their own critical angle rise on the
transmit port and fall on the reflect port: the porous sheet on a rigid wall with `tf` (table
B of the lossy-verdict case) by up to +6.26 dB and −0.233 in reflectance, table H2 as above,
the tracer tables by up to +5.45 dB and +8.99 dB behind the wall. The lossless rare sheet on a
rigid wall goes from a gain of 0.111 at every angle to 0.28 to 0.38, and with `tf` −0.5 from
nothing to −11.47 dB at 45°. Rows that held a rounding residue where nothing should pass, −160
and −313 dB, read nothing. A plate next to a dense lossy layer moves by −0.09 to −0.77 dB on
its level; next to the rare lossy host `eps 0.31 − 0.20j` the event into the host falls by
9.23 dB and the event out of it rises by 4.25 dB at 60°, which together are the 13.49 dB its
two sides were apart. The recorded gap of issue 2 fell from 7.13 dB to 1.73 dB at 75° without
being touched. 283 rows moved at 7e-15 or less or in the sign of a printed zero: `tf` at a
closed face next to a lossless medium, the rigid wall of the acoustic main road at 3.3e-16 and
the rigid mass-law sheet at 1.4e-16. A face without `tf` is bit for bit, a sheet without `tf`
on a wall without `tf` passes exactly nothing, lossless dense sheets are identical in power,
and no row of an EM type on a material with neither `tf` nor the mass column moved beyond
rounding.

Asserted in the closed-face case (diagnostic infrastructure), 25,275 assertions. The
references of four older cases carried the old rule and were adjusted, 316 assertions:
`diag_sheet_composite` and `diag_leak_series` state the pad through `diag_closed_face`, which
settled the total-reflection and the lossy-verdict case; the lossless rare sheet of the
leak-chain case is asserted on the tunnelling sheet, its two flat leaks recorded beside it;
the two sides of the closed-plate case are asserted equal, the flux splitting recorded beside
them. Three planned assertions were restated rather than weakened. *Two neighbours across the
critical angle of the outer pair* cannot be asserted close: the far cosine has a square-root
cusp there on the exact rule; the event is asserted on the reference on both sides and the
reference on having no step. *The plate with the mass column* was asserted on a number written
by hand that was wrong in the fifth digit, and is asserted on itself across thickness. *The
event of a closed plate next to a host* is its two leaks only where the host is lossless;
next to a lossy host what it passes in and what it passes out are asserted to mirror each
other, and "the exit pays the host leg on top of its leaks" is asserted on the path and not on
the one event, which out of a lossy medium is normalised to the forward wave.

Found beside it: the scalar closed face, closed above; a forced mirror at an exact match, closed since
(*The tie of a forced face*, below); and in §5 a layer rarer
than the invariant as a sheet and as a traced solid, a sheet of no thickness on a leaking
wall, the real immittance of a mass-law medium and the path powers next to a lossy medium.

**One solver.** Every sheet and every ordered stack is resolved by `qd_stack_resolve`, one
recursion, called by both public functions with one layer or with two; `qd_slab_resolve` is
the series factor of a traced solid and nothing else. Before, a single sheet was resolved by a
mode of `qd_slab_resolve` ("produce mode") and a stack by `qd_stack_resolve` ("the core"),
which itself had a matrix form for plain stacks and a scattering form for stacks with `tf`,
`att` or the mass column: three pieces of code for one physics, kept apart so that the
single-sheet path stayed bit-identical to an earlier form of itself. Measured before the
change on 475,200 values handed to both as `SHEET_RESOLVE` hands them - in air, as a glued
entry and exit, inside a host, launched inside a host, at an invariant above one and on four
closed walls, over eleven permittivities, `tf` of both signs, the mass column, `att`, `alpha`,
a resonance and a permeability, scalar, TE and TM: on an open outer pair the two agreed to 1e-9
except where a clamp of the single-sheet path acted, on the loss path of a lossy mass-law sheet
behind a host (up to 3.4 dB at 85°, the real-angle cosine against the complex one) and on
`alpha` behind a lossy host (the same cosine). On a closed outer pair the stack solver was not
told. It is told now, `far_closed`: the pad at the last face, no flux conversion, the displaced
medium at the incidence cosine, and for a mass-law sheet the two crossings of its traced solid.
The file went from 5894 to 5020 lines. With every decision of the entries below left as it was,
the unification alone moved the loss-cap rows, the held gate (next entry but four) and nothing
else beyond 1e-9; 2,165 rows in 104 tables moved at or below 1e-9, the change of form.
Bit-identity of the single-sheet path is retired with it. The loss cap of the matrix form went
with the matrix: an opaque layer transmits at its own level (§5).

**No clamp at a wave face.** A reflection above one seen from a lossy medium is the Fresnel
value, and nothing caps it: at the far face of a sheet on a closed far side, at the mirror of a
host's series that is a layer or a closed face, at the near face of a closed mass-law sheet, at
the pad of a closed layer in a stack, and at a closed bare boundary in the scalar types as in
the EM types. Before, each of those but the EM bare face capped the reflection at one before a
series was summed, and a summed series with a capped mirror is a wrong sum. Measured through
the intersector, every series summed, every path, no `tf` and no mass column anywhere:

| | before | now |
| --- | --- | --- |
| 6 mm `eps 0.31 − 0.20j` lined on both faces with 6 mm `eps 5.24`, 85°, scalar and TE: transmit against the matrix; the two ports together | +20.71 dB; 44.08 | 0.000 dB; 0.99 |
| 864 such walls per channel, coated on either face or both: worst transmit against the matrix | +20.7 dB (TM −0.27 dB) | 0.001 dB |
| 0.2 mm `eps 5.24 − 5j` on `eps 5.24` met from inside at 27°, TM: reflect port against the matrix | 0.265 below in amplitude, 2.7 dB | 1e-16 |
| the same layer as a traced solid: the power it returns | 1.1123 | 0.9861 |
| the coating over its traced solid, TM, without `tf`; with `tf` | −3.22 dB; −1.81 dB | −0.017 dB; −0.060 dB |
| 47,540 sheet events on a closed far side, lossless near medium: the two ports together | at most 1.000000 | at most 1.000000 |
| mineral wool 50 mm with its permeability fit on a rigid wall, scalar, 250 Hz, 60°: reflect port | −13.15 dB | −8.20 dB |
| the same, largest over 250 Hz to 4 kHz and 20° to 75° | −3.00 dB | −1.13 dB |
| the wool as a box against the wool as a sheet, summed and followed | 0.004 dB | 0.004 dB |

The clamp acted in TM for materials without a permeability, in scalar mode and TE for a
permeability, a real angle in a lossy medium, or a layer mirror seen from a lossy medium rarer
than air. On a rigid wall, a layer without a permeability does not reach it in scalar mode or
TE: 2,520 box and sheet rows per channel did not move beyond 1e-4 dB, and the acoustic main road
is where it was. Capping the EM bare face as well was measured and not taken: sheet and solid
then agree with each other and sit 2.7 dB off the matrix. Removing it at some faces only parts
a box from its sheet, by 1.9 dB for the wool. The references of six older cases carried the
clamp and were restated without it, 205 assertions, each of which failed by the clamp's size
before the source changed; two TM bounds went from 5e-2 to 1e-9. What is left is the clamp
inside the pair rule, at the open faces of a mass-law sheet (issue 6).

**No pole clamp.** The solver clamps no denominator, and `qd_slab_resolve` sums near a pole as
anywhere else, with a verdict handed in or without. Before, the single-sheet path forced a
denominator below 1e-2 up to 1e-2, and the series of a traced solid handed the bounces back
near a pole and clamped under a given verdict. A lossless sheet at a transmission resonance has
a small denominator and a small numerator, and the clamp cut it: eps 5.24 in air read −9.41 dB
where the closed form has −4.19 dB at 89.9° and a resonance; with `tf` −0.9, −9.54 dB against
0.00 dB from 87° on; with `tf` −0.99, −29.54 dB against −0.23 dB at every angle; a rigid sheet
without the mass column at normal incidence 0.0196 against 0.99989. With any loss it never
acted. Asserted in table E of the one-solver case and in the verdict-near-a-pole case, whose
series reads 144.46 where it was held at 100.

**`ray_mesh_interact`'s own result.** A caller without state gets what the redo writes in four
places where it did not. *An ordered stack*: the solver with both layers between air and air,
in the order bit 1 of the hit code carries, whatever the two normals are; before, a single
interface between two materials picked by the normals, with no thickness, 0.22 to 1.27 off the
redo in the complex plane, nothing passing where that pair was closed, and hit codes 111, 103
and 101 where an open stack has 79, 71 and 69. The stack takes the interface call of a single
sheet, air | air, so its segment, tube and `path_dirN` keep the incoming direction with no flag
of their own. *A corner with the sheet as the first face*: the sheet alone, 0.22 to 0.88
before and exactly zero now on all 16 rows; hit code 65 where it was 97. *A ray tube with one
edge past the critical angle*: the spine's coefficient, −6.437 dB where it was nothing; the
tube is passed undeviated and the bit is set as before, at a sheet and at a bare face alike.
*A mass-law lining left from inside a solid with types 2 and 5*: the mass term at the cosine in
air, mapped from the bent segment; 0.080, 0.790 and 4.447 dB at 10°, 30° and 60° before. What
the map cannot know is in §5. Asserted in table F of the one-solver case, the corner case, the
own-glued-composite case, table E of the mass-cosine case and the tube case of
`test_ray_mesh_interact.cpp`.

**An ordered stack inside a host.** The mirror of the host's series is both layers, backed by
the host, ahead of the leg in front of the stack and behind the leg behind it. Before, each
half of the host took the nearest layer alone as its mirror, and with every series summed the
wall was off its characteristic matrix by up to +1.09 dB on the transmit and +0.70 dB on the
reflect port (scalar and TE; −0.49 dB in TM), for materials with neither `tf` nor the mass
column; a single sheet inside a host was on its wall at 1e-4 dB. With both layers ahead alone
0.78 dB is left, the mirror behind. The previous-medium word names one layer of it. The other
travels as the stack code in the buffer word (§1), a code point no site wrote: bit 14 with an
index beside it. For a while the leg behind carried the followed mark instead, on the reading
that no code point was free, and its bounces were rays: exact in the limit, and measured
through `ray_progress` at 22 to 256 paths behind a wall of 50 mm | 6 + 6 mm | 50 mm for one
ray with every series summed, 20 interactions and a floor of −140 dB, where the wall with one
sheet has 4 to 5; +0.62 dB off the matrix for a lossless host at 85°, the series cut by the
interaction limit; 740,223 rays and 2.4 s for that one ray with 60 interactions, still
+0.99 dB off; and +96 dB to +134 dB after 20 interactions where the round trip of that leg
exceeds one, in a lossy host rarer than air between linings. With the code the same walls take
4 to 5 paths and 24 to 32 rays and sit within 3e-4 dB, 8.9e-3 dB for the lossless host at 85°,
and the walker of table G takes 6 paths where it took 546 to 2047, with both sums where they
were. The state did not grow. A stack whose last layer is mass-law is a half-space from behind
and writes the thickness code. Asserted in table G of the one-solver case and in the stack-wall
case of `test_ray_progress.cpp`. A glued solid as a two-layer list and a sheet inside a solid
as a three-layer one were measured against the sum of paths and are worth nothing: both are on
the wall as they are.

**The tie of a forced face.** A reflection forced by `tf < 0` between two media of one
admittance takes its sign from the media wherever the admittance ratio is within 1e-9 of one in
modulus. Before, only where the two agreed to the last bit; evaluated at a real angle the two
sides form the ratio from two spellings of one cosine, and rounding gave both sides the same
sign at 147 of 400 angles (65 of 161 in the fixture), where they have to be opposite. A wall
`eps 5.24 | eps 5.24 with tf −0.5` met from inside past the critical angle of its far face
returned 0.087 to 3.30 of what it received with every bounce followed, and returns 0.997 to
1.004. From air through parallel faces both sides were consistent already, 0 of 400, which is
why the walls of that kind asserted earlier held. Reached by a ray launched inside a host, a
caller without state and a wedge. The summed readings of 0.13 to 3.57 on the same probe were
the instrument: a seeded leg the state does not know to be bent; with type 5 and the bent flag
on the seed, summed and followed read 0.9998 to 1.0008. Asserted in table H of the one-solver
case. The step of a forced face at a ratio of modulus one (§5) is where it was.

**The held gate is one gate.** A mass-law sheet behind a wave held in a mass-law wall takes the
chain of held crossings where at least one face of the run leads towards the lower index, the
gate an ordered stack always had. Before, a single sheet took it only where its first face led
down. A mass-law lining that is not rigid on a rigid wall, whose far face leads down towards
air, is thereby on its solid twin: W with a lining of `eps 2.25` and `tf` 0.3, the wall met
first, −10.6779 dB to −11.9786 dB at 75°, the twin at −11.9786 dB, and the two sides within
0.008 dB, the flat first face. 18 rows moved, in tables G and H of the held-lining case.
Asserted there at 1e-3 dB. What is left is a lining both of whose faces lead upward, issue 4.
**A sheet takes no more host than lies on each side of it.** The displaced medium is taken out
per side of the mesh plane, each part of the layer at most the host there, and the transmit
port, the reflect port and the mirror of the host's series place the layer alike; the
convention is in §1. Before, one total was capped at 80 % of the host the event saw, the
reflect port took half of that total, and the mirror took no cap. Three things were wrong with
it. An ordered stack has the other sheet as the second face of its hit, so it saw the leg in
front alone: a stack that fits its host was short by 0.030 dB and 1.42° (5 mm centred in
10 mm) to 0.24 dB and 7.1° (9 mm in 10 mm) on both ports, and by up to 0.15 dB and 0.57 dB on
the two sums of the wall. With the undeviated tracer the gap ahead was the segment at the
external angle times the traced cosine, so a sheet three times its host was not capped at all
at 85°. And the three placements were three walls: a wall of lossless media with a capped
stack that fits read 1.0014 to 1.011 on its two ports together, a single sheet at 90 % of its
host 1.005, layers that do not fit up to 1.27, and a lossless sheet off the middle of a lossy
host that it does not fit 1.32 to 1.43; the checks of the time ran on a centred sheet in a
lossy host against a reference that carried the same cap, and saw none of it. Now a sheet or a
stack that fits is on the uncapped reference at rounding, the walls of lossless media read
1.00003 at most whether the layers fit or not, the off-centre sheet 0.855 at most, and both
tracers take one reference per row. The factor of 80 % went: a cap per side leaves a wall that
exists at any thickness. Moved beyond 1e-9: the rows of the budget case, up to −0.96 dB on a
chain that the undeviated tracer had left uncapped; the stack in a host of the rare-sheet case
by +0.43 dB; a 25 mm sheet inside a host in the tracer's sheet-tf case by 0.07 dB; reflections
off a stack inside a host in the stack-geometry tracer case by +0.17 dB and +0.34 dB; 40 rows
of two stack-geometry tables by 6e-4 dB; and the worst-row labels of two summary tables whose
sums stay 1.000000. 416 rows in 22 tables moved at or below 1e-9, the reflect-port factor
written over twice the part in front. No row of a material with the mass column moved. The
dead cap in `ray_mesh_interact`'s own composite, which displaces air in every case, went with
it. Asserted in the two budget cases of §1. What a stack cannot know of the host behind it is
in §5.

**A permittivity resonance is part of the medium.** A Lorentz pole used to be added to the
permittivity at every interface quantity - admittance, cosine, Snell ratio, verdict - and left
out of every propagation index: the layer phase, the displaced medium, the traversal of a traced
solid, `medium_gain` and `refractive_index`. It came with the acoustic material model, and every
later piece of code carried it. That is not a wave at oblique incidence, and with a lossy pole it
is not passive: a face with a complex admittance in front of a lossless phase amplifies at a
quarter wave. Measured on the split: a 37.5 mm sheet of `ε` 4 with a pole of −2.5j read 1.092 on
its two ports at normal incidence; a strong pole that takes `ε` 2.5 to −1.04 − 0.26j read 110 on
50 mm, 260 for its traced solid summed and 1e98 followed; 1174 of 14,400 sheets of a formula scan
read above one; a traced resonant solid read 2.11 summed, 148° and 0.87 dB from its sheet, and
its followed series 10.6 dB from the summed one; a band through `ray_progress` sat up to 3.4 dB
and 79° off the medium. A reading with the resonance in nothing that propagates - the layer
phase of `ε` at the faces' transverse invariant - puts the sheet on its solid but still read 1.25
on the lossy pole and 84 where the face and the inside disagree about evanescence. The pole is
now part of `Material::eta()`, `eta_resonance` is gone, and `medium_gain` takes its rate from the
complex index; −17 lines. No state, no verdict, no direction and no geometry moved: the hit codes,
state words and directions of 93 events were identical. Moved: every row with a resonance - up
to −0.63 dB on a resonant sheet, −0.42 dB on a stack with a resonant layer, +0.58 dB on a glued
sheet with `att`, `tf` and a pole, a 100 mm resonant host leg by its −3.70 dB decay, the matched
sheet's phase from −7.2° to 0 - and two rows of a mass-law stack by 1.4e-17. No row of a material
without a resonance. Asserted in the resonance cases of §1. What a resonance does on a material
with the mass column is issue 1.

---

## 5. Known limitations, recorded rather than open

**The pair rule clamps its reflectance.** `apply_tf_pair` clamps `R0` into `[0, 1]`, and it is
the one clamp left in the file. It is formed on a reflectance of one for the pad and on the
modulus of an admittance ratio for the transformer, where it cannot act; it acts at an open
face of a mass-law sheet next to a lossy medium, where the Fresnel reflectance is above one.
Issue 4 has what follows from it. No other reflection is capped anywhere.

**`tf < 0` forces a reflection in TM at the Brewster angle.** `tf` is per channel at every
site, so a forced reflection does not keep the ratio of the two Fresnel channels: at 66.4° on
the lossless `eps 5.24` with `tf` −0.5 the face reflects 0.731 in VV and 0.500 in HH, where
the raw face reflects nothing in HH. A rule on the TE/TM mean kept that null and paid for it
with a VV reflectance of 1.23. With `tf > 0` the reflections of a lossless face are the raw
ones times `1 − tf` in each channel. Recorded in table B of the bare-tf case.

A forced reflection at a matched boundary is a reflection, so a laminate of two identical `tf < 0`
sheets has an internal mirror, and its phase is not defined by the media: the Fresnel reflection it
takes its argument from is zero.

The mass-law route carries no internal series at any angle, so the energy reflected at its far
face is dropped rather than returned, and it does not conserve energy on a lossless palette. It
is bounded by it. An incoherent power sum would return the energy and has not been written. A
coherent sheet carries its series on both ports, under a closed outer pair as well.

**A rigid material with `tf` and `m = 0` is not reciprocal at a solid | solid face.** The rule of §1
is gated on the mass-law column. Without it a leak meeting a face towards the lower index is judged
from inside as before, flat leak past the critical angle and Fresnel below it, while the other
direction is an ordinary crossing; for an impedance ratio of 3 that is 4.5 dB at 30°. Lifting the
gate needs to know that a wave is a leak, which the materials and the angle do not determine: a ray
at 60° in air behind glass is consistent with a refracted wave and with a leak. It would cost a
state bit, and none is left (§1). `tf` is in practice set together with `m`. A rigid wall
without the column with a rigid lining is the same statement: 0.20, 4.48, 3.62 and 2.52 dB
between its two sides at 10°, 30°, 60° and 75°, no gain, the sheet and its solid twin alike;
table G of the held-lining case holds its rows.

**The first face of a leak is flat.** Under total reflection the leak is `1 − R'` with `R0 = 1`,
while the same face left from inside is a Fresnel crossing with `R0 = 1 − 4 z cos θ`. For an
air | rigid face that is 0.010 dB on wood and 0.008 dB on concrete between the two directions. It is
the recorded part of the leak-reciprocity fixture.

**The wave becoming a leak is a step.** Below the critical angle of the near face a wave is
refracted into the sheet and meets the far face near grazing; above it, it is a leak at the
incidence angle. For a wood panel on concrete the transmit port steps by 4.67 dB there,
`10·log10(0.82 / 0.28)`. It sits at 0.0084° from the normal, inside the cone in which a rigid
material admits a wave at all, and is what a leak keeping its direction means. Recorded in the chain
fixture against a 5 dB bound.

**Across the critical angle of the outer pair there is a cusp and no step.** Below it the far
face of a sheet is a transformer, above it the pad, and the two are one rule at the angle: the
reference's limits at an invariant of 1 ± 1e-10 agree to 0.0006 dB in both channels, and the
event sits on the reference 0.01° below and above at 6e-14. What is left is the square-root
cusp of the far cosine, which the exact composite has: two neighbours 0.02° apart differ by 0.1
to 1.2 dB, and a coated lossy host by 0.35 dB between an invariant of 0.99 and 1.01. Tables D2
of the closed-face case and the verdict window of the exits-and-verdicts case.

**Entry and exit of a lossy solid differ by the flux splitting.** `sec²(arg η)` per channel, exit
above entry; the numbers are in §4. It is the energy convention of §1 and is not a defect.

`mtl_prev` is one level deep, and after a sheet event it holds a sheet. What its readers then
take, and where that is not exact:

- *Air beyond every glued exit*, whatever the word holds. Exact for a body standing in air. A lined body nested
  in another medium would be resolved against air and continue in air; it is not a use case (§1).
- *The backing of the sheet behind the ray*, air for a lining and the host for a sheet inside it.
  Exact for both. Behind an ordered stack inside a host the mirror is the last layer backed by the
  host, and the other layer is not seen. After a back reflection at a bare face the word still
  holds what the leg was entered from or through, though the mirror now behind the ray is the
  bare face; no series is formed on that leg (§1), so nothing reads it as a mirror, except on a
  leg whose followed mark a deferred medium replaced.
- *A real transverse invariant for the recovery.* Exact behind a sheet and behind a flagged
  solid, at any depth. Behind a solid without the flag the solve assumes a real angle in that
  solid, which is exact one solid deep behind a face evaluated at a real angle. Two deep it is
  not: the wave in the second solid behind a launched wave or a wedge has neither a real angle
  nor a real invariant, the transition out of it sets no flag, and the third leg takes the
  named form for want of a better one. No fixture; no worse than it was.
- *A reflection off a sheet replaces the name.* It writes the sheet into the word, and behind a
  sheet the recovery solves for a real invariant. For a host that was entered from a lossy solid
  across a face evaluated at a real angle - a wedge, a launched wave - the leg before the
  reflection solved with that solid named, so the recovered cosine of that leg moves at the
  reflection. Read from the code, not traced on a reflected leg; the size of that solve on such
  a wave is the recorded one, −1.0 dB at 75° and a killed ray at 85° behind a lossy wedge. With
  the followed mark the series no longer reads the sheet a reflection writes, so not writing it
  would be possible; it is the convention of §1 and is left.
- *A lossless solid in the word is named even where the flag could be set.* The same statement,
  kept in the named spelling so those rows stay bit-identical.
- *A deferred medium displaces the thickness code.* The leg then pays its own mass term in full:
  +37.5 dB on a wall with a nested object behind a bonded solid, from one side only. Issue 8.
- *A solid in the word is a half-space mirror.* The host's series takes the solid behind the ray
  as a half-space, where a sheet there is a layer. That is the decomposition of §1 and not a
  limit of it: what the solid's far side returns arrives as paths.

The same depth limit is what stops the solver from being handed more than two layers: the hit
names two faces and the state one boundary.

A sheet between two solids is not a geometry the intersector reports (§1); nothing is designed
for it.

A mirror above one is reached by a traced chain as well: at 85° of external incidence on the lossy
host a sheet backed by air reflects 1.0197 seen from the host, and the series takes it as it is. The chain
fixture is on the matrix on that row as on every other, and records what a clamp there was worth.

`SHEET_EVENT` sets the fix-flag rather than guessing when it meets a deferred buffer
(`mtl_buffer != 0`), because no sheet code in scope carries one. A glued exit there still exits
into air and the deferred medium is dropped. A deferred medium says the ray is inside two bodies
at once, so what lies beyond the sheet is then the other body: a sheet between two solids,
reached by overlap instead of by a triple hit, and not designed for either.

**The current-medium word is corrected in one direction only.** A ray recorded in air while it
is inside a solid is put right at the next exit face, from the material of that face, with the
fix-flag. A ray recorded in a medium while it is in air is not: every wall it meets is a nested
pass-through on the way in and a same-medium pass on the way out, transparent and without a
fix-flag, a reflection off a wall is evaluated as the internal reflection of the recorded
medium against air, and `ray_subdivide_flag` never splits the beam again. The sizes are in §4,
under the glued exit behind a solid, which was the one writer of such a state. Nothing in this
file writes one now, and the layered-wall case's next-wall rows are the guard; a change that
writes `mtl_current` has to be right, because nothing downstream will notice.

**An exit out of a wave whose invariant is not real takes the real-angle map.** A wave launched
inside a lossy solid, a lossy wedge and a host whose recovery declines write
`(|sqrt(p)|, Re sqrt(1 − p))` for a complex `p`, where the phase normal of the inhomogeneous
wave in air is `(Re sqrt(p), Re sqrt(1 − p))`. Second order in the loss angle, about 0.2° for
`eps 1.50 − 0.40j` at 25° in the solid, computed and not measured. For a wedge the decay direction in the host is normal to the entry face, which the state
does not hold.

**With the undeviated tracer a wrong direction in air is not a wrong place.** The geometry is
the incoming segment, so the next wall is met where it should be. It is not harmless for that:
a sheet event in air is evaluated at the direction handed in, and a direction along the normal
cost a lining behind a rigid wall up to 11.6 dB at 75° before the leak direction was closed.
`path_dirN` in air is read by
a sheet event, by the recovery in the next wall where that wall is entered through a glued
sheet, and by the polarization basis; a bare entry and a reflection in air keep
`ray_mesh_interact`'s geometric result, and a segment with no hit resets the direction.

**With the refracting tracer the segment at a closed face is `ray_mesh_interact`'s.** That
function judges a face at a real angle along the segment. For a lossy medium the real-angle
test reads the invariant short by the length the projection lost, so for an invariant between 1
and about 1.045 on `eps 1.50 − 0.40j` it finds open a face that `ray_state_update` then finds
closed. The direction written is the wave's; the relaunched segment stays the Snell bend, 67.93°
and 68.94° at 1.01 and 1.04 against a wave at 52.27° and 53.28°, at a bare exit, a coated exit
and a transition alike, and the leg behind such a transition carries the bent flag, which
describes that polyline. `RELAUNCH` moves the segment at a closed face only where the invariant has a direction in
air, and above one it has none. It carries power only with `tf` on the far medium, and is
reached only by a host entered from a medium denser than what lies behind it, so never by a
wall standing in air. From 1.10 on the segment is the wave. Recorded in the closed-face table of
the leak-direction case.

**The direction a closed face hands on has a step at an invariant of one.** Below one it is
the direction in air, which goes to grazing; from one on it is the direction the wave has. A
host `eps 1.50 − 0.40j` entered from a lossless `eps 5.24` solid hands the rigid wall behind it
84.26° at 0.99 and 52.27° at 1.01. It is where a direction in air stops existing. Reached only
by a host entered from a medium denser than air, so never by a wall standing in air; table D
of the closed-face case. With the undeviated tracer such a hand-built leg does not have its
segment on an incidence line, so below one the wall behind it is not held either, as before.

**A caller with types 2 or 5 that does not hand in the geometry gets a wrong wall behind a
closed face.** The direction written is the leak's, the segment stays the layer's, and the
wall then reads two directions: it writes 0.0005° to 0.0028° behind itself and forms its mass
term on the layer's cosine, as it did before for the level, −61.67 dB against −66.02 dB at 45°
behind the porous layer. `ray_progress` hands the geometry in on the base-frequency transmit
call. Row `dir` of table C of the closed-face case.

**A rigid wall with `tf` and no mass column leaves along its normal.** The hold of §1 is gated
on `m > 0`. Without the column the exit into air is the ordinary crossing with the Snell ratio
of the impedance mapping: 0.0005° at 10°, 0.0020° at 45°, 0.0028° at 85°, bare, behind a sheet
and behind a traced layer alike, and with the refracting tracer that is the segment. What
stands behind such a wall is met along the wall's normal. Lifting the gate needs to know that
a leg is a leak (the entry on reciprocity above); whether the refracting tracer's state could
say so has not been examined. `tf` is in practice set together with `m`. Recorded in section J
of the lossy-verdict case and in tables A and C of the closed-face case.

**What the closed-face rule has not been measured on.** The EM types; a layer whose faces are
not parallel and a wave launched inside the layer, where the recovery declines and the
direction is the real-angle map's or the segment's; a beam through `ray_progress`, the tube
being asserted on single events in double; and, on a wall that is not rigid and rarer than the
layer's invariant, the window in which `ray_mesh_interact` finds the face open on the segment
and `ray_state_update` finds it closed, where the leg behind carries the bent flag and is not
held. For a rigid wall that window lies below 0.003°.

**Behind an air gap the two tracers agree to the last digit, not bitwise.** On a rigid wall the
refracting tracer is the undeviated one in every word, type code and gain, exactly. A second
wall behind an air gap is reached by a segment relaunched from a point each tracer builds its
own way, and the gain of its exit can differ in the last digit, 2.8e-14 dB worst over the
leak-direction case.

**A traced leg is short by the relaunch offset.** `ray_mesh_interact` relaunches a few float ulp
of the coordinate magnitude past the face, along the outgoing direction, and the ledger starts
there. The normal part is 1e-6 m with the undeviated tracer and up to 1.06e-5 m with the
refracting one at 85°, in a scene a few meters across; it grows with the distance from the
origin. Through the series of a dense slab near grazing that is 4e-3 dB. The fixtures carry it
in the reference thickness.

**The bent flag is the base frequency's.** It is written from the total-reflection verdict of
the event that bent the segment, at the base frequency. Near a critical angle of a lossless
dispersive medium another frequency of the run can have the other verdict; the flag then
describes the polyline and not that frequency's wave. A lossy medium has no such angle (§1).
A ray launched inside a medium, and a caller that keeps
no state, have no flag: such a leg is treated as a leg whose wave was not bent, as before.

**Where the cosines decide, rounding can.** With the undeviated tracer "bent" is still read off
a 1e-6 difference of two cosines. Near normal incidence the two coincide and the leg takes the
forms of a wave that was not bent (the hybrid propagation factor, below); in single precision far from the origin
the test is noisy at that level. The refracting tracer no longer depends on it for a bent leg.
For a leak leg it still does: behind a closed face above an invariant of one the segment is
`ray_mesh_interact`'s and carries the anchoring of the co-located pair before it, and on the
hand-built 2 m segments of table D of the closed-face case the wall behind is held at an
invariant of 0.50 and not from 0.90 on, where it writes 0.002°. Below one the relaunch puts
both points on one line and the hold does not depend on it.

The wedge test is unusable for a glued entry, since it compares FBS and SBS, which are co-located by
construction. The code defaults to parallel there, which is right for a coating on a flat wall and wrong
for a sheet glued to a wedge.

`Material::interact_with` keeps two spellings of the same two expressions, one for a real angle and one
for a complex cosine: the refraction-root argument and the total-reflection test. They are the same
statement and the complex form is the general one. The real form is retained because reassociating it
moved coefficients by up to 6e-14 relative and, at a matched interface, lifted the reflectance off
exactly zero, which sent a `tf`-forced reflection down the rescale branch and flipped the sign of a
coefficient whose power was unchanged. Do not merge them without re-running the bit-identity check
described in §1.

The mass-law term is a hard zero below an argument of one rather than a taper, so its frequency
dependence has a kink where its angle dependence no longer does. Removing the clip needs a knee that is
independent of the normal-incidence offset, and the columns cannot express one because `fRef` sets both:
choosing `fRef` to place `TL(0)` also places the roll-off. Bounded by 3 dB and only within a decade of
the knee, where the term is itself under 3 dB. Fixing it costs a material column and a table migration,
which is not worth it at that size.

**The mass term steps at an invariant of one.** Below one a mass-law sheet behind a refracting
layer forms its term at the cosine the invariant has in air, which goes to zero at one; from one
on it keeps the cosine the wave has in the layer (§1). The mass paid reads 7.66 dB at 0.99 and
22.60 dB at 1.01 behind `eps 1.50 − 0.40j`, 24.61 dB behind `eps 2.50`, for a 26.8 dB sheet, and
the event of a sheet inside the host steps from −4.95 dB to −20.07 dB where it was smooth. A
traced mass-law wall behind a layer has the same step in the direction it is handed. Never
reached by a ray that came from air through the face parallel to the plate: it takes a host
entered from a denser body, a ray that entered the cover through another face, or a source
inside the cover. Table D of the mass-cosine case.

**A ray the state does not know to be bent keeps the cosine of its segment in a mass term.** A
source inside a host, and a caller without state: the leg cannot be told from a leak, whose
segment is at the incidence angle already. In a lossless `eps 2.50` host a mass-law lining then
pays 0.21, 0.96 and 3.00 dB more than at the Snell cosine in air, at 10°, 20° and 30° in the host.
Table E of the mass-cosine case, its `launched` rows. `ray_mesh_interact`'s own result with
types 2 and 5 maps the segment to air and is exact in a lossless host.

**The cosine in air behind a lossy wedge is the real-angle one.** Where the recovery declines
the invariant of a wave in a lossy host is complex and it has no exact direction in air. The mass
term takes `Re sqrt(1 − ε (1 − c²))` at the traced cosine; the cosine of the direction the exit
writes, which is normalized, gives a term 0.002, 0.011 and 0.096 dB away at 10°, 30° and 60° on
`eps 2.50 − 0.80j` with the exit face tilted by 10°. A lossless wedge is exact.

**A mass-law sheet inside a host pays as the bare plate.** The thin-cover view of §1. A plate in a
host many wavelengths thick on both faces would read the host's wave admittance, which the term,
calibrated in air, does not hold.

**The thickness code is quantized.** A traced solid's surface mass reaches the next body as one of
32767 logarithmic steps of 0.044 %, worth at most 0.0038 dB on a mass term at `m = 20` and
proportionally more for a steeper slope, and the code is quantized again at every bonded face it
is carried across. The solid | solid rows are bounded at 0.01 dB for that reason and measure
7e-4 dB; the worst row three and four bodies deep is 3.0e-3 dB, at `m` 30.

**The thickness code ends at 100 m.** A thickness outside 0.1 mm to 100 m is clamped to the end
of the range. A wall is never that thick, but an equivalent thickness is the heavy body in
front expressed in the material of a lighter one, scaled by the ratio of their `fRef`, and can
be. A ratio of 2.5 is exact. With a body of `X₀` 10000 in front of 10 mm of a material ten
thousand times lighter per thickness the code clamps and the third body pays +0.74 dB from that
side, where it paid +39.9 dB before the code carried the body in front; from the other side it
is exact. Where between those two ratios the clamp begins has not been located. The low end is
reached only by a body thinner than the 1.5 mm floor met at grazing.

**Mixed slopes are folded from one side.** A bonded body whose slope `m` is not that of the
material the code is read with, and which the words do not name themselves, is left out of the
plate (§1). Two numbers would be needed, one per slope, and the state has room for one; no
frequency-independent place for the second was found. Measured over the plate at normal
incidence, far above the knee, 100 mm bodies unless noted: `m` 20 \| 20 \| 30 (50 mm) exact from
the first side and +4.29 dB from the other, +3.34 dB to +5.01 dB over 0.25 to 4 GHz; `m`
20 \| 30 (50 mm) \| 20 +0.43 dB from both sides and +5.11 dB near the knee; a lining of `m` 30
on a wall of `m` 20 holding a sheet +1.51 dB from the lining's side and exact from the other;
two sheets of `m` 20 and 30 inside a wall of `m` 20 exact from the side that crosses the `m` 20
sheet first and +1.32 dB to +1.54 dB from the other; plaster on a brick wall of another slope
+0.003 dB, the body left out being light. So such a wall transmits differently from its two
sides, by up to what it was over from both before. A table that gives every bonded material one
slope never reaches this.

**The plate near its knee can exceed the dB sum.** The single-body angle shape is quadratic in a
small `X₀·cos θ`, so two light bodies bonded pay more than the two alone near grazing: +0.44 dB at
70° for `X₀` of 6 and 2.4. That is the shape, applied consistently, and not a defect.

**A fitted slope other than 20 is not a mass law in thickness.** The plate sums impedances, so two
halves of one material with `m ≠ 20` do not reproduce the table's term for the summed thickness;
the size and where it is reachable are in §1. A wall of `m` 30 in halves of 60 mm and 40 mm
around an 8 mm sheet of its own permittivity pays 57.39 dB, the plate of the three, where the
table's term for 100 mm alone is 60.00 dB.

**The energy a mass-law body does not pass is not returned as reflection.** The mass term is a loss
factor on a `tf` leak. A mass-law sheet forms no bounce and a mass-law solid's bounce pays the term
again, so neither conserves on a lossless palette; both are bounded by it. Below its knee a
mass-law body has no term at all and is charged its two `tf` leaks, where a real panel is nearly
transparent, which is the weakest regime of the surrogate and is set by the table, not by this
file.

**The segment follows the written direction out of a lossy medium, and behind a closed face.** Where the recovery
declines - a lossy wedge, a wave launched inside a lossy solid, faces known not parallel - the
ray keeps the segment `ray_mesh_interact` built. The direction `ray_state_update` writes there
is the same real-angle map applied to the direction the wave has, which on a bent leg at the
base frequency is the segment, so the two agree to rounding and not bit for bit; on the wedge
fixture they agree in every printed digit. A lossless medium is not touched at an open face, which leaves the mass-law solid of issue 7
with 24.09° travelled and 30° written.

**A lossless leg behind a co-located pair carries that pair's anchoring.** `ray_mesh_interact`
relaunches a co-located pair from the SBS and aims at a destination measured from the FBS, so
the segment is off its own direction by the separation of the two points over its length. The
refraction at the next face amplifies that towards grazing: behind a glued entry into a
lossless host the segment out of the wall is 1.6e-6 off at 85° over a 300 m segment through
`ray_progress`, and 4e-5, 84.9976°, over the 2 m segments of the chain fixture. A relaunched
ray has both points on one line and does not carry it, and neither does an ordered stack, which
does not bend. The tilt is the gap between the two faces over the length of the segment, so a
pair whose faces coincide in the mesh has none. It can cross the 1e-6 gate on which
`ray_state_update` reads "the direction differs from the segment": behind a sheet glued on a
host with a 10 µm gap and 2 m segments the leg is 3.7e-6 off at 60°, and the exit of the host
then carries resolved type 3 where it carries 1 at 30° and behind a bare entry, with the exit
segment 1.0e-5 off the direction written and 6.9e-5 dB on the chain. The bit is right for what
the event was handed; the level is exact to the size of the tilt. Aiming the destination from
the point the ray is relaunched at would remove it and would move the segment of every
co-located event that bends; not built, and not taken for a flag.

**`ray_mesh_interact`'s own mass-law lining is approximate in a lossy host.** With types 2 and 5
the own composite forms the cosine in air from the bent segment by the real-angle map, which is
the wave in a lossless host. In a lossy host it is the approximation every real-angle map is
here (0.579 at 60° on `eps 2.50 − 0.80j` where 0.500 is right), and a host the wave is not bent
in, a leak, would be mapped wrongly; that needs a closed host without the mass column, which is
not a valid material (below). With the state kept the redo is exact. Not sized in a fixture.

**A stack inside a host continues the segment it was reached on.** That is the wave wherever
that segment is: at the base frequency behind a bare entry. Where the segment in front is off
the direction the wave has - behind a glued entry by its anchoring, where the recovery
declined, at every frequency of a band but the base one - the leg behind the stack is off by
the same, as it is behind a single sheet.

**The tube is corrected with the spine or not at all.** No fixture reaches a vertex that cannot
be formed. A tube whose edge is past the critical angle of the face ahead while the spine is
not makes `ray_mesh_interact` pass the whole tube undeviated and set the total-reflection bit.
With the state kept the redo then writes the undeviated tracer's event for that ray, bit for
bit, on a leg without the bent flag; a caller without state gets the spine's coefficient as
well, with the total-reflection bit set in the hit code: the verdict is the tube's, the
coefficient the wave's.
Measured at a glued entry, not at a bare face. The corrected tube is asserted in single
precision through `ray_progress` only.

**The geometry is the base frequency's, by the caller's convention.** `ray_state_update` moves
the geometry on any call that hands it over. A caller running a band has to do that on the
base-frequency call alone, as `ray_progress` does.

**What lies between a hit point and the next relaunch point is charged as free space.** The
ledger of a leg starts at the relaunch point, a few float ulp past the face, and across a
co-located pair past the second face. The polyline between the hit point and that point is in
the path length and in no ledger, so it is free space along the polyline, and the leg is short
by its normal part. Through two solids across a bare face with a 10 µm gap, series declined,
that is what separates the two tracers: 0.013° at 85° with each at its polyline's exit point,
0.017° with a 1 µm gap, −0.008° with 50 µm, and the phase that follows from the geometry alone
reproduces it to 0.0002° on 36 rows, lossy and lossless. It grows with the gap of a co-located
pair, which is a property of the mesh.

**A co-located pair is judged along the ray.** Two faces are one event where their hit points
are within 1 mm of each other along the segment. A pair 100 µm apart is 1.15 mm apart along a
ray at 85° and did not run as one co-located event there: the chain did not have its three
events. Seen in the phase-reference case, whose gaps stop at 50 µm for that reason; not traced
further.

**A rigid half-space of the one-parameter mapping reflects with a phase of twice the angle.**
Outside this file, recorded and parked. Seen from air, `eps 2.401e-9` with `mu 1` reflects at
unit magnitude with +20.0000°, +80.0000°, +120.0000° and +170.0000° at 10°, 40°, 60° and 85°,
which is the Fresnel coefficient of that permittivity under total reflection. The same
impedance with an index of 0.1 (`eps = z·n`, `mu = n/z`) reflects with 0.008° to 0.64°. So it is
the mapping, which fixes the index together with the impedance, and not the evaluation. A wall
behind a porous layer reflects with that Fresnel phase, 43° to 137° for
mineral wool at 20° to 75° over 250 Hz to 4 kHz, at the Fresnel magnitude, which
there is +5.3 dB to −1.2 dB. It matters wherever reflections are summed coherently. What it does to the
absorption of a wall-mounted absorber with its series resolved has not been measured. Recorded
in the last table of the phase-reference case.

**The hand-built chain fixtures do not describe a polyline.** They prime the geometric ledger
with the normal thickness and pass the geometric angle beside it. The phase their references
carry to the polyline's point follows the code with those values and has no physical size; see
the diagnostic infrastructure.

**The phase at the polyline's point is a plane-wave statement.** The field of a frequency's
wave is taken at the committed point as that of a locally plane wave, which is the assumption
the Fresnel coefficients and the slab series already make. Where the invariant is complex the
magnitude is the one at the refracted ray's point and only the phase is carried over.

**`use_vbs` gates the carry-over, on a 1e-6 difference of two cosines.** Below it the two points
are taken as one. For a medium that is all but free of dispersion, far from the base frequency
and near grazing on a body many wavelengths thick, what is dropped is `k₀·d·1e-6 / cos² θ`.

**At a real angle in a lossy medium the magnitude test stands.** Where the first medium is
lossy and the second lossless, and no cosine was recovered, the verdict is
`sqrt|ε₁μ₁ / ε₂μ₂|·sin θ ≥ 1` at the angle of the segment or of the traced direction. That
angle is not what the wave has. With the undeviated tracer `ray_mesh_interact` judges the exit
of a lossy solid at the external angle and sets the bit on a face the wave crosses, from 53.4°
for `eps 1.50 − 0.40j` (§1); `ray_state_update` re-derives it. A wave launched inside a lossy
medium, a wedge, and a caller without state keep it. The alternative, no verdict wherever the
root's argument is not real, errs the other way and was measured (§4): it opens faces that are
closed for a wave whose invariant exceeds one, and bends the segment there.

**A rigid body with a loss column is not closed.** The rule of §1 reads a lossy second medium
as one that admits a wave, and the permittivity of a rigid body is an impedance mapping, not a
propagation index. Rigid materials are lossless, and stay closed: the first face reflects
`1 − tf`. One that is given a conductivity or a resonance opens, its face is a transformer
with an all but evanescent wave behind it, and it reflects everything: on a 200 mm wall
`eps 2.401e-9` with `tf` 0.3 the first face reflects 0.700000 lossless and 1.000000 at any
loss tangent from 1e-9 to 0.04, a step of 1.55 dB at a loss of zero. Entry and exit together
read −10.46 dB lossless, −6.5, −19.7 and −33.3 dB at 10°, 45° and 75° for a tangent of 1e-3 or
0.04, which is the tunnelling of a layer of near-zero permittivity, and −139 to −183 dB at
1e-6 and 1e-9, where the two crossings no longer cancel. Nothing is amplified, which is
asserted in table F of the bare-tf case; the numbers hold nothing of the wall's impedance. No
column says that a material is not a wave medium; `m > 0` is the nearest and does not cover a
rigid material without it. Recorded in section J of the lossy-verdict case. Do not put a loss
column on a rigid material; a resonance column on one is issue 1.

**The two tracers through two solids on a resolved ray have no reference of their own.** The
five-event row compares the undeviated with the refracting tracer and says they are one wave.
That wave is one term of the wall's reflected sum (§1); the sum has a reference, the single
path has none.

**The first path through a layered wall is not the wall.** With every series resolved a wall of
two traced solids, or of a host with a sheet inside it, is the sum of the paths a run follows,
and the first one alone is off by the table in §1, up to −4.88 dB for a lossless second layer at
85°. A run that stops early, on `max_no_interactions`, on the reflection limit or on
`min_gain_dB`, keeps that much of it: each order costs four interactions, and a lossless second
layer at 85° is +1.37 dB off after 20 and +0.04 dB after 60. The later paths leave the wall
displaced and delayed, so a receiver has to collect them.

**Two regimes of one wall differ by what a gain floor cuts.** Where a series is declined its
bounces are many weak paths, and `min_gain_dB` removes them one by one where the summed regime
loses nothing. A sheet inside a wall, the default threshold over every series summed, lossy host
at 85°: −0.215 dB at a floor of −100 dB, +0.086, −0.036 and +0.005 dB at −120, −140 and −160 dB,
and +0.004 dB with 20 interactions and no floor; in reflection on the lossless host at 75°,
−0.052, +0.017, −0.002 and +0.001 dB. The floor acts on the accumulated path gain, free-space
loss included. That is why the two regimes of that wall are asserted to 60° only where the
floor is −100 dB, and at every angle without one.
With every series declined the paths of two coupled cavities multiply, and a lossless wall
near grazing does not converge within any limit that is affordable: 50 mm | 12 mm sheet | 50 mm
of `eps 1.50` at 85°, a threshold of 1, 60 interactions and a floor of −140 dB launches
2.2 million rays for one and reads −10.3 dB; at the default threshold the same wall takes 8
paths and is on its matrix.

**The followed mark shares its word with the deferred medium.** A nested or overlapping object
met on a marked leg stores its medium in the buffer word, and the mark is gone: the event that
closes that leg forms its series against the previous-medium word as it stands, which after a
bare-face reflection is not the mirror behind the ray. Nested geometry together with a declined
series and a window of the threshold. The words are asserted in the marks case, and the series
such a leg forms is recorded there at `eps` 0: +0.08 dB and −0.09 dB at 20° and 60° on the
host `eps 1.50 − 0.40j`, +0.02 dB and −0.54 dB on `eps 1.50`, around a 20 mm object in 100 mm.
The stack code shares the word in the same way: a nested object met behind a stack replaces it,
and that leg judges its series against the last layer alone, the 0.78 dB of §4.

**A verdict is the base frequency's.** Where the base frequency sums a series, every frequency
of the run sums it, also one whose own `ρ` is below the threshold and one whose denominator is
near a pole, which is exact in both; where the base frequency follows, every frequency
follows. The number of rays of a run is therefore the base frequency's choice. `ray_progress`
hands the verdict on; another caller that runs several frequencies has to do the same, and one
that does not gets a verdict per frequency as before.

**The path sums that have no fixture.** Two solids with a far coating, three solids, and every
EM channel: the sums are asserted in scalar mode for two solids, a wall coated on either side or
lined on both, a sheet in front of a wall and a wall with a sheet inside it. The reflect port of
a sheet, its reference plane and its echo gain, is asserted per channel on single events and in
scalar mode on every sum. Three solids have several paths per order.
The straight path through three solids is recorded against the whole wall, up to −3.52 dB for
`eps 5.24 | 1.50 | 5.24` at 40°.

**A followed sum and a summed series are one wall also where a mirror reflects more than one.**
Nothing clamps a mirror, so the two differ by the relaunch offsets alone; the row that carried
0.019 dB between two thresholds of one wall reads 0.0000.

**The limit of a path sum carries each path's own relaunch offsets.** The reference is given the
thickness the first path crossed. With the refracting tracer and a lossless second layer at 85°
the limit is 4.9e-3 dB off for it, and a reflected limit that is 16 dB below its front-face term
reads 1.3e-3 dB. Through `ray_progress` the reference has the nominal thickness, and the offset
is a phase that grows with frequency: a lossless wall lined on both faces reads 8.9e-3 dB at
6 GHz and 10°. A sheet against its traced solid of 20 mm reads 3.2e-3 dB on the transmit port at
85° for the same reason.

**The mid-plane is a convention of the mesh.** A sheet with one medium on both sides is centred
on its face (§1). A pane meshed on one of its surfaces is then half its thickness from where it
stands: a phase of `k₀ cos θ·thk` on its reflections and a cavity that is half a thickness
longer on one side and shorter on the other, which is a power wherever something stands behind
it. The orientation of a sheet's face is not read, so there is no way to say otherwise in the
mesh; a layer that has to stand on one side of its face is a coating, or a traced solid.

**The reference plane is a phase per frequency and no geometry.** A coating reflects as a layer
whose front face is `thk` in front of the mesh plane, and the ray still turns around at the
mesh plane: the path length and the delay of the reflected path are the mesh plane's. In a
lossy host the same factor is a magnitude above one, on the reflect port and on a mirror of the
host's series alike.

**A sheet inside a host and its traced counterpart differ by the host's interface gain.** A
sheet event charges the host nothing, the host having been entered earlier; a traced solid in
its place charges the host again on the way back into it. So the equality of a sheet with its
traced solid, on either port, is for a sheet in air and for a coating.

**A sheet that does not fit its host is capped, not refused.** A sheet thicker than the host
it lies in is not a valid geometry, and nothing stops a mesh from holding one. Each half of it
then takes the host on its side and no more, the sheet keeps its full thickness, and the wall
comes out longer than the mesh. Asserted for a sheet up to three times as thick as its host,
lossy and lossless, both tracers, to 85°: every coefficient finite, the wall passing less than
it received with the host's series declined and summed; and over every path for sheets and
stacks off the middle of their host.

**A stack thicker than its whole host, met from its long side, passes more than it received.**
The second face of a stack's hit is its other sheet, so the event takes the leg in front for
both sides, and behind the stack there may be less. Where the stack is thicker than the two
legs together it takes out more host than the ray walked: 110 mm, 140 mm and 160 mm of
`eps 1.50 | eps 1.60` with 90 mm of `eps 1.50 − 0.40j` in front and 10 mm behind read +0.24 dB,
+1.16 dB and +1.81 dB on the straight path and 1.05 to 1.62 on the two ports together, and
12 mm with 8 mm in front and 2 mm behind 1.011. From its short side the same stack is capped
and passes less. A stack that fits is exact from both sides, and one that does not fit in
front is capped. The event cannot know better: the intersector reports two faces and both are
the stack, the destination is the rest of the segment, and the hit count says only that a face
follows. A total capped at the leg in front would keep this geometry passive and leave a stack
that fits but is thicker than that leg short; it was not measured. Recorded in the `long` rows
of the budget wall case. A collapsed duplicate inside a host takes the same rule.

**An opaque layer transmits at its own level, down to underflow.** The solver carries no loss
cap: a layer with 2000 dB of one-way loss reads −2045 dB on the transmit port, on the reference,
and its reflect port is untouched. Beyond about 6000 dB the amplitude underflows to an exact
zero. The path of the `alpha` column and of a mass-law layer's loss keeps its saturation at
400 dB, which acts only where the in-layer cosine has no real part (issue 3).

**A verdict of "sum" on a leg known not parallel is not reachable by a consistent caller.**
The base frequency of a run sums on such a leg only at a threshold of zero, and there the gate
is not read with or without a verdict. `ray_progress` hands one threshold to every frequency.
The guard is kept: it is what makes a handed-in verdict mean what it says.

**A transparent transition between two indices of one material sets the fix-flag.** The exit
face names the other index, which the same-material test accepts and the mismatch test does
not: resolved type 89 where one index gives 73. A flag; the coefficient, the words and the
followed mark are those of one index.

**With the refracting tracer the segment is bent at the face of a deferred body.**
`ray_mesh_interact` holds no state and refracts at the face of a nested or overlapping body as
if it were entered from air, where the redo passes it and keeps the direction. Inside an
overlap of `eps 5.24` on `eps 1.50` the segment runs at 10.27° where the wave runs at 12.62°,
for a ray at 30°; behind the overlap direction and segment agree again, the faces being
parallel. The level of the crossing out of the overlap is the abutting one to 8e-7 dB; its
phase is the field at another polyline. Rare geometry, and with issue 8.

**The crossing into a deferred medium keeps the direction formed for medium | air.** Where the
wave was not bent the rewrite replaces the coefficient and returns. For a mass-law body that
direction is the held one and is right. For a rigid body without the mass column it is along
the normal, which is the limitation above for every exit of such a body.

**A mass-law layer becoming closed is a step.** Below the angle at which the invariant reaches
its `ε μ` a mass-law layer is crossed by two Fresnel faces, above it it is a closed plate (§1). A
sheet of `eps 0.50` with `tf` 0.2 inside `eps 2.50` reads −10.78 dB at 44.9° and −13.98 dB at
45.1° on its event, and inside `eps 2.50 − 0.80j` −11.26 dB and −14.55 dB. It is what the wave
becoming a leak is at a bare face, above. A rigid layer is past that angle from 0.003° on.
Table F1 of the closed-plate case, bounded below one.

**A rigid lining without `tf` passes the Fresnel exit of a held wave.** Met from the wall, a
rigid lining without `tf` on a rigid wall with `tf` passes −45.13, −45.88, −49.27 and −53.45 dB
at 10°, 30°, 60° and 75° before its mass term, which is the `1 − R0 = 4 z cos θ` of its face to
air; met from its own side it passes exactly nothing, its first face being flat. Two traced
solids do the same. It is the flat first face, above, at `tf` 0. A flat far face for a held
wave would close it and would part the sheet from its solid twin by the same term; not
measured. Table H of the held-lining case.

**The level of a held chain has no reference but its solid twin.** The crossings of a held
wave are formed from permittivities that are impedance mappings. What is asserted is
consistency: no gain, one wall from its two sides, one wall as a sheet and as a solid, no
step at an angle formed from two mappings.

**The reflectance of a transformed face is the pair rule's only where its two sides are in
phase.** `tf` moves the modulus of the admittance ratio of a face, and a real ratio cannot
match a face whose ratio is complex. That is every face with a wave on both sides: of a
coherent sheet and of a traced solid alike, so the reflection of a lossy wall with `tf > 0`
seen from the room is not `(1 − tf)·R0`. From the Fresnel formulas, one face seen from air,
the transformer against `(1 − tf)·R0`: `eps 5.24 − 0.63j` +0.03 dB at `tf` 0.5 and +0.1 to
+0.5 dB at 0.9; `eps 1.50 − 0.40j` +1.0 dB and +5.3 dB at normal incidence, where `R0` is
0.016; `eps 0.31 − 0.20j` at 75°, 0.79 where the pair rule has 0.42 at `tf` 0.5 and 0.74 where
it has 0.08 at 0.9. For `tf < 0` the two agree to 0.3 dB. A lossless wall is on the pair rule
exactly. A face with an evanescent wave behind it and an open verdict reflects everything at
every `tf`. It is what keeps a sheet and a solid from creating energy; a closed bare face and
the faces of a mass-law sheet keep `(1 − tf)·R0` (issue 6 for the latter).

**`tf = +1` closes the reflect port of a coherent sheet only where its faces are in phase.**
Measured on 6 mm in air and inside `eps 2.50`, the largest over 0°, 45° and 75°: exactly zero
for the lossless `eps 5.24`; 3.4e-4 for `eps 5.24 − 0.63j`, 9.4e-4 for `1.50 − 0.40j`, 1.8e-3
for `0.90 − 0.20j`, 9.7e-3 for `0.31 − 0.20j`, 1.4e-2 for `5.24 − 5j`, and 9.8e-3 for the
lossless `eps 0.31` past its critical angle. On a lossy wall a lossy sheet leaves −24 dB up to
50° and −18.7 dB at 85°. `tf = −1` closes the transmit port exactly, for every sheet.

**A forced face steps where its admittance ratio has modulus one.** `tf < 0` moves a face away
from its neighbour, and at a ratio of modulus one "away" changes direction. Between media in
phase that is a matched boundary, where a forced reflection changes sign and nothing else.
Where they are not in phase both ports of a wall step: 6 mm of `eps 0.90 − 0.20j` in air,
59.99° against 60.01°, TE and scalar, −0.29 dB at `tf` −0.1, −1.05 dB at −0.5 and −2.52 dB at
−0.9, on the transmit and on the reflect port alike, as a sheet and as a traced solid; TM has
its own angle. One face alone does not step in power, 0.001 dB; the step is the cavity of the
two faces. Both sides of the step add to less than one. `tf > 0` has none. A forcing that
fades out at modulus one would be continuous and would no longer give `R' = |tf|` at a matched
boundary, which is settled; it was not measured. Recorded in table D2 of the sheet-tf case and
in table H of the bare-tf case.

**Exactly at the tie a sheet reads a value of its own.** At 60.00° that sheet reads −9.61 dB
at `tf` −0.5 between neighbours of −0.18 and −1.22 dB, and −25.58 dB at −0.9: its two faces
form the ratio from two spellings of the sine and rounding puts them on opposite sides. The
traced solid sits on one side, −1.22 dB. A single point in the angle; negligible, recorded in
table H of the bare-tf case and dropped.

**The two ports of a sheet event do not add to at most one next to a lossy medium.** Inside
`eps 2.50 − 0.80j`, the sheet event with its host leg of 10 mm: 1.037 for a lossy sheet and
1.233 for the lossless `eps 5.24`, both with `tf` −0.5 at 85°; at the exit of that host the
characteristic matrix itself, with no `tf` anywhere, reads 1.206 for a 6 mm lining of
`eps 5.24` at 85°. The incident and the reflected wave interfere in a lossy medium and `T + R`
is not what a face conserves there; the transformer conserves the flux across the face by
construction, which is not measured for a lossy near medium. Each port is asserted.

**A rigid sheet without the mass column tunnels, with or without `tf`.** The rule of a coherent
sheet reads a permittivity as a wave medium, and a rigid body of the one-parameter mapping is rarer
than any invariant from 0.003° on. A 22 mm sheet of `eps 2.1609e-8` with `m = 0` in air passes
−0.23, −0.30 and −0.88 dB at 10°, 30° and 60° without `tf` and −0.14, −0.29 and −0.83 dB with
`tf` 0.2; inside `eps 2.50` −0.55, −0.61 and −0.79 dB and −0.26, −0.50 and −0.77 dB; as a
lining of it −0.58, −0.72 and −1.67 dB without `tf`. The number holds nothing of the wall's
impedance and is not acoustics. With the mass column the sheet is a closed mass-law layer:
exactly nothing without `tf`, −13.98 dB with `tf` 0.2, its two leaks. No column but `m > 0`
says that a permittivity is an impedance mapping. Give a rigid sheet the mass column; a rigid
wall meshed as a solid is not reached. Asserted from both sides in table G of the flat-branch
case.

**A layer rarer than the invariant is a wave medium as a sheet and a closed body as a traced
solid.** A coherent sheet tunnels and decays with its thickness, in air, inside a host and
under a closed outer pair: 10 mm of `eps 0.31` with `tf` 0.5 on a rigid wall with `tf` 0.3 at
60° passes −2.13, −2.79, −12.13 and −24.06 dB at 1, 10, 100 and 200 mm. With the mass column it
is a closed plate, its two leaks at every thickness, −4.88 dB. A lossless traced solid of that
material is closed by the verdict of its first face, as a rigid wall is, with or without the
column: it is entered as a leak, charged no decay, and its coherent sums read −4.10, −4.30,
−6.04 and −6.55 dB with path powers of one. So a lossless rare layer without the mass column
parts from its traced solid, and the traced solid steps in its own loss, by 2.0 dB at 1 mm and
17.5 dB at 200 mm between a loss of 1e-6, where it is the sheet, and zero. A meshed body in
which the wave is evanescent cannot be followed by a ray; give such a layer the mass column, or
mesh it as a sheet. Measured with a scratch probe; not asserted.

**A sheet of no thickness on a leaking wall is not the bare wall.** The leak of a closed face
is formed on the immittance of the medium in front of it, the pair that forms the face. A thin
dense film on a rigid wall with `tf` therefore raises the wall's leak and a thin rare one lowers
it, as a traced solid does: 1 mm of `eps 0.31 − 0.20j` without `tf` on a wall with `tf` 0.3 at
60° passes −5.79 dB where the bare wall passes −5.23 dB, at formula level. A sheet of air is the
bare wall exactly. Referring the leak to the medium in front of the sheet would part every
sheet from its traced solid.

**A mass-law medium is given a real immittance at a closed bare boundary.** It has no wave
inside, so the pad of its closed face is formed as in front of a lossless medium. For a
lossless mass-law medium that is the same number; for a lossy one, which no row of the suite
builds, the face reflects `R'` times the Fresnel reflectance and passes the rest, and the two
crossings of a mass-law sheet stay bounded. Without it a lossy mass-law sheet on a rigid wall
read up to 1.84 on a what-if. Issue 6 is the open question about such a material.

**A sum of path powers is not an energy statement next to a lossy medium.** A crossing out of a
lossy medium is normalised to the forward wave and is not bounded by one, and the pad of a
closed face in such a medium passes the flux of the field and not of the forward wave. The
path powers of a lossy traced layer on a rigid wall with `tf` add to up to 3.5 where its
coherent sum is 0.997, summed and followed alike. Coherent sums are the statement.

**The composite of a sheet in a lossy host can read above 0 dB.** It stands for the layer in
place of its thickness of host and divides that host out. `diag_cm_stack` for
`eps 2.50 − 0.80j | 6 mm eps 0.31 | eps 2.50 − 0.80j` without `tf` reads +0.12, +0.10 and
+0.08 dB in TE and +0.20, +0.36 and +0.33 dB in TM at 20°, 45° and 60°, with the host's 6 mm
worth −0.28 dB to −0.32 dB one way. A lossless layer in place of a lossy one is a gain
against the host, not against what was launched; the event with its host leg is below 0 dB,
and that is where passivity is asserted.

**A reflection above one is not a gain, and a ray can carry it.** TM next to a lossy medium
reads a reflectance above one: 1.178 for `eps 0.31` seen from `eps 2.50 − 0.80j` at 45°; in
scalar mode and TE it needs a permeability, a real angle in a lossy medium, or a layer mirror
seen from a lossy medium rarer than air (1.38 for 6 mm of `eps 2.50` seen from
`eps 0.31 − 0.20j` at 75°). Nothing caps it, at a sheet, a mirror or a bare face. A ray inside
a lossy solid can therefore carry an amplitude above one after a reflection; the traversal of
that solid takes it back, and every port seen from a lossless medium is at most one. Where the
bounces of such a cavity are followed and its round trip exceeds one, they grow (below).

**Exactly at its critical angle a sheet without `tf` is declined.** 6 mm of `eps 0.25` in
air is on the matrix, −0.0129 dB, at 29.999° and 30.001°. At 30.000° the cosine in the sheet is
zero, the raw exit amplitude of each face is zero and the denominator is zero, so the pair is
0/0 and the solver returns false: the event keeps what `ray_mesh_interact` wrote, which for its
own composite is nothing on either port, and an ordered stack kills the ray. Read from the code
and not measured. A single point in the angle; negligible, recorded and dropped.

**The two ports of the matrix add to more than one in a lossy host.** `diag_cm_stack` for S
inside `eps 2.50 − 2.0j` reads a sum of 1.0011 in TM, each port below 0 dB: the cross term of
the incident and the reflected wave in a lossy medium, which is not a flux. The sum of the
two ports is asserted for a lossless host only.

**A coherent sum over the paths of a leak is not an energy statement.** A lossless traced
solid rarer than air with `tf` and no mass column, past its critical angle, is entered by the
flat leak and left from inside by a Fresnel crossing, two rules for one face (the first face of
a leak, above). Every event conserves: the powers of all paths add to 1.000000 at every angle
and thickness. The coherent sums of the two ports read more: `eps 0.31` with `tf` 0.5, 50 mm,
1.02 at 45°, 1.35 at 75° and 1.69 at 85°, and 1.16 to 1.49 over 1 to 200 mm at 75°. The excess
is interference between paths whose phases no medium defines. The same class is a rigid wall
with `tf` and no mass column behind a porous solid: through `ray_progress` its coherent sums
read 1.16 together at 30° and 0.5 GHz and its power sums 1.37. A repair needs the tracer to
know that a wave is a leak, a state bit, and none is left; with the mass column the bounces pay
the mass term and are gone. Not asserted.

**A followed bounce series can diverge where the summed one is exact.** In a lossy layer rarer
than air the wave is strongly inhomogeneous near grazing and a reflector seen from inside
reflects more than one in amplitude, which is the exact Fresnel result and no gain. Between a
coating and its far face the round trip `|r1 r2 phi²|` of 6 mm of `eps 0.31 − 0.20j` is 0.79 at
60°, 1.03 at 75° and 1.19 at 85°, from the formulas and from the library's events alike, and
1.13 for `eps 0.90 − 0.20j` at 85°. The bounces grow, and a run that follows them reads 3e30
after 400 events with no `tf` anywhere; the closed form is finite. With no mirror clamped the round trip of such a leg is above any threshold the tracer
accepts, which is at most one, so its first leg is summed; measured on the walker at a
threshold of 1.5, where 59 of 90 diverging rows became finite, and not through `ray_progress`.
A leg that carries the followed mark, a leg flagged not parallel and a mass-law body still
follow, and can still grow there.
Through `ray_progress` such a leg is summed at every threshold the API accepts, 0 to 1: the
lined 6 mm layer is one path on its matrix at 1e-4 dB, its two ports together 0.976 to 0.992.
What still grows there is the coupling between two cavities: with a sheet inside that host, 6 mm
`eps 5.24` | 20 mm | 6 mm `eps 5.24 − 0.63j` | 20 mm | 6 mm `eps 5.24` of `eps 0.31 − 0.20j`,
the paths that pass from one half to the other form a sequence whose ratio exceeds one, and
the sums read +6.4 dB, +45.7 dB and +88.6 dB above the matrix at 60°, 75° and 85° after 20
interactions with every series summed. A wall as a sum of paths has no finite sum there, and
no event holds the whole wall. A lossy medium rarer than air between strong reflectors;
recorded, with no fixture.

**The hybrid propagation factor where no cosine is recovered.** Where the recovery declines,
`qd_slab_resolve` and the traversal take the magnitude of a leg from `medium_gain` and its phase
from the real index, while a recovered leg and the solver take both from `exp(−j k₀ n L cos θ)`.
The two forms are one expression but for two rounded constants in `medium_gain`, 8.686 for
`20 / ln 10` and 0.0477135 for `c₀ / (2π·10⁹)`: 2.5e-5 dB on the rows that show it. At exactly
normal incidence the traced and the geometric cosine coincide, with the undeviated tracer
nothing else says the wave was bent, and the leg takes this form, which puts 2.5e-5 dB against
a 1e-6 dB bound; both traced-chain fixtures therefore start their sweep at 10°. Exact constants
would remove it and were measured and not taken: 517 rows in 29 tables move and 424 assertions
fail whose reference helpers carry the rounded constants. Negligible; recorded.

**A ray launched inside a host is a leg the state does not know to be bent.** With types 1 and
4 its legs take the forms of a wave that was not bent, in the host and in a body behind it that
is entered without a bend, and a series summed on such a leg is off the sum of its followed
bounces: 0.13 to 3.57 on a lossless wall that returns everything, where the followed bounces
read one. A source inside a medium is not a use case; an instrument that seeds a state has to
seed the bent flag and use type 5, or read its followed rows only.

**A negative resonance strength is a gain medium.** `resS < 0` with a finite `resQ` gives the
pole a positive imaginary part at every frequency, and the medium amplifies as its permittivity
says: `ε` 4 with `resF` 1 GHz, `resQ` 5 and `resS` −0.5 reads 2.11 on the two ports of a 37.5 mm
sheet at normal incidence and 1.63 at 60°, and a followed series through its traced solid
diverges. Nothing validates `resS`; with `resQ` 1e300 the gain is 2e-300, which the fixtures use
for a lossless pole. Not a passive material.

---

## 6. Order

Before anything else: regenerate `test.txt` (see the status at the top).

No open item reaches a material without the mass column; all are acoustics. Issue 1 goes first:
it creates energy in the use the resonance columns are meant for, and it starts with an analysis,
not a patch.

1. **Issue 1.** A first-principles analysis and a literature review (the questions in §3.1), a
   decision proposal per question, then fixtures and a change. It touches the verdict of a
   closed mass-law layer and the mass-law convention, which are settled, so the decision comes
   before any patch; the result is measured on issue 6 as well.
2. **Issue 2.** Analysis with a marker and the reverse direction as the reference, then a
   decision per form. It touches the direction of a leak into a medium without the mass
   column, which is settled, so the decision comes before any patch.
3. **Issue 3.** A convention decision on the path of a lumped plate and of a tunnelling layer,
   then one path in the places that form it.
4. **Issue 4.** A decision; one condition if taken, an entry in §5 if not.
5. **Issue 5.** A convention decision, on both ports of a bonded pair.
6. **Issue 6.** A fixture first, then a decision: the transformer at the open faces of a
   mass-law sheet removes a rule, the quotient and the last clamp together.
7. **Issue 7.** A flag; with whatever touches the undeviated rule first, which may be issue 2.
8. **Issue 8.** Analysis and a what-if build before any decision; reopens the layout of the
   buffer word, whose bit 14 now carries the stack code as well as the mark. To be dropped if
   it is not feasible, with the limitation left in §5.
9. **Issue 9.** Out of scope for now; with whichever of the above last changes what the
   document has to say.

A rule for every item: the code is to stay small. A wider gate and one rule are preferred
over an exclusion kept to protect existing digits, bit-identity of existing rows is not a
requirement, and a row may move within reasonable bounds if the reason is physical; the rows
that move are measured on a what-if, explained before the source is touched, and their
references adjusted. A table that moves is read in the unit it reports, column by column and
row by row, before the source is touched: a difference in a real part hid a step of 55 dB
once, and a summary table that prints the worst row of a sweep moves in the angle and the
thickness of that row without any level moving. The acceptance criteria of the older items
below that still say "bit-identical" are to be read that way.

A rule for what enters section 2 keeps the list from growing with every measurement: an issue
is a wrong result in a valid geometry for which a fix is conceivable or cannot be ruled out
without analysis; a limitation with no fix in sight is recorded in section 5 with its measured
size, and so is anything negligible. A finding that belongs to an open item is added to that
item and does not open a new one.
A rule for a premise: a statement that something is not available - a bit, a code point, a
quantity an event could know - is checked against the decode and a marker before it decides a
design. One such statement, taken over unmeasured, cost a followed leg and a session.

## 7. Acceptance

Each assertion below replaces or joins the sanity checks in the case that measures it.

- *A mass-law panel with a resonance*: a window with the column as a sheet and as a traced solid,
  at and away from `resF`, both ports together at most one, sheet on solid at the relaunch bound,
  no step in angle, the chosen model against its closed form; the window without the column and
  every row without the mass column where it is.
- *A layer without the mass column behind a held wave*: a coherent lining on a rigid wall on
  one transmission from its two sides; a traced coherent layer behind a rigid wall passing
  what it passes in front of it; every row without a mass-law wall where it is.
- *The path of `alpha`*: a rigid mass-law sheet with the column on the traced solid of its
  thickness, in air, as a lining and behind a held wave; a tunnelling coherent sheet with the
  column passing what the convention says; every row without the column where it is.
- *Both faces upward*, if taken: a lining behind a held wave on its solid twin in both
  orders, with the rows that move listed.
- *Bonded interface gain*: the chosen convention for `att` across a bonded pair, asserted on all
  three pairings.
- *A lossy mass-law layer with `tf`*: as a sheet against its traced solid with a mass term
  that is worth something, both ports together at most one, or the convention stated; no
  reflectance clamped anywhere if the transformer is taken.
- *Deferred medium*, if taken: a wall with a nested object behind a bonded solid on the plate
  from both sides, with every other row of the plate-depth case where it is.
- *Documentation*, when it is taken up: the material-model document stating the three `tf` rules
  and the faces that take each, the energy convention, the sheet under total reflection, a
  mass-law layer in a stack, the closed mass-law layer, a sheet behind a held wave, the cosine
  of a mass term, the rare sheet and the rule that a rigid sheet carries the mass column, the
  one solver and that nothing is clamped, the state-word layout and the material limit; the
  API markdown regenerated.
- Everything in the table in §1 stays where it is. Those bounds are numerical, not physical, and a
  failure means an approximation has been reintroduced.
