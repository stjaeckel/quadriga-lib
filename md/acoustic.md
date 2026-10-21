# The acoustic add-on — the machine, what the physics requires, candidate changes

Acoustics in this library is an add-on to a wave engine. It has to be accurate on the reflect side and
make reasonable predictions on the transmit side. This document is about the machine, not about
material parameters. Per physical quantity it states what the physics requires, what the machine does
today, and which changes to the machine the literature supports, including removals. Parameter values
come afterwards. No acoustic materials are defined for the library today (`data/materials_ac.csv` and its
copy under `tools/acoustic_parameter_mapping` are out of date and are no reference for anything).

The boundary is the **mass column**: everything with `m > 0` is acoustic and described here; everything
without it is wave physics, settled and guarded, in `md/wave.md`. Every acoustic change stays behind
`m > 0` (or behind a column no wave case carries); the `[wave]` baseline and `tests/data/wave_twins.tsv`
are the check. The material model lives in `src/ray_mesh_interact.cpp`, and so does the acoustic work.
The reflect side comes first (P6); scattering is out of scope (P9).

**Sources.** Physics statements cite the literature review `/sjc/accoustic_research/review.md`
by source key in square brackets; the keys resolve in `references/index.csv` there, and "review 4.2" is a
section of that file. "Report 3.4" is a section of `md/acoustic_round1_report.md`. "Derived here" is a
closed-form evaluation made for this document, "measured here" a run of the library; both are in
`build_linux/scratch/ac2` (section 11). The review's `data/material_properties.csv` names reference
sources (section 9) and sets no parameter.

First step: `md/wave.md`, section 0. The acoustic half:
`QD_DIAG_RAY=1 OMP_NUM_THREADS=12 ./build_linux/test_bin "[acoustic]" > test_acoustic.txt` → `All tests passed (42741 assertions in 59 test cases)`.
Its cases are in `test_acoustic_mesh_interact.cpp`, `test_acoustic_progress.cpp`,
`test_acoustic_state_update.cpp` and `test_acoustic_leaf.cpp` (the leaves of 2.8); 16 of them are "<name> - the mass-law part" of a wave case. The appendix
lists every case with the section that states it. The machine described in section 2 is the one after
acoustic round 1 (decisions of section 10, taken 2026-10-15; report section 5).

## 1. The acoustic mapping

Frequencies are mapped `f_radio = f_acoustic · c_light / c_sound` (1 kHz ≈ 0.874 GHz).

**The two-parameter mapping** (N7, decided for every acoustic material): a material with impedance `Z`
and sound speed `c` is mapped as `ε = n·z`, `μ = n/z` with `z = Z_air/Z` and `n = c_air/c`, in the columns
only (no code reads the mapping). `ε` is then the compressibility ratio `ρ₀c₀²/(ρc²)` and `μ` the density
ratio `ρ/ρ₀` (derived here from the definitions). Air | water is exact as a traced half-space, in level and
critical angle (13.38°; the phase past it 4.33e-4° at 13.38°, 1.53° at 85°, report 1.8); a hard wall reflects with a phase near zero (concrete: 0.009° at 10°, 0.104° at
60°); a porous layer on it absorbs as the exact layer (Delany-Bazley, 20 mm, 500 Hz: within 0.0008 in α at
0° to 80°). Rigid bodies then have a critical angle of a few degrees (glass 3.48°, gypsum 9.44°); below it a
traced rigid solid holds every tracer. Asserted in: *Two-parameter mapping - air | water, a hard wall's
phase, a porous layer on it and every tracer below its critical angle*.

A thin layer of a wave medium in this mapping has the series impedance `j·k₀·h·μ = jωm″/Z₀` at any angle,
the mass reactance of a limp panel (derived here from the layer matrix; the cosine cancels). Measured here:
a 4 mm glass sheet with `m = 0` (`a` 1.767e−6, `e` 2083) passes the limp mass law
`1/(1 + (ωm″cos θ/(2Z₀))²)` within 0.002 dB at 0° to 85° at 500 Hz (−31.634 dB at 0°, −10.815 dB at 85°), and
within 0.0078 dB at 50 Hz–1 kHz; its phase is the layer's (the zero-thickness law lacks the air path over `h`,
4.2° at 1 kHz). Asserted in: *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf*. The route
built on it is 2.8 (M9).

**The one-parameter mapping** (the older form): `ε' = (Z_air / Z)²`, `μ = 1`. Rigid bodies are then
optically rarer than air by eight to eleven orders (glass 8.5e-10 for Z = 1.41e7 rayl, concrete 2.4e-9,
wood 2.2e-8), air | rigid is past its critical angle from about 0.003°, and a rigid half-space reflects with
a phase of twice the angle (+20° at 10° … +170° at 85° for `eps 2.401e-9`; report 3.5). A thin layer in it
carries no mass reactance. Most fixtures use it; the machine's statements hold in both mappings (round 1
measured both, report 3.4 to 3.7). Candidate M8 retires it for materials.

For a body with the mass column the mapping sets the reflection of its faces and its critical angle; its
transmission comes from the mass term (2.1) and the wave inside it is held (2.4). A mass-law material's
`ε`, `μ` are calibrated for surface impedance only.

## 2. The machine as built

Everything here is gated on `m > 0`.

### 2.1 The mass term

`Material::medium_gain` forms it from a surface-mass surrogate `M` with the angle removed (the normal
thickness) and an angle factor `c` (the cosine the wave has in air):

    X₀ = (f/fRef)·M        A = 20·log10 X₀
    TL = A·log10(1 + (X₀·c)²) / log10(1 + X₀²)

**`m > 0` switches the mass law on, 20 dB per decade**, whatever its value (M10, built 2026-10-17; until then `m`
was a fitted slope, `X₀ = ((f/fRef)·M)^(m/20)`, fitted 1.64 to 35.13 in round 1). For the infinite plate (`side` 0, below)
the term is `A` at normal incidence, zero at tangency, 20 dB per decade of `cos θ` (the cosine enters through the
fluid's admittance, not the panel's impedance). The term is zero below `X₀ = 1` (a kink in frequency, bounded by 3 dB within a decade of the knee).
With `fRef = (f_GHz/f_Hz)·ρ₀c/(π·ρ)`, a material constant (4.58e-5 GHz for glass), `X₀ = ωm′/(2ρ₀c)` is the
physical impedance ratio of a limp panel and `TL` is the limp mass law `10·log10(1 + (X₀·c)²)` wherever
`X₀ ≫ 1` [Davy_2009_SingleLeafCremer, Eqs. 3, 13]. A traced solid's leg shorter than 1.5 mm carries no term;
a sheet carries it at any thickness. The public `medium_gain` supplies `c = 1`: it returns `A` for `side` 0, and `A` up to the panel's low-frequency term
otherwise (below). Asserted in:
*Thin sheet - mass-law angle dependence* (`m` 12, which passes what `m` 20 passes); *Mass law - m switches the limp
mass law on* (a 4 mm pane with `fRef` from its density against the limp mass law, `m` 12, 20, 30 bitwise equal);
*Acoustic - medium_gain in-medium loss*.

**The finite panel (`side`, decided and built 2026-10-19; `md/grazing_report.md`).** The cosine in the term is the
radiation of the plate's forced wave: an infinite plate driven by a wave of trace wavenumber `k·sin θ` radiates
`ρ₀c/cos θ` per unit velocity on each face. Near grazing that load outgrows any mass and the infinite plate turns
transparent (`τ → 1`); below coincidence, where `X₀ > 1/cos²78° ≈ 23` (a 4 mm pane from about 300 Hz, a 152 mm
wall at every octave), more than half of what it transmits comes from the last 12° before grazing. A panel of finite size radiates its forced wave as a source of finite length: its forced radiation
efficiency `σ` is `1/cos θ` up to an angle that grows with the size in wavelengths (the finite plate leaves the
infinite one at `cos θ ≈ 1.9/√(kL)`), then stays bounded (about `0.38·√(k·side)` at grazing), and below about
`kL ≈ 4` it is under one at every angle. The library takes `σ` from Davy's closed form for a square of side `side`
[Davy_2009_ForcedRadiationEfficiency, Eqs. 19, 28, 34, 36, 37], replaces the cosine by `1/σ` and pays the finite plate's
gain `1/(σ cos θ)`: `TL = A·log10(1 + (X₀/σ)²)/log10(1 + X₀²) + 10·log10(σ·cos θ)`, clipped at zero. The column
`side` = √(panel area) in m, 2a = `side`, S = `side²` (√area is within 0.45 dB of Davy's form with the true area and
perimeter up to an aspect ratio of 3.3 : 1, 1.5 dB at 10 : 1); 3.29 m (a 4.0 × 2.7 m wall) where the column is absent;
0 is the infinite plate, bitwise; a negative side is refused; read only where `m > 0`. The plate's own impedance (the
resonance and the bending stiffness below) stays at the cosine in air. Against the windowed finite plate of
`tools/acoustic/gzlib.py` (azimuth mean), the form is within 0.4 dB diffuse and 0.8 dB per angle on single and double
leaves at sizes from 1.23 × 1.48 m to 10 × 3 m. What it changes: near grazing the panel keeps insulating (4 mm glass,
500 Hz, 89°: a term of 13.2 dB against 1.6 dB), and at 31.5 Hz a small panel insulates beyond the infinite plate even at
normal incidence (+7.8 dB for a 1.35 m pane, +3.0 dB for a 3.29 m concrete wall). Asserted in: *Finite plate - a
mass-law panel of side s radiates with Davy's forced radiation efficiency*; *Finite plate - side 0 is the infinite
plate, a table without the column is side 3.29, a wave medium ignores it, a negative side is refused, a bonded run
takes the side of the body that forms its term*; *Mass law - m switches the limp mass law on* and *Plate rule - a
rigid mass-law panel passes the incoherent series of its face and its plate and reflects what its face does not
absorb* state their references at the default side; the cases whose statements are written for the infinite plate
(the angle law, the bonded run, the mass cosine, the resonance, the bending stiffness, a closed plate, a peaked face)
carry `side` 0.

**Two impedances in series with the mass** act in `X₀` (`Material::mass_X0`), both at the cosine `c` the
wave has in air (N3, N4):

- `res*` on a mass-law material is the plate's resonance, not a permittivity pole (`eta()` adds the pole
  only where `m = 0`, so a rigid body with `res*` stays closed). `resS > 0`: the mass-air-mass resonance of a
  double unit or a leaf on its cavity, the dip it makes and no more,
  `X₀ × min(1, |1 − (f·c/resF)² + j·f·c/(resF·resQ)|)` (the cavity is stiffer by `1/c²`, so the dip moves up
  as `1/c`). `resS < 0`: the panel's own first mode, a stiffness below it,
  `X₀ × |1 − (resF/f)² − j·resF/(f·resQ)|`. Sheet and traced solid on the rule within 0.0071 dB.
- `coiF`, `coiQ` on a mass-law material are the bending stiffness of the plate (`coiF` its coincidence
  frequency, `1/coiQ` its loss factor): `X₀ × |1 − (f/coiF)²·sin⁴θ·(1 + j/coiQ)|`, so the diffuse-field dip
  comes from the angle. Coincidence is no interface loss: the angle-free Lorentzian `coiA` of the interface
  gain is removed (D2-b, 2026-10-17); a `coiA` column of an older table is read and ignored (*Coincidence column - a
  table that carries coiA loads and the column changes nothing* and its *- the mass-law part*).

Bonded bodies add these `X₀` like any other (2.2). Asserted in: *Resonance - on a mass-law material res\* is
the plate's resonance and leaves the permittivity alone*; *Coincidence - on a mass-law material coiF and coiQ
are the bending stiffness of the plate*.

**The cosine is the one the wave has in air**, `sqrt(1 − p)` for the transverse invariant `p`: for a wall in
air the incidence cosine, whatever refracting layer lies in front. `SHEET_RESOLVE` forms it where the near
medium is not air, carries no mass column, the leg is known to be bent, and the sheet (or a layer of the
stack) carries `m > 0`, as the far cosine of near | air at the recovered cosine; elsewhere the traced cosine
already is it. Above an invariant of one the term keeps the cosine the wave has in the near medium. A plate
with a cover on both faces pays as the bare plate (the thin-cover view). Asserted in: *Mass cosine - a
mass-law sheet behind a refracting layer forms its term at the cosine the wave has in air*; *Refracting
tracer - a traced bend is known from the state and the leg is exact - the mass-law part*.

### 2.2 Bonded bodies are one plate

Mass-law bodies within the co-location distance move with one velocity: their `X₀` add
(`TL = 20·log10(X₀₁ + X₀₂)` at normal incidence, the single-body shape on the sum). A 35 dB panel on a 53 dB
wall is 54.03 dB, not 88. Each `X₀` is unclipped, so the zero sits on the sum; `m = 0` is absent (a resilient
underlay between two mass-law bodies separates them, their terms then add in dB). The charging telescopes:
the first body pays its term, a later one `TL(X_run + X₀) − TL(X_run)` (35 then 19.03 dB; the other way 53
then 1.03 dB). A leg that begins at a reflection starts with `X_run = 0`. A matched sheet (its host's
permittivity) is merged into its host's leg unless both carry the mass column; a mass-law sheet in a
mass-law host cuts the leg into two bonded bodies. Asserted in: *Unentered branches - a sheet without the
mass column of its mass-law host's permittivity is merged*.

How the second body learns `X_run`:

| pair | learned from |
| --- | --- |
| sheet \| sheet | the solver's run over adjacent mass-law layers, one event |
| sheet \| solid, entry | `mtl_prev` holds the sheet (thickness is a column), the buffer flag says it was crossed |
| sheet \| solid, exit | the solid closes its leg in `SHEET_EVENT` and hands the sheet its `X₀` |
| solid \| solid | `mtl_prev` names the material, the buffer word carries the thickness code |
| a sheet inside a wall | `mtl_prev` holds the sheet with its inside flag, the buffer word the first half's code |
| deeper | the same two words: the code is an equivalent thickness of every body behind |

**The thickness code** (buffer word with bit 15 set): `d = 10^(−4 + 6·(k − 1)/32766)` m, `k = 1 … 32767`,
0.1 mm to 100 m in steps of 0.044 % (0.0038 dB), quantized again at every bonded face. `MASS_FOLD` forms the
equivalent thickness `d_eq = Σ dᵢ·fRef_R/fRefᵢ` over every bonded mass-law body, read with material `R`: `X₀` is
linear in thickness at every frequency, so no frequency enters and one code serves every frequency of a run (M10:
no exponent, no exclusion of a body of another slope; before, `d_eq = (Σ (dᵢ·fRef_R/fRefᵢ)^p)^(1/p)` with
`p = m/20` and a body of another slope left out). `MASS_CODE` writes, `MASS_RUN` reads. Two halves of one material
pay the term of the summed thickness, +6 dB per doubling. The fixtures keep bodies with `m` 30 (and 10, 12) and
write their references at slope 20, so each table guards that `m` is a switch. Asserted in: *Bonded mass - sheet on
sheet, sheet on solid and solid on solid as one plate*; *Bonded mass - a mass-law sheet inside a mass-law wall*;
*Bonded mass - the plate at any depth*; *ray_progress - the bonded plate at any depth*; *Stack with a mass-law layer
- resolved by segments, not killed* (an `m` 10 layer pays what `m` 20 pays, bitwise).

**One side per bonded run** (rule of use, like one `att` per run; decided 2026-10-19). A bonded run is one panel and
should carry one `side`. Each body pays its increment at its own side, `TL(X_run + X₀; s) − TL(X_run; s)`, so where the
sides differ the run passes `TL(X₁; s₁) + TL(X₁ + X₂; s₂) − TL(X₁; s₂)`; in the mass-controlled range each later
increment hardly depends on the side, and the run passes what the whole run passes at the side of the **first** body
the ray meets (within 0.0003 dB, two 152 mm concrete bodies of sides 1.35 \| 3.29 m, 31.5 to 500 Hz, 0° to 85°). The
result then depends on the direction. Asserted in: *Finite plate - side 0 is the infinite plate, a table without the
column is side 3.29, a wave medium ignores it, a negative side is refused, a bonded run takes the side of the body that
forms its term*.

### 2.3 The plate rule

A run of bonded mass-law bodies (2.2) is one plate with no wave inside. It reflects at the face met what
that face does not absorb, `R = 1 − a_met`, and passes the incoherent series of its faces and its bodies
(N1, R1h):

    T = a·τ / (a + τ − a·τ)

with `a` the smaller absorption of its two outer faces and `τ` the bodies' term: `T ≈ τ` where `τ ≪ a`,
`T → a` for a light panel or at a resonance, never more than either, smooth in `τ`. What it neither passes
nor reflects (`a − T`) is absorbed, which is the split transmitted / dissipated / reflected of a panel
[Davy_2009_SingleLeafCremer]. A heavy wall passes its mass law, no longer the mass law times the absorption
of both faces (at bb336dc a 200 mm wall with `tf` 0.3 passed 10.45 dB less).

**The absorption of an outer face** is what its pad passes (`Material::closed_face`): `tf` in front of a
lossless medium where the face is closed (every rigid face from 0.003°, one-parameter mapping; from the
critical angle, two-parameter), the transformer's `1 − R′` where it is open (a rigid face exactly at normal
incidence: `1 − (1 − tf)·R0`, `R0` its Fresnel reflectance), and next to a lossy medium the flux the pad
hands over. A face without `tf` absorbs nothing and the plate passes nothing; rule of use: `tf` is set
together with `m`. Where `tf` carries a peak the face is its series impedance (2.6) and absorbs
`1 − (1 − tf)·|r_ζ(θ)|²`, so `a` depends on the angle: over-damped it rises toward grazing, and the plate passes
more there. The 12.5 mm board of 2.6 passes within 0.06 dB of the flat absorption at 0°, 1.07 dB more at 60° and
6.35 dB more at 85° (63 Hz); field-incidence TL per octave falls by at most 0.96 dB on the board and by 0.21 / 0.17 dB
on 4 mm glass and 4-16-4 with peaked faces (the library on the formula copy of report 3.6 to 0.0000 dB,
`md/m6_report.md` 0.5). None of the 13 transmission fits of report 3.6 carries a peak. A face between two bodies of
the plate passes everything and reflects nothing.

**Where it is paid.** The first face passes `a_in`; each body's leg pays its telescoped term (2.2); the face
out of the plate into a medium without the mass column pays the rest, `F(min(a_in, a_out), τ) / (a_in·τ)`.
For a traced plate `a_in` is re-formed at the exit from the previous-medium word (`PLATE_IN`, one level deep,
section 7); for a sheet or a stack the solver carries it along the run (`PLATE_FIX`). A mass-law sheet that
is a whole plate under a closed outer pair (glued to a medium the wave cannot enter, the leak chain) passes
the rule of its two faces in one event.

Numbers: 4 mm glass and 152 mm concrete with their measured absorption, sheet and traced solid, both tracers,
both mappings: on the rule within 0.0022 dB, the reflection within 0.0008 dB. Wood 20 mm | concrete 100 mm
(`tf` 0.2 | 0.1): `a` = 0.1, the same from both sides within 0.0007 dB as two solids, sheet on solid and
stack. A wood panel glued on concrete without its mass law is reciprocal within 0.00094 dB (the flat first
face, section 7). Asserted in: *Plate rule - a rigid mass-law panel passes the incoherent series of its face
and its plate and reflects what its face does not absorb*; *Plate rule - a wall bonded of two mass-law bodies
passes the same in both directions*; *Closed plate - a mass-law layer the wave is evanescent in passes the
plate rule of its two faces*; *Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two
traced solids*; *Total reflection of the outer pair - the sheet stays in the result - the mass-law part*;
*Leak chain - each face at the cosine the wave has and the series on a closed far face - the mass-law part*;
the four *Bonded mass* cases of 2.2; *ray_progress - the plate rule and the absorption peak at every
frequency of a band*.

**A closed mass-law layer** (lossless, `Im(ε·μ) = 0`, and the transverse invariant at or above its `ε·μ`:
asked of the layer, so both faces close together) carries the leak as a flux. Seen from a side with a wave (a
coherent layer, a medium without the mass column) its face is the pad of a closed face: on the way in the
flux it passes, on the way out the amplitude that flux launches. A stack of a coherent layer and a mass-law
layer is one wall in both orders where the mass-law layer is closed; where it is not closed the two orders
differ by 0.107 / 0.127 / 0.201 dB at 10° / 30° / 60° (section 7). Asserted in: *Closed face - tf at a closed
face is a pad and the two ports of a sheet on it add up*; *Unentered branches - a stack of a coherent and a
mass-law layer against the laminate meshed as a solid with a lining*; the mass-law parts of *Sheet tf*,
*Bare tf*, *Rare sheet* and *Flat branch*.

**In the solver** a mass-law layer is not made coherent: it ends the coherent segment as a half-space of its
own admittance, contributes one real factor (`qd_mass_amp`: `alpha` over the held path `thk / c` with `c`
the cosine in air, under a 400 dB cap, and the mass term), starts the next segment, and displaces no host.
Nothing returns across it: its reflect port is its near face. A mass-law sheet named as a mirror in
`qd_slab_resolve` is a half-space. Asserted in: *Lumped columns - a bounding sheet: tf is a layer, a mass-law
sheet a half-space*; *Stack with a mass-law layer - resolved by segments, not killed*; *Lumped columns - an
ordered stack: what the core carries, by matrix and by scattering - the mass-law part*; *Unentered branches -
ray_mesh_interact's own glued composite against the redo - the mass-law part*; *Held path - alpha on a
lumped plate is charged over the path the held wave takes* (22 mm at 100 dB/m: 2.19 / 2.22 / 2.52 / 4.31 /
11.74 dB at 0° / 10° / 30° / 60° / 80°, sheet and traced solid within 0.001 dB).

### 2.4 The held wave

A mass-law body holds the wave undeviated, whatever its permittivity and the tracer (a body with `ε` above
one included). `interact_with` takes an `undeviated` flag: inside a plate a face passes what reaches it;
towards a medium without the mass column the held wave carries the transverse invariant it had in air, so it
is refracted as from air and closed where air would be past that medium's critical angle, and
`APPLY_TRANSITION` writes the bent flag for the leg that begins there. So sound through a wall continues in
the direction it arrived in, a layered wall has one angle throughout, and the leg behind a wall into a wave
medium is the refraction from air. `ray_mesh_interact` sets the flag always; `ray_state_update` where no
refracted direction exists or the cosines coincide, unless the leg carries the bent flag.

**A layer behind a held wave.** A mass-law sheet on the far face of a mass-law body, or inside it, is part of
the plate. A layer without the mass column glued on the far face (a lining) is the same wall as its traced
layer, for every tracer, within 0.015 dB (dense `eps 2.50 − 0.80j` and porous `eps 0.31 − 0.20j`, 10° to
75°); the two sides of such a lining differ by the lossy medium's convention (A2, section 7). A non-rigid
mass-law body (`eps 1.50`, `m` 20) passes at every angle, tracer 5 on tracer 4.

**A closed face hands the wall behind a traced layer the direction in air** (the leak's direction), so the
wall's mass term is formed at the incidence cosine.

Asserted in: *Held exit - a lining behind a rigid wall is the same as its traced layer, for every tracer*;
*Held exit - a non-rigid mass-law body holds every tracer and passes at every angle*; *Held lining - a
mass-law sheet behind a wave held in a mass-law wall is the same wall as its traced solid*; *Ports of a
mass-law face from inside - the reflect port is the other direction's*; *Leak direction - a closed face hands
the wall behind a traced layer the direction the wave has in air*; *Leak direction - a wave held in a
mass-law body keeps its direction and a closed face hands on the wave's*; *Phase reference - a traversal
carries the phase at the point the polyline leaves at - the mass-law part* (the held reference, dense and
rigid, 0.0000°); *Direction - a ray leaves a parallel slab parallel to how it arrived - the mass-law part*;
*Stack geometry - an ordered stack leaves the ray on the line it arrived on - the mass-law part*;
*ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band*;
*ray_progress - refracting tracer: the segment out of a lossy medium is the direction the wave has - the
mass-law part*.

### 2.5 Nothing is reflected back into a mass-law body

`qd_slab_resolve` returns false for a mass-law slab, and since round 1 (N2) a face reached from inside a held
mass-law body reflects nothing: a traced plate passes in one path. A 4 mm pane and a 20 mm wood panel with
every bounce followed pass one path, equal to the summed series; the reflect port of a held wave at a lining
is gone (−999 in *Held lining* D). Asserted in: *Plate rule - a traced mass-law body passes in one path:
nothing is reflected back into it*; *ray_state_update - mass-law material is excluded from slab resolution*
(its exit into air an ordinary one, type 1, at 0° as at 0.1°); *Measurement - the mirror behind a second solid
and behind a sheet inside a wall - the mass-law part*.

### 2.6 The room-side absorption

`tf` is the absorption of a rigid face (2.3) and, with three columns (A10), it can peak:

    tf(f) = tfR·(f/fRef)^tfB + tfP / (1 + (tfQ·(f/tfF − tfF/f))²)

clamped to `[−1, 1]` as before; `tfF ≥ 0` is required (`tfQ` may be negative: the branch, below), and the peak acts
where `tfP ≠ 0` and `tfF > 0`. The peak is the lumped resonator's (M5, built 2026-10-16; 3.2): symmetric in log
frequency, zero at both ends, `|tfQ|` the absorber's `Q_α`. Near the peak it equals the linear detuning of 4720a92,
`tfP / (1 + (tfQ′·(f − tfF)/tfF)²)`, with `tfQ′ = 2·Q_α`; that form kept a floor of `tfP/(1 + tfQ′²)` as `f → 0`.
A 12.5 mm board on battens (floor 0.05, peak 0.25 at 150 Hz, `Q_α` 0.75) absorbed the formula at every angle under M5
(238444d), as sheet and traced solid, and passed the plate rule within 0.001 dB; through `ray_progress` every
frequency of a band equals its single-frequency run. Its peak part at band centers: 0.020 / 0.079 / 0.232 / 0.152 /
0.040 at 31.5 / 63 / 125 / 250 / 500 Hz (on the linear detuning with `tfQ` 1.5: 0.104 / 0.142 / 0.235 / 0.125 /
0.019; `md/m5_a21_report.md` 1.2).

**A peaked face is a locally reacting face** (M6, built 2026-10-16, `md/m6_report.md`). Where a mass-law face carries
a peak (`m > 0`, `tfP > 0`, `tfF > 0`) the peak is the series impedance of a resonator per `Z₀`,

    ζ = R′ + j·(1 + R′)·|tfQ|·(f/tfF − tfF/f),   tfP = 4R′/(1 + R′)²,

over-damped (`R′ ≥ 1`) by default, under-damped (`1/R′`) where `tfQ < 0`; `tfP` above one is `R′ = 1`, where the two
branches meet. `ζ` replaces the reactance of the bare face (the mapping's calibration of a hard wall), behind a
matched pad of the power-law part, and is met against the immittance of the side the wave is on: in air
`r = (ζ c − 1)/(ζ c + 1)` at the cosine `c`, the TM channel of EM types `(ζ/c − 1)/(ζ/c + 1)` (an impedance boundary),
and `|r|² = (1 − tf)·|r_ζ|²`. At normal incidence in air it absorbs the law above less `tf` times the peak
(`tf + α_peak − tf·α_peak`: 0.2875 against 0.3000 at 150 Hz on the board). The face is the same on both sides of a
critical angle (no step: 1.9e-7 between neighbors 0.001° apart at the two-parameter critical angle) and in both
mappings. A wave medium (`m = 0`) with a peak keeps the transformer of the flat law, and a negative peak forces
reflection as before; without that gate a peaked wave medium created energy (R + T 1.36, `md/m6_report.md` 0.3).

Measured on the board (`md/m6_report.md` 0.1, 0.4): over-damped (`R′` 13.93) the phase runs from −4.12° at 83.6 Hz to
+4.12° at 269.2 Hz at 0° (±5.83° at 45°, ±8.26° at 60°, 1/96-octave sweep), `r` +0.844 at the peak; under-damped
(`R′` 0.0718) the phase passes through 180° at 150 Hz, `r` −0.844. The over-damped face absorbs more toward grazing
(0.288 / 0.319 / 0.467 / 0.991 at 0° / 30° / 60° / 85°, 150 Hz), the under-damped one less (0.288 / 0.259 / 0.177 /
0.074): the random-incidence value (Paris 0°–90°) is 0.420 and 0.214 against the same `α₀` 0.288 (9). Behind a lining
the face is met against the lining's immittance, the transfer matrix of a locally reacting face behind a layer
(within 6.2e-13 glued, 3.4e-5 traced, dense `eps 2.50 − 0.80j`): a face of fixed impedance whatever lies in front (A20
for the peak). A panel over a partitioned cavity (`cot(k₀d)`, no cosine inside) is the lumped `ζ` within 0.012 in
`|r|` and 0.0021 in the Paris value per octave to 1 kHz on 25 to 100 mm of cavity; an unpartitioned cavity is M9's
(2.8). Asserted in: *Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg,
as a sheet and as a traced solid, on both branches* (both mappings, types 4, 5, 3 and EM type 1, the gate, the
lining, the critical angle); *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a
mass-law face* and *ray_progress - the plate rule and the absorption peak at every frequency of a band* (restated:
the absorption at the angle and the plate rule with it); *ray_progress - the random-incidence absorption of a peaked
lining is the Paris integral of its series impedance* (within 4.3e-6 of the closed form). A material CSV carries the three columns like any other; the OBJ writer's
canonical order and the Python / MATLAB readers' key order list them after `tfB`. Asserted in: *Absorption peak -
tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face*; *ray_progress - the plate rule
and the absorption peak at every frequency of a band*.

A rigid face without a peak reflects `R = 1 − tf` at every angle, in both mappings (|r| 0.9486 for `tf` 0.1, report
3.4); that `tf` does not change the phase of the face (report 3.5). The coefficient type this makes `tf` is in 3.1.

Rule of use: a rigid body carries the mass column, and `tf` is set together with `m`. A rigid permittivity
without `m` is a wave medium (`md/wave.md`, section 8). A rigid body is lossless: a loss column opens its
faces, and a 200 mm wall with `tf` 0.3 and a loss tangent of 0.04 then reflects 1.000000 from 10° on and
passes −88 to −102 dB (*Lossy verdict - a lossy second medium has no critical angle - the mass-law part*, J,
recorded). One `att` per bonded run (A5).

### 2.7 The columns on a mass-law material

| quantity (per band) | column | settable on its own | round-1 evidence |
| --- | --- | --- | --- |
| room-side absorption of a rigid face | `tf`, `tfB` (power law), `tfP`, `tfF`, `tfQ` (a resonant peak) | yes | monotone rows within 0.051 in α (report 3.4); the peak on the resonator law: every lining and pane row within 0.017, cork board 0.047 (report 3.8 refitted, `md/m5_a21_report.md` 1.2); as a locally reacting face (M6) the same residuals on the Paris target (`md/m6_report.md` 0.6) |
| transmission level | `fRef` with `thk` | yes: the plate passes about `τ` where `τ ≪ a` | the 13 constructions within 2.70 dB at 0–90° with the finite panel at their laboratory side (`md/grazing_report.md` 1.5; the table below); light panels at their mass (glass 4 / 6 mm 1.13 / 0.90 of it, gypsum 0.64, plywood 0.88), the units at 0.41 / 0.36, heavy walls and linings at 0.02 to 0.39 (under the infinite plate glass read 1.42 / 1.42: its level bought back the grazing transmission) |
| the mass law | `m > 0`, a switch (20 dB per decade, M10) | yes | the slope is the mass law's; until 2026-10-17 `m` was fitted, 1.64 to 35.13 (report 3.6) |
| panel size (finite-plate radiation) | `side` = √(panel area), m; 3.29 where absent, 0 the infinite plate | yes; one per bonded run | the 13 at their laboratory side (1.349 m panes and units, 1.697 m boards, 3.523 m walls and linings) within 2.70 dB; all at 3.29 m within 2.70 dB as well (`md/grazing_report.md` 1.5) |
| coincidence dip | `coiF`, `coiQ` (bending stiffness in `X₀`; `coiA` removed) | yes | light panels at their coincidence (glass 2967 / 1975 Hz against 2989 / 1993, gypsum 2763 against 2500, the units 3073 / 2967 against 2989; plywood 1563 against 1146); on heavy walls and linings off the physics above coincidence: concrete 6925 Hz (`coiQ` 1.19), brick 203 18.9 kHz (1.0), plaster 3734 Hz (1.61), block 8151 Hz (3.0), the furred lining at the 30 Hz bound (1.0). The finite panel does not change this: the heavy walls are above coincidence from 125 Hz, where grazing carries nothing (`md/grazing_report.md` 0.6) |
| mass-air-mass dip of a double or triple unit | `res*`, `resS > 0` | yes | 4-16-4 0.71 dB, triple 0.77 dB (report 3.6, 0–78°); at 0–90° with the finite panel the fits take the first-mode form at 211 / 192 Hz: 1.56 / 2.67 dB (`md/grazing_report.md` 1.5) |
| a panel's first mode, the stiffness region below it | `res*`, `resS < 0` | yes | the fits choose the sign unstably (report 3.8; with the finite panel six of 13 take `resS < 0`) |
| extra loss per entry (frame, seals) | `att`, `attB` | yes; one per bonded run (A5) | with the finite panel at 0–90°: 0 to 1.9 dB on light panels, 5.4 / 7.2 dB on the units, 2.3 to the 12 dB bound on heavy walls and linings (the furred lining and the board on dabs at it; concrete 7.0, block 8.9, plaster 9.9): a level below the mass with a flat `att` stands for what the infinite or finite plate does not model above coincidence (`md/grazing_report.md` 0.6, 1.5) |
| porous and empirical absorbers | `a`–`h`, `alpha` (wave media) in the two-parameter mapping | yes (`md/wave.md`; section 1) | Knauf TP 120 A within ±0.025 (20 mm), ±0.032 (30 mm), 31.5 Hz–8 kHz (report 3.4) |
| bonded layers | the plate rule | yes | 2.3 |
| reflection phase of a hard wall, of a resonant face | the two-parameter mapping (N7); `tfP`, `tfF`, `tfQ` as a series impedance (M6, 2.6) | yes | within 0.1° to 60° (report 3.5); a peaked face on its series impedance to 1e-12 (*Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches*); a flat pad keeps the bare face's phase |


**The 13 fits as library columns** (decided 2026-10-19: the finite panel at each construction's laboratory side; the
procedure of section 9, 8 seeds and the M10 0–90° fit as a start, selection at 768 × 12; `md/grazing_report.md` 1.5).
Worst residual 2.70 dB (the furred lining; gypsum, a model reference, 2.54); before, under the infinite plate, 2.97 dB.
`thk` the construction's, `tfB` the class's absorption law of report 3.6, `att` in dB at 1 kHz, `coiF` and `res` in Hz
(the library takes GHz: × 8.7403e-4); `fRef` and `tf` are library values. A 0 in the residuals is a band inside its
target or without one. Through the built library: sheet and formula copy within 0.0000 dB, traced solid within 0.0042
dB, R + T below one in every row, a band through `ray_progress` on its single frequencies bitwise (`md/grazing_report.md`
2.4).

| construction | worst | residual per octave 31.5 Hz … 8 kHz (dB) | `side` (m) | `m″` level (of mass) | `att` / `attB` | `coiF` Hz / `coiQ` | `res` Hz / Q / sign | `fRef` (GHz) | `tf` at `fRef` |
| −-- | −-- | −-- | −-- | −-- | −-- | −-- | −-- | −-- | −-- |
| glass 4 mm | 0.85 | 0 +0.09 −0.49 +0.51 +0.30 −0.51 +0.25 +0.52 −0.85 | 1.349 | 11.3 (1.13) | 0.05 / +1.35 | 2967 / 284 | 138 / 2.33 / + | 4.0562e-05 | 5.661 |
| glass 6 mm | 1.78 | 0 0 +1.78 +1.75 −0.64 −1.78 +1.78 +1.77 −1.25 | 1.349 | 13.5 (0.90) | 0.00 / −0.43 | 1975 / 300 | 26 / 49.7 / + | 5.1068e-05 | 3.682 |
| gypsum 12.5 (model) | 2.54 | +2.54 +0.59 +0.54 0 −1.30 −2.54 +2.54 +2.54 +0.53 | 1.697 | 5.5 (0.64) | 0.81 / −0.78 | 2763 / 300 | 30 / 38.3 / − | 0.0002614 | 7.349 |
| plywood 18 (model) | 0.31 | 0 +0.14 +0.22 +0.08 −0.31 +0.30 −0.31 +0.22 +0.06 | 1.697 | 9.5 (0.88) | 1.93 / +0.49 | 1563 / 300 | 734 / 50 / + | 0.00021843 | 0.2127 |
| concrete 152 | 0.57 | +0.47 0 −0.56 +0.56 −0.56 +0.57 −0.56 +0.09 0 | 3.523 | 88.8 (0.26) | 7.01 / −0.09 | 6925 / 1.19 | 78 / 1.44 / − | 0.00019663 | 0.0002549 |
| brick 95 | 0.01 | 0 0 0 0 0 0 0 +0.01 0 | 3.523 | 66.0 (0.35) | 2.43 / +1.04 | 5621 / 11.9 | 205 / 2.77 / + | 0.00016602 | 0.004283 |
| brick 203 | 0.58 | −0.52 0 +0.28 +0.58 −0.58 +0.58 −0.57 −0.10 −0.31 | 3.523 | 158.6 (0.39) | 2.27 / −0.34 | 18910 / 1 | 119 / 6.78 / + | 0.00014723 | 0.004146 |
| block 190 | 0.49 | −0.49 0 +0.49 −0.49 +0.49 −0.49 +0.21 +0.04 0 | 3.523 | 51.2 (0.23) | 8.93 / +0.25 | 8151 / 3.03 | 580 / 2.87 / + | 0.00042658 | 0.004688 |
| 4-16-4 (AGC) | 1.56 | 0 0 +1.55 −1.08 +1.55 −1.56 −1.42 +1.56 0 | 1.349 | 8.2 (0.41) | 5.35 / +0.54 | 3073 / 300 | 211 / 49.6 / − | 0.00011189 | 1.341 |
| 4-12-4-12-4 (AGC) | 2.67 | 0 0 +2.66 −1.41 +2.67 −2.67 −2.33 +2.67 0 | 1.349 | 10.8 (0.36) | 7.17 / +0.53 | 2967 / 299 | 192 / 49.4 / − | 0.00012741 | 1.279 |
| brick 203 + furred board | 2.70 | +2.60 +2.61 +2.16 +1.60 −1.79 −2.70 −2.32 +2.66 0 | 3.523 | 7.7 (0.02) | 12.00 / −0.36 | 30 / 1 | 165 / 48.9 / + | 0.0030373 | 0.6398 |
| SCR 140 + board on dabs | 1.95 | 0 0 +1.60 −1.73 +1.51 −1.95 +0.80 +0.64 0 | 3.523 | 86.5 (0.31) | 12.00 / +0.95 | 2673 / 290 | 203 / 3.29 / − | 0.00018593 | 1.183 |
| brick 105 + plaster one face | 0.00 | 0 0 0 0 0 0 0 0 0 | 3.523 | 35.2 (0.17) | 9.90 / +0.09 | 3734 / 1.61 | 113 / 1.03 / − | 0.00034267 | 0.006445 |

### 2.8 Leaves as wave-medium sheets

A light leaf over an unpartitioned cavity (a panel or membrane absorber, a double leaf, a curtain, an MPP as an
equivalent fluid) is built from wave media, not from the mass column (M9). Under the two-parameter mapping a sheet
with `m = 0` is a limp leaf (section 1); a box of air behind it forms the leaf's resonance, its phase and its angle
shift, with no lumped column. Measured in `build_linux/scratch/m9` (`md/m9_fixtures_report.md`, phase 0).

**Rule of use.**
- The leaf: a sheet (`thk`) of a wave medium, `m = 0`, `ε = n·z`, `μ = n/z` of its material (`μ = ρ/ρ₀`); its loss (a
  mounting resistance) a constant loss ratio `rr` on `μ`: `g = e·rr/17.98`, `h = 1`, `fRef = 1`.
- The cavity: a meshed box of near-air material (`ε` 1, `μ` 1.000001; a material identical to air is air), the leaf
  glued on its front face. A glued leaf lies in front of the box, so the cavity is the box's depth.
- The wall: glued on the box's back face, a mass-law body as everywhere. With `tf` 0 it reflects the Fresnel
  coefficient of its two-parameter half-space (the leaf reads the same as on the `m = 0` half-space, bitwise), with
  `tf` a pad on it. A rigid backing (`r = 1`) is not the reference: the half-space's small phase moves R by up to
  0.023 at a sharp resonance.
- The box's series is summed in one event (any `thin_slab_threshold` up to 0.15 sums it): 5 traced events per ray
  on a half-space, 7 on a finite wall. Through `ray_progress` a band equals its single frequencies (the box and the
  sheets do not bend).
- A double leaf: two leaves glued on the two faces of one box.
- A double unit is one lumped sheet with its fitted columns (2.7), or two wave-medium leaves on a box; never two
  meshed mass-law sheets. Measured 2026-10-19 (`md/grazing_report.md` 1.4), 4|16|4 as two 4 mm mass-law sheets 16 mm
  apart: with a near-air box between, the unbridged double leaf above its resonance (70.1 / 104.3 dB at 1 / 4 kHz, 0°,
  against the transfer matrix's 70.2 / 104.7; the measured unit, whose spacer bridges the panes, reads 36.4 / 36.5 in
  those octaves) and no mass-air-mass dip (29.2 / 39.8 dB at 125 / 250 Hz against 21.9 / 23.7): a mass-law sheet carries
  its term as a power factor and the box's series cannot resonate on it. With a plain gap, two uncoupled plates
  (37.6 dB at 1 kHz). Before 2026-10-15 the meshed unit read 105 dB at 1 kHz.
- A porous fill: a traced porous body glued at the back of the box (air in front, the fill against the wall), its
  columns fitted to the chosen model (A15). Each body is summed; the exchange between the two is followed as paths,
  55 to 170 events per ray for Shearer's build (20 mm air, 75 mm rock wool), up to 3,314 for 10 mm of air on 15 mm
  at 80 Hz. Behind a fill the wall takes `tf` 0 (A20).

**What it reaches.**
- The single leaf: the limp mass law within 0.0078 dB (50 Hz–1 kHz, 0° to 85°), its phase the layer's.
- The board (12.5 mm, 8.6 kg/m², `rr` 0.2) on 25 mm: the transfer matrix of fluid layers within 6.5e-5 in R summed,
  4.7e-6 followed; its peak at 128.4 / 148.6 / 182.4 / 258.8 Hz at 0° / 30° / 45° / 60°, the angle shift
  1.158 / 1.421 / 2.016 (`1/cos θ` within 0.8 %).
- 4|16|4: within 7.0e-3 dB in T; the dip at 211.3 / 244.0 / 298.8 / 422.6 Hz at 0° / 30° / 45° / 60° (`1/cos θ`
  within 0.02 %; the lumped mass-air-mass frequency 211.4 Hz).
- Measured absorbers at random incidence (Paris integral): Shearer's MDF membrane peaks at 111 Hz empty and 83 Hz
  filled against the measured 100 and 80 Hz thirds; Sakagami's 3 mm plywood over 30 / 90 / 150 mm at 1.07 / 0.80 /
  0.99 times the measured chamber peaks (A19). The heights depend on the assumed `rr`.
- The fill (Shearer's build): on the transfer matrix of its columns within 2.6e-5 in R.

**Limits.**
- Wave media keep the infinite plate at grazing: `side` is read only where `m > 0`, so a leaf (or a leaf of a meshed
  double wall) is the infinite fluid layer at every angle, transparent toward grazing like the infinite plate; its
  finite size is not modeled (7).
- No bending stiffness: a fluid layer has none. Coincidence stays with the mass column, and the stiffness of a
  small clamped panel's edge is not in it (section 5, A19).
- A meshed gap in place of the box is exact where its path series converges, but needs 1,126 to more than 8,000
  events per ray (more toward higher frequencies), and `ray_progress` stops at 255 interactions: 3.7e-3 in R on the
  board, 9.1 dB in T on 4|16|4. Out of use.
- A fill only at the back of the box. A fill directly behind the leaf with air behind it has no finite path sum (the
  leaf seen from inside the fill reflects `|r|` 1.57 at 50 Hz); an ordered stack of the leaf and a porous sheet glued on
  the box runs away; a porous sheet between the box and the wall is outside the glued-exit convention (`md/wave.md`
  6, 8): the intersector reports two of its three co-located faces, by mesh order, and the result depends on which
  (A21). Since 2026-10-16 nothing is created there (the wave left in air inside the wall is killed), but the
  reflection can be that of box | sheet | air. A porous layer at the back of the box is a traced body.
- The followed series of a lossy body between strong reflectors diverges (the board on 25 mm of fill: round trip
  1.145 at 50 Hz, by formula); only the summed mode, which `ray_progress` takes, means anything there (`md/wave.md`
  7).
- A leaf past its critical angle tunnels; the thin-layer result holds while the leaf is thin against the wavelength
  in it.
- A curtain's `R_s ∥ jωm` is a relaxation in `μ` that the power-law columns carry only on the random-incidence
  average (M16).
- The summed box carries a residual at the resonance that the followed series does not (6.5e-5 in R on the board,
  7.0e-3 dB in T on 4|16|4, against 2e-6 followed); not a depth or density offset, cause not traced.

Asserted in: *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf*; *Leaf - a leaf on a box of air
is the transfer matrix of leaf and cavity at every angle*; *Leaf - two leaves on a box of air resonate at the
mass-air-mass frequency*; *ray_progress - a leaf on a box of air at every frequency of a band*; *Leaf - a porous fill
at the back of the box is the transfer matrix of leaf, air, fill and wall*. They evaluate wave media for the leaf and
the box; fixtures 1 and 3 evaluate wave media only and guard this route (`md/wave.md` 0).

## 3. Requirements and expressibility

One row per quantity: what the physics requires, with its model form, inputs, validity and source; what the
machine does today; the candidate of section 4 that would move it. Status words: **expressible** (some
setting of today's columns reaches it, with the evidence named), **partial** (reachable in part, the missing
part named), **not expressible**. "Not measured" marks a statement without a run. Scope decisions are in
section 5.

### 3.1 Absorption and reflection of a surface

The four coefficients of the literature are not interchangeable [Balint_2023_HundredYearDebate]:
**normal-incidence** `α₀` (ISO 10534); **random-incidence** `α_st`, the Paris integral
`∫ α(θ)·sin 2θ dθ` over 0 to 90° [Nocke_2023_HowTo, Eq. 5]; **field-incidence**, the same integral cut near
78°, an empirical choice [Jeong_2012_BoundaryConditionsPhasedGA, Eq. 5]; and the **Sabine** coefficient `α_s`
of ISO 354, which includes edge diffraction and the non-diffuse incidence of real chambers and can exceed one
[Balint_2023_HundredYearDebate, p. 20].

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| absorption per octave of a rigid face | for energy tracing a random-incidence coefficient per band; normal-incidence coefficients give the largest errors [Jeong_2012_BoundaryConditionsPhasedGA, Sec. V-VI] | `tf` power law, `R = 1 − tf`; a peak as a locally reacting face (2.6) | expressible; report 3.4 (monotone rows within 0.051); peaked rows within 0.017, cork 0.047, on the resonator law (report 3.8 refitted, `md/m5_a21_report.md` 1.2), and on their Paris targets as locally reacting faces with the same residuals (`md/m6_report.md` 0.6) | M5, M6 (built) |
| the coefficient type `tf` is | `α_st ≤ 0.951` for a locally reacting surface, reached at a normalized impedance of 1.567 [Jeong_2009_AngularDistributionCorrection, Sec. I]; `α_s` exceeds `α_st` by 0.1 to 0.35 at mid frequencies for porous layers [Dupont_2023_SoundFieldSynthesis, Sec. 4.3]; ISO 11654 caps practical coefficients at 1.00 [Balint_2023_HundredYearDebate, p. 19] | `R = 1 − tf` at every angle, so `tf` without a peak is `α(θ)` at every angle and equals `α₀`, `α_st` and the field value at once; `tf` up to one. A peak is a locally reacting face (2.6): the law gives its `α₀`, and its random-incidence value is the Paris integral of the face (0.420 against `α₀` 0.288 on the board of 2.6, over-damped) | expressible for a random-incidence reference; a Sabine reference has to be converted before it is a `tf` (A13); a peaked row is fitted on the Paris integral (9) | — |
| angle dependence of a locally reacting face | `α(θ) = 1 − \|R(θ)\|²`, `R(θ) = (Z_s cos θ − ρ₀c₀)/(Z_s cos θ + ρ₀c₀)` [Savioja_2015_GeometricalModeling, Eqs. 4-5]; for a high-impedance face `α_st` exceeds `α₀` by up to a factor of about 1.7 (impedances 40 / 10 / 4: `α₀` 0.10 / 0.33 / 0.64, `α_st` 0.17 / 0.49 / 0.79) [Jeong_2012_BoundaryConditionsPhasedGA, Table I]; angle-dependent coefficients improve accuracy only marginally over angle-independent ones (Jeong et al. in [Savioja_2015_GeometricalModeling, p. 719]); measured `α(θ)` exists for two surfaces only [Aspoeck_2020_BRAS, Sec. 3.2.1] | flat in angle for `tf` without a peak; a peaked face is this locally reacting face (M6, 2.6) | expressible for resonant faces (*Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches*); partial for a flat `tf` (energy right on average, no angle shape), which the literature gives little weight | M6 (built) |
| a porous layer on a rigid backing, extended reaction | an equivalent fluid (`Z_c`, `k`) per frequency; the angle enters through the refracted wavenumber in the layer [Pind_2020_ExtendedReactingDG, Sec. II]; local reaction under-predicts on a rigid backing, error below 10 % (eight-octave mean) above 120 / 100 / 60 / 40 mm for 5 / 9 / 22 / 40 kPa·s/m² [Jeong_2011_LocalReactionGuideline, Sec. 4] | a coherent layer of a wave medium (`m = 0`, columns `a`–`h`) on a rigid mass-law wall, two-parameter mapping: extended reaction by construction | expressible; *Two-parameter mapping* (within 0.0008 in α at 0° to 80°); report 3.4 (within 0.003 per angle; the one-parameter backing off by up to 0.53 at 125 Hz, 60°) | M8 |
| a porous layer over an air gap | a transfer matrix with the angle-dependent cavity impedance `−jρ₀c₀·cot(k₀d₀ cos θ)/cos θ` [Pind_2020_ExtendedReactingDG, Sec. II; Masovic_2021_RoomAcousticsNotes, Sec. 8.4]; behind a gap local reaction over-predicts below 4 kHz, and per band local and extended reaction differ by 0.14 to 0.31 at 63 and 125 Hz for 10 mm of 40 kPa·s/m² over 150 to 350 mm [Jeong_2011_LocalReactionGuideline, Figs. 9-10]; one panel tripled `α_p` at 125 Hz with a 150 mm larger cavity [Ecophon_2021_MasterBrochure] | a leaf glued on a meshed box of air with a porous body at its back (2.8) | a leaf with the fill at the back expressible (*Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall*: within 2.6e-5 of the matrix of its columns); a porous layer directly behind a leaf with air behind it not expressible (no finite path sum, 2.8); a porous layer over a gap without a leaf not measured | M9 |
| a panel over an unpartitioned cavity | the peak moves by more than two octaves with incidence angle; a honeycomb partition removes the shift [Toyoda_2011_HoneycombPanelAbsorbers, Fig. 2]; an MPP curve shifts up by about `1/cos θ` [Maa_1987_MPPWideband, Eq. 27] | `tf` peak at `tfF` at every angle (a locally reacting face, M6: the partitioned cavity); the transmission dip (`resS > 0`) does move as `1/c` (2.1); or the leaf as a wave-medium sheet over the cavity (M9) | as a `tf` peak partial (a partitioned cavity expressible, an open one not); as a wave-medium leaf on a box of air expressible with the angle shift (2.8; *Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle*: the 12.5 mm board on 25 mm peaks at 128.4 / 148.6 / 182.4 / 258.8 Hz at 0° / 30° / 45° / 60° on a 1/96-octave sweep, on the transfer-matrix reference within 6.5e-5 in R; A18's 125 / 160 / 200 / 250 Hz were the nearest points of a 14-frequency grid) | M9 |
| perforated facing on a porous layer | nearly angle independent: equal absorption at 0°, 45° and 78° for a perforated gypsum ceiling [Hengst_2020_GelochteAkustikdecke] | flat `tf` | expressible | — |
| the edge effect of a finite absorber | `α_s = α_∞ + β·E`, `E` the free edge per area, 1.13 to 1.29 m⁻¹ for ISO samples [Vercammen_2012_SampleSize, Eq. 1]; `β` about 0.11 / 0.21 / 0.19 / 0.12 / 0.06 at 200 / 320 / 500 / 1000 / 2000 Hz for 50 mm mineral wool [Kuhl_1960_EdgeEffect] | a meshed panel absorbs its `tf` per face area; no edge term | not expressible as a material property; a small panel's excess can be carried in its `tf` | — (a parameter rule, A13) |

### 3.2 Resonant absorbers

All lumped resonant absorbers are a series resistance `R`, mass `m″` and stiffness `s″` per unit area; the
normal-incidence absorption is `α₀ = 4RZ₀ / ((R + Z₀)² + (ωm″ − s″/ω)²)`, with
`f₀ = (1/2π)·√(s″/m″)` and `α_max = 4R′/(1 + R′)²`, `R′ = R/Z₀`
[Masovic_2021_RoomAcousticsNotes, Eqs. 8.57-8.60; Kristl_2023_Verbundplattenresonatoren, Eqs. 3.1-3.18]. Written
as a shape, `α₀/α_max = 1 / (1 + (Q_α·(f/f₀ − f₀/f))²)` with `Q_α = 2πf₀m″/(R + Z₀)` (the review derives
`Q_α` [review 4.2]; the shape is that formula rearranged, derived here). The detuning is `f/f₀ − f₀/f`.

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| peak shape in frequency | the law above: symmetric in log frequency, zero at both ends, `f²` below the peak and `f⁻²` above it | the law, `tfP / (1 + (tfQ·(f/tfF − tfF/f))²)` with `tfQ` = `Q_α` (M5, built 2026-10-16). Before: the linear detuning `tfP / (1 + (tfQ·(f − tfF)/tfF)²)`, equal near the peak for `tfQ = 2·Q_α`, wide below and narrow above, with a floor of `tfP/(1 + tfQ²)` as `f → 0`. The board of *Absorption peak* (0.25 at 150 Hz), measured through the library at band centers, law with `Q_α` 0.75 against the old form with `tfQ` 1.5: 0.020 / 0.104 at 31.5 Hz, 0.079 / 0.142 at 63 Hz, 0.232 / 0.235 at 125 Hz, 0.152 / 0.125 at 250 Hz, 0.040 / 0.019 at 500 Hz (`md/m5_a21_report.md` 1.2) | expressible; *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face*; the six peaked rows of report 3.8 refitted on the law: within 0.017, cork board 0.047 (0.070 on the old form, converged; report 3.8 gave 0.076); Paris 0°–90° and field 0°–78° equal for a flat `tf` face; for a peaked face (M6) they differ, by up to 0.017 per octave on the six refitted rows (`md/m6_report.md` 0.6) | — (M5 built) |
| peak frequency of a membrane | `f₀ = (c₀/2π)·√(ρ₀/(m″d))` [Masovic_2021_RoomAcousticsNotes, Eq. 8.58], the constant 59.9 (60) with adiabatic air, about 50.6 with porous fill (isothermal) [review 4.2]; the 50 form with errors up to 10 % for either [Oldfield_2006_ImprovedMembraneAbsorbers, Eqs. 2.30-2.31]; measured peaks of 3 mm plywood 1.1 to 1.6 times the 60 value [Sakagami_2007_NonUniformCavity, Sec. 2.2]; edge clamping 1.24 and 1.54 times [Oldfield_2006_ImprovedMembraneAbsorbers, Sec. 4.4.2]; measured resonances 0.7 to 1.6 times the model [review 6.4] | `tfF` and `resF` are free numbers; under M9 the leaf on a box of its real depth (2.8) | expressible; the constant is a rule for deriving the column (A19, revised); under M9 the angle shift lifts the random-incidence peak 1.24 to 1.30 times above `f₀`, and the framed chamber peaks read 0.80 to 1.11 (A19) | — |
| height and width | `α_max` from `R′`, width from `Q_α`; light, deep resonators broad, heavy panels over shallow cavities narrow [Kristl_2023_Verbundplattenresonatoren, Sec. 3.1]; a loose porous layer of flow resistivity `Ξ` and thickness `d_a` adds about `Ξ·d_a/3` [Kristl_2023_Verbundplattenresonatoren, Sec. 3.1] | `tfP`, `tfQ` | expressible | — (`tfQ` is `Q_α` since M5) |
| the reflection phase of a resonant absorber | the complex `R` of the series impedance: a 12.5 mm board on 25 mm air swings from −18° to +15° through its resonance (model, report 3.5) | the phase of the face's series impedance (M6, built 2026-10-16; 2.6): ±4.12° on the board of 2.6 over-damped, through 180° under-damped | expressible; *Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches* | — (M6 built) |
| microperforated panel | Maa's model from hole radius, panel thickness, perforation ratio; half-absorption band about two octaves [Randeberg_2000_PerforatedPanelAbsorbers, Eqs. 2.49-2.52]; measured MPP ceilings peak near 0.75 at 160 / 400 / 630 / 800 Hz for 400 / 200 / 100 / 50 mm cavities [Fuchs_2006_MPPReview, Fig. 4]; double and triple MPPs widen the band to three and four octaves (modeled) [Cobo_2019_MultipleLayerMPPReview, Sec. 3]; a perforate is an equivalent fluid layer with porosity the perforation ratio, `σ = 8η/(φr²)`, `Λ = Λ′ = r` [Atalla_2007_PerforatedPlatesJCA, Sec. 2] | a `tf` peak; or a wave-medium sheet over a cavity | partial: the peak expressible (one band of about two octaves), the angle shift not (M7), the layer form not measured | M5 (built), M7, M9 |
| level dependence | linear behavior ends near a peak pressure difference of about 10 Pa across an MPP, stated as 114 dB [Maa_1987_MPPWideband, Eq. 15]; orifices become nonlinear near 5 m/s [Randeberg_2000_PerforatedPanelAbsorbers, Sec. 2.3.2] | the tracer is linear | not expressible (section 5) | — |
| gypsum cavity walls | `α(f) = 0.44·(f_MAM/f)² + α_r` above `f_MAM = 1900·√((m₁ + m₂)/(m₁m₂d))` (1362 with fill), `α_r` 0.045 single, 0.06 double or filled, validated from 80 Hz [Bradley_1997_GypsumCavityAbsorption, Eqs. 1-3] | `tfR` with `tfB = 0` for `α_r`, the peak's upper flank (falling as `f⁻²` in both forms) for the first term | expressible above `f_MAM`; not measured | M5 (built) |
| curtains | a fabric sheet is a specific airflow resistance `R_s` in parallel with its mass impedance `jωm`, the gaps as distributed elements; within 0.01 ± 0.05 of tube measurements for 24 fabrics [Pieren_2015_MultilayerCurtains, Secs. 2, 3.4]; fullness has no model [DelRey_2013_CurtainFullness] | a `tf` per hanging (power law and peak); or a resistive wave-medium leaf on a box of air (2.8) | expressible as a fitted `tf`; as a leaf the power-law `μ` columns reach the parallel form within 0.008 to 0.039 in random-incidence α for seven fabrics of [DelRey_2013_CurtainFullness] at 145 mm, fitted on that average, but not per angle (up to 0.35) where `f_t = R_s/(2πm)` lies in the band (`m9_fixtures_report.md`, item 6) | M9, M16 |

### 3.3 Porous models

All equivalent-fluid models supply `Z_c` and `k` per frequency; the surface impedance follows from thickness
and backing [Jaouen_2026_APMRMotionlessSkeleton].

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| which model the columns implement | — | none: `a`, `b` (`ε′` power law), `c`, `d` (conductivity power law), `e`, `f`, `g`, `h` (the same for `μ`) are fitted per material to a model's `ε = n·z`, `μ = n/z`. Round 1 fitted Delany-Bazley in its range and Miki below it (report 3.4); the fixture uses Delany-Bazley at one frequency; `tools/acoustic_parameter_mapping` uses only Delany-Bazley's attenuation term (report 1.5) | expressible within ±0.025 / ±0.032 for the Knauf board (report 3.4) | — |
| Delany-Bazley validity | `0.01 < f/σ < 1` (SI), non-physical `Re(Z_s)` at low `f/σ` for thin layers [Jaouen_2026_APMRDelanyBazley]; the lower bound is 50 / 100 / 200 / 400 Hz for σ 5 / 10 / 20 / 40 kPa·s/m² [review 4.1], so typical mineral wool is outside it at 31.5 to 125 Hz; some sources write `ρ₀f/σ`, a factor 1.2 [Jaouen_2026_APMRDelanyBazley] | the columns are fitted, not the model | a rule for deriving the columns: Miki, Komatsu for mineral wool, over the whole band (A15, decided) | — |
| the low-frequency model | Miki is positive-real and causal below `f/σ = 0.01` [Miki_1990_ModificationsDelanyBazley, Sec. 4]; Komatsu is better for glass and rock wool at `f/σ < 0.01` and `> 0.1` m³/kg [Komatsu_2008_ImprovementDelanyBazley, Sec. 4-5]; JCAL and Horoshenkov's Padé model have the right low-frequency asymptotes [Jaouen_2026_APMRJCAL; Jaouen_2026_APMRHoroshenkov] | power laws fitted to the chosen model | expressible where a power law follows the model within the guideline (report 3.4: 31.5 Hz–8 kHz) | — |
| frame motion | rigid frame only well above `f_d = σφ²/(2πρ₁)` [Jaouen_2026_APMRDecouplingFrequency]; for three characterized materials `f_d` about 890, 57 and 186 Hz [review 4.1]; limp-frame model for light free-hanging layers [Doutres_2007_LimpModelValidity] | an equivalent fluid of fitted columns; a limp frame changes `ρ`, so `μ`, which the columns can carry | expressible as a fit; not measured | — |
| carpets | tufted 12 mm wool carpet on a rigid backing below 0.1 up to 250 Hz (normal incidence) [Broda_2025_TuftedCarpets, Sec. 3.2]; carpet tiles 0.00 to 0.03 at 125 Hz [BCTC_2014_EcosoftCarpetTile] | a thin porous layer on a rigid wall (as 3.1), or a `tf` | expressible; not measured | — |

### 3.4 Transmission

The thin infinite plate: `τ(θ) = |1 + Z(θ)·cos θ/(2ρ₀c₀)|⁻²`,
`Z(θ) = mω·[j(1 − r²sin⁴θ) + η·r²sin⁴θ]`, `r = f/f_c`, three parameters (m, f_c, η)
[Davy_2009_SingleLeafCremer, Eqs. 3, 7, 8].

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| mass law, angle | `τ(0) = 1/(1 + a²)`, `a = ωm/(2ρ₀c₀)`; diffuse average with weight `2 cos θ sin θ` [Davy_2009_SingleLeafCremer, Eqs. 1, 13; London_1949_SingleWalls] | the limp mass law, `m > 0` a switch at 20 dB per decade (2.1, M10); also a wave-medium sheet (section 1, 2.8) | expressible; *Mass law - m switches the limp mass law on*; *Thin sheet - mass-law angle dependence*; *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf* | M9 |
| coincidence and loss | the `η·r²sin⁴θ` term is resistive and adds to the one inside the modulus [Davy_2009_SingleLeafCremer, Eq. 8] | `X₀·\|1 − (f/coiF)²·sin⁴θ·(1 + j/coiQ)\|`, then `TL` from the modulus as a reactance: the resistive part enters in quadrature. At coincidence the physical `τ` is `1/(1 + X₀ηc)²`, the library's about `1/(1 + (X₀ηc)²)`, 3 dB apart where `X₀ηc ≈ 1` (derived here) | expressible where `X₀ηc` is large; *Coincidence - …*; 13 constructions within 2.97 dB at 0–90° under M10 (`md/m10_m15_report.md` 1.0) | M15 (proposal) |
| the installed loss factor | `η = η_int + η_edge + 2η_rad` [Davy_2009_SingleLeafCremer, Eq. 9]; for laboratory specimens `η = η_int + m/(485√f)` [Davy_2009_SingleLeafCremer, Eq. 11]; installed values 1.3 to 4 times the isolated ones [Davy_2019_MassAirMassResonance; Quirt_1995_GypsumWallsSummaryIR693]; round 1: the coupling loss is the largest single effect on heavy walls (report 3.1) | `1/coiQ`, constant in frequency | partial; the fits reach the guideline without it (report 3.6; with it no better, `md/m10_m15_report.md` 0.2) | M13 (dropped) |
| finite size below `f_c` | infinite-plate theory to 90° under-predicts measured R; a limiting angle of 85° fitted NRC data better than 78° [Northwood_1968_PlasterboardWalls, p. 2]; a finite panel's forced radiation efficiency replaces it [Davy_2009_ForcedRadiationEfficiency]; a 1.55 × 0.95 m specimen read about 5 dB above a 2.4 × 4.8 m one [Wareing_2015_SampleSizeVariations]; round 1: a lab pane 7.8 to 11.2 dB above the infinite plate at 31.5 Hz (report 3.1) | the finite panel in the mass term (2.1): Davy's forced radiation efficiency of a square of side `side` in place of `1/cos θ`, within 0.4 dB diffuse and 0.8 dB per angle of the windowed finite plate; the 78° / 85° conventions are this at one size and frequency each: the limit that makes the infinite plate match the finite one is `cos θ_l ≈ 1.53/√(kL)`, 78° at 0.9 kHz on a 4.0 × 2.7 m wall and at 2.2 kHz on a lab pane, 85° at 5 kHz on the wall, none at 31.5 Hz (`md/grazing_report.md` 0.3, 0.5); a measured size effect of a small specimen reproduced in sign and about half its size (1.1 to 2.8 dB, 1.9 to 3.5 with the niche, of Wareing's "more than 5 dB"; the rest edges and rooms) | expressible: `side` (2.1, decided 2026-10-19) | (d) built, M14 closed |
| single-plate accuracy | for nine homogeneous plates the spectral RMS error over 50 Hz to 5 kHz was 6 to 7 dB for mass law and Sharp, about 3 dB for Davy's finite-size model [Kosala_2020_SingleBafflesAccuracy, Table 2] | fitted per construction (P1) | — | — |
| double leaves | below the mass-air-mass resonance one leaf of the combined mass, above it a steep rise until stud bridging [Davy_2019_MassAirMassResonance]; double glazing gains about 3 dB per doubling of the air space instead of 6 [Quirt_1982_WindowsI, Sec. II.D]; of 17 models against 52 walls none was acceptable for all wall types, the best below 10 dB per band, the choice set by stud type and fill [Hongisto_2006_DoubleWallsOverview, Secs. 5.2-5.3]; the mass-air-mass constant 60 adiabatic, about 51 isothermal, about 80 in Sharp's model [review 9.10] | one sheet per unit with `resS > 0` (the dip, capped at one, B4m by P1) | expressible as a fit per measured construction (4-16-4 0.71 dB, triple 0.77 dB); not predictive. Two wave-medium leaves (`m = 0`, two-parameter mapping) resonate: 4\|16\|4 on an air body within 0.007 dB of the transfer-matrix reference at 50 Hz–1 kHz, 0° to 60°, the dip at 211.3 / 244.0 / 298.8 / 422.6 Hz at 0° / 30° / 45° / 60° on a 1/96-octave sweep (2.8; *Leaf - two leaves on a box of air resonate at the mass-air-mass frequency*; A18's 200 and 315 Hz were grid points) | M9 |
| a lining on a wall | a leaf over a cavity: the lined wall's R falls around the leaf's resonance and rises above it; gypsum linings on masonry lower R at 50 to 160 Hz by up to 8 dB without fill [Zeitler_2015_ConcreteBlockRR334, Table A1.2] | the lined wall as one material with `res*` and bending (report 3.8: furred board 1.70 dB, board on dabs 2.51 dB) | expressible as a fit | M9 |
| flanking | per junction three paths with the vibration reduction index `K_ij` [Mahn_2023_GuideAirborneRR331, Eqs. 1.1, 2.3-2.4]; cannot be a transmission coefficient of one surface [review 2.5] | — | not expressible (section 5) | — |
| the low octaves | R at 31.5 Hz: no open data; at 50 to 80 Hz the test rooms are modal [review 6.1, 6.3]; round 1: model spreads 4.5 to 13.9 dB (heavy) and 5.2 to 13.5 dB (light) at 31.5 / 63 Hz (report 3.1) | the mass term, `res*` in both forms | expressible within the spread (P7; report 3.8) | — |

### 3.5 Ground

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| plane-wave reflection with phase | the complex `R_p(θ)` of the ground impedance [Fuchs_2025_GroundImpedanceNTACOU104, Eqs. 4a-4d] | a wave half-space or a layer on a rigid backing (two-parameter mapping, columns fitted to the model) | expressible as in 3.1; not measured on a ground | — |
| spherical-wave reflection | `Q = R_p + (1 − R_p)·F(w)`, `w = √(j·k·r_r/2)·(cos θ + 1/Z)`; toward grazing `R_p → −1` and the ground wave governs the field; an energy coefficient cannot reproduce the ground dip [Fuchs_2025_GroundImpedanceNTACOU104, Eqs. 4a-4d] | an event has no path length to the receiver | not expressible (section 5) | — |
| model-specific parameters | fitted σ_e depends on the model: grassland 41 to 1932 kPa·s/m² (Delany-Bazley) against 45 to 2251 (slit pore) on the same 26 sites [Bashir_2014_PhDRoughMixedImpedance, Table 4.6]; one-parameter models can acquire a negative real part at low frequency [Attenborough_2016_ExploitingGroundEffects, Sec. 3]; hard-backed layers of 0.01 to 0.05 m [Bashir_2014_PhDRoughMixedImpedance] | fitted columns per material; a layer on a backing | expressible; the model tag belongs to the material table | — |

### 3.6 Scattering and diffusion (out of scope, P9)

The literature requires a per-band scattering coefficient `s`, the specular part `(1 − α)(1 − s)` and the
scattered part `(1 − α)·s` [Vorlaender_2004_ScatteringCaseStudies, Sec. 2], distributed by Lambert's law by ray
splitting, random choice, vector mixing or diffuse rain [Savioja_2015_GeometricalModeling, Sec. IV.B], with
a size- and distance-dependent edge term for small surfaces [Shtrepi_2017_DistanceDiffusive, Eq. 1;
Odeon_2025_UserManualV19, Sec. 9.5] and estimation rules that differ by about a factor of 10 at 125 Hz
[review 4.6]. The library has no scattering mechanism: the resolved-type table of `ray_state_update` reserves
the codes from 192 with "scattering not implemented" (`src/ray_mesh_interact.cpp`, the type table of
`resolved_typeN`). A band shares one polyline (`md/wave.md` 6), so a scattered direction could not differ
per frequency within a ray [Savioja_2015_GeometricalModeling, Sec. IV.B]. Status: not expressible; out of
scope (section 5).

### 3.7 Objects, furniture and people

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| absorption of an object | equivalent absorption area per object in octaves 125 Hz to 4 kHz (ISO 354, ISO 20189) [Thorsson_2023_FurnitureAbsorptionN10]; for simulation an object coefficient `α_obj = A_obj/S_E`, `S_E` the exposed area of a simplified cuboid [Nocke_2018_ObjectAbsorption, Eqs. 4-5] | a meshed proxy box of rigid mass-law faces with `tf = α_obj` | expressible; not measured | — |
| audience and seating | coefficients per floor area; placement as a plane, a raised plane or a block changed predicted STI by up to 0.08 [Hammond_2018_StandingAudience, Sec. 3]; edge strips 0.4 to 0.8 m per aisle side [KathKuhl_1964_PublikumAbsorption, Sec. 3] | a proxy surface or block with `tf` | expressible as geometry and `tf` | — |
| furniture as volume scatterers | mean free path `4/(n·q)` [Picaut_2012_SPPS, Sec. 2.2.2.4]; equivalent scattering absorption area, room dependent [Nilsson_2007_NTTR606, Sec. 4.3] | — | not expressible; out of scope (P9) | — |

### 3.8 The reflection phase (A12)

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| phase in a pressure-based tracer | the complex `R(θ)` of the boundary; neglecting the phase raised level errors by up to about 2.4 dB in phased beam tracing against BEM [Jeong_2012_BoundaryConditionsPhasedGA, Sec. V-VI]; phase in early reflections improves mid-frequency accuracy (Jeong et al. in [Savioja_2015_GeometricalModeling, p. 719]) | every path carries a complex amplitude | expressible | — |
| a hard wall | a phase near zero at every angle | the two-parameter mapping: 0.009° at 10°, 0.104° at 60° (concrete), 3.1° at 89° (glass) (report 3.5) | expressible; *Two-parameter mapping* | M8 |
| a porous layer on a wall | the phase of the layer impedance | the coherent layer (3.1) | expressible | — |
| the pad of a rigid face (`1 − tf`) and a resonant absorber | the phase of the face's impedance; ±18° through a panel absorber's resonance (model, report 3.5) | a resonant (peaked) face: the phase of its series impedance (M6, 2.6); a flat pad keeps the reactance of the bare face (`Material::closed_face`), `tf` sets its resistance only | expressible for locally reacting resonant faces, every type (*Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches*); a flat pad has the hard wall's phase | M6 (built) |
| finite-size reflection | a finite-size reflection coefficient reduced phased-GA errors at low frequency and degraded them above the Schroeder frequency [Jeong_2013_ReflectionAbsorptionCoefficients, Sec. 2-3] | — | not expressible; not proposed | — |

### 3.9 The modal region

The Schroeder frequency `f_s = 2000·√(T/V)` marks the lower limit of statistical room acoustics
[Odeon_2023_UserManualV18, Sec. 9.1]; for a 75.6 m³ room it is 126 to 178 Hz for T of 0.3 to 0.6 s [review 6.5],
so 31.5 to 125 Hz is modal there.

| quantity | physics requires | today | status | candidate |
| --- | --- | --- | --- | --- |
| modes from a coherent tracer | complex impedance boundaries; a phase-carrying ray method reproduced the lowest modes of a rectangular room, 3 to 5 % reverberation-time deviation at 125 Hz for real impedance, lowest-mode peaks under-estimated [Prislan_2017_RayTracingExtendedReaction, Conclusion] | coherent paths with complex boundary factors; wavelengths of 5.4 m (63 Hz) and 10.9 m (31.5 Hz) against room size (report 1.9) | partial: the boundary phase of hard walls, porous layers and locally reacting resonant faces is right (3.8); not measured on a room | M6 (built) |
| energy storage of a resonator | a plate absorber tuned within 0.1 Hz of a room mode gave a beating decay [Nell_2019_TieffrequenteAbsorption, Sec. 6.3]; an energy tracer with per-band `α` cannot represent this coupling [review 6.5] | a frequency-domain tracer carries a resonator through its complex reflection factor | the factor exists (M6, 2.6); the coupling rests on the reflection order traced; not measured on a room | M6 (built) |
| position-dependent low-frequency absorption | one absorber gave about 1.07 and 0.43 to 0.67 at two modes 0.5 Hz apart [Nell_TIP_TieftonAbsorber, Table 8, Sec. 7.1]; edge and corner placements read 1.20 to 1.77 in situ [Huber_2018_AufnahmeraumTieffrequent, Table 1] and 4 to 6 per face area [Fuchs_2020_BreitbandSchallabsorber, Bild 5]; these describe a placement in a modal field and are not material properties [review 6.2] | a coherent field forms them from the geometry | expressible as a field result; such values must not enter `tf` (A13) | — |

## 4. Candidate changes to the machine

Every item is a proposal; Stephan decides. Ordered by value for the reflect side (P6). Kind is addition,
modification or removal / simplification. "Rows" are the rows of section 3 the candidate moves. Invariants for
every candidate (section 9): both ports at most one, sheet on traced solid, no step in angle, a band on its
single frequencies, the `[wave]` baseline and twins untouched. The per-ray state cannot grow (section 7);
none of these candidates needs a state bit.

**M5 — the `tf` peak on the lumped-resonator law** (modification).
- What: `tfP / (1 + (tfQ·(f/tfF − tfF/f))²)` in `Material::tf`; `tfQ` becomes `Q_α` (about half today's value
  for the same peak).
- Basis: the series-resonator absorption [Masovic_2021_RoomAcousticsNotes, Eqs. 8.57-8.60;
  Kristl_2023_Verbundplattenresonatoren, Eqs. 3.1-3.18].
- Rows: 3.2 peak shape, gypsum cavity walls, MPP height. Gain: below the peak, the board of 2.6 absorbs
  0.104 / 0.142 at 31.5 / 63 Hz today against 0.020 / 0.079 on the law (derived here); above it 0.125 / 0.019
  at 250 / 500 Hz against 0.152 / 0.040. The 125 Hz octave and up moved little in round 1's fits (report 3.8).
- Cost: one line; no column; no state.
- Risk: `tf()` is shared with wave faces; no `[wave]` case carries `tfP` (the baseline did not move when A10
  was built), so no gate is needed, and a gate on `m > 0` is the alternative. Moves *Absorption peak* and
  *ray_progress - the plate rule and the absorption peak at every frequency of a band* (references from the
  column definition, restated).
- Acceptance: the peaked rows of report 3.8 refitted (on the linear detuning within 0.017, cork 0.070 converged;
  report 3.8 gave 0.076); the low-band references
  of 9 (the 6 mm MDF membrane at 63 to 125 Hz [Shearer_2016_AbsorptionBelow100Hz, Fig. 5], the plywood peaks
  [Sakagami_2007_NonUniformCavity]).
- Priority: high.
- Decided (Stephan, 2026-10-15): with M5, `tfQ` is `Q_α`.
- Built (2026-10-16, `md/m5_a21_report.md`): the line in `Material::tf`, no gate (no wave case carries `tfP`; `[wave]`
  identical). The six peaked rows of report 3.8 refitted on the law (minimax on the octave means, 150 starts):
  window glass 0.013, plasterboard on frame 0.017, double glazing 0.008, plasterboard on 25 mm battens 0.006, wooden
  lining 0.010, cork board **0.047** (0.070 on the linear detuning); every fitted peak lies in the band (92 to 493 Hz),
  where two old-form fits used the floor below 25 Hz as a flat lift. Through `ray_progress` the fitted faces absorb
  the formula within 1e-6, Paris and field alike. The two cases restated with the board's `Q_α` 0.75 (the width near
  the peak of `tfQ` 1.5), names and bounds unchanged. The low-band references of 9 (MDF membrane, plywood peaks) are
  not part of this change.

**M6 — the room-side face as a series impedance: phase and angle** (addition, A12).
- What: where a peak is set, the pad's resistance (from `tf`) and the face's reactance become the series
  impedance `ζ = R′ + j·(1 + R′)·tfQ·(f/tfF − tfF/f)` per unit `Z₀`, with `R′` from `tfP = 4R′/(1 + R′)²`,
  met as `ζ·cos θ` (local reaction). Its normal-incidence absorption is M5's law; its reflection carries the
  phase and the angle dependence. The power-law part stays a resistive pad, `|r|² = (1 − tf_bg)·|r_ζ|²`.
- Basis: [Masovic_2021_RoomAcousticsNotes, Eqs. 8.57-8.60; Savioja_2015_GeometricalModeling, Eqs. 4-5]; the phase
  matters up to 2.4 dB [Jeong_2012_BoundaryConditionsPhasedGA, Sec. V-VI].
- Rows: 3.1 angle dependence (resonant faces), 3.2 phase, 3.8 pad and resonant absorber, 3.9 energy storage.
  Gain: the ±18° swing of a panel absorber (report 3.5); the random-incidence value of a locally reacting
  peak is below its `α₀` (up to 0.951, [Jeong_2009_AngularDistributionCorrection]), so the fit target becomes
  the Paris integral of the face.
- The damping branch: `α_max` has two roots, `R′` and `1/R′` (under- and over-damped), with one absorption
  curve at normal incidence. They reflect with opposite sign at `f₀` (`r = (R′ − 1)/(R′ + 1)`, derived here):
  the under-damped face is a pressure-release boundary at its resonance, and its absorption falls toward oblique
  incidence where the over-damped one rises. The lumped model ties the width to the mass,
  `1 + R′ = ωm″/(Q_α·Z₀)` (from the formulas of 3.2): the round-1 board (8.6 kg/m² on 25 mm, `ωm″/Z₀` ≈ 16.9 at
  129 Hz) reaches the measured linings' low, broad peaks (about 0.3, one to two octaves, report 1.10) only
  over-damped (`R′` ≈ 10.5 for 0.32 with `Q_α` ≈ 1.5; the under-damped root 0.095 needs `Q_α` ≈ 15), and the
  round-1 panel model is over-damped (`R′` ≈ 3.4, `r` positive at the peak, −12° to +15°,
  `p3/ab/liningphase.out`). Under-damped faces are light, lightly damped resonators (an unfilled Helmholtz
  resonator, a thin foil). The branches converge as `tfP → 1`.
- Decided (Stephan, 2026-10-15): over-damped by default, `tfQ < 0` selects under-damped (a relaxed
  `require`); the mass relation is the parameter session's rule for the branch, not code (from the mass it
  would over-determine `tfP` and `tfQ` against P1). The detailed design follows the A18 measurement.
- Cost: `Material::closed_face` and `Material::tf_face` (the immittance the pad is formed on); the cosine in air
  at the face (known at the event, as `SHEET_RESOLVE`'s mass cosine); no column; no state.
- Risk: the plate rule reads the face's absorption through `PLATE_ABS` (`interact_with`), so `a` becomes
  angle dependent for a peaked face; sheet on traced solid must hold (the pad lives at the face in both);
  restates *Absorption peak*, the band case of M5, and the phase rows of a peaked face; no wave face carries
  `tfP`.
- Acceptance: the phase and `|r|` of a peaked face against the series-impedance reference at 0° to 85°, as
  sheet and traced solid; the random-incidence value of the A10 rows.
- Priority: high (A12 is a priority, P8). Depends on M5.
- Decided (Stephan, 2026-10-15, after A18): lower priority, for locally reacting peaks only (a partitioned
  cavity, a face known from catalog data); an unpartitioned construction with a known build is M9.
- Built (2026-10-16, `md/m6_report.md`), the design of phase 0 as decided (section 10): D1 `ζ` met against the
  immittance of the side the wave is on (local reaction: the transfer matrix of the face behind a lining; the cosine in
  air alone was off by up to 0.68 in `|r|` there and created energy behind a porous lining); D2 `ζ` replaces the bare
  face's reactance (combined with it, the one-parameter mapping brought back its `2θ` phase and almost no absorption);
  D3 the power-law part a matched pad in front of `ζ` (folded into the resistance it cannot be built: `ζ`'s reactance
  bounds the absorption; added in α it is no element); D4 gated on `m > 0` (a peaked wave medium created energy without
  it), EM TE `ζ·c`, TM `ζ/c`; D5 `tfQ < 0` the under-damped root, the `require` on `tfQ` removed, `R′` from
  `min(tfP, 1)`. In `Material`: `peaked()`, `zeta()`, the load of `closed_face` and the open face of `tf_face`;
  `src/ray_mesh_interact.cpp` 5132 lines against 5089. The six peaked rows reach their Paris targets with the residuals
  of M5 (window glass 0.0132, plasterboard on frame 0.0171, double glazing 0.0075, battens 0.0056, wooden lining
  0.0101, cork 0.0476, over-damped); their field values 0°–78° differ from the Paris values by up to 0.017 per octave. The 13 transmission
  fits of report 3.6 carry no peak and do not move. Asserted in the four cases of 2.6.

**M7 — the cavity's angle shift on the peak** (addition).
- What: for a panel over an unpartitioned cavity the peak sits at `tfF/c` (the cosine in air), flagged by a
  sign (`tfF < 0`, today required non-negative); with M6 the stiffness term carries `1/c²`.
- Basis: the rear impedance of an air cavity at oblique incidence, `−jρ₀c₀·cot(k₀d₀ cos θ)/cos θ`
  [Pind_2020_ExtendedReactingDG, Sec. II]; the shift of more than two octaves [Toyoda_2011_HoneycombPanelAbsorbers,
  Fig. 2]; about `1/cos θ` for an MPP [Maa_1987_MPPWideband, Eq. 27].
- Rows: 3.1 panel over an unpartitioned cavity, 3.2 MPP. Gain: the angle distribution of a peaked face. Its
  random-incidence value moves too: the board of 2.6 on the law with `Q_α` 0.75, peak at `f₀/cos θ`, Paris
  integral to 90°, peaks at 209 Hz with 0.198 against 150 Hz with 0.25 (derived here, `ac2/peak2.out`). A fixed
  peak fitted to a diffuse-field measurement reaches that value already, so the gain is at non-diffuse
  incidence: early reflections and the listening seat.
- Cost: the cosine into `Material::tf` (four calls inside `Material`) or into M6's immittance; a relaxed
  `require`; no column; no state.
- Risk: as M6; the transmission dip of `resS > 0` already moves as `1/c` (2.1), so the two stay consistent.
- Acceptance: angle-resolved against the panel-over-cavity model with the extended rear impedance, 0° to 85°;
  its Paris integral against the measured random-incidence row.
- Priority: medium-high.
- Decided (Stephan, 2026-10-15, after A18): dropped. A leaf on an air body (M9) forms the shift from the wave
  physics; a partitioned cavity has none.

**M8 — retire the one-parameter mapping for materials** (removal, rule of use).
- What: section 1 and the rules of use name only the two-parameter mapping for acoustic materials. The
  fixtures that use the one-parameter mapping stay as guards of the machine.
- Basis: N7; the 2θ phase of a rigid half-space has no physical counterpart (report 3.5); a porous layer on a
  one-parameter wall absorbs up to 0.39 too much at 125 Hz and 0.53 at 60° (report 3.4).
- Rows: 3.1 porous layer, 3.8 hard wall.
- Cost: documentation only (no code reads the mapping).
- Risk: none in the suite; a material table built in the old form is wrong in phase and in a layer's absorption.
- Acceptance: none beyond the existing *Two-parameter mapping* case.
- Priority: high (cheap).

**M9 — a leaf as a wave-medium sheet** (addition; later a possible simplification).
- What: a light leaf (a panel absorber's face, a curtain, an MPP as an equivalent fluid, a pane of a meshed
  double unit) is a coherent sheet of a wave medium (`m = 0`) in the two-parameter mapping, its loss as a loss
  of `μ` (a mounting resistance `r_m·ωm″` is a constant loss ratio on `μ`: `g` with `h = 1`, derived here, not
  measured). The cavity behind it is a meshed box of near-air material (2.8).
- Basis: the series impedance `jωm″/Z₀` of the thin layer (section 1; measured here on 4 mm glass within
  0.002 dB of the limp mass law, 0° to 85°, 500 Hz); a perforate as an equivalent fluid
  [Atalla_2007_PerforatedPlatesJCA, Sec. 2]; a curtain as a resistive limp sheet [Pieren_2015_MultilayerCurtains,
  Sec. 2].
- Rows: 3.1 porous layer over an air gap and panel over a cavity, 3.2 MPP and curtains, 3.4 double leaves and
  linings. Gain: the mass-air-mass and leaf-over-cavity resonances, their phase and their angle shift from the
  wave physics, with no lumped column; the old item 5 (a meshed double leaf cannot resonate) was measured
  with mass-law sheets in the one-parameter mapping, whose layers carry no mass reactance.
- Measured (A18, 2026-10-15, `ac2/a18.py`, the walker through the library at 50 Hz–1 kHz, 0° to 60°, against a
  transfer-matrix reference of fluid layers): the panel absorber (12.5 mm board, 8.6 kg/m², a mounting resistance
  as a loss of `μ`, over 25 mm on concrete) over a meshed gap within 2e-6 in R, over a traced body of air
  (`ε` 1, `μ` 1.000001, the leaf glued on it) within 6.7e-5 summed in one event; 4|16|4 of two 4 mm glass leaves
  over an air body within 0.007 dB. The meshed gap is exact where its path series converges and needs 800 to more
  than 4,000 events per ray (a lossless double leaf at 1 kHz did not converge in 4,000: up to 7.3 dB off). An
  ordered stack is two sheets in one medium and cannot carry the wall, so the summed route is the air body. A
  fluid layer has no bending stiffness: coincidence stays with the mass column (section 5), and so does the
  stiffness of a small clamped panel's edge; the framed chamber peaks of A19 it reaches at full depth through the
  angle shift alone (0.80 to 1.11, A19 revised).
- Cost: no code for the leaf, the cavity and the double leaf; the air body is a meshed box of near-air material
  behind the leaf. Measured after A18 (M9 phase 0, `md/m9_fixtures_report.md`): a band through `ray_progress`
  equals its single frequencies; tracer 5 equals tracer 4 on the box; a porous fill only at the back of the box;
  the cost per ray (2.8).
- Risk: a wave-medium leaf past its critical angle tunnels (a rigid body in the two-parameter mapping is rarer
  than air from a few degrees); the thin-layer result holds while the leaf is thin against the wavelength in
  it; a traced rigid solid without the mass column is closed by its first face (`md/wave.md` 8), so traced
  walls keep the mass column.
- Acceptance: a single leaf in a fixture against the limp mass law with phase; leaf | gap | wall against the
  transfer-matrix model and 60/√(m″d) (report 1.10: 129 / 91 / 65 Hz for 8.6 kg/m² on 25 / 50 / 100 mm); 4|16|4
  against its mass-air-mass resonance (211.4 Hz, report 1.4) and the measured unit (AGC, report 1.4).
- Decided (Stephan, 2026-10-15, after A18): M9 is the route for an unpartitioned construction with a known build
  (a leaf over a cavity, a double leaf, a curtain, an MPP over an open cavity): the leaf a wave-medium sheet glued on
  a meshed box of near-air material (`μ` 1.000001). A partitioned cavity and a face known only from catalog data stay
  on the `tf` peak (M5, M6). Transmission of a double unit through its coincidence stays on the lumped sheet (N3,
  N4). Next: the A18 runs as fixtures, then the open measurements (a band, tracer 5, a porous fill, a curtain, the
  shallower box of A19).
- Built (2026-10-15): the rule of use 2.8 and five fixtures in `test_acoustic_leaf.cpp`; no library change. A18
  closed; the curtain's relaxation is M16, the open conventions A20 and A21.
- Later simplification, if it holds: the mass column on sheets duplicates the mapping; it stays for traced
  solids, the held wave, bonded plates and bending.
- Priority: medium (the reflect side through panel absorbers and curtains, the transmit side through double
  leaves).
- Decided (Stephan, 2026-10-15): measured first (A18), the ordered stack (leaf | air sheet | wall, the cavity
  series summed in one event) first and the meshed gap for comparison. A meshed gap behind a heavy leaf is a
  long series of separate paths: 8.6 kg/m² passes about 1.5 % per crossing at 125 Hz (derived here from
  `1/(1 + X₀²)`, `X₀` 8.2).

**M10 — `m` fixed at 20** (simplification).
- What: the mass term as the physical mass law; `m` becomes a switch.
- Basis: `TL = 10·log10(1 + a²)` at normal incidence, 20 dB per decade [Davy_2009_SingleLeafCremer, Eq. 13];
  measured heavy walls rise about 6 dB per octave (report 1.3).
- Rows: 3.4 mass law. Gain: the mixed-slope limitation of section 7 and the `0.301·(20 − m)` offset of 2.2 go;
  `MASS_FOLD` loses its exponent and its slope exclusion.
- Measured (`md/m10_m15_report.md` 0.2, 1.0): the 13 constructions within 2.80 dB at 0–78° and 2.97 dB at 0–90°
  (gypsum, a model reference; every measured construction within ±3 dB). The light panels then fit with their mass
  and coincidence; on heavy walls and linings the slope's job moves to `att` and the level (a level far below the
  mass with a flat `att`, up to the 12 dB bound). Not together with M11 (the furred lining 3.33 dB in measured
  octaves).
- Decided (Stephan, 2026-10-17): adopted. The restated cases keep their `m` 30 / 12 palettes and write their
  references at slope 20; "where the slopes agree" leaves the two plate-depth case names.
- Built (2026-10-17, `md/m10_m15_report.md` 2): `mass_X0` without the exponent, `MASS_FOLD` the plain sum over every
  bonded mass-law body, the comments and the public `medium_gain` text; with D2-b `src/ray_mesh_interact.cpp` 5116
  lines against 5132. Six cases restated (288 assertions), the `ray_progress` plate-depth twin renamed, the case *Mass
  law - m switches the limp mass law on* new; five print-only cases move (*Leak direction* twice, *ray_progress - the
  wall behind a traced layer …*, *Lossy verdict - … the mass-law part*, *Refracting tracer - … the mass-law part*: the
  levels of an `m` 30 wall).

**M11 — no `att` on a mass-law material** (simplification).
- What: rule of use `att = 0` with `m`.
- Basis: none in the review for a per-entry loss of a plate; flanking and leaks are not a surface property
  [review 2.5]. The fits used `att` as a level (0 to 5.44 dB, `p3/tl/rmb_chk.out`).
- Measured (`md/m10_m15_report.md` 0.2): alone within 1.51 dB (the block wall needs no `att`: 0.33 dB with slope 7.5),
  but with M10 the furred lining reads 3.33 dB in the measured octaves 250 Hz–4 kHz: one non-physical knob has to
  shape heavy walls and linings, the slope or the flat loss.
- Dropped (Stephan, 2026-10-17): `att` stays, the declared empirical level of the constructions the infinite plate
  does not describe (report 3.1); A5 stays a rule of use.

**M12 — `coiA` ignored where `m > 0`** (simplification).
- What: `interface_gain` skips the coincidence Lorentzian on a mass-law material; the rule of use becomes code.
- Measured (`md/m10_m15_report.md` 0.3): no case reaches it; the one condition moves no row.
- Superseded (Stephan, 2026-10-17) by **D2-b: `coiA` removed altogether**. No physical reason remained: coincidence on
  a mass-law plate is `coiF` / `coiQ` in `X₀`, no material file uses `coiA`, and an angle-free Lorentzian in dB is no
  physical element. Built (2026-10-17): out of `MaterialCols`, `Material`, `same_as`, `interface_gain` and the public
  texts, out of the column lists of the OBJ writer and the Python / MATLAB readers and of
  `tools/acoustic/band_probe.cpp`; `src/quadriga_lib_material_helpers.hpp` deleted. A table that still carries the
  column loads and the column is ignored (*Coincidence column - a table that carries coiA loads and the column changes
  nothing*, `[wave]`, and its *- the mass-law part*). The three `[wave]` cases that tested the Lorentzian are restated
  (`md/wave.md`), the only wave change of this round.

**M13 — the installed loss factor in frequency** (modification).
- What: `η(f) = 1/coiQ + C·(f/coiF)^(−1/2)` or a fixed `1/√f` coupling term.
- Basis: [Davy_2009_SingleLeafCremer, Eq. 11]; report 3.1 (coupling loss the largest single effect, fitted
  `C_e`).
- Measured (`md/m10_m15_report.md` 0.2): no variant's worst residual moves by more than 0.08 dB, with or without M10,
  with the coincidence free or at its physical frequency; the fitted column is not Davy's (`C` 0 on every light panel,
  0.2 to 3.9 on walls). Davy's fixed law needs no column under M10 (`fRef` carries `m″`) but the acoustic mapping's
  constants in the library; the column touches the OBJ writer, both readers and the format document.
- Dropped (Stephan, 2026-10-17).

**M14 — a limiting angle for finite size** (modification).
- What: the mass cosine bounded below by `cos 78°` to `cos 85°` [Northwood_1968_PlasterboardWalls, p. 2].
- Basis: [Northwood_1968_PlasterboardWalls, p. 2]. Risk: a kink in angle (no step); moves *Thin sheet -
  mass-law angle dependence* (16 assertions at 85°, `md/m10_m15_report.md` 0.4); +6 lines.
- Measured (`md/m10_m15_report.md` 0.2): the waves between 78° and 90° carry 4.3 % of the diffuse weight but pass far
  more than the average (the mass term goes to zero at grazing; a plate with its coincidence below the band is at
  coincidence there): round 1's columns, fitted on the 0–78° average, pass up to 15.3 dB more per octave on the 0–90°
  average a room sees (concrete, brick 203, block, the linings; the glazing 3.8 to 7.1 dB). Refitted at 0–90° the 13
  constructions reach 1.78 dB today and 1.17 dB with a limit at 78° (2.97 → 2.80 dB under M10).
- Open (Stephan, 2026-10-17): not in this round; pending a first-principles study of grazing incidence (the
  infinite plate against the finite plate). No new open item.
- **Closed (Stephan, 2026-10-19): superseded by the finite panel (2.1).** The study (`md/grazing_report.md` 0) found a
  fixed limit right in one octave per size only (off by up to 14 dB at 31.5–125 Hz), and every clamp or soft limit
  4 to 34 dB off the finite plate per angle past 78°; Davy's forced radiation efficiency with a size column follows it
  within 0.8 dB per angle. Built with the column `side` (decision 2 of 10).
- Side finding, open (for the double-leaf rows of 3.4, not this round): the cavity spring of `res*` with `resS > 0`
  is formed at the cosine in air, so the mass-air-mass dip sweeps up as `f₀/cos θ` and through the band at grazing. A
  cavity partitioned by battens, studs or a pane's spacer is locally reacting and does not sweep (a honeycomb partition
  removes the angle shift of a panel absorber [Toyoda_2011_HoneycombPanelAbsorbers, Fig. 2]). With the spring at normal
  incidence and the finite panel the double units and linings fit to 0.04 to 1.12 dB (4-16-4 0.04, board on dabs 0.16,
  furred lining 1.12 with its `coiF` off the 30 Hz bound; worst of the 13 2.53 dB; `md/grazing_report.md` 0.6).

**M15 — the loss of the bending term as a resistance** (modification, proposal).
- What: `TL` from `|1 + X₀·c·(j·(1 − w) + w/coiQ)|²` in place of the reactance modulus.
- Basis: [Davy_2009_SingleLeafCremer, Eq. 8]. Gain: up to 3 dB at coincidence where `X₀·c/coiQ ≈ 1` (derived
  here); on the *Coincidence* pane the mass term 7.88 → 10.22 dB at 59.8°, 4 kHz, T 0.70 dB lower (the plate rule
  holds T near `a`). Cost: +5 lines, a complex `X₀` through nine declarations; moves only *Coincidence - …*.
- Measured (`md/m10_m15_report.md` 0.2): neutral on the 13 fits (≤ 0.02 dB): a fitted `coiQ` absorbs it.
- Proposal (Stephan, 2026-10-17): to be decided when the parameter session decides where `coiQ` comes from (a
  measured loss factor needs M15; a fitted one does not).

**M16 — a relaxation term on `μ` for fabrics** (addition, proposal).
- What: a Debye term on the permeability, `μ = μ_∞ + Δμ/(1 + j·f/f_t)`, as two columns (or a pole form on `μ`, as
  `res*` is on `ε`).
- Basis: a fabric is its airflow resistance `R_s` in parallel with its mass impedance `jωm`
  [Pieren_2015_MultilayerCurtains, Sec. 2]; as a fluid leaf of thickness `h` that is
  `μ(f) = (m/(ρ₀h)) / (1 + jωm/R_s)`, a relaxation at `f_t = R_s/(2πm)` (derived here).
- Rows: 3.2 curtains. Gain: the fluid leaf with the exact relaxation is within 0.003 (normal incidence) and 0.015
  (random incidence) in α of the parallel form for seven fabrics of [DelRey_2013_CurtainFullness] at 145 mm; the
  power-law columns hold 0.05 only on the random-incidence average they are fitted on (0.008 to 0.039) and miss single
  angles by up to 0.35 where `f_t` lies in the band (179 to 322 Hz for five of the seven) (M9 phase 0, item 6).
- Cost: two columns read in `Material`'s permeability; no state.
- Risk: it is a wave-medium column (the leaf has `m = 0`), so a change on the wave side, checked against `[wave]`; no
  wave case carries it.
- Acceptance: the seven fabrics per angle against the parallel form, through the walker on a box; a G-100 hanging.
- Priority: low (a fitted `tf` reaches a measured hanging; class 9). Proposal only (Stephan, 2026-10-15).

**Considered, not proposed:** per-band tabulated columns in place of power laws (a measured-data record
[review 7.2]; the power laws reach the porous reference within ±0.032, report 3.4); a physical parameter set
(`m″`, `d`, `R`) for the peak (the same count as `tfP`, `tfF`, `tfQ`; a rule for deriving them); tying `tfF`
to `resF` (a pane's mode and a lining's cavity are different resonances in one material); scattering and edge
diffraction (P9).

## 5. What stays out of reach

Even with every candidate. "In scope" items are recorded limitations of the machine; "out of scope" items are
not the material model's to carry.

| what | scope | why |
| --- | --- | --- |
| scattering, diffusion, edge diffraction of small surfaces, furniture as volume scatterers | out (P9) | needs new directions or rays per event, a change to the beam-tracing engine beyond acoustics. The evidence for it: reverberation time is sensitive to `s` where absorption is concentrated on a few surfaces (T30 settled from `s` about 0.7) [Pereira_2025_ScatteringSensitivity, Sec. 4]; near a scattering wall (within about 2 m) an equivalent `s` changed EDT and C80 audibly less than the explicit geometry [Shtrepi_2019_DiffusiveModelingDetail, Table 1]; difference limens of `s` 0.27 to 0.7 [Vitale_2015_ScatteringPerception, Tables 6.2-6.3] |
| flanking | out | a junction property, `K_ij` per path [Mahn_2023_GuideAirborneRR331]; out of reach of a sheet (P7) |
| the ground wave `F(w)` | out | needs the source-receiver geometry at the event [Fuchs_2025_GroundImpedanceNTACOU104]; outdoor, not the reflect side of a room |
| level dependence of MPPs and orifices | out | the tracer is linear and knows no level at an event; near walls at 31.5 to 63 Hz listening levels can approach the MPP limit [review 4.2] |
| a panel's size (first mode, finite radiation) as a column | in, as a row | the low octaves of a panel depend on its size (report 3.1; a 1.25 m × 1.5 m pane has its first mode at 10.7 Hz for 4 mm and 16.0 Hz for 6 mm); a material row per size class carries it (A11) |
| a double leaf by stud type, bridging | in, as a fit | no single model fits all wall types [Hongisto_2006_DoubleWallsOverview]; one lumped sheet per measured construction (P1) |
| bending stiffness of a wave-medium leaf (M9) | in | a fluid layer has none; coincidence needs the mass column |
| a porous layer directly behind a leaf with air behind it (M9) | in | two coupled bodies with no finite path sum; a fill at the back of the box is the form (2.8) |
| a corner bass trap as an acceptance class | out of the expressibility test | its data are m² per object in a chamber corner, 1.3 to 1.7 m² at 125 Hz [GIK_2013_244BassTrapRAL; GIK_2022_TurboTrapRAL], with no standardized method for spaced objects [GIK_2022_TurboTrapRAL, p. 4]: an object in a placement (3.7, 3.9), for a later room-level check |
| curtain fullness | in, as a fit | no model [DelRey_2013_CurtainFullness]; a `tf` per fullness |
| absorption values of a placement in a modal field (above one in corners, at one mode) | out as a material property | they are field results [review 6.2]; the coherent field forms them from geometry |
| time-domain coupling of a resonator and a mode beyond its complex factor | in | a frequency-domain tracer carries the factor (M6) and the reflection orders it traces; nothing else |
| the state one level deep | in | section 7; the per-ray state cannot grow |

## 6. Open items

**Decided, still open**

| id | statement | size | where it goes |
| --- | --- | --- | --- |
| A8 | A deferred medium displaces the thickness code: a nested or overlapping object met on a leg that carries a code replaces it, and the leg pays its own term in full | +37.50 dB (a solid on a wall with an object nested in it), +23.10 dB (a lining on a solid overlapping a second one), from one side only; +25.8 / +11.6 dB at 75° | low priority; a freed bit could flag "an object pending"; may be dropped |
| A9 | `html_docu/quadriga_lib_material_model.md` is out of date (the three `tf` rules, the closed layer, the held wave, the mass cosine, the plate rule, the state layout) | a document | after the machine is settled |
| A12 | The reflection phase. Closed for a hard wall and a porous layer by N7 (3.8), and for a locally reacting resonant face of every type by M6 (built 2026-10-16, 2.6: the face's series impedance, TE and the scalar types `ζ·c`, TM `ζ/c`). What remains: a flat `tf` (no peak) keeps the hard wall's phase and is flat in angle (section 7); a panel over an unpartitioned cavity takes the M9 route (2.8) | ±18° through a panel absorber's resonance (model, report 3.5); ±4.12° on the board of 2.6 over-damped, through 180° under-damped; level errors up to 2.4 dB without phase [Jeong_2012_BoundaryConditionsPhasedGA] | closed for locally reacting resonant faces (M6); the flat pad as stated |
| A13 | The coefficient type of a `tf` reference. `tf` is a random-incidence coefficient (3.1). Round 1's absorption rows were compilation values of unstated type (report 1.2), its averages field incidence 0° to 78° (report 1.1). Sabine values above one, edge effects and placement values are no `tf` | `α_s` above `α_st` by 0.1 to 0.35 for porous layers at mid frequencies [Dupont_2023_SoundFieldSynthesis]; by 0.12 on average for 50 mm mineral wool [Jeong_2013_ReflectionAbsorptionCoefficients, Sec. 4-5] | analyzed (2026-10-15, `ac2/a13.py`, `a13.out`) on three materials with an independent random-incidence reference (glass wool and PU foam 50 mm, chamber against sound-field synthesis [Dupont_2023_SoundFieldSynthesis]; rock wool 50 mm, chamber against tube [Kosten_1960_InternationalComparison]), every band with a reference: (a) the chamber value capped at one, max 0.273, rms 0.139, mean +0.109; (c) `β·E` [Kuhl_1960_EdgeEffect; review 7.2], max 0.112 (the foam; 0.053 on the two wools), rms 0.041; (b) Jeong's conversion [Jeong_2013_ReflectionAbsorptionCoefficients] with Thomasson's field impedance of a 3 × 4 m sample [Thomasson_1980_AbsorptionCoefficient, Eq. 26] and one Miki σ per product, extended reaction, max 0.057, rms 0.031, mean −0.010 (local reaction rms 0.039; without the finite-size term rms 0.049 and σ two to three times too high). Decided 2026-10-15 (section 10): a tiered rule. Sample size checked (`ac2/a13b.out`): Kosten's rock wool as 4, 8 and 12 m² (shapes taken as 2 × 2, 2 × 4, 3 × 4 m) converts to one curve, spread at most 0.010, within 0.039 of the tube; `β·E` with each sample's edge per area is up to 0.083 off at 4 m². Found with A14: the conversion holds only where the model family fits the product and the sample is mounted as the model assumes (a flush patch); where σ is measured, the fitted σ is the check (PU foam 10.3 against 10.6 kPa·s/m², glass wool 19.0 against 13.1, polyester 65.5 against 15.0, framed stone wool 51.7 against 30.3). Proposal: add that check to the rule; a conversion whose σ is off by more than about 1.5 times is marked, not used. Since M6 a peaked face is locally reacting, so its field (0°–78°) and Paris (0°–90°) values differ (by up to 0.017 per octave on the six refitted rows, and a Paris fit misses its targets by up to 0.062 in the field value, `md/m6_report.md` 0.6): the coefficient type of a peaked row's reference matters, and its target is the Paris integral (9) |
| A14 | The class-4 reference. The Knauf TP 120 A catalog values (`α_s` 0.11 / 0.26 / 0.52 / 0.70 / 0.84 / 0.95) are not a 20 mm layer of its declared 11 kPa·s/m² on a rigid backing: as a finite 3 × 4 m sample that layer reads 0.43 / 0.58 / 0.71 at 500 Hz / 1 kHz / 2 kHz, and (b) needs σ ≈ 85 kPa·s/m² to fit (mounting not stated, report 1.5) | `ac2/a13.out` | the machine reaches every random-incidence target within 0.003 to 0.010 in α (Paris 0°–90°; the glass wool's sound-field-synthesis curve within 0.010), confirmed through the walker within 0.0013 per angle (`ac2/a14.out`, `a14w.out`); the capped chamber values of the glass wool (about 1.0) stay 0.077 out of reach. Decided 2026-10-15 (section 10): class 4 on the glass wool 50 mm of [Dupont_2023_SoundFieldSynthesis] (its random-incidence curve measured, no conversion). The two Scrosati rows (`ac2/a13b.out`) do not convert: the polyester mat 17.8 mm (unframed) fits only with σ 65.5 kPa·s/m² against the measured 15.0, the Miki model at the measured σ missing its chamber values by up to 0.13 (Miki is fitted to mineral wool; polyester follows other regressions [Hurrell_2020_PolymericFibres, Sec. 4.3]); the stone wool 105 mm sat in an 18 mm plywood frame, which the flush-patch model does not describe (σ 51.7 against 30.3, the model at the measured σ up to 0.22 off). Proposal: both rows out until a polyester model and a framed-sample model exist |
| A19 | The membrane and cavity constant for `tfF` and `resF`: 60, about 50 with fill, about 80 in Sharp's model; edge clamping raises `f₀` 1.2 to 1.5 times | [Oldfield_2006_ImprovedMembraneAbsorbers; Shearer_2016_AbsorptionBelow100Hz; review 7.2, 9.10] | analyzed (2026-10-15, `ac2/a19.py`, `a19.out`) on 10 measured resonances with their construction ([Shearer_2016_AbsorptionBelow100Hz; Sakagami_2007_NonUniformCavity; Oldfield_2006_ImprovedMembraneAbsorbers; Nell_2019_TieffrequenteAbsorption]; the design values of [Kristl_2023_Verbundplattenresonatoren, Table 4.1] are computed with 60 and left out). Free edges (a loudspeaker surround, a free-floating plate; n = 3): measured over 60/50 (60 empty, 50 filled) 0.81 to 1.10, geometric mean 0.99; over 60 always 0.82. Framed and clamped edges (n = 7): over 60/50 1.06 to 1.84, mean 1.34. With 60/50 and a factor 1.34 on framed edges all 10 within ×/1.19 rms, worst 0.80 and 1.38. Decided 2026-10-15 (section 10). **Revised 2026-10-15 after M9 phase 0 (section 10):** the framed factor 1.34 was mostly the `1/cos θ` angle shift of an unpartitioned cavity under diffuse incidence. A leaf on a box of its full depth (2.8), at random incidence, reads 1.11 / 1.07 / 0.80 / 0.99 against the framed chamber peaks (Shearer empty; Sakagami 30 / 90 / 150 mm; geometric mean 0.98) and 1.03 against Shearer filled; a box shallower by 1/1.34² reads 1.05 to 1.49 (mean 1.30). Rule: the measured `f₀` first; else 60 empty, 50 filled; edge stiffening noted only for small clamped panels (Oldfield, normal incidence, 1.24 / 1.54) |
| A20 | A wall's `tf` behind a porous fill. The library forms the pad on the immittance of the medium in front (2.3: next to a lossy medium the flux the pad hands over), so the wall absorbs according to what lies against it; the alternative is an absorption fixed as seen from air; physically a wall has a fixed surface impedance | the board on 25 mm of fill (Miki, 25.1 kPa·s/m²) on a wall with `tf` 0.02: the library within 4.3e-5 of the pad on the modulus of the fill's impedance, 0.040 in R off the pad referred to air (M9 phase 0, F1) | closed for the peak by M6 (2.6: a peaked face is a fixed series impedance met against whatever lies in front, the transfer matrix behind a lining); open for the power-law part of `tf`, still a pad matched to the medium in front; the fixtures use `tf` 0 behind a fill (section 10) |
| A21 | A sheet co-located with the exit face of one body and the entry face of the next (body \| sheet \| body). The intersector reports two of the three faces, by mesh order (a tie): box exit + sheet → a glued exit into air, the wave left "in air" inside the far body; box exit + far entry → body \| body, the sheet ignored; sheet + far entry → the glued entry from the host, exact (the fluid layers to 6 digits). **Narrowed 2026-10-16** (`md/m5_a21_report.md`): a ray a glued exit left in air that meets a body from inside is killed, every type (`md/wave.md` 6, 8); energy creation and the `ray_progress` runaway are closed | before: R up to 42.19 (scalar and EM TE) and 14.46 (EM TM), R + T 1.075 with a mass-law body in front; `ray_progress` at threshold 1.0 ran away (12.7 GB in 300 s, leaf on an `m = 0` wall). After: R + T ≤ 0.902 over 8 variants, six orders, scalar and EM. Left: the reflect port of box \| sheet \| air (R 0.154 for 0.994 at 125 Hz, 0°) or the sheet ignored (R 1.000 for 0.745 at 1 kHz), by mesh order; a sheet glued on the far body's back face is met as air \| sheet \| air (code 69, not killed, R + T ≤ 0.48, nothing created) | open: the order dependence needs the intersector or `ray_progress` to report the third face (outside `ray_mesh_interact.cpp`); open: how often a glued exit is followed by a face met from inside in a real scene (the repository scenes have no glued sheets; their isolated sheets never meet the condition). Rule of use: a porous layer between a box and a wall is a traced body (2.8) |

**Proposed (for a decision)**

| id | statement | size and source | proposal |
| --- | --- | --- | --- |
| A17′ | Class 8 as the family of resonant absorbers. All lumped resonant absorbers are one series circuit (3.2) and form one continuum from panel to microperforated absorbers [Sakagami_2009_MPPPanelRelationship]; they differ in where the mass sits: the leaf (panel, membrane, composite plate, distributed-mode panel), the air in the openings of a perforated or slotted panel (a Helmholtz array; perforated gypsum ceilings, slats), or the air in sub-millimeter holes (MPP) | perforated gypsum ceilings 0.30 to 0.40 at 125 Hz and 0.75 to 0.90 at 500 Hz [Rigips_2026_RigitoneActivAir618R; Rigips_2026_RigitoneActivAir818R]; the lumped Helmholtz formula holds for cavities below about λ/16 [Randeberg_2000_PerforatedPanelAbsorbers, Sec. 2.1.1]; below 250 Hz a slat absorber acts through its porous layer only [Riionheimo_2018_SlatStructures, Sec. 4-5] | class 8 becomes "resonant absorbers" with rows 8a panel and membrane, 8b perforated and slotted panels, 8c microperforated panels (class 10 moves into it); a single Helmholtz resonator is an object, as a corner trap (section 5) |

## 7. Accepted limitations (acoustic, until the machine changes)

- The mass term steps at an invariant of one (7.66 dB at 0.99, 22.60 dB at 1.01 behind `eps 1.50 − 0.40j` for a
  26.8 dB sheet); never reached by a ray from air through a face parallel to the plate.
- A ray the state does not know to be bent (a source inside a host, a caller without state) keeps its
  segment's cosine in a mass term: 0.21 / 0.96 / 3.00 dB at 10° / 20° / 30° in `eps 2.50`.
- Behind a lossy wedge the cosine in air is the real-angle one (0.002 to 0.096 dB).
- The thickness code is quantized (0.044 % per step, 0.0038 dB; ≤ 3.0e-3 dB three and four bodies deep as measured
  before M10) and ends at 100 m (an equivalent thickness of a heavy body in a light material can reach it).
- Closed (M10, 2026-10-17): mixed slopes no longer exist, so the fold from one side is gone. The former mixed stacks
  of *Bonded mass - the plate at any depth* read as their one-slope twins: `A20 100 | B30 50 | C20 100` light +7.66
  / +6.78 dB (as named / from the other side) → +1.32 / +0.0045 dB, `A20 100 | B20 100 | C30 50` heavy from the other
  side +4.30 → +0.0070 dB; what remains (+1.77 dB on a three-body stack as named) is the one-level entry face of the
  next item (`md/m10_m15_report.md` 0.4).
- The state is one level deep. A traced plate re-forms its first face's absorption at the exit from the
  previous-medium word, which names only the body before: a plate of three or more bodies pays the rule within
  +3.02 / −3.01 dB on one slope (*plate depth* S: a sheet inside a wall between two bodies; plaster \| brick \|
  plaster), and a wall with an ordered stack inside it is up to 3.01 dB from its traced twin (*Held lining* E:
  K \| T W 3.01, K \| W T 1.76, T \| K W 1.25 dB). The per-ray state cannot grow (section 2.2's two words;
  `md/wave.md` 6).
- The plate near its knee can exceed the dB sum (+0.44 dB at 70° for `X₀` 6 and 2.4).
- A plate passes at most what its face absorbs: below its knee a light panel passes about `a`, not one; what it
  neither passes nor reflects (`a − T`) is absorbed.
- The first face of a leak is flat: through the plate rule 0.00013–0.00094 dB between the two directions of a
  wood panel on concrete without its mass law (10°–70°). The step of a wave becoming a leak is gone (0.0000 dB
  at 0.0084°). A non-rigid mass-law layer becoming closed is a step (`eps 0.50`, 22 mm, `tf` 0.2, mass at
  nothing, inside `eps 2.50 − 0.8j`: −5.85 dB at 44.9°, −7.49 dB at 45.1°; −5.39 / −6.99 dB in the lossless
  host).
- A rigid face without `tf` absorbs nothing, so its plate passes nothing, from either side (rule of use: `tf`
  with `m`).
- A non-rigid mass-law layer beside a lossy coherent layer depends on the order: 0.107 / 0.127 / 0.201 dB
  between the two orders at 10° / 30° / 60° (*Closed plate* E), 0.016–0.024 dB in a stack.
- Lossy media split the flux (the wave convention, `md/wave.md`): a face next to a lossy medium can pass more
  than one in flux. The two sides of a lossy lining on a rigid wall differ, the wall's side higher: 0.07–1.38 dB
  (dense `eps 2.50 − 0.80j`) and 0.52–2.70 dB (porous `eps 0.31 − 0.20j`) at 10°–75°, scalar. A mass-law sheet
  without `tf` behind a lossy host, under a closed outer pair, takes the smaller face, one: −0.39 to −1.83 dB
  against the face's flux gain (*Mass cosine* D, invariant 1.01–1.45).
- An event can pass more than one. A plate's exit gives back the absorption of its first face, up to `1/a_in`
  (+8.85 dB on wood 50 mm \| concrete 50 mm); no path passes more than the rule. `ray_progress` stops a
  continuation below `min_gain_dB` (default −140 dB) on the running gain, which inside a plate is up to
  `−10·log10 a_in` below the level the path will reach (10 dB for `tf` 0.1, 20 dB for `tf` 0.01), so a path
  ending within that margin above the threshold can be cut. Not measured on a scene.
- The HH rows (type 1, k 3) of a mass-law material are a formal EM row and no use case: they part from the traced
  twin by up to 3.9 dB (porous lining) and the two sides by up to 7.5 dB.
- The level of a held chain has no reference but its solid twin; the crossings are formed from impedance
  mappings.
- `ray_mesh_interact`'s own mass-law lining is approximate in a lossy host (the redo with state is exact).
- A rigid face's absorption is flat in angle, except a peaked face's (2.6, 3.1).
- A followed bounce series behind a traced lining rarer than air can diverge on a peaked face: the one-parameter porous
  lining `eps 0.31 − 0.20j` (10 mm) on the peaked board wall, at 85° and 31.5 / 63 Hz, R + T 10.8 / 151 followed, where
  M5 converges. The face's reflection in the lining's frame exceeds one (1.04 to 1.09: the in-layer immittance is
  mostly reactive, −67° to −82°, the lossy-medium convention of `md/wave.md` 8); summed rows stay below one (0.99867)
  and are the transfer matrix. Behind two-parameter rock wool the candidate converges where M5's followed series
  diverged (4.0e14 at 125 Hz, 10°); `ray_progress` runs neither away nor above one there (`md/m6_report.md` 1.0, 2.4).
- At exactly the critical angle of a lossless lining rarer than air (eps 0.5, 45.000°), traced in front of a mass-law
  wall, R + T reads 1.0005 (1.0002 under M5) at that one angle and 1.000000 past it; the glued sheet stays below one.
- `ray_progress` with tracer 4 under-transmits out of a summed lossy lining rarer than air, traced in front of a held
  mass-law wall (one-parameter mapping only): on the peaked board at 85° it hands 0.00126 in power to the path through
  the wall against 0.0774 for tracer 5 and the walker (both tracers), so R + T is 0.065 / 0.044 low at 31.5 / 63 Hz;
  under M5 the same leg carries 2.6e-5 against 3.5e-3, invisible because M5's face passes little at 85°. The lining's
  series is summed (`ρ` 1.006 / 1.029); the cause is not traced (`md/m6_report.md` 2.4, S1).
- Wave media, the leaves of a meshed double wall included, keep the infinite plate at grazing: the finite panel
  (`side`) acts only where `m > 0` (2.1, 2.8). Their finite size is not modeled.
- **Closed (2026-10-20): a traced leg far from the origin** (`md/precision_report.md`). `ray_mesh_interact` relaunches
  a ray 8·2⁻²³ of the hit point's largest coordinate past it, about 1 µm per metre, and the leg inside a solid was
  measured from there: short by that much. A 4 mm traced pane at normal incidence passed 0.2 dB more 100 m from the
  origin (−31.454 against −31.658 dB at 500 Hz) and 2.4 dB more at 1 km. The ledger now counts the offset, so the leg is
  measured from the hit point: the same pane passes what it passes at the origin to 5.4e-11 dB up to 1 km (*Far from the
  origin - a traced mass-law wall passes what it passes at the origin, its leg measured from the hit point*). The 89.9°
  row of the same finding (a 152 mm wall missing its plate rule by up to 3.42 dB) was the fixture's: `r1::wall_g`'s first
  segment ended inside the wall and the walker added the preloaded leg again. The segment now reaches past the wall and
  *Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency* traces its solid to 89.9°.
- **Closed (2026-10-20): the false −26 dB of a float mass-law wall.** In single precision, on a short segment 10 m or
  more from the origin, rounding of the segment's ends made the 1e-6 test of `use_vbs` count a held leg as a bent one;
  the plate rule was skipped and the wall passed its two leaks, 26 dB too little (in 16 of 72 rows at 100 m; through
  `ray_progress` only on segments of a few metres). In a mass-law medium in float the test allows that rounding (*Far
  from the origin - in single precision a mass-law leg is not taken for a bent one*).
- **Rule of use:** build scenes in local coordinates near the origin: mm-wave through `ray_progress` within about 100 m;
  never raw georeferenced coordinates, where the nudge exceeds a wall's thickness. **Limits** (`md/wave.md` 8): a body
  thinner than the nudge is skipped; rays within about 3° of grazing can be lost in float or with the AVX2 kernel; the
  float position rounding leaves an EM phase of about 3.8° at 28 GHz at 1 km. Recorded, no item: a 4 mm pane on large
  turned faces in float is lost or runs away even at the origin.
- A peak set on a non-rigid mass-law body makes its face the series impedance: `eps 1.50`, `m` 20 with the board's peak
  reflects 0.844 at 150 Hz, 0°, where its Fresnel face with the flat pad read 0.085. No use case.

## 8. Side findings

- Closed (2026-10-17): `src/quadriga_lib_material_helpers.hpp`, a second `medium_gain_impl` / `medium_loss_dB` and
  `eta_resonance` with its own `coiA` copy that nothing included, is deleted (with D2-b).
- `quadriga_lib::medium_gain` carries an `ISSUE` comment in `src/ray_mesh_interact.cpp`: for a mass-law
  medium it needs two distances (the refracted path and the geometric path), and its caller
  `calc_diffraction_gain` passes one. The diffraction gain of a path through a mass-law body takes the mass
  term at `c = 1`.
- The fitted knobs of round 1 carry non-physical values on heavy walls and linings (`m` 1.64 to 35.13, `coiF`
  49 to 9933 Hz, `p3/tl/rmb_chk.out`): they shape what finite size, coupling loss and modes do (report 3.1, 3.6).
  Round 1's 2.51 dB came from an unconverged fit (16-start Nelder-Mead); refitted with differential evolution and a
  selection on the 768 × 12 evaluation grid, today's knobs reach 1.23 dB at 0–78° (`md/m10_m15_report.md` 0.1, 0.2;
  round 1's `conf/cols.tsv` holds its earlier unbounded fits). Whichever knob is removed, another non-physical one
  takes over its job (0.2): the slope (today, M11), a flat `att` with a level far below the mass (M10, built), a broad
  coincidence at 3.7 to 18 kHz with `coiQ` about 1. Under M10 at 0–90° the light panels fit with their mass and
  coincidence and the heavy walls and linings sit at the knob bounds (1.0, 2.7). M13 and M15 change none of it, nor
  does the finite panel (2.1, closing M14): the heavy walls are above coincidence from 125 Hz, where grazing carries
  nothing; their knobs stand for the plate physics above coincidence (`md/grazing_report.md` 0.6).
- The test-case name *Unentered branches - a sheet without the mass column of its mass-law host's permittivity
  is merged* states the rule of 2.2: a matched sheet merges unless both carry the mass column. The previous
  text of 2.2 said a sheet with its host's permittivity is not merged into a mass-law host, which holds for a
  mass-law sheet only.
- The review cites four keys that its `references/index.csv` does not list: `PTB_2012_AbsorptionTable`,
  `PTB_2002_RR3Data`, `Scheibler_2018_Pyroomacoustics` (data sets; their rows are in its
  `data/material_properties.csv`) and `Nell_TIP_TieftonAbsorber`. Sections 3 and 9 cite the first and the last.

## 9. How the machine is accepted: the expressibility test

The machine is accepted when every reference construction can be reached by some setting of the columns, as
sheet and as traced solid, within the guideline tolerance in every band. The stack will be calibrated against
measurements later; this test shows that the knobs exist.

- **Reference:** measured data where they exist (P1); physical models only where no measurement covers a band
  or to fill in a construction's shape. The type of each reference is stated (below), and a Sabine reference is
  converted or marked (A13).
- **Bands:** octaves from 31.5 Hz to 8 kHz (the mapping's range is 30 Hz to 8 kHz), each on enough single
  frequencies to resolve a peak or a dip (round 1: 6 to 12 per third, report 3.6); resonances and coincidence
  dips are narrower than an octave [Quirt_1982_WindowsI, Sec. II.C]. No open source gives an independently
  measured random-incidence absorption coefficient at 31.5 Hz for any common material, nor a laboratory R at
  31.5 Hz [review 6.1, 6.3]: those bands take the model spread (P7).
- **Tolerance, a guideline and no hard bound (P2), read in two tiers (A16):** transmission loss within about
  ±3 dB, absorption within about ±0.05, coincidence bands reported apart; bands without a measurement the model
  spread (P7). A band outside the guideline but within the reproducibility of its reference is reported "within
  reproducibility" and does not fail: in α the `s_R` of the reference's type at its level and band (below); in
  TL at 63 Hz and below the larger of the model spread and the difference between laboratories.
- **What the guideline stands against:**

  | band | ISO 354 `s_R` at `α_s` = 0.1 / 1 [Wittstock_2018_InterlabAbsorptionPart2, Table II] | pooled `SD_R`, 13 to 17 labs [Rosario_2025_ProficiencyISO354, Table 1] |
  | --- | --- | --- |
  | 63 Hz | 0.06 / 0.465 (fit on 7 results) | — |
  | 100 Hz | 0.039 / 0.255 | 0.023 |
  | 125 Hz | 0.033 / 0.195 | — |
  | 250 Hz | 0.024 / 0.105 | 0.040 |
  | 500 Hz | 0.020 / 0.065 | 0.050 |
  | 1 kHz | 0.019 / 0.055 | 0.062 |
  | 4 kHz | 0.043 / 0.070 | 0.070 |

  Impedance tube `α₀`: `SD_R` 0.03 to 0.07 [Pompoli_2017_Reproducible, Table VIII]; converted to random
  incidence the chamber spread falls to 0.008 to 0.040 [Jeong_2015_ReproducibilityConverted]. Laboratory R: ±1 dB
  from 315 Hz, 2 dB at 200 to 250 Hz, 3 dB below (95 %) [Quirt_1982_WindowsI]; 3 to 7 dB per octave between
  nominally identical lightweight walls [Halliwell_1998_GypsumBoardWalls]; 7 to 10 dB between two laboratories at
  50 and 63 Hz [Prato_2015_LowFreqModalApproach, Table 5.14]; small specimens read about 5 dB high below `f_c`
  [Wareing_2015_SampleSizeVariations], and a reduced test opening 10 to 20 dB above the infinite-plate mass law at
  63 and 125 Hz [Kosala_2020_SingleBafflesAccuracy].
- **Classes** (P3) and their reference sources (coefficient type in brackets; R is laboratory R, ISO 10140 or
  ASTM E90):
  1. light single panels: glass 4 and 6 mm, gypsum board, a wood panel or a door. R: EN 12758 and AGC (report
     1.2), [Quirt_1981_WindowsTLMeasurements], [AGC_2023_PocketStratophone], a single 12.7 mm gypsum board
     [Northwood_1968_PlasterboardWalls], thin steel, aluminum and PMMA from a reduced opening
     [Kosala_2020_SingleBafflesAccuracy]. Absorption: [PTB_2012_AbsorptionTable] rows 207-209, 339-343, 396
     (compilation, Sabine, type and mounting mostly missing); BRAS window fits 0.13 to 0.19 at 125 Hz
     [Aspoeck_2020_BRAS] (fitted to a room).
  2. heavy walls: 100 to 200 mm concrete or masonry. R: concrete 150 and 200 mm [Mahn_2023_GuideAirborneRR331];
     hollow block 190 mm with 50 to 80 Hz [Zeitler_2015_ConcreteBlockRR334]; the round-1 walls (report 1.3).
     Absorption: [PTB_2012_AbsorptionTable] rows 30, 31, 36, 345, 346 (compilation); concrete room surfaces 0.006
     to 0.008 at 31.5 to 125 Hz from measured T30 [Nell_TIP_TieftonAbsorber, Table 6] (derived in the review).
  3. a double or triple glazing unit, lumped as one sheet. R: [AGC_2023_PocketStratophone], [Quirt_1982_WindowsI].
     Absorption: [PTB_2012_AbsorptionTable] rows 207-209 (compilation).
  4. a porous absorber on a rigid wall (absorption). [Isover_2021_AkusticSSP2], [BASF_2021_BasotectGplus],
     [GIK_2020_B4TestResults] (practical or Sabine); the 200 mm stone-wool reference absorber, interlaboratory mean
     0.67 / 0.72 at 100 / 125 Hz in 8 rooms [Scrosati_2020_InterLaboratoryTest, Table 5] (Sabine); 100 mm mineral
     wool about 0.35 at 76 Hz by the modal method [Curovic_2024_ModalAbsorptionSmallRoom, Table 2]; flow
     resistivities of 24 glass and rock wools [Komatsu_2008_ImprovementDelanyBazley, Table 3]; the reference (A14,
     decided): glass wool 50 mm, σ 13.1 kPa·s/m² and its JCA set, chamber and sound-field synthesis
     (random incidence, measured) [Dupont_2023_SoundFieldSynthesis]. A carpet on a
     rigid floor is a row of this class: [BCTC_2014_EcosoftCarpetTile; Salford_2020_CarpetTileSpecimen] (Sabine);
     [Broda_2025_TuftedCarpets] (normal incidence).
  5. a lining on a rigid wall. R: the round-1 BIA walls (report 1.6); linings on masonry
     [Zeitler_2015_ConcreteBlockRR334, Table A1.2]. Absorption: [PTB_2012_AbsorptionTable] rows 339-343, 400, 401
     (compilation); gypsum cavity walls from 80 Hz [Bradley_1997_GypsumCavityAbsorption] (model fitted to
     measurement).
  6. a layered wall, plaster | brick | plaster: the round-1 references (report 1.7; no spectrum plastered on both
     faces was found).
  7. air | water, reflection first, transmission second: the fluid-fluid reference with sourced constants (report
     1.8: `c` = 1482.36 m/s, `Z` = 1.4797e6 rayl at 20 °C); exact in the two-parameter mapping.
- **Classes for the reflect side** (A17, decided 2026-10-15; class 8 as a family is proposed, A17′):
  8. a panel absorber (leaf over a cavity): 3 mm plywood over 30 / 90 / 150 mm [Sakagami_2007_NonUniformCavity]
     (random incidence, measured); 6 mm MDF over 95 mm, empty and filled, 40 to 315 Hz
     [Shearer_2016_AbsorptionBelow100Hz] (Sabine, below 100 Hz modal); a wall of plywood-fronted boxes
     [Bork_2005_RR3PartI]; a membrane trap [Primacoustic_2008_FullTrapRAL]; composite plate resonators
     [Fuchs_2005_LowFrequencyDomain] (Sabine, sample size not stated); a free-floating steel plate at 31.5 to
     125 Hz [Nell_TIP_TieftonAbsorber]. Tests M5, M6, M7, M9.
  9. a curtain, after the A18 measurement: G-100 reports [ZimmerRohde_2020_KakaduCurtain;
     ZimmerRohde_2022_Fabric10936Curtain; ADO_2020_BreakingDawnCurtain; ADO_2018_CannyCurtain] (practical); light
     curtains by fullness and wall distance [DelRey_2013_CurtainFullness]; the model
     [Pieren_2015_MultilayerCurtains] (normal incidence). Tests M9, M16 and the fitted `tf`.
  10. a microperforated panel or foil over a cavity: [Fuchs_2006_MPPReview] (Sabine, measured); Maa's model
     [Maa_1987_MPPWideband]. Tests M5, M7, M9.

  A corner bass trap is no class (section 5).
- **Procedure:** for each class, fit the knobs on a formula copy of the library (the window's sits on the walker
  to 0.011 dB); confirm the best fit through the walker as sheet and as traced solid, and through `ray_progress`
  as a band. Report the residual per band, today and per candidate of section 4. An absorption reference is
  compared with the Paris integral over 0° to 90° of the face (A13). A peaked face is locally reacting (2.6): its
  Paris value is not its `α₀` (`tools/acoustic/paris_cf.py`). By the mass relation `1 + R′ = ωm″/(Q_α·Z₀)` the
  default branch (over-damped) fits four of the six peaked rows of report 3.8 physically; double glazing needs
  `tfQ < 0` (16.1 kg/m² under-damped against 182 kg/m² over-damped, at the same residual 0.0075;
  `md/m6_report.md` 0.6).
- **A transmission reference is fitted on the library's own 0–90° average** (random incidence, weight `cos θ sin θ`),
  not on the 0–78° field-incidence average (decided 2026-10-17). A ray tracer meets a wall at every angle; the waves
  between 78° and 90° carry 4.3 % of the diffuse weight but pass far more than the average. Round 1's columns
  (report 3.6), fitted at 0–78°, pass up to 15.3 dB more per octave at 0–90° and their worst residual grows from
  1.23 to 15.8 dB (furred lining; concrete 9.6, brick 203 10.3); refitted at 0–90° the 13 constructions reach 1.78 dB
  today and 2.97 dB under M10 (`md/m10_m15_report.md` 0.2, 1.0). Fit at 192 angles × 6 frequencies per third or
  coarser, but select and report at 768 × 12: a polish or a selection on a coarser grid picks knobs that alias it
  (report 0.2: 5.71 against 3.33 dB). The formula copy and the fitter are `tools/acoustic/txlib.py` and `txfit.py`.
  **Since 2026-10-19 each construction is fitted at its laboratory side** (`side` = √(lab area): 1.349 m for the panes
  and units, 1.697 m for the boards, 3.523 m for the walls and linings of the Riverbank opening), the size its reference
  was measured at; `txfit.py --variant` takes `side` as `"lab"`, a number (one nominal side for all) or 0 (the infinite
  plate). At the lab side the 13 reach 2.70 dB, at 3.29 m for all 2.70 dB (`md/grazing_report.md` 1.5).
- **Invariants per candidate:** both ports at most one, sheet on solid at the relaunch bound, no step in angle, a
  band on its single frequencies, the `[wave]` baseline and twins untouched, line counts.

## 10. Decisions

**Taken (Stephan, 2026-10-14):**

- P1. **The reference is measured data.** Pragmatic: the whole stack will be calibrated against measurements
  later. Where theory and measurement differ, the knobs have to reach the measurement (e.g. B4m over B4 for
  double glazing above its resonance).
- P2. **Octaves, including the low ones** (31.5 Hz up). The tolerances are guidelines, not hard bounds.
- P3. **The reference classes of section 9, with air | water added.**
- P4. **A peaked room-side absorption on a rigid face needs analysis:** item A10.
- P5. **The settled acoustic conventions are open:** the two leaks of a closed plate, the bounce inside a
  mass-law body, the two ports of a face seen from inside a mass-law body. M1 and M2 are candidates for a change,
  measured like the others.

**Taken (Stephan, 2026-10-15), after round 1 step 0 (report 3.1, 3.2):**

- P6. **The reflect side first.** The goal is in-room wavefields (speaker placement, room treatment): room-side
  absorption per octave, the peaked absorption (A10) and the reflection phase (A12, with N7) come before
  transmission, which has looser requirements (sound insulation across rooms).
- P7. **Bands without a measurement take the model spread as their tolerance** (report 3.1: heavy walls
  4.5–13.9 dB, light panels 5.2–13.5 dB at 31.5 / 63 Hz). Flanking through the building is out of reach of a
  sheet (section 5).
- P8. **The rigid-wall reflection phase is an open item** (A12), no longer an accepted limitation.

**Taken (Stephan, 2026-10-15), after round 1 phase 3 (report 3.10); fixtures first, then the source:**

- N1. **The plate rule is R1h**: a closed mass-law plate reflects `1 − a` and passes `a·τ/(a + τ − a·τ)`, `a` the
  smaller absorption of its two outer faces, `τ` its bodies' term (report 2.4, 3.6).
- N2. **M2 with N1**: nothing is reflected back into a held mass-law body; a traced plate passes in one path.
- N3. **`res*` on a mass-law material is the plate's resonance in X0 (B4s)**, not a permittivity pole: `resS > 0`
  the dip of a double unit or lining (capped at one), `resS < 0` the panel's first mode (stiffness below `resF`).
- N4. **M4a**: on a mass-law material `coiF` and `coiQ` are the bending stiffness of the plate in X0
  (`|1 − (f/coiF)² sin⁴θ (1 + j/coiQ)|`); M4b is dropped; rule of use: `coiA = 0` on a mass-law material
  (superseded 2026-10-17: `coiA` removed, D2-b).
- N5. Closed (measured; no change). Every traced entry into a mass-law solid has its second hit (4,204 entries,
  report 2.2); its only use was A10 (a), which was rejected.
- N6. A2: the exit fix and the held exit into a wave medium judged and bent by the invariant of air (with the
  bent flag); A3: `alpha` over the held path in `qd_mass_amp`; A4, A6: closed by N1; A5: keep the sum, rule of
  use one `att` per bonded run; A7: closed by N1 and the held exit, the non-rigid phase rows restated to the held
  reference.
- N7 / A12. **The two-parameter mapping** (`ε = n·z`, `μ = n/z`) for every acoustic material.
- A10. **A resonant peak on the absorption, three new columns** `tfP`, `tfF`, `tfQ`:
  `tf(f) = tfR·(f/fRef)^tfB + tfP / (1 + (tfQ·(f − tfF)/tfF)²)`, clamped as today. Of the candidates, (a) a sheet
  reflecting `1 − max(tf, τ)` put window glass at TL 6.7 / 8.9 dB at 125 / 250 Hz against EN 12758's 17 / 20
  and was rejected; (c) a wave medium with a pole was not measured (report 3.8).
- A11. Both forms of `res*` (N3); the low-octave reference is the model spread (P7).

**Taken (Stephan, 2026-10-15), on the outline of this document:**

- P9. **Scattering is out of scope.** It would change the beam-tracing engine beyond acoustics; the material
  model and the acoustic work stay in `src/ray_mesh_interact.cpp`. A separate project only on very strong
  evidence (section 5 records the evidence).

**Taken (Stephan, 2026-10-15), on the decision list of this document:**

- M5: with the resonator law, `tfQ` is `Q_α`.
- M6: over-damped by default, `tfQ < 0` selects under-damped; the mass relation `1 + R′ = ωm″/(Q_α·Z₀)` is the
  parameter session's rule for the branch (section 4).
- M9 / A18: measured first, the ordered stack before the meshed gap.
- A13, A14: analysis with a scratch rig against the options and the reported results before a rule is chosen
  (section 6).
- A15: porous columns are derived from Miki, or Komatsu for mineral wool, over the whole band; the Delany-Bazley
  bound no longer enters.
- A16: the tolerances are read in two tiers (section 9).
- A17: classes 8 panel absorber, 9 curtain (after A18), 10 microperforated panel; the carpet a row of class 4; no
  corner-trap class.
- A19: analysis against the measured resonances before a rule is chosen (section 6).

**Taken (Stephan, 2026-10-15), on the recommendations after the scratch analyses (section 6):**

- A13: a tiered rule for a `tf` reference. A hard face (α up to about 0.2) is used as measured (the chamber
  excess of a locally reacting face of normalized impedance 40 is 0.022 at 125 Hz and less above, `ac2/a13_hard.out`);
  an absorbing product of known construction is converted with Jeong's method (`ac2/a13.py`); mineral wool
  without construction data takes `β·E`; anything else is marked, usable for parameters, left out of the
  acceptance. Checked after the decision: one curve from Kosten's 4, 8 and 12 m² samples (section 6, A13); the
  conversion needs a model family that fits the product and a flush sample (proposed check: the fitted σ against a
  measured one).
- A14: class 4 on the glass wool 50 mm of [Dupont_2023_SoundFieldSynthesis]; the polyester mat 17.8 mm and the
  stone wool 105 mm of [Scrosati_2020_InterLaboratoryTest], converted by A13, as further rows; Knauf out. The two
  Scrosati rows did not convert (section 6, A14); proposed out for now.
- A19: the measured `f₀` where one exists; else 60 empty, 50 filled, ×1.34 for a framed or clamped leaf, ×/1.2.
  Under M9 a framed leaf's rise is to be tested as a box shallower by 1/1.34² (derived, not measured). Revised
  after M9 phase 0 (below).
- M9: the route for an unpartitioned construction with a known build; M7 dropped; M5 kept; M6 lower, for locally
  reacting peaks (section 4).

**Built (2026-10-15)** as decided above, fixtures first; the source change, the restated cases and every moved
table are in the report, section 5. `src/ray_mesh_interact.cpp` 5082 lines against 5016 at bb336dc (+230 /
−164); the OBJ / API column lists carry `tfP`, `tfF`, `tfQ` after `tfB` (report 5.1); the leak chain on the plate
rule (report 5.2). Commit 4720a92.

**Taken (Stephan, 2026-10-15), after M9 phase 0 (`md/m9_fixtures_report.md`):**

- The M9 fixtures go into a new file, `tests/catch2_tests/test_acoustic_leaf.cpp`, tagged `[acoustic]`. Fixtures 1
  (the leaf alone) and 3 (4|16|4) evaluate wave media only; they stay there as guards of the acoustic route
  (`md/wave.md` 0 says so).
- The rows of the phase-0 table with its bounds; one fill fixture on Shearer's filled build, summed, against the
  fluid layers with the same columns; behind a fill the wall takes `tf` 0 (the convention is open, A20).
- No curtain fixture; the fabric's relaxation is candidate M16 (proposal). A21 open (a library change).
- **A19 revised.** The framed factor 1.34 was mostly the `1/cos θ` angle shift of an unpartitioned cavity under
  diffuse incidence: a leaf on a box of its full depth reads 0.80 to 1.11 against the framed chamber peaks, a box
  shallower by 1/1.34² 1.05 to 1.49 (section 6). New rule: the measured `f₀` first; else 60 empty, 50 filled; edge
  stiffening is noted only for small clamped panels (Oldfield, normal incidence, 1.24 / 1.54). It replaces the A19
  rule above (×1.34 and the shallower box).

**Built (2026-10-15)**: the rule of use 2.8 and five cases in `test_acoustic_leaf.cpp`, every reference written in
the case from fluid layers or the limp mass law, one case through `ray_progress`; no library change. A18 is
closed. `[acoustic]` 36705 assertions in 51 cases, whole suite 1311364 in 1137; `[wave]` unchanged.

**Taken (Stephan, 2026-10-15 and 16), after M5 / A21 phase 0 to 2 (`md/m5_a21_report.md`):**

- M5: the fixture board's `tfQ` 0.75 (`Q_α`), the two cases restated, bounds and names unchanged.
- A21: the kill of a ray a glued exit left in air when it meets a body from inside, first gated to the scalar types,
  then for every type once EM was measured to create energy as well (R + T 42.19 TE / 14.46 TM); the order dependence
  recorded (section 6, `md/wave.md` 8). The fixture `[acoustic]` in `test_acoustic_leaf.cpp`; `lw::walk` collects air
  only outside the structure (no M9 row moved).

**Built (2026-10-16)**: M5 (the line in `Material::tf`) and the A21 kill in `ray_state_update`;
`src/ray_mesh_interact.cpp` 5089 lines against 5082. The two peak cases restated, the new case *Sheet between two
bodies - nothing is created and the wave left in air inside the far body is killed*, scalar types 4 and 5 and EM type
1 (TE and TM, the faces as dielectrics); `lw::walk` takes EM types on one channel. `[acoustic]` 36813 assertions in 52
cases, whole suite 1311472 in 1138; `[wave]` identical, the stored twins unchanged.

**Taken (Stephan, 2026-10-16), after M6 phase 0 to 2 (`md/m6_report.md`):**

- M6 as recommended: D1 `ζ` met against the incidence side's immittance (equal to `ζ·c` in air); D2 `ζ` replaces the
  bare face's reactance; D3 the power-law part is the existing pad in front of `ζ` (0° absorbs `tf + α₀ − tf·α₀`); D4
  the gate `m > 0 && tfP > 0 && tfF > 0` (a wave medium and a negative peak stay on M5), EM TE `ζ·c` and TM `ζ/c` with a
  `tm` flag in `closed_face` and `tf_face`; D5 `tfQ < 0` selects `1/R′`, the `require` on `tfQ` removed, `R′` from
  `min(tfP, 1)`.
- Fixtures: both peak cases restated (the first in both mappings, the plate rule with `a(θ)`), the phase case and the
  random-incidence case new; references from `ζ` in the tests; bounds 2 to 3 times the measured maxima, never below
  1e-12 (rounding-level maxima move with the compiler, FMA contraction and libm).
- The findings of the checks C1 and S1 go to section 7, no new item; `ray_state_update`'s description of `eps`
  (`ρ` is not bounded by one; a round trip at or above one is always summed) and `md/wave.md` 8 corrected; the peak
  columns in `tools/acoustic/band_probe.cpp`, and `tools/acoustic/paris_cf.py`.

**Built (2026-10-16)**: M6 in `Material` (`src/ray_mesh_interact.cpp` 5132 lines against 5089), the two peak cases
restated, two cases new (*Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85
deg, as a sheet and as a traced solid, on both branches*; *ray_progress - the random-incidence absorption of a peaked
lining is the Paris integral of its series impedance*). `[acoustic]` 39371 assertions in 54 cases, whole suite
1314030 in 1140; `[wave]` identical, the stored twins unchanged.

**Taken (Stephan, 2026-10-17), after the M10 to M15 phase 0 (`md/m10_m15_report.md`):**

- M10 adopted: `m > 0` switches the mass law on at 20 dB per decade; `MASS_FOLD` without exponent or exclusion. The
  restated cases keep their `m` 30 / 12 palettes and write their references at slope 20; "where the slopes agree"
  leaves the two plate-depth case names.
- M11 dropped (`att` stays; A5 stays a rule of use). M13 dropped. M15 a proposal, decided when the parameter session
  decides where `coiQ` comes from.
- The fit convention: a transmission reference is fitted on the library's 0–90° average (section 9).
- M14 not in this round: a candidate, open pending a first-principles study of grazing incidence (section 4).
- M12 replaced by D2-b: `coiA` removed altogether (no physical reason remained); a table that still carries it loads
  and the column is ignored. The three `[wave]` cases that tested it restated, the only wave change allowed.

**Built (2026-10-17)**: M10 and D2-b (`md/m10_m15_report.md` 1, 2): `src/ray_mesh_interact.cpp` 5116 lines against
5132; `coiA` out of the OBJ writer, the Python and MATLAB readers and `tools/acoustic/band_probe.cpp`;
`src/quadriga_lib_material_helpers.hpp` deleted. Six cases restated for M10 and the `ray_progress` plate-depth twin
renamed, one case new (*Mass law - m switches the limp mass law on*); three `[wave]` cases and the *Coincidence* case
restated for D2-b, the guard *Coincidence column - a table that carries coiA loads and the column changes nothing*
new in `[wave]` with its *- the mass-law part* in `[acoustic]`; the OBJ header checks in the C++, Python and MATLAB
tests. `[wave]` 165258 assertions in 180 cases (output identical apart from the count line, the stored twins
unchanged), `[acoustic]` 40050 in 56, whole suite 1314857 in 1143. The 13 fits of report 1.0 through the built
library: sheet on traced solid within 0.0018 dB, a band on its single frequencies bitwise, R + T at most 0.99324.

**Taken (Stephan, 2026-10-19), after the grazing study (`md/grazing_report.md` 0, 1):**

- (d) adopted: Davy's forced radiation efficiency `σ` in the mass term in place of `1/cos θ`, with the finite plate's
  gain `1/(σ cos θ)` and the transmission clipped at one per frequency, on the radiation cosine only; the plate's own
  impedance (bending at the true `sin θ`, the `res*` cavity spring at the true cosine) unchanged. Mass-law materials
  only (`m > 0`); wave media, the leaves of a meshed double wall included, keep the infinite plate (2.8, 7).
- The size is a column, `side` = √(panel area) in m (2a = `side`, S = `side²`), 3.29 m where absent (a 4.0 × 2.7 m wall),
  0 the infinite plate, < 0 refused, read only where `m > 0`; `same_as` compares it where `m > 0`.
- One side per bonded run, a rule of use with no code (2.2); a mixed run takes the side of the first body the ray meets.
- M14 closed, superseded (section 4). The locally reacting cavity of `res*` not in this round: a side finding for the
  double-leaf rows (section 4, M14 entry).
- The precision of a traced leg far from the origin or near grazing is a separate investigation (7; closed 2026-10-20,
  below).
- Transmission fits at each construction's laboratory side (section 9).

**Built (2026-10-19)**: the finite panel (`md/grazing_report.md` 2): `src/ray_mesh_interact.cpp` 5159 lines against
5116 (`MaterialCols::side` and its check, `Material::side`, `same_as`, `Material::davy_sigma`, `mass_TL` with the
frequency and the side, its three call sites; the public `medium_gain` text); `side` after `tfQ` in the OBJ writer
(default 3.29), the Python and MATLAB readers and `tools/acoustic/band_probe.cpp`; the slab-threshold comments of
`include/quadriga_tools.hpp` and `calc_diffraction_gain` made to match `ray_state_update`'s `eps`. Two cases new (*Finite
plate - …*, 2.1, 2.2), two restated at the default side (*Mass law - m switches the limp mass law on*, *Plate rule - a
rigid mass-law panel passes …*), fifteen given `side` 0 where their statement is the infinite plate's; the OBJ header
checks in the C++, Python and MATLAB tests. `[wave]` 165258 assertions in 180 cases (output identical, the stored
twins unchanged), `[acoustic]` 41523 in 58, whole suite 1316330 in 1145. The 13 fits at their lab side through the
built library: sheet and formula within 0.0000 dB, traced solid within 0.0042 dB, R + T below one in every row, a band
on its single frequencies bitwise. CPU time unchanged (whole suite 46.0 against 45.9 s single-threaded).

**Taken (Stephan, 2026-10-19), after the precision study (`md/precision_report.md` 0, 1):**

- W1 adopted, a decided wave change: each in-medium leg starts with the relaunch offset it skipped, so it is measured
  from the hit point; the refracted ledger's share is the offset times `|g·N| / |dir_next·N|` (`g` the incoming segment,
  1 for types 2 and 5); preloaded legs untouched; the offset itself unchanged. The `[wave]` twins move (260 of 736
  rows, 5 tables, at most 8.8e-3 relative); the fixtures that wrote "the thickness crossed" from `origN` are restated to
  the hit point.
- W3″ adopted: in a mass-law medium and the float template the `use_vbs` tolerance is
  `1e-6 + 8·2⁻²³·max(|O|, |F|, 1) / |O − F|`.
- W2 (a nudge with a normal component) dropped: rays within about 3° of grazing in float or AVX2 stay a limit.
- `r1::wall_g` repaired; *Finite plate - …* traced to 89.9°.
- One rule of use and one line of limits (7, `md/wave.md` 8); no new open items; the large turned faces in float
  recorded only.

**Built (2026-10-20)**: W1 and W3″ in `ray_state_update` (`md/precision_report.md` 2): `src/ray_mesh_interact.cpp` 5211
lines against 5159; the `acc_dist` documentation of `ray_state_update` and `ray_progress` says the ledgers carry the
offset. Three cases new (*Far from the origin - …*, two `[acoustic]`, one `[wave]`); the fixtures that wrote a thickness
from `origN` restated to the hit point (10 `[wave]` cases of them failed on the library before); `tests/data/wave_twins.tsv`
rewritten (260 rows). `[wave]` 165594 assertions in 181 cases, `[acoustic]`
42391 in 60, whole suite 1317534 in 1148; the built library's run equals the what-if's line for line; printed rows
move in 76 cases (38 `[wave]`, 38 `[acoustic]`), every one accounted for. CPU time unchanged (+0.2 / −1.6 / −2.1 % on
three test files, single-threaded, alternated).

**Round history** (detail in the report and in `build_linux/scratch/i47/NOTES.md`):

- What was entangled before round 1 (the window analysis, i47, against a thin-plate impedance model of 4 mm,
  6 mm and 4-16-4 glazing). (1) `tf` had two jobs, the room-side absorption and twice the transmission: no
  setting reached both the absorption and the transmission of a pane (+3.3 to +42.7 dB above EN 12758 with the
  measured absorption; the plate model within 2.5 dB). Resolved by N1. (2) `coi*` needed an `att` baseline.
  Resolved by N4. (3) A pole on a rigid mass-law body broke the closed verdict (up to +135.6 dB). Resolved by
  N3. (4) The bounce compensated the leaks; at a sharp resonance the coherent sum over the bounces exceeded one
  (A1). Resolved by N2. (5) Meshed double leaves could not resonate: two 4 mm mass-law sheets 16 mm apart read
  −105 dB at 1 kHz against EN 35 dB, no mass-air-mass resonance (the model has a 30 dB peak at 211 Hz). Stands
  for mass-law sheets; two wave-medium leaves under N7 resonate (A18, M9).
- The candidates of round 1. M1 (`tf` on a closed plate is the room-side absorption, transmission from the
  plate): rules R1h, R1m, R2 reached the 13 constructions within 2.51 / 2.50 / 1.78 dB with resonance and
  bending (report 3.6); R1h decided (N1). M2 (no reflection back into a mass-law body; measured as B4k: sheet on
  solid within 0.0063 dB; on the conventions before round 1 it cost 2 cases, 53 assertions and 774 rows): N2. M3 (`res*` in `X₀`; variants A, B, B4, B4m, C measured in i47, B4s built): N3.
  M4 (coincidence; M4a in `X₀`, M4b a negative `coiA` without `att`): N4.
- The open items of round 1 closed by the source change: A1 (on the 84 rows of the resonance case `T + R` is at
  most 0.9951), A2 (the held exit judged by the invariant of air, the copy R1hG of report 3.9; the lining sheet
on its traced layer within 0.015 dB for every tracer), A3 (sheet on traced
  solid within 0.001 dB), A4 (K \| W one plate, both sides and twin 0.0000 dB), A5 (a rule of use), A6 (no pair
  rule and no clamp at mass-law faces), A7 (a mass-law body holds every tracer; the dense phase rows on the
  held reference), A10 (three columns), A11 (both forms of `res*`).

## 11. Rig

The rigs below are scratch: `build_linux/scratch` is untracked and removed by `make clean`, so a rig named here or in
the reports may be gone. The record is the numbers stated in this document and in the reports; a file name next to a
number says where it was produced. The instruments that survive are in `tests/catch2_tests/ray_test_helpers.hpp`
(`md/wave.md` 11), `tests/catch2_tests/test_acoustic_leaf.cpp` (`lw::walk`, a walker without recursion that collects a path in air only outside the structure;
`lw::stack`, the transfer matrix of fluid layers) and `tools/acoustic` (`sabine_to_random.py`, the A13 conversion with
its validation as `check`; `band_probe.cpp`, a band through `ray_progress` on a palette and plane faces; `paris_cf.py`;
`txlib.py` and `txfit.py` with `tl_refs.json`, the formula copy of the mass term and plate rule and the transmission fit
of section 9; `cases.py`, moved rows per case between two runs; `README.md`).

- `build_linux/scratch/tx` (`md/m10_m15_report.md`; `README.txt` lists every file): the candidate fits of M10 to M15
  (`batch.py`, `select.py`, the fits in `fits/`), the what-if generator `mkc.py`, the walker probe `probe.cpp`, the
  band and walker confirmations `band.py`, `conf.py`.
- `build_linux/scratch/ac2` (this document): `sprobe.cpp` (one sheet through `ray_mesh_interact`, scalar types 3
  and 4; build with `../a1/water/build_w.sh`), `s.in` / `s.out` (the wave-medium sheet against the limp mass law
  and the mass-law sheet, section 1), `peak.py` / `peak2.py` (the `tf` peak against the resonator law, and the
  random-incidence value of a peak at `f₀/cos θ`; outputs `peak.out`, `peak2.out`). `a19.py` (measured resonances
  against the cavity rules). `walker3.cpp` / `w3` (the a1 walker, rebuilt on today's library with `buildw.sh -
  walker3.cpp <abs out>`; run from the repository root with the test spec "P4 walker" and `ulimit -s unlimited`, the
  walk recurses once per event), `a18.py` (leaf, gap, air body and double leaf against fluid-layer transfer
  matrices; `a18.out`, `a18_sum.out`). `zf.py` (Thomasson's field impedance of a rectangular patch), `a13.py` (the
  options of A13, cache `zf_3x4.npz`), `a14.py` (the class-4 columns per target, walker specs `a14_spec_*`),
  `a14w.py` (their walker check).
- `build_linux/scratch/m9` (2.8, M9 phase 0; `README.txt` lists every file): `w4` (`walker4.cpp`, the walker with
  every column and an event count; `QD_SUMONLY=1`), `rp4` (`rp4.cpp`, the `ray_progress` band probe on a face list),
  `m9lib.py` (fluid-layer references), `b1.py` to `b8.py` (the items of the phase-0 report), `porcol.py` (Miki to
  power-law columns), `b6.py` (Shearer and Sakagami at random incidence, the shallower box). The walker recurses once
  per event: a followed box needs `ulimit -s unlimited`; the fixtures use an iterative copy.
- `build_linux/scratch/i47`: `panel.py` (thin plate and double unit), `fit.py` (EN octaves, absorption rows),
  `libwin.py` (formula copy of the rigid sheet), `mk47.py` (switch copy with a marker), `mkv47.py` (variants A, B,
  B4, C as real code), `rmi_B4m.cpp`, `rmi_B4k.cpp`, walkers `probe3.cpp` / `probe4.cpp` (P4 collects only outside
  the structure, for meshed cavities), band probe `rp2.cpp`, scans `runwin.sh`, `runx.sh`, `runscan.sh`.
  `NOTES.md` has the sources.
- `build_linux/scratch/ts1`: the marker and split tools of the test split.
- `build_linux/scratch/a1`: round 1. `wi/` the variant generators (`mkc.py`, `mkc_p3.py`) and builds
  (`build.sh <lib.cpp|-> <probe|-> <abs out>`, `bld1.sh`), `water/` the one-face probe (`wprobe.cpp`,
  `build_w.sh`), `p3/` step 0 to 7 rigs (`tl/rmb_chk.out` the fitted knobs), `p4/` fixture summaries, `p5/` the
  restatement scripts and `fails_by_case.py`, `p6/` the per-table diff (`tabdiff.py old new`,
  `rows.py old new <header prefix>`, `colmax.py run <prefix> col|a-b`), the what-if without the leak-chain fix
  (`rmi_noleak.cpp`) and the baselines before this change (`old_baselines/`).
- The test instruments and traps of `md/wave.md`, section 11, apply here too. Two more: `build.sh` needs absolute
  output paths; two traced solids with a cavity explode the outside-only walker's tree.

## Appendix: the `[acoustic]` cases by section

| case | section |
| --- | --- |
| *Thin sheet - mass-law angle dependence* | 2.1 |
| *Acoustic - medium_gain in-medium loss* | 2.1 |
| *Resonance - on a mass-law material res\* is the plate's resonance and leaves the permittivity alone* | 2.1 |
| *Coincidence - on a mass-law material coiF and coiQ are the bending stiffness of the plate* | 2.1 |
| *Coincidence column - a table that carries coiA loads and the column changes nothing - the mass-law part* | 2.1, 4 (D2-b) |
| *Mass law - m switches the limp mass law on* | 2.1, 4 (M10) |
| *Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency* | 2.1, 4 (M14), 10 |
| *Finite plate - side 0 is the infinite plate, a table without the column is side 3.29, a wave medium ignores it, a negative side is refused, a bonded run takes the side of the body that forms its term* | 2.1, 2.2 |
| *Mass cosine - a mass-law sheet behind a refracting layer forms its term at the cosine the wave has in air* | 2.1, 7 |
| *Refracting tracer - a traced bend is known from the state and the leg is exact - the mass-law part* | 2.1 |
| *Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate* | 2.2, 2.3 |
| *Bonded mass - a mass-law sheet inside a mass-law wall* | 2.2, 2.3 |
| *Bonded mass - the plate at any depth* | 2.2, 2.3, 7 |
| *ray_progress - the bonded plate at any depth* | 2.2, 2.3 |
| *Unentered branches - a sheet without the mass column of its mass-law host's permittivity is merged* | 2.2 |
| *Plate rule - a rigid mass-law panel passes the incoherent series of its face and its plate and reflects what its face does not absorb* | 2.3 |
| *Plate rule - a wall bonded of two mass-law bodies passes the same in both directions* | 2.3 |
| *Closed plate - a mass-law layer the wave is evanescent in passes the plate rule of its two faces* | 2.3, 7 |
| *Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids* | 2.3, 7 |
| *Total reflection of the outer pair - the sheet stays in the result - the mass-law part* | 2.3 |
| *Leak chain - each face at the cosine the wave has and the series on a closed far face - the mass-law part* | 2.3 |
| *Closed face - tf at a closed face is a pad and the two ports of a sheet on it add up* | 2.3 |
| *Unentered branches - a stack of a coherent and a mass-law layer against the laminate meshed as a solid with a lining* | 2.3 |
| *Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the invariant - the mass-law part* | 2.3 |
| *Bare tf - a bare face with tf is a transformer where it is open and a flat leak where it is closed - the mass-law part* | 2.3 |
| *Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the characteristic matrix - the mass-law part* | 2.3 |
| *Lumped columns - a bounding sheet: tf is a layer, a mass-law sheet a half-space* | 2.3 |
| *Stack with a mass-law layer - resolved by segments, not killed* | 2.2, 2.3 |
| *Lumped columns - an ordered stack: what the core carries, by matrix and by scattering - the mass-law part* | 2.3 |
| *Unentered branches - ray_mesh_interact's own glued composite against the redo - the mass-law part* | 2.3, 7 |
| *Held path - alpha on a lumped plate is charged over the path the held wave takes* | 2.3 |
| *ray_progress - the plate rule and the absorption peak at every frequency of a band* | 2.3, 2.6, 3.2 |
| *ray_progress - the random-incidence absorption of a peaked lining is the Paris integral of its series impedance* | 2.6, 3.1, 9 |
| *Held exit - a lining behind a rigid wall is the same as its traced layer, for every tracer* | 2.4 |
| *Held exit - a non-rigid mass-law body holds every tracer and passes at every angle* | 2.4 |
| *Held lining - a mass-law sheet behind a wave held in a mass-law wall is the same wall as its traced solid* | 2.4, 2.5, 7 |
| *Ports of a mass-law face from inside - the reflect port is the other direction's* | 2.4 |
| *Leak direction - a closed face hands the wall behind a traced layer the direction the wave has in air* | 2.4 |
| *Leak direction - a wave held in a mass-law body keeps its direction and a closed face hands on the wave's* | 2.4 |
| *Phase reference - a traversal carries the phase at the point the polyline leaves at - the mass-law part* | 2.4 |
| *Direction - a ray leaves a parallel slab parallel to how it arrived - the mass-law part* | 2.4 |
| *Stack geometry - an ordered stack leaves the ray on the line it arrived on - the mass-law part* | 2.4 |
| *ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band* | 2.4 |
| *ray_progress - refracting tracer: the segment out of a lossy medium is the direction the wave has - the mass-law part* | 2.4 |
| *Plate rule - a traced mass-law body passes in one path: nothing is reflected back into it* | 2.5 |
| *ray_state_update - mass-law material is excluded from slab resolution* | 2.5 |
| *Measurement - the mirror behind a second solid and behind a sheet inside a wall - the mass-law part* | 2.5 |
| *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face* | 2.3, 2.6, 3.2 |
| *Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches* | 2.6, 2.7, 3.1, 3.2, 3.8, 6 (A12) |
| *Lossy verdict - a lossy second medium has no critical angle - the mass-law part* | 2.6 |
| *Two-parameter mapping - air \| water, a hard wall's phase, a porous layer on it and every tracer below its critical angle* | 1, 3.8 |
| *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf* | 1, 2.8, 3.4 |
| *Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle* | 2.8, 3.1 |
| *Leaf - two leaves on a box of air resonate at the mass-air-mass frequency* | 2.8, 3.4 |
| *ray_progress - a leaf on a box of air at every frequency of a band* | 2.8 |
| *Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall* | 2.8, 3.1 |
| *Sheet between two bodies - nothing is created and the wave left in air inside the far body is killed* | 2.8, 6 (A21) |
| *Far from the origin - a traced mass-law wall passes what it passes at the origin, its leg measured from the hit point* | 7, 10 |
| *Far from the origin - in single precision a mass-law leg is not taken for a bent one* | 7, 10 |
