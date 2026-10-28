# Acoustic engine: issues for a later engine stage (2b)

The acoustic interface engine was closed at the end of step 2 (see `md/acoustic_engine_notes.md`). Step 3 builds the
acoustic material table and measures every material and its common combinations through a test rig
(`tests/catch2_tests/acoustic_material_rig.hpp`, its self-checks in `test_acoustic_materials.cpp`, the material cases in
`test_acoustic_mat_<group>.cpp`). Where the engine cannot express a construction, the miss is
recorded here with its size, the independent reference it was measured against, the materials it affects, and an
interim modeling guideline. Nothing here is worked on during step 3; a later engine stage (2b) takes this list as its
input, and the rig's cases become its regression tests.

Each issue: what happens physically; how it was measured; the size; what it affects in typical rooms; the interim
guideline; how to reproduce it.

**Engine-limit marks in the rig.** A comparison in the rig's test files (`tests/catch2_tests/acoustic_material_rig.hpp`,
`test_acoustic_mat*`) that misses because of an engine limit listed here carries a comment
`ENGINE LIMIT (<name>):` with the issue's name in its heading below (the code cites no document or number); it asserts
nothing, and the measured and the reference values are printed in the case's diagnostics (`QD_DIAG_RAY=1`). Known
engine issues therefore never make the suite fail; the marks (`grep "ENGINE LIMIT"`) are the regression list of the
engine stage that fixes them.

**Bounce cap.** Production scenes are traced with a cap of about 20 to 30 interactions per ray: a cavity that traps
rays multiplies the paths at every bounce (each splits into reflection and transmission), and about 150 million traced
rays is the practical limit. The rig takes the transmission of a box of air in front of a fill at a cap of 10 and
prints the reflection at that cap beside the uncapped one.

## Issues

### 1. A porous layer on a rigid wall adds too much insulation at low frequency ("porous layer on a rigid wall")

- **Construction.** 50 mm glass wool (the columns of page 11.3: Komatsu's model at a flow resistivity of
  13112 Pa·s/m²) as a sheet glued on 152 mm cast concrete (the preliminary row of page 11.4), sound from the wool side.
- **Physics.** Pressed against a heavy wall, the air in a thin porous layer hardly moves at low frequency, so its flow
  resistance has almost nothing to act on: the layer adds next to nothing to the wall's insulation below a few
  hundred hertz and rises toward its own loss at high frequency. In free air the same layer is a resistive screen
  (3.5 dB at 31.5 Hz).
- **Measured** (2026-10-23, normal incidence, `calc_diffraction_gain` with one straight path, scalar mode,
  `thin_slab_threshold` 0): the insulation the wool adds to the wall, in dB.

  | octave (Hz) | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | library | 8.04 | 6.71 | 5.38 | 4.09 | 3.42 | 4.78 | 7.32 | 9.32 | 13.02 |
  | transfer matrix: the same fluid layer on a heavy limp plate (347 kg/m²) | −0.02 | −0.07 | −0.18 | −0.28 | 0.16 | 2.53 | 5.98 | 8.79 | 13.24 |
  | wool alone in air, library and transfer matrix (equal) | 3.52 | 3.85 | 4.20 | 4.56 | 4.92 | 5.44 | 6.85 | 9.33 | 13.00 |

  The reference reproduces the wool alone exactly, so the layer's mapping is the same in both. The wall's fitted knobs
  cancel in the difference (with and without the layer).
- **Size.** Library minus reference: 8.06 / 6.78 / 5.56 / 4.37 / 3.26 / 2.25 / 1.34 / 0.53 / −0.22 dB from 31.5 Hz
  to 8 kHz, so up to 8.1 dB too much insulation at 31.5 Hz and 5.6 dB at 125 Hz, within 0.6 dB from 4 kHz. Room-side reflection of the same construction is right (within 6.3e-3 of the transfer matrix).
  Random incidence and other layers and walls are measured by the rig's combination cases.
- **Affects.** Transmission through any wall, floor or ceiling carrying porous treatment: absorbers on walls, carpets
  on floors, bass traps against walls. It matters for planning treatment across rooms; it does not move the sound
  field inside the treated room.
- **Likely cause (not proven).** The wave-medium layer is crossed as if air were behind it, and the mass-law wall's
  plate rule follows as a separate loss, so the layer's free-air loss is paid on top of the wall's.
- **Interim guideline.** Do not count porous treatment on a wall toward the insulation between rooms below about
  1 kHz; where it matters, fit the treated wall as one row (as page 11.4 does for lined walls), which loses the
  layer's angle-dependent absorption on the room side.
- **Through the rig** (2026-10-23, `Materials rig - a porous layer on a rigid wall: the added insulation`,
  342.76271 m/s, one tone per third, 50 angles), the insulation the wool adds, in dB:

  | incidence | quantity | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0° | library | 7.78 | 6.54 | 5.53 | 4.21 | 3.49 | 4.59 | 7.06 | 9.08 | 12.54 |
  | 0° | fluid layer on a limp plate of 346.7 kg/m² | −0.02 | −0.07 | −0.17 | −0.26 | 0.11 | 2.19 | 5.59 | 8.41 | 12.61 |
  | random | library | 3.09 | 3.13 | 2.81 | 2.98 | 4.27 | 6.74 | 9.87 | 14.78 | 21.55 |
  | random | fluid layer on an infinite limp plate | 1.27 | 2.80 | 4.23 | 5.84 | 7.95 | 10.86 | 14.24 | 18.27 | 25.02 |
  | random | fluid layer on a finite limp plate (Davy's σ at `side` 3.52 m, as the library's wall) | 0.16 | 0.38 | 0.81 | 1.75 | 3.78 | 7.10 | 11.00 | 15.51 | 22.71 |

  With the finite plate the wall's radiation cancels in the difference: at random incidence the library adds
  +2.93 / +2.75 / +2.00 / +1.23 / +0.49 / −0.36 / −1.13 / −0.73 / −1.16 dB more than the model (31.5 Hz–8 kHz), too
  much at low frequency as at 0°, slightly too little above 1 kHz.
- **The same mechanism in front of a pane** (2026-10-27): a curtain (3 mm of the rock-wool columns of the rig) on a
  100 mm box of cavity air before a 4 mm pane (page 11.4 row, sheet at the box's exit). Reflection is right (7.1e-4
  against the pane's half-space behind its pad). The insulation the curtain and box add, in dB:

  | incidence | quantity | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0° | library | 0.35 | 0.28 | 0.30 | 0.55 | 1.28 | 1.65 | 0.73 | 1.00 | 1.69 |
  | 0° | fluid layers on a limp plate of 10.0 kg/m² | 0.16 | 0.03 | 0.09 | 0.43 | 1.28 | 1.70 | 0.70 | 0.99 | 1.70 |
  | random | library | 0.19 | 0.16 | 0.20 | 0.27 | 0.70 | 1.87 | 3.47 | 2.24 | 2.00 |
  | random | fluid layers on a finite limp plate (side 1.35 m) | 0.12 | 0.02 | −0.00 | 0.10 | 0.68 | 1.69 | 2.05 | 2.55 | 4.30 |
  | random | the layers alone in air (the loss paid on top of the pane's) | 0.72 | 0.71 | 0.71 | 0.71 | 0.74 | 0.84 | 1.14 | 1.80 | 3.04 |

  At 0° the library is within 0.25 dB of the model. At random incidence the two estimates bracket where the result
  should be; the library lies inside that band except +0.18 dB at 1 kHz, +1.42 dB at 2 kHz and −1.04 dB at 8 kHz.
  Small for a curtain, but the same mechanism as the wall.
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Materials rig - a porous layer on a rigid wall*"` and
  `"Materials rig - engine limits printed*"` (`ENGINE LIMIT (porous layer on a rigid wall)`) print the rows above.

### 2. A band run near grazing can take the wrong bounce decision for a leaf on a box of air ("band decision near grazing")

- **Construction.** A 12.5 mm board (8.6 kg/m²) glued on a 25 mm box of air on a mass-law wall (page 10.4), one ray
  carrying a band of frequencies through `ray_progress` at the default `thin_slab_threshold` 0.15.
- **Physics.** Every frequency of a band takes the sum-or-follow decision made at the band's first frequency. Near
  grazing the first frequency follows the box's series, and higher frequencies inherit a decision that is wrong for
  them.
- **Measured** (2026-10-23): at 89° and 1816 Hz the band reflects 1.073 (energy created) where the same frequency
  alone and the transfer matrix give 0.955; with 125 Hz as the first frequency the reflection is off by up to 0.164
  (1.6 kHz, 89.5°); a traced porous layer through `calc_diffraction_gain` at 0.15 is off by 1.0 dB in transmission.
  At threshold 0 every band equals its single frequencies.
- **Size.** 8.3e-4 in octave absorption at random incidence (only angles past 85° are hit); the filled box costs 12
  times more at 0.15.
- **Interim guideline.** Run acoustic scenes with `thin_slab_threshold` 0.

### 3. Two traced wave media back to back on one straight path ("wave media back to back")

- **Construction.** Two wave-medium layers (glass wool | board) both traced as solids, glued face to face, through
  `calc_diffraction_gain` with one straight path.
- **Measured** (2026-10-23): transmission 4.9 dB off the transfer matrix at 16 Hz and above one at 88° (+3.1 dB);
  through `ray_progress` the exchange between the two bodies is followed as paths and depends on the gain floor
  (0.106 in reflection at −120 dB, 0.027 at −250 dB). Two bonded traced mass-law bodies deviate from the bonded plate
  by up to 2.0 dB at 16 kHz (0.05 dB at 1 kHz); the sheet | solid form is exact.
- **A filled box in a wall** (2026-10-27): the same mechanism where a box of cavity air lies in front of a traced
  porous fill (the precise drywall recipe: 12.5 mm board | 25 mm of air | 50 mm rock wool | 12.5 mm board at the
  exit; panel absorbers with a fill). The sound bounces between the leaf and the fill face, and every bounce is a
  path of its own. Transmission against the transfer matrix of the layers, largest per node:

  | method | off by |
  | --- | ---: |
  | one straight path (`calc_diffraction_gain`) | 9.47 dB (100 Hz, 30°) |
  | the traced paths, at most 10 interactions | 5.04 dB (100 Hz, 30°) |
  | the traced paths, at most 60 interactions | 0.29 dB (63 Hz, 0°) |
  | without the second board: one path / the traced paths (60) | 4.57 / 0.14 dB |

  The scene tracer's LOS modes 2–6 drop the traced paths that arrive by transmission only and take the direct path
  through a wall from the diffraction gain, so a scene's direct sound through a filled wall carries the one-path
  miss; at the production cap of 20 to 30 interactions the traced paths are between the last two rows. Reflection
  moves with the cap too: at 10 interactions the precise drywall recipe's octave absorption changes by up to 0.042
  (0.23 per angle and tone), the MDF panel absorber with a fill by up to 8.2e-3.
- **A dense leaf traced behind a box of air diverges** (2026-10-27, independent review of the rig): board sheet |
  75 mm box of air | 12.5 mm board as a traced solid reflects up to 1.2e13 at 125 Hz between 40° and 53° (267 of
  1566 nodes above one, 61 calls), octave α down to −1.9e12; transmission 20.8 dB off through the paths, 11.5 dB on
  one path. The same back board as a sheet at the box's exit is right within 3.0e-3 dB. The rig refuses the traced
  form.
- **Affects.** Transmission through stud walls with a filled cavity drawn as the precise recipe, and through walls
  with filled panel absorbers; reflection of filled cavities at a low bounce cap.
- **Interim guideline.** Build stacks with the room-side layer as a sheet and the layers behind it as traced solids;
  a leaf behind a box of air is a sheet at the box's exit, never a traced body. For insulation between rooms through a filled stud wall, use the lumped `drywall_filled*` rows, not the precise
  recipe.
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Materials rig - engine limits printed*"` (`ENGINE LIMIT (wave media back to back)` rows).

### 4. Three or more bonded mass-law bodies: the state is one level deep ("three bonded bodies")

- **Construction.** A masonry block with plaster bonded on both faces: three mass-law bodies that move as one plate
  (the standard residential wall).
- **Physics.** Bonded mass-law bodies are one plate whose surface masses add (material page, section 5.4). The ray
  state remembers only the body before the current one, so a traced plate re-forms its first face's absorption at its
  exit from that word alone; with three or more bodies the plate rule is paid within a bound, not exactly.
- **Known size** (material page, section 13; engine notes 9.1): a plate of three or more bonded bodies pays its rule
  within +3.02 / −3.01 dB; a wall with an ordered stack inside it is up to 3.01 dB from its traced twin.
- **Measured on a plastered wall** (2026-10-27; 13 mm plaster, 16.9 kg/m², on both faces of a 215 mm block,
  396 kg/m², 430 kg/m² in all as Hopkins' wall, one `side` of 3.05 m): with the plaster as a sheet on the block's
  front face and a sheet at its exit, every face is a two-face stack and the block carries the state: **exact**,
  4.8e-5 dB per node against the bonded plate, also with unequal face absorptions. With the far plaster a traced
  third body, random-incidence `tl` against the bonded plate, in dB:

  | octave | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | library | 38.59 | 41.12 | 40.78 | 42.80 | 53.89 | 63.93 | 74.07 | 82.47 | 92.38 |
  | bonded plate | 41.57 | 44.11 | 41.61 | 44.49 | 55.92 | 66.01 | 76.82 | 84.97 | 92.19 |
  | Hopkins, measured (not fitted) | | 35.4 | 39.6 | 42.7 | 52.2 | 58.3 | 65.8 | 70.6 | |

  Up to 6.56 dB per node (160 Hz, 89.1°). Exactness of the sheet form needs one `side` for the bonded run (page
  rule 3): plaster at 3.05 m on concrete at 3.52 m is 0.80 dB off per node, a 4 mm pane on concrete 6.7 dB.
- **Affects.** Bonded runs of three traced bodies (a finish solid on a screed solid on a slab solid); not plaster or
  a finish drawn as sheets on a solid's two faces.
- **Interim guideline.** The masonry rows of the material table are the finished wall, plastered on both faces, as
  one row; plaster sheets on both faces of a block solid are an equivalent recipe. A bonded floor is the slab and its
  screed as one solid with the finish solid on top; keep traced bonded runs to two bodies.
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Materials rig - engine limits printed*"` (`ENGINE LIMIT (three bonded bodies)`); the
  sheet form is asserted in `"Materials rig - combinations*"`.

### 5. Three co-located faces: a rug on a leaf floor ("three co-located faces")

- **Construction.** A rug (a porous sheet) on boards on joists (a board leaf sheet on a box of cavity air) or on a
  floating laminate (a leaf sheet on a traced underlay): two leaves, or a leaf and a fill, in one plane.
- **Physics.** The engine forms each event from two co-located faces (`src/ray_mesh_interact.cpp:2617`: "At most two
  co-located faces form a stack; a third layer is a separate event"); a sheet | sheet pair is resolved only with air
  behind it. A third face in the same plane is resolved by whichever two faces the intersector reports. Making the
  leaf a traced solid instead gives two traced wave media back to back (issue 3).
- **Measured** (2026-10-27, a scratch probe of the rig; reflection against the fluid layers on the backing's
  half-space behind its pad): a 10 mm rug (the glass-wool columns) on a 15 mm parquet sheet on 152 mm concrete is off
  by up to 0.98 in R per node (25 Hz, 87.3°), octave α 0.42 where the layers give 0.005 at 31.5 Hz; a rug on a
  12.5 mm board leaf on a 50 mm box of air is off by up to 0.96, octave α 0.48 against 0.03 at 31.5 Hz; both within
  0.003 from 1 kHz. Which two of the three faces the engine pairs is not determined, so the size is not repeatable.
  The recipe with the parquet as a traced finish solid is within 1.3e-2.
- **A modeling limit, likely to stay.** The rig's stack rule refuses three co-located faces; there is no fixture.
- **Affects.** Rugs on wooden floors on joists and on floating laminate, the common residential floors; any object
  standing on a sheet.
- **Interim guideline.** Hard finishes (parquet, vinyl, tile) are thin traced solids, so a rug lies as a sheet on a
  solid; rugs on leaf floors are left out or drawn on a solid patch; objects stand on solids.

### 6. A ray grazing a solid's face is relaunched onto the same face and gains energy ("grazing relaunch")

- **Construction.** A closed box (1.0 × 0.5 × 2.0 m, a rigid mass-law row with `tf` 0.1) standing on a rigid floor,
  plane waves from the upper hemisphere (the object-test probe of the rig, 2026-10-27).
- **Physics.** A ray about 0.2° from grazing a face hits that face again at its own launch point at every
  `ray_progress` call; each hit splits it into a reflected and a transmitted ray, so the ray count doubles per call
  and the summed power grows.
- **Measured.** One such ray of unit power: 1.000 → 1.868 in ten calls (rays 2 → 364), at the origin, not far from
  it as the lost rays near grazing of page section 13. In the object test (16 directions, cap 10, at most 3
  reflections, −60 dB floor) no direction hits it and no power is left at the cap; on 64 directions it appears in 2
  directions of a 1.5 × 0.9 × 0.8 m porous body, with rays doubling past about 12 interactions.
- **Affects.** Any traced solid hit near grazing: objects, walls seen along their face; energy is created and the
  path count grows with the cap.
- **Interim guideline.** Keep the bounce cap (production runs at 20 to 30 interactions).
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Materials rig - objects*"` (`ENGINE LIMIT (grazing relaunch)`, the power per call).

### 7. A heavy wall's coincidence: one constant loss factor, a fixed coincidence frequency ("heavy-wall coincidence")

- **Construction.** Cast concrete walls 150–203 mm (the `concrete` rows), and any heavy generic solid drawn at another
  thickness.
- **Physics.** A concrete wall's coincidence lies near 114 Hz at 150 mm (fc·h 17.1 Hz·m) and moves as 1/h. Above it,
  measured walls gain about 4.5 dB per octave because their total loss factor falls with frequency (0.04 at 125 Hz to
  0.007 at 4 kHz, NRC data). The mass-law body has one constant loss factor (`coiQ`), and `coiF` is a fixed column,
  while the mass term scales with the traced thickness.
- **Measured** (2026-10-27, the concrete fit, side 3.523 m, against the Litvin pair mass-shifted to 150 and 203 mm,
  125 Hz–4 kHz): the physical plate (coincidence 114 Hz, η 0.005) misses by up to 8.3 dB, with η 0.02 up to 11.4 dB;
  `coiF` pinned at 114 Hz and every other knob fitted misses by 1.25 dB with knobs at their bounds and needs about 4000
  angles; `coiF` free (4.4 kHz) misses by 0.25 dB. The shipped row takes the free fit, so this is no miss of the row.
- **Side effect.** The fitted mass level (`thk`/`fRef`) stands for 183 kg/m² instead of 342 kg/m² at 150 mm, so mass
  bonded to the row counts about twice: a 15 mm plaster sheet (19.5 kg/m²) adds 0.22–1.24 dB on one face and
  0.41–2.33 dB on both (random incidence, octaves 31.5 Hz–8 kHz), against 0.48 / 0.94 dB from the mass law (scratch
  probe, exact against the bonded plate to 2.3e-6 dB).
- **Affects.** Heavy walls outside their fitted range; plaster and finishes bonded to concrete (within the ±1 dB class
  tolerance at the measured sizes).
- **Interim guideline.** Use the concrete rows within 150–203 mm; read added insulation from bonded plaster or finishes
  on concrete as an upper bound.
- **Heavy masonry** (2026-10-28): the measured walls gain 7.9–9.4 dB per doubling of surface mass at 125 Hz–1 kHz
  against 6.02 dB of the mass term. One fit holds within the walls' own scatter over 150–250 mm (worst single wall
  2.23 dB); at 300 mm it reads −4.04 dB against the thickest brick wall, at 100 mm up to +3.31 dB above the class
  trend, so the rows are asserted at 150–250 mm only.
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Material - concrete"` (the row against the Litvin pair);
  `"Material - masonry_heavy"` (the printed ends).

### 8. A pane's coincidence dip is capped by its face absorption ("pane coincidence capped by the face absorption")

- **Construction.** Single panes (`glass_4mm`, `glass_6mm`) and any thin mass-law sheet whose coincidence lies in the
  band.
- **Physics.** At coincidence a bare plate passes nearly all the sound at the coincidence angle. In the engine the face's
  transmission factor `tf`, which also sets the face's absorption, caps the transmission at every angle, so the depth
  of the coincidence dip is set by the face absorption, not by the plate's loss factor.
- **Measured** (2026-10-28, the physical 4 mm plate, coincidence 3.12 kHz, loss factor 0.01, side 1.349 m, fine grid):
  the coincidence octave (4 kHz) reads 29.89 dB at `tf` 0.017 (a pane's measured absorption) and 21.74 dB at `tf` 1;
  6 mm (2 kHz octave) 30.32 dB against 22.24 dB. The shipped rows meet the laboratory panes within ±3 dB with the
  measured absorption, so the case carries no mark.
- **Affects.** Windows and thin panels near coincidence: the dip cannot be deeper than the face absorption allows.
- **Interim guideline.** None needed for the glass rows; a row that must reproduce a deep dip cannot keep a hard face.
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Material - glass"`.

## Modeling guidelines collected in step 3

Rules for scene models that follow from measured behavior; they go to the material page (section 12) at the end of
step 3 where they are not there already.

- **Treatment is a sheet on a rigid solid.** The room-side layer of a stack (carpet, porous absorber, the leaf of a
  panel absorber) is a sheet; the layers behind it are traced solids. Not: two traced wave media back to back
  (issue 3), two mass-law sheets on one face (page 12, rule 9), a thin lossless wave layer as a traced solid (it
  reflects totally past its critical angle instead of tunneling).
- **The order of co-located sheets is the order of their faces in the mesh:** the first face is the room side.
  Reversing it turns wool | board into board | wool (reflection up to 0.999 different). Check the face order after
  an export.
- **An air gap behind a panel is a box of cavity air** (`e` 1.000001) glued exactly between the panel and the wall,
  never a free air space between two objects: the box's bounce series is summed in one event, a free gap is followed
  bounce by bounce and runs out of interactions (page 12, rules 9 and 16).
- **`thin_slab_threshold` 0** for acoustic runs (issue 2).
- **Frequencies in the library's unit:** acoustic frequencies map to the library's frequency unit with the sound
  speed 342.76271 m/s (`src/qd_atmospheric_absorption.hpp`; the wrappers convert with it). Material columns in GHz
  are made with the same constant.
