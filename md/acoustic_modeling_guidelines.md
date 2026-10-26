# Modeling a venue for the acoustic tracer (draft 3)

Draft 3, 2026-10-26 (draft 0 revised after two review rounds and an independent review). How to build a room model
in Blender so that the acoustic material table (`md/acoustic_materials.md`) gives meaningful results. The rules are collected from the material page
(`html_docu/quadriga_lib_material_model.md`, sections 2, 7, 10, 12, 13; "rule n" means its section 12) and from the
engine issues measured during the material work (`md/acoustic_engine_stage2b_issues.md`). Rules that are the
drafter's own proposal are marked **(proposed)**.

## 1. Run settings

- **Scalar interaction types.** Use types 3–5 for acoustics; the EM TM channel of a mass-law material is formal (up
  to 3.9 dB from its traced twin, rule 8).
- **`thin_slab_threshold` 0.** Every bounce series inside a body is then summed in one event, and a band of
  frequencies equals its single frequencies. At `ray_progress`'s default 0.15 a band near grazing can take the wrong
  bounce decision (stage-2b issue 2: reflection up to 1.073 for a board on a box of air; 8.3e-4 in octave absorption).
  `calc_diffraction_gain` already defaults to 0.
- **Frequencies at the library's sound speed.** Acoustic frequencies map to the library's frequency unit with
  342.76271 m/s (`qd_acoustic_freq_scale` in `src/qd_atmospheric_absorption.hpp`): 1 kHz is 0.87464 GHz,
  125 Hz is 0.10933 GHz. The wrappers and `qrt_ray_trace` convert with it. The material page and the starting rows
  of its section 11.4 use 343 m/s; the difference is at most 0.017 dB in transmission and 4e-4 in absorption.
- **`min_gain_dB` with a margin** near walls: inside a plate the running gain lies up to −10·log₁₀(`tf`) below the
  level the path will reach (10 dB for `tf` 0.1; rule 15).
- **Local coordinates.** Build the scene near the origin, never in georeferenced coordinates; a body thinner than the
  relaunch offset (about 1 µm per meter of coordinate) is skipped (rule 1, page section 13).

## 2. Material names in Blender

- The Blender material name selects the table row. The reader (`src/obj_file_read.cpp`, `resolve_csv`) looks the
  name up exactly, then by its part before the first dot, and nothing else.
- **The thickness suffix marks a sheet.** A row name ending in `_<NN>mm` (always last) is a sheet of that
  thickness; a name without it is a traced solid. `concrete` and `drywall_filled` are solids, `concrete_150mm` and
  `drywall_filled_100mm` sheets.
- Names are case-sensitive and snake_case; row names contain no dot. Blender's duplicates (`concrete.001`) and
  variants for display (`concrete.gray`, `concrete.white`) therefore all use the row `concrete`.
- **An unmatched name silently becomes air** (index 0, no material). Load the scene with `csv_strict` set to get an
  error naming the material instead. **(proposed)** Use `csv_strict` for every production run.
- A material file replaces the built-in table.

## 3. Geometry: solids and sheets

- **Traced solid** (row without `thk`, or `thk` 0): a closed mesh; the tracer follows the ray through its interior.
  The mass term of a mass-law solid uses the traced normal thickness (legs longer than 1.5 mm), so **the thickness
  comes from the geometry**: model the wall as thick as it is.
- **Sheet** (row with `thk` > 0, name ending in `_<NN>mm`): the layer of that thickness is resolved in one event and
  the face needs no mesh thickness.
- **A sheet material goes on ONE plane.** A wall drawn as a single plane takes a sheet row (`concrete_150mm`,
  `drywall_filled_125mm`). A closed box with a sheet material on every face counts the wall twice (each face is a
  whole wall). Walls drawn as boxes, with their thickness, take the solid rows (`concrete`, `drywall_filled`).
- **When to use which.** Sheets for thin layers on one plane: panes, glazing units, open boards, leaves, carpets,
  curtains, finishes. Traced solids for walls drawn with thickness, for any layer that carries another layer
  (a screed under tile), for closed objects (hard boxes, porous bodies), for a porous fill behind a box (rule 10) and
  for boxes of cavity air. A sheet and its traced twin agree within the limits of page section 13.
- **One material per closed body; finishes as sheets** (to be confirmed in phase 3): a glass front on a metal box is
  a glass sheet glued on that face.
- **Face orientation.** A face is hit on its front side when its vertices are counter-clockwise as seen from the ray;
  a front-side hit enters the body that owns the face. Keep normals pointing out of solids (check after an export).
- **Glue co-located faces exactly.** Hits closer than 1 mm along the ray are one event; near grazing, faces much
  closer than 1 mm in normal distance become two events with air between them (rule 12).
- **No overlapping geometry.** At most two co-located sheets form a stack; a third layer is a separate event
  (rule 13).

### 3.1 Checklist: forbidden and limited configurations

Check every recipe against this list; the material list's recipes avoid them.

| configuration | where it arises | size | instead |
| --- | --- | --- | --- |
| three co-located faces (a sheet between two bodies, or three sheets) | a finish sheet on a screed sheet on a slab; a carpet sheet under a standing object; an absorber sheet on a plaster sheet on masonry | reflection 0.154 where the fluid layers give 0.994 (page 13); such rays are killed | the carrying layer as a traced solid; cut the carpet under objects |
| two mass-law sheets on one face | tile, vinyl or parquet on a screed sheet; a double glazing as two sheets | not allowed (rule 9; the rig refuses it) | the screed as a solid; glazing as one lumped sheet |
| three or more bonded mass-law bodies | plaster sheets on both faces of a block solid | within +3.02 / −3.01 dB (page 13; stage-2b issue 4) | the finished-wall masonry rows (plaster included) |
| a sheet behind a traced solid | plaster on the far face of a wall | the rig cannot build it today | the finished-wall rows |
| two traced wave media back to back | a cushion on a sofa body; a person on a sofa; porous bodies touching | 4.9 dB off, transmission above one at 88° (stage-2b issue 3) | merge cushions into the body; keep porous bodies apart |
| a free air space | a closed box of mass-law sheets (a casing); an air gap left open between a panel and the wall | up to 0.29 in reflection at 255 interactions (rule 9) | a hard box; a box of cavity air |

## 4. Panel size `side`

- `side` is √(panel area) of a mass-law material; it sets the panel's radiation and so its low-frequency
  transmission. It is a row value, not read from the geometry.
- Each mass-law row carries the √area of a typical panel of its use (the table in section 1.8 of the material list:
  a wall of 4 × 2.6 m, a pane of 1.2 × 1.4 m, a door of 0.9 × 2.0 m). The modeler does not change it, and the engine
  does not read the drawn panel: it sees one triangle, not the panel.
- **The drift is accepted.** Where a drawn panel differs from the typical size, the result drifts with it. Its size,
  as an example: a 4 mm pane read at 3.29 m instead of 1.349 m transmits 1.7 dB more at 125 Hz and 4.8 dB more at
  31.5 Hz (page 11.2). There are no size-variant rows.
- **One `side` per bonded run** (rule 3): glued finishes carry the host's `side`.
- Leaves and wave media keep the infinite plate: `side` acts only on mass-law materials (page section 13).

## 5. Class 1: walls, floors, ceilings, windows

- **A rigid body carries the mass column and `tf`.** A rigid face without `tf` absorbs nothing and its plate passes
  nothing (rule 4); a rigid permittivity without `m` is a wave medium and tunnels as a sheet (rule 5). Keep rigid
  bodies lossless (rule 6). These are row properties; the modeler only picks the row.
- **One `att` per bonded run or glued stack** (rule 2): glued finishes carry `att` 0.
- **Masonry by what you see.** Concrete has its own rows. Other masonry takes `masonry_heavy` (solid brick, dense
  block, sand-lime), `masonry_hollow` (hollow brick and block) or `masonry_light` (aircrete, lightweight block): each
  is the finished wall, plastered on both faces. Do not add plaster sheets to them; `plaster_15mm` is for an
  unplastered wall that gets plaster.
- **Drywall by what the plan shows.** `drywall` or `drywall_filled` (solid, wall drawn with thickness) or the sheet
  of the nearest total thickness (`drywall_100mm`, `_125mm`, `_150mm`, `drywall_filled_*`). The rows assume one
  12.5 mm gypsum board each side on metal studs, the rest cavity, empty unless filled. The precise option, for a
  build that matters: `gypsum_leaf_12_5mm` | box of cavity air (| a mineral-wool fill as a traced body at the back
  of the box, rule 10) | leaf; the studs are not modeled.
- **Dry lining** on dabs or furring: `gypsum_leaf_12_5mm` on a box of cavity air of the dab or furring depth, on the
  masonry solid.
- **Bonded toppings and finishes.** A layer that carries another layer is a traced solid: a screed under tile, vinyl
  or parquet is a `screed` solid; the finish is a sheet on it. Two bonded traced mass-law bodies deviate from the
  bonded plate by at most 2.0 dB at 16 kHz (stage-2b issue 3).
- **Floating screed:** `screed` solid on a resilient layer on the slab; how the resilient layer is carried is open
  (material list, 4.6).
- **Glazing units** are one lumped sheet; never two meshed mass-law sheets for a double unit (rule 9). Two separate
  sashes with a wide air space are two leaves on a box of cavity air.
- **Floors on joists** **(proposed recipe)**: `board_leaf_22mm` on a box of cavity air of the joist depth on the slab
  solid. Wood panelling on battens: `wood_leaf_12mm` on a box of the batten depth. A stage is modeled in detail from
  the same parts.
- **Suspended ceilings** **(proposed recipe)**: `gypsum_leaf_12_5mm` (or a tile sheet) on a box of cavity air of the
  plenum depth.
- **One board, two rows.** A board used as a free panel takes its plate row (`gypsum_board_12_5mm`); used as a leaf
  on a box of air it takes its leaf row (`gypsum_leaf_12_5mm`, named `<material>_leaf_<NN>mm`).
- **Thickness sheets** exist at round thicknesses (`concrete_150mm`, `concrete_200mm`); draw the wall at its real
  thickness with the generic solid where the geometry has thickness.

## 6. Class 2: hard furnishing

- **Open single panels** (table tops, shelves) are `furniture_board_*` sheets.
- **Closed objects are hard boxes** from the asset library (section 8): `box_cabinet` (wardrobes, cupboards),
  `box_bookshelf`, `box_metal` (fridges, freezers, bar counters, metal cabinets), `box_casing` (TVs, speaker
  cabinets, appliances). Never a closed box of sheets: it holds a free air space (3.1).
- **(proposed)** Leave out objects much smaller than the wavelengths of interest (legs, tubes, pipes, small
  appliances): their effect is scattering and diffraction, which the engine does not model (page section 13).

## 7. Class 3: soft furnishing and treatment on surfaces

- **Treatment is a sheet on a rigid solid.** The room-side layer of a stack (carpet, porous absorber, the leaf of a
  panel absorber, a curtain) is a sheet; the layers behind it are traced solids (stage-2b issue 3). Not: two traced
  wave media back to back (4.9 dB off and transmission above one), two mass-law sheets on one face, a thin lossless
  wave layer as a traced solid.
- **Place treatment separately on the finished wall or floor.** The wall or floor is modeled with its own row first;
  the carpet or absorber is a sheet glued to its face. There are no combination rows.
- **Face order of co-located sheets.** The order of their faces in the mesh decides the stack order; the first face
  is the room side (stage-2b collection; the page states the stack as "the layer hit on its front side first").
  Reversing it turns wool | board into board | wool (reflection up to 0.999 different). Check the face order after an
  export.
- **An air gap is a box of cavity air** (`cavity_air`, `e` 1.000001) glued exactly between the panel, curtain or
  absorber and the wall; never a free air space between two objects. The box's bounce series is summed in one
  event; a free gap is followed bounce by bounce and runs out of interactions (a 12.5 mm board on 25 mm of air: up to
  0.29 in reflection at 255 interactions; rules 9 and 16).
- **Curtains** hang as a fabric sheet on the front face of a box of cavity air that reaches the wall or window:
  `curtain_flat_*` for curtains hung flat, `curtain_pleated_*` for curtains with fullness (effective columns; the
  pleats are not drawn).
- **Underlay** below a carpet or a laminate leaf is a thin traced body (`underlay`) of its real thickness.
- **Small cushions and pillows** that lie free: a traced porous solid of the object's shape (`pu_foam`,
  `polyester_fiber`); the fabric cover is ignored. A cushion that lies on a sofa is merged into the sofa's body.

## 8. Objects: hard boxes, porous bodies, absorbing volumes (classes 2–4)

- **Use the assets.** Objects come as OBJ meshes generated by script, with the material assigned, and as a Blender
  asset library built on them: drag, place, and scale only where the asset allows it. Do not build object meshes
  with other rows.
- **Hard box** (bookshelf, cabinet, metal-clad furniture, casings): a closed box of the object's outer size whose
  faces absorb, per octave, the measured per-object absorption area divided by the box's surface. Scaling keeps the
  face absorption, so the object's absorption area grows with its surface.
- **Porous body** (sofa, armchair, bed, mattress, large cushions; a person; an audience or rows of seats): a low-poly
  shape or a cushion-like box of a porous row, fitted so that the body's absorption area matches the per-object
  data. It reflects at its face, absorbs inside and transmits some. Its absorption area does not scale with its
  surface: porous assets come in size variants (1-, 2-, 3-seat sofas) and are **not scaled**.
- **Absorbing volume** (clutter, foliage): a closed volume matched to air with a loss per meter. Items on a desk or
  shelf are one volume drawn over them: `clutter_sparse`, `clutter_medium`, `clutter_dense`. Dishes and glasses are
  left out or counted in a clutter volume. Clutter mainly scatters, which is out of scope; the volume gives only its
  absorption. For furniture an absorbing volume is wrong: it reflects nothing.
- **Placement.**
  - Assets stand on the floor as measured (the cabinet's measured surface excludes its floor face); against a wall,
    the share of the back face is lost.
  - Cut the carpet sheet under a standing object, or keep a clearance of a few millimeters: a carpet sheet between
    the floor and an object is three co-located faces (3.1).
  - Porous bodies must not touch each other (a person on a sofa, a cushion on a body): merge or keep a gap.
- **Per-object convention.** A per-object area measured in a reverberation chamber is used like for like; the rig's
  object test computes the asset's absorption area in a diffuse field. Placement values (an absorber measured in a
  corner) are no material property.
- **(proposed)** Plants: leave small ones out; a large plant is an absorbing volume of its foliage outline. Clothes
  racks: a porous body around the hanging clothes. A person: a porous body of a low-poly human shape. A dense
  audience or rows of seats: a porous box over the seating area, of the height of seated persons.
- Bodies reflect specularly where real objects scatter and diffract.

## 9. Class 5: acoustic treatment

- Porous panels: a sheet on the wall or ceiling solid. Panels stood off the wall: absorber sheet | box of cavity air |
  wall.
- Panel absorbers and microperforated panels: leaf sheet | box of cavity air (| porous fill as a traced body at the
  back of the box) | wall. The wall keeps its own row behind a fill (its reflection is within 6.3e-3 of the transfer
  matrix, stage-2b issue 1); no separate `tf` 0 row.
- Bass traps **(proposed recipe)**: a porous body of the trap's shape in the corner, from the asset library.
- Suspended absorbing ceilings **(proposed recipe)**: tile sheet | box of cavity air of the plenum depth | slab.
- Diffusers: a hard box (`box_diffuser`, proposed) gives their absorption; their scattering is out of scope.

## 10. Class 6: outdoor

- **Ground** is a porous half-space: a traced solid much thicker than the sound penetrates, of a ground row
  (`ground_grass`, `ground_soil`, `ground_gravel_loose`, `ground_gravel_compacted`, `ground_snow`,
  `ground_forest_floor`). The rows are Delany-Bazley at the class flow resistivity. Loose gravel and a compacted
  gravel road differ by a factor of about 200 in flow resistivity: pick by what you see.
- **At short range the ground acts nearly rigid at low frequency** (63–250 Hz excess attenuation +5.3 to +6.0 dB
  for σ 12.5–200 kPa·s/m² at 20–200 m, review sizing); the uncertainty lies at higher frequencies and in the
  spherical wave.
- **Spherical-wave limit.** The engine reflects a plane wave; the ground wave of a spherical source is not modeled.
  The difference (review sizing) is at most 0.6 dB on grass (σ 200 kPa·s/m²) to 200 m, but 2.8 dB (source at 1 m)
  and 3.8 dB (source at 0.5 m) on snow (σ 12.5 kPa·s/m²) and 2.5 dB on forest floor (σ 31.5 kPa·s/m²) at 200 m.
  Keep sources over soft ground at least about 1 m high, or accept the size.
- **Asphalt** is a hard solid (`asphalt`); porous asphalt is a thin porous sheet (`asphalt_porous_40mm`) on it.
  Open water is `water_20c`.
- **Vegetation** (foliage of trees, hedges, shrubs) is a closed absorbing volume of the `vegetation` row drawn around
  the foliage: matched to air, it reflects nothing and attenuates along the path.
- Keep coordinates local (section 1).

## 11. Engine limits to work around

One line each, with the size measured.

| limit | size | what to do |
| --- | --- | --- |
| A porous layer on a rigid wall adds too much insulation at low frequency (stage-2b issue 1) | up to 8.1 dB at 31.5 Hz, 5.6 dB at 125 Hz, within 0.6 dB from 4 kHz; room-side reflection right within 6.3e-3 | do not count porous treatment toward the insulation between rooms below about 1 kHz |
| A band near grazing takes the first frequency's bounce decision (issue 2) | reflection up to 1.073 at 0.15; 8.3e-4 in octave absorption | `thin_slab_threshold` 0 |
| Two traced wave media back to back (issue 3) | 4.9 dB off at 16 Hz, transmission above one at 88° | room-side layer as a sheet, the rest traced |
| Two bonded traced mass-law bodies (issue 3) | up to 2.0 dB at 16 kHz (0.05 dB at 1 kHz) | the outer layer as a sheet |
| Three or more bonded bodies; the state is one level deep (page 13, stage-2b issue 4) | within +3.02 / −3.01 dB | the finished-wall masonry rows; keep bonded runs to two bodies |
| A sheet between two bodies, three co-located faces (page 13) | reflection 0.154 where the fluid layers give 0.994 | make the middle layer a traced body (rule 10) |
| Reversed face order of a co-located stack | reflection up to 0.999 different | check the face order after export |
| A free air gap between objects (rule 9) | up to 0.29 in reflection at 255 interactions | a box of cavity air |
| The first path through a layered wall is not the wall (page 13) | −4.87 dB for a lossless second layer at 85°; +1.37 dB after 20 interactions, +0.04 dB after 60 | allow enough interactions |
| The two sides of a lossy lining on a rigid wall differ (page 13) | 0.07–1.38 dB (dense), 0.52–2.70 dB (porous), 10° to 75° | read the result from the treated side |
| Leaves have no bending stiffness and keep the infinite plate (page 13) | coincidence not expressible (box windows: 2–3 kHz) | a mass-law row where coincidence matters |
| A curtain's relaxation in μ holds on the random-incidence average only (page 13) | — | do not read a curtain per angle |
| A source inside a medium with an unknown bend (rule 11) | 0.21 / 0.96 / 3.00 dB at 10° / 20° / 30° | seed the bent bit, use type 5 |
| The ground wave of a spherical source (page 13) | ≤ 0.6 dB on grass to 200 m; 2.8 dB (source 1 m) and 3.8 dB (0.5 m) on snow, 2.5 dB on forest floor at 200 m (review sizing) | sources at least about 1 m above soft ground |
| Out of scope (page 13) | — | scattering, diffusion, edge diffraction, flanking through the building, level-dependent absorbers |
