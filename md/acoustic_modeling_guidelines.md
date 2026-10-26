# Modeling a venue for the acoustic tracer (draft 2)

Draft 2, 2026-10-26 (draft 0 revised after two review rounds). How to build a room model in Blender so that the acoustic material table
(`md/acoustic_materials.md`) gives meaningful results. The rules are collected from the material page
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
- **When to use which.** Sheets for thin layers: panes, glazing units, boards, leaves, carpets, curtains, plaster,
  linings. Traced solids for thick bodies whose interior matters (sources or receivers inside, refraction offset),
  for porous bodies that are not plane layers (cushions, bass traps), for a porous fill behind a box (rule 10) and
  for boxes of cavity air. A sheet and its traced twin agree within the limits of page section 13.
- **Face orientation.** A face is hit on its front side when its vertices are counter-clockwise as seen from the ray;
  a front-side hit enters the body that owns the face. Keep normals pointing out of solids (check after an export).
- **Glue co-located faces exactly.** Hits closer than 1 mm along the ray are one event; near grazing, faces much
  closer than 1 mm in normal distance become two events with air between them (rule 12).
- **No overlapping geometry.** At most two co-located sheets form a stack; a third layer is a separate event
  (rule 13).

## 4. Panel size `side`

- `side` is √(panel area) of a mass-law material; it sets the panel's radiation and so its low-frequency
  transmission. It is a row value, not read from the geometry.
- Each mass-law row carries the √area of a typical panel of its use, stated in the material list (for example a
  wall of 4 × 2.6 m for concrete, a pane of 1.2 × 1.4 m for glass, a door of 0.9 × 2.0 m). The modeler does not
  change it, and the engine does not read the drawn panel: it sees one triangle, not the panel.
- **The drift is accepted.** Where a drawn panel differs from the typical size, the result drifts with it. Its size,
  as an example: a 4 mm pane read at 3.29 m instead of 1.349 m transmits 1.7 dB more at 125 Hz and 4.8 dB more at
  31.5 Hz (page 11.2). A modeler who needs a very different panel size uses a row made for it.
- **One `side` per bonded run** (rule 3): each body of a bonded run pays its increment at its own side, so mixed
  sides make the result depend on the direction.
- Leaves and wave media keep the infinite plate: `side` acts only on mass-law materials (page section 13).

## 5. Class 1: walls, floors, ceilings, windows

- **A rigid body carries the mass column and `tf`.** A rigid face without `tf` absorbs nothing and its plate passes
  nothing (rule 4); a rigid permittivity without `m` is a wave medium and tunnels as a sheet (rule 5). Keep rigid
  bodies lossless (rule 6). These are row properties; the modeler only picks the row.
- **One `att` per bonded run or glued stack** (rule 2): the loss is paid at every entry.
- **Bonded mass-law layers are one plate.** Plaster on brick or screed on a slab: the plaster or screed as a sheet on
  the solid (sheet | solid is exact). Avoid two traced mass-law bodies bonded together (up to 2.0 dB from the bonded
  plate at 16 kHz, stage-2b issue 3) and runs of three or more bonded bodies (within ±3.02 dB, page section 13).
- **Glazing units** are one lumped sheet; never two meshed mass-law sheets for a double unit (a mass-law sheet cannot
  resonate with a cavity, rule 9). Two separate sashes with a wide air space are two leaves on a box of cavity air.
- **Masonry by what you see.** Concrete has its own rows; other masonry takes `masonry_heavy` (solid brick, dense
  block, sand-lime), `masonry_hollow` (hollow brick and block) or `masonry_light` (aircrete, lightweight block), as
  a solid drawn at its thickness or as a round-thickness sheet on one plane.
- **Drywall by what the plan shows.** Pick `drywall` or `drywall_filled` (solid, wall drawn with thickness) or the
  sheet of the nearest total thickness (`drywall_100mm`, `_125mm`, `_150mm`, `drywall_filled_100mm`, ...). The rows
  assume one 12.5 mm gypsum board each side on metal studs, the rest cavity, empty unless filled. The precise
  option, for a build that matters: `gypsum_leaf_12_5mm` | box of cavity air (| a mineral-wool fill as a traced body
  at the back of the box, rule 10) | leaf; the studs are not modeled.
- **Suspended boards** **(proposed recipe)**: `gypsum_leaf_12_5mm` on a box of cavity air of the plenum depth.
- **Floors on joists** **(proposed recipe)**: `board_leaf_22mm` on a box of cavity air of the joist depth on the
  slab. A stage is modeled in detail from the same parts.
- **One board, two rows.** A board used as a panel takes its plate row (`gypsum_board_12_5mm`); used as a leaf on a
  box of air it takes its leaf row (`gypsum_leaf_12_5mm`, named `<material>_leaf_<NN>mm`).
- **Thickness sheets** exist at round thicknesses (`concrete_150mm`, `concrete_200mm`); draw the wall at its real
  thickness with the generic solid (`concrete`) where the geometry has thickness.

## 6. Class 2: hard furnishing

- **(proposed)** Open furniture boards (table tops, shelves, speaker cabinets) are sheets on the object's faces.
  A closed box of sheets is a hollow casing (each face one board): right for a TV or a speaker cabinet, wrong for a
  wall (section 3).
- **Hard closed furniture** (bookshelf, wardrobe, cupboard, fridge) is a hard box from the asset library (section 8).
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
- **Curtains** hang as a fabric sheet on the front face of a box of cavity air that reaches the wall or window.
  Pleats are not modeled; a pleated curtain uses a row with effective columns.
- **Small cushions and pillows:** a traced porous solid of the object's shape (`pu_foam`, `polyester_fiber_20mm`);
  the fabric cover is ignored. Sofas, armchairs, beds, mattresses and large cushions are porous bodies from the asset
  library (section 8).

## 8. Objects: hard boxes and porous bodies (classes 2–4)

Decided 2026-10-26; the per-object data and the fits are built in phase 3.

- **Use the assets.** Objects come as Blender assets with the material assigned: drag the asset into the scene,
  place it and scale it. Do not build object meshes with other rows.
- **Hard box** (bookshelf, wardrobe, cupboard and similar hard closed furniture): a closed box of the object's outer
  size whose faces absorb, per octave, the measured per-object absorption area divided by the box's surface. The
  box reflects at its faces and passes nothing of interest **(proposed)**. Scaling the asset keeps the face
  absorption, so the object's absorption area grows with its surface.
- **Porous body** (sofa, armchair, bed, mattress, large cushions): a low-poly shape or a cushion-like box of the
  `upholstery` row, fitted so that the body's absorption area matches the per-object data. It reflects at its face,
  absorbs inside and transmits some.
- **Never an air-like absorbing volume.** A volume with the impedance of air that only absorbs reflects nothing;
  it is not a model of furniture.
- **Clutter box.** Items on a desk or shelf are one box drawn over them with a clutter row: `clutter_hard_` or
  `clutter_soft_` with `sparse`, `medium` or `dense` (fill fraction about 10, 25, 40 %). Clutter mainly scatters,
  which is out of scope; the box gives its absorption and a little delay.
- **(proposed)** Plants: leave small ones out; a large plant is a porous body of its foliage outline. Clothes racks:
  a porous body around the hanging clothes. A person: a porous body of a low-poly human shape. A dense audience or
  rows of seats: a porous box over the seating area, of the height of seated persons. Dishes, glasses and clutter:
  leave out.
- Bodies reflect specularly where real objects scatter and diffract; scattering and diffraction are out of scope.
- Placement values (an absorber measured in a corner or on the floor) are no material property.

## 9. Class 5: acoustic treatment

- Porous panels: a sheet on the wall or ceiling solid. Panels stood off the wall: absorber sheet | box of cavity air |
  wall.
- Panel absorbers and microperforated panels: leaf sheet | box of cavity air (| porous fill as a traced body at the
  back of the box) | wall. Behind a fill the wall's row takes `tf` 0 (page 10.4).
- Bass traps **(proposed recipe)**: a porous body of the trap's shape in the corner, from the asset library.
- Suspended absorbing ceilings **(proposed recipe)**: tile sheet | box of cavity air of the plenum depth | slab.

## 10. Class 6: outdoor

- **Ground** is a porous half-space: a traced solid much thicker than the sound penetrates, of a ground row
  (`ground_grass`, `ground_soil`, `ground_gravel`, `ground_snow`, `ground_forest_floor`; proposed names). Below
  about 200 Hz the ground rows are a guess.
- **Asphalt** is a hard solid (`asphalt`); porous asphalt is a thin porous sheet (`asphalt_porous_40mm`) on it.
  **Ice** is a hard solid (`ice`); open water is `water_20c`.
- **Vegetation** (foliage of trees, hedges, shrubs) is a closed volume of the `vegetation` row drawn around the
  foliage: a medium matched to air that reflects nothing and attenuates along the path. This air-like absorbing
  volume is right for foliage, unlike for furniture (section 8).
- Scenes far from the origin: keep coordinates local (section 1).

## 11. Engine limits to work around

One line each, with the size measured.

| limit | size | what to do |
| --- | --- | --- |
| A porous layer on a rigid wall adds too much insulation at low frequency (stage-2b issue 1) | up to 8.1 dB at 31.5 Hz, 5.6 dB at 125 Hz, within 0.6 dB from 4 kHz; room-side reflection right within 6.3e-3 | do not count porous treatment toward the insulation between rooms below about 1 kHz |
| A band near grazing takes the first frequency's bounce decision (issue 2) | reflection up to 1.073 at 0.15; 8.3e-4 in octave absorption | `thin_slab_threshold` 0 |
| Two traced wave media back to back (issue 3) | 4.9 dB off at 16 Hz, transmission above one at 88° | room-side layer as a sheet, the rest traced |
| Two bonded traced mass-law bodies (issue 3) | up to 2.0 dB at 16 kHz (0.05 dB at 1 kHz) | the outer layer as a sheet |
| Three or more bonded bodies; the state is one level deep (page 13) | within +3.02 / −3.01 dB | lump them into one row or keep runs short |
| A sheet between two bodies, three co-located faces (page 13) | reflection 0.154 where the fluid layers give 0.994 | make the middle layer a traced body (rule 10) |
| Reversed face order of a co-located stack | reflection up to 0.999 different | check the face order after export |
| A free air gap between objects (rule 9) | up to 0.29 in reflection at 255 interactions | a box of cavity air |
| The first path through a layered wall is not the wall (page 13) | −4.87 dB for a lossless second layer at 85°; +1.37 dB after 20 interactions, +0.04 dB after 60 | allow enough interactions |
| The two sides of a lossy lining on a rigid wall differ (page 13) | 0.07–1.38 dB (dense), 0.52–2.70 dB (porous), 10° to 75° | read the result from the treated side |
| Leaves have no bending stiffness and keep the infinite plate (page 13) | coincidence not expressible (box windows: 2–3 kHz) | a mass-law row where coincidence matters |
| A curtain's relaxation in μ holds on the random-incidence average only (page 13) | — | do not read a curtain per angle |
| A source inside a medium with an unknown bend (rule 11) | 0.21 / 0.96 / 3.00 dB at 10° / 20° / 30° | seed the bent bit, use type 5 |
| Out of scope (page 13) | — | scattering, diffusion, edge diffraction, flanking through the building, level-dependent absorbers |
