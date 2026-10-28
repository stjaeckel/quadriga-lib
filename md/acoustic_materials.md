# Acoustic materials: the wish-list (draft 4)

Draft 4, 2026-10-26 (draft 0 revised after two review rounds and two independent reviews). This list is the input to
the work on the shipped acoustic material table (`data/materials_acoustic.csv`). The rows, their columns and their
tests are built later, material by material, from the entries below. Nothing here is a row yet, except where the
status says so. Everything still marked **(proposed)** is the drafter's proposal and waits for review: impact
ratings, typical panel sizes, sheet thicknesses chosen by the drafter, row names of objects, recipes for tier-3
materials and objects, the phase-3 start list.

The companion document `md/acoustic_modeling_guidelines.md` says how a venue is modeled for the acoustic tracer. Its
checklist of forbidden and limited configurations applies to every recipe below.

**Research row ids.** A research row is cited by its stable id `[<source key>#<n>]`, the column `id` of
`data/material_properties.csv` in the research library at its commit d501809 (2026-10-27); an id never changes and is
never reused. Every citation is converted; the library's `data/id_map_2dad906.csv` gives the id of a line of that file
at commit 2dad906 (2026-10-22, header = line 1), the form of earlier drafts.

## 1. Purpose, classes, tiers, rules

### 1.1 Purpose

A venue or a residential room, indoors or outdoors, is modeled in Blender. Materials are chosen mostly by informed
guess, and the simulation then optimizes speaker placement and acoustic treatment. The table must therefore offer,
for every surface a modeler meets, a row that is physically sound, named so that it can be picked from what the
modeler sees (a plan, a photo, a site visit) without reading the literature, and tested against the best reference
available. The quantity that moves the sound field inside a room is the room-side absorption; transmission matters
between rooms and for light boundaries at low frequency. This list says which rows are wanted, how each is used in a
model, what evidence backs each of the two quantities and what is missing.

### 1.2 The six material classes

1. **Structural building materials** ("empty rooms"): concrete, masonry, single-pane window glass, double glazing,
   ceiling boards, floor boards, floor screed, floor finishes, laminate flooring, metal sheets, drywall with an air or
   mineral-wool fill, linings, doors.
2. **Hard furnishing:** tables, chairs, wardrobes, cupboards, bookshelves, metal-clad furniture and appliances,
   casings of electronics (TV screens, speaker cabinets, lamps), sports equipment. A stage is modeled in detail from
   class-1 rows and has no entry here.
3. **Soft furnishing:** cushions and pillows as meshed 3-D shapes, blankets, carpets and rugs, curtains and blinds,
   mattresses, upholstery.
4. **Complex assemblies:** racks of clothing, couches, books in a bookshelf, plants, clutter, a single person and a
   dense audience, dishes and glasses.
5. **Acoustic treatment:** porous absorbers, wood-wool panels, bass traps, panel absorbers, microperforated panels,
   suspended absorbing ceilings, ceiling clouds and baffles, diffusers.
6. **Outdoor:** ground (grass, soil, gravel, snow, forest floor), asphalt, vegetation.

Auxiliary rows that belong to no class (the cavity air of a box, water) are listed in section 2.7.

### 1.3 Evidence tiers: one for absorption, one for transmission

Every entry carries two tiers: **A** for room-side absorption and **T** for transmission.

| tier | meaning |
| --- | --- |
| 1 | A measured reference of that quantity exists, with the row's own build and a known coefficient type. For absorption: a random-incidence or normal-incidence reference; a hard face in that band (up to about 0.2, used as measured: material page, section 10.5); a per-object area used like for like (2.4); or a Sabine value of a class whose conversion is decided (porous layers of known construction; mineral wool by the edge correction). A leaf on a box takes A 1 where its construction's reference is a hard face. |
| 1p | **Tier 1 pending conversion:** the absorption reference is a Sabine (chamber) value of a class whose conversion is not decided (panel absorbers, microperforated panels, hanging fabrics, green walls, per-area values of seat and audience blocks). Printed, not asserted, until the conversion is decided; that decision comes first in phase 3. |
| 2 | First-principles physics gives the columns (plate theory, a transfer matrix, a porous or ground model at a known flow resistivity, a leaf on a box of air); references are absent, of a neighbor build (printed), or compilation values of unstated type (printed). |
| 3 | An informed guess is the best available; the guess and its basis are stated in the case. |
| — | the quantity does not apply (an object's transmission, a volume's absorption) |

**Conversion error budget.** Converted absorption carries the conversion's own error beside the ±0.05: Jeong's
conversion with Thomasson's field impedance reached an rms of 0.031 (maximum 0.057) on three materials with an
independent random-incidence reference; the edge-effect correction 0.041 (engine notes, "Rules for absorption
references").

Compilation values (Cox and D'Antonio Appendix A, the pyroomacoustics library) whose coefficient type is not stated
do not lift a material to tier 1. PTB rows are typed Sabine; they count where they are hard faces of the row's own
build.

### 1.4 Impact and work order

How much the material moves the simulated sound field of a typical room or site **(proposed ratings throughout)**:

- **high:** covers a large share of the surface in most rooms or of an outdoor site, or dominates the low frequencies
  where a room's modes are, or is the main absorber of a furnished room;
- **medium:** common but smaller in area, or large in area but close to a material already in the table;
- **low:** small objects, rare items, or items below the wavelengths that matter.

**Work order (decided):** concrete first as the rig check; then by the absorption tier, then by impact, then by the
transmission tier. Every relevant material goes in, each with its diagnostics, before the engine is reopened; engine
findings are recorded as stage-2b issues in the meantime. A complete table matters more than the exact order.

### 1.5 Status values

| status | meaning |
| --- | --- |
| wish | listed; nothing built |
| in work | a row or its case exists and is being built or redone |
| confirmed | the row is in the CSV and its case passes against its references |
| recorded miss | the engine cannot express the material; the miss and its size are recorded in `md/acoustic_engine_stage2b_issues.md` |
| miss candidate | expected to be a recorded miss; measured in phase 3 before it is recorded |

### 1.6 How rows are named

- **The thickness suffix marks a sheet.** A row whose name ends in `_<NN>mm` is a sheet of that thickness; the
  suffix always comes last. A row without it is a traced solid, whose thickness comes from the geometry. A decimal
  point becomes an underscore. Examples: `concrete` (solid), `concrete_150mm` (sheet); `parquet` (solid),
  `parquet_15mm` (sheet); `gypsum_leaf_12_5mm` (sheet).
- **Generic solid and round sheets.** Sheet rows exist at round thicknesses. A measured reference is compared with
  the sheet of the nearest round thickness, the difference stated in the case, up to about 10 % of the thickness
  **(proposed limit)**; a sheet with no reference within that distance is printed (tier 2). Product thicknesses keep
  their size (`gypsum_board_12_5mm`, `plywood_19mm`). Finished walls are measured including their plaster.
- **Literature constructions are test references, not rows.**
- **One board, two rows.** A board used as a mass-law panel and as a wave-medium leaf on a box of air has two rows:
  the plate row (`gypsum_board_12_5mm`) and a leaf row `<material>_leaf_<NN>mm` (`gypsum_leaf_12_5mm`,
  `steel_leaf_1mm`).
- **Names say what a modeler sees:** `curtain_flat_*`, `curtain_pleated_*`; `masonry_heavy`; `drywall_filled_125mm`.
- **Every row is named,** object rows included.
- **No combination rows.** Walls and floors are modeled first; rugs, curtains and absorbers are placed on them as
  separate rows. A combination is a rig test, and only where a measurement of the combination exists (section 3).
  Drywall and masonry rows are no combinations: each is the finished wall a modeler sees.
- **Reader rules** (`src/obj_file_read.cpp`, `base_material_name` and `resolve_csv`): the Blender material name is
  looked up exactly, then by its part before the first dot; there is no other fallback. Names are case-sensitive,
  snake_case, without dots, so that Blender's duplicates (`concrete.001`) map to the row `concrete`. A name that
  matches no row becomes "no material" (index 0, air) without a warning unless the reader is called with
  `csv_strict`.

### 1.7 Rules for designing rows

1. **Sheet rows are the generic row's columns plus `thk`.** Drawing a wall as one plane with the sheet or as a box
   with the solid gives the same wall (the twin assertion of the rig).
2. **Lumped rows carry the visible thickness in `thk`** (decided). The surface mass of a mass-law row is set by
   `fRef` with `thk`, and `fRef` is also the reference frequency of every power law (`att`·(f/`fRef`)^`attB`,
   `tf`·(f/`fRef`)^`tfB`, and so for `a`, `c`, `e`, `g`, `alpha`: `src/ray_mesh_interact.cpp:600`, :612, :41–43). So
   `thk` × k and `fRef` × k, **and every power-law value at `fRef` rescaled by k^exponent**: `tf`·k^`tfB`,
   `att`·k^`attB`, `alpha`·k^`alphaB`, `a`·k^`b`, `c`·k^`d`, `e`·k^`f`, `g`·k^`h`. Then every column is unchanged at
   every frequency; the absolute frequencies (`resF`, `coiF`, `tfF`) stay. Example (the page 11.4 row of double glazing
   4-16-4 at 342.76271 m/s, k = 3): scaling `fRef` alone multiplies `tf` by 1.4851 and `att` by 0.5553, so the absorption at 125 Hz
   goes from 0.1124 to 0.1670 and `att` at 8 kHz from 16.29 to 9.05 dB. A mass-law sheet's `thk` displaces nothing of
   the host medium (`src/ray_mesh_interact.cpp:1320–1324`); wave-medium sheets keep their true thickness, which the
   engine takes back out of the host. The first lumped case asserts that the rescaled row equals the original on the
   grid. Every lumped entry states its surface mass.
3. **Glued rows carry `att` 0 and the host's `side`** (page rules 2 and 3).
4. **A layer that carries another layer or an object is a traced solid.** Hard floor finishes (parquet, vinyl,
   ceramic tile) are thin traced solids on the slab wherever a rug lies or an object stands on them; their sheet rows
   serve only where nothing lies or stands on them. A rug is a sheet on the finish solid. Three co-located faces (any
   three faces in one plane) are not resolvable by the engine (`src/ray_mesh_interact.cpp:2617`: "At most two
   co-located faces form a stack; a third layer is a separate event"); allowed are a sheet on a solid and a
   sheet | sheet pair with air behind.
5. **A bonded floor is two bodies.** Slab and bonded screed are drawn as one solid of their full thickness (screed
   2000 kg/m³, [Hopkins_2007_SoundInsulation#172], is close to concrete), the finish solid on top. Three bonded mass-law
   bodies are within ±3 dB (stage-2b issue 4). The `screed` rows stay for bare and floating screeds.
6. **Behind a fill the wall keeps its own row.** The rig plan measured 2.8e-3 in octave absorption for the wall's own
   `tf` against the pad referred to air; the engine notes (9.2) record 0.040 in R off that pad for a board on 25 mm of
   fill as an open convention. The page's "behind a fill the wall takes `tf` 0" (section 10.4) goes on the phase-4
   list of page corrections (1.12).
7. **One material per closed body; finishes as sheets** (to be confirmed in phase 3). A glass front on a metal box is
   a glass sheet glued on that face; it then carries a window's room-side absorption, which includes what a window
   passes outdoors; phase 3 uses a hard-face glass value for such fronts **(proposed)**.
8. **Closed casings are hard boxes (traced solids),** never closed boxes of mass-law sheets: a hollow casing holds a
   free air space (page rule 9: up to 0.29 in reflection at 255 interactions). Sheets are for open single panels.
9. **One measurement never fits two rows that are stacked in one model.** Alternative models of one construction (a
   lumped row and its recipe) may share a reference.
10. **A mass-law row carries its density in `e`** (`e` = ρ/ρ₀): the rig shifts measured walls of another thickness by
    the surface mass `e`·ρ₀·`thk`. A lumped row's case passes its stated surface mass instead.

### 1.8 Panel size `side`

`side` is a row value: the √area of a typical panel of that use. There is no engine change: the engine sees one
triangle, not the panel. Where a drawn panel differs from the typical size, the drift is accepted (guidelines,
section 4). `side` acts only on mass-law rows. A reference is evaluated at its own specimen side, the row carries the
typical size. Every CSV cell is filled. Typical sizes **(proposed)**, √(width × height) computed by script:

| rows | typical panel | `side` |
| --- | --- | ---: |
| `concrete*`, `masonry_*`, `drywall*`, `gypsum_board_12_5mm`, `clt*`, `screed*`, slabs and ceilings | wall 4 × 2.6 m | 3.22 m |
| glued finishes (`plaster_15mm`, `parquet*`, `ceramic_tile*`, `vinyl*`) | the host's (written as 3.22 m) | 3.22 m |
| `glass_*mm`, `glazing_*` | pane 1.2 × 1.4 m | 1.30 m |
| `plywood_19mm` | sheet 2.6 × 1.2 m | 1.77 m |
| `door_*` | door 0.9 × 2.0 m | 1.34 m |
| `steel_1mm`, `aluminum_2mm` | appliance side 0.6 × 1.8 m | 1.04 m |
| `furniture_board*` | table top 1.6 × 0.9 m | 1.20 m |

A floor of 4 × 5 m would be 4.47 m; on a heavy slab the effect is small (page 11.1: the 152 mm wall at the default
3.29 m against its side 3.52 m reads 0.04–0.14 dB higher from 63 Hz).

### 1.9 Notation

- **sheet | solid:** a sheet glued on the face of a traced solid, room side first (page section 7).
- **sheet | sheet:** two sheets in one plane with air behind (page section 2).
- **box of cavity air:** a closed box of `cavity_air` glued exactly between a leaf and a wall; its bounce series is
  summed in one event (page 10.4).
- **finish solid:** a thin traced solid of a hard floor finish on the slab solid.
- **hard box:** a closed traced solid of a rigid row whose faces absorb a per-object area (2.4.1).
- **porous body:** a closed traced solid of a porous row (2.4.2).
- **absorbing volume:** a closed traced solid of a medium matched to air with an `alpha` column (dB/m along the
  path); it reflects nothing and is not taken for air, since `Material::same_as` compares `alpha`
  (`src/ray_mesh_interact.cpp:635`).

### 1.10 How phase 3 tests a row family

- **One test case per row family** (decided): a generic row and its sheets share one lean case.
- **Procedure (decided):** measure first; judge against the known range of measurements and models; if the result
  needs an engine fix, mark it as an engine limit; if not, assert. A comment in the case states what "reasonable" meant: the range
  of measurements and models it was judged against. The rig documents itself; this list is updated from the result.
- **Engine-limit marks (decided; named since 2026-10-27):** a rig comparison that misses because of an engine limit
  carries a comment `ENGINE LIMIT (<name>):` with the name of the stage-2b issue (code and tests cite no document or
  issue number) and asserts nothing; the measured and the reference values go into the
  diagnostics (for a per-node comparison the largest deviation and its node stand for the reference); the issue has
  an entry in `md/acoustic_engine_stage2b_issues.md`. Known engine issues never make the suite fail. The transmission of a box of air in front of a fill is taken from the traced paths at a cap of 10
  interactions (production runs: 20 to 30); reflection runs to the engine's limit, its change at the cap is printed.
- **Tolerances (decided; may move with more data):**

  | check | against | tolerance | assertion |
  | --- | --- | --- | --- |
  | independent physics | the model written in the test (plate, transfer matrix, porous layer) | at or above 2–3 × the measured maxima of the rig's self-checks (wide gates are preferred) | CHECK |
  | measured single reference | the converted reference at its own specimen side | ±3 dB / ±0.05 (plus the conversion budget) | CHECK, or printed as an engine limit |
  | class row | the envelope of the class's walls, each shifted to the evaluated thickness by the mass law (20·log₁₀ of the surface-mass ratio; the row's surface mass is `e`·ρ₀·`thk`, a lumped row's the stated one) | ±1 dB | as above |
  | regression guard | the library's own measured result | about 1 dB / 0.01 in α | CHECK |

  Worked example: `concrete_150mm` against Litvin 76-66 ([Litvin_1980_ConcreteWallsTL#1], 152 mm, opening 4.37 × 2.84 m,
  side 3.523 m by script): the row is evaluated at side 3.523 m, the 2 mm difference shifts the reference by
  20·log₁₀(152/150) = 0.115 dB (script), and each octave from 125 Hz to 4 kHz is asserted within ±3 dB; 31.5, 63 Hz and
  8 kHz take the model spread.
- **Thickness range (decided):** a generic row is asserted where the trend miss of its class stays inside the
  tolerance, and printed outside. The range is stated in the column "valid".
- **Rig prerequisites** (built 2026-10-27): a thickness argument for rows with `thk` 0; a specimen side and surface
  mass per measured reference and the mass-shifted class envelope; a 0° comparison; a sheet glued at a traced solid's
  exit; a stack rule that refuses three co-located faces; one case per family (`Material - <family>`, several
  families joined by ` + `; the family is the name without `_<NN>mm`); Miki, Komatsu and Delany-Bazley written in the
  test; a helper that rescales a lumped row with `fRef`; the object test (a body on a rigid floor, plane waves from 16
  directions of the upper hemisphere, about 65 rays each, octave tones 125 Hz–4 kHz, at most 10 interactions and 3
  reflections, −60 dB floor: within 3 % of α·S for a hard box, equal to the chords for an absorbing volume, 0.24 s
  CPU for three bodies).

### 1.11 Phase-3 start list (proposed)

In the decided order (concrete first, then absorption tier, impact, transmission tier), with what each needs first:

| # | family | A / T | needs |
| --- | --- | --- | --- |
| 1 | `concrete` | 1 / 1 | thickness argument; specimen side |
| 2 | `masonry_heavy` | 1 / 1 | thickness argument; mass-shifted envelope; fit to the brick series |
| 3 | `masonry_hollow` | 1 / 1 | as 2 |
| 4 | `glass_wool` | 1 / 2 | thickness argument (solid); Komatsu written in the test |
| 5 | `gypsum_leaf_12_5mm` and `drywall` | 1 / 1–2 | sheet at a solid's exit (precise recipe) |
| 6 | `board_leaf_22mm` | 1 / 2 | leaf on a box (exists) |
| 7 | `carpet_tile_11mm` | 1 at 63 Hz and to 500 Hz / 2 | sheet on a solid (exists) |
| 8 | `parquet`, `vinyl` | 1 / 2 | thickness argument; sheet \| sheet \| solid self-check and the stack rule |
| 9 | `masonry_light` | 1 / 1 | as 2 |
| 10 | `rock_wool`, `pu_foam_50mm` | 1 / 2 | as 4 (Miki written in the test) |

### 1.12 Page corrections for phase 4

- `tf` 0 behind a fill (page 10.4): the wall keeps its own row (1.7.6).
- Miki or Komatsu for porous columns (page 10.5): ground rows are Miki fits to the class impedance (2.6).
- 343 m/s (page 10.1 and the examples): the library's 342.76271 m/s.
- The two-face limit (page section 2): two sheets form a stack only with air behind; never two sheets on a solid.

## 2. The materials per class

Each table gives: name, intended use, recipe, **A** and **T** tiers, **valid** (the thickness and band range where the
row is asserted; printed outside), impact, status. The references and gaps follow per entry below the table: key,
id, coefficient type, and **(row)** where they test the row or **(comb.)** where they are combination references
of section 3. R = laboratory sound reduction index; σ = flow resistivity.

### 2.1 Class 1: structural building materials

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | `concrete`, `concrete_150mm`, `concrete_200mm` | cast walls, slabs, ceilings; slab with bonded screed | solid of the full thickness (slab plus bonded screed) or sheet | 1 | 1 | 150–203 mm; R 125 Hz–4 kHz | high | confirmed |
| 1.2 | `masonry_heavy`, `masonry_heavy_150mm`, `_200mm`, `_250mm` | finished heavy wall plastered both faces (solid clay brick, dense block, sand-lime, about 1700–2400 kg/m³) | solid or sheet | 1 | 1 | about 140–250 mm (proposed from the trend; 250 mm decided); R 125 Hz–4 kHz; 63 Hz printed | high | wish |
| 1.3 | `masonry_hollow`, `masonry_hollow_100mm`, `_200mm`, `_250mm` | finished hollow-unit wall plastered both faces (hollow clay brick, hollow block of any aggregate) | solid or sheet | 1 | 1 | about 90–250 mm, set in phase 3 by the same rule; measured plastered both faces at 166, 206, 240 and 270 mm | high | wish |
| 1.4 | `masonry_light`, `masonry_light_100mm`, `_150mm` | finished light wall, plastered (aircrete, expanded-clay and lightweight-aggregate solid units) | solid or sheet; 200 and 250 mm printed only | 1 | 1 | about 90–190 mm; 100 mm against the four bare 10 cm walls | medium | wish |
| 1.5 | `plaster_15mm` | plaster on `concrete` or `clt` | sheet on the solid; `att` 0, host's `side` | 1 | 2 | — | low | wish |
| 1.6 | `glass_4mm`, `glass_6mm` | single-pane windows | sheet only | 2 | 1 | R 125 Hz–4 kHz | high | wish |
| 1.7 | `glazing_double_24mm`, `glazing_triple_36mm` | sealed units 4-16-4 (20 kg/m²) and 4-12-4-12-4 (30 kg/m²) | sheet, one lumped body | 2 | 1 | R 125 Hz–4 kHz | high | wish |
| 1.8 | `glass_leaf_4mm` | two sashes with a wide air space (box windows) | leaf sheet \| box of cavity air \| leaf (sheet at the box's exit) | 2 | 1 | — | low | wish |
| 1.9 | `gypsum_board_12_5mm` | a single free gypsum panel; the reference row of the leaf | sheet | 2 | 1 | R 63 Hz–4 kHz | low | wish |
| 1.10 | `gypsum_leaf_12_5mm` | gypsum ceilings over a plenum, linings, the precise drywall recipe | leaf sheet on a box of cavity air | 1 (empty cavity); 1p (filled) | 2 | — | high | wish |
| 1.11 | `drywall`, `drywall_filled`; `drywall_100mm`, `drywall_125mm`, `drywall_150mm`; `drywall_filled_100mm`, `drywall_filled_125mm`, `drywall_filled_150mm` | stud partitions as a plan shows them | lumped mass-law row of the standard build (one 12.5 mm gypsum board each side, metal studs, the rest cavity, empty unless filled; boards 17.0 kg/m² from 680 kg/m³, [Hopkins_2007_SoundInsulation#135]); `thk` the visible thickness | 1 (empty); 1p (filled) | 1 for `drywall_filled_125mm` and the solid at 115 mm; 2 (double-wall model) for the rest | filled 115 mm | high | wish |
| 1.12 | dry lining (no row: a recipe) | gypsum board on dabs or on furring over masonry | `gypsum_leaf_12_5mm` \| box of cavity air (dab or furring depth) \| masonry solid | 1p | 2 | — | medium | wish |
| 1.13 | `plywood_19mm` | free wooden panels | sheet | 1 | 1 | R 125 Hz–4 kHz | low | wish |
| 1.14 | `wood_leaf_12mm` | wood panelling on battens | leaf sheet on a box of cavity air (batten depth) | 2 | 2 | — | medium | wish |
| 1.15 | `board_leaf_22mm` | floor boards on joists | leaf sheet on a box of cavity air (joist depth) on the slab solid; nothing lies or stands on it in the model | 1 | 2 | — | high | wish |
| 1.16 | `parquet`, `parquet_15mm` | parquet glued on a slab | finish solid on the slab solid; the sheet only where nothing lies or stands on it | 1 | 2 | — | medium | wish |
| 1.17 | `laminate_leaf_8mm` | floating laminate on a foam underlay | leaf sheet \| `underlay` solid \| slab solid; nothing lies or stands on it in the model | 2 | 2 | — | medium | wish |
| 1.18 | `screed`, `screed_50mm` | a bare screed; the screed of a floating floor | solid, or the sheet for a bare screed | 3 | 2 | — | low | wish |
| 1.19 | floating screed (recipe; or a lumped row: 4.6) | screed on a resilient layer | `screed` solid \| resilient layer \| slab solid | 3 | 2 | — | medium | wish |
| 1.20 | `ceramic_tile`, `ceramic_tile_10mm` | tiled floors and walls | finish solid; the sheet only where nothing lies or stands on it; `att` 0, host's `side` | 3 | 2 | — | low | wish |
| 1.21 | `vinyl`, `vinyl_3mm` | linoleum or vinyl stuck to the slab | finish solid; the sheet as 1.20 | 1 | 2 | — | low | wish |
| 1.22 | `steel_1mm`, `aluminum_2mm` | metal sheets, metal doors, cladding | sheet | 2 | 1 | R 63 Hz–4 kHz | low | wish |
| 1.23 | `door_solid_40mm` | solid timber doors | sheet, lumped; surface mass 23.9 kg/m² (the reference door; 22.5–25.2 kg/m² over the references) | 1 | 1 | R 125 Hz–4 kHz (sealed leaf) | medium | wish |
| 1.24 | `door_hollow_core_40mm` | hollow-core interior doors | sheet, lumped; surface mass 9.3 kg/m² (NBS stated) | 3 | 1 | R 125 Hz–4 kHz (sealed leaf) | medium | wish |
| 1.25 | `clt`, `clt_100mm` | cross-laminated timber walls and floors | solid or sheet | 2 | 1 | 78–245 mm; R 63 Hz–4 kHz | low | wish |

References and gaps:

1.1 **Concrete.** (row) R: Litvin [Litvin_1980_ConcreteWallsTL#1] (152 mm, 346.7 kg/m²), [Litvin_1980_ConcreteWallsTL#2]
(203 mm, 471.6 kg/m²); Mahn [Mahn_2023_GuideAirborneRR331#8], [Mahn_2023_GuideAirborneRR331#10]. α: PTB#30
[PTB_2012_AbsorptionTable#26] (Sabine, hard face), Nell [Nell_TIP_TieftonAbsorber#4] (chamber boundary, from 31.5 Hz).
Constants: Hopkins [Hopkins_2007_SoundInsulation#58], [Hopkins_2007_SoundInsulation#59],
[Hopkins_2007_SoundInsulation#60]; Mahn η [Mahn_2023_GuideAirborneRR331#9], [Mahn_2023_GuideAirborneRR331#11].
[Litvin_1980_ConcreteWallsTL#2] minus [Litvin_1980_ConcreteWallsTL#1]: 1.7 / 0.8 / 2.1 / 1.6 / 1.6 / 2.7 dB
(125 Hz–4 kHz), against 2.67 dB from the mass law of the two surface masses (script; 2.51 dB is the thickness ratio).
**Built (2026-10-27):** one generic solid and its two sheets, density 2281 kg/m³ (`e` 1901, Litvin 76-66), the face
absorption one power law over PTB#30 and Nell (0.0204 at 1 kHz, exponent 0.3965), the mass-law knobs fitted at
342.76271 m/s to the mass-shifted Litvin pair at 150 and 203 mm. At side 3.523 m the row is at most 0.27 dB outside the
pair's envelope at both thicknesses, within 0.74 dB of 76-66 and 2.14 dB of 76-77 (250 Hz, the spread of the pair
itself); α within 0.015 of PTB#30 and 0.002 of Nell. `coiF` (4.4 kHz) and `att` are fit knobs: Hopkins' fc·h of
17.1 Hz·m ([Hopkins_2007_SoundInsulation#58]) gives 114 Hz at 150 mm, and a plate with the physical coincidence and one
constant loss factor misses the pair by 7–11 dB, because the measured loss factor falls from 0.04 to 0.007 (Mahn
[Mahn_2023_GuideAirborneRR331#9], [Mahn_2023_GuideAirborneRR331#11]; stage-2b issue 7). Mahn
[Mahn_2023_GuideAirborneRR331#8], [Mahn_2023_GuideAirborneRR331#10] (thirds at the octave centres, CON150 of unknown
size) are printed only: 3 dB below the pair at 250 Hz, 9 dB above at 4 kHz. Gaps: 63 Hz R; R outside 150–203 mm.

1.2 **Heavy masonry.** References (decided 2026-10-28): walls measured plastered on both faces are the references where
they exist; only where none exists, the bare walls of the class plus the measured plaster increment, with its spread,
for each missing plaster face. The case prints the miss of the rule "bare wall plus one increment per face" against
Weise as a measured finding. (row) plastered both faces: Weise [Weise_2003_PruefstandAbweichungen#1] (sand-lime blocks
240 mm, thin gypsum plaster both faces, 250 mm finished, 440 kg/m²: 1760 kg/m³ by script; each third the mean of 12
laboratory curves, computed by the review and taken as measured (decided); from 63 Hz, spread between the laboratories
31.6–47.5 dB at 63 Hz and 35.0–47.5 dB at 125 Hz; specimen size not stated; no 8 kHz); Hopkins
[Hopkins_2007_SoundInsulation#68] (dense block 215 mm plastered both faces, 430 kg/m², 241 mm finished: 1784 kg/m³ by
script); printed: [Weise_2003_PruefstandAbweichungen#2] (the same wall of frost-damaged units, three further
laboratories, +0.5 to +2.7 dB); Secchi [Secchi_2003_LaterizioLabInSitu#1] (lightweight clay block with mortar-filled
holes, 21 cm finished, 400 kg/m², 1905 kg/m³ by script, from 100 Hz: a heavy wall by density at 210 mm, printed here and
in 1.3). (row) plastered one face: Hopkins [Hopkins_2007_SoundInsulation#67] (100 mm, 200 kg/m², 113 mm: 1770 kg/m³);
BIA [BIA_2000_TN5AClayMasonry#1] (brick 92 + 13 mm one face). (row) bare walls: BIA [BIA_2000_TN5AClayMasonry#3]
(95 mm), [BIA_2000_TN5AClayMasonry#7] (SCR brick 140 mm), [BIA_2000_TN5AClayMasonry#12] (203 mm),
[BIA_2000_TN5AClayMasonry#6] (241 mm), [BIA_2000_TN5AClayMasonry#11] (305 mm), surface masses 189.0–569.8 kg/m² (data
text); Warnock [Warnock_1984_ConcreteBlockWalls#1] (90 mm solid concrete blocks, 195.2 kg/m²). Plaster increment of one
face (matched pairs, script, 125 Hz–4 kHz): [BIA_2000_TN5AClayMasonry#8] minus [BIA_2000_TN5AClayMasonry#7] = +2.5 /
+1.1 / +1.5 / +2.0 / +1.5 / +0.0 dB; [BIA_2000_TN5AClayMasonry#14] minus [BIA_2000_TN5AClayMasonry#15] = +2.7 / +1.7 /
+2.5 / +3.0 / +1.6 / +1.0 dB; Warnock [Warnock_1984_ConcreteBlockWalls#5] minus [Warnock_1984_ConcreteBlockWalls#4]
(290 mm hollow normal-weight block, the other face sealed, 337 → 352 kg/m²; the added mass alone gives 0.38 dB) = +1.7 /
+1.7 / +2.3 / +2.6 / +1.8 / +2.6 dB. The bare brick series rises per doubling of surface mass 7.41 / 8.60 / 9.60 / 9.13
/ 7.43 / 5.81 dB (least squares over [BIA_2000_TN5AClayMasonry#3], [BIA_2000_TN5AClayMasonry#7],
[BIA_2000_TN5AClayMasonry#12], [BIA_2000_TN5AClayMasonry#6], [BIA_2000_TN5AClayMasonry#11], surface masses from the data
text), against 6.02 dB for the engine's mass term; walls scatter 2.5–4.0 dB about the trend, and a fit at the centre
misses the ends by up to 2.85 dB at 500 Hz. Measured finding (script, 125 Hz–4 kHz):
[Weise_2003_PruefstandAbweichungen#1] lies on the bare trend at its own total mass (+1.8 / −0.5 / +0.6 / +2.0 / +1.9 /
−0.7 dB) and below the trend plus two mean one-face increments by −2.8 / −3.5 / −3.6 / −3.1 / −1.3 / −3.1 dB, beyond
±3 dB at 250 Hz–1 kHz: the increments were measured with 13–16 mm sanded plaster and overstate a thin plaster. Against
[Hopkins_2007_SoundInsulation#68] it reads +3.0 / +1.4 / +0.2 / +1.3 / −0.9 / −3.9 dB, where the mass law gives 0.20 dB.
[Secchi_2003_LaterizioLabInSitu#1] lies on the bare trend at its mass: +3.4 / +1.0 / −1.7 / −1.2 / −0.9 dB
(125 Hz–2 kHz). Hence the asserted range of about 140–250 mm; 95–140 and 250–305 mm are printed. Walls of nearly equal
mass differ widely: [BIA_2000_TN5AClayMasonry#6] minus [BIA_2000_TN5AClayMasonry#12] = 5.3 / 7.4 / 8.4 / 9.2 / 6.2 /
5.5 dB against 1.07 dB from the mass law (script). Starting columns come from a fit to the series, not from the page
rows (whose `att` and `coiF` differ wall by wall). Printed checks: NBS [NBS_1955_BMS144SoundInsulation#33] to
[NBS_1955_BMS144SoundInsulation#38] (clay brick 100 and 200 mm and on edge, 16 mm plaster both faces, tested 1926–1928,
warble tones 256–2048 Hz, side 1.74 m, masses 154–474 kg/m² weighed with the frame; the 200 mm walls
[NBS_1955_BMS144SoundInsulation#35] to [NBS_1955_BMS144SoundInsulation#37] read 1.9–10.5 dB below the trend plus two
increments at 500 Hz–2 kHz, script). 63 Hz: [Weise_2003_PruefstandAbweichungen#1] only, printed (model spread). α:
plaster PTB#344 [PTB_2012_AbsorptionTable#38], PTB#346 [PTB_2012_AbsorptionTable#40] (Sabine, hard face). Density:
Hopkins [Hopkins_2007_SoundInsulation#32], [Hopkins_2007_SoundInsulation#70]. [BIA_2000_TN5AClayMasonry#10] is a lined
wall ([BIA_2000_TN5AClayMasonry#10] minus [BIA_2000_TN5AClayMasonry#12] = −0.3 / −1.4 / 8.4 / 9.9 / 8.6 / 5.4 dB) and
belongs to 1.12. Gaps: third-octave R of solid clay brick plastered both faces (NBS tones only); bare sand-lime and
sand-lime of 115–175 mm; R below 100 Hz other than the sand-lime wall; the specimen size of Weise.

1.3 **Hollow masonry.** Hollow = cored units of any aggregate; light = solid units of aircrete, expanded clay or
lightweight aggregate. (row) plastered both faces: Warnock [Warnock_1984_ConcreteBlockWalls#3] (140 mm hollow
normal-weight block, 13 mm sand plaster both faces, 166 mm finished; from 125 Hz; specimen size not stated) with its
bare wall [Warnock_1984_ConcreteBlockWalls#2] (about 199 kg/m² from the block mass, mortar not counted): +5.4 / +6.5 /
+5.9 / +5.8 / +5.3 / +5.5 dB (script), mostly by sealing the porous units (about 0.6 dB by mass, source). `_200mm`
(decided): Granzotto [Granzotto_2019_ClayHollowBrickWalls#2] (wall B, 206 mm, 321 kg/m², 37 % voids, 1558 kg/m³ by
script; plaster thickness not stated; thirds 100–5000 Hz from a figure). `_250mm` (reinstated, decided): Granzotto
[Granzotto_2019_ClayHollowBrickWalls#3] (wall C, 240 mm, 305 kg/m²), [Granzotto_2019_ClayHollowBrickWalls#5] (wall E,
270 mm, 359 kg/m²), 0.96 and 1.08 times 250 mm (script). Granzotto B minus (Zeitler [Zeitler_2015_ConcreteBlockRR334#3]
plus two mean one-face increments of 1.2) = +1.9 / +3.2 / −0.3 / −1.0 / −0.6 / −2.5 dB, against 2.60 dB from the mass
law (script). Printed: Secchi [Secchi_2003_LaterizioLabInSitu#1] (wall A, 18 cm lightweight clay block with
mortar-filled holes, 21 cm finished, 400 kg/m², 1905 kg/m³ by script: a filled core, offered to 1.2 as a heavy wall at
210 mm; on the same basis as wall B +3.0 / +3.5 / +0.1 / +0.1 / +1.0 dB, against 4.51 dB),
[Secchi_2003_LaterizioLabInSitu#2] to [Secchi_2003_LaterizioLabInSitu#4] (28–45 cm finished, 371–470 kg/m²),
[Secchi_2003_LaterizioLabInSitu#5] (double wall with a 3 cm cavity); all from a figure, thirds 100–3150 Hz, 15 mm
plaster both faces. Field values (`R_apparent_dB`) are printed only (decided): Secchi [Secchi_2003_LaterizioLabInSitu#6]
to [Secchi_2003_LaterizioLabInSitu#10] (the same walls in a test building) read 0.3–9.7 dB by band and 3–6 dB in Rw
below the laboratory (script), by flanking, which the engine does not trace. Printed further: Granzotto
[Granzotto_2019_ClayHollowBrickWalls#1], [Granzotto_2019_ClayHollowBrickWalls#4],
[Granzotto_2019_ClayHollowBrickWalls#6], [Granzotto_2019_ClayHollowBrickWalls#7] (280–330 mm, 327–377 kg/m²); NBS hollow
tile, cinder and pumice walls [NBS_1955_BMS144SoundInsulation#39] to [NBS_1955_BMS144SoundInsulation#57] (1926–1941); Rw
pairs Oliveira [Oliveira_2021_CeramicBlockCoating#1] to [Oliveira_2021_CeramicBlockCoating#14] (seven 140 mm walls bare
and rendered, Rw only). (row) bare walls: Zeitler [Zeitler_2015_ConcreteBlockRR334#3] (190 mm normal-weight, from
63 Hz), [Zeitler_2015_ConcreteBlockRR334#2] (140 mm hollow lightweight aggregate, without leakage;
[Zeitler_2015_ConcreteBlockRR334#1] is the same wall with the leakage of unsealed faces, printed); Warnock
[Warnock_1995_BlockConstructions#1]; BIA [BIA_2000_TN5AClayMasonry#13] (194 mm hollow clay tile),
[BIA_2000_TN5AClayMasonry#15] (90 mm). [Warnock_1995_BlockConstructions#1] minus [Zeitler_2015_ConcreteBlockRR334#3]:
−0.6 / 3.2 / 0.0 / 0.2 / −0.5 / 1.3 dB (script). The increment of two plaster faces on porous hollow units (+5.3 to
+6.5 dB) is about twice the one-face increments of 1.2 and is the class's own. Starting columns: a fit to these walls.
α: plaster as 1.2. Gaps: hollow clay walls of 80–200 mm plastered both faces (spectra exist for 206–450 mm only);
specimen sizes of Secchi and Granzotto; Oliveira's spectra (not extracted).

1.4 **Light masonry.** (row) plastered: Hopkins [Hopkins_2007_SoundInsulation#4] (aircrete 115 mm + 13 mm plaster one
side, 71 kg/m², 128 mm: 555 kg/m³ by script); Warnock [Warnock_1995_BlockConstructions#2] (190 mm lightweight block,
plastered), [Warnock_1995_BlockConstructions#4] (90 mm wood-fiber aggregate, plastered). (row) bare at 100 mm (decided):
Granzotto [Granzotto_2026_AACAirborneInsulation#2] to [Granzotto_2026_AACAirborneInsulation#5] (aircrete blocks 10 cm,
thin-bed mortar, block density stated only as "within 480–800 kg/m³", Rw 34–37 dB; ISO 10140, thirds from 100 Hz, figure
vector data; specimen size not stated). `masonry_light_100mm` is tested against the range of the four walls (spread 5.8
/ 4.6 / 2.5 / 3.4 / 3.8 / 4.9 dB, energy mean 29.3 / 30.7 / 27.6 / 35.8 / 43.8 / 50.1 dB, 125 Hz–4 kHz, script); they
share the row's thickness, so no thickness shift enters. The row's density is chosen inside 480–800 kg/m³ and checked
against [Hopkins_2007_SoundInsulation#4] (555 kg/m³) and Davy [Davy_2017_AACPanels#2] (509 kg/m³); the stated range is
48–80 kg/m², 4.44 dB of mass law (script), which the case states as the mass uncertainty. Printed: the fitted mass law
Rw = 29.0 lg m′ − 15.5 over 50 walls (unplastered 31.1 lg m′ − 19.4, plastered 24.6 lg m′ − 6.0;
[Granzotto_2026_AACAirborneInsulation#10], [Granzotto_2026_AACAirborneInsulation#11]), which at each 10 cm wall's own Rw
gives 51–65 kg/m² (509–646 kg/m³, script), circular as the mass of the test; [Granzotto_2026_AACAirborneInsulation#1]
(7 cm, 600 kg/m³ blocks), [Granzotto_2026_AACAirborneInsulation#6], [Granzotto_2026_AACAirborneInsulation#7] (12 cm,
500–600 kg/m³), [Granzotto_2026_AACAirborneInsulation#8] (12.5 cm, build unknown, low),
[Granzotto_2026_AACAirborneInsulation#9] (12.5 cm with 20 mm plaster both faces, build inferred, low); Davy
[Davy_2017_AACPanels#2] (bare 75 mm panels, 38.2 kg/m², side 3.29 m as stated). Fitted coincidence fc 400–630 Hz, fc·h
44–60 Hz·m (source), bears on `coiF`. Unsealed blocks leak ([Warnock_1995_BlockConstructions#3] reads 24.5 dB at 125 Hz
against 34.1 dB for [Warnock_1995_BlockConstructions#2]; [Warnock_1995_BlockConstructions#5] 6.9 against 32.5 dB) and
are not references. Density: Hopkins [Hopkins_2007_SoundInsulation#12] (aircrete 400–800 kg/m³),
[Hopkins_2007_SoundInsulation#78], [Hopkins_2007_SoundInsulation#106]; Davy [Davy_2017_AACPanels#1]. α: plaster as 1.2.
Gaps: aircrete spectra with each wall's own surface mass; walls of 15–20 cm; a matched bare/plastered pair.

1.5 **Plaster.** Only on hosts without plaster: `concrete`, `clt`. (comb.) matched pairs BIA
[BIA_2000_TN5AClayMasonry#7] / [BIA_2000_TN5AClayMasonry#8] and [BIA_2000_TN5AClayMasonry#15] /
[BIA_2000_TN5AClayMasonry#14], Warnock [Warnock_1984_ConcreteBlockWalls#4] / [Warnock_1984_ConcreteBlockWalls#5] (one
face, +1.7 to +2.6 dB on a sealed 290 mm block), the bare wall written inline in the case; printed:
[Warnock_1984_ConcreteBlockWalls#2] / [Warnock_1984_ConcreteBlockWalls#3] (two faces on porous units, +5.3 to +6.5 dB,
by sealing), Oliveira [Oliveira_2021_CeramicBlockCoating#1] to [Oliveira_2021_CeramicBlockCoating#14] (Rw +1 to +9 dB,
single numbers). α: PTB#344 [PTB_2012_AbsorptionTable#38], PTB#346 [PTB_2012_AbsorptionTable#40].

1.6 **Glass.** (row) R: Pilkington [Pilkington_2020_OptiphonBrochure#3], [Pilkington_2020_OptiphonBrochure#4] (EN 12758
generic); Quirt [Quirt_1981_WindowsTLMeasurements#24], [Quirt_1981_WindowsTLMeasurements#25]; AGC
[AGC_2023_PocketStratophone#11]. α: PTB#207 [PTB_2012_AbsorptionTable#11] is a 3 mm pane (neighbor build, printed): A
tier 2. Starting columns: page 11.4 rows "glass 4 mm" (also the rig's `GLASS4`) and "glass 6 mm", side 1.3492 m. The
generic values and Quirt's small panes differ by 5.6 to 6.6 dB at 125/250 Hz (4.4). The page row absorbs 0.305 at 125 Hz
against PTB#207's 0.08: its absorption is refitted. Constants: Hopkins [Hopkins_2007_SoundInsulation#84],
[Hopkins_2007_SoundInsulation#85], [Hopkins_2007_SoundInsulation#86].

1.7 **Glazing units.** (row) R: AGC [AGC_2023_PocketStratophone#2], Pilkington [Pilkington_2020_OptiphonBrochure#8],
Quirt [Quirt_1981_WindowsTLMeasurements#11], [Quirt_1981_WindowsTLMeasurements#13]; AGC [AGC_2023_PocketStratophone#14].
Quirt [Quirt_1981_WindowsTLMeasurements#26] is 3-10-3-10-3 (neighbor, printed). α: PTB#208 [PTB_2012_AbsorptionTable#12]
is 2–3 mm glass with a 10 mm gap (neighbor, printed): A tier 2. Starting columns: page 11.4 rows "double glazing 4-16-4"
and "triple glazing 4-12-4-12-4" with `thk` 0.008 and 0.012; under rule 1.7.2 they become `thk` 0.024 and 0.036 with
`fRef` × 3 and every power-law value rescaled.

1.8 **Box windows.** (row) R: Quirt [Quirt_1981_WindowsTLMeasurements#10], [Quirt_1981_WindowsTLMeasurements#9],
[Quirt_1981_WindowsTLMeasurements#12]. A leaf has no bending stiffness, so the coincidence dip at 2–3 kHz is not
expressible (page section 13).

1.9 **Gypsum board.** (row) R: Hopkins [Hopkins_2007_SoundInsulation#132], Northwood
[Northwood_1968_PlasterboardWalls#2]; Wareing [Wareing_2015_SampleSizeVariations#1],
[Wareing_2015_SampleSizeVariations#2] are 10 mm boards (neighbors, printed). η: Quirt
[Quirt_1995_GypsumWallsSummaryIR693#20], Hopkins [Hopkins_2007_SoundInsulation#136]. Density: Hopkins
[Hopkins_2007_SoundInsulation#135]. Starting columns: page 11.4 row "gypsum board 12.5 mm" (side 1.6971 m); Hopkins'
specimen side is √(3.53 × 2.63) = 3.047 m (script). Gaps: 31.5 Hz, 8 kHz.

1.10 **Gypsum leaf.** (comb.) section 3; α: PTB#341 [PTB_2012_AbsorptionTable#35] (empty cavity, Sabine, hard face: A
1), PTB#342 [PTB_2012_AbsorptionTable#36] (filled, above 0.2: 1p), Bradley [Bradley_1997_GypsumCavityAbsorption#2]
(model of chamber data). Starting columns: the rig's limp leaf `BOARD` (8.6 kg/m², loss ratio 0.2): `thk` 0.0125, `a`
4.691e-5, `e` 573.0, `g` 573.0·0.2/17.98, `h` 1.0. Gap: `rr` for room-size leaves.

1.11 **Drywall.** (row) `drywall_filled_125mm`: Halliwell [Halliwell_1998_GypsumBoardWalls#24],
[Halliwell_1998_GypsumBoardWalls#47] (G13 SS90 GFB90, about 115 mm). Printed, other builds: Halliwell
[Halliwell_1998_GypsumBoardWalls#27], [Halliwell_1998_GypsumBoardWalls#30] (16 mm boards, filled),
[Halliwell_1998_GypsumBoardWalls#35] (16 mm boards, empty, about 122 mm), [Halliwell_1998_GypsumBoardWalls#48] (G13
SS150 GFB150, about 175 mm: the check of the double-wall model, not a reference of `_150mm`); Fuchs 2005
[Fuchs_2005_LowFrequencyDomain#4] (12.5 / 100 / 12.5 mm, studs and fill not stated). Depth alone moves R:
[Halliwell_1998_GypsumBoardWalls#48] minus [Halliwell_1998_GypsumBoardWalls#47] = −0.6 / 7.6 / 9.9 / 11.0 / 6.2 / −1.9 /
0.6 dB (63 Hz–4 kHz, from the data rows); the fill: [Halliwell_1998_GypsumBoardWalls#27] minus
[Halliwell_1998_GypsumBoardWalls#35] up to 10.3 dB (from the data rows). [Halliwell_1998_GypsumBoardWalls#47] rises 13.5
and 14.5 dB per octave from 125 to 500 Hz, then dips at coincidence: fitting `drywall_filled_125mm` early checks whether
lumped double walls are expressible. α: as 1.10. Gaps: no wall of the build at 100 or 150 mm, none empty with 12.5 mm
boards. The sheets other than `_125mm` are tier 2 from a double-wall model.

1.12 **Dry lining.** (comb.) Hopkins [Hopkins_2007_SoundInsulation#65] (dabs on 100 mm dense block, ΔR),
[Hopkins_2007_SoundInsulation#64] (dabs on expanded clay, ΔR); BIA [BIA_2000_TN5AClayMasonry#9] against bare
[BIA_2000_TN5AClayMasonry#7] (on foam dabs); BIA [BIA_2000_TN5AClayMasonry#10] against [BIA_2000_TN5AClayMasonry#12] (on
wood furring); Zeitler linings [Zeitler_2015_ConcreteBlockRR334#14], [Zeitler_2015_ConcreteBlockRR334#15],
[Zeitler_2015_ConcreteBlockRR334#16], [Zeitler_2015_ConcreteBlockRR334#17], [Zeitler_2015_ConcreteBlockRR334#18],
[Zeitler_2015_ConcreteBlockRR334#19], [Zeitler_2015_ConcreteBlockRR334#20], [Zeitler_2015_ConcreteBlockRR334#21],
[Zeitler_2015_ConcreteBlockRR334#22], [Zeitler_2015_ConcreteBlockRR334#23], [Zeitler_2015_ConcreteBlockRR334#24],
[Zeitler_2015_ConcreteBlockRR334#25], [Zeitler_2015_ConcreteBlockRR334#26], [Zeitler_2015_ConcreteBlockRR334#27],
[Zeitler_2015_ConcreteBlockRR334#28], [Zeitler_2015_ConcreteBlockRR334#29], [Zeitler_2015_ConcreteBlockRR334#30],
[Zeitler_2015_ConcreteBlockRR334#31], [Zeitler_2015_ConcreteBlockRR334#32], [Zeitler_2015_ConcreteBlockRR334#33],
[Zeitler_2015_ConcreteBlockRR334#34] (ΔR) on base walls [Zeitler_2015_ConcreteBlockRR334#3],
[Zeitler_2015_ConcreteBlockRR334#2]. Page 11.4 rows "brick 140 mm, gypsum board on dabs", "brick 203 mm, gypsum board on
furring". The furring is not modeled.

1.13 **Plywood.** (row) R: Wareing [Wareing_2016_OrthotropicPlywood#5] (small specimen),
[Wareing_2016_OrthotropicPlywood#4], [Wareing_2016_OrthotropicPlywood#6], [Wareing_2016_OrthotropicPlywood#7]. α: Quirt
[Quirt_1983_WindowsII#1] (Sabine, hard face; the specimen lay on the chamber floor, while the row is a free panel: a
mounting difference, stated in the case). Constants: Hopkins [Hopkins_2007_SoundInsulation#151],
[Hopkins_2007_SoundInsulation#152], [Hopkins_2007_SoundInsulation#153], [Hopkins_2007_SoundInsulation#154]. Starting
columns: page 11.4 row "plywood 18 mm" (side 1.6971 m). Gaps: density of the specimens (710 kg/m³ assumed); 31.5/63 Hz.

1.14 **Wood panelling.** α (compilation): Cox [Cox_2016_AbsorbersDiffusers#119], Scheibler
[Scheibler_2018_Pyroomacoustics#78], [Scheibler_2018_Pyroomacoustics#97].

1.15 **Floor boards on joists.** α: PTB#393 [PTB_2012_AbsorptionTable#47] (Sabine, hard face, max 0.15; the same six
values appear in Cox [Cox_2016_AbsorbersDiffusers#161] and, for boards on battens over concrete, in Junckers
[Junckers_ND_AcousticsFloorConstructions#1]: one compilation value); compilation Cox [Cox_2016_AbsorbersDiffusers#57],
Scheibler [Scheibler_2018_Pyroomacoustics#95]; BRAS [Aspoeck_2020_BRAS#19], [Aspoeck_2020_BRAS#20] (room fits). (comb.)
R of wood-joist floor-ceilings with ceilings: Berendt [Berendt_1967_NBSMonograph77#34] to
[Berendt_1967_NBSMonograph77#46], NBS [NBS_1955_BMS144SoundInsulation#5] to [NBS_1955_BMS144SoundInsulation#32] (41
floors, from 125 Hz, warble tones, side 2.01–2.05 m; the six 1928 floors [NBS_1955_BMS144SoundInsulation#27] to
[NBS_1955_BMS144SoundInsulation#32] low confidence; joists bridge the leaves, which the leaf | box | leaf stack does not
model: printed). Moduli, densities and modes: Foy [Foy_2019_WoodFloorsVibroacoustic#1] to
[Foy_2019_WoodFloorsVibroacoustic#5] (tier-2 inputs). A rug on it is a miss candidate (stage-2b issue 5). Gaps: measured
α with build; R of a bare floor; 63 Hz.

1.16 **Parquet.** α: PTB#394 [PTB_2012_AbsorptionTable#48] (glued, Sabine, hard face); Cox
[Cox_2016_AbsorbersDiffusers#100]; BRAS [Aspoeck_2020_BRAS#58]; parquet on battens PTB#395
[PTB_2012_AbsorptionTable#49].

1.17 **Laminate.** None measured. Printed neighbours: Junckers [Junckers_ND_AcousticsFloorConstructions#2],
[Junckers_ND_AcousticsFloorConstructions#3] (floating solid-wood clip floors 22 and 14 mm on foam over concrete, type
not stated, catalog, maxima 0.19 and 0.13 at 250 Hz). A rug on it is a miss candidate (stage-2b issue 5). Gap:
absorption of a floating laminate floor (request 8: none found).

1.18 **Screed.** Constants: Hopkins [Hopkins_2007_SoundInsulation#171], [Hopkins_2007_SoundInsulation#172],
[Hopkins_2007_SoundInsulation#173], [Hopkins_2007_SoundInsulation#174], [Hopkins_2007_SoundInsulation#175],
[Hopkins_2007_SoundInsulation#176]. A bonded screed is drawn with the slab (1.7.5).

1.19 **Floating screed.** Resilient layers: Hopkins [Hopkins_2007_SoundInsulation#55],
[Hopkins_2007_SoundInsulation#83], [Hopkins_2007_SoundInsulation#91], [Hopkins_2007_SoundInsulation#92],
[Hopkins_2007_SoundInsulation#93], [Hopkins_2007_SoundInsulation#94], [Hopkins_2007_SoundInsulation#164],
[Hopkins_2007_SoundInsulation#165], [Hopkins_2007_SoundInsulation#166] (dynamic stiffness). Mass-law bodies do not
resonate with a fill (4.6).

1.20 **Ceramic tile.** α (compilation): Cox [Cox_2016_AbsorbersDiffusers#73]; BRAS [Aspoeck_2020_BRAS#89] (initial
estimate).

1.21 **Vinyl.** α: PTB#601 [PTB_2012_AbsorptionTable#73] (Sabine, hard face); Cox [Cox_2016_AbsorbersDiffusers#72]; Bork
[Bork_2000_RR2#8].

1.22 **Metal sheets.** (row) R: Kosala [Kosala_2020_SingleBafflesAccuracy#51] (steel),
[Kosala_2020_SingleBafflesAccuracy#6] (aluminum 2 mm); Northwood [Northwood_1968_PlasterboardWalls#6] (0.8 mm steel).
Constants: Hopkins [Hopkins_2007_SoundInsulation#186], [Hopkins_2007_SoundInsulation#187],
[Hopkins_2007_SoundInsulation#188], [Hopkins_2007_SoundInsulation#5], [Hopkins_2007_SoundInsulation#6],
[Hopkins_2007_SoundInsulation#7]. Gaps: specimen size of the references; installed loss factor.

1.23 **Solid door.** Thickness 40 mm (decided): `thk` is the visible thickness, the surface mass carries R, and a 44 mm
leaf maps to the same row. (row) R of a sealed leaf: Quirt [Quirt_1986_DoubleDoors#1] (45 mm solid core, hardboard
faces, fully sealed, 914 × 2134 mm, 46.7 kg, 23.9 kg/m², side 1.397 m by script; ASTM E90, thirds 80 Hz–5 kHz from a
figure, medium); Morin [Morin_1993_AccessDoorsCMHC#1], [Morin_1993_AccessDoorsCMHC#2] (44 mm, 25.2 kg/m², gasketed /
caulked, 250 Hz–2 kHz, sound intensity, side 1.427 m by script), [Morin_1993_AccessDoorsCMHC#3],
[Morin_1993_AccessDoorsCMHC#4] (57 mm, 30.5 kg/m², printed); Berendt [Berendt_1967_NBSMonograph77#14] (NBS 632, 44 mm,
22.5 kg/m², gaskets and caulking, warble tones, side 1.397 m). The field-incidence mass law R = 20 lg(m f) − 47 at
Quirt's R needs 27.4 / 26.1 / 17.0 / 5.6 / 3.6 / 3.3 kg/m² (125 Hz–4 kHz, script): within 1.2 dB of the stated
23.9 kg/m² at 125–250 Hz, far below from 500 Hz (coincidence and leaks; Quirt 28.0–31.6 dB at 500 Hz–2 kHz): `coiF` and
`att` are fit knobs, as for concrete. (comb.) seal states [Quirt_1986_DoubleDoors#2] to [Quirt_1986_DoubleDoors#4]
(magnetic, foam, none), [Morin_1993_AccessDoorsCMHC#1] / [Morin_1993_AccessDoorsCMHC#2] (section 3). Printed: NBS
panelled doors [Berendt_1967_NBSMonograph77#17] to [Berendt_1967_NBSMonograph77#19] (64 mm, 38.1 kg/m²), the 64–67 mm
NBS doors [NBS_1955_BMS144SoundInsulation#1] to [NBS_1955_BMS144SoundInsulation#4]. α: PTB#396
[PTB_2012_AbsorptionTable#50] (Sabine, hard face); Cox [Cox_2016_AbsorbersDiffusers#146], Scheibler
[Scheibler_2018_Pyroomacoustics#96]; Pinhede [Pinhede_2025_ActiveSemiAnechoic#3] (room fit). Gaps: absorption of a door
beyond PTB#396 (compilation); 63 Hz; a modern ISO 10140 door report with thirds from 50 Hz.

1.24 **Hollow-core door.** Thickness 40 mm as 1.23 (decided). Surface mass 9.3 kg/m² (decided): NBS 633/634
[Berendt_1967_NBSMonograph77#15], [Berendt_1967_NBSMonograph77#16] (44 mm hollow-core flush door, conventional and
caulked, warble tones, 9.3 kg/m² stated). (row) R, the spectrum reference: Lee [Lee_2010_InteriorDoorAperture#1] (author
M.-J. Kim; 35 mm lattice-core leaf with 4.5 mm MDF faces, 875 × 2215 mm, all joints sealed, side 1.392 m by script; KS F
2808 (= ISO 140-3), thirds 63 Hz–5 kHz from a figure, the 63 and 80 Hz thirds outside the standard and the 63 Hz octave
blank; surface mass not stated). The field mass law at Kim's R gives 7.6–9.2 kg/m² over 500 Hz–4 kHz (script), within
1.75 dB of the NBS mass; NBS 634 itself reads 13.1–17.3 dB below Kim at 1–4 kHz (printed). Kim's mass-air-mass dip at
315 Hz (3–5 dB) is not expressible by a lumped sheet: printed as an engine limit. (comb.) bottom gaps 2–17 mm
[Lee_2010_InteriorDoorAperture#2] to [Lee_2010_InteriorDoorAperture#14] (the 6 mm gap [Lee_2010_InteriorDoorAperture#4],
the survey mean 6.4 mm, gives Rw 23 against 30 sealed; section 3). Gaps: surface mass of a modern hollow-core leaf; a
second modern spectrum; absorption; a 50 Hz third.

1.25 **CLT.** (row) R: Hoeller [Hoeller_2017_CLTBuildingsRR335#1], [Hoeller_2017_CLTBuildingsRR335#2],
[Hoeller_2017_CLTBuildingsRR335#3]; η: [Hoeller_2017_CLTBuildingsRR335#4], [Hoeller_2017_CLTBuildingsRR335#5],
[Hoeller_2017_CLTBuildingsRR335#6].

### 2.2 Class 2: hard furnishing

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.1 | `furniture_board`, `furniture_board_19mm`, `furniture_board_25mm` | table tops, shelves, open furniture panels | sheet for an open single panel; solid for a thick top or a frame | 3 | 2 | — | medium | wish |
| 2.2 | `box_cabinet` (proposed name) | wardrobes, cupboards, closed cabinets | hard box (2.4.1), asset standing on the floor | 1 | — | per object | medium | wish |
| 2.3 | `box_bookshelf` (proposed name) | a filled bookshelf | hard box | 3 | — | — | medium | wish |
| 2.4 | `box_metal` (proposed name) | fridges, freezers, bar counters, metal cabinets | hard box; a glass front as a `glass_4mm` sheet on that face | 3 | — | — | low | wish |
| 2.5 | `box_casing` (proposed name) | casings of TVs, speaker cabinets, lamps, appliances | hard box; a screen as a glass sheet on its face | 3 | — | — | low | wish |
| 2.6 | table and chairs (no row) | dining and office furniture | table top as a `furniture_board_*` sheet; legs and chair frames left out (proposed) | 1 | — | per object | low | wish |
| 2.7 | sports equipment (no row) | metal pipes and bars | leave out (proposed) | — | — | — | low | wish |

References and gaps:

2.1 Constants: Hopkins MDF [Hopkins_2007_SoundInsulation#114], [Hopkins_2007_SoundInsulation#115],
[Hopkins_2007_SoundInsulation#116], chipboard [Hopkins_2007_SoundInsulation#25], [Hopkins_2007_SoundInsulation#26],
[Hopkins_2007_SoundInsulation#27] (both 760 kg/m³), softwood [Hopkins_2007_SoundInsulation#179],
[Hopkins_2007_SoundInsulation#180]. No R or α of a free board; its role is a reflector. The rig's `MDF` leaf is the leaf
form for panel absorbers.

2.2 (row) Nocke [Nocke_2018_ObjectAbsorption#2] (cabinet without acoustic front, per object),
[Nocke_2018_ObjectAbsorption#4] (α on its cuboid, Sabine); with acoustic front [Nocke_2018_ObjectAbsorption#1],
[Nocke_2018_ObjectAbsorption#3]. Nocke's cabinet is 0.80 × 0.46 × 1.09 m; its S_E of 3.11 m² is the cuboid without its
floor face (3.115 m² against 3.483 m² for all faces, script). (row) Tsuchiya [Tsuchiya_2013_FurnitureAbsorption#8],
[Tsuchiya_2013_FurnitureAbsorption#9] (wardrobe with hinged doors, 0.59 × 0.515 × 1.88 m, empty, centre / against a
wall), [Tsuchiya_2013_FurnitureAbsorption#10], [Tsuchiya_2013_FurnitureAbsorption#11] (tall chest of 8 drawers, empty,
centre / wall), [Tsuchiya_2013_FurnitureAbsorption#12], [Tsuchiya_2013_FurnitureAbsorption#13] (the same chest with
clothes in the drawers) (JIS A 1409, one unit, per object, figure vector data; the 63 Hz octave from reference thirds
outside the chamber's range, printed). Hard-box S without the floor face 4.459 m² (all faces 4.762 m², without floor and
back 3.349 m², script); face α = A/S of the empty wardrobe in the centre 0.291 / 0.274 / 0.070 / 0.059 / 0.034 / 0.050 /
0.041 (63 Hz–4 kHz, script), the 63–125 Hz peak from the box interior. The asset stands on the floor; against a wall,
the back face's share is lost. Gaps: a wardrobe with hanging clothes behind its doors; open doors.

2.3 Odeon [Odeon_2025_UserManualV19#2] (scattering only). Aretz [Aretz_2011_DAGA#1] to [Aretz_2011_DAGA#4] (a wooden
bookcase pair in four fill states, empty to full, against a side wall of the chamber; `alpha_obj`, typed Sabine,
reference area not stated, so no absorption area per object follows; 8 kHz from thirds outside ISO 354): full / empty
1.87 / 1.64 / 1.78 / 1.78 / 1.82 / 1.61 / 1.38 (125 Hz–8 kHz, script), printed as guidance; tier 3 stays. Gap: the
bookcase's size.

2.4, 2.5 No per-object data. Casing constants: Hopkins PMMA [Hopkins_2007_SoundInsulation#158],
[Hopkins_2007_SoundInsulation#159]; Kosala PMMA R [Kosala_2020_SingleBafflesAccuracy#19],
[Kosala_2020_SingleBafflesAccuracy#21], [Kosala_2020_SingleBafflesAccuracy#17], polypropylene
[Kosala_2020_SingleBafflesAccuracy#34], [Kosala_2020_SingleBafflesAccuracy#35], [Kosala_2020_SingleBafflesAccuracy#36],
[Kosala_2020_SingleBafflesAccuracy#37].

2.6 Per object: Nilsson [Nilsson_2007_NTTR606#6] (table with two chairs), [Nilsson_2007_NTTR606#7] (one chair),
[Nilsson_2007_NTTR606#5] (10 tables + 19 chairs; values include scattering toward an absorbing ceiling); BRAS
[Aspoeck_2020_BRAS#10], [Aspoeck_2020_BRAS#11] (room fits).

### 2.3 Class 3: soft furnishing

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | `carpet_tile_11mm` | carpet tiles, low-pile rugs | sheet on the slab or finish solid; cut out under standing objects | 1 at 63 Hz and to 500 Hz; above: 1 once σ is measured | 2 | 63 Hz–4 kHz | high | wish |
| 3.2 | `carpet_tufted_12mm` | tufted carpet without underlay | sheet on a solid | 1 | 2 | 125 Hz–4 kHz, 0° | medium | wish |
| 3.3 | `underlay` | felt or foam underlay | traced solid of its real thickness: sheet \| `underlay` \| floor solid | 2 | 2 | — | medium | wish |
| 3.4 | `curtain_flat_0_2mm` | curtains hung flat; roller and vertical blinds (recipe) | fabric leaf sheet \| box of cavity air \| wall or window (sheet at the box's exit) | 1p | 2 | — | high | wish |
| 3.5 | `curtain_pleated_0_8mm` | curtains with fullness, stage drapes | as 3.4, effective columns | 1p | 2 | — | high | wish |
| 3.6 | `pu_foam` | free small cushions and pillows | porous body of the cushion's shape | 2 | 2 | — | medium | wish |
| 3.7 | `polyester_fiber`, `polyester_fiber_20mm` | fiber pillows (solid), blankets (sheet) | solid, or sheet on a solid | 2 | 2 | — | low | wish |
| 3.8 | `upholstery` | the body of soft furniture: sofas, armchairs, beds, mattresses, large cushions | porous body (2.4.2), assets in size variants; the fit is phase-3 work | 2 (measured padding and cover inputs) | — | per object | high | wish |

References and gaps:

3.1 (comb.) Salford [Salford_2020_CarpetTileSpecimen#1] (Sabine, 11 mm on concrete; hard face to 500 Hz, 0.09), BCTC
[BCTC_2014_EcosoftCarpetTile#1] (Sabine, cushion backed; 0.40 at 500 Hz, printed above 250 Hz); cushion- and
fleece-backed tiles CSIRO [CSIRO_2019_WellbacComfortTile#1], [CSIRO_2019_WellbacComfortPlus#1],
[CSIRO_2019_WellbacFunctionTile#1], TFI [TFI_2018_NylonCarpetTileCushion#1] (−0.01 at 125 Hz, read as zero), Kiwa
[Kiwa_2021_EForceComfortBack#1], Salford [Salford_2022_FlotexNext#1], Milliken [Milliken_ND_CarpetTileCertificates#1],
[Milliken_ND_CarpetTileCertificates#2] (Sabine, laboratory reports, other products, printed; the cushion backs act as a
mass on a spring and reach 0.30–0.35 at 500 Hz). 63 Hz (decided): A 1 from the four neighbour builds
[Milliken_ND_CarpetTileCertificates#1], [TFI_2018_NylonCarpetTileCushion#1], [Kiwa_2021_EForceComfortBack#1], Bietz
[Bietz_2025_CombiningRoomVolumes#2] (0.00–0.01 whatever the backing: a hard face in that band). Cox
[Cox_2016_AbsorbersDiffusers#15], [Cox_2016_AbsorbersDiffusers#16], [Cox_2016_AbsorbersDiffusers#17] (compilation
spread); PTB#155 [PTB_2012_AbsorptionTable#4] is 5 mm needle felt (another product, printed). Above 500 Hz the porous
conversion applies once σ is measured. Gap: flow resistivity of a tile's pile and backing.

3.2 (comb.) Broda [Broda_2025_TuftedCarpets#3] (normal incidence, rigid backing); SAA [Broda_2025_TuftedCarpets#2].
Compared at 0° in the rig. Printed: Ford [Ford_1980_CarpetsUnderlays#5] (tufted wool carpet on concrete, Sabine, figure,
build not stated).

3.3 α alone: Broda [Broda_2025_TuftedCarpets#1] (normal incidence, range only); Ford underlays
[Ford_1980_CarpetsUnderlays#3], [Ford_1980_CarpetsUnderlays#6] (Sabine, figure, types not stated: printed); (comb.)
Broda [Broda_2025_TuftedCarpets#4], PTB#156–158 [PTB_2012_AbsorptionTable#5], [PTB_2012_AbsorptionTable#6],
[PTB_2012_AbsorptionTable#7], Cox [Cox_2016_AbsorbersDiffusers#147], [Cox_2016_AbsorbersDiffusers#149]; Ford carpets on
underlay [Ford_1980_CarpetsUnderlays#1], [Ford_1980_CarpetsUnderlays#4] (builds not stated: printed). σ inputs (tier 2):
rebond PU Mickovski [Mickovski_2022_RebondedPUFoam#1], [Mickovski_2022_RebondedPUFoam#2] (3.8 and 5.0 kPa·s/m² at 40 and
60 kg/m³, slabs 45–95 mm, thicker than underlay); nonwovens Lupescu [Lupescu_2025_MilkweedCarpets#1],
[Lupescu_2025_MilkweedCarpets#3], [Lupescu_2025_MilkweedCarpets#5], [Lupescu_2025_MilkweedCarpets#7],
[Lupescu_2025_MilkweedCarpets#9], [Lupescu_2025_MilkweedCarpets#11] (σ from a figure). The floor below must be a
mass-law body (two traced wave media back to back: stage-2b issue 3). Gap: σ of felt and rubber underlays.

3.4 (comb.) DelRey [DelRey_2013_CurtainFullness#14], [DelRey_2013_CurtainFullness#16] (Sabine, flat; fabric 0.2 mm,
300 g/m², 144 Pa·s/m), [DelRey_2013_CurtainFullness#18]; σ: [DelRey_2013_CurtainFullness#28]. Blinds: compilation
Scheibler [Scheibler_2018_Pyroomacoustics#13] (vertical blinds 15 cm from the wall, half open),
[Scheibler_2018_Pyroomacoustics#14] (open), as a `curtain_flat_*` recipe. A curtain in front of a window needs the sheet
at the box's exit (the glazing). Gap: a random-incidence route for a hanging fabric.

3.5 (comb.) Zimmer+Rohde [ZimmerRohde_2022_Fabric10936Curtain#1] (342 g/m², 0.79 mm, 1421 Pa·s/m, 100 % fullness, 100 mm
from the wall, Sabine), [ZimmerRohde_2020_KakaduCurtain#1]; ADO [ADO_2018_CannyCurtain#1],
[ADO_2020_BreakingDawnCurtain#1] (Sabine, 100 % fullness, 100 mm from the wall); velour (compilation) Cox
[Cox_2016_AbsorbersDiffusers#64], [Cox_2016_AbsorbersDiffusers#65], [Cox_2016_AbsorbersDiffusers#74],
[Cox_2016_AbsorbersDiffusers#75]; PTB#159 [PTB_2012_AbsorptionTable#8]. The two curtain references differ by a factor of
9.9 in flow resistance (1421 / 144, [ZimmerRohde_2022_Fabric10936Curtain#1], [DelRey_2013_CurtainFullness#28]) and 1.14
in mass (342 / 300): fullness and resistance, not mass, separate them.

3.6 Dupont [Dupont_2023_SoundFieldSynthesis#6], [Dupont_2023_SoundFieldSynthesis#7],
[Dupont_2023_SoundFieldSynthesis#8], [Dupont_2023_SoundFieldSynthesis#9] (σ 10605 Pa·s/m²,
[Dupont_2023_SoundFieldSynthesis#7]). Porous columns from Miki or Komatsu at the measured σ; generic solids of wave
media are exact at any drawn thickness. Porous bodies must not touch each other.

3.7 σ: Garai (fiber blankets 10–70 mm; [Garai_2005_PolyesterModel#1], [Garai_2005_PolyesterModel#2],
[Garai_2005_PolyesterModel#3], [Garai_2005_PolyesterModel#4], [Garai_2005_PolyesterModel#5],
[Garai_2005_PolyesterModel#6], [Garai_2005_PolyesterModel#7], [Garai_2005_PolyesterModel#8],
[Garai_2005_PolyesterModel#9], [Garai_2005_PolyesterModel#10], [Garai_2005_PolyesterModel#11],
[Garai_2005_PolyesterModel#12], [Garai_2005_PolyesterModel#13], [Garai_2005_PolyesterModel#14],
[Garai_2005_PolyesterModel#15], [Garai_2005_PolyesterModel#16], [Garai_2005_PolyesterModel#17],
[Garai_2005_PolyesterModel#18], [Garai_2005_PolyesterModel#19], [Garai_2005_PolyesterModel#20],
[Garai_2005_PolyesterModel#21], [Garai_2005_PolyesterModel#22], [Garai_2005_PolyesterModel#23],
[Garai_2005_PolyesterModel#24], [Garai_2005_PolyesterModel#25], [Garai_2005_PolyesterModel#26],
[Garai_2005_PolyesterModel#27], [Garai_2005_PolyesterModel#28], [Garai_2005_PolyesterModel#29],
[Garai_2005_PolyesterModel#30], [Garai_2005_PolyesterModel#31], [Garai_2005_PolyesterModel#32],
[Garai_2005_PolyesterModel#33], [Garai_2005_PolyesterModel#34], [Garai_2005_PolyesterModel#35],
[Garai_2005_PolyesterModel#36], [Garai_2005_PolyesterModel#37], [Garai_2005_PolyesterModel#38] span 12–110 kg/m³);
[Garai_2005_PolyesterModel#39] to [Garai_2005_PolyesterModel#42] are a validity limit, a model deviation and two
regressions, not σ rows (printed). Polyester model unresolved (research handoff, A14).

3.8 Per object: see 4.1, 4.2; SP [SP_2017_ChairDecibel#1], Nilsson [Nilsson_2007_NTTR606#7]; the per-area values of
seating rows (PTB#484 [PTB_2012_AbsorptionTable#71], PTB#483 [PTB_2012_AbsorptionTable#70], Kath and Kuhl
[KathKuhl_1965_UpholsteredChairs#3]; compilation Cox [Cox_2016_AbsorbersDiffusers#152],
[Cox_2016_AbsorbersDiffusers#153], [Cox_2016_AbsorbersDiffusers#154], PTB#480 [PTB_2012_AbsorptionTable#67], Scheibler
[Scheibler_2018_Pyroomacoustics#34], [Scheibler_2018_Pyroomacoustics#35]) belong to 4.10, not to a sofa body. Model
inputs (A 2 for the body: a resistive sheet on a porous body): padding Rubino [Rubino_2026_UpholsteredSeatingCircular#9]
to [Rubino_2026_UpholsteredSeatingCircular#14] (PU and textile-waste nonwoven: σ, density, porosity), covers
[Rubino_2026_UpholsteredSeatingCircular#15], [Rubino_2026_UpholsteredSeatingCircular#16] (eco-leather 2482 Pa·s/m,
fabric 516 Pa·s/m); simulated layups [Rubino_2026_UpholsteredSeatingCircular#3] to
[Rubino_2026_UpholsteredSeatingCircular#8] (modelled, low: printed).

### 2.4 Class 4: complex assemblies, and the object recipes

Objects that are not plane layers are modeled in one of three ways. Scattering and diffraction stay out of scope.
Objects stand on the slab or on a finish solid, never on a finish sheet or a rug (1.7.4).

**2.4.1 Hard box** (hard closed furniture). A closed box of the object's outer size. Its faces carry a rigid row
(mass column) whose `tf` is, per octave, the measured per-object absorption area A divided by the box's absorbing
surface S, fitted as a power law (a `tf` without a peak is flat in angle and is a random-incidence α: page 10.3).
S must be at least the largest A. **(proposed)** The box's mass is set so that it passes nothing of interest. Scaling
the asset scales S; the face α stays, so the absorption area scales with the surface.

**2.4.2 Porous body** (soft furniture; a person; an audience or rows of seats). A low-poly shape of the object, or a
cushion-like box, of a porous row, fitted so that the body's absorption area matches the per-object data. It
reflects at its face, absorbs inside and transmits some. Its absorption area does not scale with its surface, so
porous assets come in size variants (for example 1-, 2- and 3-seat sofas) and are not scaled freely. An air-like
absorbing volume is rejected for furniture: it reflects nothing.

**2.4.3 Absorbing volume** (clutter, foliage). A closed volume of a medium matched to air (`a` = `e` = 1) with an
`alpha` column in dB/m. The start value from a per-object area A of the volume V is Sabine's air term A = 4mV:
`alpha` = 4.343·A/(4V) dB/m. It holds only for optically thin volumes; a non-reflecting body saturates toward its
surface S. Monte Carlo over cosine-weighted chords of a cube, true area divided by 4mV:

| 4mV/S | 0.05 | 0.1 | 0.25 | 0.5 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| true / formula | 0.966 | 0.935 | 0.850 | 0.733 | 0.564 |

So `alpha` is fitted with the object test. Vegetation (0.12 dB/m at 8 kHz, [Witte_2018_ForestAttenuation#3]) is far
inside the thin limit.

**Per-object rule.** A per-object area measured in a reverberation chamber is used like for like, without conversion:
the object test places the asset in a diffuse field (rays from all directions) and computes its absorption area, which
in a diffuse field is 4 × the absorption cross-section averaged over directions. The placement of the measurement
(on the floor, in a corner) is stated with each reference; the object test runs the asset on the bare floor. Placement
values (corners) are no material property.

**Assets.** OBJ meshes generated by a script, so that they are the reader's input and testable through
`obj_file_read`; a Blender asset library is built on top.

**Object test (phase 3).** Rays from all directions onto the body give its absorption area per octave. Designed and
timed as its own rig step before the first object row.

| # | name | object | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 | `upholstery` (assets `sofa_1seat`, `sofa_2seat`, `sofa_3seat`, `armchair`, proposed) | couch, sofa, armchair | porous body | 1 (leather sofa); 2 (fabric seats; Caimi printed as the upper bound) | — | per object | high | wish |
| 4.2 | `upholstery` on `furniture_board` (asset `bed`, proposed) | bed | porous body (the mattress) on a traced solid of the frame | 1 (single bed without bedding); 3 (double bed, bedding) | — | per object | medium | wish |
| 4.3 | `box_bookshelf` | books in a bookshelf | hard box (2.3) | 3 | — | — | medium | wish |
| 4.4 | `clothes_rack` (proposed name) | rack of clothing | porous body of a light fiber medium around the hanging clothes (proposed) | 3 | — | — | low | wish |
| 4.5 | `plant_foliage`, `green_wall_100mm` (proposed names) | plants indoors, green walls | potted plant: leave out if small, else an absorbing volume of the foliage outline (proposed); green wall: a porous sheet | 1 (1 m palm, 2.3 m conifer); 3 (other potted); 1p (green wall) | — | — | low | wish |
| 4.6 | `clutter_sparse`, `clutter_medium`, `clutter_dense` | items on desks and shelves, dishes on a counter | absorbing volume (2.4.3) drawn over the items; revisable with data | 3 | — | — | medium | wish |
| 4.7 | `person` (proposed name; assets `person_seated`, `person_standing`) | a listener, a performer | porous body of a low-poly human shape (proposed) | 1 | — | 125 Hz–2 kHz | medium | wish |
| 4.8 | `audience_seated` (proposed name) | dense audience | porous box over the seating area, of the height of seated persons (proposed) | 1p | — | 125 Hz–4 kHz | high (venues) | wish |
| 4.9 | dishes and glasses (no row) | kitchens, bars | leave out, or inside a clutter volume (proposed) | — | — | — | low | wish |
| 4.10 | `seating_empty` (proposed name) | unoccupied rows of seats, pews | porous box over the seating area, as 4.8 (proposed) | 1p | — | 250 Hz–2 kHz | medium (venues) | wish |

References and gaps:

4.1 Per object (used like for like): synthetic-leather sofa Tsuchiya [Tsuchiya_2013_FurnitureAbsorption#6],
[Tsuchiya_2013_FurnitureAbsorption#7] (JIS A 1409, one unit, 50 Hz–5 kHz, centre / wall): the tier-1 reference of
`upholstery` (decided). Its printed size "W1260 × D2700 mm" is implausible (the depth would exceed 2.7 m); the asset's
size is open. Fabric acoustic seating in a one- to three-seat size series Caimi [Caimi_2025_VolumiSpecification#1] to
[Caimi_2025_VolumiSpecification#24], [Caimi_2025_CilindroSpecification#1] to [Caimi_2025_CilindroSpecification#4] (ISO
354 as stated, manufacturer data sheet, laboratory, placement and unit count not named; 125 Hz–4 kHz): printed as the
upper bound of fabric seats (decided). Printed: acoustic high-back sofas SP [SP_2016_GotessonsHutSofa#1] to
[SP_2016_GotessonsHutSofa#3] (ISO 354, with a wall variant); armchair without cover Rubino
[Rubino_2026_UpholsteredSeatingCircular#1], [Rubino_2026_UpholsteredSeatingCircular#2]. Fabric acoustic seats absorb 1.7
to 9.5 times the leather sofa by band (script). Ordinary fabric domestic seats: tier 2 (Caimi as the upper bound, Rubino
without cover as a neighbour). Gaps: an ordinary fabric domestic sofa and armchair; occupied seats; 63 Hz inside a
standard's range.

4.2 (row) Tsuchiya [Tsuchiya_2013_FurnitureAbsorption#1] (single bed with storage base, frame 0.99 × 2.0 m, spring
mattress, no bedding, centre), [Tsuchiya_2013_FurnitureAbsorption#2] (against a wall: −0.4 to +0.1 m² against the
centre, 63 Hz–4 kHz, script), [Tsuchiya_2013_FurnitureAbsorption#3] (mattress in its delivery bag); frame alone
[Tsuchiya_2013_FurnitureAbsorption#4], [Tsuchiya_2013_FurnitureAbsorption#5] (with and without drawers and deck: the
frame's recipe; its 63–125 Hz values come from its storage box). The mattress (#1 minus #4) carries 0.51 / 0.21 / 1.41 /
2.16 / 2.31 / 2.28 / 1.91 m² (63 Hz–4 kHz, script). Under 1.7.9 the frame-alone rows test the frame and the whole bed
tests the recipe. Gaps: double bed; bedding.

4.3 Aretz ratios as 2.3; tier 3 stays (request 9).

4.4 None (request 13: clothing racks and hanging garments not found).

4.5 Potted plants (row, per object): Vinade [Vinade_2015_DecorativePlantArrangements#1] (1 m Buriti palm in a grow bag;
ISO 354, 12 units, per unit), [Vinade_2015_DecorativePlantArrangements#3] (the same palm in a 0.5 m MDF box with moss),
[Vinade_2015_DecorativePlantArrangements#2] (the empty box, 0.05–0.08 m², a 160–200 Hz box resonance); RAL
[RAL_2022_FraserFirTrees#1], [RAL_2022_DecoratedChristmasTrees#1] (2.3 m firs, undecorated and decorated, ASTM C423,
three trees, per tree; 63 Hz blank); printed: Costa [Costa_1995_VegetationNoiseControl#1] to
[Costa_1995_VegetationNoiseControl#7] (unit not stated, low), Yang [Yang_2013_NaturalMaterialsNoiseThesis#1],
[Yang_2013_NaturalMaterialsNoiseThesis#2] (box shrubs, foliage only, 0.002–0.009 m² per plant: "leave out if small"
holds). Green walls: Thomazelli [Thomazelli_2016_GreenWallsPOMA#3] to [Thomazelli_2016_GreenWallsPOMA#7] (Sabine); Davis
[Davis_2017_VerticalGarden#2] (Sabine, single number only). Gap: broadleaf plants of 1.5–2 m (closed source).

4.6 Groups of furniture only: Nilsson [Nilsson_2007_NTTR606#5], [Nilsson_2007_NTTR606#6] (include scattering); room
level per m² floor: Nilsson [Nilsson_2017_InputDataPublicRooms#1] to [Nilsson_2017_InputDataPublicRooms#3] (A_sc, three
office densities, in situ, including sound scattered into an absorbing ceiling),
[Nilsson_2026_ReverberationOrdinaryRooms#1] to [Nilsson_2026_ReverberationOrdinaryRooms#3] (furnishing area, three
classes, preprint; the dense class 0.33 m²/m² at 1 kHz: the check for a dense scene; further rows
[Nilsson_2026_ReverberationOrdinaryRooms#4] to [Nilsson_2026_ReverberationOrdinaryRooms#13]); printed: Baruch
[Baruch_2020_OfficeDeskAbsorbers#1] (1:8 scale-model desk, per sample, low). Clutter mainly scatters, which is out of
scope; the volume gives only absorption. Gap: per-object or per-area data of a desk or shelf with items (request 10:
none).

4.7 (row) per person: Kath and Kuhl [KathKuhl_1964_PublikumAbsorption#7] (seated, isolated),
[KathKuhl_1964_PublikumAbsorption#8] (standing); Nishidate [Nishidate_2024_PersonAbsorption#1] (design value),
[Nishidate_2024_PersonAbsorption#2], [Nishidate_2024_PersonAbsorption#3] (in situ); Chagok
[Chagok_2013_AbsorptionPerPerson#2] not usable. Gap: 63 Hz.

4.8 Per area of a block (1p until the per-area convention for blocks is decided): Kath and Kuhl
[KathKuhl_1964_PublikumAbsorption#3], [KathKuhl_1964_PublikumAbsorption#4] (Sabine),
[KathKuhl_1964_PublikumAbsorption#5], [KathKuhl_1964_PublikumAbsorption#6] (in situ); Bradley
[Bradley_1996_OccupiedSeating#1], [Bradley_1996_OccupiedSeating#2] (occupied seats, infinite-block intercept, Sabine; up
to 1.051 / 1.088, values above one marked); Carvalho [Carvalho_2012_ChurchPews#2] (per object),
[Carvalho_2012_ChurchPews#4] (Sabine per area 1.26–1.65, marked); Kath and Kuhl [KathKuhl_1965_UpholsteredChairs#1],
[KathKuhl_1965_UpholsteredChairs#2]. Edge effects: Kath and Kuhl [KathKuhl_1964_PublikumAbsorption#1],
[KathKuhl_1964_PublikumAbsorption#2] (strip widths).

4.10 Kath and Kuhl [KathKuhl_1965_UpholsteredChairs#3] (Sabine), [KathKuhl_1965_UpholsteredChairs#4] (in situ); PTB#484
[PTB_2012_AbsorptionTable#71]; Carvalho [Carvalho_2012_ChurchPews#3] (pews, per object).

### 2.5 Class 5: acoustic treatment

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 | `glass_wool`, `glass_wool_50mm`, `glass_wool_100mm` | porous panels, cavity fill, ceiling clouds and baffles (recipe) | sheet on a rigid solid; solid for a fill, a block, a cloud or a baffle hung in air | 1 | 2 | 250 Hz–4 kHz | high | wish |
| 5.2 | `rock_wool`, `rock_wool_50mm` | porous panels, cavity fill | as 5.1 | 1 | 2 | 250 Hz–4 kHz | medium | wish |
| 5.3 | `pu_foam_50mm` | foam panels | sheet on a rigid solid | 1 | 2 | 250 Hz–4 kHz | medium | wish |
| 5.4 | `melamine_foam_50mm` | studio foam | sheet on a rigid solid | 2 | 2 | — | medium | wish |
| 5.5 | `polyester_fiber_50mm` | PET felt panels | sheet on a rigid solid | 2 | 2 | — | medium | wish |
| 5.6 | `wood_wool_50mm` | wood-wool panels on walls and ceilings | sheet on a rigid solid, or on a box of cavity air | 2 | 2 | — | medium | wish |
| 5.7 | porous panel with an air gap (no row) | panels stood off the wall | absorber sheet \| box of cavity air \| wall | 2 | 2 | — | high | wish |
| 5.8 | bass trap (no row) | corner traps | porous body of the trap's shape in the corner, as an asset (proposed) | 2 | — | — | high | wish |
| 5.9 | `mdf_leaf_6mm` | panel absorber, empty or filled cavity | leaf sheet \| box of cavity air (\| fill solid) \| wall | 1p | 2 | — | high | wish |
| 5.10 | `plywood_leaf_3mm` | panel absorber | leaf sheet \| box of cavity air \| wall | 1p | 2 | — | medium | wish |
| 5.11 | `steel_leaf_1mm` | compound panel absorber (steel bonded to foam) | steel leaf sheet bonded to a fill solid \| wall | 1p | 2 | — | medium | miss candidate |
| 5.12 | `steel_leaf_2_5mm` | free-floating plate absorber | leaf on a fiber-mat fill | 1p | 2 | — | low | miss candidate |
| 5.13 | `mpp_steel_0_5mm` | microperforated cassette ceiling | MPP leaf sheet (equivalent fluid) \| box of cavity air \| slab | 1p | 2 | — | medium | wish |
| 5.14 | `mpp_foil_0_1mm` | MPP foil, sails, aprons | leaf \| box \| wall; an apron as a sheet in air | 2 | 2 | — | low | wish |
| 5.15 | `ceiling_tile_20mm` | suspended absorbing ceiling | porous tile sheet \| box of cavity air (the plenum) \| slab | 1 (Focus A, E-50 and E-200) | 2 | 125 Hz–4 kHz; 30–180 mm air | high (offices, venues) | wish |
| 5.16 | `perforated_gypsum_12_5mm` | perforated plasterboard ceilings | perforated leaf \| box \| slab | 1p | 2 | — | medium | wish |
| 5.17 | slotted steel covers on a porous layer (no row) | slotted ceilings | not a plane layer | 3 | — | — | low | miss candidate |
| 5.18 | diffusers (`box_diffuser`, proposed name) | QRD and skyline diffusers | absorption as a hard box; the scattering is out of scope | 3 | — | — | medium | miss candidate |

References and gaps:

5.1 (comb.) Dupont [Dupont_2023_SoundFieldSynthesis#3] (random incidence, sound-field synthesis),
[Dupont_2023_SoundFieldSynthesis#1] (JCA prediction), [Dupont_2023_SoundFieldSynthesis#4] (Sabine, above one, marked); σ
[Dupont_2023_SoundFieldSynthesis#2] (13112 Pa·s/m²). Starting columns: page 11.3 (also the rig's `WOOL`), Komatsu at
13112 Pa·s/m², `fRef` 0.87403: `a` 1.1362, `b` −0.043932, `c` 0.0023911, `d` 1.2533, `e` 2.2495, `f` −0.45515, `g`
0.088044, `h` 0.14944, `thk` 0.05. Transmission through a treated wall is over-insulated up to 8.1 dB at 31.5 Hz
(stage-2b issue 1). Clouds and baffles hung in air are porous bodies (solid) or sheets with air on both sides; their
Sabine data are per object. Printed: glass-fibre boards van Hout [VanHout_2016_SuspendedCeilings#14] to
[VanHout_2016_SuspendedCeilings#17] (100 kg/m³, 15–100 mm, Type A, Sabine, no σ; above one at 250 Hz–4 kHz for
[VanHout_2016_SuspendedCeilings#16] and [VanHout_2016_SuspendedCeilings#17]), [VanHout_2016_SuspendedCeilings#18] (90 mm
batts, 15 kg/m³); Froger [Froger_2026_NedpendladeUndertak#4] (JCA set of a 40 mm glass wool from the manufacturer, σ
63 kPa·s/m², catalog). Gaps: 31.5/63 Hz; 125 Hz only as a model.

5.2 (comb.) Kosten [Kosten_1960_InternationalComparison#7] (random incidence from the tube, locally reacting), Kuhl
[Kuhl_1960_EdgeEffect#2]; Sabine: Kosten [Kosten_1960_InternationalComparison#4] (round robin), Kuhl
[Kuhl_1960_EdgeEffect#4] (both above one, marked). The rig's `ROCKWOOL` is Miki at 32.8 kPa·s/m² (`fRef` 0.87403, `a`
1.31823, `b` −0.0125593, `c` 0.00481048, `d` 1.80573, `e` 0.823363, `f` 0.0954384, `g` 0.221513, `h` −0.0150261), a
leaf-test fill, not fitted to Kosten's product. Printed: as 5.1 (van Hout, Froger). Gap: σ.

5.3 (comb.) Dupont [Dupont_2023_SoundFieldSynthesis#8], [Dupont_2023_SoundFieldSynthesis#6],
[Dupont_2023_SoundFieldSynthesis#9]; σ [Dupont_2023_SoundFieldSynthesis#7]. Fitted-σ check 10.3 against 10.6 kPa·s/m²
(engine notes 9.3).

5.4 BASF [BASF_2021_BasotectGplus#3] (Sabine, mounting not stated, marked); σ range Cox
[Cox_2016_AbsorbersDiffusers#76].

5.5 σ: Garai [Garai_2005_PolyesterModel#1] to [Garai_2005_PolyesterModel#38] (as 3.7). α: Szelag
[Szelag_2022_FurnitureScaleModel#2], [Szelag_2022_FurnitureScaleModel#3] (PET felt on and under furniture, modelled,
type not stated: printed).

5.6 PTB#407 [PTB_2012_AbsorptionTable#62] (50 mm wood wool set in mortar, Sabine, 0.08–0.65); σ 500–2000 Pa·s/m² for
350–500 kg/m³ (Masovic [Masovic_2021_RoomAcousticsNotes#5], a secondary compilation). A tier 2 until σ of a product is
measured, then 1 by the porous route.

5.7 No typed measurement with an air gap: compilation Cox [Cox_2016_AbsorbersDiffusers#44] (25 mm fibreglass on 25 mm
air, type not stated); practical Ecophon [Ecophon_2021_MasterBrochure#1] (40 mm glass wool at 200 mm o.d.s.); GIK
[GIK_2020_B4TestResults#2] (per object in corners), [GIK_2020_B4TestResults#1] (A-mount on the floor, no air gap; Sabine
above one).

5.8 Per object in corners (placement values, printed): GIK [GIK_2013_244BassTrapRAL#1], [GIK_2022_TurboTrapRAL#1];
Auralex [Auralex_2021_LENRDCornerRAL#1]; Primacoustic [Primacoustic_2008_FullTrapRAL#1] (Sabine, above one, marked); the
GIK note [GIK_2020_B4TestResults#2] reads 0.6 m² flat against 1.34 m² in the corner at 125 Hz. Printed neighbours: GIK
D-mount [GIK_2017_ImpressionDMount#1] (Sabine above one at 125–250 Hz; 8 kHz equals 4 kHz) and J-mount
[GIK_2017_ImpressionJMount#1] (per object).

5.9 (comb.) Shearer [Shearer_2016_AbsorptionBelow100Hz#3], [Shearer_2016_AbsorptionBelow100Hz#4] (Sabine); f₀
[Shearer_2016_AbsorptionBelow100Hz#1], [Shearer_2016_AbsorptionBelow100Hz#2]. Starting columns: the rig's `MDF` leaf.
Printed neighbour (5.9, 5.10): unperforated 6 mm birch plywood over 300 mm, Jang [Jang_2024_PerforatedBirchPlywood#1].
Gaps: `rr`; panel conversion.

5.10 (comb.) Sakagami [Sakagami_2007_NonUniformCavity#2], [Sakagami_2007_NonUniformCavity#3],
[Sakagami_2007_NonUniformCavity#4] (Sabine; 30/90/150 mm); f₀ [Sakagami_2007_NonUniformCavity#5],
[Sakagami_2007_NonUniformCavity#6], [Sakagami_2007_NonUniformCavity#7].

5.11 (comb.) Fuchs [Fuchs_2005_LowFrequencyDomain#3] (CPA on melamine foam), [Fuchs_2005_LowFrequencyDomain#2] (on
polyester fleece) (Sabine, octave means; above one at 125 Hz for [Fuchs_2005_LowFrequencyDomain#2]: placement);
[Fuchs_2013_AppliedAcoustics#13], [Fuchs_2013_AppliedAcoustics#14] are the same two tests (equal at 63 and 125 Hz since
the octave-mean correction, centre thirds above), not independent: printed. Resonates on the fill's frame stiffness ("a
fill bonded to the plate acts through its own stiffness", research handoff), which a rigid-frame fluid lacks; the
steel's coincidence is not a leaf's.

5.12 Nell [Nell_TIP_TieftonAbsorber#1], [Nell_TIP_TieftonAbsorber#2] (Sabine, marked); f₀
[Nell_2019_TieffrequenteAbsorption#1] (20.5 Hz); placement moves 125 Hz from 0.41 to 0.64. Resonates on the mat's
stiffness.

5.13 (comb.) Fuchs [Fuchs_2006_MPPReview#7], [Fuchs_2006_MPPReview#6] (Sabine), [Fuchs_2006_MPPReview#8],
[Fuchs_2006_MPPReview#9]. The stated open area 0.45 % does not match the stated grid (0.64 %). Printed: Rusher
[Rusher_2022_PerforatedMetalTileRAL#1] (25 % open, not micro-perforated, fibrous pad, E-400); Borelli
[Borelli_2021_PerforatedMetalPorous#1] to [Borelli_2021_PerforatedMetalPorous#8] (normal incidence, 4.9–30 % open steel
on PET fibre; facings from 20 % open area are transparent, a tier-2 model input).

5.14 Cox [Cox_2016_AbsorbersDiffusers#85], [Cox_2016_AbsorbersDiffusers#150], [Cox_2016_AbsorbersDiffusers#86] (type not
stated); aprons Fuchs [Fuchs_2006_MPPReview#3], [Fuchs_2006_MPPReview#4], [Fuchs_2006_MPPReview#5] (Sabine, strips hung
from a concrete ceiling).

5.15 (comb.) Focus A (decided: A 1 at both depths) Ecophon [Ecophon_2014_FocusAODS50#1], [Ecophon_2014_FocusAODS200#1]
(20 mm glass wool, 1.5 kg/m², E-50 and E-200; CSTB, COFRAC accredited; Sabine, 10.5 m², side 3.24 m; above one at
1–4 kHz, 1.01–1.06, marked; σ 54.7 ± 8.8 kPa·s/m² as stated by the manufacturer, quoted in Emmerich
[Emmerich_2025_DataDrivenTwoMic#2]). E-200 minus E-50 = +0.38 / +0.48 / +0.11 / −0.16 / −0.01 / +0.03 (125 Hz–4 kHz,
script). The case prints both depths; if their fitted σ differ by more than 1.5 times, it marks the plenum. Printed:
Aoki [Aoki_2022_PerforatedCeilingGrazing#1] (40 mm hard mineral wool, 69.7 kg/m³, 160 mm air; JCA set
[Aoki_2022_PerforatedCeilingGrazing#3] of unstated origin, σ 32.5 kPa·s/m²; specimen size not stated; 1 kHz blank).
Printed, tier 2, other tiles without σ: Ecophon (12 reports, among them [Ecophon_2003_AccessAThirdOctave#1], the thirds
of the 25 mm Access A at 200 mm), Rockfon (5), Armstrong [Armstrong_2025_ULAcousticalSummary#1],
[Armstrong_2025_ULAcousticalSummary#2], Vercammen [Vercammen_2007_OpenCeilings#1] to [Vercammen_2007_OpenCeilings#3],
Froger [Froger_2026_NedpendladeUndertak#1] to [Froger_2026_NedpendladeUndertak#3]; tiles with measured Rs at 300–800 mm
Goecke [Goecke_2023_AbgehaengteUnterdecken#1] to [Goecke_2023_AbgehaengteUnterdecken#8] (σ = Rs/d computed by the
review; 63 Hz outside ISO 354); front and back faces van Hout [VanHout_2016_SuspendedCeilings#1] to
[VanHout_2016_SuspendedCeilings#13] (the back faces: plenum-face absorption for the box of air, a tier-2 input).
Practical: Ecophon [Ecophon_AccessA_ODS200#1] (200 mm plenum), Nilsson [Nilsson_2007_NTTR606#1],
[Nilsson_2007_NTTR606#2], [Nilsson_2007_NTTR606#3]; compilation Cox [Cox_2016_AbsorbersDiffusers#6],
[Cox_2016_AbsorbersDiffusers#7], Scheibler [Scheibler_2018_Pyroomacoustics#28]. 50–80 Hz thirds lie outside ISO 354
(printed). Gaps: one tile with σ at three or more plenum depths; Goecke full data and the WSP series (on request);
31.5 Hz.

5.16 (comb.) Aoki [Aoki_2022_PerforatedCeilingGrazing#2] (12.5 mm, 8 mm holes at 18 mm, 15.5 % open, fleece Rs
265 Pa·s/m [Aoki_2022_PerforatedCeilingGrazing#5], 40 mm wool σ 5 kPa·s/m² [Aoki_2022_PerforatedCeilingGrazing#4] as JCA
input, 200 mm; Sabine from a figure: 1p); Heggemann [Heggemann_2026_PositionResonanzabsorber#1] (8/18 R over an empty
100 mm cavity, 9.49 m², from 63 Hz, figure, a bachelor thesis, low). Practical: Rigips
[Rigips_2026_RigitoneActivAir818R#3], [Rigips_2026_RigitoneActivAir818R#1]; Hengst
[Hengst_2018_KlassenraumAbsorberverteilung#2] (10.5 % cassette, 50 mm wool, 250 mm); PTB#316
[PTB_2012_AbsorptionTable#29] (Sabine). Gap: perforated gypsum at several plenum depths.

5.17 Fuchs [Fuchs_2013_AppliedAcoustics#39], [Fuchs_2013_AppliedAcoustics#38], [Fuchs_2013_AppliedAcoustics#40] (Sabine,
figure). Printed neighbours (no steel): slats Riionheimo [Riionheimo_2018_SlatStructures#1] to
[Riionheimo_2018_SlatStructures#23], Ballagh [Ballagh_2002_PerforatedSlotted#1] to [Ballagh_2002_PerforatedSlotted#3],
WEAL [WEAL_2015_PlybooLinearSoundLS11#1], Decustik [Decustik_2015_GroovedPanelD004#1],
[Decustik_2015_GroovedPanelD004#2], [Decustik_2015_GroovedPanelD006#1], [Decustik_2015_GroovedPanelD006#2].

5.18 Compilation: Cox [Cox_2016_AbsorbersDiffusers#1], [Cox_2016_AbsorbersDiffusers#2] (1D QRD),
[Cox_2016_AbsorbersDiffusers#3], [Cox_2016_AbsorbersDiffusers#4] (2D QRD; covered and uncovered).

### 2.6 Class 6: outdoor

**Ground model (decided route).** The ground classes are defined by Delany-Bazley at a class σ (Fuchs
[Fuchs_2025_GroundImpedanceNTACOU104#2] to [Fuchs_2025_GroundImpedanceNTACOU104#9]: the Harmonoise/Nord2000 impedance
classes A–H of the one-parameter model; their CNOSSOS ground factors G: Kephalopoulos [Kephalopoulos_2012_CNOSSOS#1] to
[Kephalopoulos_2012_CNOSSOS#8]). Delany-Bazley cannot be written into the columns over most of the band: in the engine's
mapping its compressibility needs a gain (`c` < 0) below ρ₀f/σ ≈ 0.014, at 6 to 25 of 25 thirds (closed form in the
engine's mapping, thirds 31.5 Hz–8 kHz: loose gravel 6, snow 7, forest floor 11, grass C 15, grass D 19, soil 23, gravel
road 25, porous asphalt 14), and `c` ≥ 0 is enforced when the table is read. The standard route is therefore a **Miki
fit with passive columns to the class impedance** (Delany-Bazley's surface impedance and wavenumber); Delany-Bazley
itself is printed. The reflection target stays physical, since a Delany-Bazley half-space keeps a positive surface
resistance. At short range the ground acts nearly rigid at low frequency (closed-form estimate for a locally reacting
half-space, Weyl–van der Pol: 63–250 Hz excess attenuation +5.3 to +6.0 dB for σ 12.5–200 kPa·s/m² at 20–200 m). The
class σ are local-reaction values while a traced porous solid reacts extendedly, and the engine reflects plane waves;
both limits are in the modeling guidelines.

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1 | `ground_snow` | snow cover | porous half-space (thick traced solid) | 2 | — | — | high (winter sites) | wish |
| 6.2 | `ground_forest_floor` | forest floor, moss, heather | porous half-space; a layer form is the alternative | 2 | — | — | medium | wish |
| 6.3 | `ground_grass` | lawns, pasture, sports fields | porous half-space | 2 | — | — | high | wish |
| 6.4 | `ground_soil` | compacted soil, earth | porous half-space | 2 | — | — | medium | wish |
| 6.5 | `ground_gravel_loose` | loose gravel, gravel beds, gabion fill | porous half-space, or a layer on soil | 2 | — | — | medium | wish |
| 6.6 | `ground_gravel_compacted` | gravel roads, car parks | porous half-space | 2 | — | — | medium | wish |
| 6.7 | `asphalt`, `asphalt_porous_40mm` (proposed thickness) | dense asphalt and concrete paving; porous asphalt | dense: rigid traced solid with a small face absorption; porous: thin porous sheet on the `asphalt` solid | 1 (dense 250 Hz–1 kHz; porous 500 Hz–1 kHz); 2 outside | — | dense 250 Hz–1 kHz; porous 500 Hz–1 kHz | high | wish |
| 6.8 | `vegetation` | foliage of trees, hedges, shrubs | absorbing volume (2.4.3) around the foliage; `alpha`, `alphaB` fitted to the excess attenuation per meter | — | 2 | 63 Hz–8 kHz, paths 20–200 m | medium | wish |

References and gaps:

6.1 Class A 12.5 kPa·s/m² (Fuchs [Fuchs_2025_GroundImpedanceNTACOU104#2]; G 1, Kephalopoulos
[Kephalopoulos_2012_CNOSSOS#1]); Ishida [Ishida_1965_SnowAcoustic#1] (2–25 kPa·s/m², laboratory); Piercy
[Piercy_1981_PropagationReview#3] (15–30); Bashir [Bashir_2014_PhDRoughMixedImpedance#154]; FHWA
[FHWA_2010_GroundPavementEffects#8] (powder 10, granular 40). Snow of finite depth is a layer (proposed).

6.2 Class B 31.5 kPa·s/m² (Fuchs [Fuchs_2025_GroundImpedanceNTACOU104#3]; G 1, Kephalopoulos
[Kephalopoulos_2012_CNOSSOS#2]); Bashir [Bashir_2014_PhDRoughMixedImpedance#1], [Bashir_2014_PhDRoughMixedImpedance#2],
[Bashir_2014_PhDRoughMixedImpedance#3], [Bashir_2014_PhDRoughMixedImpedance#4], [Bashir_2014_PhDRoughMixedImpedance#5],
[Bashir_2014_PhDRoughMixedImpedance#6], [Bashir_2014_PhDRoughMixedImpedance#8],
[Bashir_2014_PhDRoughMixedImpedance#131], [Bashir_2014_PhDRoughMixedImpedance#132],
[Bashir_2014_PhDRoughMixedImpedance#133], [Bashir_2014_PhDRoughMixedImpedance#134]; Nordtest
[Nordtest_1999_NTACOU104#1], [Nordtest_1999_NTACOU104#7]; Piercy [Piercy_1981_PropagationReview#5]. Half-space fits miss
by 24.5 / 26.3 / 28.1 dB ([Bashir_2014_PhDRoughMixedImpedance#2], [Bashir_2014_PhDRoughMixedImpedance#4],
[Bashir_2014_PhDRoughMixedImpedance#6]), layer fits by 17.0 / 11.8 / 6.0 dB ([Bashir_2014_PhDRoughMixedImpedance#1],
[Bashir_2014_PhDRoughMixedImpedance#3], [Bashir_2014_PhDRoughMixedImpedance#5]; fit errors in the data notes).

6.3 Class C 80 (Fuchs [Fuchs_2025_GroundImpedanceNTACOU104#4]; G 1, Kephalopoulos [Kephalopoulos_2012_CNOSSOS#3]), class
D 200 (Fuchs [Fuchs_2025_GroundImpedanceNTACOU104#5]; G 1, Kephalopoulos [Kephalopoulos_2012_CNOSSOS#4]); Piercy
[Piercy_1981_PropagationReview#6] (150–300); Bashir [Bashir_2014_PhDRoughMixedImpedance#9],
[Bashir_2014_PhDRoughMixedImpedance#10], [Bashir_2014_PhDRoughMixedImpedance#11],
[Bashir_2014_PhDRoughMixedImpedance#12], [Bashir_2014_PhDRoughMixedImpedance#13],
[Bashir_2014_PhDRoughMixedImpedance#14]; Van Renterghem [VanRenterghem_2012_VegetationBelts#1],
[VanRenterghem_2015_NaturalMeans#3]; FHWA [FHWA_2010_GroundPavementEffects#4], [FHWA_2010_GroundPavementEffects#5],
[FHWA_2010_GroundPavementEffects#6], [FHWA_2010_GroundPavementEffects#8]; Nordtest [Nordtest_1999_NTACOU104#5].
Attenborough [Attenborough_2016_ExploitingGroundEffects#1] (summer, 80 kPa·s/m², layer 0.035 m) and
[Attenborough_2016_ExploitingGroundEffects#2] (winter, 200 kPa·s/m², layer 0.011 m) are layer fits: printed checks of a
layer form (4.9).

6.4 Class E 500 (Fuchs [Fuchs_2025_GroundImpedanceNTACOU104#6]; G 0.7, Kephalopoulos [Kephalopoulos_2012_CNOSSOS#5]);
Piercy [Piercy_1981_PropagationReview#4] (rain-packed earth 4000–8000); Bashir [Bashir_2014_PhDRoughMixedImpedance#7];
Attenborough [Attenborough_2016_ExploitingGroundEffects#3] (soil with winter wheat, 170); Nordtest
[Nordtest_1999_NTACOU104#2], [Nordtest_1999_NTACOU104#3]; FHWA [FHWA_2010_GroundPavementEffects#7],
[FHWA_2010_GroundPavementEffects#8].

6.5 Van Renterghem [VanRenterghem_2015_NaturalMeans#4] (gravel 10 kPa·s/m²); Bashir
[Bashir_2014_PhDRoughMixedImpedance#121], [Bashir_2014_PhDRoughMixedImpedance#122],
[Bashir_2014_PhDRoughMixedImpedance#123], [Bashir_2014_PhDRoughMixedImpedance#124],
[Bashir_2014_PhDRoughMixedImpedance#125], [Bashir_2014_PhDRoughMixedImpedance#126],
[Bashir_2014_PhDRoughMixedImpedance#127], [Bashir_2014_PhDRoughMixedImpedance#128],
[Bashir_2014_PhDRoughMixedImpedance#129], [Bashir_2014_PhDRoughMixedImpedance#130] (layers 0.04–0.22 m and
semi-infinite), [Bashir_2014_PhDRoughMixedImpedance#155]. Loose and compacted gravel differ by 2000 / 10 = 200 in σ.

6.6 Class F 2000 (Fuchs [Fuchs_2025_GroundImpedanceNTACOU104#7]; G 0.3, Kephalopoulos [Kephalopoulos_2012_CNOSSOS#6]);
Nordtest [Nordtest_1999_NTACOU104#2]; Piercy [Piercy_1981_PropagationReview#7].

6.7 Dense: class G 20 000 [Fuchs_2025_GroundImpedanceNTACOU104#8] (G 0, [Kephalopoulos_2012_CNOSSOS#7]), class H 200 000
[Fuchs_2025_GroundImpedanceNTACOU104#9] (G 0, [Kephalopoulos_2012_CNOSSOS#8]); FHWA [FHWA_2010_GroundPavementEffects#1],
[FHWA_2010_GroundPavementEffects#2]; Piercy [Piercy_1981_PropagationReview#1]. (row) dense, hard face 250 Hz–1 kHz:
Ongel [Ongel_2007_AcousticalAbsorptionAsphalt#8], [Ongel_2007_AcousticalAbsorptionAsphalt#9] (field cores, normal
incidence, 0.12 / 0.05 / 0.05 and 0.02 / 0.02 / 0.03), [Ongel_2007_AcousticalAbsorptionAsphalt#1] (mean 0.04 over
200–1700 Hz, 18 sections); Rochat [Rochat_2012_PavementAbsorptionUS#2] (in situ, 0.02 at 500 Hz–1 kHz, low). (row)
porous 40 mm, 500 Hz–1 kHz: cores of 36.2–43.6 mm Praticò [Pratico_2021_AcousticFunctionalPorousAsphalt#1],
[Pratico_2021_AcousticFunctionalPorousAsphalt#7], [Pratico_2021_AcousticFunctionalPorousAsphalt#15],
[Pratico_2021_AcousticFunctionalPorousAsphalt#17] (within 10 % of 40 mm, script; normal incidence, means 0.14–0.19 over
400–630 Hz and 0.51–0.56 over 800–1600 Hz, first peaks 0.81–0.92 at 1.07–1.18 kHz; σ of the upper face on the even ids,
ISO 9053 part uncertain); Bell [Bell_2024_PorousAsphaltAbsorption#2] (EPA7 40 ± 3 mm, in situ, 500 Hz–2 kHz, low);
printed: Hübelt [Huebelt_2006_PorousRoadSurfaces#1] to [Huebelt_2006_PorousRoadSurfaces#12] (in situ, thickness not
published). Model inputs: Bashir [Bashir_2014_PhDRoughMixedImpedance#142] (67.5 kPa·s/m², porosity 0.18, tortuosity
2.36, layer 36 mm), [Bashir_2014_PhDRoughMixedImpedance#141] (2–15 kPa·s/m², non-acoustic); σ and tortuosity Praticò
[Pratico_2021_PoreShapeSustainability#1] to [Pratico_2021_PoreShapeSustainability#19], Kuijpers
[Kuijpers_2000_TwoLayerPorousModeling#1] to [Kuijpers_2000_TwoLayerPorousModeling#4]. Tier 2 outside the asserted bands
(ground model). Gaps: below 200 Hz; a clean 4 kHz octave; dense spectra beyond two field sets.

6.8 ISO 9613-2 Annex A through Witte [Witte_2018_ForestAttenuation#3] (0.02 dB/m at 63 Hz to 0.12 dB/m at 8 kHz),
[Witte_2018_ForestAttenuation#2] (10–20 m, absolute dB); Witte [Witte_2018_ForestAttenuation#5] (vegetation strip); Van
Renterghem [VanRenterghem_2014_Hedges#1] (hedges). Tier 2: an engineering table of a standard, typed catalog. The
air-matched volume is right for foliage, unlike for furniture.

### 2.7 Auxiliary rows

| name | use | recipe | A | T | status |
| --- | --- | --- | --- | --- | --- |
| `cavity_air` | the air of a box behind a leaf, a curtain, a panel, a floor on joists, a suspended ceiling | traced closed box; `a` 1, `e` 1.000001 (rig `AIRBOX`): a material identical to air is air | definition | definition | wish |
| `water_20c` | ponds and pools | traced solid; c 1482.36 m/s, ρ 998.207 kg/m³, Z 1.4797e6 Pa·s/m (NPL [NPL_2025_SoundPureWaterPhysics#1], EURAMET [EURAMET_2025_GravimetricVolumeCG19#1], [EURAMET_2025_GravimetricVolumeCG19#2]) | 1 | 1 | wish |

The rig tests `water_20c` as a sheet (its stack rule refuses a lossless wave medium with a critical angle as a traced
solid); the half-space form is untested, and a thick-solid case is proposed for phase 3.

## 3. Test references that are combinations (measured only)

Each combination is a rig test of the materials involved, room side first. "Rig today" says whether the rig can build
it now.

| combination | materials tested | measured whole (id, type) | rig today | notes |
| --- | --- | --- | --- | --- |
| carpet on concrete | `carpet_tile_11mm`, `carpet_tufted_12mm`, `concrete` | Salford [Salford_2020_CarpetTileSpecimen#1], BCTC [BCTC_2014_EcosoftCarpetTile#1] (Sabine); Broda [Broda_2025_TuftedCarpets#3] (normal incidence); compilations Cox [Cox_2016_AbsorbersDiffusers#148], [Cox_2016_AbsorbersDiffusers#62], [Cox_2016_AbsorbersDiffusers#166]; cushion- and hard-backed tiles CSIRO [CSIRO_2019_WellbacComfortTile#1], [CSIRO_2019_WellbacComfortPlus#1], [CSIRO_2019_WellbacFunctionTile#1], TFI [TFI_2018_NylonCarpetTileCushion#1], Kiwa [Kiwa_2021_EForceComfortBack#1], Salford [Salford_2022_FlotexNext#1], Milliken [Milliken_ND_CarpetTileCertificates#1], [Milliken_ND_CarpetTileCertificates#2], Bietz [Bietz_2025_CombiningRoomVolumes#2] (Sabine; other products, printed) | yes | transmission touches stage-2b issue 1; 63 Hz a hard face (3.1) |
| carpet with underlay on concrete | `carpet_*`, `underlay`, `concrete` | Broda [Broda_2025_TuftedCarpets#4] (normal); compilations PTB#156–158 [PTB_2012_AbsorptionTable#5], [PTB_2012_AbsorptionTable#6], [PTB_2012_AbsorptionTable#7], Cox [Cox_2016_AbsorbersDiffusers#63]; printed: Ford woven and tufted carpets on underlay [Ford_1980_CarpetsUnderlays#1], [Ford_1980_CarpetsUnderlays#4] (Sabine, figure, builds not stated), compilation Cox [Cox_2016_AbsorbersDiffusers#167] to [Cox_2016_AbsorbersDiffusers#177] | sheet \| solid \| solid: yes | — |
| rug on a hard finish | `carpet_tile_11mm` on `parquet` solid on `concrete` | none | yes | plausibility |
| porous absorber on a wall | `glass_wool_50mm`, `pu_foam_50mm`, `rock_wool_50mm` on `concrete` | Dupont [Dupont_2023_SoundFieldSynthesis#3], [Dupont_2023_SoundFieldSynthesis#8]; Kosten [Kosten_1960_InternationalComparison#7] | yes (rig case exists for glass wool) | — |
| porous absorber with an air gap | wool \| `cavity_air` \| wall | none typed | yes | plausibility only |
| panel absorber over a cavity | `mdf_leaf_6mm`, `plywood_leaf_3mm`, `cavity_air`, fill | Shearer [Shearer_2016_AbsorptionBelow100Hz#3], [Shearer_2016_AbsorptionBelow100Hz#4]; Sakagami [Sakagami_2007_NonUniformCavity#2], [Sakagami_2007_NonUniformCavity#3], [Sakagami_2007_NonUniformCavity#4] (Sabine) | yes | panel conversion pending |
| curtain in front of a wall | `curtain_flat_0_2mm`, `curtain_pleated_0_8mm`, `cavity_air` | DelRey [DelRey_2013_CurtainFullness#14], [DelRey_2013_CurtainFullness#16]; Zimmer+Rohde [ZimmerRohde_2022_Fabric10936Curtain#1], [ZimmerRohde_2020_KakaduCurtain#1]; ADO [ADO_2018_CannyCurtain#1], [ADO_2020_BreakingDawnCurtain#1] | yes | fabric route pending |
| curtain in front of a window | curtain \| `cavity_air` \| glazing sheet | none | yes; transmission printed (stage-2b issue 1) | — |
| drywall, precise recipe | `gypsum_leaf_12_5mm`, `cavity_air`, `glass_wool` | Halliwell [Halliwell_1998_GypsumBoardWalls#24], [Halliwell_1998_GypsumBoardWalls#47] (the build), [Halliwell_1998_GypsumBoardWalls#27], [Halliwell_1998_GypsumBoardWalls#30], [Halliwell_1998_GypsumBoardWalls#33], [Halliwell_1998_GypsumBoardWalls#35], [Halliwell_1998_GypsumBoardWalls#48] (other builds); Fuchs [Fuchs_2005_LowFrequencyDomain#4] | yes; transmission printed (stage-2b issue 3) | the same walls test the lumped rows (an alternative model, 1.7.9) |
| plasterboard on frame, room side | `gypsum_leaf_12_5mm`, `cavity_air` | PTB#341 [PTB_2012_AbsorptionTable#35], PTB#342 [PTB_2012_AbsorptionTable#36]; Bradley [Bradley_1997_GypsumCavityAbsorption#2] | yes | — |
| suspended ceiling over a plenum | `ceiling_tile_20mm`, `cavity_air`, `concrete` | Focus A Ecophon [Ecophon_2014_FocusAODS50#1], [Ecophon_2014_FocusAODS200#1]; Aoki [Aoki_2022_PerforatedCeilingGrazing#1] (printed); printed series (5.15) (Sabine) | yes (sheet \| box \| solid) | the references of 5.15 |
| perforated gypsum over a plenum | `perforated_gypsum_12_5mm`, `cavity_air`, wool, `concrete` | Aoki [Aoki_2022_PerforatedCeilingGrazing#2] (with wool), Heggemann [Heggemann_2026_PositionResonanzabsorber#1] (empty cavity) (Sabine) | yes | 1p |
| dry lining on masonry | `gypsum_leaf_12_5mm`, `cavity_air`, masonry | Zeitler [Zeitler_2015_ConcreteBlockRR334#14], [Zeitler_2015_ConcreteBlockRR334#15], [Zeitler_2015_ConcreteBlockRR334#16], [Zeitler_2015_ConcreteBlockRR334#17], [Zeitler_2015_ConcreteBlockRR334#18], [Zeitler_2015_ConcreteBlockRR334#19], [Zeitler_2015_ConcreteBlockRR334#20], [Zeitler_2015_ConcreteBlockRR334#21], [Zeitler_2015_ConcreteBlockRR334#22], [Zeitler_2015_ConcreteBlockRR334#23], [Zeitler_2015_ConcreteBlockRR334#24], [Zeitler_2015_ConcreteBlockRR334#25], [Zeitler_2015_ConcreteBlockRR334#26], [Zeitler_2015_ConcreteBlockRR334#27], [Zeitler_2015_ConcreteBlockRR334#28], [Zeitler_2015_ConcreteBlockRR334#29], [Zeitler_2015_ConcreteBlockRR334#30], [Zeitler_2015_ConcreteBlockRR334#31], [Zeitler_2015_ConcreteBlockRR334#32], [Zeitler_2015_ConcreteBlockRR334#33], [Zeitler_2015_ConcreteBlockRR334#34] (ΔR); Hopkins [Hopkins_2007_SoundInsulation#65], [Hopkins_2007_SoundInsulation#64]; BIA [BIA_2000_TN5AClayMasonry#9] / [BIA_2000_TN5AClayMasonry#7], [BIA_2000_TN5AClayMasonry#10] / [BIA_2000_TN5AClayMasonry#12] | yes (leaf \| box \| wall; the bare wall inline) | issue 1 for a fill |
| plaster on a bare wall | `plaster_15mm` on a bare brick written inline | BIA [BIA_2000_TN5AClayMasonry#7] / [BIA_2000_TN5AClayMasonry#8], [BIA_2000_TN5AClayMasonry#15] / [BIA_2000_TN5AClayMasonry#14]; Warnock [Warnock_1984_ConcreteBlockWalls#4] / [Warnock_1984_ConcreteBlockWalls#5] (one face) | sheet \| solid: yes | gives the plaster increment of 1.2 |
| plaster on both faces of a block | plaster sheets on both faces of a block solid | Hopkins [Hopkins_2007_SoundInsulation#68] (printed); Warnock [Warnock_1984_ConcreteBlockWalls#2] / [Warnock_1984_ConcreteBlockWalls#3] (porous hollow block, sealing; printed) | yes (exact); the far plaster traced is the engine-limit fixture of three bonded bodies | — |
| box window | `glass_leaf_4mm` \| `cavity_air` \| `glass_leaf_4mm` | Quirt [Quirt_1981_WindowsTLMeasurements#10], [Quirt_1981_WindowsTLMeasurements#9], [Quirt_1981_WindowsTLMeasurements#12] | yes | — |
| parquet on concrete | `parquet`, `concrete` | PTB#394 [PTB_2012_AbsorptionTable#48] | yes | — |
| solid door, seal states | `door_solid_40mm` in a wall | Quirt [Quirt_1986_DoubleDoors#2] to [Quirt_1986_DoubleDoors#4]; Morin [Morin_1993_AccessDoorsCMHC#1] / [Morin_1993_AccessDoorsCMHC#2] | no: a leak is not a plane layer | printed only |
| hollow-core door with a bottom gap | `door_hollow_core_40mm` | Lee [Lee_2010_InteriorDoorAperture#2] to [Lee_2010_InteriorDoorAperture#14] (gaps 2–17 mm; 6 mm: Rw 23 against 30) | no: as above | printed; modelling guidance |
| wood-joist floor-ceiling | `board_leaf_22mm` \| `cavity_air` \| `gypsum_leaf_12_5mm` | Berendt [Berendt_1967_NBSMonograph77#34] to [Berendt_1967_NBSMonograph77#46], NBS [NBS_1955_BMS144SoundInsulation#5] to [NBS_1955_BMS144SoundInsulation#32] (R from 125 Hz, warble tones) | yes (leaf \| box \| leaf at the exit); joist bridging not modelled | miss candidate for T |
| carpet on timber | `carpet_tile_11mm` on `board_leaf_22mm` or `laminate_leaf_8mm` | none (request 6) | **no: three co-located faces** (stage-2b issue 5) | miss candidate |

## 4. Open design questions for phase 3

1. **The generic solid across thickness.** One generic fit, asserted where the trend miss stays inside the tolerance.
   First on `masonry_heavy` (the 95–305 mm series), then concrete within 150–203 mm. `coiF`, `resF`, `att` are fixed
   columns while fc·h of 17.1 Hz·m gives 171 Hz at 100 mm and 28.5 Hz at 600 mm; if the miss is too large, a
   thickness-scaled coincidence goes to the stage-2b list.
2. **`side` values.** The rule is decided; the typical sizes of 1.8 are proposed.
3. **Objects.** The object test's design and cost; the porous columns fitted for a soft body; the asset size variants;
   the kind proposed for clothes racks, plants, persons and audiences; the per-area convention for seat and audience
   blocks (now 1p).
4. **Reference row per class.** Glass (EN 12758 generic against Quirt), carpets (Cox minimum / mean / maximum),
   concrete (Litvin against Mahn).
5. **Free porous bodies.** A cushion, a cloud or a bass trap as a traced porous solid in air has no reference as a
   free body; the object test and the per-object data are the only checks.
6. **Lumped rows against recipes for building assemblies.** Stud partitions of other builds, floating floors: one
   sheet fitted to the measured construction, or the recipe. Mass-law bodies do not resonate with a fill.
7. **A solid drywall row at other drawn thicknesses.** Its mass is mostly the boards, while the mass term of a traced
   solid scales with the traced thickness; the measured depth effect comes from the cavity resonance.
8. **Masonry class boundaries.** Whether three classes cover what a modeler sees. Sand-lime: closed, Weise
   [Weise_2003_PruefstandAbweichungen#1] is the wall (1.2).
9. **Ground.** One grass row or two by season; a layer form for snow, loose gravel and forest floor.
10. **Conversions first.** The routes for panel absorbers, microperforated panels, hanging fabrics, green walls and
    per-area blocks decide every "1p" entry.
11. **One material per closed body, finishes as sheets.** To be confirmed in phase 3.

## 5. Procedure for adding a material later

1. **Who decides.** The maintainer decides whether a material enters the list; anyone may propose it.
2. **Fields of a new entry.** Name (1.6), class, intended use, recipe, A and T tiers, valid range, references
   (research row id at the pinned research commit, coefficient type, row or combination reference), gaps, impact,
   status `wish`; a lumped entry states its surface mass; a mass-law row its typical `side` (1.8). The rules of 1.7
   and the guidelines' checklist apply.
3. **References.** Search the research library first (read-only). If nothing usable exists, write a draft research
   request in section 6. The maintainer passes it to the research manager; the answer comes back as data rows with
   source, location and coefficient type.
4. **Phase-3 pickup.** By the work order of 1.4. One worker builds its row family and its lean case by the procedure
   of 1.10; engine-limited checks are marked `ENGINE LIMIT (<name>)`; the maintainer reviews and commits; the status moves to
   `confirmed`, or to `recorded miss`. Object rows also get their generated asset.
5. **CSV traps.** An empty cell reads 0, not the column default; a text cell in a numeric column is an error; a
   material file replaces the built-in table.

## 6. Draft research requests, ordered by impact

0. **A stable id per data row** (done at research commit d501809). The column `id` of
   `data/material_properties.csv` survives regeneration and sorting; the material table's source comments cite it.
1. **Sofas, armchairs, beds** (classes 3 and 4; high). Per-object equivalent absorption area, ISO 354, 63 Hz–4 kHz,
   with the object's size, upholstery and placement. Manufacturer reports acceptable. Blocks 3.8, 4.1, 4.2.
   **Research answer (2026-10-27):** answered in part (single bed, leather sofa, acoustic fabric series); remaining: an
   ordinary fabric domestic sofa and armchair (a factor 2 to 3 open at 1 kHz), a double bed, bedding.
2. **Wooden floor on joists** (class 1; high). Measured absorption with build (board thickness, joist depth, cavity
   fill), Sabine or random incidence, 63 Hz–4 kHz; laboratory R of the bare floor if available. Not blocking.
   **Research answer (2026-10-27):** answered with combination R only; remaining: absorption with build, R of a bare
   floor, 63 Hz.
3. **Doors** (class 1; medium). Laboratory R and surface mass of a solid timber door and of a hollow-core door with
   build, size and seals, 63 Hz–4 kHz; absorption of a hollow-core door. Blocks the transmission of both door rows.
   **Research answer (2026-10-27):** answered for R (solid and hollow-core); remaining: absorption of any door, the
   surface mass of a modern hollow-core leaf, a 50 Hz third.
4. **Suspended absorbing ceilings with build** (class 5; high in venues). Sabine or random-incidence α of
   mineral-fiber or glass-wool ceiling tiles with thickness, density or σ, and plenum depth, 63 Hz–4 kHz.
   **Research answer (2026-10-27):** answered (Focus A with σ at two depths, 17 further reports); remaining: σ at three
   or more depths; Goecke and WSP on request.
5. **Perforated and slotted panels with full build** (class 5; medium). Sabine or random-incidence α with geometry,
   open area, plate thickness, cavity depth and fill, sample size, 125 Hz–4 kHz.
   **Research answer (2026-10-27):** answered in part (one full-geometry curve); remaining: several plenum depths, metal
   tiles with hole geometry, narrow slots, perforated wood.
6. **Carpet on a timber floor** (class 3; medium). Absorption of a carpet of stated build on boards on joists, and on
   concrete; flow resistivity of carpet piles; 63 Hz–4 kHz. Not blocking (the configuration is a miss candidate).
   **Research answer (2026-10-27):** not answered; remaining: the whole request (a commissioned test).
7. **Flow resistivity of carpets and underlays** (class 3; medium). With thickness and density. Turns the carpet rows
   to A 1 above 500 Hz.
   **Research answer (2026-10-27):** answered for rugs, rebond foam and nonwovens; remaining: σ of a tufted or woven
   pile, a backing, a felt or rubber underlay (closed).
8. **Floating laminate floors** (class 1; medium). Absorption of laminate on foam underlay over concrete, with build,
   63 Hz–4 kHz.
   **Research answer (2026-10-27):** not answered; remaining: the whole request (two printed neighbours).
9. **Filled bookshelves** (classes 2 and 4; medium). Absorption per object of a shelf filled with books, ISO 354,
   125 Hz–4 kHz, with the shelf's size.
   **Research answer (2026-10-27):** answered without size; remaining: the bookcase's size.
10. **Cluttered desks and shelves** (class 4; medium). Absorption per object or per area of a desk or shelf with
    typical items, ISO 354, 125 Hz–4 kHz.
    **Research answer (2026-10-27):** answered at room level only; remaining: per-object or per-area data.
11. **Masonry walls missing in the data** (class 1; medium). Laboratory R of plain aircrete and sand-lime walls with
    density and thickness; clay brick and hollow units plastered on both faces; 63 Hz–4 kHz.
    **Research answer (2026-10-27):** answered in part (sand-lime, hollow plastered both faces, aircrete); remaining:
    third-octave R of solid brick plastered both faces, hollow clay 80–200 mm plastered both faces, aircrete with its
    own mass, below 100 Hz.
12. **Absorption of asphalt** (class 6; medium). Absorption coefficient of dense and porous asphalt (in situ or tube,
    with build), 125 Hz–4 kHz.
    **Research answer (2026-10-27):** answered (dense and porous, 200 Hz–1.7 kHz); remaining: below 200 Hz, a clean
    4 kHz octave.
13. **Potted plants, clothing racks, full wardrobes** (classes 2 and 4; low). Per-object absorption area, ISO 354,
    with placement.
    **Research answer (2026-10-27):** answered for a wardrobe, a chest, a palm and firs; remaining: clothing racks,
    hanging clothes, broadleaf plants of 1.5–2 m.

(The numbering in section 2 refers to these requests by number.)
