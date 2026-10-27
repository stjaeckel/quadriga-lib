# Acoustic materials: the wish-list (draft 4)

Draft 4, 2026-10-26 (draft 0 revised after two review rounds and two independent reviews). This list is the input to
the work on the shipped acoustic material table (`data/materials_acoustic.csv`). The rows, their columns and their
tests are built later, material by material, from the entries below. Nothing here is a row yet, except where the
status says so. Everything still marked **(proposed)** is the drafter's proposal and waits for review: impact
ratings, typical panel sizes, sheet thicknesses chosen by the drafter, row names of objects, recipes for tier-3
materials and objects, the phase-3 start list.

The companion document `md/acoustic_modeling_guidelines.md` says how a venue is modeled for the acoustic tracer. Its
checklist of forbidden and limited configurations applies to every recipe below.

**Research line numbers.** Every "line n" below is the line of `data/material_properties.csv` (header = line 1) of
the research library at its commit 2dad906 (2026-10-22). The file is generated and sorted, so a regeneration moves
the lines; phase 3 remaps each cited line by its (source, material, configuration, property) with a script before it
writes a source comment. A stable id column per data row is requested from the research manager (section 6,
request 0).

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
| 1 | A measured reference of that quantity exists, with the row's own build and a known coefficient type. For absorption: a random-incidence or normal-incidence reference; a hard face (all bands up to about 0.2, used as measured: material page, section 10.5); a per-object area used like for like (2.4); or a Sabine value of a class whose conversion is decided (porous layers of known construction; mineral wool by the edge correction). A leaf on a box takes A 1 where its construction's reference is a hard face. |
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
   2000 kg/m³, line 1788, is close to concrete), the finish solid on top. Three bonded mass-law bodies are within
   ±3 dB (stage-2b issue 4). The `screed` rows stay for bare and floating screeds.
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
  needs an engine fix, mark it ISSUE; if not, assert. A comment in the case states what "reasonable" meant: the range
  of measurements and models it was judged against. The rig documents itself; this list is updated from the result.
- **ISSUE convention (decided):** a rig comparison that misses because of an engine limit carries a comment
  `ISSUE <n>:` naming the stage-2b issue and asserts nothing; the measured and the reference values go into the
  diagnostics (for a per-node comparison the largest deviation and its node stand for the reference); the issue has
  an entry in `md/acoustic_engine_stage2b_issues.md`. Known engine issues never make the suite fail. The transmission of a box of air in front of a fill is taken from the traced paths at a cap of 10
  interactions (production runs: 20 to 30); reflection runs to the engine's limit, its change at the cap is printed.
- **Tolerances (decided; may move with more data):**

  | check | against | tolerance | assertion |
  | --- | --- | --- | --- |
  | independent physics | the model written in the test (plate, transfer matrix, porous layer) | at or above 2–3 × the measured maxima of the rig's self-checks (wide gates are preferred) | CHECK |
  | measured single reference | the converted reference at its own specimen side | ±3 dB / ±0.05 (plus the conversion budget) | CHECK, or printed with ISSUE |
  | class row | the envelope of the class's walls, each shifted to the evaluated thickness by the mass law (20·log₁₀ of the surface-mass ratio; the row's surface mass is `e`·ρ₀·`thk`, a lumped row's the stated one) | ±1 dB | as above |
  | regression guard | the library's own measured result | about 1 dB / 0.01 in α | CHECK |

  Worked example: `concrete_150mm` against Litvin 76-66 (line 228, 152 mm, opening 4.37 × 2.84 m, side 3.523 m by
  script): the row is evaluated at side 3.523 m, the 2 mm difference shifts the reference by 20·log₁₀(152/150) =
  0.115 dB (script), and each octave from 125 Hz to 4 kHz is asserted within ±3 dB; 31.5, 63 Hz and 8 kHz take the
  model spread.
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
| 7 | `carpet_tile_11mm` | 1 to 500 Hz / 2 | sheet on a solid (exists) |
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
line, coefficient type, and **(row)** where they test the row or **(comb.)** where they are combination references
of section 3. R = laboratory sound reduction index; σ = flow resistivity.

### 2.1 Class 1: structural building materials

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | `concrete`, `concrete_150mm`, `concrete_200mm` | cast walls, slabs, ceilings; slab with bonded screed | solid of the full thickness (slab plus bonded screed) or sheet | 1 | 1 | 150–203 mm; R 125 Hz–4 kHz | high | confirmed |
| 1.2 | `masonry_heavy`, `masonry_heavy_150mm`, `_200mm`, `_250mm` | finished heavy wall plastered both faces (solid clay brick, dense block, sand-lime, about 1800–2400 kg/m³) | solid or sheet | 1 | 1 | about 140–241 mm (proposed from the trend); R 125 Hz–4 kHz | high | wish |
| 1.3 | `masonry_hollow`, `masonry_hollow_100mm`, `_200mm` | finished hollow-unit wall plastered both faces (hollow clay brick, hollow block of any aggregate) | solid or sheet | 1 | 1 | about 90–200 mm, set in phase 3 by the same rule | high | wish |
| 1.4 | `masonry_light`, `masonry_light_100mm`, `_150mm` | finished light wall, plastered (aircrete, expanded-clay and lightweight-aggregate solid units) | solid or sheet; 200 and 250 mm printed only | 1 | 1 | about 90–190 mm | medium | wish |
| 1.5 | `plaster_15mm` | plaster on `concrete` or `clt` | sheet on the solid; `att` 0, host's `side` | 1 | 2 | — | low | wish |
| 1.6 | `glass_4mm`, `glass_6mm` | single-pane windows | sheet only | 2 | 1 | R 125 Hz–4 kHz | high | wish |
| 1.7 | `glazing_double_24mm`, `glazing_triple_36mm` | sealed units 4-16-4 (20 kg/m²) and 4-12-4-12-4 (30 kg/m²) | sheet, one lumped body | 2 | 1 | R 125 Hz–4 kHz | high | wish |
| 1.8 | `glass_leaf_4mm` | two sashes with a wide air space (box windows) | leaf sheet \| box of cavity air \| leaf (sheet at the box's exit) | 2 | 1 | — | low | wish |
| 1.9 | `gypsum_board_12_5mm` | a single free gypsum panel; the reference row of the leaf | sheet | 2 | 1 | R 63 Hz–4 kHz | low | wish |
| 1.10 | `gypsum_leaf_12_5mm` | gypsum ceilings over a plenum, linings, the precise drywall recipe | leaf sheet on a box of cavity air | 1 (empty cavity); 1p (filled) | 2 | — | high | wish |
| 1.11 | `drywall`, `drywall_filled`; `drywall_100mm`, `drywall_125mm`, `drywall_150mm`; `drywall_filled_100mm`, `drywall_filled_125mm`, `drywall_filled_150mm` | stud partitions as a plan shows them | lumped mass-law row of the standard build (one 12.5 mm gypsum board each side, metal studs, the rest cavity, empty unless filled; boards 17.0 kg/m² from 680 kg/m³, line 1284); `thk` the visible thickness | 1 (empty); 1p (filled) | 1 for `drywall_filled_125mm` and the solid at 115 mm; 2 (double-wall model) for the rest | filled 115 mm | high | wish |
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
| 1.23 | `door_solid_40mm` (proposed thickness) | solid timber doors | sheet, lumped; surface mass set from the reference door (none in hand) | 1 | 3 | — | medium | wish |
| 1.24 | `door_hollow_core_40mm` (proposed thickness) | hollow-core interior doors | sheet, lumped; surface mass unknown | 3 | 3 | — | medium | wish |
| 1.25 | `clt`, `clt_100mm` | cross-laminated timber walls and floors | solid or sheet | 2 | 1 | 78–245 mm; R 63 Hz–4 kHz | low | wish |

References and gaps:

1.1 **Concrete.** (row) R: Litvin 228 (152 mm, 346.7 kg/m²), 229 (203 mm, 471.6 kg/m²); Mahn 369, 371. α: PTB#30
1559 (Sabine, hard face), Nell 1730 (chamber boundary, from 31.5 Hz). Constants: Hopkins 374–376; Mahn η 370, 372.
Line 229 minus 228: 1.7 / 0.8 / 2.1 / 1.6 / 1.6 / 2.7 dB (125 Hz–4 kHz), against 2.67 dB from the mass law of the two
surface masses (script; 2.51 dB is the thickness ratio). **Built (2026-10-27):** one generic solid and its two sheets,
density 2281 kg/m³ (`e` 1901, Litvin 76-66), the face absorption one power law over PTB#30 and Nell (0.0204 at 1 kHz,
exponent 0.3965), the mass-law knobs fitted at 342.76271 m/s to the mass-shifted Litvin pair at 150 and 203 mm. At side
3.523 m the row is at most 0.27 dB outside the pair's envelope at both thicknesses, within 0.74 dB of 76-66 and
2.14 dB of 76-77 (250 Hz, the spread of the pair itself); α within 0.015 of PTB#30 and 0.002 of Nell. `coiF` (4.4 kHz)
and `att` are fit knobs: Hopkins' fc·h of 17.1 Hz·m (line 374) gives 114 Hz at 150 mm, and a plate with the physical
coincidence and one constant loss factor misses the pair by 7–11 dB, because the measured loss factor falls from 0.04
to 0.007 (Mahn 370, 372; stage-2b issue 7). Mahn 369, 371 (thirds at the octave centres, CON150 of unknown size) are
printed only: 3 dB below the pair at 250 Hz, 9 dB above at 4 kHz. Gaps: 63 Hz R; R outside 150–203 mm.

1.2 **Heavy masonry.** References (decided): the bare walls of the class plus the measured plaster increment, with
its spread, for each missing plaster face. (row) bare walls: BIA 273 (95 mm), 1795 (SCR brick 140 mm), 1847 (203 mm),
1729 (241 mm), 1846 (305 mm), surface masses 189.0–569.8 kg/m² (data text); Warnock 349 (90 mm solid concrete blocks,
195.2 kg/m²). (row) plastered: Hopkins 448 (dense block 215 mm plastered both faces, 430 kg/m², 241 mm finished:
1784 kg/m³ by script), 447 (100 mm plastered one side, 200 kg/m², 113 mm: 1770 kg/m³); BIA 265 (brick 92 + 13 mm one
face). Plaster increment of one face (matched pairs, script): 1796 minus 1795 = +2.5 / +1.1 / +1.5 / +2.0 / +1.5 /
+0.0 dB, 1904 minus 1905 = +2.7 / +1.7 / +2.5 / +3.0 / +1.6 / +1.0 dB (125 Hz–4 kHz). The bare brick series rises per
doubling of surface mass 7.41 / 8.60 / 9.60 / 9.13 / 7.43 / 5.81 dB (least squares over 273, 1795, 1847, 1729,
1846, surface masses from the data text), against 6.02 dB for the engine's mass term; walls scatter 2.5–4.0 dB about the trend, and a fit at
the centre misses the ends by up to 2.85 dB at 500 Hz. Hence the asserted range of about 140–241 mm; 95–140 and
241–305 mm are printed. Walls of nearly equal mass differ widely: 1729 minus 1847 = 5.3 / 7.4 / 8.4 / 9.2 / 6.2 /
5.5 dB against 1.07 dB from the mass law (script). Starting columns come from a fit to the series, not from the page
rows (whose `att` and `coiF` differ wall by wall). α: plaster PTB#344 1571, PTB#346 1573 (Sabine, hard face).
Density: Hopkins 267, 450. Line 1845 is a lined wall (1845 minus 1847 = −0.3 / −1.4 / 8.4 / 9.9 / 8.6 / 5.4 dB) and
belongs to 1.12. Gaps: a clay-brick wall plastered both faces; sand-lime (request 11).

1.3 **Hollow masonry.** Hollow = cored units of any aggregate; light = solid units of aircrete, expanded clay or
lightweight aggregate. (row) bare walls: Zeitler 352 (190 mm normal-weight, from 63 Hz), 351 (140 mm hollow
lightweight aggregate, without leakage; 350 is the same wall with the leakage of unsealed faces, printed); Warnock
354; BIA 1903 (194 mm hollow clay tile), 1905 (90 mm), with the plaster increment as 1.2. Line 354 minus 352: −0.6 /
3.2 / 0.0 / 0.2 / −0.5 / 1.3 dB (script). Starting columns: a fit to these walls. α: plaster as 1.2. Gap: hollow
walls plastered both faces (request 11). `masonry_hollow_250mm` dropped (no reference).

1.4 **Light masonry.** (row) plastered: Hopkins 24 (aircrete 115 mm + 13 mm plaster one side, 71 kg/m², 128 mm:
555 kg/m³ by script); Warnock 355 (190 mm lightweight block, plastered), 357 (90 mm wood-fiber aggregate,
plastered). Unsealed blocks leak (356 reads 24.5 dB at 125 Hz against 34.1 dB for 355; 358 6.9 against 32.5 dB) and
are not references. Density: Hopkins 88 (aircrete 400–800 kg/m³), 498, 1040; Davy 93. α: plaster as 1.2. Gap: a
plain aircrete wall (request 11).

1.5 **Plaster.** Only on hosts without plaster: `concrete`, `clt`. (comb.) matched pairs BIA 1795 / 1796 and 1905 /
1904, the bare wall written inline in the case. α: PTB#344 1571, PTB#346 1573.

1.6 **Glass.** (row) R: Pilkington 536, 537 (EN 12758 generic); Quirt 610, 612; AGC 611. α: PTB#207 1544 is a 3 mm
pane (neighbor build, printed): A tier 2. Starting columns: page 11.4 rows "glass 4 mm" (also the rig's `GLASS4`) and
"glass 6 mm", side 1.3492 m. The generic values and Quirt's small panes differ by 5.6 to 6.6 dB at 125/250 Hz (4.4).
The page row absorbs 0.305 at 125 Hz against PTB#207's 0.08: its absorption is refitted. Constants: Hopkins 561–563.

1.7 **Glazing units.** (row) R: AGC 592, Pilkington 1000, Quirt 579, 581; AGC 622. Quirt 614 is 3-10-3-10-3 (neighbor,
printed). α: PTB#208 1545 is 2–3 mm glass with a 10 mm gap (neighbor, printed): A tier 2. Starting columns: page 11.4
rows "double glazing 4-16-4" and "triple glazing 4-12-4-12-4" with `thk` 0.008 and 0.012; under rule 1.7.2 they
become `thk` 0.024 and 0.036 with `fRef` × 3 and every power-law value rescaled.

1.8 **Box windows.** (row) R: Quirt 578, 577, 580. A leaf has no bending stiffness, so the coincidence dip at
2–3 kHz is not expressible (page section 13).

1.9 **Gypsum board.** (row) R: Hopkins 1274, Northwood 956; Wareing 961, 962 are 10 mm boards (neighbors, printed).
η: Quirt 898, Hopkins 1285. Density: Hopkins 1284. Starting columns: page 11.4 row "gypsum board 12.5 mm" (side
1.6971 m); Hopkins' specimen side is √(3.53 × 2.63) = 3.047 m (script). Gaps: 31.5 Hz, 8 kHz.

1.10 **Gypsum leaf.** (comb.) section 3; α: PTB#341 1568 (empty cavity, Sabine, hard face: A 1), PTB#342 1569 (filled,
above 0.2: 1p), Bradley 923 (model of chamber data). Starting columns: the rig's limp leaf `BOARD` (8.6 kg/m², loss
ratio 0.2): `thk` 0.0125, `a` 4.691e-5, `e` 573.0, `g` 573.0·0.2/17.98, `h` 1.0. Gap: `rr` for room-size leaves.

1.11 **Drywall.** (row) `drywall_filled_125mm`: Halliwell 937, 1049 (G13 SS90 GFB90, about 115 mm). Printed, other
builds: Halliwell 940, 943 (16 mm boards, filled), 948 (16 mm boards, empty, about 122 mm), 1050 (G13 SS150 GFB150,
about 175 mm: the check of the double-wall model, not a reference of `_150mm`); Fuchs 2005 478 (12.5 / 100 / 12.5 mm,
studs and fill not stated). Depth alone moves R: 1050 minus 1049 = −0.6 / 7.6 / 9.9 / 11.0 / 6.2 / −1.9 / 0.6 dB
(63 Hz–4 kHz, from the data rows); the fill: 940 minus 948 up to 10.3 dB (from the data rows). 1049 rises 13.5 and
14.5 dB per octave from 125 to 500 Hz, then dips at coincidence: fitting `drywall_filled_125mm` early checks whether
lumped double walls are expressible. α: as 1.10. Gaps: no wall of the build at 100 or 150 mm, none empty with
12.5 mm boards. The sheets other than `_125mm` are tier 2 from a double-wall model.

1.12 **Dry lining.** (comb.) Hopkins 444 (dabs on 100 mm dense block, ΔR), 443 (dabs on expanded clay, ΔR); BIA 1797
against bare 1795 (on foam dabs); BIA 1845 against 1847 (on wood furring); Zeitler linings 1066–1086 (ΔR) on base
walls 352, 351. Page 11.4 rows "brick 140 mm, gypsum board on dabs", "brick 203 mm, gypsum board on furring". The
furring is not modeled.

1.13 **Plywood.** (row) R: Wareing 1334 (small specimen), 1333, 1335, 1336. α: Quirt 1344 (Sabine, hard face; the
specimen lay on the chamber floor, while the row is a free panel: a mounting difference, stated in the case).
Constants: Hopkins 1322–1325. Starting columns: page 11.4 row "plywood 18 mm" (side 1.6971 m). Gaps: density of the
specimens (710 kg/m³ assumed); 31.5/63 Hz.

1.14 **Wood panelling.** α (compilation): Cox 1320, Scheibler 1692, 1711.

1.15 **Floor boards on joists.** α: PTB#393 1580 (Sabine, hard face, max 0.15); compilation Cox 539, Scheibler 1709;
BRAS 123, 124 (room fits). A rug on it is a miss candidate (stage-2b issue 5). Gaps: measured α with build; any R
(request 2).

1.16 **Parquet.** α: PTB#394 1581 (glued, Sabine, hard face); Cox 1232; BRAS 162; parquet on battens PTB#395 1582.

1.17 **Laminate.** None found (query "laminate" returns only laminated glass and CLT). A rug on it is a miss
candidate (stage-2b issue 5). Gap: absorption of a floating laminate floor (request 8).

1.18 **Screed.** Constants: Hopkins 1787–1792. A bonded screed is drawn with the slab (1.7.5).

1.19 **Floating screed.** Resilient layers: Hopkins 294, 503, 605–608, 1750–1752 (dynamic stiffness). Mass-law
bodies do not resonate with a fill (4.6).

1.20 **Ceramic tile.** α (compilation): Cox 1090; BRAS 193 (initial estimate).

1.21 **Vinyl.** α: PTB#601 1606 (Sabine, hard face); Cox 1088; Bork 1087.

1.22 **Metal sheets.** (row) R: Kosala 1874 (steel), 40 (aluminum 2 mm); Northwood 1877 (0.8 mm steel). Constants:
Hopkins 1867–1869, 33–35. Gaps: specimen size of the references; installed loss factor.

1.23 **Solid door.** α: PTB#396 1583 (Sabine, hard face); Cox 1850, Scheibler 1710; Pinhede 459 (room fit). Gap:
laboratory R and surface mass of a door (request 3).

1.24 **Hollow-core door.** None found (queries "hollow core", "door"). Request 3.

1.25 **CLT.** (row) R: Hoeller 395, 396, 397; η: 398–400.

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

2.1 Constants: Hopkins MDF 1094–1096, chipboard 242–244 (both 760 kg/m³), softwood 1836, 1837. No R or α of a free
board; its role is a reflector. The rig's `MDF` leaf is the leaf form for panel absorbers.

2.2 (row) Nocke 1211 (cabinet without acoustic front, per object), 1213 (α on its cuboid, Sabine); with acoustic
front 1210, 1212. Nocke's cabinet is 0.80 × 0.46 × 1.09 m; its S_E of 3.11 m² is the cuboid without its floor face
(3.115 m² against 3.483 m² for all faces, script): the asset stands on the floor; against a wall, the back face's
share is lost. Wardrobe and cupboard rest on this one object. Gap: full wardrobes (request 13).

2.3 Odeon 104 (scattering only). Request 9.

2.4, 2.5 No per-object data. Casing constants: Hopkins PMMA 1345, 1346; Kosala PMMA R 1360, 1362, 1358,
polypropylene 1438–1441.

2.6 Per object: Nilsson 264 (table with two chairs), 1216 (one chair), 262 (10 tables + 19 chairs; values include
scattering toward an absorbing ceiling); BRAS 114, 115 (room fits).

### 2.3 Class 3: soft furnishing

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | `carpet_tile_11mm` | carpet tiles, low-pile rugs | sheet on the slab or finish solid; cut out under standing objects | 1 to 500 Hz; above: 1 once σ is measured | 2 | 125 Hz–4 kHz | high | wish |
| 3.2 | `carpet_tufted_12mm` | tufted carpet without underlay | sheet on a solid | 1 | 2 | 125 Hz–4 kHz, 0° | medium | wish |
| 3.3 | `underlay` | felt or foam underlay | traced solid of its real thickness: sheet \| `underlay` \| floor solid | 2 | 2 | — | medium | wish |
| 3.4 | `curtain_flat_0_2mm` | curtains hung flat; roller and vertical blinds (recipe) | fabric leaf sheet \| box of cavity air \| wall or window (sheet at the box's exit) | 1p | 2 | — | high | wish |
| 3.5 | `curtain_pleated_0_8mm` | curtains with fullness, stage drapes | as 3.4, effective columns | 1p | 2 | — | high | wish |
| 3.6 | `pu_foam` | free small cushions and pillows | porous body of the cushion's shape | 2 | 2 | — | medium | wish |
| 3.7 | `polyester_fiber`, `polyester_fiber_20mm` | fiber pillows (solid), blankets (sheet) | solid, or sheet on a solid | 2 | 2 | — | low | wish |
| 3.8 | `upholstery` | the body of soft furniture: sofas, armchairs, beds, mattresses, large cushions | porous body (2.4.2), assets in size variants; the fit is phase-3 work | 3 | — | per object | high | wish |

References and gaps:

3.1 (comb.) Salford 227 (Sabine, on concrete; hard face to 500 Hz, 0.09), BCTC 226 (Sabine, cushion backed; 0.40 at
500 Hz, not a hard face there: printed above 250 Hz); Cox 223–225 (compilation spread); PTB#155 1537 is 5 mm needle
felt (another product, printed). Above 500 Hz the porous conversion applies once σ is measured. Gap: flow
resistivity; 63 Hz.

3.2 (comb.) Broda 1930 (normal incidence, rigid backing); SAA 1929. Compared at 0° in the rig.

3.3 α alone: Broda 1186 (normal incidence, range only); (comb.) Broda 1931, PTB#156–158 1538–1540, Cox 1909, 1911.
The floor below must be a mass-law body (two traced wave media back to back: stage-2b issue 3). Gap: σ of underlays
(request 7).

3.4 (comb.) DelRey 428, 430 (Sabine, flat; fabric 0.2 mm, 300 g/m², 144 Pa·s/m), 432; σ: 1370. Blinds: compilation
Scheibler 1627 (vertical blinds 15 cm from the wall, half open), 1628 (open), as a `curtain_flat_*` recipe. A curtain
in front of a window needs the sheet at the box's exit (the glazing). Gap: a random-incidence route for a hanging
fabric.

3.5 (comb.) Zimmer+Rohde 412 (342 g/m², 0.79 mm, 1421 Pa·s/m, 100 % fullness, 100 mm from the wall, Sabine); 410, 406,
408; velour (compilation) Cox 976, 977, 1100, 1101; PTB#159 1541. The two curtain references differ by a factor of
9.9 in flow resistance (1421 / 144, lines 412, 1370) and 1.14 in mass (342 / 300): fullness and resistance, not mass,
separate them.

3.6 Dupont 1446–1449 (σ 10605 Pa·s/m², line 1447). Porous columns from Miki or Komatsu at the measured σ; generic
solids of wave media are exact at any drawn thickness. Porous bodies must not touch each other.

3.7 σ: Garai 1385–1426 (fiber blankets 10–70 mm; lines 1385–1422 span 12–110 kg/m³). Polyester model unresolved
(research handoff, A14).

3.8 Per object: SP 239, Nilsson 1216; the per-area values of seating rows (PTB#484 1604, PTB#483 1603, Kath and Kuhl
1939; compilation Cox 1943–1945, PTB#480 1600, Scheibler 1648, 1649) belong to 4.10, not to a sofa body. Gap:
per-object area of sofas, armchairs, beds (request 1).

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

So `alpha` is fitted with the object test. Vegetation (0.12 dB/m at 8 kHz, line 546) is far inside the thin limit.

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
| 4.1 | `upholstery` (assets `sofa_1seat`, `sofa_2seat`, `sofa_3seat`, `armchair`, proposed) | couch, sofa, armchair | porous body | 3 | — | — | high | wish |
| 4.2 | `upholstery` on `furniture_board` (asset `bed`, proposed) | bed | porous body (the mattress) on a traced solid of the frame | 3 | — | — | medium | wish |
| 4.3 | `box_bookshelf` | books in a bookshelf | hard box (2.3) | 3 | — | — | medium | wish |
| 4.4 | `clothes_rack` (proposed name) | rack of clothing | porous body of a light fiber medium around the hanging clothes (proposed) | 3 | — | — | low | wish |
| 4.5 | `plant_foliage`, `green_wall_100mm` (proposed names) | plants indoors, green walls | potted plant: leave out if small, else an absorbing volume of the foliage outline (proposed); green wall: a porous sheet | 3 (potted); 1p (green wall) | — | — | low | wish |
| 4.6 | `clutter_sparse`, `clutter_medium`, `clutter_dense` | items on desks and shelves, dishes on a counter | absorbing volume (2.4.3) drawn over the items; revisable with data | 3 | — | — | medium | wish |
| 4.7 | `person` (proposed name; assets `person_seated`, `person_standing`) | a listener, a performer | porous body of a low-poly human shape (proposed) | 1 | — | 125 Hz–2 kHz | medium | wish |
| 4.8 | `audience_seated` (proposed name) | dense audience | porous box over the seating area, of the height of seated persons (proposed) | 1p | — | 125 Hz–4 kHz | high (venues) | wish |
| 4.9 | dishes and glasses (no row) | kitchens, bars | leave out, or inside a clutter volume (proposed) | — | — | — | low | wish |
| 4.10 | `seating_empty` (proposed name) | unoccupied rows of seats, pews | porous box over the seating area, as 4.8 (proposed) | 1p | — | 250 Hz–2 kHz | medium (venues) | wish |

References and gaps:

4.1, 4.2 No per-object area of a sofa, armchair or bed (request 1).

4.4 None (query "clothing", "garment" finds nothing); request 13.

4.5 Green walls: Thomazelli 677–680, 1953 (Sabine); Davis 1951 (Sabine, single number only). Potted plants: request
13.

4.6 Groups of furniture only: Nilsson 262, 264 (include scattering). Request 10. Clutter mainly scatters, which is out
of scope; the volume gives only absorption.

4.7 (row) per person: Kath and Kuhl 1263 (seated, isolated), 1264 (standing); Nishidate 1260 (design value), 1265,
1266 (in situ); Chagok 1262 not usable. Gap: 63 Hz.

4.8 Per area of a block (1p until the per-area convention for blocks is decided): Kath and Kuhl 80, 81 (Sabine), 82, 83
(in situ); Bradley 1202, 1203 (occupied seats, infinite-block intercept, Sabine; up to 1.051 / 1.088, values above one
marked); Carvalho 250 (per object), 252 (Sabine per area 1.26–1.65, marked); Kath and Kuhl 1937, 1938. Edge effects:
Kath and Kuhl 77, 79 (strip widths).

4.10 Kath and Kuhl 1939 (Sabine), 1940 (in situ); PTB#484 1604; Carvalho 251 (pews, per object).

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
| 5.15 | `ceiling_tile_25mm` | suspended absorbing ceiling | porous tile sheet \| box of cavity air (the plenum) \| slab | 2 | 2 | — | high (offices, venues) | wish |
| 5.16 | `perforated_gypsum_12_5mm` | perforated plasterboard ceilings | perforated leaf \| box \| slab | 2 | 2 | — | medium | wish |
| 5.17 | slotted steel covers on a porous layer (no row) | slotted ceilings | not a plane layer | 3 | — | — | low | miss candidate |
| 5.18 | diffusers (`box_diffuser`, proposed name) | QRD and skyline diffusers | absorption as a hard box; the scattering is out of scope | 3 | — | — | medium | miss candidate |

References and gaps:

5.1 (comb.) Dupont 647 (random incidence, sound-field synthesis), 645 (JCA prediction), 648 (Sabine, above one,
marked); σ 646 (13112 Pa·s/m²). Starting columns: page 11.3 (also the rig's `WOOL`), Komatsu at 13112 Pa·s/m²,
`fRef` 0.87403: `a` 1.1362, `b` −0.043932, `c` 0.0023911, `d` 1.2533, `e` 2.2495, `f` −0.45515, `g` 0.088044, `h`
0.14944, `thk` 0.05. Transmission through a treated wall is over-insulated up to 8.1 dB at 31.5 Hz (stage-2b issue
1). Clouds and baffles hung in air are porous bodies (solid) or sheets with air on both sides; their Sabine data are
per object. Gaps: 31.5/63 Hz; 125 Hz only as a model.

5.2 (comb.) Kosten 1770 (random incidence from the tube, locally reacting), Kuhl 1145; Sabine 1767, 1147 (above one,
marked). The rig's `ROCKWOOL` is Miki at 32.8 kPa·s/m² (`fRef` 0.87403, `a` 1.31823, `b` −0.0125593, `c`
0.00481048, `d` 1.80573, `e` 0.823363, `f` 0.0954384, `g` 0.221513, `h` −0.0150261), a leaf-test fill, not fitted
to Kosten's product. Gap: σ.

5.3 (comb.) Dupont 1448, 1446, 1449; σ 1447. Fitted-σ check 10.3 against 10.6 kPa·s/m² (engine notes 9.3).

5.4 BASF 1108 (Sabine, mounting not stated, marked); σ range Cox 1102.

5.5 σ: Garai 1385–1426; Szelag 1215, 1935 (type not stated).

5.6 PTB#407 1595 (50 mm wood wool set in mortar, Sabine, 0.08–0.65); σ 500–2000 Pa·s/m² for 350–500 kg/m³ (line 1976,
secondary compilation). A tier 2 until σ of a product is measured, then 1 by the porous route.

5.7 No typed measurement with an air gap: compilation 513; practical 655; GIK 555 (per object in corners), 554
(A-mount on the floor, no air gap; Sabine above one).

5.8 Per object in corners (placement values, printed): GIK 553, 557; Auralex 86; Sabine 1511 (above one, marked);
the note at 555 reads 0.6 m² flat against 1.34 m² in the corner at 125 Hz.

5.9 (comb.) Shearer 1120, 1121 (Sabine); f₀ 1115, 1116. Starting columns: the rig's `MDF` leaf. Gaps: `rr`; panel
conversion.

5.10 (comb.) Sakagami 1314, 1315, 1316 (Sabine; 30/90/150 mm); f₀ 1317–1319.

5.11 (comb.) Fuchs 319, 318, 327, 328 (Sabine; above one: placement). Resonates on the fill's frame stiffness ("a
fill bonded to the plate acts through its own stiffness", research handoff), which a rigid-frame fluid lacks; the
steel's coincidence is not a leaf's.

5.12 Nell 549, 550 (Sabine, marked); f₀ 1302 (20.5 Hz); placement moves 125 Hz from 0.41 to 0.64. Resonates on the
mat's stiffness.

5.13 (comb.) Fuchs 1178, 1177 (Sabine), 1179, 1180. The stated open area 0.45 % does not match the stated grid
(0.64 %).

5.14 Cox 1130, 1933, 1140 (type not stated); aprons Fuchs 1131–1133 (Sabine, strips hung from a concrete ceiling).

5.15 Practical: Ecophon 1908 (200 mm plenum), Nilsson 235–237; compilation Cox 15, 16, Scheibler 1642. Request 4.

5.16 Practical: Rigips 1252, 1250; PTB#316 1562 (Sabine). Request 5.

5.17 Fuchs 1823, 1822, 1824 (Sabine, figure).

5.18 Compilation: Cox 3, 4 (1D QRD), 8, 9 (2D QRD; covered and uncovered).

### 2.6 Class 6: outdoor

**Ground model (decided route).** The ground classes are defined by Delany-Bazley at a class σ (lines 699–712:
Harmonoise/Nord2000 impedance classes of the one-parameter model). Delany-Bazley cannot be written into the columns
over most of the band: in the engine's mapping its compressibility needs a gain (`c` < 0) below ρ₀f/σ ≈ 0.014, at 6
to 25 of 25 thirds (closed form in the engine's mapping, thirds 31.5 Hz–8 kHz: loose gravel 6, snow 7, forest floor 11, grass C 15, grass D 19, soil
23, gravel road 25, porous asphalt 14), and `c` ≥ 0 is enforced when the table is read. The standard route is
therefore a **Miki fit with passive columns to the class impedance** (Delany-Bazley's surface impedance and
wavenumber); Delany-Bazley itself is printed. The reflection target stays physical, since a Delany-Bazley half-space
keeps a positive surface resistance. At short range the ground acts nearly rigid at low frequency (closed-form
estimate for a locally reacting half-space, Weyl–van der Pol: 63–250 Hz excess attenuation +5.3 to +6.0 dB for σ 12.5–200 kPa·s/m² at 20–200 m).
The class σ are local-reaction values while a traced porous solid reacts extendedly, and the engine reflects plane
waves; both limits are in the modeling guidelines.

| # | name | intended use | recipe | A | T | valid | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1 | `ground_snow` | snow cover | porous half-space (thick traced solid) | 2 | — | — | high (winter sites) | wish |
| 6.2 | `ground_forest_floor` | forest floor, moss, heather | porous half-space; a layer form is the alternative | 2 | — | — | medium | wish |
| 6.3 | `ground_grass` | lawns, pasture, sports fields | porous half-space | 2 | — | — | high | wish |
| 6.4 | `ground_soil` | compacted soil, earth | porous half-space | 2 | — | — | medium | wish |
| 6.5 | `ground_gravel_loose` | loose gravel, gravel beds, gabion fill | porous half-space, or a layer on soil | 2 | — | — | medium | wish |
| 6.6 | `ground_gravel_compacted` | gravel roads, car parks | porous half-space | 2 | — | — | medium | wish |
| 6.7 | `asphalt`, `asphalt_porous_40mm` (proposed thickness) | dense asphalt and concrete paving; porous asphalt | dense: rigid traced solid with a small face absorption; porous: thin porous sheet on the `asphalt` solid | 2 | — | — | high | wish |
| 6.8 | `vegetation` | foliage of trees, hedges, shrubs | absorbing volume (2.4.3) around the foliage; `alpha`, `alphaB` fitted to the excess attenuation per meter | — | 2 | 63 Hz–8 kHz, paths 20–200 m | medium | wish |

References and gaps:

6.1 Class A 12.5 kPa·s/m² (699; G 1, 698); Ishida 1831 (2–25 kPa·s/m², laboratory); Piercy 480 (15–30); Bashir 869;
FHWA 1925 (powder 10, granular 40). Snow of finite depth is a layer (proposed).

6.2 Class B 31.5 kPa·s/m² (701; G 1, 700); Bashir 714–719, 721, 846–849; Nordtest 547, 1833; Piercy 548. Half-space
fits miss by 24.5 / 26.3 / 28.1 dB (715, 717, 719), layer fits by 17.0 / 11.8 / 6.0 dB (714, 716, 718; fit errors in
the data notes).

6.3 Class C 80 (702; G 1, 703), class D 200 (705; G 1, 704); Piercy 670 (150–300); Bashir 722–727; Van Renterghem 671,
672; FHWA 666–668, 1925; Nordtest 1776. Attenborough 844 (summer, 80 kPa·s/m², layer 0.035 m) and 845 (winter,
200 kPa·s/m², layer 0.011 m) are layer fits: printed checks of a layer form (4.9).

6.4 Class E 500 (707; G 0.7, 706); Piercy 482 (rain-packed earth 4000–8000); Bashir 720; Attenborough 870 (soil with
winter wheat, 170); Nordtest 674, 675; FHWA 1844, 1925.

6.5 Van Renterghem 673 (gravel 10 kPa·s/m²); Bashir 834–843 (layers 0.04–0.22 m and semi-infinite), 871. Loose and
compacted gravel differ by 2000 / 10 = 200 in σ.

6.6 Class F 2000 (709; G 0.3, 708); Nordtest 674; Piercy 1214.

6.7 Dense: class G 20 000 (711; G 0, 710), class H 200 000 (712; G 0, 713); FHWA 74, 75; Piercy 76. Porous: Bashir
857 (67.5 kPa·s/m², porosity 0.18, tortuosity 2.36, layer 36 mm), 856 (2–15 kPa·s/m², non-acoustic). No absorption
coefficient of asphalt in the data (request 12).

6.8 ISO 9613-2 Annex A through Witte 546 (0.02 dB/m at 63 Hz to 0.12 dB/m at 8 kHz), 545 (10–20 m, absolute dB);
Witte 1949 (vegetation strip); Van Renterghem 456 (hedges). Tier 2: an engineering table of a standard, typed
catalog. The air-matched volume is right for foliage, unlike for furniture.

### 2.7 Auxiliary rows

| name | use | recipe | A | T | status |
| --- | --- | --- | --- | --- | --- |
| `cavity_air` | the air of a box behind a leaf, a curtain, a panel, a floor on joists, a suspended ceiling | traced closed box; `a` 1, `e` 1.000001 (rig `AIRBOX`): a material identical to air is air | definition | definition | wish |
| `water_20c` | ponds and pools | traced solid; c 1482.36 m/s, ρ 998.207 kg/m³, Z 1.4797e6 Pa·s/m (lines 1959, 1957, 1958) | 1 | 1 | wish |

The rig tests `water_20c` as a sheet (its stack rule refuses a lossless wave medium with a critical angle as a traced
solid); the half-space form is untested, and a thick-solid case is proposed for phase 3.

## 3. Test references that are combinations (measured only)

Each combination is a rig test of the materials involved, room side first. "Rig today" says whether the rig can build
it now.

| combination | materials tested | measured whole (line, type) | rig today | notes |
| --- | --- | --- | --- | --- |
| carpet on concrete | `carpet_tile_11mm`, `carpet_tufted_12mm`, `concrete` | Salford 227, BCTC 226 (Sabine); Broda 1930 (normal incidence); compilations 1910, 973, 1981 | yes | transmission touches stage-2b issue 1 |
| carpet with underlay on concrete | `carpet_*`, `underlay`, `concrete` | Broda 1931 (normal); compilations PTB#156–158 1538–1540, 974 | sheet \| solid \| solid: yes | — |
| rug on a hard finish | `carpet_tile_11mm` on `parquet` solid on `concrete` | none | yes | plausibility |
| porous absorber on a wall | `glass_wool_50mm`, `pu_foam_50mm`, `rock_wool_50mm` on `concrete` | Dupont 647, 1448; Kosten 1770 | yes (rig case exists for glass wool) | — |
| porous absorber with an air gap | wool \| `cavity_air` \| wall | none typed | yes | plausibility only |
| panel absorber over a cavity | `mdf_leaf_6mm`, `plywood_leaf_3mm`, `cavity_air`, fill | Shearer 1120, 1121; Sakagami 1314–1316 (Sabine) | yes | panel conversion pending |
| curtain in front of a wall | `curtain_flat_0_2mm`, `curtain_pleated_0_8mm`, `cavity_air` | DelRey 428, 430; Zimmer+Rohde 412, 410; ADO 406, 408 | yes | fabric route pending |
| curtain in front of a window | curtain \| `cavity_air` \| glazing sheet | none | yes; transmission printed (stage-2b issue 1) | — |
| drywall, precise recipe | `gypsum_leaf_12_5mm`, `cavity_air`, `glass_wool` | Halliwell 937, 1049 (the build), 940, 943, 946, 948, 1050 (other builds); Fuchs 478 | yes; transmission printed (stage-2b issue 3) | the same walls test the lumped rows (an alternative model, 1.7.9) |
| plasterboard on frame, room side | `gypsum_leaf_12_5mm`, `cavity_air` | PTB#341 1568, PTB#342 1569; Bradley 923 | yes | — |
| dry lining on masonry | `gypsum_leaf_12_5mm`, `cavity_air`, masonry | Zeitler 1066–1086 (ΔR); Hopkins 444, 443; BIA 1797 / 1795, 1845 / 1847 | yes (leaf \| box \| wall; the bare wall inline) | issue 1 for a fill |
| plaster on a bare wall | `plaster_15mm` on a bare brick written inline | BIA 1795 / 1796, 1905 / 1904 | sheet \| solid: yes | gives the plaster increment of 1.2 |
| plaster on both faces of a block | plaster sheets on both faces of a block solid | Hopkins 448 (printed) | yes (exact); the far plaster traced is the ISSUE 4 fixture | — |
| box window | `glass_leaf_4mm` \| `cavity_air` \| `glass_leaf_4mm` | Quirt 578, 577, 580 | yes | — |
| parquet on concrete | `parquet`, `concrete` | PTB#394 1581 | yes | — |
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
8. **Masonry class boundaries.** Whether three classes cover what a modeler sees; sand-lime without a reference.
9. **Ground.** One grass row or two by season; a layer form for snow, loose gravel and forest floor.
10. **Conversions first.** The routes for panel absorbers, microperforated panels, hanging fabrics, green walls and
    per-area blocks decide every "1p" entry.
11. **One material per closed body, finishes as sheets.** To be confirmed in phase 3.

## 5. Procedure for adding a material later

1. **Who decides.** The maintainer decides whether a material enters the list; anyone may propose it.
2. **Fields of a new entry.** Name (1.6), class, intended use, recipe, A and T tiers, valid range, references
   (research key, line at the pinned research commit, coefficient type, row or combination reference), gaps, impact,
   status `wish`; a lumped entry states its surface mass; a mass-law row its typical `side` (1.8). The rules of 1.7
   and the guidelines' checklist apply.
3. **References.** Search the research library first (read-only). If nothing usable exists, write a draft research
   request in section 6. The maintainer passes it to the research manager; the answer comes back as data rows with
   source, location and coefficient type.
4. **Phase-3 pickup.** By the work order of 1.4. One worker builds its row family and its lean case by the procedure
   of 1.10; engine-limited checks are marked ISSUE; the maintainer reviews and commits; the status moves to
   `confirmed`, or to `recorded miss`. Object rows also get their generated asset.
5. **CSV traps.** An empty cell reads 0, not the column default; a text cell in a numeric column is an error; a
   material file replaces the built-in table.

## 6. Draft research requests, ordered by impact

0. **ISSUE for the research manager: a stable id per data row.** An `id` column in `data/material_properties.csv`
   that survives regeneration and sorting, so that the material table's source comments cite a key, not a line.
   Needed before the research round on the gaps; until then lines are pinned to commit 2dad906 and remapped by
   script.
1. **Sofas, armchairs, beds** (classes 3 and 4; high). Per-object equivalent absorption area, ISO 354, 63 Hz–4 kHz,
   with the object's size, upholstery and placement. Manufacturer reports acceptable. Blocks 3.8, 4.1, 4.2.
2. **Wooden floor on joists** (class 1; high). Measured absorption with build (board thickness, joist depth, cavity
   fill), Sabine or random incidence, 63 Hz–4 kHz; laboratory R of the bare floor if available. Not blocking.
3. **Doors** (class 1; medium). Laboratory R and surface mass of a solid timber door and of a hollow-core door with
   build, size and seals, 63 Hz–4 kHz; absorption of a hollow-core door. Blocks the transmission of both door rows.
4. **Suspended absorbing ceilings with build** (class 5; high in venues). Sabine or random-incidence α of
   mineral-fiber or glass-wool ceiling tiles with thickness, density or σ, and plenum depth, 63 Hz–4 kHz.
5. **Perforated and slotted panels with full build** (class 5; medium). Sabine or random-incidence α with geometry,
   open area, plate thickness, cavity depth and fill, sample size, 125 Hz–4 kHz.
6. **Carpet on a timber floor** (class 3; medium). Absorption of a carpet of stated build on boards on joists, and on
   concrete; flow resistivity of carpet piles; 63 Hz–4 kHz. Not blocking (the configuration is a miss candidate).
7. **Flow resistivity of carpets and underlays** (class 3; medium). With thickness and density. Turns the carpet rows
   to A 1 above 500 Hz.
8. **Floating laminate floors** (class 1; medium). Absorption of laminate on foam underlay over concrete, with build,
   63 Hz–4 kHz.
9. **Filled bookshelves** (classes 2 and 4; medium). Absorption per object of a shelf filled with books, ISO 354,
   125 Hz–4 kHz, with the shelf's size.
10. **Cluttered desks and shelves** (class 4; medium). Absorption per object or per area of a desk or shelf with
    typical items, ISO 354, 125 Hz–4 kHz.
11. **Masonry walls missing in the data** (class 1; medium). Laboratory R of plain aircrete and sand-lime walls with
    density and thickness; clay brick and hollow units plastered on both faces; 63 Hz–4 kHz.
12. **Absorption of asphalt** (class 6; medium). Absorption coefficient of dense and porous asphalt (in situ or tube,
    with build), 125 Hz–4 kHz.
13. **Potted plants, clothing racks, full wardrobes** (classes 2 and 4; low). Per-object absorption area, ISO 354,
    with placement.

(The numbering in section 2 refers to these requests by number.)
