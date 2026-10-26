# Acoustic materials: the wish-list (draft 3)

Draft 3, 2026-10-26 (draft 0 revised after two review rounds and an independent review). This list is the input to
the work on the shipped acoustic material table (`data/materials_acoustic.csv`). The rows, their columns and their
tests are built later, material by material, from the entries below. Nothing here is a row yet, except where the
status says so. Everything still marked **(proposed)** is the drafter's proposal and waits for review: impact
ratings, typical panel sizes, sheet thicknesses chosen by the drafter, row names of objects, recipes for tier-3
materials and objects.

The companion document `md/acoustic_modeling_guidelines.md` says how a venue is modeled for the acoustic tracer. The
modeling rules quoted here in short come from there; its checklist of forbidden and limited configurations applies to
every recipe below.

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
   ceiling boards, floor boards, floor screed, laminate flooring, metal sheets, drywall with an air or mineral-wool
   fill, linings, doors.
2. **Hard furnishing:** tables, chairs, wardrobes, cupboards, bookshelves, metal-clad furniture and appliances,
   casings of electronics (TV screens, speaker cabinets, lamps), sports equipment. A stage is modeled in detail from
   class-1 rows and has no entry here.
3. **Soft furnishing:** cushions and pillows as meshed 3-D shapes, blankets, carpets, curtains, mattresses,
   upholstery.
4. **Complex assemblies:** racks of clothing, couches, books in a bookshelf, plants, clutter, a single person and a
   dense audience, dishes and glasses.
5. **Acoustic treatment:** porous absorbers, bass traps, panel absorbers, microperforated panels, suspended
   absorbing ceilings, diffusers.
6. **Outdoor:** ground (grass, soil, gravel, snow, forest floor), asphalt, vegetation.

Auxiliary rows that belong to no class (the cavity air of a box, water) are listed in section 2.7.

### 1.3 Evidence tiers: one for transmission, one for absorption

Every entry carries two tiers: **T** for transmission and **A** for room-side absorption.

| tier | meaning | what checks the row |
| --- | --- | --- |
| 1 | A measured reference of that quantity exists, with a known build and coefficient type. For absorption, only a random-incidence or normal-incidence reference, a hard face (α up to about 0.2, used as measured: material page, section 10.5), a per-object area used like for like (2.4), or a Sabine value of a class whose conversion is decided. | the measured values after their conversion, within about ±3 dB in transmission and ±0.05 in absorption (plus the conversion budget below), and the independent physical model |
| 1p | **Tier 1 pending conversion:** the absorption reference is a Sabine (chamber) value of a class whose conversion to random incidence is not decided yet (panel absorbers, microperforated panels, hanging fabrics, green walls). Printed, not asserted, until the conversion is decided; that decision comes first in phase 3. | the independent model; the Sabine value printed |
| 2 | First-principles physics gives the columns (plate theory, a transfer matrix, a porous or ground model at a known flow resistivity, a leaf on a box of air); references are absent or are compilation values of unstated type, which are shown but not asserted. | the independent physical model written in the test; compilation values printed |
| 3 | An informed guess is the best available (a recipe built from neighbors, a per-object value of a similar object). | plausibility only; the guess and its basis are stated in the case |
| — | the quantity does not apply (an object's transmission, a volume's absorption) | — |

**Conversion error budget.** Converted absorption carries the conversion's own error beside the ±0.05: Jeong's
conversion with Thomasson's field impedance reached an rms of 0.031 (maximum 0.057) on three materials with an
independent random-incidence reference; the edge-effect correction 0.041 (engine notes, "Rules for absorption
references"). Today only porous layers of known construction (Jeong) and mineral wool without construction data
(edge correction) have a decided route.

Compilation values (Cox and D'Antonio Appendix A, the pyroomacoustics library) whose coefficient type is not stated
do not lift a material to tier 1. PTB rows are typed Sabine; they count where they are hard faces.

### 1.4 Impact and work order

How much the material moves the simulated sound field of a typical room or site **(proposed ratings throughout)**:

- **high:** covers a large share of the surface in most rooms (walls, floor, ceiling, windows) or of an outdoor
  site (the ground), or dominates the low frequencies where a room's modes are (floors on joists, bass traps, panel
  absorbers), or is the main absorber of a furnished room (sofas, carpets, curtains, audience);
- **medium:** common but smaller in area, or large in area but close to a material already in the table;
- **low:** small objects, rare items, or items below the wavelengths that matter.

**Work order (decided):** tier, then impact. Every relevant material goes in, each with its diagnostics, before the
engine is reopened; engine findings are recorded as stage-2b issues, not fixed, in the meantime.

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
  point becomes an underscore. Examples: `concrete` (solid), `concrete_150mm` (sheet); `drywall_filled` (solid),
  `drywall_filled_125mm` (sheet); `gypsum_leaf_12_5mm` (sheet).
- **Generic solid and round sheets.** A generic solid is named for the material (`concrete`, `masonry_heavy`); sheet
  rows exist at round thicknesses (`concrete_150mm`, `concrete_200mm`). A measured reference is compared with the
  sheet of the nearest round thickness (Litvin's 152 mm wall with `concrete_150mm`), the thickness difference stated
  in the case. Product thicknesses keep their size (`gypsum_board_12_5mm`, `plywood_19mm`).
- **Literature constructions are test references, not rows.** Their names do not appear in the table.
- **One board, two rows.** A board that is used both as a mass-law panel and as a wave-medium leaf on a box of air
  has two rows: the plate row (`gypsum_board_12_5mm`) and a leaf row named `<material>_leaf_<NN>mm`
  (`gypsum_leaf_12_5mm`).
- **Names say what a modeler sees:** `curtain_flat_*` (hung flat) and `curtain_pleated_*` (with fullness);
  `masonry_heavy` rather than a block product; `drywall_filled_125mm` from a plan's wall thickness and its fill note.
- **Every row is named,** object rows included. The existing preliminary row `concrete_cast_152mm` is a known
  exception until it is redone as `concrete_150mm`.
- **No combination rows.** Walls and floors are modeled first; carpets, curtains and absorbers are placed on them
  as separate rows. A combination is a rig test, and only where a measurement of the combination exists (section 3).
  Drywall and masonry rows are no combinations: each is the finished wall a modeler sees.
- **Reader rules** (`src/obj_file_read.cpp`, `base_material_name` and `resolve_csv`): the Blender material name is
  looked up exactly, then by its part before the first dot; there is no other fallback. Names are case-sensitive.
  Row names are snake_case and contain no dot, so that Blender's duplicates (`concrete.001`) and variants
  (`concrete.gray`) map to the row `concrete`. A name that matches no row becomes "no material" (index 0, air)
  without a warning unless the reader is called with `csv_strict`.

### 1.7 Rules for designing rows

1. **Sheet rows are the generic row's columns plus `thk`.** Drawing a wall as one plane with the sheet or as a box
   with the solid gives the same wall (the twin assertion of the rig).
2. **Lumped rows carry the visible thickness in `thk`, with `fRef` scaled** so that the surface mass is unchanged
   (decided). A mass-law sheet's `thk` does not enter the phase and displaces nothing of the host medium
   (`src/ray_mesh_interact.cpp:1320–1324`: only non-mass-law layers are taken back out of the host), so the rule is
   neutral for rigid lumped rows and makes the sheet and the solid of the visible thickness consistent. Wave-medium
   sheets (leaves, porous layers, fabrics) keep their true thickness, which the engine takes back out of the host
   medium. Every lumped entry states its surface mass.
3. **Glued rows carry `att` 0 and the host's `side`** (page rules 2 and 3: one `att` per bonded run or stack, one
   `side` per bonded run). This applies to finishes, linings and toppings (plaster, screed, tile, vinyl, parquet).
4. **A layer that carries another layer is a traced solid** (screed under tile, vinyl or parquet; the slab under a
   screed): a finish sheet on a solid is sheet | solid; a finish sheet on a screed sheet on a slab would be three
   co-located faces or two mass-law sheets, both forbidden. Two bonded traced mass-law bodies deviate from the bonded
   plate by at most 2.0 dB at 16 kHz and 0.05 dB at 1 kHz (stage-2b issue 3).
5. **Behind a fill the wall keeps its own row.** With the wall's own `tf`, the reflection of wool on concrete is
   within 6.3e-3 of the transfer matrix (stage-2b issue 1); no separate `tf` 0 wall row is needed.
6. **One material per closed body; finishes as sheets** (to be confirmed in phase 3). A glass front on a metal box is
   a glass sheet glued on that face.
7. **Closed casings are hard boxes (traced solids),** never closed boxes of mass-law sheets: a hollow casing holds a
   free air space, followed bounce by bounce (page rule 9: up to 0.29 in reflection at 255 interactions). Sheets are
   for open single panels (table tops, shelves).

### 1.8 Panel size `side`

`side` is a row value: the √area of a typical panel of that use. There is no engine change: the engine sees one
triangle, not the panel. Where a drawn panel differs from the typical size, the drift is accepted (modeling
guidelines, section 4). `side` acts only on mass-law rows; leaves and porous rows keep the infinite plate. A
reference is evaluated at its own specimen side, the row carries the typical size. The engine default is 3.29 m.
Typical sizes **(proposed)**, √(width × height) computed by script:

| rows | typical panel | `side` |
| --- | --- | ---: |
| `concrete*`, `masonry_*`, `drywall*`, `gypsum_board_12_5mm`, `clt*`, `screed*` (as the slab) | wall 4 × 2.6 m | 3.22 m |
| `glass_*mm`, `glazing_*` | pane 1.2 × 1.4 m | 1.30 m |
| `plywood_19mm` | sheet 2.6 × 1.2 m | 1.77 m |
| `door_*` | door 0.9 × 2.0 m | 1.34 m |
| `steel_1mm`, `aluminum_2mm` | appliance side 0.6 × 1.8 m | 1.04 m |
| `furniture_board*` | table top 1.6 × 0.9 m | 1.20 m |
| glued finishes (`plaster_15mm`, `parquet_15mm`, `ceramic_tile_10mm`, `vinyl_3mm`) | the host's side | — |

### 1.9 Notation

- **sheet | solid:** a sheet glued on the face of a traced solid, room side first (page section 7).
- **box of cavity air:** a closed box of `cavity_air` glued exactly between a leaf and a wall; its bounce series is
  summed in one event (page 10.4).
- **hard box:** a closed traced solid of a rigid row whose faces absorb a per-object area (2.4.1).
- **porous body:** a closed traced solid of a porous row (2.4.2).
- **absorbing volume:** a closed traced solid of a medium matched to air with an `alpha` column (dB/m along the
  path); it reflects nothing and is not taken for air, since `Material::same_as` compares `alpha`
  (`src/ray_mesh_interact.cpp:635`).

### 1.10 How phase 3 tests a row family

- **One test case per row family** (decided): a generic row and its sheets share one lean case, accepted by the
  table case.
- **Procedure (decided):** measure first; judge against the known range of measurements and models; if the result
  needs an engine fix, mark it ISSUE; if not, assert tightly (about 1 dB against the measured library result) as a
  regression guard. A comment in the case states what "reasonable" meant: the range of measurements and models it
  was judged against. The rig documents itself; this list is updated from the result.
- **ISSUE convention (decided):** a rig check that fails because of an engine limit carries a comment
  `ISSUE <n>:` naming the stage-2b issue; its assertion becomes a non-failing check (Catch2 `CHECK_NOFAIL` or
  equivalent) that still prints and is counted; the measured and the reference values go into the diagnostics; the
  issue has an entry in `md/acoustic_engine_stage2b_issues.md`. Known engine issues never make the suite fail.
- **Class tolerance (decided):** a class row (the masonry and drywall classes) is asserted against the envelope of
  its class's measured walls ±1 dB, in the rig's interval form (`lo`, `hi` per band).
- **Thickness range (decided):** a generic row is asserted within the measured thickness range of its references
  and printed outside it. The range is stated in the column "valid" of each table.
- **Rig prerequisites (before the first material that needs them):** a thickness argument for rows with `thk` 0
  (generic solids, porous bodies, ground rows; `check_material` requires `thk` > 0 today); a specimen side per
  measured reference; a 0° comparison (normal-incidence references); the object test as its own rig step, designed
  and timed against the budget of about 0.3 s per case; a self-check per recipe form (sheet | sheet | solid; plaster
  sheets on both faces of a block solid; a porous body on the floor) before its first material.

## 2. The materials per class

Columns: **recipe** (sheet, solid, stack, box of air, hard box, porous body, absorbing volume; what the modeler
sets); **T** and **A** the two tiers; **valid** the thickness and band range the references cover (asserted inside,
printed outside); **references** give the research key, the line in the research library's
`data/material_properties.csv` (header = line 1) and the coefficient type, marked **(row)** where they test the row
and **(comb.)** where they are combination references of section 3; one measurement is never used for both. R =
laboratory sound reduction index; σ = flow resistivity. Line numbers and values come from the material-list
analysis of 2026-10-23, the independent review of 2026-10-26 or queries of the research data on 2026-10-26.

### 2.1 Class 1: structural building materials

| # | name | intended use | recipe | T | A | valid | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | `concrete`, `concrete_150mm`, `concrete_200mm` | cast walls, slabs, ceilings | solid (thickness from geometry) or sheet | 1 | 1 | 150–203 mm; R 125 Hz–4 kHz | (row) R: Litvin 228 (152 mm), 229 (203 mm); Mahn 369, 371. α: PTB#30 1559 (Sabine, hard face), Nell 1730 (chamber boundary, from 31.5 Hz). Constants: Hopkins 374–376; Mahn η 370, 372 | 63 Hz R; R outside 150–203 mm | high | in work |
| 1.2 | `masonry_heavy`, `masonry_heavy_100mm`, `_150mm`, `_200mm`, `_250mm` **(proposed thicknesses)** | finished heavy wall, plastered both faces: solid clay brick, dense concrete block, sand-lime, about 1800–2400 kg/m³ | solid or sheet; `plaster_15mm` only for exceptions | 1 | 1 | about 100–305 mm; 63 Hz–4 kHz | (row) plastered: Hopkins 448 (dense block 215 mm, plaster both faces), 447 (100 mm, plaster one side); BIA 265 (brick 92 + 13 mm plaster one face), 1796 (SCR brick 140 + plaster one face). Bare walls of the class, printed: BIA 273, 1847, 1846, 1729, 1795; Warnock 349. α: plaster PTB#344 1571, PTB#346 1573 (Sabine, hard face). Density: Hopkins 267, 450 | no wall plastered on both faces in clay brick (request 11); sand-lime not in the data | high | wish |
| 1.3 | `masonry_hollow`, `masonry_hollow_150mm`, `_200mm`, `_250mm` **(proposed thicknesses)** | finished hollow-unit wall, plastered both faces: hollow clay brick, hollow concrete block of any aggregate | solid or sheet | 1 | 1 | about 90–200 mm | (row) plastered: BIA 1904 (hollow clay tile 90 mm + 16 mm plaster one face). Bare, printed: Zeitler 352 (190 mm normal-weight), 351 (140 mm lightweight aggregate, without leakage; 350 the same wall with the leakage of unsealed faces); Warnock 354; BIA 1903, 1905. α: plaster PTB#344 1571, PTB#346 1573 | plastered hollow walls on both faces | high | wish |
| 1.4 | `masonry_light`, `masonry_light_100mm`, `_150mm`, `_200mm`, `_250mm` **(proposed thicknesses)** | finished light wall, plastered: aircrete, expanded-clay and lightweight-aggregate solid units | solid or sheet | 1 | 1 | about 90–190 mm | (row) Hopkins 24 (aircrete 115 mm + plaster one side, 71 kg/m²); Warnock 355 (190 mm lightweight block, plastered), 357 (90 mm wood-fiber aggregate, plastered). Density: Hopkins 88 (aircrete 400–800 kg/m³), 498, 1040; Davy 93. α: plaster as 1.2 | plain aircrete wall (request 11) | medium | wish |
| 1.5 | `plaster_15mm` | plaster for exceptions (an unplastered row that gets plaster later) | sheet on the masonry solid; `att` 0, host's `side` | 2 | 1 | — | (comb.) matched pairs BIA 1795 / 1796, 1905 / 1904. α: PTB#344 1571, PTB#346 1573 | — | low | wish |
| 1.6 | `glass_4mm`, `glass_6mm` | single-pane windows | sheet only | 1 | 1 | R 125 Hz–4 kHz | (row) R: Pilkington 536, 537 (EN 12758 generic); Quirt 610, 612; AGC 611. α: PTB#207 1544 (Sabine, hard face). Constants: Hopkins 561–563 | reference row choice (4.4); the page row absorbs 0.305 at 125 Hz against PTB#207's 0.08; 31.5/63 Hz | high | wish |
| 1.7 | `glazing_double_24mm`, `glazing_triple_36mm` | sealed units 4-16-4 and 4-12-4-12-4 | sheet, one lumped body: 20 and 30 kg/m² | 1 | 1 | R 125 Hz–4 kHz | (row) R: AGC 592, Pilkington 1000, Quirt 579, 581; AGC 622. Quirt 614 is 3-10-3-10-3 (another unit, printed). α: PTB#208 1545 (Sabine, hard face) | 31.5/63 Hz R | high | wish |
| 1.8 | `glass_leaf_4mm` | two sashes with a wide air space (box windows) | leaf sheet \| box of cavity air \| leaf | 1 | 2 | — | (row) R: Quirt 578, 577, 580 | coincidence not expressible by a leaf | low | wish |
| 1.9 | `gypsum_board_12_5mm` | a single free gypsum panel; the reference row of the leaf | sheet | 1 | 2 | R 63 Hz–4 kHz | (row) R: Hopkins 1274, Northwood 956; Wareing 961, 962 are 10 mm boards (neighbors, printed). η: Quirt 898, Hopkins 1285. Density: Hopkins 1284 | 31.5 Hz, 8 kHz | low | wish |
| 1.10 | `gypsum_leaf_12_5mm` | gypsum ceilings over a plenum, linings, the precise drywall recipe | leaf sheet on a box of cavity air | 2 | 1p | — | (comb.) section 3; α: PTB#341 1568 (Sabine, hard face), PTB#342 1569 (Sabine, above 0.2), Bradley 923 (model of chamber data) | leaf loss ratio `rr` for room-size leaves; panel conversion | high | wish |
| 1.11 | `drywall`, `drywall_filled`; `drywall_100mm`, `drywall_125mm`, `drywall_150mm`; `drywall_filled_100mm`, `drywall_filled_125mm`, `drywall_filled_150mm` | stud partitions as a plan shows them: total thickness, maybe "filled" | lumped mass-law row of the standard build (one 12.5 mm gypsum board each side, metal studs, the rest cavity, empty unless filled; boards 17.0 kg/m² from Hopkins' 680 kg/m³); `thk` the visible thickness | 1 for `drywall_filled_125mm` and the solid within 115–122 mm; 2 (double-wall model) for the rest | 1p | filled 115–122 mm | (row) `drywall_filled_125mm`: Halliwell 937, 1049 (G13 SS90 GFB90, 115 mm). Printed, other builds: Halliwell 940, 943 (16 mm boards, filled), 948 (16 mm boards, empty, 122 mm), 1050 (G13 SS150 GFB150, 175 mm: the check of the double-wall model, not a reference of `_150mm`); Fuchs 2005 478 (12.5 / 100 / 12.5 mm, studs and fill not stated). α: as 1.10 | no measured wall of the build at 100 or 150 mm, none empty with 12.5 mm boards | high | wish |
| 1.12 | dry lining (no row: a recipe) | gypsum board on dabs or on furring over masonry | `gypsum_leaf_12_5mm` \| box of cavity air (dab or furring depth) \| masonry solid | 2 | 1p | — | (comb.) Hopkins 444 (dabs on 100 mm dense block, ΔR), 443 (dabs on expanded clay, ΔR); BIA 1797 against bare 1795 (on foam dabs); BIA 1845 against 1847 (on wood furring; moved here from the heavy-masonry references); Zeitler linings 1066–1086 (ΔR) on base walls 352, 351. Page 11.4 rows "brick 140 mm, gypsum board on dabs" and "brick 203 mm, gypsum board on furring" | the furring is not modeled | medium | wish |
| 1.13 | `plywood_19mm` | free wooden panels | sheet | 1 | 1 | R 125 Hz–4 kHz | (row) R: Wareing 1334 (small specimen), 1333, 1335, 1336. α: Quirt 1344 (Sabine, hard face). Constants: Hopkins 1322–1325 | density of the specimens (710 kg/m³ assumed); 31.5/63 Hz | low | wish |
| 1.14 | `wood_leaf_12mm` | wood panelling on battens | leaf sheet on a box of cavity air (batten depth) on the wall | 2 | 2 | — | α (compilation): Cox 1320 (plywood panelling 10 mm), Scheibler 1692, 1711 | build of the compilation rows | medium | wish |
| 1.15 | `board_leaf_22mm` | floor boards on joists | leaf sheet on a box of cavity air (joist depth) on the slab solid | 2 | 1 | — | α: PTB#393 1580 (Sabine, hard face); compilation Cox 539, Scheibler 1709; BRAS 123, 124 (room fits) | measured α with build; any R (request 2) | high | wish |
| 1.16 | `parquet_15mm` | parquet glued on a slab | sheet on the slab or screed solid; `att` 0, host's `side` | 2 | 1 | — | α: PTB#394 1581 (Sabine, hard face); Cox 1232; BRAS 162; parquet on battens PTB#395 1582 | — | medium | wish |
| 1.17 | `laminate_leaf_8mm` | floating laminate on a foam underlay | leaf sheet \| `underlay` solid \| slab solid | 2 | 2 | — | none found (query "laminate" returns only laminated glass and CLT) | absorption of a floating laminate floor (request 8) | medium | wish |
| 1.18 | `screed`, `screed_50mm` | sand-cement screed, bonded | solid (always a solid where a finish lies on it); the sheet only for a bare screed | 2 | 3 | — | constants: Hopkins 1787–1792 | — | medium | wish |
| 1.19 | floating screed (recipe; or a lumped row: 4.6) | screed on a resilient layer, the standard European floor | `screed` solid \| resilient layer \| slab solid | 2 | 3 | — | resilient layers: Hopkins 294, 503, 605–608, 1750–1752 (dynamic stiffness) | mass-law bodies do not resonate with a fill; R of a floating floor | medium | wish |
| 1.20 | `ceramic_tile_10mm` | tiled floors and walls | sheet on the screed or wall solid; `att` 0, host's `side` | 2 | 3 | — | α (compilation): Cox 1090; BRAS 193 (initial estimate) | — | low | wish |
| 1.21 | `vinyl_3mm` | linoleum or vinyl stuck to the screed | sheet on the screed solid; `att` 0, host's `side` | 2 | 1 | — | α: PTB#601 1606 (Sabine, hard face); Cox 1088; Bork 1087 | — | low | wish |
| 1.22 | `steel_1mm`, `aluminum_2mm` | metal sheets, metal doors, cladding | sheet | 1 | 2 | R 63 Hz–4 kHz | (row) R: Kosala 1874 (steel), 40 (aluminum 2 mm); Northwood 1877 (0.8 mm steel). Constants: Hopkins 1867–1869, 33–35 | specimen size of the references; installed loss factor | low | wish |
| 1.23 | `door_solid_40mm` **(proposed thickness)** | solid timber doors | sheet, lumped | 3 | 1 | — | α: PTB#396 1583 (Sabine, hard face); Cox 1850, Scheibler 1710; Pinhede 459 (room fit) | laboratory R of a door (request 3) | medium | wish |
| 1.24 | `door_hollow_core_40mm` **(proposed thickness)** | hollow-core interior doors (the common residential door) | sheet, lumped | 3 | 3 | — | none found (query "hollow core", "door") | R and α of a hollow-core door (request 3) | medium | wish |
| 1.25 | `clt`, `clt_100mm` | cross-laminated timber walls and floors | solid or sheet | 1 | 2 | 78–245 mm; 63 Hz–4 kHz | (row) R: Hoeller 395, 396, 397; η: 398–400 | — | low | wish |

Notes:

1. **Concrete.** Starting columns: the preliminary row `concrete_cast_152mm` in `data/materials_acoustic.csv` (the
   page 11.4 row, fitted at 343 m/s to Litvin at side 3.5229 m): `fRef` 0.00019663, `a` 3.85e-06, `e` 1901, `thk`
   0.152, `att` 14.459, `attB` −0.086239, `m` 1, `resF` 0.068191, `resQ` 1.4354, `resS` −1, `coiF` 6.0529, `coiQ`
   1.19, `tf` 0.00025486, `tfB` 0.52, `side` 3.5229. It is redone as `concrete_150mm` against the research data's
   Litvin values (42.1 / 46.5 / 52.6 / 56.5 / 61.7 / 64.6 dB, line 228). Cast concrete is measured only at
   150–203 mm; from 152 to 203 mm Litvin rises 1.7 / 0.8 / 2.1 / 1.6 / 1.6 / 2.7 dB against 2.51 dB for the mass
   law (review). The fitted `coiF` corresponds to about 6.9 kHz, while Hopkins' fc·h of 17.1 Hz·m (line 374) gives
   114 Hz at 150 mm: the knobs do not scale with thickness (4.1). As the host of glued finishes it keeps its `att`
   once (rule 1.7.3).
2. **Heavy masonry.** The row is the finished wall plastered on both faces (decided), fitted to the plastered walls;
   the bare walls of the class are printed. The class has a measured series from 95 to 305 mm (surface masses 189 to
   569.8 kg/m², densities 1868–2003 kg/m³; review), the first real test of the generic solid across thickness. Walls
   of nearly equal mass in the class differ widely: line 1729 (241 mm) minus 1847 (203 mm) gives 5.3 / 7.4 / 8.4 /
   9.2 / 6.2 / 5.5 dB, against 1.07 dB from the mass law (review); the class tolerance is the envelope ±1 dB
   (1.10). Line 1845 is a lined wall (1845 minus bare 1847: −0.3 / −1.4 / 8.4 / 9.9 / 8.6 / 5.4 dB) and moved to
   1.12. Plaster on both faces of a block is three bonded mass-law bodies, within ±3 dB in the engine (stage-2b
   issue 4); the lumped finished-wall row avoids it. Starting columns: page 11.4 rows "face brick 95 mm", "solid
   brick 203 mm" and "brick 105 mm, plaster one face".
3. **Hollow masonry.** Replaces `concrete_block_190mm`. Where hollow ends and light begins: hollow = cored units of
   any aggregate (Zeitler 350/351 are hollow lightweight-aggregate blocks); light = solid units of aircrete,
   expanded clay or lightweight aggregate. In this class the spread is small: Warnock 354 minus Zeitler 352 ranges
   from −0.6 to +3.2 dB (review). Starting columns: page 11.4 row "concrete block 190 mm" (fitted to Warnock).
4. **Light masonry.** Sealed or plastered walls only: unsealed porous blocks leak through their pores (Warnock 356,
   the unplastered twin of 355, reads 24.5 dB at 125 Hz against 34.1 dB; 358 against 357: 6.9 against 32.5 dB).
   Its references are row references only, not plaster combinations.
5. **Plaster.** Only for exceptions; the masonry rows include their plaster.
6. **Glass.** Starting columns: page 11.4 rows "glass 4 mm" (also the rig's inline `GLASS4`) and "glass 6 mm", side
   1.3492 m. The EN 12758 generic values and Quirt's small panes differ by 5.6 to 6.6 dB at 125/250 Hz. The page row's
   absorption (0.305 at 125 Hz, fitted to a compilation) is about 4 times PTB#207 (0.08, line 1544): the absorption
   fit is redone against the hard-face reference.
7. **Glazing units.** Starting columns: page 11.4 rows "double glazing 4-16-4" and "triple glazing 4-12-4-12-4" with
   `thk` 0.008 and 0.012; under rule 1.7.2 they become `thk` 0.024 and 0.036 with `fRef` scaled by 3, so the mass is
   unchanged and the name is true. The rig's printed bare mass law then needs the stated surface mass (20 and
   30 kg/m²).
8. **Box windows.** A leaf has no bending stiffness, so the coincidence dip at 2–3 kHz is not expressible (page
   section 13).
9. **Gypsum board.** Mainly the reference row of the leaf (ceilings are leaves on a plenum). Starting columns: page
   11.4 row "gypsum board 12.5 mm" (side 1.6971 m); Hopkins' specimen side is 3.047 m.
10. **Gypsum leaf.** Starting columns: the rig's inline limp leaf `BOARD` (8.6 kg/m², loss ratio 0.2): `thk`
    0.0125, `a` 4.691e-5, `e` 573.0, `g` 573.0·0.2/17.98, `h` 1.0.
11. **Drywall.** From what a plan shows. Depth alone moves R by up to 11.0 dB (line 1050 minus 1049: −0.6 / 7.6 /
    9.9 / 11.0 / 6.2 / −1.9 / 0.6 dB, 63 Hz–4 kHz), the fill by up to 10.3 dB (940 minus 948; review). Fitting
    `drywall_filled_125mm` to 1049 and 937 early is the expressibility check of lumped double walls (1049 rises 13.5
    and 14.5 dB per octave from 125 to 500 Hz, then dips at coincidence). The other rows are tier 2 from a double-wall
    model. The precise option stays: `gypsum_leaf_12_5mm` | box of cavity air (| porous fill solid) | leaf. How a
    solid `drywall` row behaves at other drawn thicknesses is open (4.7).
12. **Dry lining.** A recipe, since the masonry row underneath stays the same.
13. **Plywood.** Starting columns: page 11.4 row "plywood 18 mm" (side 1.6971 m). Boarded floors on joists use
    1.15, panelling on battens 1.14; this plate is for free panels.
14. **Wood panelling.** Compilation values only.
15. **Floor boards on joists.** A board leaf on a box of air; the joists are not modeled.
16. **Parquet.** Glued parquet on a solid; parquet on battens is a leaf on a box (as 1.15).
17. **Laminate.** A thin HDF leaf on `underlay` (3.3), a traced porous body: sheet | solid | solid with a wave-medium
    body on a mass-law slab.
18. **Screed.** A solid wherever something lies on it (rule 1.7.4).
19. **Floating screed.** The resilient layer is a fill; a lumped row or the recipe is open (4.6).
20. **Ceramic tile, vinyl.** Hard faces glued on a solid.
21. **Metal sheets.** Kosala's opening is reduced and of unstated size.
22. **Doors.** Solid and hollow-core doors as lumped sheets; no R of either.
23. **CLT.** Measured R with mass; relevant for timber buildings only.

### 2.2 Class 2: hard furnishing

| # | name | intended use | recipe | T | A | valid | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.1 | `furniture_board`, `furniture_board_19mm`, `furniture_board_25mm` | table tops, shelves, open furniture panels (MDF, chipboard, solid wood) | sheet for an open single panel; solid for a thick top or a frame | 2 | 3 | — | constants: Hopkins MDF 1094–1096, chipboard 242–244 (both 760 kg/m³), softwood 1836, 1837 | no R or α of a free board | medium | wish |
| 2.2 | `box_cabinet` **(proposed name)** | wardrobes, cupboards, closed cabinets | hard box (2.4.1), asset standing on the floor | — | 1 | per object | (row) Nocke 1211 (cabinet without acoustic front, per object), 1213 (α on its cuboid, Sabine); with acoustic front 1210, 1212 | full wardrobes (request 13) | medium | wish |
| 2.3 | `box_bookshelf` **(proposed name)** | a filled bookshelf | hard box (2.4.1), asset | — | 3 | — | Odeon 104 (scattering only, no absorption) | absorption of a filled shelf (request 9) | medium | wish |
| 2.4 | `box_metal` **(proposed name)** | a hard box covered in metal: fridges, freezers, bar counters, metal cabinets | hard box (2.4.1); a glass front as a `glass_4mm` sheet glued on that face | — | 3 | — | the steel face: as 1.22 (no per-object data) | per-object area of a fridge | low | wish |
| 2.5 | `box_casing` **(proposed name)** | casings of TVs, speaker cabinets, lamps, appliances | hard box (2.4.1); a screen as a `glass_4mm` sheet on its face | — | 3 | — | constants of casing plastics: Hopkins PMMA 1345, 1346; Kosala PMMA R 1360, 1362, 1358 and polypropylene 1438–1441 (for the face) | — | low | wish |
| 2.6 | table and chairs (no row) | dining and office furniture | table top as a `furniture_board_*` sheet; legs and chair frames left out **(proposed)** | — | 1 | per object | per object: Nilsson 264 (table with two chairs), 1216 (one chair), 262 (10 tables + 19 chairs); BRAS 114, 115 (room fits) | Nilsson's values include scattering toward an absorbing ceiling | low | wish |
| 2.7 | sports equipment (no row) | metal pipes and bars | leave out (below the wavelengths of interest; scattering out of scope) **(proposed)** | — | — | — | none | — | low | wish |

Notes:

1. **Furniture board.** One family for MDF, chipboard and solid wood: its role is a reflector, and no R or α of any
   free board is in the data. The rig's inline `MDF` leaf is the leaf form for panel absorbers (`mdf_leaf_6mm`).
2. **Cabinet.** Nocke's cabinet (0.80 × 0.46 × 1.09 m) gives a per-object area and an effective α of its cuboid.
   S_E 3.11 m² is the cuboid without its floor face (3.115 against 3.483 m² for all faces, review): the asset stands
   on the floor; against a wall, the back face's share is lost. Wardrobe and cupboard rest on this one object and are
   one row.
3. **Bookshelf.** Tier 3 until request 9 returns.
4. **Metal box.** One material per closed body; finishes as sheets (rule 1.7.6).
5. **Casings.** Closed casings are hard boxes (rule 1.7.7), not boxes of sheets.
6. **Tables and chairs.** Items on a desk are clutter (4.6).

### 2.3 Class 3: soft furnishing

| # | name | intended use | recipe | T | A | valid | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | `carpet_tile_11mm` | carpet tiles, low-pile carpet | sheet on the floor solid; cut out under standing objects | 2 | 1 to 500 Hz; 1p above | 125 Hz–4 kHz | (comb.) Salford 227 (Sabine, on concrete), BCTC 226 (Sabine, cushion backed); Cox 223–225 (compilation spread). PTB#155 1537 is 5 mm needle felt (another product, printed) | flow resistivity, which the conversion at 1–4 kHz needs; 63 Hz | high | wish |
| 3.2 | `carpet_tufted_12mm` | tufted carpet without underlay | sheet on the floor | 2 | 1 | 125 Hz–4 kHz, 0° | (comb.) Broda 1930 (normal incidence, rigid backing); SAA 1929 | flow resistivity; random incidence | medium | wish |
| 3.3 | `underlay` | felt or foam underlay below a carpet or laminate | traced solid of its real thickness: sheet \| `underlay` \| floor solid | 2 | 2 | — | α alone: Broda 1186 (normal incidence, range only); (comb.) Broda 1931, PTB#156–158 1538–1540, Cox 1909, 1911 | flow resistivity of underlays (request 7) | medium | wish |
| 3.4 | `curtain_flat_0_2mm` | curtains hung flat | fabric leaf sheet \| box of cavity air (the wall distance) \| wall or window | 2 | 1p | — | (comb.) DelRey 428, 430 (Sabine, flat; fabric 0.2 mm, 300 g/m², 144 Pa·s/m), 432; σ: 1370 | random-incidence route for a hanging fabric; 31.5/63 Hz | high | wish |
| 3.5 | `curtain_pleated_0_8mm` | curtains with fullness, stage drapes | as 3.4, effective columns for the pleated fabric | 2 | 1p | — | (comb.) Zimmer+Rohde 412 (342 g/m², 0.79 mm, 1421 Pa·s/m, 100 % fullness, 100 mm from the wall, Sabine); 410, 406, 408; velour (compilation) Cox 976, 977, 1100, 1101; PTB#159 1541 | pleats are no plane: effective columns only | high | wish |
| 3.6 | `pu_foam` | small cushions, pillows, seat pads as meshed 3-D shapes | porous body of the cushion's shape; merged into a larger body it lies on | 2 | 2 | — | the foam: Dupont 1446–1449 (σ 10605 Pa·s/m², line 1447) | a free foam body has no reference; the object test | medium | wish |
| 3.7 | `polyester_fiber`, `polyester_fiber_20mm` | fiber pillows (solid), blankets and duvets (sheet) | solid, or sheet laid on a solid | 2 | 2 | — | σ: Garai 1385–1426 (fiber blankets 10–70 mm) | polyester model unresolved (research handoff, A14) | low | wish |
| 3.8 | `upholstery` | the body of soft furniture: sofas, armchairs, beds, mattresses, large cushions | porous body (2.4.2), assets in size variants; the fit is phase-3 work | — | 3 (1 for rows of seats, 4.10) | per object | per area of seating rows: PTB#484 1604, PTB#483 1603, Kath and Kuhl 1939 (Sabine); per object: SP 239, Nilsson 1216; compilation Cox 1943–1945, PTB#480 1600, Scheibler 1648, 1649 | per-object area of sofas, armchairs, beds (request 1) | high | wish |

Notes:

1. **Carpet tile.** Bands up to 500 Hz are hard-face (used as measured); 1–4 kHz need the porous conversion, which
   needs the flow resistivity, a listed gap. Every measured carpet lies on concrete or a rigid backing, so its
   references are combination references (section 3).
2. **Tufted carpet.** A normal-incidence reference: the rig needs its 0° comparison first.
3. **Underlay.** A traced porous body under a sheet; the floor below must be a mass-law body (two traced wave media
   back to back are off by 4.9 dB, stage-2b issue 3).
4. **Curtains.** Named by what a modeler sees, flat or pleated. What separates the two references is the fullness
   and the flow resistance (a factor of 9.9: 1421 against 144 Pa·s/m), not the mass (342 against 300 g/m², a
   factor of 1.14; review). A curtain away from the wall needs the box of cavity air.
5. **Foam bodies.** The porous columns follow from Miki or Komatsu at the measured σ (page 10.5). Generic solids of
   wave media are exact at any drawn thickness (the thickness enters the transfer matrix). Porous bodies must not
   touch each other (stage-2b issue 3).
6. **Fiber.** Garai's σ rows (lines 1385–1422) span fiber blankets of 12 to 110 kg/m³.
7. **Upholstery.** One porous row; its columns are fitted in phase 3 so that each asset's absorption area matches
   the per-object data. The per-area values hold for rows of seats, not for a single sofa.

### 2.4 Class 4: complex assemblies, and the object recipes

Objects that are not plane layers are modeled in one of three ways. Scattering and diffraction stay out of scope.

**2.4.1 Hard box** (hard closed furniture: bookshelf, cabinet, metal-clad furniture, casings). A closed box of the
object's outer size. Its faces carry a rigid row (mass column) whose `tf` is, per octave, the measured per-object
absorption area A divided by the box's absorbing surface S, fitted as a power law (a `tf` without a peak is flat in
angle and is a random-incidence α: page 10.3). S must be at least the largest A. **(proposed)** The box's mass is set
so that it passes nothing of interest. Scaling the asset scales S; the face α stays, so the absorption area scales
with the surface.

**2.4.2 Porous body** (soft furniture: sofa, armchair, bed, mattress, large cushions; a person; an audience). A
low-poly shape of the object, or a cushion-like box, of a porous row, fitted so that the body's absorption area
matches the per-object data. The body reflects at its face, absorbs inside and transmits some. Its absorption area
does not scale with its surface, so porous assets come in size variants (for example 1-, 2- and 3-seat sofas) and
are not scaled freely. An air-like absorbing volume is rejected for furniture: it reflects nothing.

**2.4.3 Absorbing volume** (clutter, vegetation). A closed volume of a medium matched to air (`a` = `e` = 1) with an
`alpha` column in dB/m. Its loss follows from a per-object area A of the volume V through Sabine's air term
A = 4mV: `alpha` = 4.343·A/(4V) dB/m (review).

**Per-object rule.** A per-object area measured in a reverberation chamber (Sabine, per object) is used like for
like, without conversion: the object test of the rig places the asset in a diffuse field (rays from all directions)
and computes its absorption area, which in a diffuse field is 4 × the absorption cross-section averaged over
directions. The placement of the measurement (on the floor, in a corner) is stated with each reference; an asset is
placed the same way, and placement values (corners) are no material property.

**Assets.** OBJ meshes generated by a script, so that they are the reader's input and testable through
`obj_file_read`; a Blender asset library is built on top (drag, place, scale where allowed). Placement rules are in
the modeling guidelines (section 8).

**Object test (phase 3).** Rays from all directions onto the body give its absorption area per octave, compared with
the per-object data. Designed and timed as its own rig step before the first object row.

| # | name | object | recipe | T | A | valid | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 | `upholstery` (assets `sofa_1seat`, `sofa_2seat`, `sofa_3seat`, `armchair` **(proposed)**) | couch, sofa, armchair | porous body (2.4.2) | — | 3 | — | as 3.8 (per seat or per floor area only) | per-object area of sofas and armchairs (request 1) | high | wish |
| 4.2 | `upholstery` on `furniture_board` (asset `bed` **(proposed)**) | bed | porous body (the mattress) on a traced solid of the frame | — | 3 | — | none | per-object area (request 1) | medium | wish |
| 4.3 | `box_bookshelf` | books in a bookshelf | hard box (2.2) | — | 3 | — | Odeon 104 (scattering only) | request 9 | medium | wish |
| 4.4 | `clothes_rack` **(proposed name)** | rack of clothing | porous body: a box of a light fiber medium around the hanging clothes **(proposed)** | — | 3 | — | none (query "clothing", "garment" finds nothing) | per-object area (request 13) | low | wish |
| 4.5 | `plant_foliage` **(proposed name)**; green wall as `green_wall_100mm` **(proposed name)** | plants indoors, green walls | potted plant: leave out if small, else an absorbing volume of the foliage outline (as 6.8) **(proposed)**; green wall: a porous sheet on the wall | — | 3 (potted); 1p (green wall) | — | green walls: Thomazelli 677–680, 1953 (Sabine); Davis 1951 (Sabine, single-number only) | per-object area of potted plants (request 13) | low | wish |
| 4.6 | `clutter_sparse`, `clutter_medium`, `clutter_dense` | items on desks and shelves, cluttered corners, dishes on a counter | absorbing volume (2.4.3) drawn over the items; revisable when data arrive | — | 3 | — | groups of furniture only: Nilsson 262, 264 (include scattering) | measured absorption of cluttered desks and shelves (request 10) | medium | wish |
| 4.7 | `person` **(proposed name)** (assets `person_seated`, `person_standing`) | a listener, a performer | porous body of a low-poly human shape **(proposed)** | — | 1 | 125 Hz–2 kHz | (row) per person: Kath and Kuhl 1263 (seated, isolated), 1264 (standing); Nishidate 1260 (design value), 1265, 1266 (in situ); Chagok 1262 not usable | 63 Hz | medium | wish |
| 4.8 | `audience_seated` **(proposed name)** | dense audience | porous box over the seating area, of the height of seated persons, fitted to the absorption per seating area **(proposed)** | — | 1 | 125 Hz–4 kHz | (row) per area: Kath and Kuhl 80, 81 (Sabine), 82, 83 (in situ); Bradley 1202, 1203 (occupied seats, Sabine; up to 1.051 / 1.088, values above one marked); Carvalho 250 (per object), 252 (Sabine per area 1.26–1.65, above one, marked); Kath and Kuhl 1937, 1938 | edge effects (Kath and Kuhl 77, 79: strip widths) | high (venues) | wish |
| 4.9 | dishes and glasses (no row) | kitchens, bars | leave out, or inside a clutter volume **(proposed)** | — | — | — | none | — | low | wish |
| 4.10 | `seating_empty` **(proposed name)** | unoccupied rows of seats, pews | porous box over the seating area, as 4.8 **(proposed)** | — | 1 | 250 Hz–2 kHz | (row) Kath and Kuhl 1939 (Sabine), 1940 (in situ); PTB#484 1604; Carvalho 251 (pews, per object) | 125 Hz in the chamber value | medium (venues) | wish |

Limits: the body reflects specularly where the object scatters; a per-object area measured in a diffuse field
contains the edge and diffraction effects of the object in that field; per-object areas measured in groups
(Nilsson's classroom units) include scattering toward an absorbing ceiling and are room-dependent below 500 Hz
(line 264 note); placement values are no material property. Clutter mainly scatters, which is out of scope; its
volume gives only absorption.

### 2.5 Class 5: acoustic treatment

| # | name | intended use | recipe | T | A | valid | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 | `glass_wool`, `glass_wool_50mm`, `glass_wool_100mm` | porous panels, cavity fill | sheet on a rigid solid; solid for a fill or a block | 2 | 1 | 250 Hz–4 kHz | (comb.) Dupont 647 (random incidence, sound-field synthesis), 645 (JCA prediction), 648 (Sabine, above one, marked); σ 646 (13112 Pa·s/m²) | 31.5/63 Hz; 125 Hz only as a model | high | wish |
| 5.2 | `rock_wool`, `rock_wool_50mm` | porous panels, cavity fill | as 5.1 | 2 | 1 | 250 Hz–4 kHz | (comb.) Kosten 1770 (random incidence from the tube, locally reacting), Kuhl 1145; Sabine 1767, 1147 (above one, marked) | σ not measured | medium | wish |
| 5.3 | `pu_foam_50mm` | foam panels | sheet on a rigid solid | 2 | 1 | 250 Hz–4 kHz | (comb.) Dupont 1448 (sound-field synthesis), 1446 (prediction), 1449 (Sabine); σ 1447 | 31.5/63 Hz | medium | wish |
| 5.4 | `melamine_foam_50mm` | studio foam | sheet on a rigid solid | 2 | 2 | — | BASF 1108 (Sabine, mounting not stated, marked); σ range Cox 1102 | mounting; σ of the product | medium | wish |
| 5.5 | `polyester_fiber_50mm` | PET felt panels | sheet on a rigid solid | 2 | 2 | — | σ: Garai 1385–1426; Szelag 1215, 1935 (type not stated) | polyester model (A14) | medium | wish |
| 5.6 | porous panel with an air gap (no row) | panels stood off the wall | absorber sheet \| box of cavity air \| wall | 2 | 2 | — | compilation 513 (type not stated); practical 655 (ISO 11654); GIK 555 (per object in corners, J-mount) and 554 (A-mount on the floor, no air gap; Sabine above one): no typed measurement of a panel with an air gap | a typed measurement with an air gap | high | wish |
| 5.7 | bass trap (no row) | corner traps | porous body of the trap's shape (5.1–5.3) in the corner, as an asset **(proposed)** | — | 2 | — | per object in corners (placement values, printed): GIK 553, 557; Auralex 86; Sabine 1511 (above one, marked); the note at 555 reads 0.6 m² flat against 1.34 m² in the corner at 125 Hz | the same products measured away from corners | high | wish |
| 5.8 | `mdf_leaf_6mm` | panel absorber, empty or filled cavity | leaf sheet \| box of cavity air (\| fill solid) \| wall | 2 | 1p | — | (comb.) Shearer 1120, 1121 (Sabine); f₀ 1115, 1116 | `rr` for room-size leaves; panel conversion | high | wish |
| 5.9 | `plywood_leaf_3mm` | panel absorber | leaf sheet \| box of cavity air \| wall | 2 | 1p | — | (comb.) Sakagami 1314, 1315, 1316 (Sabine; 30/90/150 mm); f₀ 1317–1319 | figure readings of low confidence | medium | wish |
| 5.10 | `cpa_steel_1mm` | compound panel absorber (steel bonded to foam) | steel leaf sheet bonded to a fill solid \| wall | 2 | 1p | — | (comb.) Fuchs 319, 318, 327, 328 (Sabine; above one: placement) | resonance on the fill's frame stiffness, which a rigid-frame fluid lacks; coincidence of the steel | medium | miss candidate |
| 5.11 | `steel_plate_2_5mm` | free-floating plate absorber | leaf on a fiber-mat fill | 2 | 1p | — | Nell 549, 550 (Sabine, marked); f₀ 1302 (20.5 Hz) | resonance on the mat's stiffness; placement moves 125 Hz from 0.41 to 0.64 | low | miss candidate |
| 5.12 | `mpp_steel_0_5mm` | microperforated cassette ceiling | MPP leaf sheet (equivalent fluid) \| box of cavity air \| slab | 2 | 1p | — | (comb.) Fuchs 1178, 1177 (Sabine), 1179, 1180 | 31.5/63 Hz; perforation ratio inconsistent; MPP conversion | medium | wish |
| 5.13 | `mpp_foil_0_1mm` | MPP foil, sails | leaf \| box \| wall | 2 | 2 | — | Cox 1130, 1933, 1140 (type not stated) | coefficient type | low | wish |
| 5.14 | `ceiling_tile_25mm` | suspended absorbing ceiling | porous tile sheet \| box of cavity air (the plenum) \| slab | 2 | 2 | — | practical: Ecophon 1908 (200 mm plenum), Nilsson 235–237; compilation: Cox 15, 16, Scheibler 1642 | random-incidence or Sabine α with build (request 4) | high (offices, venues) | wish |
| 5.15 | `perforated_gypsum_12_5mm` | perforated plasterboard ceilings | perforated leaf \| box \| slab | 2 | 2 | — | practical: Rigips 1252, 1250; PTB#316 1562 (Sabine) | full build with a typed curve (request 5) | medium | wish |
| 5.16 | slotted steel covers on a porous layer (no row) | slotted ceilings | not a plane layer | — | 3 | — | Fuchs 1823, 1822, 1824 (Sabine, figure) | — | low | miss candidate |
| 5.17 | diffusers (no row) | QRD and skyline diffusers | absorption as a hard box `box_diffuser` **(proposed name)**; the scattering is out of scope | — | 3 | — | compilation: Cox 3, 4 (1D QRD), 8, 9 (2D QRD; covered and uncovered) | scattering | medium | miss candidate |

Notes:

1. **Glass wool.** Starting columns: page 11.3 (also the rig's inline `WOOL`), Komatsu at 13112 Pa·s/m², `fRef`
   0.87403: `a` 1.1362, `b` −0.043932, `c` 0.0023911, `d` 1.2533, `e` 2.2495, `f` −0.45515, `g` 0.088044,
   `h` 0.14944, `thk` 0.05. Transmission through a treated wall is over-insulated up to 8.1 dB at 31.5 Hz (stage-2b
   issue 1); room-side reflection is right. T is tier 2: the fluid layer of the columns is exact.
2. **Rock wool.** The rig's inline `ROCKWOOL` is Miki at 32.8 kPa·s/m² (`fRef` 0.87403, `a` 1.31823, `b`
   −0.0125593, `c` 0.00481048, `d` 1.80573, `e` 0.823363, `f` 0.0954384, `g` 0.221513, `h` −0.0150261), a leaf-test
   fill, not fitted to Kosten's product; Kosten's curve is locally reacting, the library's layer reacts extendedly.
3. **PU foam.** Fitted-σ check: 10.3 against 10.6 kPa·s/m² (material-list analysis).
4. **Melamine, polyester.** No usable typed reference.
5. **Air gap.** No typed measurement with an air gap: tier 2 (transfer matrix).
6. **Bass traps.** All references are corner values (placement); the porous body is checked by the object test.
7. **Panel absorbers.** The loss ratio `rr`, the damping branch and the Sabine-to-random conversion for panels are
   decided with the first panel material. Starting columns: the rig's inline `MDF` leaf.
8. **CPA, Nell plate.** Both resonate on the frame stiffness of a bonded fill or mat ("a fill bonded to the plate
   acts through its own stiffness", research handoff); the engine's fills are rigid-frame fluids and a leaf has no
   bending stiffness. Miss candidates.
9. **MPP.** The stated open area 0.45 % does not match the stated grid (0.64 %).
10. **Ceiling tile.** Practical coefficients (ISO 11654) are another quantity.
11. **Diffusers.** Standard listening-room treatment; their scattering is out of scope, their absorption can be a
    hard box. Miss candidate for the scattering.

### 2.6 Class 6: outdoor

Ground and porous asphalt are porous media at the measured flow resistivity. **Ground model (decided):**
Delany-Bazley at the class σ, as the classes are defined (data lines 699–712: Harmonoise/Nord2000 impedance classes of the Delany-Bazley
one-parameter model), with a passivity check; Miki at the same σ is printed; where Delany-Bazley fails the passivity
check, Miki is refitted to the class impedance. At short range the ground acts nearly rigid at low frequency: in the
review's sizing (σ 12.5–200 kPa·s/m², 20–200 m) the 63–250 Hz excess attenuation lies between +5.3 and +6.0 dB with
both models. The spherical-wave limit (the engine reflects plane waves) is in the modeling guidelines.

| # | name | intended use | recipe | T | A | valid | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1 | `ground_snow` | snow cover | porous half-space (thick traced solid) | — | 2 | — | class A 12.5 kPa·s/m² (699; G 1, 698); Ishida 1831 (2–25 kPa·s/m², laboratory); Piercy 480 (15–30 kPa·s/m²); Bashir 869; FHWA 1925 (powder 10, granular 40 kPa·s/m²) | snow depth (a layer) | high (winter sites) | wish |
| 6.2 | `ground_forest_floor` | forest floor, moss, heather | porous half-space; a layer form is the alternative | — | 2 | — | class B 31.5 kPa·s/m² (701; G 1, 700); Bashir 714–719, 721, 846–849; Nordtest 547, 1833; Piercy 548 | half-space fits miss by 24.5–28.1 dB (715, 717, 719), layer fits by 6.0–17.0 dB (714, 716, 718; review) | medium | wish |
| 6.3 | `ground_grass` | lawns, pasture, sports fields | porous half-space | — | 2 | — | lawn by season: Attenborough 844 (summer, 80 kPa·s/m²), 845 (winter, 200 kPa·s/m²); class C 80 (702; G 1, 703), class D 200 (705; G 1, 704); Piercy 670 (150–300); Bashir 722–727; Van Renterghem 671, 672; FHWA 666–668, 1925; Nordtest 1776 | one row or two by season (4.9) | high | wish |
| 6.4 | `ground_soil` | compacted soil, earth | porous half-space | — | 2 | — | class E 500 kPa·s/m² (707; G 0.7, 706); Piercy 482 (rain-packed earth 4000–8000 kPa·s/m²); Bashir 720; Attenborough 870 (soil with winter wheat, 170 kPa·s/m²); Nordtest 674, 675; FHWA 1844, 1925 | — | medium | wish |
| 6.5 | `ground_gravel_loose` | loose gravel, gravel beds, gabion fill | porous half-space, or a layer on soil | — | 2 | — | Van Renterghem 673 (gravel 10 kPa·s/m²); Bashir 834–843 (layers 0.04–0.22 m and semi-infinite, 9.5–85 kPa·s/m², review), 871 | — | medium | wish |
| 6.6 | `ground_gravel_compacted` | gravel roads, car parks | porous half-space | — | 2 | — | class F 2000 kPa·s/m² (709; G 0.3, 708); Nordtest 674; Piercy 1214 | — | medium | wish |
| 6.7 | `asphalt`, `asphalt_porous_40mm` **(proposed thickness)** | dense asphalt and concrete paving; porous (drainage) asphalt | dense: a rigid traced solid with a small face absorption; porous: a thin porous sheet on the `asphalt` solid | — | 2 | — | dense: class G 20 000 kPa·s/m² (711; G 0, 710), class H 200 000 (712; G 0, 713); FHWA 74, 75; Piercy 76. Porous: Bashir 857 (67.5 kPa·s/m², porosity 0.18, tortuosity 2.36, layer 36 mm), 856 (2–15 kPa·s/m², non-acoustic) | no absorption coefficient of asphalt in the data (request 12) | high | wish |
| 6.8 | `vegetation` | foliage of trees, hedges, shrubs | absorbing volume (2.4.3) around the foliage; `alpha` and `alphaB` fitted to the excess attenuation per meter | 2 | — | 63 Hz–8 kHz, paths 20–200 m | ISO 9613-2 Annex A through Witte 546 (0.02 dB/m at 63 Hz to 0.12 dB/m at 8 kHz), 545 (10–20 m, absolute dB; consistent with 546 within 0.6 dB, review); Witte 1949 (vegetation strip); Van Renterghem 456 (hedges, insertion loss) | — | medium | wish |

Notes:

1. **Ground rows** are per class, not per site; the site spreads are printed beside the class value.
2. **Snow.** A layer of finite depth matters at low frequency; the half-space holds for deep snow **(proposed)**.
3. **Gravel** is two materials: loose (about 10 kPa·s/m²) and compacted (class F, 2000 kPa·s/m²), a factor of about
   200 (review).
4. **Asphalt.** A hard face whose small absorption follows from the ground model at its σ; no measured absorption
   coefficient was found, so A is tier 2.
5. **Vegetation.** The air-matched absorbing volume is right for foliage (it reflects little and attenuates along
   the path), unlike for furniture. Tier 2: the values are an engineering table of a standard, typed catalog.
6. **Ice** was dropped (no data; a hard face like `asphalt`).

### 2.7 Auxiliary rows

| name | use | recipe | T | A | status |
| --- | --- | --- | --- | --- | --- |
| `cavity_air` | the air of a box behind a leaf, a curtain, a panel, a floor on joists, a suspended ceiling | traced closed box; columns `a` 1, `e` 1.000001 (rig inline `AIRBOX`): a material identical to air is air | definition | definition | wish |
| `water_20c` | ponds and pools; a half-space test of the mapping | traced solid; constants c 1482.36 m/s, ρ 998.207 kg/m³, Z 1.4797e6 Pa·s/m (lines 1959, 1957, 1958); the rig tests it as a sheet, since its stack rule refuses a lossless wave medium with a critical angle as a traced solid | 1 | 1 | wish |

## 3. Test references that are combinations (measured only)

Each combination is a rig test of the materials involved, room side first. "Rig today" says whether the rig can
build it now.

| combination | materials tested | measured whole (line, type) | rig today | notes |
| --- | --- | --- | --- | --- |
| carpet on concrete | `carpet_tile_11mm`, `carpet_tufted_12mm`, `concrete` | Salford 227, BCTC 226 (Sabine); Broda 1930 (normal incidence, rigid backing); compilations 1910, 973, 1981 | yes; Broda needs the 0° comparison | transmission touches stage-2b issue 1 |
| carpet with underlay on concrete | `carpet_*`, `underlay`, `concrete` | Broda 1931 (normal); compilations PTB#156–158 1538–1540, 974 | sheet \| solid \| solid: yes; 0° comparison missing | — |
| porous absorber on a wall | `glass_wool_50mm`, `pu_foam_50mm`, `rock_wool_50mm` on `concrete` | Dupont 647, 1448 (random incidence); Kosten 1770 (from the tube) | yes (rig case exists for glass wool) | — |
| porous absorber with an air gap | wool \| `cavity_air` \| wall | none typed (513 compilation, 655 practical) | yes | plausibility only |
| panel absorber over a cavity | `mdf_leaf_6mm`, `plywood_leaf_3mm`, `cavity_air`, fill | Shearer 1120, 1121; Sakagami 1314–1316 (Sabine) | yes | panel conversion pending |
| curtain in front of a wall | `curtain_flat_0_2mm`, `curtain_pleated_0_8mm`, `cavity_air` | DelRey 428, 430 (flat); Zimmer+Rohde 412, 410; ADO 406, 408 (pleated) | yes | fabric route pending |
| drywall, precise recipe | `gypsum_leaf_12_5mm`, `cavity_air`, `glass_wool` | Halliwell 937, 1049 (filled, the build), 940, 943, 946, 948, 1050 (other builds); Fuchs 2005 478 (fill not stated) | leaf \| box \| fill \| leaf: needs a self-check | the same walls test the lumped `drywall_*` rows |
| plasterboard on frame, room side | `gypsum_leaf_12_5mm`, `cavity_air` | PTB#341 1568, PTB#342 1569 (Sabine); Bradley 923 (model of chamber data) | yes | — |
| dry lining on masonry | `gypsum_leaf_12_5mm`, `cavity_air`, `masonry_*` | Zeitler 1066–1086 (ΔR) with base walls 352, 351; Hopkins 444, 443 (dabs, ΔR); BIA 1797 / 1795 (dabs), 1845 / 1847 (furring) | yes (leaf \| box \| wall) | issue 1 for a fill |
| plaster on masonry (exceptions) | `plaster_15mm` on a bare masonry solid | BIA 1795 / 1796, 1905 / 1904 (plaster one face) | sheet \| solid: yes | the masonry rows themselves are plastered |
| plaster on both faces of a block | plaster sheets on both faces of a block solid | Hopkins 448 (also the heavy-masonry row reference; here only as the fixture of stage-2b issue 4) | no: needs a sheet behind a solid | ±3 dB, ISSUE 4 |
| parquet on concrete | `parquet_15mm`, `concrete` | PTB#394 1581 | yes | — |
| carpet on timber | `carpet_tile_11mm`, `plywood_19mm` or `board_leaf_22mm` | none in the data (request 6) | yes | plausibility fixture until measured |

## 4. Open design questions for phase 3

1. **The generic solid across thickness.** One generic fit across the measured thicknesses, asserted inside the
   measured range and printed outside (decided). Test it first on `masonry_heavy` (the 95–305 mm series), then
   concrete within 150–203 mm. `coiF`, `resF`, `att` are fixed columns while fc·h of 17.1 Hz·m gives 171 Hz at
   100 mm and 28.5 Hz at 600 mm; if the miss is too large, a thickness-scaled coincidence goes to the stage-2b list.
2. **`side` values.** The rule is decided; the typical sizes of 1.8 are proposed.
3. **Objects.** The object test's design and cost; the porous columns fitted for a soft body (σ, or σ and a loss
   column); the asset size variants; the kind proposed for clothes racks, plants, persons and audiences.
4. **Reference row per class.** Glass (EN 12758 generic against Quirt), carpets (Cox minimum / mean / maximum),
   concrete (Litvin against Mahn). The masonry and drywall classes use the envelope ±1 dB (decided).
5. **Free porous bodies.** A cushion or a bass trap as a traced porous solid in air has no reference as a free body;
   the object test and the per-object data are the only checks.
6. **Lumped rows against recipes for building assemblies.** Stud partitions of other builds, floating floors and
   screeds: one sheet fitted to the measured construction, or the recipe. The recipe keeps the room-side absorption
   and the cavity resonance; the lumped row keeps the measured transmission including the studs; mass-law bodies do
   not resonate with a fill.
7. **A solid drywall row at other drawn thicknesses.** Its mass is mostly the boards, not proportional to its
   thickness, while the mass term of a traced solid scales with the traced thickness (page 5.4); the measured depth
   effect comes from the cavity resonance, so the miss will be large with a predictable sign (review).
8. **Masonry class boundaries.** Whether three classes cover what a modeler sees, and whether sand-lime (not in the
   data) belongs to `masonry_heavy` without a reference.
9. **Ground.** One grass row or two by season (844, 845); a layer form for snow, loose gravel and forest floor.
10. **Conversions first.** The Sabine-to-random routes for panel absorbers, microperforated panels, hanging fabrics
    and green walls decide every "1p" entry; they come first in phase 3.
11. **One material per closed body, finishes as sheets.** To be confirmed in phase 3 (a sheet glued on a hard box's
    face).

## 5. Procedure for adding a material later

1. **Who decides.** The maintainer decides whether a material enters the list; anyone may propose it.
2. **Fields of a new entry.** Name (1.6: no suffix for a solid, `_<NN>mm` last for a sheet), class, intended use,
   recipe, T and A tiers, valid range, references in hand (research key, line, coefficient type, row or combination
   reference), gaps, impact, status `wish`; a lumped entry states its surface mass; a mass-law row its typical
   `side` (1.8). The rules of 1.7 apply.
3. **References.** Search the research library first (`data/material_properties.csv`, read-only). If nothing usable
   exists, write a draft research request in section 6 (column or class, quantity and coefficient type, bands,
   whether closed sources are acceptable, whether the work can proceed without it). The maintainer passes it to the
   research manager; the answer comes back as data rows with source, location and coefficient type.
4. **Phase-3 pickup.** The material enters the work order by tier and impact. One worker builds its row family and
   its lean case in `tests/catch2_tests/test_acoustic_materials.cpp` by the procedure of 1.10; engine-limited checks
   are marked ISSUE; the maintainer reviews and commits; the status moves to `confirmed`, or to `recorded miss` with
   the issue in `md/acoustic_engine_stage2b_issues.md`. Object rows also get their generated asset.
5. **CSV traps.** An empty cell reads 0, not the column default; a text cell in a numeric column is an error; a
   material file replaces the built-in table.

## 6. Draft research requests, ordered by impact

1. **Sofas, armchairs, beds** (classes 3 and 4; high). Per-object equivalent absorption area, ISO 354, 63 Hz–4 kHz,
   with the object's size, upholstery and placement. Manufacturer reports (tier B) acceptable. Blocks 3.8, 4.1, 4.2
   above tier 3.
2. **Wooden floor on joists** (class 1; high). Measured absorption with build (board thickness, joist depth, cavity
   fill), Sabine or random incidence, 63 Hz–4 kHz; laboratory R of the bare floor if available. Closed sources
   acceptable. Not blocking.
3. **Doors** (class 1; medium). Laboratory R (ISO 10140 or ASTM E90) of a solid timber door and of a hollow-core
   door with build, size and seals, 63 Hz–4 kHz; absorption of a hollow-core door. Open sources first. Blocks the
   transmission of both door rows.
4. **Suspended absorbing ceilings with build** (class 5; high in venues). Sabine or random-incidence α of
   mineral-fiber or glass-wool ceiling tiles with tile thickness, density or σ, and plenum depth, 63 Hz–4 kHz.
5. **Perforated and slotted panels with full build** (class 5; medium). Sabine or random-incidence α with hole or
   slot geometry, open area, plate thickness, cavity depth and fill, sample size, 125 Hz–4 kHz (63 Hz if possible).
   Tier A first. Blocks 5.15, 5.16.
6. **Carpet on a timber floor** (class 3; medium). Absorption of a carpet of stated build laid on boards on joists,
   and the same carpet on concrete; flow resistivity of carpet piles with build; 63 Hz–4 kHz. Not blocking.
7. **Flow resistivity of carpet underlays** (class 3; medium). Felt, foam and rubber underlays with thickness and
   density. Not blocking.
8. **Floating laminate floors** (class 1; medium). Absorption of laminate on foam underlay over concrete, Sabine or
   random incidence, with build, 63 Hz–4 kHz. Not blocking.
9. **Filled bookshelves** (classes 2 and 4; medium). Absorption per object of a shelf filled with books, ISO 354,
   125 Hz–4 kHz, with the shelf's size. Blocks `box_bookshelf` above tier 3.
10. **Cluttered desks and shelves** (class 4; medium). Absorption per object or per area of a desk or shelf with
    typical items, ISO 354, 125 Hz–4 kHz, with a description of the fill. Revises the clutter rows.
11. **Masonry walls missing in the data** (class 1; medium). Laboratory R of plain aircrete and sand-lime walls with
    density and thickness; clay brick and hollow units plastered on both faces; 63 Hz–4 kHz. Tier A search. Not
    blocking.
12. **Absorption of asphalt** (class 6; medium). Absorption coefficient of dense and porous asphalt (in situ or tube,
    with build), 125 Hz–4 kHz. Not blocking.
13. **Potted plants, clothing racks, full wardrobes** (classes 2 and 4; low). Per-object absorption area, ISO 354,
    with placement. Not blocking.

(The numbering in section 2 refers to these requests by number.)
