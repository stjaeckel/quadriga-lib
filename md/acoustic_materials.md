# Acoustic materials: the wish-list (draft 2)

Draft 2, 2026-10-26 (draft 0 revised after two review rounds). This list is the input to the work on the shipped
acoustic material table (`data/materials_acoustic.csv`). The rows, their columns and their tests are built later,
material by material, from the entries below. Nothing here is a row yet, except where the status says so.
Everything still marked **(proposed)** is the drafter's proposal and waits for review: impact ratings, typical panel
sizes, sheet thicknesses chosen by the drafter, recipes for tier-3 materials and objects.

The companion document `md/acoustic_modeling_guidelines.md` says how a venue is modeled for the acoustic tracer. The
modeling rules quoted here in short come from there.

## 1. Purpose, classes, tiers, impact, status, names

### 1.1 Purpose

A venue or a residential room, indoors or outdoors, is modeled in Blender. Materials are chosen mostly by informed
guess, and the simulation then optimizes speaker placement and acoustic treatment. The table must therefore offer,
for every surface a modeler meets, a row that is physically sound, named so that it can be picked from what the
modeler sees (a plan, a photo, a site visit) without reading the literature, and tested against the best reference
available. This list says which rows are wanted, how each is used in a model, what evidence backs it and what is
missing.

### 1.2 The six material classes

1. **Structural building materials** ("empty rooms"): concrete, masonry, single-pane window glass, double glazing,
   ceiling boards, floor boards, floor screed, laminate flooring, metal sheets, drywall with an air or mineral-wool
   fill.
2. **Hard furnishing:** wooden tables, wooden and metal chairs, wardrobes, cupboards, bookshelves, metal fridges,
   plastic casings of electronics (TV screens, toasters, speaker cabinets, lamps), sports equipment (metal pipes and
   bars). A stage is modeled in detail from class-1 rows and has no entry here.
3. **Soft furnishing:** cushions and pillows as meshed 3-D shapes, blankets, carpets, curtains, mattresses,
   upholstery.
4. **Complex assemblies:** racks of clothing, couches, books in a bookshelf, plants, clutter, a single person and a
   dense audience, dishes and glasses.
5. **Acoustic treatment:** porous absorbers, bass traps, panel absorbers, microperforated panels, suspended
   absorbing ceilings.
6. **Outdoor:** ground (grass, soil, gravel, snow, forest floor), asphalt, ice, vegetation.

Auxiliary rows that belong to no class (the cavity air of a box, water) are listed in section 2.7.

### 1.3 Evidence tiers

| tier | meaning | what checks the row |
| --- | --- | --- |
| 1 | A measured reference of the material's own quantity exists, with a known build and coefficient type (laboratory transmission loss, chamber or random-incidence absorption, per-object absorption area, measured flow resistivity, excess attenuation). | the measured values after their coefficient-type conversion, within about ±3 dB in transmission and ±0.05 in absorption, plus the independent physical model |
| 2 | First-principles physics gives the columns (plate theory, a transfer matrix, a porous or ground model at a known flow resistivity, a leaf on a box of air); references are absent or are compilation values of unstated type, which are shown but not asserted. | the independent physical model written in the test; compilation values printed as plausibility marks |
| 3 | An informed guess is the best available (a recipe built from neighbors, a per-object value of a similar object). | plausibility only; the guess and its basis are stated in the case |

Compilation values (Cox and D'Antonio Appendix A, the PTB table, the pyroomacoustics library) whose coefficient type
is not stated do not lift a material to tier 1. Hard faces (α up to about 0.2) are used as measured (material page,
section 10.5).

### 1.4 Impact

How much the material moves the simulated sound field of a typical room or site **(proposed ratings throughout)**:

- **high:** covers a large share of the surface in most rooms (walls, floor, ceiling, windows) or of an outdoor
  site (the ground), or dominates the low frequencies where a room's modes are (floors on joists, bass traps, panel
  absorbers), or is the main absorber of a furnished room (sofas, carpets, curtains, audience);
- **medium:** common but smaller in area, or large in area but close to a material already in the table;
- **low:** small objects, rare items, or items below the wavelengths that matter.

The work order is tier 1, then tier 2, then tier 3; within a tier, high before medium before low.

### 1.5 Status values

| status | meaning |
| --- | --- |
| wish | listed; nothing built |
| in work | a row or its case exists and is being built or redone |
| confirmed | the row is in the CSV and its case passes against its references |
| recorded miss | the engine cannot express the material; the miss and its size are recorded in `md/acoustic_engine_stage2b_issues.md` |

### 1.6 How rows are named

- **The thickness suffix marks a sheet.** A row whose name ends in `_<NN>mm` is a sheet of that thickness; the
  suffix always comes last. A row without it is a traced solid, whose thickness comes from the geometry. A decimal
  point becomes an underscore. Examples: `concrete` (solid), `concrete_150mm` (sheet); `drywall`, `drywall_filled`
  (solids), `drywall_filled_100mm` (sheet); `gypsum_leaf_12_5mm` (sheet).
- **Generic solid:** a traced solid named for the material (`concrete`, `masonry_heavy`). The mass term uses the
  traced normal thickness (material page, section 5.4). One set of columns is fitted across the measured thicknesses
  (open question 4.1).
- **Sheets at round thicknesses:** sheet rows exist at round thicknesses (`concrete_150mm`, `concrete_200mm`). A
  measured reference is compared with the sheet of the nearest round thickness (Litvin's 152 mm wall with
  `concrete_150mm`, his 203 mm wall with `concrete_200mm`), the thickness difference stated in the case. Product
  thicknesses keep their size (`gypsum_board_12_5mm`, `plywood_19mm`). Some materials exist only as sheets (window
  glass, glazing units, leaves, carpets, curtains).
- **Literature constructions are test references, not rows.** Their names do not appear in the table.
- **One board, two rows.** A board that is used both as a mass-law panel (a single pane, a wall leaf with coincidence)
  and as a wave-medium leaf on a box of air (a limp leaf that resonates with the cavity, without coincidence) has two
  rows: the plate row (`gypsum_board_12_5mm`) and a leaf row named `<material>_leaf_<NN>mm`
  (`gypsum_leaf_12_5mm`).
- **No combination rows.** Walls and floors are modeled first; carpets, curtains and absorbers are placed on them
  as separate rows. A combination is a rig test, and only where a measurement of the combination exists (section 3).
  A drywall row is no combination: it is the wall a modeler sees on a plan (2.1, entry 1.11).
- **Object rows.** Hard and soft objects (section 2.4) have rows fitted to per-object data and are delivered as
  Blender assets with the material assigned.
- **Reader rules** (`src/obj_file_read.cpp`, `base_material_name` and `resolve_csv`): the Blender material name is
  looked up exactly, then by its part before the first dot; there is no other fallback. Names are case-sensitive.
  Row names are snake_case and contain no dot, so that Blender's duplicates (`concrete.001`) and variants
  (`concrete.gray`) map to the row `concrete`. A name that matches no row becomes "no material" (index 0, air)
  without a warning unless the reader is called with `csv_strict`.

### 1.7 Panel size `side`

`side` is a row value: the √area of a typical panel of that use, stated per entry below. There is no engine change:
the engine sees one triangle, not the panel. Where a drawn panel differs from the typical size, the drift is accepted
(modeling guidelines, section 4). `side` acts only on mass-law rows; leaves and porous rows keep the infinite plate.
The typical sizes below are **(proposed)**; where a row is fitted to a laboratory reference, the reference is
evaluated at its own specimen side and the row carries the typical size. The engine default is 3.29 m.

## 2. The materials per class

Columns: **recipe** says how the row is used in a model (sheet, solid, stack, box of air, hard box, porous body,
absorbing volume) and what the modeler sets, including the typical `side` **(proposed)** for mass-law rows;
**references** give the research key, the line in the research library's `data/material_properties.csv` (header =
line 1) and the coefficient type; R = laboratory sound reduction index; σ = flow resistivity. Line numbers and
values come from the material-list analysis of 2026-10-23 or from queries of the research data on 2026-10-26.
Typical sides are √(width × height) of the stated panel, and nearest round thicknesses were picked, by script. Notes
below each table are numbered by entry and hold the starting columns where they exist.

### 2.1 Class 1: structural building materials

| # | name | intended use | recipe | tier | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | `concrete`, `concrete_150mm`, `concrete_200mm` | cast walls, slabs, ceilings | solid 100–600 mm (thickness from geometry) or sheet; `side` 3.22 m (wall 4 × 2.6 m) | 1 | R: Litvin 228 (152 mm), 229 (203 mm); Mahn 369, 371 (alternatives). α: PTB#30 1559 (Sabine), Nell 1730 (chamber boundary, from 31.5 Hz). Constants: Hopkins 374–376; Mahn η 370, 372 | 63 Hz R of cast concrete; 31.5 Hz; the miss of one fit across 100–600 mm (4.1) | high | in work |
| 1.2 | `masonry_heavy`, `masonry_heavy_100mm`, `_150mm`, `_200mm`, `_250mm` **(proposed thicknesses)** | solid clay brick, dense concrete block, sand-lime, about 1800–2400 kg/m³ | solid or sheet; `side` 3.22 m | 1 | R: BIA 273 (brick 95 mm), 1847 (brick 203 mm), 1845 (203 mm, two wythes), 1846 (305 mm), 1729 (241 mm), 1795 (SCR brick 140 mm); Warnock 349 (90 mm solid concrete blocks, 195.2 kg/m²). α: PTB#34 1566 (Sabine). Density: Hopkins clay brick 267, dense aggregate block 450 | sand-lime not in the data; 31.5/63 Hz R | high | wish |
| 1.3 | `masonry_hollow`, `masonry_hollow_150mm`, `_200mm`, `_250mm` **(proposed thicknesses)** | hollow clay brick, hollow concrete block | solid or sheet; `side` 3.22 m | 1 | R: Zeitler 352 (190 mm hollow normal-weight block, from 63 Hz), 350, 351 (140 mm hollow lightweight-aggregate block); Warnock 354 (190 mm hollow); BIA 1903 (194 mm hollow clay tile), 1905 (90 mm). α: Cox 346, 347 (compilation, spread) | measured α; 31.5 Hz | high | wish |
| 1.4 | `masonry_light`, `masonry_light_100mm`, `_150mm`, `_200mm`, `_250mm` **(proposed thicknesses)** | aircrete, lightweight and expanded-clay block (sealed or plastered) | solid or sheet; `side` 3.22 m | 1 | R: Hopkins 24 (aircrete 115 mm + plaster one side); Warnock 355 (190 mm lightweight block, plastered), 357 (90 mm wood-fiber aggregate, plastered). Density: Hopkins 88 (aircrete 400–800 kg/m³), 498 (expanded clay), 1040 (lightweight aggregate); Davy 93 (AAC panel 38.2 kg/m²). Constants: Hopkins 87, 89 | R of a plain aircrete wall (request 11) | medium | wish |
| 1.5 | `plaster_15mm` | plaster on masonry | sheet on the masonry solid (sheet \| solid); `side` as the wall | 1 | α: PTB#344 1571, PTB#346 1573 (Sabine). Combinations: section 3 (13 mm plaster compared at 15 mm) | plaster alone has no R; checked through its combinations | medium | wish |
| 1.6 | `glass_4mm`, `glass_6mm` | single-pane windows | sheet only; `side` 1.30 m (pane 1.2 × 1.4 m) | 1 | R: Pilkington 536, 537 (EN 12758 generic); Quirt 610, 612; AGC 611. α: PTB#207 1544 (Sabine). Constants: Hopkins 561–563 | reference row choice (4.4); 31.5/63 Hz | high | wish |
| 1.7 | `glazing_double_24mm`, `glazing_triple_36mm` | sealed units 4-16-4 and 4-12-4-12-4 | sheet only, one lumped body; `side` 1.30 m | 1 | R: AGC 592, Pilkington 1000, Quirt 579, 581; AGC 622, Quirt 614. α: PTB#208 1545 (Sabine) | 31.5/63 Hz R; unit absorption | high | wish |
| 1.8 | `glass_leaf_4mm` | two sashes with a wide air space (box windows) | leaf sheet \| box of cavity air \| leaf | 1 | R: Quirt 578, 577, 580 | coincidence not expressible by a leaf | low | wish |
| 1.9 | `gypsum_board_12_5mm` | ceiling boards, a single gypsum panel | sheet; `side` 3.22 m (partition 4 × 2.6 m) | 1 | R: Hopkins 1274, Northwood 956, Wareing 961, 962. η: Quirt 898, Hopkins 1285. Density: Hopkins 1284 | 31.5 Hz, 8 kHz | high | wish |
| 1.10 | `gypsum_leaf_12_5mm` | gypsum leaf over a box of air (suspended ceilings, linings, the precise drywall recipe) | leaf sheet on a box of cavity air | 1 | whole constructions in section 3 (Halliwell, Fuchs 2005, Zeitler); α: PTB#341 1568, PTB#342 1569 (Sabine, compilation), Bradley 923 | leaf loss ratio `rr` for room-size leaves | high | wish |
| 1.11 | `drywall`, `drywall_filled`; `drywall_100mm`, `drywall_125mm`, `drywall_150mm`; `drywall_filled_100mm`, `drywall_filled_125mm`, `drywall_filled_150mm` | stud partitions as a plan shows them: total thickness, maybe "filled" | lumped mass-law row fitted to measured walls of a standard build: one 12.5 mm gypsum board each side, metal studs, cavity = the rest, empty unless filled. Solid for walls drawn with thickness, sheet for walls drawn as one plane; `side` 3.22 m | 1 | filled, steel studs: Halliwell 937, 1049 (G13 SS90 GFB90, 115 mm → `drywall_filled_125mm`), 1050 (G13 SS150 GFB150, 175 mm → `_150mm`); 940, 943 (16 mm boards, 122 mm → `_125mm`). Empty: Halliwell 948 (16 mm boards, 122 mm → `drywall_125mm`); Fuchs 2005 478 (12.5/100/12.5, studs and fill not stated, 125 mm). α: PTB#341 1568, PTB#342 1569 (compilation) | no measured wall of the standard build at 100 mm; no empty wall with 12.5 mm boards; wood studs (931, 934) and resilient channels (951, 1052) are other builds | high | wish |
| 1.12 | `plywood_19mm` | panelling, boarded floors and ceilings | sheet; `side` 1.77 m (sheet 2.6 × 1.2 m) | 1 | R: Wareing 1334 (small specimen), 1333, 1335, 1336. α: Quirt 1344 (Sabine). Constants: Hopkins 1322–1325 | density of the measured specimens (710 kg/m³ assumed); 31.5/63 Hz | medium | wish |
| 1.13 | `board_leaf_22mm` | floor boards on joists | leaf sheet on a box of cavity air (joist depth) on the slab | 2 | α (compilation): PTB#393 1580, Cox 539, Scheibler 1709; BRAS 123, 124 (room fits) | measured α with build; any R (request 2) | high | wish |
| 1.14 | `parquet_15mm` | parquet glued on a slab | sheet on the slab solid | 2 | α (compilation): PTB#394 1581, Cox 1232; BRAS 162 (room fit); parquet on battens PTB#395 1582 | measured α with build | medium | wish |
| 1.15 | `laminate_leaf_8mm` | floating laminate on a foam underlay | leaf sheet \| `underlay` solid \| slab (or a lumped row: 4.6) | 2 | none found (query "laminate" returns only laminated glass and CLT) | absorption of a floating laminate floor (request 8) | medium | wish |
| 1.16 | `screed`, `screed_50mm` | sand-cement screed, bonded | solid or sheet on the slab; `side` as the slab | 2 | constants: Hopkins 1787–1792 | floating screed on a resilient layer (4.6) | medium | wish |
| 1.17 | `ceramic_tile_10mm` | tiled floors and walls (kitchens, bathrooms) | sheet on the screed or wall | 3 | α (compilation): Cox 1090; BRAS 193 (initial estimate) | none needed beyond a hard face | low | wish |
| 1.18 | `vinyl_3mm` | linoleum or vinyl stuck to the screed | sheet on the screed | 3 | α (compilation): Cox 1088, PTB#601 1606; Bork 1087 | — | low | wish |
| 1.19 | `steel_1mm`, `aluminum_2mm` | metal sheets, metal doors, cladding, appliance casings | sheet; `side` 1.04 m (appliance side 0.6 × 1.8 m) | 1 | R: Kosala 1874 (steel), 40 (aluminum 2 mm); Northwood 1877 (0.8 mm steel). Constants: Hopkins 1867–1869, 33–35 | specimen size of the references; installed loss factor | low | wish |
| 1.20 | `door_solid_wood_40mm` **(proposed thickness)** | interior and entrance doors | sheet; `side` 1.34 m (door 0.9 × 2.0 m) | 3 | α (compilation): PTB#396 1583, Cox 1850, Scheibler 1710; Pinhede 459 (room fit) | laboratory R of a door (request 3) | medium | wish |
| 1.21 | `clt`, `clt_100mm` | cross-laminated timber walls and floors | solid or sheet; `side` 3.22 m | 1 | R: Hoeller 395, 396, 397; η: 398–400 | — | low | wish |

Notes:

1. **Concrete.** Starting columns: the preliminary row `concrete_cast_152mm` in `data/materials_acoustic.csv` (the
   page 11.4 row, fitted at 343 m/s to Litvin at side 3.5229 m): `fRef` 0.00019663, `a` 3.85e-06, `e` 1901, `thk`
   0.152, `att` 14.459, `attB` −0.086239, `m` 1, `resF` 0.068191, `resQ` 1.4354, `resS` −1, `coiF` 6.0529, `coiQ`
   1.19, `tf` 0.00025486, `tfB` 0.52, `side` 3.5229. Its case `Material - concrete_cast_152mm` exists; it is redone
   as `concrete_150mm` against the research data's Litvin values (42.1 / 46.5 / 52.6 / 56.5 / 61.7 / 64.6 dB, line
   228). The fitted `coiF` corresponds to about 6.9 kHz, while Hopkins' fc·h of 17.1 Hz·m (line 374) gives 114 Hz at
   150 mm: the knobs carry what the plate does above coincidence and do not scale with thickness (4.1).
2. **Heavy masonry.** One class for what a modeler sees as a solid, heavy wall. The literature walls (the BIA bricks,
   Warnock's solid blocks) are its test references; their spread at equal surface mass sets the class's tolerance.
   Starting columns: page 11.4 rows "face brick 95 mm" and "solid brick 203 mm" (side 3.5229 m), compared with
   `masonry_heavy_100mm` and `masonry_heavy_200mm`. No sand-lime row was found (queries "sand-lime", "calcium
   silicate"). Plastered walls are tested as combinations (section 3).
3. **Hollow masonry.** Replaces `concrete_block_190mm` (a product size a modeler cannot see). Zeitler 350 has unsealed faces with the air leakage included; 351 is the same wall calculated without leakage. Starting columns: page
   11.4 row "concrete block 190 mm" (fitted to Warnock), compared with `masonry_hollow_200mm`. Zeitler 352 has 63 Hz,
   mass and side. The spread of the hollow block and hollow clay tile walls sets the tolerance.
4. **Light masonry.** Sealed or plastered walls only: unsealed porous blocks leak through their pores (Warnock 356,
   the unplastered twin of 355, reads 24.5 dB at 125 Hz against 34.1 dB; 358 against 357: 6.9 against 32.5 dB).
5. **Plaster.** The bonded form sheet | solid is exact in the engine (stage-2b issue 3: only two traced mass-law
   bodies deviate). A lumped row "brick 105 mm, plaster one face" exists on page 11.4 but is a combination and becomes
   a test reference.
6. **Glass.** Starting columns: page 11.4 rows "glass 4 mm" (also the rig's inline `GLASS4`) and "glass 6 mm", side
   1.3492 m (the test size, close to the typical pane). The EN 12758 generic values and Quirt's measured small panes
   differ by 5.6 to 6.6 dB at 125/250 Hz; which one is the reference is open (4.4).
7. **Glazing units.** Starting columns: page 11.4 rows "double glazing 4-16-4" and "triple glazing 4-12-4-12-4".
   One lumped mass-law body: two meshed mass-law sheets cannot resonate with a cavity (page rule 9). The name gives
   the unit's total thickness; the page rows carry `thk` 0.008 and 0.012 (the glass), which the row will keep for its
   mass **(proposed; review the name)**.
8. **Box windows.** Two 4 mm leaves on a 100 mm box of cavity air; a leaf has no bending stiffness, so the
   coincidence dip at 2–3 kHz is not expressible (page section 13).
9. **Gypsum board.** Starting columns: page 11.4 row "gypsum board 12.5 mm" (fitted to a thin-plate model, side
   1.6971 m, no measured spectrum then); the measured spectra above now exist (Hopkins specimen side 3.047 m).
10. **Gypsum leaf.** Starting columns: the rig's inline limp leaf `BOARD` (8.6 kg/m², loss ratio 0.2): `thk`
    0.0125, `a` 4.691e-5, `e` 573.0, `g` 573.0·0.2/17.98, `h` 1.0. Its loss ratio has no measured value for a
    room-size leaf (research handoff, gaps).
11. **Drywall.** The rows start from what a modeler sees: a plan gives the total thickness and perhaps "filled".
    Each is one lumped mass-law row fitted to measured walls of the standard build (boards 12.5 mm each side, metal
    studs, the rest cavity; glass or mineral wool if filled). The thicknesses of the measured walls were mapped to the
    nearest round row by script (115 mm → 125, 122 mm → 125, 175 mm → 150). The precise option stays documented:
    `gypsum_leaf_12_5mm` | box of cavity air (| porous fill solid) | `gypsum_leaf_12_5mm` (section 3); it keeps the
    room-side absorption and the cavity resonance and loses the studs. How a solid `drywall` row behaves at other
    drawn thicknesses is open (4.7).
12. **Plywood.** Starting columns: page 11.4 row "plywood 18 mm" (thin-plate model, side 1.6971 m). The 19 mm
    small specimen has side 1.21 m.
13. **Floor boards on joists.** A board leaf on a box of air; the joists are not modeled.
14. **Parquet.** Glued parquet is a thin sheet on the slab; parquet on battens is a leaf on a box (as 1.13).
15. **Laminate.** A thin HDF leaf on a few millimeters of foam underlay (`underlay`, 3.3): a leaf on a fill. No
    reference.
16. **Screed.** Bonded screed is a mass-law body bonded to the slab (one plate, page 5.4).
17. **Ceramic tile, vinyl.** Hard faces; the row matters for the face absorption only.
18. **Metal sheets.** Kosala's opening is reduced and of unstated size.
19. **Door.** Waiting for a measured R; the absorption exists as a compilation value.
20. **CLT.** Measured R with mass; relevant for timber buildings only.

### 2.2 Class 2: hard furnishing

| # | name | intended use | recipe | tier | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.1 | `mdf_19mm`, `chipboard_19mm` | table tops, shelves, open furniture panels, speaker cabinets | sheet faces of the object's mesh; `side` 1.20 m (table top 1.6 × 0.9 m) | 2 | constants: Hopkins MDF 1094–1096, chipboard 242–244 | no R or α of a free board | medium | wish |
| 2.2 | `solid_wood_25mm` | solid-wood tables and furniture | sheet; `side` 1.20 m | 2 | constants: Hopkins softwood 1836, 1837 | — | low | wish |
| 2.3 | `box_wardrobe`, `box_cupboard` **(proposed names)** | wardrobes, cupboards, closed cabinets | hard box (2.4.1), Blender asset | 1 | per object: Nocke 1211 (cabinet without acoustic front), 1213 (α on its cuboid, Sabine); with acoustic front 1210, 1212 | wardrobes full of clothes (request 13) | medium | wish |
| 2.4 | `box_bookshelf` **(proposed name)** | a filled bookshelf | hard box (2.4.1), Blender asset | 3 | Odeon 104 (scattering only, no absorption) | absorption of a filled shelf (request 9) | medium | wish |
| 2.5 | table and chairs (no row) | dining and office furniture | table top as a 2.1 sheet; legs and chair frames left out **(proposed)** | 1 | per object: Nilsson 264 (table with two chairs), 1216 (one chair), 262 (10 tables + 19 chairs); BRAS 114, 115 (room fits) | Nilsson's values include scattering into an absorbing ceiling | low | wish |
| 2.6 | `box_fridge` **(proposed name)** | fridges, metal cabinets, lockers | hard box (2.4.1) with the face of `steel_1mm` **(proposed)** | 3 | as 1.19 (no per-object data) | — | low | wish |
| 2.7 | `plastic_3mm` | casings of TVs, toasters, lamps; TV screens | sheet faces of a hollow box; `side` 0.92 m (TV 1.2 × 0.7 m) | 2 | R: Kosala PMMA 1360 (3 mm), 1362 (5 mm), 1358 (15 mm). Constants: Hopkins PMMA 1345, 1346; Kosala polypropylene 1438–1441 | ABS and polycarbonate not in the data | low | wish |
| 2.8 | sports equipment (no row) | metal pipes and bars | leave out (below the wavelengths of interest, scattering out of scope); a large frame as a thin closed body of `steel_1mm` faces **(proposed)** | 3 | none | — | low | wish |

Notes:

1. **Boards.** MDF and chipboard share Hopkins' density 760 kg/m³ (lines 1095, 243) and differ in fc·h (25.3 and
   29.5 Hz·m). The rig's inline `MDF` leaf (`a` 4.5e-5, `e` 775/1.2, `g` 775/1.2·0.2/17.98, `h` 1.0) is the leaf
   form for panel absorbers (`mdf_leaf_6mm`), not a furniture board.
2. **Solid wood.** Only softwood constants (440 kg/m³) are in the data.
3. **Wardrobes and cupboards.** Nocke's cabinet (0.80 × 0.46 × 1.09 m, S_E 3.11 m²) gives a per-object area and an
   effective α of its cuboid: the reference for the hard box. Its contents are part of the measured object.
4. **Bookshelf.** A hard box of the filled shelf; no absorption data yet, so tier 3 until request 9 returns.
5. **Tables and chairs.** Thin legs and tubes are below the wavelengths of interest; the top is a reflector. Items
   on a desk are clutter (4.6).
6. **Fridges.** A closed steel casing: a hard box whose face is the steel sheet's absorption.
7. **Plastic.** PMMA stands in for casing plastics.
8. **Sports equipment.** Left out unless large.

### 2.3 Class 3: soft furnishing

| # | name | intended use | recipe | tier | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | `carpet_tile_11mm` | carpet tiles, low-pile carpet | sheet on the floor (sheet \| solid) | 1 | α: Salford 227 (Sabine), BCTC 226 (Sabine, cushion backed); Cox 223–225 (compilation spread); PTB#155 1537 | flow resistivity; 63 Hz | high | wish |
| 3.2 | `carpet_tufted_12mm` | tufted carpet without underlay | sheet on the floor | 1 | α: Broda 1930 (normal incidence, rigid backing); SAA 1929 | flow resistivity; random incidence | medium | wish |
| 3.3 | `underlay` | felt or foam underlay below a carpet or laminate | traced solid between the carpet or laminate sheet and the floor (sheet \| `underlay` \| floor solid) | 2 | α alone: Broda 1186 (normal incidence, range only); with carpet: Broda 1931, PTB#156–158 1538–1540, Cox 1909, 1911 | flow resistivity of underlays (request 7) | medium | wish |
| 3.4 | `curtain_thin_0_2mm` | thin curtain hung flat | fabric leaf sheet \| box of cavity air (the wall distance) \| wall or window | 1 | α: DelRey 428, 430 (Sabine, flat; fabric 0.2 mm, 300 g/m²), 432 (alternative); σ: 1370 | 31.5/63 Hz | high | wish |
| 3.5 | `curtain_heavy_0_8mm` | heavy pleated curtains, stage drapes | as 3.4, effective columns for the pleated fabric | 1 | α: Zimmer+Rohde 412 (342 g/m², 0.79 mm, 100 % fullness, 100 mm from the wall, Sabine); alternatives 410, 406, 408; velour (compilation) Cox 976, 977, 1100, 1101; PTB#159 1541 | pleats are no plane: effective columns only | high | wish |
| 3.6 | `pu_foam` | small cushions, pillows, seat pads as meshed 3-D shapes | traced porous solid of the cushion's shape (fabric cover ignored) | 2 | the foam: Dupont 1446–1449 (σ 10605 Pa·s/m², line 1447) | a free foam body has no reference; the object test (4.3) | medium | wish |
| 3.7 | `polyester_fiber`, `polyester_fiber_20mm` | blankets and duvets (sheet), fiber pillows (solid) | sheet laid on a surface, or solid | 2 | σ: Garai 1385–1426 (fiber blankets 10–70 mm) | polyester model unresolved (research handoff, A14) | low | wish |
| 3.8 | `upholstery` | the body of soft furniture: sofas, armchairs, beds, mattresses, large cushions | porous body (2.4.2), Blender asset; the fit is phase-3 work | 1 for seating, 3 for beds and mattresses | per area or per object: PTB#484 1604, PTB#483 1603, Kath and Kuhl 1939 (Sabine); SP 239 (per object); Nilsson 1216; seating per floor area (compilation): Cox 1943–1945, PTB#480 1600, Scheibler 1648, 1649 | per-object area of sofas, armchairs, beds (request 1) | high | wish |

Notes:

1. **Carpet tile.** Starting columns: none yet (no flow resistivity). Bands up to 500 Hz are hard-face (as
   measured); 1–4 kHz need the Sabine-to-random conversion for a porous layer. Every measured carpet lies on
   concrete or a rigid backing, so its tier-2 check is the combination carpet | concrete (section 3).
2. **Tufted carpet.** A normal-incidence reference needs a 0° comparison in the rig (not there yet).
3. **Underlay.** A traced porous body under a sheet: sheet | solid is allowed; two traced wave media back to back
   are not (stage-2b issue 3), so the floor below must be a mass-law body. The underlay is drawn as a thin closed
   body of its real thickness.
4. **Thin curtain.** A curtain away from the wall needs the box of cavity air between it and the wall; a free air
   gap is followed bounce by bounce and runs out of interactions (page rule 9).
5. **Heavy curtain.** The G-100 reports give build and Sabine thirds; the fabric relaxation term is pending
   (research handoff, class 9).
6. **Foam bodies.** The porous columns follow from Miki or Komatsu at the measured σ (page 10.5). A traced porous
   body reflects at its face, absorbs along the path and transmits the rest; diffraction around it is out of scope.
7. **Fiber.** Garai's σ rows (lines 1385–1422) span fiber blankets of 12 to 110 kg/m³.
8. **Upholstery.** One porous row whose columns are fitted in phase 3 so that the asset's absorption area matches
   the per-object data (2.4.2). The per-area seating values hold for rows of seats, not for a single sofa; no sofa,
   armchair, bed or mattress is measured as an object.

### 2.4 Class 4: complex assemblies, and the object recipes

Objects that are not plane layers are modeled in one of three ways. The hard box and the porous body are delivered
as Blender assets with the material assigned: the modeler drags the asset into the scene, places it and scales it.
Scattering and diffraction stay out of scope.

**2.4.1 Hard box** (hard closed furniture: bookshelf, wardrobe, cupboard and similar). A closed box of the object's
outer size. Its faces carry a rigid row (mass column) whose `tf` is, per octave, the measured per-object
absorption area A divided by the box's surface S, fitted as a power law (a `tf` without a peak is flat in angle and
is a random-incidence α: page 10.3). S must be at least the largest A, since `tf` above one is no absorption
coefficient. **(proposed)** The box's mass is set so that it passes nothing of interest; scaling the asset scales
S, so the face α stays and the absorption area scales with the surface.

**2.4.2 Porous body** (soft furniture: sofa, armchair, bed, mattress, large cushions). A low-poly shape of the object,
or a cushion-like box, of the `upholstery` row. Its porous columns are fitted so that the body's absorption area
matches the per-object data. The body reflects at its face, absorbs inside and transmits some, as soft furniture
does. An air-like absorbing volume (a box that absorbs but has the impedance of air) is rejected for furniture: it
reflects nothing.

**2.4.3 Clutter box.** A box drawn over a desk, a shelf or a cluttered floor area, of a clutter row: a porous body of
an effective medium. The rows form a 2 × 3 grid, hard or soft by sparse, medium, dense (fill fraction about 10, 25,
40 %). Hard clutter is an effective medium of rigid objects in air: slightly denser and slower than air, little
loss. Soft clutter adds flow resistance. Tier 3; clutter mainly scatters, which is out of scope.

**Object test (phase 3).** Each object row needs a rig test that sends rays from all directions onto the body and
reports its absorption area per octave, for comparison with the per-object data. The rig has no such test yet.

| # | object | intended use | recipe | tier | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 | couch, sofa, armchair | living rooms, lounges | porous body of `upholstery` (2.4.2) | 3 | as 3.8 (per seat or per floor area only) | per-object area of sofas and armchairs (request 1) | high | wish |
| 4.2 | bed | bedrooms | porous body of `upholstery` (the mattress) on a hard frame box | 3 | none | per-object area (request 1) | medium | wish |
| 4.3 | books in a bookshelf | living rooms, studies, libraries | hard box `box_bookshelf` (2.2) | 3 | Odeon 104 (scattering only, no absorption) | absorption per object of a filled shelf (request 9) | medium | wish |
| 4.4 | rack of clothing | open wardrobes, shops, backstage | porous body, a box of a light fiber row around the hanging clothes **(proposed)** | 3 | none (query "clothing", "garment" finds nothing) | per-object area (request 13) | low | wish |
| 4.5 | plants indoors | potted plants, green walls | potted plant: leave out if small, else a porous body of the foliage's outline **(proposed)**; green wall: a porous sheet on the wall | 3 (potted), 1 (green wall) | green walls: Thomazelli 677–680, 1953 (Sabine); Davis 1951 (Sabine, single-number only) | per-object area of potted plants (request 13) | low | wish |
| 4.6 | `clutter_hard_sparse`, `clutter_hard_medium`, `clutter_hard_dense`, `clutter_soft_sparse`, `clutter_soft_medium`, `clutter_soft_dense` | items on desks and shelves, cluttered corners | clutter box (2.4.3) | 3 | per object of furniture groups only: Nilsson 262, 264 (include scattering) | measured absorption of cluttered desks and shelves (request 10) | medium | wish |
| 4.7 | single person | a listener, a performer | porous body of a body's low-poly shape **(proposed)** | 1 | per person: Kath and Kuhl 1263 (seated, isolated), 1264 (standing); Nishidate 1260 (design value), 1265, 1266 (in situ); Chagok 1262 not usable | 63 Hz | medium | wish |
| 4.8 | dense audience | concerts, lectures | porous body: a box over the seating area of the height of seated persons, fitted to the absorption per seating area **(proposed)** | 1 | per area: Kath and Kuhl 80, 81 (Sabine), 82, 83 (in situ); Bradley 1202, 1203 (occupied seats, Sabine); Carvalho 250, 252 (occupied pews); Kath and Kuhl 1937, 1938 (occupied upholstered chairs) | edge effects (Kath and Kuhl 77, 79: strip widths) | high (venues) | wish |
| 4.9 | dishes and glasses | kitchens, bars | leave out, or count them in a clutter box **(proposed)** | 3 | none | — | low | wish |
| 4.10 | unoccupied seating | auditorium seats, pews | porous body over the seating area, as 4.8 **(proposed)** | 1 | Kath and Kuhl 1939, 1940; PTB#484 1604; Carvalho 251 (pews, per object) | — | medium (venues) | wish |

Limits of the recipes: the body reflects specularly where the object scatters; a per-object area measured in a
diffuse field contains the edge and diffraction effects of the object in that field; per-object areas measured in
groups (Nilsson's classroom units) include scattering toward an absorbing ceiling and are room-dependent below
500 Hz (line 264 note); placement values (corners, floors) are no material property.

### 2.5 Class 5: acoustic treatment

| # | name | intended use | recipe | tier | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 | `glass_wool`, `glass_wool_50mm`, `glass_wool_100mm` | porous panels, cavity fill | sheet on a rigid solid; solid for a fill or a free-standing block | 1 | α: Dupont 647 (random incidence, sound-field synthesis), 645 (JCA prediction), 648 (Sabine, marked); σ 646 (13112 Pa·s/m²) | 31.5/63 Hz; 125 Hz only as a model | high | wish |
| 5.2 | `rock_wool`, `rock_wool_50mm` | porous panels, cavity fill | as 5.1 | 1 | α: Kosten 1770 (random incidence from the tube, locally reacting), Kuhl 1145; Sabine 1767, 1147 (marked) | σ not measured; 125 Hz random incidence | medium | wish |
| 5.3 | `pu_foam_50mm` | foam panels | sheet on a rigid solid | 1 | α: Dupont 1448 (sound-field synthesis), 1446 (prediction), 1449 (Sabine); σ 1447 (10605 Pa·s/m²) | 31.5/63 Hz | medium | wish |
| 5.4 | `melamine_foam_50mm` | studio foam | sheet on a rigid solid | 2 | α: BASF 1108 (Sabine, mounting not stated, marked); σ range Cox 1102 | mounting; σ of the product | medium | wish |
| 5.5 | `polyester_fiber_50mm` | PET felt panels | sheet on a rigid solid | 2 | σ: Garai 1385–1426; on-furniture PET felt Szelag 1215, 1935 (type not stated) | polyester model (A14) | medium | wish |
| 5.6 | porous panel with an air gap (no row) | panels stood off the wall | absorber sheet \| box of cavity air \| wall | 1 | compilation 513; practical 655; per object GIK 555; Sabine 554 (> 1, marked) | — | high | wish |
| 5.7 | bass trap (no row) | corner traps | porous body of the trap's shape (5.1–5.3) in the corner, as an asset **(proposed)** | 1 | per object: GIK 553, 557; Auralex 86; Sabine 1511 (> 1, marked) | build of the products | high | wish |
| 5.8 | `mdf_leaf_6mm` | panel absorber, empty or filled cavity | leaf sheet \| box of cavity air (\| fill solid) \| wall | 1 | α: Shearer 1120, 1121 (Sabine); f₀ 1115, 1116 | `rr` for room-size leaves; Sabine only | high | wish |
| 5.9 | `plywood_leaf_3mm` | panel absorber | leaf sheet \| box of cavity air \| wall | 1 | α: Sakagami 1314, 1315, 1316 (Sabine; 30/90/150 mm); f₀ 1317–1319 | figure readings of low confidence | medium | wish |
| 5.10 | `cpa_steel_1mm` | compound panel absorber (steel bonded to foam) | steel leaf sheet bonded to a fill solid \| wall | 1 | α: Fuchs 319, 318, 327, 328 (Sabine; values above one are placement) | a mass-law steel leaf with coincidence is not a wave-medium leaf | medium | wish |
| 5.11 | `steel_plate_2_5mm` | free-floating plate absorber (31.5 Hz) | leaf on a fiber-mat fill | 2 | α: Nell 549, 550 (Sabine, marked); f₀ 1302 | mat stiffness; placement moves 125 Hz from 0.41 to 0.64 | low | wish |
| 5.12 | `mpp_steel_0_5mm` | microperforated cassette ceiling | MPP leaf sheet (equivalent fluid) \| box of cavity air \| slab | 1 | α: Fuchs 1178, 1177 (Sabine), 1179, 1180 | 31.5/63 Hz; perforation ratio inconsistent | medium | wish |
| 5.13 | `mpp_foil_0_1mm` | MPP foil, sails | leaf \| box \| wall | 2 | α: Cox 1130, 1933, 1140 (type not stated) | coefficient type | low | wish |
| 5.14 | `ceiling_tile_25mm` | suspended absorbing ceiling | porous tile sheet \| box of cavity air (the plenum) \| slab | 2 | practical α: Ecophon 1908 (200 mm plenum), Nilsson 235–237; compilation: Cox 15, 16, Scheibler 1642 | random-incidence or Sabine α with build (request 4) | high (offices, venues) | wish |
| 5.15 | `perforated_gypsum_12_5mm` | perforated plasterboard ceilings | perforated leaf \| box \| slab | 2 | practical: Rigips 1252, 1250; PTB#316 1562 (Sabine, compilation) | full build with a typed curve (request 5) | medium | wish |
| 5.16 | slotted steel covers on a porous layer | slotted ceilings | not a plane layer | 3 | Fuchs 1823, 1822, 1824 (Sabine, figure) | recorded-miss candidate | low | wish |

Notes:

1. **Glass wool.** Starting columns: page 11.3 (also the rig's inline `WOOL`), Komatsu at 13112 Pa·s/m², `fRef`
   0.87403: `a` 1.1362, `b` −0.043932, `c` 0.0023911, `d` 1.2533, `e` 2.2495, `f` −0.45515, `g` 0.088044,
   `h` 0.14944, `thk` 0.05. Transmission through a treated wall is over-insulated up to 8.1 dB at 31.5 Hz (stage-2b
   issue 1); room-side reflection is right.
2. **Rock wool.** The rig's inline `ROCKWOOL` is Miki at 32.8 kPa·s/m² (`fRef` 0.87403, `a` 1.31823, `b`
   −0.0125593, `c` 0.00481048, `d` 1.80573, `e` 0.823363, `f` 0.0954384, `g` 0.221513, `h` −0.0150261), a leaf-test
   fill, not fitted to Kosten's product. Kosten's curve is a locally reacting infinite layer; the library's layer
   reacts extendedly.
3. **PU foam.** Fitted-σ check: 10.3 against 10.6 kPa·s/m² (material-list analysis).
4. **Melamine.** Only a manufacturer curve of unstated mounting.
5. **Polyester.** Garai's model rows exist; model and chamber disagree for polyester mats (research handoff, A14).
6. **Air gap.** The gap is a box of cavity air glued between the absorber and the wall (modeling guidelines).
7. **Bass traps.** Corner placement values are no material property; the trap is checked with the object test
   (2.4) against its per-object area.
8. **Panel absorbers.** The loss ratio `rr`, the damping branch and the Sabine-to-random conversion for panels are
   decided with the first panel material. Starting columns: the rig's inline `MDF` leaf.
9. **Plywood leaf.** Three cavity depths of one leaf; confirms `rr` across depths.
10. **CPA.** The membrane constant with a bonded fill (60 or 50) is decided here.
11. **Nell plate.** The only absorber row at 31.5 Hz; placement values.
12. **MPP.** The stated open area 0.45 % does not match the stated grid (0.64 %).
13. **MPP foil.** Coefficient type not stated.
14. **Ceiling tile.** Practical coefficients (ISO 11654) are another quantity; no random-incidence curve with build
    is in the data.
15. **Perforated gypsum.** Waiting for a full build.
16. **Slotted panels.** Slots and excitable covers are not a plane layer; candidate for a recorded miss.

### 2.6 Class 6: outdoor

Ground and porous asphalt are porous media from a ground model at the measured flow resistivity (Miki's model, page
10.5). The research handoff (sections 3 and 6) gives ground flow resistivities with model tags only and calls
one-parameter models valid above about 200 Hz: below that the rows are a guess. Ground classes A–H (CNOSSOS-EU and
Harmonoise/Nord2000, lines 698–713) give one σ per class with a ground factor G; site measurements give spreads.

| # | name | intended use | recipe | tier | references in hand | gaps (request) | impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1 | `ground_snow` **(proposed name)** | snow cover | porous half-space: a thick traced solid | 2 | class A 12.5 kPa·s/m² (699; G 1, 698); Ishida 1831 (2–25 kPa·s/m², laboratory, porosity 0.6–0.9); Piercy 480 (new snow 15–30 kPa·s/m²); Bashir 869; FHWA 1925 (powder 10, granular 40 kPa·s/m²) | below 200 Hz a guess; snow depth (a layer on the ground) | high (winter sites) | wish |
| 6.2 | `ground_forest_floor` **(proposed name)** | forest floor, moss, heather | porous half-space | 2 | class B 31.5 kPa·s/m² (701; G 1, 700); Bashir 714–719, 721, 846–849; Nordtest 547, 1833; Piercy 548 | below 200 Hz | medium | wish |
| 6.3 | `ground_grass` **(proposed name)** | lawns, pasture, sports fields | porous half-space | 2 | class C 80 kPa·s/m² (702; G 1, 703), class D 200 kPa·s/m² (705; G 1, 704); Piercy 670 (150–300 kPa·s/m²); Bashir 722–727; Van Renterghem 671, 672; FHWA 666–668, 1925 (field grass 150, lawn 300 kPa·s/m²); Nordtest 1776 | below 200 Hz; which class is "grass" (4.9) | high | wish |
| 6.4 | `ground_soil` **(proposed name)** | compacted soil, earth | porous half-space | 2 | class E 500 kPa·s/m² (707; G 0.7, 706); Piercy 482 (rain-packed earth 4000–8000 kPa·s/m²); Bashir 720, 870; Nordtest 674, 675; FHWA 1844, 1925 (hard soil 5000 kPa·s/m²) | below 200 Hz | medium | wish |
| 6.5 | `ground_gravel` **(proposed name)** | gravel, gravel roads, car parks | porous half-space, or a gravel layer on soil | 2 | class F 2000 kPa·s/m² (709; G 0.3, 708); Bashir 834–843 (layers 0.04–0.22 m and semi-infinite), 871; Van Renterghem 673; Piercy 1214 | below 200 Hz | medium | wish |
| 6.6 | `asphalt` | dense asphalt, concrete paving | rigid face (mass column) with small absorption: a traced solid | 2 | class G 20 000 kPa·s/m² (711; G 0, 710), class H 200 000 kPa·s/m² (712; G 0, 713); FHWA 74 (new, 5000–15 000 kPa·s/m²), 75 (old, 25 000–30 000); Piercy 76 (> 20 000) | no absorption coefficient of asphalt found (queries over the whole file); the decision called it tier 1 for absorption | high | wish |
| 6.7 | `asphalt_porous_40mm` **(proposed thickness)** | porous (drainage) asphalt | thin porous sheet on the `asphalt` solid | 2 | Bashir 857 (67.5 kPa·s/m², porosity 0.18, tortuosity 2.36, layer 36 mm), 856 (2–15 kPa·s/m², non-acoustic) | absorption of porous asphalt (request 12) | medium | wish |
| 6.8 | `ice` | ice on ponds, rinks | rigid hard face (mass column), a traced solid | 2 | none found (queries "ice", "frozen") | — | low | wish |
| 6.9 | `vegetation` | foliage of trees, hedges, shrubs between source and receiver | absorbing volume: a lossy medium matched to air (no reflection), its `alpha` column (dB/m) and `alphaB` from the foliage excess attenuation | 1 **(proposed)** | ISO 9613-2 Annex A through Witte 546 (0.02 dB/m at 63 Hz to 0.12 dB/m at 8 kHz, paths 20–200 m), 545 (10–20 m, absolute dB); Witte 1949 (vegetation strip); Van Renterghem 456 (hedges, insertion loss) | — | medium | wish |

Notes:

1. **Ground model.** A porous half-space: a traced solid much thicker than the penetration depth, with Miki columns
   at the class σ; the face reflects and the body absorbs. Rows per site class, not per measured site; the site
   spreads set the tolerance.
2. **Snow.** A layer of finite depth on the ground matters at low frequency; the half-space form holds for deep snow
   **(proposed)**.
3. **Grass.** Classes C and D both carry grass examples; a single `ground_grass` row needs a choice (4.9).
4. **Gravel.** Bashir gives layers of known depth over a hard base; a gravel layer as a porous sheet on soil is the
   alternative **(proposed)**.
5. **Asphalt.** Dense asphalt is class G or H: a hard face whose small absorption follows from the ground model at
   its σ. The review decision called asphalt tier 1 for absorption, but no absorption row of asphalt was found in
   the research data (only flow resistivities and the ground factor), so it is listed as tier 2 here.
6. **Porous asphalt.** Bashir's slit-pore fit (line 857) gives σ, porosity, tortuosity and layer thickness: a
   thin porous sheet on a rigid base.
7. **Ice.** A hard face like concrete; nothing measured.
8. **Vegetation.** The air-like absorbing volume rejected for furniture is right here: foliage reflects little and
   attenuates along the path. The `alpha` column is the loss along the path in dB/m (page section 4), fitted as a power
   law to line 546. ISO 9613-2 values are typed catalog in the data.

### 2.7 Auxiliary rows

| name | use | recipe | tier | status |
| --- | --- | --- | --- | --- |
| `cavity_air` | the air of a box behind a leaf, a curtain, a panel, a floor on joists, a suspended ceiling | traced closed box; columns `a` 1, `e` 1.000001 (rig inline `AIRBOX`): a material identical to air is air | definition | wish |
| `water_20c` | water: ponds, pools; a half-space test of the mapping | traced solid; constants c 1482.36 m/s, ρ 998.207 kg/m³, Z 1.4797e6 Pa·s/m (lines 1959, 1957, 1958); in the rig as a sheet, since the stack rule refuses a lossless wave medium with a critical angle as a traced solid | 1 | wish |

## 3. Test references that are combinations (measured only)

Each combination is a rig test of the materials involved, room side first. "Whole" means the combination is measured
as a whole. References at non-round thicknesses are compared with the nearest round sheet.

| combination | materials tested | measured whole (line, type) | notes |
| --- | --- | --- | --- |
| carpet on concrete | `carpet_tile_11mm`, `carpet_tufted_12mm`, `concrete` | Salford 227, BCTC 226 (Sabine); Broda 1930 (normal incidence, rigid backing); compilations 1537, 1910, 973, 1981 | transmission touches stage-2b issue 1 |
| carpet with underlay on concrete | `carpet_*`, `underlay`, `concrete` | Broda 1931 (normal); compilations PTB#156–158 1538–1540, 974 | — |
| porous absorber on a wall | `glass_wool_50mm`, `pu_foam_50mm`, `rock_wool_50mm` on `concrete` | Dupont 647, 1448 (random incidence); Kosten 1770 (from the tube) | rig case exists for glass wool on 152 mm concrete |
| porous absorber with an air gap | wool \| `cavity_air` \| wall | compilation 513; practical 655 | — |
| panel absorber over a cavity | `mdf_leaf_6mm`, `plywood_leaf_3mm`, `cavity_air`, fill | Shearer 1120, 1121; Sakagami 1314–1316 (Sabine) | — |
| curtain in front of a wall | `curtain_thin_0_2mm`, `curtain_heavy_0_8mm`, `cavity_air` | DelRey 428, 430 (flat); Zimmer+Rohde 412, 410; ADO 406, 408 (pleated) | — |
| drywall, precise recipe, empty | `gypsum_leaf_12_5mm`, `cavity_air` | Halliwell 948 (R); Fuchs 2005 478 (R, 12.5 / 100 / 12.5 mm) | the same walls test the lumped `drywall_*` rows |
| drywall, precise recipe, filled | `gypsum_leaf_12_5mm`, `cavity_air`, `glass_wool` | Halliwell 937, 1049, 1050, 940, 943, 946 (R) | studs not modeled |
| plasterboard on frame, room side | `gypsum_leaf_12_5mm`, `cavity_air` | PTB#341 1568, PTB#342 1569 (Sabine, compilation); Bradley 923 (model of chamber data) | — |
| lining on a masonry wall | `gypsum_leaf_12_5mm`, `cavity_air`, fill, `masonry_hollow` | Zeitler 1079 (ΔR) with base wall 352; BIA 1797 against bare 1795 | issue 1 for the fill |
| plaster on masonry | `plaster_15mm`, `masonry_heavy`, `masonry_hollow`, `masonry_light` | BIA 265 (92 mm + 13 mm, one face); matched pairs 1795 / 1796, 1905 / 1904; Hopkins 448, 447 (dense block plastered both faces / one face), 24 (aircrete plastered one side); Warnock 355, 357 (plastered) | — |
| parquet on concrete | `parquet_15mm`, `concrete` | PTB#394 1581 (compilation) | marked only |
| carpet on timber | `carpet_tile_11mm`, `plywood_19mm` or `board_leaf_22mm` | none in the data (request 6) | plausibility fixture until measured |

## 4. Open design questions for phase 3

1. **The generic solid across thickness (decided approach, size open).** Phase 3 fits one generic set of columns
   across the measured thicknesses (concrete 152 and 203 mm; the masonry classes' walls) and measures the miss from
   100 to 600 mm. The mass term scales with the traced thickness (page 5.4), but `coiF`, `resF`, `att` are fixed
   columns: the fitted 152 mm row puts coincidence near 6.9 kHz, while fc·h of 17.1 Hz·m gives 171 Hz at 100 mm and
   28.5 Hz at 600 mm. If the miss is too large, a thickness-scaled coincidence goes to the stage-2b list. Affects
   concrete, the three masonry classes, screed, CLT.
2. **`side` (decided rule, values open).** A row value at a typical size, no engine change, drift accepted where a
   drawn panel differs. Open: the typical sizes proposed per entry, and the size of the drift a modeler should
   expect (page 11.2: a 4 mm pane at 3.29 m instead of 1.349 m transmits 1.7 dB more at 125 Hz).
3. **Objects (decided recipes, test open).** Hard box, porous body and clutter box (2.4). Open: the object test in the
   rig; which porous columns are fitted for a soft body and for clutter (σ, or σ and a loss column; the effective
   density and speed of hard clutter); whether the per-seat and per-area values are enough for sofas until per-object
   data arrive; the kind proposed for plants, clothes racks, persons and audiences.
4. **Reference row per class.** Glass (EN 12758 generic against Quirt), carpets (Cox minimum / mean / maximum),
   concrete (Litvin against Mahn), hollow masonry (Zeitler against Warnock): which reference a row is asserted
   against; for the masonry classes, the tolerance from the spread of their walls.
5. **Free porous bodies.** A cushion or a bass trap as a traced porous solid in air has no reference as a free body;
   the object test and the per-object data (GIK, Auralex) are the only checks.
6. **Lumped rows against recipes for building assemblies.** Drywall now has lumped rows by visible thickness and
   fill, with the leaf | box | leaf recipe as the precise option. Still open for stud partitions of other builds
   and for floating floors and screeds: one sheet fitted to the measured wall or floor, or the recipe. The recipe
   keeps the room-side absorption and the cavity resonance; the lumped row keeps the measured transmission including
   the studs; mass-law bodies do not resonate with a fill.
7. **A solid drywall row at other drawn thicknesses.** The mass of a stud wall is mostly its boards, not proportional
   to its thickness, while the mass term of a traced solid scales with the traced thickness (page 5.4). A `drywall`
   solid drawn at 100 and at 150 mm would read different masses; phase 3 measures the miss and decides whether the
   solid rows hold or only the sheet rows are offered.
8. **Masonry classes.** Whether three classes cover what a modeler sees, and whether sand-lime (not in the data)
   belongs to `masonry_heavy` without a reference.
9. **Outdoor ground below 200 Hz and class choice.** One-parameter ground models are not valid below about 200 Hz;
   which class each ground row takes (grass between C and D), and whether a layer form is needed for snow and
   gravel.

## 5. Procedure for adding a material later

1. **Who decides.** The maintainer decides whether a material enters the list; anyone may propose it.
2. **Fields of a new entry.** Name (the naming rules of 1.6: no suffix for a solid, `_<NN>mm` last for a sheet),
   class, intended use, recipe (with the typical `side` for a mass-law row), evidence tier, references in hand
   (research key, line, coefficient type), gaps, impact, status `wish`.
3. **References.** Search the research library first (`data/material_properties.csv`, read-only). If nothing usable
   exists, write a draft research request in section 6 (column or class, quantity and coefficient type, bands,
   whether closed sources are acceptable, whether the work can proceed without it). The maintainer passes it to the
   research manager; the answer comes back as data rows with source, location and coefficient type.
4. **Phase-3 pickup.** The material enters the work order by tier and impact. One worker builds its row and its
   material case in `tests/catch2_tests/test_acoustic_materials.cpp` (the row as a traced solid and as a sheet, the
   twin assertion, the independent model, the measured references; an object row with the object test); the
   maintainer reviews and commits; the status moves to `confirmed`, or to `recorded miss` with the issue in
   `md/acoustic_engine_stage2b_issues.md`. Object rows also get their Blender asset.
5. **CSV traps.** An empty cell reads 0, not the column default; a text cell in a numeric column is an error; a
   material file replaces the built-in table.

## 6. Draft research requests, ordered by impact

1. **Sofas, armchairs, beds** (classes 3 and 4; high). Per-object equivalent absorption area, ISO 354, 63 Hz–4 kHz,
   with the object's size and upholstery. Manufacturer reports (tier B) acceptable. Blocks 3.8, 4.1, 4.2 above
   tier 3.
2. **Wooden floor on joists** (class 1; high). Measured absorption with build (board thickness, joist depth, cavity
   fill), Sabine or random incidence, 63 Hz–4 kHz; laboratory R of the bare floor if available. Closed sources
   acceptable. Not blocking.
3. **Doors** (class 1; medium). Laboratory R (ISO 10140 or ASTM E90) of a solid timber door with build, size and
   seals, 63 Hz–4 kHz. Open sources first. Blocks the transmission of `door_solid_wood_40mm`.
4. **Suspended absorbing ceilings with build** (class 5; high in venues). Sabine or random-incidence α of
   mineral-fiber or glass-wool ceiling tiles with tile thickness, density or σ, and plenum depth, 63 Hz–4 kHz.
   Today only practical coefficients (another quantity).
5. **Perforated and slotted panels with full build** (class 5; medium). Sabine or random-incidence α with hole or
   slot geometry, open area, plate thickness, cavity depth and fill, sample size, 125 Hz–4 kHz (63 Hz if possible).
   Tier A first. Blocks 5.15, 5.16.
6. **Carpet on a timber floor** (class 3; medium). Absorption of a carpet of stated build laid on boards on joists,
   and the same carpet on concrete; flow resistivity of carpet piles with build; 63 Hz–4 kHz. Closed sources
   acceptable at moderate cost. Not blocking.
7. **Flow resistivity of carpet underlays** (class 3; medium). Felt, foam and rubber underlays with thickness and
   density. Not blocking.
8. **Floating laminate floors** (class 1; medium). Absorption of laminate on foam underlay over concrete, Sabine or
   random incidence, with build, 63 Hz–4 kHz. Not blocking (tier 2 by physics).
9. **Filled bookshelves** (classes 2 and 4; medium). Absorption per object of a shelf filled with books, ISO 354,
   125 Hz–4 kHz, with the shelf's size. Blocks `box_bookshelf` above tier 3.
10. **Cluttered desks and shelves** (class 4; medium). Absorption per object or per area of a desk or shelf with
    typical items (hard and soft), ISO 354, 125 Hz–4 kHz, with a description of the fill. Blocks the clutter rows
    above tier 3.
11. **Masonry walls missing in the data** (class 1; medium). Laboratory R of a plain aircrete wall and of a
    sand-lime wall with density and thickness; clay brick plastered on both faces; 63 Hz–4 kHz. Tier A search. Not
    blocking (Hopkins' dense block plastered both faces, line 448, and aircrete plastered one side, line 24, stand
    in).
12. **Absorption of asphalt** (class 6; medium). Absorption coefficient of dense and porous asphalt (in situ or
    tube, with build), 125 Hz–4 kHz. Not blocking (ground model at the measured σ).
13. **Potted plants, clothing racks, full wardrobes** (classes 2 and 4; low). Per-object absorption area, ISO 354.
    Not blocking.

(The numbering in section 2 refers to these requests by number.)
