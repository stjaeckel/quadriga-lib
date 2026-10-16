# M5 and A21 — report

Brief: `md/m5_a21_brief.md`. One section per phase. Scratch rig: `build_linux/scratch/m5` (`README.txt` lists every
file); scratch is not kept, so every number used below is stated here.

## 0. First step (2026-10-15)

`git status` clean at 04b92bd (the M9 work is committed). `cmake --build build_linux --parallel`, then the three runs of
`md/wave.md` 0:

| run | result | expected |
| --- | --- | --- |
| `[wave]` | 165110 assertions in 179 cases, all passed | 165110 / 179 |
| `[acoustic]` | 36705 in 51, all passed | 36705 / 51 |
| whole suite | 1311364 in 1137, all passed (58.6 s wall) | 1311364 / 1137 |

`src/ray_mesh_interact.cpp`: 5082 lines. The plain what-if copy (`plain.cpp`, `md/wave.md` 10) reproduces all three
baselines; only the random-seed line differs.

## 1. Phase 0 — analysis (2026-10-15)

No edits outside scratch; this report is the only new file in the repository.

### 1.1 M5, item 1: the new law as a what-if

The switch copy changes one line in `Material::tf`:

    x = tfQ * (fGHz / tfF - tfF / fGHz)        (today: x = tfQ * (fGHz - tfF) / tfF)

inside the existing `if (tfP != 0.0 && tfF > 0.0)`, so the peak still acts for `tfF > 0` only and the clamp to
[−1, 1] is applied afterwards, unchanged. Every public entry point rejects a frequency ≤ 0 (`center_frequency <= 0`
throws in `ray_mesh_interact`, `ray_state_update`, `medium_gain`, `interface_gain`, `refractive_index`, `ray_progress`),
so `tfF / fGHz` is finite. Edge cases measured through the library (sheet, 45°, `edge.out`), at 0.01 Hz, 1 Hz,
150 Hz, 20 kHz, 1 MHz:

| setting | old law | new law |
| --- | --- | --- |
| `tfP` 0.25, `tfQ` 0 (a flat lift) | 0.30 everywhere | 0.30 everywhere (no NaN at 0.01 Hz) |
| `tfP` 1.2 at 150 Hz, `tfQ` 0.75 | 0.818 / 0.822 / 1.0 (clamp) / 0.050 / 0.05 | 0.05 / 0.050 / 1.0 (clamp) / 0.050 / 0.05 |
| `tfF` 0 | 0.05 (no peak) | 0.05 (no peak) |
| `tfP` −0.3 at 150 Hz | −0 (forced) at the low end | 0.05 at the low end, forced only near the peak |

The old law's floor `tfP/(1 + tfQ²)` as `f → 0` is visible in row 2 (0.818 at 0.01 Hz); the new one goes to the
power law at both ends.

Whole suite on the switch copy, against the baselines:

| run | switch off | new law |
| --- | --- | --- |
| `[wave]` | — | 165110 / 179 passed, **0 rows moved** |
| `[acoustic]` | — | 84 assertions fail in 2 cases |
| whole suite | identical to the baseline | 84 assertions fail in the same 2 cases, nothing else moves |

The two cases are the expected ones: *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side
absorption of a mass-law face* (44 failing: α max 0.116 off, T max 2.04 dB) and *ray_progress - the plate rule and the
absorption peak at every frequency of a band* (40 failing: α max 0.116, T max 2.22 dB). Printed rows: 64 move, 32 in
each case's table (every row except 150 Hz, where both laws give `tfP`); no other row of the suite moves.

### 1.2 M5, item 2: the A10 rows refitted, and the low bands

**The fit.** The rows and the procedure of report 3.8 (rig `a1/p3/ab/a10fit.py`): six peaked rows, octaves 31.5 Hz to
8 kHz, each octave the mean of its three thirds, minimax in α over the bands of `absfit.targets` (31.5 / 63 Hz a band
between the flat continuation of 125 Hz and the round-1 model; 8 kHz measured or flat), five knobs
(`tf(1 kHz)`, `tfB`, `tfP`, `tfF`, `tfQ`). Nelder–Mead from 150 starts per row and law (40 starts gave the same maxima
within 0.0001). The old law is refitted with the same rig to check it: it reproduces report 3.8 or does slightly
better.

| row | old law (report 3.8) | old law, refit here | **new law** | new-law knobs: `tfP`, `tfF` (Hz), `tfQ` = `Q_α` |
| --- | --- | --- | --- | --- |
| window glass (MHA) | 0.017 | 0.0170 | **0.0133** | 0.140, 131.7, 0.364 |
| plasterboard on frame, 100 mm | 0.012 | 0.0119 | **0.0170** | 0.395, 91.9, 4.03 |
| double glazing | 0.004 | 0.0037 | **0.0075** | 0.302, 93.1, 5.40 |
| plasterboard on 25 mm battens | 0.005 | 0.0055 | **0.0056** | 0.402, 181.1, 1.18 |
| wooden lining | 0.013 | 0.0116 | **0.0100** | 0.143, 130.2, 0.107 |
| cork board on battens | 0.076 | 0.0696 | **0.0470** | 0.445, 493.3, 0.904 |

Every row stays within today's worst lining / pane residual (0.017); cork board drops from 0.070–0.076 to 0.047, inside
the ±0.05 guideline. Two rows lose a little (plasterboard on frame +0.005, double glazing +0.004), both far inside the
guideline. The old law's best fits for window glass and the wooden lining use its low-frequency floor as a flat lift
(`tfF` 18 / 22 Hz with `tfQ` 0.05 / 0.025, peaks below the band); the new law puts every peak inside the band (92 to
493 Hz).

**Paris against field, through the library.** Each fitted row was run through `ray_progress` (band probe with the
three peak columns, `bp5`; two-parameter concrete with `m` 20 as the rigid face, as a sheet and as a 200 mm traced
solid; tracer 4; 27 third-octave frequencies; 24-point Gauss–Legendre in θ with weight `sin 2θ`):

- Paris 0°–90° and field 0°–78° give the **same residual as the formula fit, row for row** (the table above): the
  library's octave values are within 1.0e-6 of the formula (largest |library − formula| 6.6e-7 Paris, 1.04e-6 field),
  sheet and traced solid identical. The face absorbs `tf` at every angle (`md/acoustic.md` 2.6); the open region
  below the critical angle (about 4.9° for this concrete) adds at most 1.8e-4 there and almost no weight.
- So the comparison `md/acoustic.md` 9 asks for (an absorption reference against the Paris integral of the face)
  changes nothing for a `tf` face: the coefficient is flat in angle, and Paris = field = `tf`. (A locally reacting
  peak whose Paris value falls below its `α₀` is M6, not M5.)

**The low bands of the fixture board** (`tf` 0.05, `tfP` 0.25 at 150 Hz; old law `tfQ` 1.5, new law 0.75), through
the library at 45°, sheet, single frequencies; the peak part is α − 0.05 (`board_low.out`):

| f (Hz) | α old | peak old | α new | peak new | octave mean of the peak, old / new |
| --- | --- | --- | --- | --- | --- |
| 31.5 | 0.1540 | 0.1040 | 0.0698 | 0.0198 | 0.1048 / 0.0212 |
| 63 | 0.1923 | 0.1423 | 0.1290 | 0.0790 | 0.1447 / 0.0838 |
| 125 | 0.2853 | 0.2353 | 0.2824 | 0.2324 | 0.2275 / 0.2196 |
| 250 | 0.1750 | 0.1250 | 0.2024 | 0.1524 | 0.1316 / 0.1550 |
| 500 | 0.0689 | 0.0189 | 0.0905 | 0.0405 | 0.0215 / 0.0434 |
| 1000 | 0.0534 | 0.0034 | 0.0600 | 0.0100 | 0.0038 / 0.0108 |

The peak parts at band centers are the numbers of `md/acoustic.md` 3.2 (0.104 / 0.020 at 31.5 Hz, 0.142 / 0.079 at
63 Hz, 0.235 / 0.232, 0.125 / 0.152, 0.019 / 0.040), now measured through the library.

### 1.3 M5, item 3: the restatement

Proposal: in both cases `tfQ` 0.75 (the board's `Q_α`; it keeps the near-peak width, since near the peak the old form
equals the new one for `tfQ_old = 2·Q_α`), the reference written from the new column definition
(`x = tfQ·(f/tfF − tfF/f)` in the case, not library code), every bound unchanged (α 1e-3, T 0.01 dB sheet / traced
solid, 0.02 dB and 1e-4 dB band against alone through `ray_progress`). Case names and statements stay; the comments
quoting the law and "Q 1.5" change.

Measured on scratch copies of the two test files linked in place of theirs (`rs_*.cpp`):

| library | `[acoustic]` | *Absorption peak* | *ray_progress - the plate rule and the absorption peak …* |
| --- | --- | --- | --- |
| new law | 36705 / 51, all passed | passes | passes; |α − ref| 0, |T − rule| ≤ 0.001 dB, band − alone 0 |
| today | 98 assertions fail in the 2 cases | 52 fail: α 32 (max 0.131), T 20 (max 1.23 dB) | 46 fail: α 32 (max 0.131), T 14 (max 1.37 dB) |

Durations (`--durations yes`, load 1.3): *Absorption peak* 0.001 s, the `ray_progress` case 0.004 s. Against the
baseline the restated run moves the same 64 printed rows as 1.1 (32 per case, 150 Hz unchanged) and nothing else.

### 1.4 A21, item 1: the configuration reproduced

Rebuilt from the brief in a probe TU that includes `test_acoustic_leaf.cpp` (`p21.cpp`, `md/wave.md` 11): faces
x = 0 (box entry), 0.010 (box exit, flipped), 0.010 (porous sheet), 0.010 (wall entry), 0.6 (wall exit, flipped), the
leaf at 0 where present; the porous sheet is the rock wool of the fill case (Miki 32.8 kPa·s/m², 15 mm); the wall
two-parameter concrete, `m` 0 or `m` 20 with `fRef` 5.0212e-5, `tf` 0, the wall medium as the sink. Reference:
[leaf] | air 10 mm | porous 15 mm on the concrete half-space (`lw::stack`).

**A trap first.** `lw::walk` collects a path wherever the state says air. After the triple face the state says air
(below), so `lw::walk` books the wave at x = 10 mm as transmitted (T 0.37) and never follows it into the wall: it
shows a lossy, wrong R and **hides** the energy creation. The M9 walker `w4` collected air only outside the structure.
The probe adds `walk_out`, `lw::walk` with that rule (a ray in air is collected only at x ≤ 0 running −x or x ≥ 0.6
running +x). Re-running M9's own `b4v.py` reproduces its F2 numbers (R 48.43 at 125 Hz, 60°, `m` 0; 0.10158 for 0.99373
on the mass-law wall; with 25.1 kPa·s/m² columns).

`walk_out`, scalar type 4 (type 5 identical in every row to the last digit), summed (`eps` 0) and followed (`eps`
1.5); R, T as power; reference R:

| leaf | wall | mode | 125 Hz 0° | 125 Hz 60° | 1 kHz 0° | 1 kHz 60° |
| --- | --- | --- | --- | --- | --- | --- |
| — | `m` 0 | summed | **6.947** | **42.18** | 0.144 | 0.292 |
| — | `m` 0 | followed | 0.140 | 0.085 | 0.073 | 0.122 |
| — | `m` 20 | both | 0.154 | 0.060 | 0.133 | 0.065 |
| — | reference | | 0.9936 | 0.9825 | 0.7453 | 0.5131 |
| leaf | `m` 0 | summed | **2.110** | **7.814** | **1.006** | **1.024** |
| leaf | `m` 20 | summed | 0.927 | 0.842 | 0.9938 | 0.9873 |
| leaf | reference | | 0.4287 | 0.2811 | 0.9939 | 0.9875 |

T is below 0.004 in every row. Energy is created on the `m = 0` wall (wave media only, as the brief says), summed
mode, with and without the leaf; on the mass-law wall nothing is created but R is 0.15 for 0.99 (85 % lost). With a
mass-law body in front of the sheet (concrete with `tf` 0.1 as the near body) R + T reaches 1.075 on the `m = 0` wall
(`energy_w0.out`).

**Through `ray_progress`** (band probe, plain library, tracer 4, band 125 Hz + 1 kHz): at the default threshold 0.15
nothing is created (no leaf: R 0.159 / 0.063 at 125 Hz 0° / 60° against 0.994 / 0.983); with the threshold at 0 the
same. **With the threshold at 1.0, leaf on the `m = 0` wall, the run did not finish in 300 s and held 12.7 GB of
memory** (a runaway tree; killed by its timeout). A fixture must not run this against today's library.

### 1.5 A21, item 2: which branch the event takes

**The intersector reports two faces.** `ray_triangle_intersect` returns FBS, SBS and a total count; three faces at one
x are a tie broken by mesh index. `ray_mesh_interact` and `ray_state_update` see the two reported faces only;
`ray_state_update`'s M1 / M2 are the caller's material words of `fbs_ind` / `sbs_ind`. The source says so in
`SHEET_EVENT`: "a sheet between two solids is not a geometry the intersector reports; if one occurs the next exit
corrects the current medium".

**Marker build** (`mk.cpp`: a counter at the sheet classification of both functions, and a brute-force scan of the mesh
for a third face co-located with a co-located pair, the FBS / SBS face's own second triangle excluded). In the brief's
mesh order (box exit, sheet, wall entry) every variant takes the same branch, transmit pass, at x = 10 mm:

| near body | sheet | far body | types | hit code → resolved | words after |
| --- | --- | --- | --- | --- | --- |
| air box / mass-law body (`tf` 0.1) | porous / porous with `m` 20 | `m` 0 / `m` 20 | 1, 2 (EM), 4, 5 | 69 → 71: **glued exit, box \| sheet \| air** | prev = sheet, cur = 0 (air), buf 0 |

The wall's entry face is the unreported third: the ray is relaunched past all three and travels "in air" inside the
wall; the far body never enters the event (`m` 0 and `m` 20 give bitwise the same event). Coming back from the wall
side the pair reads as a glued entry air | sheet | box (71) with cur = 0. Without the sheet (box | wall) the event is
the ordinary media-to-media transition (5 → 13).

**The order of the three faces decides the physics** (`walk_out`, no leaf, both walls, summed and followed, 125 Hz /
1 kHz, 0° / 60°):

| mesh order of the tie | pair reported | what the library computes | R at 125 Hz 0° / 60°, 1 kHz 0° / 60° |
| --- | --- | --- | --- |
| box, sheet, wall / sheet, box, wall | box exit + sheet | glued exit into air | 6.95 / 42.2 / 0.144 / 0.292 (`m` 0), 0.154 / 0.060 / 0.133 / 0.065 (`m` 20) |
| box, wall, sheet / wall, box, sheet | box exit + wall entry | body \| body, the sheet ignored | 0.9998 / 1.0000 / 0.9998 / 1.0000 |
| sheet, wall, box / wall, sheet, box | sheet + wall entry | glued entry from the host box (code 79) | **0.993591 / 0.982525 / 0.745250 / 0.513055 — the reference to 6 digits** |

So where the intersector happens to report the sheet and the far body's face, today's solver resolves body | sheet |
body exactly (the host in the current word is the near medium).

**The whole suite on the marker build** (all 1137 cases pass, rows unchanged): one case meets a third co-located face,
*One solver - a sheet and a stack are one sum and no mirror is clamped* (`[wave]`), 48 events, table B: a traced
solid of 0.2 mm (and 1 mm) inside the `eps 5.24` host, i.e. host exit, solid entry and solid exit within the
co-location distance, **no sheet** (the documented degenerate "solid thinner than `colocation_dist`"). No existing case
meets body | sheet | body. A detection gated on a sheet among the three moves no row.

### 1.6 A21, item 3: the options, measured

Two kills were built as what-ifs in the switch copy, besides (b) and (c):

- **(a) scan.** At every co-located pair `ray_mesh_interact` scans the mesh for a further co-located face (point-in-plane
  within `colocation_dist`, inside the triangle, the pair's own second triangles excluded); if the faces include a sheet
  and two bodies it sets **bit 7 of the hit code (unused today)** and zeroes its own output; `ray_state_update` kills
  (resolved type 0) on bit 7. This is the brief's (a): the event is detected where it happens.
- **(a′) state.** No scan. `ray_state_update` kills where a ray that a sheet crossing put in air (current word 0,
  previous word a sheet) meets a body from inside (an exit-class hit code: 1, 5, 7, 13, 21, 29, plus the total-reflection
  variants). That is the first event after the triple face in the orders that create energy.

| | (a) scan | (a′) state | (b) resolve as glued on both sides | (c) record only |
| --- | --- | --- | --- | --- |
| lines in `ray_mesh_interact.cpp` | about 63 (scan 50, bit 7 write 10, kill 3) plus the doc of bit 7 | 4 | not buildable in this file (below) | 0 |
| rows moved, whole suite | 0 (1137 pass) | 0 (1137 pass) | — | 0 |
| `[wave]` rows | 0 | 0 | — | 0 |
| suite cases that reach it | 0 | 0 (marker `A2KILL`: only the A21 probes) | — | — |
| energy after it, 8 variants × 2 tracers × 2 modes × 2 f × 2 θ | none: max R + T 0.900 (mass-law body in front) | none: max R + T 0.901 | — | R up to 42.2, R + T 1.075 with a mass-law body |
| energy over the six face orders | none (every order killed, R 0) | none: max R + T 1.000001 (rounding at the bare face) | — | R up to 42.2 |
| `ray_progress`, threshold 0 / 0.15 / 1.0, leaf or not, both walls | all complete; no leaf R 0, leaf R 0.942 / 0.869 (the leaf's own reflection) | all complete; R 0.154 / 0.060 (no leaf), 0.927 / 0.842 (leaf) | — | runaway at 1.0 with the leaf on `m` 0 |
| what the room sees at the event | nothing passes the triple face; the ray is killed there in every order, also in the two orders that are exact today | box \| sheet \| air reflect port (R 0.154 for 0.994) or the sheet ignored (R 1.0 for 0.745), by mesh order; exact in the two good orders | — | as today |
| cost | **≈ 8 ns per mesh face per co-located event**: 1000 rays at a legitimate glued exit, 1.6 ms per call without the scan, 9.7 / 83 / 795 ms with it at 1e3 / 1e4 / 1e5 faces (one thread, load 2.7–3.0, `scan_cost.out`) | none (one test per event on words already decoded) | — | — |

Further notes:

- **(a)'s scan has its own trap**: on a legitimate glued exit it flagged 1 ray in 1000, a hit on a quad's diagonal where
  FBS and SBS are the box face's own two triangles and the sheet counts as the third. A production detection would have
  to merge a face's triangles first. Its cost is the main objection: every lined wall in a scene has co-located pairs,
  and the scan is linear in the mesh, where the intersector itself is accelerated by sub-mesh bounding boxes that
  `ray_mesh_interact` does not receive.
- **(b) is not buildable inside `ray_mesh_interact.cpp`.** The resolve the brief describes exists already (the third row
  of the order table is exact), but `ray_state_update` only learns the materials of the two reported faces from its
  caller. In the order box exit + sheet it never learns the far body; in box exit + wall entry it never learns the
  sheet. Even with (a)'s scan, `ray_mesh_interact` has no channel to hand a third material over (the hit code has one
  free bit; the state words have none). It needs the caller (`ray_progress`, the intersector) to report the third face:
  outside this brief's rule.
- **(a′) is a wider rule than A21.** Any ray in air whose previous-medium word is a sheet and which meets a body from
  inside is killed, in every type. Today such a ray is corrected (`md/wave.md` 8: "the current-medium word is corrected
  in one direction only (air → medium at the next exit, with the fix-flag)"). Besides A21 this covers a lined body
  nested in another medium (not a use case, `md/wave.md` 8) and a scene with open or flipped faces met after a sheet.
  No suite case reaches it, EM included; the risk for EM scenes with such geometry is not measured. A gate on the
  scalar types would leave EM exactly as today (and A21 in EM unhandled).
- (a′) does not touch the event at the triple face: in the orders box exit + sheet the reflect port is still that of
  box | sheet | air (R 0.154 for 0.994, energy lost, silent), in the orders box exit + wall entry the sheet is still
  ignored. Both lose energy; neither creates it.

### 1.7 A21, item 4: recommendation

**(a′) together with (c)**: the four-line kill in `ray_state_update`, and the remaining limitation recorded. Reasons:

- It removes everything that creates energy or runs away, measured over 8 variants (near body air / mass-law, sheet
  with and without `m`, far wall `m` 0 / `m` 20), both tracers, summed and followed, all six face orders, and through
  `ray_progress` at three thresholds; it moves no row of the suite and no `[wave]` row, and no existing case reaches it.
- It costs nothing at run time; (a) costs a mesh scan at every co-located event (795 ms against 1.6 ms per 1000 rays at
  1e5 faces) and about 63 lines, needs a face merge it does not have yet, and kills the two face orders that today's
  library resolves exactly.
- (b) cannot be built within this brief's rule; (c) alone leaves R 42.
- What stays wrong is bounded and one-sided: the room sees a reflection that is too small (the box | sheet | air port) or
  the bare body | body face, never more than it sent. Recorded with the rule of use that already works: a porous layer
  between a box and a wall is a traced body glued on the wall (*Leaf - a porous fill at the back of the box is the
  transfer matrix of leaf, air, fill and wall*, within 5e-4 in R), not a sheet; a sheet is glued on one body, with air on
  its other side (`md/wave.md` 6).

The phase-1 fixture for (a′), as I would propose it: wave media only (air box, porous sheet, `m` 0 wall, the R 42
configuration) plus the mass-law-wall row, so `[wave]` if Stephan agrees, else `[acoustic]`; through the
outside-collecting walker (`walk_out` would go into the test file next to `lw::walk`, or `lw::walk` gains the two
bounds), summed and followed, tracers 4 and 5, 125 Hz / 1 kHz, 0° / 60°: assert R + T ≤ 1 + 1e-6 and that the
continuation into the wall is killed (resolved type 0 at the wall's exit face, one traced event); record the reflect
port against the reference, not assert it. Against today's library it fails on R + T (42.2) in a few milliseconds (7
events); it must not use `ray_progress` with the threshold at 1.0, which runs away today.

### 1.8 For Stephan's decision

1. **A21**: (a′) + (c) as recommended; or (a′) gated on the scalar types (EM unchanged); or (a) with its scan cost; or
   (c) only.
2. **The A21 fixture**: `[wave]` (the creating configuration is wave media only) or `[acoustic]`; a walker that collects
   only outside the structure in the test file (`lw::walk` with bounds, or a second walker).
3. **M5**: the fixture board's `tfQ` 0.75 and the restatement of 1.3 (bounds unchanged, names unchanged).

Line counts so far: `src/ray_mesh_interact.cpp` 5082, untouched. Expected after phase 2: M5 one line changed plus its
comment, (a′) four lines.

## 2. Phase 1 — fixtures (2026-10-16)

Stephan's decisions on phase 0: A21 by (a′) **gated to the scalar types (3, 4, 5)**, EM unchanged, plus (c), recorded
as open in `md/acoustic.md` 6 and as an accepted limitation in `md/wave.md` 8, EM included; the fixture `[acoustic]` in
`test_acoustic_leaf.cpp` as proposed in 1.7; `lw::walk` with the outside-only collection as the default if no M9 row
moves; M5 with `tfQ` 0.75 and the restatement of 1.3; in the documents the converged old-law cork residual 0.070 and the
new 0.047. The document records (including the sentence for `md/wave.md` 0) follow the brief's order and go into phase 3.

### 2.1 `lw::walk`: one walker, collecting air only outside the structure

`lw::walk` now collects a path in air only where it leaves the structure's x range (the mesh's smallest and largest x;
at the face, within 1e-6 m), running −x at the front or +x at the back, or where it has entered the sink medium (the
sink test now reads `sink != 0 && cur == sink`; before, `sink` 0 made it the air test again). A ray the state puts in air
inside that range is followed on. Checked first in scratch (a copy of the file linked in place of its object): all 51
`[acoustic]` cases pass and **no row moves** (every printed row of the five M9 cases bitwise identical, event counts
included). So it is the default, and the comment says why (A21). `cf::walk` in `ray_test_helpers.hpp` keeps the old
rule (not used here).

### 2.2 The changes

| file | change | lines |
| --- | --- | --- |
| `test_acoustic_state_update.cpp` | *Absorption peak - …*: `tfQ` 0.75, reference `x = tfQ·(f/tfF − tfF/f)`, comment (the law, `Q_α`, why 0.75); name and bounds unchanged | +8 / −6 |
| `test_acoustic_progress.cpp` | *ray_progress - the plate rule and the absorption peak at every frequency of a band*: the same; name and bounds unchanged | +7 / −7 |
| `test_acoustic_leaf.cpp` | `lw::walk` (2.1); new case *Sheet between two bodies - nothing is created and the wave left in air inside the far body is killed* | +80 / −5 |

**The new case.** Box of near-air material 0–10 mm, the fill case's rock-wool columns as a 15 mm sheet, the wall
(two-parameter concrete, `tf` 0, `m` 0 and `m` 20) from 10 mm to 0.6 m as the sink; faces at 10 mm in the order box
exit, sheet, wall entry. `lw::walk`, tracers 4 and 5, summed and followed, 125 Hz / 1 kHz, 0° / 60°. Asserted:
`R + T ≤ 1 + 1e-6` (32 rows); a three-event chain per wall, tracer and angle (box entry, the triple face, the wall's exit
face): the premise, hit code 69 (glued exit) at the triple face and the state in air after it; and at the wall's exit
face a hit (`nh > 0`) with resolved type 0 in both passes (the tracer's and type 3). Recorded, not asserted: R against
air 10 mm | porous 15 mm | the concrete half-space (`lw::stack`), the event count and the dropped power.

### 2.3 Against today's library

| run | result |
| --- | --- |
| `[wave]` | 165110 / 179, all passed, **0 rows moved** |
| `[acoustic]` | 36785 assertions in 52 cases, 118 fail in 3 cases |
| whole suite | 1311444 in 1138, 118 fail in the same 3 cases |

Failing assertions per case:

| case | failing | deviation |
| --- | --- | --- |
| *Absorption peak - …* | 52: α 32, T 20 | α up to 0.131, T up to 1.23 dB |
| *ray_progress - the plate rule and the absorption peak …* | 46: α 32, T 14 | α up to 0.131, T up to 1.37 dB |
| *Sheet between two bodies - …* | 20: R + T 4, kill 16 | R + T 6.95 and 42.19 (`m` 0, summed, 125 Hz, 0° / 60°, both tracers); the wall's exit face resolves 17 / 21 (transmit) and 153 / 157 (reflect pass): an exit with the fix flag, not a kill — every one of the 16 |

The premise holds everywhere (hit code 69 at the triple face, the state in air after it). The followed rows and the
mass-law wall pass the energy bound today (R 0.154 for 0.994: lost, not created); they guard the kill and the bound
after the change.

**Rows.** Against the baselines nothing moves but the restated rows and the new table: in `[acoustic]` and the whole
suite 64 rows change (32 of each restated case, every row except 150 Hz) and 34 lines are added (the new case's two
header lines and 32 rows); `[wave]` identical.

**Durations** (`--durations yes`, load 4–12 from other sessions): the new case 0.007 s, *Absorption peak* 0.001 s, the
`ray_progress` peak case 0.003 s; the five M9 cases with the new walker 0.001 / 0.110 / 0.026 / 0.022 / 0.106 s (as
before).

### 2.4 EM: does body | sheet | body create energy?

Measured on today's library (scratch probe `p1/p21em.cpp`): the same faces and columns, EM types 1 (transmit) and 0
(reflect), the materials as dielectrics (`m` 0 wall), an outside-only walker per channel (TE and TM of `xprmat`), summed
and followed, 125 Hz / 1 kHz equivalents, 0° / 60°, all six mesh orders of the tie. Largest R + T:

| pair the intersector reports (orders) | TE summed | TE followed | TM summed | TM followed |
| --- | --- | --- | --- | --- |
| box exit + sheet (BSW, SBW) | **42.19** | 0.140 | **14.46** | 0.205 |
| box exit + wall entry (BWS, WBS) | 1.000 | 1.000 | 1.000 | 1.000 |
| sheet + wall entry (SWB, WSB) | 0.994 | 0.994 | 0.997 | 0.997 |

**Yes: EM creates energy in the same two orders**, TE identical to the scalar type to every printed digit (the scalar
types run the TE form), TM up to 14.46 at 60°. In the sheet + wall-entry orders TE equals the fluid-layer reference
within 1e-6. The scalar gate decided for phase 2 leaves this in EM.

For your decision only, not built: the ungated (a′) of the phase-0 what-if removes it in EM as well (largest R + T in
the two bad orders 0.154 TE, 0.318 TM; the other orders unchanged), and moved no row of the suite (phase 0, 1.6). The
second open question, whether the (a′) condition occurs in a real EM scene (open or flipped faces met after a sheet),
is not measurable in the suite and stays open.

### 2.5 Next (phase 2, after the OK)

`Material::tf`: the one line and its comment. `ray_state_update`: (a′) gated to `is_scalar`, about four lines before the
reflection / transmission dispatch. Expected: the three cases pass, `[wave]` identical (tables and twins), `[acoustic]`
36785 / 52, whole suite 1311444 / 1138.

## 3. Phase 2 — source (2026-10-16)

Stephan's decision on phase 1: the kill applies to **every type**, EM included (EM creates energy too, 2.4). Two checks
in scratch came first (`build_linux/scratch/m5/p2`).

### 3.1 Check 1: can the kill be narrowed to "after a glued exit"?

It already is. The transmit path of `SHEET_EVENT` returns early, writing no state word, where the ray is in air on both
sides of the sheet (`mtl_current == 0 && m_next == 0`: an isolated sheet in air), so the previous-medium word keeps what
it had: 0 at launch, or the solid the ray last left through a bare face. The only writer of "current 0, previous a
sheet" is a glued exit (solid | sheet | air). The reflect pass writes the words only inside a medium, and every entry
from air writes the previous word anew. Measured (`p2/narrow.cpp`, EM type 1 and scalar type 4, 30°):

| chain | event | today | with the kill |
| --- | --- | --- | --- |
| isolated sheet (`eps 4`, 4 mm) at x = 0 in air, then a solid face at 0.1 m met from behind (an open mesh) | 2 | words after the sheet: prev 0, cur 0; the face resolves 49 (exit, fix flag) | the same, not killed |
| solid 0–50 mm with a sheet glued on its exit face, then the same face | 3 | words after the glued exit: prev = the sheet, cur 0; resolves 49 | killed (0) |

So the condition as measured is the narrowed one, and no separate build was made: the A21 energy checks below are run on
the built library instead. What the words cannot separate is a legitimate glued exit (wall | wallpaper | air) followed by
a face met from behind; that ray is killed as well.

### 3.2 Check 2: the repository scenes through `ray_progress`

`p2/smoke.cpp`: `tests/data/test_scene.obj` (48 faces) and `test_scene_pbr.obj` (180 faces), the default material table
(no `thk` column, so no sheet at all) and a copy with sheets (`itu_glass` 6 mm, `itu_plasterboard` 12.5 mm,
`itu_ceiling_board` 15 mm); EM at 2.4 GHz and scalar; an icosphere of 32000 rays from the scene's center and from 1 m
above it; `ray_progress` with up to 20 interactions until no ray is left (20 to 21 steps); today's library against the
kill build (the phase-0 what-if, ungated), and a marker build counting the condition. Per run a bitwise hash of every live
coefficient after every step.

| | 16 runs (2 scenes × 2 tables × EM / scalar × 2 sources) |
| --- | --- |
| kill condition met (today's library, marker) | **0** in every run |
| today against the kill build | **bitwise identical** in every run (hash of every coefficient after every step): no path lost; repeated with the built library of 3.3: 16 of 16 identical |
| sheet events in the sheet runs | 17.7 to 144.8 million isolated-sheet events per run, **no glued pair** (no sheet sits on a solid face in these scenes) |

So the kill never touches a window in an open mesh (tens of millions of isolated-sheet events, EM and scalar), but these
scenes cannot say how often a glued exit is followed by a face met from behind in a real scene: they have no glued
sheets. That question stays open.

### 3.3 The change

`src/ray_mesh_interact.cpp`, 5089 lines against 5082 (+10 / −3):

- `Material::tf`: `x = tfQ * (fGHz / tfF - tfF / fGHz)`; the comment names the lumped resonator, `tfQ` as `Q_α`, and
  that every caller's frequency is positive.
- `ray_state_update`, before the reflect / transmit dispatch: where the ray is in air with a sheet in the
  previous-medium word and the hit is an exit-class code (1, 5, 7, 13, 21, 29, with or without the total-reflection
  bit), `KILL_RAY()`; every type. Three comment lines say why (A21).

### 3.4 Whole suite

| run | result | against the baseline |
| --- | --- | --- |
| `[wave]` | 165110 / 179, all passed | **identical** (`test_wave.txt`); `tests/data/wave_twins.tsv` unchanged and asserted |
| `[acoustic]` | 36785 / 52, all passed | 64 rows of the two restated cases (32 each, every row except 150 Hz) and the new case's 34 lines |
| whole suite | 1311444 / 1138, all passed | the same rows, nothing else |

Against the phase-1 run (today's library) the same 64 rows move (they print the library's values, now on the new law)
and the 32 rows of the new table (the kill). Every moved row is one of these.

- The restated cases: α on the reference in every row, T within 0.001 dB (bounds 1e-3 and 0.01 / 0.02 dB), band against
  alone 0.
- The new case, every row: R + T ≤ 0.154 (`m` 0 and `m` 20, summed and followed, both tracers); 5 events summed, 6
  followed (before: 7, and 66 to 94 followed on the `m` 0 wall). R 0.154 / 0.060 / 0.133 / 0.065 against 0.994 / 0.983 /
  0.745 / 0.513 (125 Hz, 1 kHz; 0°, 60°): the box | sheet | air port, recorded.
- Durations (load 5 to 30 from other sessions): the new case 0.002 s, *Absorption peak* 0.001 s, the `ray_progress` peak
  case 0.003 s.

### 3.5 The A21 energy checks on the built library

Probes linked against the built `libquadriga.a` (`p21.cpp`, `p1/p21em.cpp`), the outside-only walker, summed and
followed, 125 Hz / 1 kHz, 0° / 60°. Largest R + T:

| check | before (today's library) | after |
| --- | --- | --- |
| scalar, 8 variants (air box / mass-law body in front, porous sheet with and without `m`, wall `m` 0 / `m` 20), tracers 4 and 5 | 42.19; 1.075 with the mass-law body | **0.902** |
| scalar, six face orders | 42.19 | **1.000001** (body \| body orders: the bare lossless face, R 0.99982 + T 0.00018, as before) |
| EM types 1 / 0, six face orders, TE / TM | 42.19 / 14.46 | **1.000000** (body \| body orders); bad orders 0.154 TE, 0.318 TM |
| EM, 8 variants, TE / TM | — | **0.902** |

The order-dependent physics stays as recorded: the box | sheet | air reflect port where the box exit and the sheet are
reported, the sheet ignored where the box exit and the wall entry are, exact where the sheet and the wall entry are.

### 3.6 Next (phase 3, after the OK)

`md/acoustic.md`: 2.6 (the law, `tfQ` as `Q_α`, the restated numbers), 3.2 (the peak-shape row expressible; cork 0.070 →
0.047, the old-law residual converged from report 3.8's 0.076), 4 (M5 built), 6 (A21 narrowed: energy creation and the
runaway closed for every type; the order dependence open; open whether the condition occurs in a real scene with glued
sheets and faces met from behind), 10 (built, dated), the appendix (the new case). `md/wave.md`: 0 (the sentence next to
the M9 one), 6 / 8 (the kill for all types; the order-dependent physics as an accepted limitation, EM included; the EM
measurement, R + T 42.19 TE / 14.46 TM before the kill), 11 (`lw::walk` collects only outside the structure).

## 4. The gap after phase 2: a glued exit out of the far body (2026-10-16)

Stephan's question: the kill covers the exit-class codes that start inside a plain body, not a glued exit (69 / 101) out
of the far body. Does A21 create energy after phase 2 where the wall carries a glued sheet on its back face, a sheet
stack there, or an abutting body?

Measured on the A21 probe (`p3/back.cpp`): box | porous sheet | wall (0.010 to 0.6 m), and at x = 0.6 m (a) the gypsum
leaf glued on the wall's back face, (b) a stack of the leaf and the porous sheet there, (c) a dielectric body (`eps 2`,
0.6 to 0.8 m) abutting it; wall `m` 0 and `m` 20; scalar types 4 and 5 (`lw::walk`), EM types 1 / 0 TE and TM; summed and
followed; 125 Hz / 1 kHz; 0° / 60°; the six orders of the triple face; on today's library and on the phase-2 build.

**No energy is created, before or after phase 2.** In the two orders where the wave is left in air inside the wall
(BSW, SBW) the largest R + T is 0.480 (scalar, TE) and 0.285 (TM) with a sheet or a stack on the back face, and 0.154 /
0.318 with the abutting body, the same on both libraries except the abutting body (0.160 / 0.334 before).

Why, from the traced chain (box entry, triple face, the face at 0.6 m; types 4 and 1, both passes):

| back of the wall | hit code at 0.6 m | resolved | what the library does |
| --- | --- | --- | --- |
| a glued sheet, or a sheet stack | 69 (glued exit; the stack reports the wall's exit and the leaf) | 71 transmit, 199 reflect | a glued exit taken with the current medium air: the composite is air \| sheet \| air, passive; not killed |
| an abutting body | 5 (media to media) | 0 | killed by the phase-2 condition |

So no extension of the condition was made (Stephan's rule "if no: say why and record it"): it is recorded in
`md/acoustic.md` 6 and `md/wave.md` 8 (the far body's back sheet met as air | sheet | air, nothing created). The
physics there is wrong (the wall taken as air, R + T 0.48 where the concrete reflects nearly everything) but bounded.

The only R + T above one in the probe, 1.000022 (sheet or stack behind) and 1.000177 (body behind), is in the orders
box exit + wall entry (the porous sheet ignored), followed mode, `m = 0` wall, 125 Hz, 0°: the followed series of the
lossless 0.59 m wall (12,000 to 32,000 events, dropped power 3e-5 to 1e-4). It reads the same without any porous
sheet and on today's library (summed: 0.999998 / 1.000000); not A21.

## 5. Phase 3 — documents (2026-10-16)

**`md/acoustic.md`:**
- 0: `[acoustic]` 36785 assertions in 52 cases.
- 2.6: the law, `tfQ` as `Q_α`, its relation to the old form, the board with `Q_α` 0.75 and its measured band values on
  both forms; "Asserted in" gains the `ray_progress` case.
- 2.7: the peak row: every lining and pane row within 0.017, cork board 0.047.
- 2.8 (limits): the porous sheet between box and wall, the mesh-order dependence, nothing created since the kill.
- 3.1 and 3.2: the peak-shape row expressible on the law, measured board values, the refit (cork 0.047; 0.070 on the old
  form, converged; report 3.8 gave 0.076); the height-and-width, MPP and gypsum rows point at M5 as built.
- 4: M5 acceptance with the converged 0.070, and "Built (2026-10-16)" with the refit.
- 6: A21 narrowed: what the intersector reports by order, the kill for every type, the numbers before (42.19 / 14.46,
  1.075, the runaway) and after (≤ 0.902), what is left (reflect port, sheet ignored, a back sheet as air | sheet | air),
  open: the order dependence (needs the third face, outside this file) and the occurrence of the condition in a real
  scene with glued sheets.
- 10: the decisions of 2026-10-15 / 16 on M5 and A21, and "Built (2026-10-16)" with the counts and 5089 lines.
- 11: `lw::walk` collects air only outside the structure. Appendix: the new case (2.8, 6), 3.2 on the two peak cases.

**`md/wave.md`:**
- 0: the counts (`[acoustic]` 36785 / 52, whole suite 1311444 / 1138; `[wave]` unchanged) and the sentence next to the M9
  one: the new case evaluates wave media and guards the kill for the scalar types, EM measured, not asserted.
- 6: the previous-medium word gains its reader: only a glued exit leaves the ray in air with a sheet in the word; meeting
  a body from inside in that state, the ray is killed, every type. "Asserted in" gains the new case.
- 8 (state, depth and directions): the accepted limitation, EM included: the tie of three faces and its three outcomes,
  the EM measurement before the kill (R + T 42.19 TE / 14.46 TM; scalar 42.19), what remains, the back-face sheet, the
  legitimate glued exit followed by a face met from behind (killed, not seen in the repository scenes), and what a full
  resolve needs.
- 11 (traps): `bf::walk` and `cf::walk` collect wherever the state says air; such a geometry needs `lw::walk`'s rule.

Not touched: the reports of earlier sessions (`md/m9_fixtures_report.md` keeps F2 as found); the brief.

**State at the end.** Uncommitted: `src/ray_mesh_interact.cpp` (5089 lines against 5082, +10 / −3), the three test
files (+95 / −18), `md/acoustic.md`, `md/wave.md`, this report. Last runs (phase 2, unchanged since): `[wave]` 165110 /
179 identical to `test_wave.txt`, twins unchanged; `[acoustic]` 36785 / 52; whole suite 1311444 / 1138. The baselines in
`test*.txt` are still those of 04b92bd; after the commit they are to be regenerated (`md/wave.md` 0).

## 6. EM rows in the A21 case (2026-10-16)

Stephan: the kill applies to every type because EM created energy, so the case asserts EM too.

- `lw::walk` takes EM types: a trailing argument `ch` picks the channel of `xprmat` (0 TE, 3 TM; the walk multiplies
  that channel along each path), and an EM type's reflect pass is type 0. Scalar calls take the same path as before:
  every scalar row of `[acoustic]` is unchanged, the five M9 cases included.
- *Sheet between two bodies - …* gains, on the `m = 0` wall (the dielectric configuration of 2.4), EM type 1 per
  channel, TE and TM, summed and followed, 125 Hz / 1 kHz, 0° / 60°: R + T ≤ 1 + 1e-6 (16 rows), and the kill chain with
  type 1 (premise hit code 69, the wall's exit face killed in passes 1 and 0). The mass-law wall stays scalar only (an
  EM row on a mass-law material is formal, `md/acoustic.md` 7).

**Against a what-if without the kill** (the built source with the condition switched off): 28 of 108 assertions fail.
R + T: scalar types 4 and 5 6.948 / 42.186 (125 Hz, 0° / 60°, summed), EM TE 6.948 / 42.186, **EM TM 6.948 / 14.458**;
the kill: 16 scalar and 4 EM.

**On the built library:** the case passes (108 assertions, 0.003 s, load 4 to 30 from other sessions); the EM rows reach
R + T 0.318 at most (TM, 125 Hz, 60°).

| run | result |
| --- | --- |
| `[wave]` | 165110 / 179, identical to `test_wave.txt`; twins unchanged |
| `[acoustic]` | **36813** assertions in 52 cases, all passed; against phase 2 only the 16 EM rows are added |
| whole suite | **1311472** in 1138, all passed; the same 16 rows added, nothing else moves |

Documents: the counts in `md/acoustic.md` 0 and 10 and `md/wave.md` 0; the `md/wave.md` 0 sentence now says the case
guards the kill for the scalar types and for EM. Test file: `test_acoustic_leaf.cpp`, now +97 / −8 against 04b92bd.
