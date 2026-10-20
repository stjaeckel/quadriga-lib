# Precision of a traced leg — report

Brief: `md/precision_brief.md`. One section per phase.

## 0. Phase 0 — analysis (2026-10-19)

### 0.0 Baseline and rig

- At the start: last commit de60194; `git status` shows only `md/acoustic_manager.md` (modified) and the brief (new).
  Nothing under `src/`, `tests/` or `md/acoustic.md` was modified.
- Baselines (the 12-thread commands of `md/wave.md` 0), identical line for line to `test_wave.txt`, `test_acoustic.txt`,
  `test.txt`: `[wave]` 165258 assertions in 180 cases, `[acoustic]` 41523 in 58, whole suite 1316330 in 1145, all pass.
  `src/ray_mesh_interact.cpp` 5159 lines. `python3 tools/acoustic/check_docs.py` passes.
- `build_linux` is configured with `ENABLE_AVX2=OFF` and `ENABLE_CUDA=OFF` (`CMakeCache.txt`), so the suite runs the
  GENERIC kernel only. A default user build (`ENABLE_AVX2` defaults to ON in `CMakeLists.txt`) picks AVX2. The rig
  compiles `quadriga_lib.cpp`, `ray_triangle_intersect.cpp`, `ray_triangle_intersect_avx2.cpp` and the two AVX2 check files
  with `BUILD_WITH_AVX2=1` and links them in front of `libquadriga.a`, which makes `use_kernel` 2 available. CUDA was not
  measured. The GPU had two compute processes of other sessions, the build has no CUDA, and the CUDA kernel is single
  precision like AVX2 by its documentation.
- Scratch `build_linux/scratch/prec` (not kept; every number below is stated here):
  - `rmi_plain.cpp`: a plain copy of `src/ray_mesh_interact.cpp`. Linked in front of the library it reproduces all three
    baselines line for line.
  - `rmi_trace.cpp`: the same with `PREC_TRACE` prints at the hit points, the relaunch, the leg, the medium gain and the
    ledger.
  - `rmi_w.cpp`: the what-ifs of 0.6 behind the bitmask `QD_PREC`. With the mask unset it reproduces all three baselines
    line for line.
  - `probe.cpp`: includes `test_acoustic_state_update.cpp` and adds the cases `[prec]` and `[prec_hits]`. `[prec]` is a
    copy of `bf::walk` with the template and kernel as parameters, the wall translated (`Xm`, `Ym`) and turned (`rot`),
    the ray translated (`Xr`, `Yr`) as `diag_ray_at` (`g=0`) or as `r1::wall_g` (`g=1`), and the first transmitted path
    kept apart. `[prec_hits]` compares each kernel's hit points with the exact plane intersection.
  - `rp.cpp`: `ray_progress` (float) on the same walls, with the direct path, T and R summed as the band probe does, and
    the path's float length.
  - Drivers in Python: `prl.py`, `map_ac.py`, `map_em.py`, `s4_ac.py`, `whatif.py` and others.
- Kernel, template and caller are named in every table. "Walker" means `ray_triangle_intersect` → `ray_mesh_interact` →
  `ray_state_update` called directly, one event at a time, as `bf::walk` does. "`ray_progress`" means the float caller.
  All probes ran single-threaded under `timeout`, in a pool of at most 10 processes, and never alongside a suite.

### 0.1 In short (for Stephan)

**The two findings come from two different causes, and neither is the cosine.**

1. **The 0.2 dB at 100 m comes from the relaunch nudge.** `ray_mesh_interact` relaunches every ray from
   `origN = F + off·FD`, where `off = qd_relaunch_offset(F) = 8·2⁻²³·max(|F|∞, 1)`: 0.95 µm per metre of coordinate.
   The next event measures the in-medium leg from `origN`, not from the hit point, so **every in-medium leg is short by
   `off`**. That holds for both templates, every kernel, and both the walker and `ray_progress`.
   - At y = −100 m the 4 mm pane is traced as 3.9046 mm. Its mass term is 31.394 dB instead of 31.602 dB, which is the
     whole 0.2 dB (traced: leg 0.0039046325680 m, nudge 9.5367432e-5 m).
   - The hit points themselves are exact: `ray_mesh_interact` recomputes F and S against the hit triangle's plane in
     double, whatever the kernel (≤ 2e-11 m off the exact intersection at 10 km in double).
2. **The 89.9° miss is a fixture artefact, not precision.** `r1::wall_g`'s first segment ends inside the 152 mm wall
   (`dl = min(1, 50/tan θ)` = 0.087 m at 89.9°). The entry event preloads F→D (50.000 m) as `md/wave.md` 11 describes.
   The walker then relaunches from `origN` and adds that part again: the leg is 137.09 m instead of 87.09 m. With a first
   segment that reaches past the wall the row matches the plate rule to 0.00000 dB at all five frequencies. At 89.5° the
   segment already reached past the wall.

**Three more mechanisms showed up while mapping:**

- **(N2) Rays are lost near grazing.** The nudge moves the relaunch point off the face by only `off·cos θ`.
  - From about 87°, when the face's own coordinate is 1 m or more from the origin, a float kernel rounds that point back
    onto the face and hits the same face again. The ray is then lost (no transmitted path).
  - This affects AVX2 in the double template, every kernel in the float template, and `ray_progress` with either kernel,
    EM included. The double template with GENERIC (the suite's configuration) is immune.
- **(N3) A thin wall can be skipped.** Where the nudge exceeds a wall's thickness, the relaunch point lies beyond the
  exit face and the ray is lost at any template. A 4 mm pane at normal incidence is traced as 3.05 mm at 1 km and is lost
  at 10 km; by the nudge's formula the crossover is at |F| = 4.2 km.
- **(N4) Float rounding flips the VBS gate.** The gate `|cos_t − cos| > 1e-6` flips on float rounding of a short
  segment 10 m or more from the origin. A mass-law leg then counts as refracted, the plate rule is skipped, and the wall
  passes 26 dB too little (the two leaks, a²τ).
  - Direct float callers meet it at most angles.
  - `ray_progress` meets it only on short remaining segments: at 1 km in 5 of 89 angles with 20 m left, 44 with 5 m,
    74 with 2 m.

**Size (today, double, GENERIC, walker; `ray_progress` the same to the third decimal):**

| quantity | 10 m | 100 m | 1 km | 10 km |
| --- | --- | --- | --- | --- |
| in-medium leg, absolute | −9.9e-6 m | −9.6e-5 m | −9.5e-4 m | −9.5e-3 m (a 4 mm pane is lost) |
| 4 mm pane, mass term, normal incidence, 500 Hz | +0.02 dB | +0.21 dB | +2.33 dB | lost |
| 13 constructions, diffuse TL, worst over 63 Hz–4 kHz | — | −0.14 dB | −1.55 dB (4 mm glass; −2.53 dB per angle) | — |
| EM 28 GHz, direct path through 4 mm glass, normal incidence | +0.002 dB / +0.45° | +0.019 dB / +4.8° | +0.19 dB / +48.5° | lost |
| EM 28 GHz, 152 mm concrete | +0.004 dB / +0.4° | +0.042 dB / +4.1° | +0.43 dB / +41.4° | +4.3 dB |
| EM summed slab (Airy), worst over 4 walls × 4 frequencies, T / R | 0.13 / 0.25 dB | 0.99 / 2.69 dB | 6.9 / 10.6 dB | — |

- **EM crosses the guideline first.** The phase error is `k0·(n−1)·off`. At about 10 m it already moves a coherent slab
  sum by more than 0.1 dB at 28 and 60 GHz (4 mm glass: R 0.15 / 0.25 dB). At 100 m it moves a thin slab's sum by up to
  1.4 dB (28 GHz) and 2.7 dB (60 GHz).
- **Acoustics stays inside ±3 dB up to 1 km.** The worst is 4 mm glass at 2.5 dB per angle. The absorption change is
  ≤ 0.01.
- **The free-space part of an EM path is more precise, except near grazing.** In `ray_progress` the float path length
  moves by up to 6.2e-5 m at 1 km for 0–60° (5.2e-6 m at 100 m), against 1.4e-3 m (1.4e-4 m) for the in-medium leg's
  `(n−1)·off`. Near grazing (85–89°) the float path itself moves by 4e-5 to 6e-4 m and reaches the in-medium size. The
  walker's free-space path is exact.
- **What the documents say today.** `md/wave.md` 8 records the nudge as "1e-6 m … up to 4e-3 dB", the value near the
  origin, without its growth with the coordinate. The float `use_vbs` rounding is recorded only "near normal incidence".

**Recommendation (0.7):**

- **W1, the core fix.** The ledger starts each in-medium leg with the nudge it skipped. It lives in `ray_state_update`,
  about 22 lines; no state, no API, no other file. With it the double leg is exact to 1e-12 relative up to 10 km, and
  every acoustic and EM row of the map moves by 0.000 dB / 0.0° against the wall at the origin.
- **W3″ for N4.** Only in a mass-law medium, the VBS gate's tolerance is scaled to float precision (about 6 lines). EM is
  unaffected by construction.
- **W2 for N2 (optional).** The nudge keeps a normal component of `off`, about 7 lines. It removes every lost ray up to
  1 km through `ray_progress`.
- **Plus:** `r1::wall_g` repaired, and documentation of N3 and of float position rounding as limits.

W1 is a wave change: it moves `[wave]` rows by today's nudge at the suite's coordinates, so Stephan decides.

### 0.2 Step 1 — the two findings reproduced

Probe `[prec]`, the plain copy, 4 mm glass with ρ = 2500, `a` 8.4794e-10, `m` 20, `tf` 0.05, `side` 0 (the infinite
plate, as at the time of the finding), types 4 and 5, 500 Hz; reference the plate rule `a·τ/(a + τ − a·τ)` with the
infinite plate's term:

| ray | GENERIC (`K` 1) | AVX2 (`K` 2) |
| --- | --- | --- |
| `diag_ray_at`, crossing at y = 0.37, mesh at the origin | −31.6585 dB (+0.0022 against the rule) | the same |
| `diag_ray_at`, crossing at y = −100 m, mesh at the origin (`ub::build`, 400 m faces) | −31.4537 (+0.2070) | the same |
| `wall_g`, mesh and ray moved to y = −100 m (crossing at local y = 0.37) | −31.4545 (+0.2062) | the same |
| `wall_g`, mesh and ray moved to x = +100 m (along the normal) | −31.4537 (+0.2070) | the same |

The finding reproduced: −31.4537 against −31.6585 dB. Moving the wall with the ray gives the same error, so this is the
distance from the origin and not the distance along the face.

152 mm concrete (ρ = 347/0.152, `a` 2.026e-9, `m` 20, `tf` 0.05, `side` 0), `wall_g`, types 4 and 5 identical, walker −
plate rule:

| | 31.5 Hz | 63 | 125 | 500 | 2000 |
| --- | --- | --- | --- | --- | --- |
| 89.5° (`wall_g`) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 89.9° (`wall_g`, first segment `dl` 0.087 m, ends inside the wall) | −0.0067 | −0.0268 | −0.1035 | −1.1724 | −3.4165 |
| 89.9°, `dl` 0.2 m or 1 m (the segment reaches past the wall) | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

The finding reproduced: −0.007 / −0.027 / −0.10 / −1.17 / −3.42. With a segment past the wall it is exact.

### 0.3 Step 2 — where the digits go

**One ray through the chain.** 4 mm glass at normal incidence, 100 m from the origin (`diag_ray_at` moved by −100.37 m in
y), 500 Hz, type 4, double, GENERIC; prints with `%.17g` (`PREC_TRACE`):

| step | the library carries | exact | error |
| --- | --- | --- | --- |
| entry hit F (`ray_mesh_interact`, recomputed against the plane in double) | (0, −100, 0) | (0, −100, 0) | 0 |
| exit hit S | (0.0040000000000000036, −100, 0) | (0.004, −100, 0) | 3.5e-18 m |
| relaunch `origN` = F + off·FD | (9.5367432e-05, −100, 0) | — | `off` = 8·100·2⁻²³ = 9.5367432e-5 m |
| leg at the exit, `dist_orig_fbs` = \|F_exit − origN\| | 0.0039046325680000002 | 0.004 | −9.5367e-5 m (−2.38 %) |
| cosine `\|cos(θ + π/2)\|` | 1 | 1 | 0 |
| thickness surrogate `dist_geo·cos θ` | 0.0039046325680 | 0.004 | −2.38 % |
| X0, mass term | 37.1295, 31.3944 dB | 38.0273, 31.6019 dB | −0.2075 dB |
| ledger written (`acc_geo` out) | 1.19609536743 (the preload F→D of the exit segment) | — | — |

The same ray at float (`K` 2 and `K` 1 alike): leg 0.0039046327583 m and term 31.3944 dB. Through `ray_progress` (float,
AVX2): leg 0.0039046327583 m, term 31.3944 dB, T −31.453 dB. So **one step loses the digits, the relaunch, by `off`**.
Everything after it is exact to the template's precision.

**The hit points per kernel** (`[prec_hits]`, a 4 mm pane, the segment 1 m before to 1 m beyond, the wall moved with the
ray; error against the exact plane intersection in long double):

| | double, GENERIC | double, AVX2 | float, either kernel |
| --- | --- | --- | --- |
| kernel FBS / SBS (`ray_triangle_intersect` output) | 0 / ≤ 5e-13 m | 0 / up to 5.4e-3 m (10 km, 89°) | up to 6.5e-4 / 5.4e-3 m (10 km) |
| `ray_mesh_interact` F / S (what the leg is formed from) | ≤ 2e-11 m | ≤ 2e-11 m (kernel-independent) | the float input's rounding: 4.8e-9 m (origin) … 6.5e-4 m (10 km) |
| relaunch nudge `\|origN − F\|` | 9.5e-7 (0 to 1 m) … 9.5e-3 m (10 km) | the same | the same ± the float store |
| leg error | −`off`, exactly | −`off`, exactly | −`off` ± float rounding (at 10 m, 89°, offset along the normal: −2.6e-5 m where `off` is 9.5e-6 m) |

- **The kernel never reaches the leg.** `ray_mesh_interact` uses only the face indices; brief candidate (a), the leg from
  double hit points against the hit triangle's plane, is already in place for the hit points.
- **The leg starts at the relaunch point, not at the hit point.** That is the entire error in double, and the dominant
  one in float up to 10 m–1 km, depending on the angle.

**Relative size.**
- The leg error is −`off` along the path. Its relative size is `off·cos θ_t / d` for a wall of normal thickness `d`, so
  it is largest at normal incidence and shrinks toward grazing.
- At 1 km, normal incidence: 4 mm −23.9 %, 12.5 mm −7.6 %, 152 mm −0.63 %, 300 mm −0.32 %.
- The cosine contributes nothing measurable. `fbs_angle` and the plane normal come from double vectors, and the trace
  shows `cos` equal to the exact one to 1e-15 (double) or 1e-15 relative (float inputs at the origin).

### 0.4 Step 3 — the error over what a scene meets

**Acoustic, walker** (types 4 and 5 equal to the third decimal):
- Walls: glass 4 mm (ρ 2500), gypsum 12.5 mm (ρ 800), concrete 152 mm (ρ 2283), concrete 300 mm (ρ 2300). All
  `tf` 0.05, `side` 0, 500 Hz, `wall_g` geometry with the first segment past the wall, faces of 2 km.
- Each cell: transmitted power against the formula (dB); "lost" means no transmitted path.
- Double, GENERIC, offset in the wall's plane (y); along its normal (x) the same, except for the lost rays of N2 below:

| wall | angle | 1 m | 10 m | 100 m | 1 km | 10 km |
| --- | --- | --- | --- | --- | --- | --- |
| glass 4 | 0° | 0 | +0.02 | +0.21 | +2.33 | lost |
| | 60° | 0 | +0.01 | +0.10 | +1.04 | lost |
| | 85° | 0 | 0 | +0.01 | +0.06 | +0.62 |
| | 89°–89.99° | 0 | 0 | 0 | 0 | ≤ +0.01 |
| gypsum 12.5 | 0° | 0 | +0.01 | +0.07 | +0.68 | +11.65 |
| concrete 152 | 0° | 0 | 0 | +0.01 | +0.05 | +0.56 |
| concrete 300 | 0° | 0 | 0 | 0 | +0.03 | +0.28 |

- The error falls toward grazing (relative leg error `off·cos θ/d`).
- **Lost rays (N2), offset along the normal.** The double template with AVX2 loses rays where the face's own coordinate
  is 1 m or more:

  | wall | 87° | 88° | 89°, 89.9° |
  | --- | --- | --- | --- |
  | 4 mm, 12.5 mm | lost at 1 m | lost at 1–100 m and 10 km | lost at every distance from 1 m |
  | 152 mm | lost at 1 m | lost at 10 m, 100 m and 10 km | lost from 10 m |
  | 300 mm | — | lost at 10 m, 100 m and 10 km | lost from 10 m |

  GENERIC in double loses none.
- **Float template (`K` 2):** the same leg errors, the same lost rays, and the −26 dB of N4 at most angles from 10 m.

**N4 in numbers.** Angles 0–88° in 1° steps where T moved by more than 5 dB against the origin, 4 mm glass / 152 mm
concrete:

| caller | 10 m | 100 m | 1 km |
| --- | --- | --- | --- |
| walker, float, AVX2 | 38 / 8 | 76 / 57 | 82 / 79 |
| `ray_progress` with today's 3 km launch segment | 0 | 0 | 0 |
| `ray_progress` with 20 m left | 0 | 0 | 5 |
| `ray_progress` with 5 m left | 0 | 0 | 44 |
| `ray_progress` with 2 m left | 0 | 24 | 74 |

- The gate compares the direction the previous event wrote (`path_dir`, a float unit vector) with the incidence angle of
  the segment `origN → D`. Both ends of that segment are rounded to float.
- So the gate's noise is about `2⁻²³·|coordinate| / |segment|`. It crosses 1e-6 on short segments far from the origin.
- **EM (wave media) is not affected the same way.** A dielectric with type 1 has `use_vbs` set by its genuine
  refraction, and type 2 has `bent` set. The flip matters where a leg should read as not bent: mass-law media.

**EM, direct path** (`eps` 1.5, first transmitted path), double, GENERIC, offset in-plane:
- Walls from `data/materials_rf.csv`: 4 mm `itu_glass`, 12.5 mm `itu_plasterboard`, 152 and 300 mm `itu_concrete`.
- Each cell: in-medium loss change dB / phase change ° against the same wall at the origin; the worst of VV and HH;
  100 m then 1 km.

| wall | type | θ | 1 GHz | 10 GHz | 28 GHz | 60 GHz |
| --- | --- | --- | --- | --- | --- | --- |
| glass 4 | 1, 2 | 0° | +0.000/+0.2, +0.002/+1.7 | +0.005/+1.7, +0.049/+17.3 | +0.019/+4.8, +0.194/+48.5 | +0.054/+10.3, +0.538/+103.9 |
| glass 4 | 1 | 85° | 0/0, 0/+0.2 | 0/+0.2, +0.005/+2.2 | +0.002/+0.6, +0.018/+6.2 | +0.005/+1.3, +0.051/+13.3 |
| glass 4 | 2 | 85° | as at 0° | as at 0° | as at 0° | as at 0° |
| plasterboard 12.5 | 1, 2 | 0° | +0.001/+0.1, +0.008/+0.7 | +0.007/+0.7, +0.070/+7.5 | +0.018/+2.1, +0.184/+20.9 | +0.037/+4.5, +0.376/+44.8 |
| concrete 152 / 300 | 1, 2 | 0° | +0.003/+0.1, +0.031/+1.5 | +0.019/+1.5, +0.190/+14.8 | +0.042/+4.1, +0.426/+41.4 | +0.077/+8.8, +0.774/+88.6 |
| concrete 152 / 300 | 1 | 85° | 0/0, +0.003/+0.2 | +0.002/+0.2, +0.018/+2.0 | +0.004/+0.5, +0.041/+5.5 | +0.007/+1.2, +0.075/+11.8 |

- **The refracting tracer (type 2)** is short by `off` along the refracted ray at every angle. So its error does not fall
  toward grazing.
- **The undeviated tracer (type 1)** follows `cos θ_i`.
- **Frequency and distance:** the phase error is linear in frequency and in distance, `k0·(n − 1)·off`. The loss error
  is the rate times `off`.

**`ray_progress`** (float; GENERIC and AVX2 give identical rows):
- **In the medium:** the same leg error (−8.9e-6 m at 10 m, −9.5e-5 at 100 m, −9.5e-4 at 1 km, −9.5e-3 at 10 km) and the
  same direct-path change as the walker. For example, 152 mm concrete at 28 GHz, normal incidence:
  +0.004 / +0.042 / +0.426 / +4.265 dB and +0.4° / +4.1° / +41.4° / +53.9° (wrapped).
- **Rays lost near grazing:** at 89° when the offset is along the normal, from 1 m (glass) or 10 m (concrete). In the
  13-construction run below, 1 of 48 angles (89.06°) is lost per octave.
- **Free space.** The path's float length (`path::length`) and the double polyline through its float points move
  against the origin by (EM rows, worst):

  | distance | 0–60° | 85° | 89° |
  | --- | --- | --- | --- |
  | 10 m | 9.5e-7 m | 9.3e-6 m | 9.8e-6 m |
  | 100 m | 5.2e-6 m | 4.1e-5 m | 2.1e-4 m |
  | 1 km | 6.2e-5 m | 6.0e-4 m | 2.4e-4 m |
  | 10 km | 5.2e-4 m | 5.5e-3 m | 7.5e-3 m |

  The table is the worst of the float length and the double polyline through the same float points. The float sum adds
  its own rounding on top of the points': the two differ by 3.9e-7 m at the origin, 2.9e-6 m at 100 m, 2.1e-5 m at 1 km,
  4.9e-4 m at 10 km.

- **In-medium against free space.** The in-medium leg's effective path error `(n − 1)·off` is 1.4e-5 m at 10 m,
  1.4e-4 m at 100 m and 1.4e-3 m at 1 km for glass (1.2e-3 m for concrete). Up to 60° it is 15 to 27 times what the float
  free-space path carries; near grazing (85–89°, offset along the normal) the two are of the same size.
- **The direct callers' free-space length is exact** (the walker sums double hit points).
- **Documentation:** the float storage of positions and lengths is documented only by type (`arma::fmat`,
  `float len_acc`), not as a precision statement.

**Face size.** A 4 mm pane turned by −30° about z, so its planes are not axis-aligned. Faces of half size 200 m against
1.5 m (wall-sized), distance in y:
- **Double:** identical rows. The leg error is the nudge (−2.39e-2 relative at 100 m at either size).
- **Float with 1.5 m faces:** the same as double, plus rounding.
- **Float with 200 m faces:** the leg error grows to −2.45e-2 at 100 m, the vertex rounding at 200 m adding 5e-6 m.
- **Float with large turned faces is erratic even at the origin.** At normal incidence the ray is lost with AVX2 at
  200 m and with GENERIC at 50 m; GENERIC at 200 m runs away (122233 transmitted paths); 10 m and 1.5 m faces are fine.
  The cause is not traced. It concerns the float mesh of a turned face, not the leg; listed in 0.7 as an open finding.
- **Axis-aligned faces:** face size does not matter in either template.

### 0.5 Step 4 — what it does in practice

**Acoustics, the 13 reference constructions** (the decided columns, `grazing/fits/P1_davy_side_lab.json`, with their
`side`) as traced solids, type 4, 48 angles 0.94–89.06°, octaves 63 Hz–4 kHz. Change against the same wall at the
origin, the largest over the 7 octaves:
- **Walker** (double, GENERIC): offset in the plane and along the normal give the same rows.
- **`ray_progress`** (float, AVX2): the same, except where noted below.
- Each cell: diffuse ΔTL / per-angle ΔTL / Δα, all in dB except α:

| construction | 100 m | 1 km |
| --- | --- | --- |
| glass 4 mm | −0.14 / −0.22 / −0.0008 | −1.55 / −2.53 / −0.010 |
| glass 6 mm | −0.08 / −0.14 / −0.0003 | −0.81 / −1.50 / −0.003 |
| 4-16-4 | −0.06 / −0.10 / −0.0002 | −0.59 / −1.10 / −0.002 |
| 4-12-4-12-4 | −0.04 / −0.07 / −0.0001 | −0.39 / −0.72 / −0.001 |
| gypsum 12.5 | −0.04 / −0.07 / −0.0003 | −0.40 / −0.70 / −0.003 |
| plywood 18 | −0.03 / −0.05 / −0.0001 | −0.26 / −0.47 / −0.002 |
| concrete 152, brick 95 / 203, block 190, SCR 140, brick 105 + plaster, brick 203 + board | ≤ 0.005 / 0.009 / 0 | ≤ 0.055 / 0.087 / 0 |

- **`ray_progress` with the offset along the normal** loses the 89.06° angle in every octave (N2). The diffuse TL of the
  heavy walls then rises by +0.16 to +0.62 dB at 100 m and 1 km.
- **Against the guidelines:** ±3 dB transmission and ±0.05 absorption hold everywhere up to 1 km, except for the lost
  rays. The worst is 4 mm glass at −2.53 dB per angle at 1 km.
- **Beyond:** at 10 km a 4 mm pane is lost (N3), and 12.5 mm gypsum is +11.65 dB at normal incidence.
- **Float callers** add the −26 dB of N4 wherever the remaining segment is short.

**EM, the coherent sum of a traced slab.** `eps` 0, the series summed by `qd_slab_resolve` from the leg; double,
GENERIC; both offsets; types 1 and 2; VV and HH; angles 0–89.99°. The worst |ΔT| / |ΔR| in dB against the same wall at
the origin (* = some rays lost):

| wall | GHz | 1 m | 10 m | 100 m | 1 km | 10 km |
| --- | --- | --- | --- | --- | --- | --- |
| glass 4 | 1 | 0.002 / 0.000 | 0.019 / 0.013 | 0.188 / 0.134 | 2.12 / 1.49 | 1.96 / 6.88 * |
| | 10 | 0.001 / 0.000 | 0.014 / 0.010 | 0.128 / 0.102 | 0.61 / 0.68 | 2.54 / 1.33 * |
| | 28 | 0.008 / 0.006 | 0.083 / **0.148** | 0.740 / 1.357 | 2.20 / 5.63 | * |
| | 60 | 0.012 / 0.010 | **0.128 / 0.254** | 0.993 / 2.693 | 6.93 / 10.64 | * |
| plasterboard 12.5 | 10 | 0.005 / 0.001 | 0.050 / 0.024 | 0.485 / 0.255 | 6.29 / 3.60 | 9.6 / 17.2 |
| | 28 | 0.007 / 0.002 | 0.066 / 0.044 | 0.614 / 0.450 | 1.49 / 3.82 | 3.8 / 8.2 |
| concrete 152 / 300 | 28 | 0.000 / 0.000 | 0.004 / 0.000 | 0.043 / 0.000 | 0.43 / 0.00 | 4.27 / 0.00 |
| | 60 | 0.001 / 0.000 | 0.008 / 0.000 | 0.077 / 0.000 | 0.77 / 0.00 | 7.74 / 0.00 |

- **The guideline of about 0.1 dB on a coherent sum is crossed:**
  - at about 10 m for a thin low-loss slab at 28 and 60 GHz;
  - at 100 m for every thin slab from 10 GHz up, and for 4 mm glass at 1 GHz as well (12.5 mm plasterboard at 1 GHz:
    0.05 dB at 100 m, 0.52 dB at 1 km);
  - for the lossy 152 / 300 mm concrete, only through its loss: 0.04 dB at 100 m, 0.43 dB at 1 km at 28 GHz. Its series
    is negligible.
- **The followed series** (`eps` 1.5, every bounce summed coherently) moves by the same amounts: 4 mm glass at 28 GHz
  2.0 / 5.9 dB at 1 km.
- **The float template** changes these by at most 0.03 dB.
- **EM correctness in a real scene therefore depends on where the scene sits.** A site model in the usual coordinates
  (hundreds of metres to kilometres) gets in-slab phases wrong by tens of degrees at mm-wave.

### 0.6 Step 5 — fix candidates

**What each needs.** `ray_state_update` knows the relaunch the event performs: the base point `A0` (F, or S of a
co-located pair), `off = qd_relaunch_offset(F)`, the direction, and the face normal. The next event knows the leg's
segment exactly. The what-ifs live in `rmi_w.cpp` behind `QD_PREC`:

- **W1 (bit 1), the ledger starts each in-medium leg with the nudge.** At an event that relaunches the ray into a medium
  (`next_current` ≠ air, ledger ending at the relaunch point, not preloaded to D), the ledger is written as
  - `dist_geo += off`;
  - `dist_refract += off·|FD·N| / |dir_next·N|`, where FD is the segment direction ray_mesh_interact used: the incoming
    segment for types 1 and 4, otherwise the written direction. This is exact for parallel faces.

  The next event then measures `|F₂ − origN|` and the sum is the leg from the hit point.
  - About 22 lines in `ray_state_update`.
  - No per-ray state, no API change, `ray_triangle_intersect` and `ray_progress` untouched.
  - The preloaded legs (`DIST_SET(dist_fbs_dest)`) are left alone.
- **W2 (bit 2), the nudge keeps its normal component.** `off / max(|FD·N|, 1e-3)` along FD, in `ray_mesh_interact`'s
  relaunch and in `RELAUNCH`, with W1 carrying the longer nudge. About 7 lines.
  - Below 89.94° the relaunch point lies `off` off the face. Beyond that the floor holds it at `1e-3·off/…`.
  - Along the ray the point moves up to 1000·`off` (0.95 m at 1 km at the floor). A face that lies within that stretch
    of the ray and is not parallel (a corner met at grazing) could be skipped. Not seen in the map, but a risk to state.
- **W3″ (bit 16), the VBS gate at float precision in mass-law media only.** Where the current medium has `m > 0` and the
  template is float, the gate's tolerance becomes `1e-6 + 8·2⁻²³·max(|O|, |F|, 1) / |O − F|`. About 6 lines.
  - **Two variants were rejected by the suite:**
    - W3 (bit 4) is the same tolerance for every medium. It fails *Unentered branches - sheet and stack and slab series
      at float against double* (4 rows whose verdict codes differ from double), because it hides real refractions of
      wave media in float.
    - W3′ (bit 8) skips the gate in mass-law media. It fails *Leak reciprocity - a mass-law panel …*, whose guard sends a
      genuinely refracted wave into a mass-law solid, which only the gate detects.
  - W3″ keeps that guard: a real bend differs by 0.1 or more, rounding by about 1e-5.

**Effect on the numbers of 0.4** (`whatif.py`; walls and angles of the map; 28 GHz for EM). "Lost" counts rays out of
the 72 rows per distance; each cell gives 10 m / 100 m / 1 km / 10 km:

| switch | acoustic, double, GENERIC: worst ΔT, dB | acoustic, double, AVX2: lost | acoustic, float, AVX2 | EM direct path, double, GENERIC, ΔdB / Δ° |
| --- | --- | --- | --- | --- |
| today | +0.021 / +0.208 / +2.33 / +11.7 | 16 / 16 / 16 / 22 | −25.9 (N4) from 10 m; lost as AVX2 | +0.004/+0.5 · +0.042/+4.8 · +0.43/+48.5 · +4.3/+171 |
| W1 | 0.000 at every distance; relative leg error ≤ 3e-10 | 16 / 16 / 16 / 22 | N4 unchanged; lost unchanged | 0.000 / 0.0 at every distance (lost rays aside) |
| W1 + W2 | 0.000 | 0 / 0 / 0 / 16 (10 km: N3) | lost 0 / 0 / 0 / 16; N4 unchanged | 0.000 / 0.0 |
| W1 + W3″ | 0.000 | 16 / 16 / 16 / 22 | +0.001 / +0.013 / −0.059 / +1.28 dB, no flips; lost as AVX2 | 0.000 / 0.0 |
| W1 + W2 + W3″ | 0.000 | 0 / 0 / 0 / 16 | +0.001 / +0.013 / +0.020 / +0.146 dB, no flips, none lost to 1 km | 0.000 / 0.0 |

Notes on the table:
- **What W1 leaves in float.** Float position rounding remains: EM direct path at 28 GHz, type 1, 1 km: +0.024 dB /
  ±3.8°; summed slab +0.51 dB (glass, 1 km). Today the same rows are +0.45 dB / 48.5° and 2.2 dB. This residual is the
  size of the float free-space error of 0.4.
- **N4 under W3″:** 0 of 89 angles moved by more than 5 dB, walker (float, AVX2) at 10 m, 100 m and 1 km and
  `ray_progress` with 2 m segments at 100 m and 1 km (today 24 and 74).
- **N2 through `ray_progress` with W1 + W2:** 0 of 7 angles at 85–89.9° lost at 1, 10, 100 m and 1 km. Today 1 to 5 of 7
  are lost, EM concrete at 28 GHz included.
- **N3 stays.** At 10 km the nudge (9.5 mm) is wider than a 4 mm or 12.5 mm wall, and W2 makes that true at every angle.

**Rows the switches move** (`tb_w_noavx`, the test files as they are; QD_PREC unset reproduces every baseline line):

| switch | `[wave]` | `[acoustic]` | whole suite |
| --- | --- | --- | --- |
| W1 | 14 cases fail | pass | 14 fail |
| W1 + W2 | 19 fail | 2 fail | 21 fail |
| W3 (all media) | 1 fails | pass | 1 fails |
| W3′ | pass | 1 fails | 1 fails |
| W1 + W3′ | 14 fail | 1 fails | 15 fail |
| W3″ | pass, every line identical | pass, every line identical | pass, every line identical |
| W1 + W3″ | 14 fail (the 14 of W1) | pass | 14 fail |
| W1 + W2 + W3″ | 19 fail (as W1 + W2) | 2 fail (as W1 + W2) | 21 fail |

**What W1 moves, and why.**
- **The every-digit twins:** 260 of 736 rows in 5 tables move, by up to 8.8e-3 relative (QD_TWIN_WRITE under W1 against
  `tests/data/wave_twins.tsv`):

  | table | rows moved | largest relative change |
  | --- | --- | --- |
  | *bare tf T* | 84/84 | 4.0e-3 |
  | *flat branch H* | 56/136 | 8.8e-3 |
  | *rare sheet F* | 48/60 | 6.3e-3 |
  | *scalar closed face C* | 12/18 | 5.9e-4 |
  | *sheet tf H* | 60/268 | 1.3e-3 |

  These are thin sheets and tf faces near their resonances: the 1 µm of leg the nudge took at the suite's coordinates.
  W2 moves *stack geometry C1* as well (56/56, the rows print `origN`; measured as W1 + W2 + W3).
- **The other W1 failures are fixtures that state today's short leg.** Nineteen reference sites in
  `test_wave_state_update_branch.cpp` (5), `test_wave_state_update_chain.cpp` (13) and `test_acoustic_state_update.cpp`
  (1) build "the thickness the ray crossed" as `X − E.origN(0,0)` (or `+ origN`).
  - With those 19 sites restated to the hit point (`E.fbs(0,0)`, a scratch copy of the three files), **today's library
    fails 7 `[wave]` cases by the nudge**, and W1 passes all 7:
    - *Refracting tracer - … the leg is exact* (1.4e-3 dB against 1e-3);
    - *Phase reference - …* (3.0e-3 against 1e-3);
    - *Unentered branches - a verdict handed in … near a pole* (0.505 against 1e-6: εr 1e4 lossless, tuned to a
      resonance of the leg);
    - *… the displacement budget …*, *… the core's loss cap …*, *One solver - …*, *Sheet reflect port - …*.
  - What still fails under W1 with those references is the same pattern elsewhere, not a defect of W1:
    - references formed from `o` after `ub::onward` (the displacement budget's `hf = 0.5·L − o(0,0)`);
    - `C.ox` (*Measurement - the mirror behind a second solid …*);
    - a model of ledger against polyline (*Phase reference - …*, two solids with a gap, 0.0177° against 0.005°);
    - exact ledger values (*ray_state_update - accumulated distance ledger*: "a non-terminal entry restarts the
      in-medium distance at zero", now `off` = 1.43e-6 m; *Diagnostic - Host layer carried through a matched sheet*:
      `L` + 9.5e-7 m);
    - *One solver - …* at 1.2e-5 against 1e-6.
  - Phase 1 would restate them to the hit point.
- **What W2 adds:**
  - `test_wave_mesh_interact.cpp` cases that assert `origN` at the old nudge (*Ray-Mesh Interact - Medium to Medium*,
    *Dielectric Medium to Air (float)*, *Conductive to Dielectric*);
  - *Stack geometry - …* (twins of `origN`);
  - *Exits and verdicts - …* (a reference built as `L − origN + origN`, 1.13e-3 against 1e-3);
  - two `[acoustic]` bitwise comparisons of two runs (*Refracting tracer - …* 3331, *Leak direction - …* 3730), equal
    to 1e-15 but no longer bit for bit.

**Run time.** CPU time (user + sys) of single-threaded runs, switches off against W1 + W2 + W3″, alternated, 4 pairs
under `flock /tmp/bench.lock`, load 1.8–3.7 (`cputime.sh`):

| spec | off, median | W1 + W2 + W3″, median | paired ratios | median change |
| --- | --- | --- | --- | --- |
| `[#test_wave_progress]` | 12.02 s | 12.16 s | 1.007–1.018 | +0.9 % |
| `[#test_wave_state_update_chain]` | 9.34 s | 9.34 s | 0.998–1.006 | +0.3 % |

The switches are read once; the per-event cost is a few multiplications per ray.

**Public behaviour each one changes:**
- **W1:** the `acc_dist_geometricN` / `acc_dist_refractedN` outputs of `ray_state_update` (and `ray_progress`'s
  `acc_dist`) carry the nudge from the event on. Their documentation ("accumulated … distance leaving this call") would
  say so.
- **W2:** `origN` of `ray_mesh_interact` is documented as "nudged … by ~8 float ULP at the interaction-point coordinate
  magnitude". It would read "… with a normal component of that size".
- **W3″:** no documented output changes.

**The other candidates of the brief:**

- **(a) The leg from double hit points against the hit triangle's plane.** Already in place for the hit points
  (`ray_mesh_interact` recomputes F and S in double; 0.3). It does not reach the leg's start, which is the relaunch
  point. W1 is the form of (a) that does.
- **(b) The thickness from the distance between the two faces' planes.**
  - At the exit event only the exit plane is known. The entry plane would need per-ray state: a face index is
    4 bytes/ray (0.6 GB at 150 M rays); a normal and offset in float is 16 bytes/ray (2.4 GB).
  - It would also help only the mass term and not EM, and with W1 `dist·cos θ` is already exact (1e-12 relative in
    double).
  - Not recommended.
- **(c) A local origin per event** (positions relative to the hit triangle).
  - It needs the origin per ray, 12 bytes/ray (1.8 GB at 150 M rays), or a `ray_progress` API with relative positions.
  - It would also cure the float rounding residual and N3, which nothing else here does (by construction; not measured).
  - Not recommended unless Stephan wants kilometre scenes in float. Then it is the only route, and a project of its own.
- **(d) The GENERIC double kernel for the second hit only.**
  - It does not touch the leg: the kernel's points are not used for it.
  - In the double template it would cure N2 the way GENERIC already does. `ray_progress` stores positions in float, so
    there it cures nothing.
  - Not recommended.
- **(e) Document the limit and a rule of use.** Needed in any case for what stays:
  - N3: a body thinner than the nudge, 8·2⁻²³ of the coordinate, about 1 mm at 1 km and 10 mm at 10 km;
  - the float position rounding: EM phase about `k0·(n − 1)` times one float ulp of the coordinate, 3.8° at 28 GHz at
    1 km after W1;
  - large turned faces in float.

  The rule of use:
  - **Today**, the 0.1 dB guideline on a coherent slab sum holds only within about 10 m of the origin (thin glass at
    28 / 60 GHz is already 0.15 / 0.25 dB in R at 10 m). Acoustics stays within ±3 dB to 1 km.
  - **With W1**, in double there is no distance limit to 10 km apart from N3. In float (`ray_progress`) the residual on a
    summed glass slab at 28 GHz is 0.02 to 0.05 dB at 100 m and 0.51 dB at 1 km: keep mm-wave scenes within about
    100 m of the origin.

### 0.7 Recommendation and decisions

**Recommendation:**
- **W1 for N1**, because EM correctness comes first.
  - It makes every in-medium leg exact in double at any distance. Every acoustic and EM row of the map moves by
    0.000 dB / 0.0° against the wall at the origin.
  - It cuts the float error about 13-fold (48.5° → 3.8° at 28 GHz, 1 km).
  - It is the least code that closes N1 in both templates and both callers (about 22 lines, one function), and it closes
    the item for every mode rather than gating one.
- **W3″ for N4** (about 6 lines, mass-law media only, EM untouched by construction).
- **`r1::wall_g` repaired** so its first segment reaches past the wall. The *Finite plate - …* case could then trace to
  89.9° again; the comment there that blames precision is wrong.
- **N3 and the float rounding documented** as limits with their numbers (e).
- **W2 offered as an option.** It closes N2: rays lost from about 87° when a face's coordinate is ≥ 1 m, through
  `ray_progress` and the AVX2 kernel, EM included. It costs more of the suite (5 more `[wave]` cases, 2 bitwise
  `[acoustic]` ones) and carries the corner risk of 0.6.

**Decisions for Stephan:**

1. **W1, a wave change.**
   - It moves the `[wave]` twins: 260/736 rows in 5 tables, ≤ 8.8e-3 relative.
   - The fixtures that build their reference from `origN` would be restated to the hit point: 19 sites, plus the ledger
     and `C.ox`/`o` references of 0.6.
   - Phase 1 would add the far-from-origin and grazing cases of the brief.
2. **W2, yes or no**, and if yes the floor `1e-3`, which sets how far along the ray the nudge may go at grazing.
3. **W3″** (acoustic only).
4. **`r1::wall_g`:** repair it and restore the traced rows to 89.9° in *Finite plate - …*.
5. **What to document as the limit:** N3 and the float rounding (`md/wave.md` 8, `md/acoustic.md` 7, the
   `ray_progress` / `ray_mesh_interact` documentation blocks), and the rule of use.
6. **The open finding:** a 4 mm pane with large turned faces in float is lost or runs away even at the origin. Its cause
   is not traced; it is a separate item (it concerns the float mesh, not the leg).

### 0.8 Rig and traps

- The suite build has no AVX2. Any statement about a default user build needs the AVX2 objects linked in front, with
  `use_kernel` fixed; the auto kernel picks CUDA from 500 rays where it exists.
- `qd_relaunch_offset` is continuous in `|F|`, not quantized to ulps: the closing event could reconstruct it from
  `origN`, to 1e-13 m.
- Fixtures often write a slab's thickness as `L − origN`, the traced, nudged one. A fix of the leg moves them; restate to
  `fbs`.
- `r1::wall_g` and `ub::onward`: a first segment that ends inside a body preloads F→D and the walker adds it again. Keep
  `dl` above the wall's thickness.
- The probe's T/R sorting by global x breaks for turned walls: sort along the wall normal (`probe.cpp` does now).
- `thin_slab_threshold` of `ray_progress` must lie in [0, 1]: 1 follows every bounce.
- A dB ratio of two rows both below −120 dB (concrete at 89.5–89.99° at 28 GHz) is noise, not a change.
- Python pools: use threads (`ThreadPool`) or a `__main__` guard. The forkserver default of Python 3.14 re-imports the
  driver.

## 1. Phase 1 — fixtures (2026-10-19)

Decided (Stephan, 2026-10-19):
1. **W1 adopted**, a decided wave change. Each in-medium leg starts with the nudge it skipped; the nudge itself is
   unchanged. The twins are rewritten in phase 2. The fixtures that build "the thickness crossed" from `origN`, from `o`
   after `ub::onward`, from `C.ox` or from exact ledger values are restated to the hit point.
2. **W2 dropped.** Rays lost within about 3° of grazing in float or AVX2 stay a documented limit.
3. **W3″ adopted** (float, mass-law media only).
4. **`r1::wall_g` repaired**, and *Finite plate - …* traced to 89.9°.
5. **Documents in phase 3**, with one rule of use and one line of limits.
6. **The large turned faces in float:** recorded here only.

Scratch: `build_linux/scratch/prec` (the phase 0 rig) and `prec/p1`. The what-if is `rmi_w.cpp` with `QD_PREC=17` (W1 +
W3″), linked in front of `libquadriga.a` as `tb_p1w`.

### 1.0 Start state

- `git status` as at the end of phase 0: `md/acoustic_manager.md` modified; the brief and this report new. Commit de60194.
- **The old tests on today's library reproduce the whole-suite baseline.** `p1/tb_old` is HEAD's versions of the 10 ray
  test files and their header, compiled in scratch (`p1/mkold.sh`) and linked with today's library. It gives 1316330
  assertions in 1145 cases, all passing, every line identical to `test.txt`. It is the reference run for `cases.py`.

### 1.1 What the tests say now

**Helpers** (`ray_test_helpers.hpp`, +43 lines):
- `qd_relaunch_nudge(x)` is ray_mesh_interact's relaunch offset written in the test, `8·2⁻²³·max(|x|, 1)`. Its comment
  states the decision: the ledger carries the offset from the event on.
- `qd_leg_start(E, c)` is where the leg after an event starts: the far face of a co-located pair (closer than 1 mm),
  else the hit point. A slab's thickness is now always the next hit point minus this.
- `bf::walk_far(pal, x, mat, flip, tr, f, thd, eps, dx, dy)` is `bf::walk` through a wall of `ub::build` shifted with
  its ray by (dx, dy). The ray has the `wall_g` geometry, its first segment reaching past the last face.

**Restated to the hit point** (no other change in these cases):

| where | before | now |
| --- | --- | --- |
| the 19 sites of report 0.6 (branch 5, chain 13, acoustic 1) | `X − E.origN(0,0)` | `X − E.fbs(0,0)` |
| *Unentered branches - the displacement budget …* (both tables) | `0.5·L − o(0,0)`, `K.xs − o(0,0)`, with `o` after `ub::onward` | `… − E0.fbs(0,0)` |
| *One solver - …*, the leg behind the stack | `d1 + G + d2 − b0[1].fbs` (one of the 19 sites) | `− b0[1].sbs`: the pair relaunches at its SBS, 10 µm on |
| the four chain helpers with `C.ox` (chain ×3, acoustic ×1; 8 thickness references) | `ox` = `origN` x | `bx` = `qd_leg_start(E, 0)`; the local `Ev` keeps `sbs` |
| *Phase reference - …*, the ledger model (`led`, `ledx`, `ledy`) | from the segment origin (`origN`) | from `qd_leg_start` of the previous event; the comment of block E says the ledger starts at the hit point or a pair's far face, and only the pair's gap is "in no ledger" |
| *Measurement - the mirror behind a second solid …*, the paths' limit | 1e-3 dB, 1e-2 at tracer 5 / B 6 / 85° ("short by its own relaunch offsets") | 1e-3 dB, 2e-3 at that row: W1 leaves 1.34e-3 dB there (2.4e-3 today), not traced further |
| *ray_state_update - accumulated distance ledger* | "a non-terminal entry restarts the in-medium distance at zero" | "… at the relaunch offset it skips": both ledgers equal `qd_relaunch_nudge(1.5)` to 1e-12 |
| *Diagnostic - Host layer carried through a matched sheet* | the host ledger `L` to 1e-9 | `L + qd_relaunch_nudge(L)` to 1e-12 |

**`r1::wall_g` and the finite plate.**
- `wall_g`'s first segment is now `max(min(1, 50/tan θ), thickness + 0.05)` long, so it always reaches past the last
  face.
- *Finite plate - a mass-law panel of side s …* traces its solid to 89.9°. The `thd > 89.6` skip and its comment, which
  blamed precision, are gone, and the case's introduction says "0 to 89.9 deg".
- The new 89.9° traced rows pass on today's library. That was a fixture artefact, not a library one.

**New cases:**

- ***Far from the origin - a traced mass-law wall passes what it passes at the origin, its leg measured from the hit
  point*** (`[acoustic]`).
  - 4 mm glass and 152 mm concrete, limp, `side` 0, `tf` 0.05; types 4 and 5; 125 / 500 / 2000 Hz; 0 / 45 / 80°.
  - The wall moved with its ray to 10 m, 100 m and 1 km, in its plane and along its normal (`bf::walk_far`).
  - It asserts T against the same wall at the origin to 1e-5 dB, and against the plate rule of the true thickness to
    5e-3 dB. The face at exact normal incidence costs 4 mm glass 1.75e-3 dB at 125 Hz, at any distance.
- ***Far from the origin - in single precision a mass-law leg is not taken for a bent one*** (`[acoustic]`). The direct
  path (entry and exit event, `ub::traced<float>` against `<double>`) through the same two walls 100 m from the origin,
  types 4 and 5, 0 to 85° in 5° steps: float within 0.05 dB of double.
- ***Far from the origin - a thin dielectric slab transmits what it does at the origin, summed and followed***
  (`[wave]`).
  - 4 mm `itu_glass` at 28 GHz (ε 6.31 − j·17.98·σ/f, σ = 0.0036·28^1.3394, written in the test); types 1 and 2;
    `eps` 0 and 1.5; 0 / 30 / 60 / 80°.
  - The slab moved to 10 m, 100 m and 1 km in both directions.
  - It asserts T_TE (VV) and T_TM (HH) against `diag_slab_p_pol` of the true thickness to 1e-4 dB.

The three new cases run in 0.05 s together (whole-suite run, 12 threads).

Test diff: 263 insertions, 72 deletions in 6 files (`ray_test_helpers.hpp`, `test_acoustic_state_update.cpp`,
`test_wave_state_update_{branch,chain,sheet,spec}.cpp`).

### 1.2 Against today's library

`test_bin` built from the new test files:

| run | result |
| --- | --- |
| `[wave]` | 181 cases, 165594 assertions; **11 cases fail, 575 assertions** |
| `[acoustic]` | 60 cases, 42391 assertions; **2 cases fail, 372 assertions** |
| whole suite | 1148 cases, 1317534 assertions; **13 cases fail, 947 assertions** |

The failing cases are the 10 restated `[wave]` fixtures and the 3 new cases. Each fails by the nudge, or by the gate
flip in the float case:

| case | today, largest deviation |
| --- | --- |
| *Far from the origin - a traced mass-law wall …*, against the wall at the origin: glass / concrete | 10 m 0.019 / 5.1e-4 dB; 100 m 0.21 / 5.4e-3 dB; 1 km 2.4 / 0.055 dB |
| the same, against the rule | at the origin 3.5e-3 / 5.7e-5 dB (inside 5e-3) |
| *Far from the origin - in single precision …* | 16 of 72 rows by 25.9–26.0 dB (glass 25, 40, 55, 60°; concrete 20, 25, 55, 65°; both types). The other 56 within 0.0034 dB |
| *Far from the origin - a thin dielectric slab …*, summed / followed | origin 8.0e-3 / 5.7e-3 dB; 10 m 0.082 / 0.058; 100 m 0.72 / 0.53; 1 km 2.2 / 2.3 dB |
| *Unentered branches - the displacement budget …* | dS 2.0e-7, dT 2.2e-5, dR 4.2e-4 (against 1e-9), 184 assertions |
| *Unentered branches - the core's loss cap …* | 2.8e-6 (against 1e-9) |
| *Unentered branches - a verdict … near a pole* | 0.505 (against 1e-6): εr 1e4 lossless, tuned to the leg's resonance |
| *One solver - …* | 1.28e-6 and 1.13e-6 (against 1e-6) |
| *Sheet reflect port - …* | 2 assertions |
| *Refracting tracer - … the leg is exact* | 1.38e-3 dB (against 1e-3), 3.4e-3 (against 2e-3) |
| *Phase reference - …* | 3.0e-3 (against 1e-3); 0.0177° (against 0.005°) |
| *Measurement - the mirror behind a second solid …* | 9.5e-7 / 1.0e-5 (against 1e-9); 1.13e-3 and 2.4e-3 dB (against 1e-3 / 2e-3) |
| *Diagnostic - Host layer carried through a matched sheet* | 9.5367e-7 m (against 1e-12), the nudge at x = 0.1 |
| *ray_state_update - accumulated distance ledger* | 1.43051e-6 m (against 1e-12), the nudge at x = 1.5 |

**Every other line of the baselines is unchanged.** `cases.py` compares the old tests against the new, both on today's
library, with `--durations yes`:
- 19 cases change: 18 print different rows or fail as `cases.py` counts them, plus *ray_state_update - accumulated
  distance ledger*, which prints no rows and fails.
- Every one of the 19 holds an edit or calls an edited helper (`p1/movemap.py`, matching full case names).
- The 4 other cases that call `wall_g` or a new helper (*Absorption peak - …*, *Closed face - …*, *Finite plate - side 0
  …*, *Lossy verdict - …*) print identical rows.
- No other case moves a line.
- The moved rows of the passing cases are references printed beside the result, now from the hit point:
  - *Exits and verdicts - …* 24 lines;
  - *Sheet reflection - …* 24;
  - *Measurement - the mirror behind the ray after a back reflection* 230;
  - *Unentered branches - a sheet without the mass column …* 24;
  - *Measurement - … the mass-law part* 15;
  - *Finite plate - …* 20 (the 89.9° traced rows, new).

### 1.3 Against the what-if (W1 + W3″)

The same test files linked with `rmi_w.cpp`, `QD_PREC=17`:

| run | result |
| --- | --- |
| `[acoustic]` | 60 cases, 42391 assertions, all pass |
| `[wave]` | 181 cases, 165594 assertions; **5 cases fail, 260 assertions: the twins** |
| whole suite | 1148 cases, 1317534 assertions; the same 5 cases, 260 assertions |

**The new cases under the fix:**

| case | largest deviation |
| --- | --- |
| *Far from the origin - a traced mass-law wall …*, against the origin | 5.4e-11 dB at 1 km |
| the same, against the rule | 1.75e-3 dB (the face at normal incidence, as at the origin) |
| *Far from the origin - in single precision …* | 0 of 72 rows over 0.05 dB; the largest 0.0037 dB |
| *Far from the origin - a thin dielectric slab …* | 1.2e-5 dB at every distance, summed and followed |

**The twin tables that move in phase 2.** These are `twin()` rows written with `QD_TWIN_WRITE` under the what-if, against
`tests/data/wave_twins.tsv`; 260 of 736 rows, the same as in phase 0:

| table | rows | largest relative change |
| --- | --- | --- |
| *bare tf T* | 84 / 84 | 4.0e-3 |
| *flat branch H* | 56 / 136 | 8.8e-3 |
| *rare sheet F* | 48 / 60 | 6.3e-3 |
| *scalar closed face C* | 12 / 18 | 5.9e-4 |
| *sheet tf H* | 60 / 268 | 1.3e-3 |

**Printed rows the fix moves.** These are new tests, today against the what-if (`cases.py`, `p1/moves_w_full.txt`):
- 76 cases print different rows, 38 `[wave]` (6449 lines) and 38 `[acoustic]` (4855 lines). All pass except the 5 twin
  cases.
- The largest by line count are *Resonant medium - …* (1045 lines), *Bonded mass - the plate at any depth* (967),
  *ray_progress - the bonded plate at any depth* (518), *Leaf - …* (437) and *Measurement - ray_progress and the mirror
  …* (415).
- Phase 2 accounts for them against this list.

### 1.4 A correction to the phase-0 what-if (for phase 2)

The first run of the new EM case on the what-if showed type 1 with every bounce followed off by 0.07 dB at 10 m and
8.8 dB at 1 km. The summed series and type 2 were exact.

- **The cause was in my W1 what-if, not in the decision.** On a reflect pass inside the slab, W1 took the relaunch
  direction to be the written wave direction. For the undeviated tracer, `ray_mesh_interact` mirrors the geometric
  segment, and the two differ. The refracted ledger therefore got `off` instead of `off·cos θ_i / cos θ_t` on every
  followed bounce.
- Phase 0's map compared only the direct path and the summed series for type 1, so it did not show this.
- **The corrected rule:** the refracted ledger's factor is `|g·N| / |dir_next·N|`, with `g` the incoming segment's
  direction for every type but 2 and 5. A mirror has the same normal cosine. For 2 and 5 the factor is 1.
- With it the case reads 1.2e-5 dB at every distance.
- The numbers of report 0.4–0.6 for direct paths and summed series are unchanged. The twins move as measured there
  (identical counts and largest changes).
- Phase 2 implements W1 with this rule.

### 1.5 For phase 2

- **W1 in `ray_state_update`** with the rule of 1.4, and **W3″**: the float tolerance of the `use_vbs` test where the
  current medium has `m > 0`.
- **Expected outcome:**
  - the 13 failing cases pass;
  - `[wave]` fails only the 5 twin tables until `tests/data/wave_twins.tsv` is rewritten with `QD_TWIN_WRITE` (260 rows);
  - the printed rows move in the 76 cases of 1.3;
  - nothing else moves.
- **Run time:** measure as CPU time against today's library (phase 0: +0.3 to +0.9 % for W1 + W2 + W3″).

## 2. Phase 2 — source (2026-10-20)

Accepted (2026-10-19, the manager's rerun of phase 1 confirmed the counts):
- the corrected W1 rule of 1.4;
- the measured exception of *Measurement - the mirror behind a second solid …* (2e-3 dB at its one row), not traced
  further.

Scratch: `build_linux/scratch/prec/p2`. Before the build, today's `test_bin` (today's library with the phase 1 tests)
and `libquadriga.a` were saved there as `tb_today` and `lib_today.a`.

### 2.1 What changed

**`src/ray_mesh_interact.cpp`** (5159 → 5211 lines, +54 / −2), all inside `ray_state_update`:

- **W1, the leg from the hit point** (block `LEG_FROM_HIT_POINT`, written just before the state words).
  - `DIST_ADD` and `DIST_SET` set `ledger_at_relaunch`: true where the ledger ends at the point this event relaunches
    from (a reset, or the leg added up to the hit point), false where it was preloaded to the destination.
  - Where the ray goes on inside a medium (`next_current` not air, the ray alive, a hit), both ledgers take the relaunch
    offset `qd_relaunch_offset(F)`. The geometric ledger takes it as it is.
  - The refracted ledger takes it times `|g·N| / |dir_next·N|`, with `g` the incoming segment's direction and `N` the
    normal of the face relaunched from (the SBS of a co-located pair, else the FBS). The factor is 1 for types 2 and 5.
  - The nudge itself is unchanged.
  - Comments: the ledger declaration says a leg is measured from the hit point. The block says why the factor is what
    it is and that preloaded legs are left alone.
- **W3″, the float tolerance of the bend test.** Where the template is float and the current medium has `m > 0`, the
  `use_vbs` test allows `1e-6 + 8·2⁻²³·max(|O|, |F|, 1) / |O − F|`; elsewhere it stays 1e-6. The comment says why: a held
  wave, and the rounding of a short segment's ends far from the origin.
- **Documentation block of `ray_state_update`:** `acc_dist_geometricN` and `acc_dist_refractedN` now say they include
  the relaunch offset where the ray goes on inside a medium, so the next leg measured from `origN` adds up to the leg
  from the hit point. The refracted one is projected onto the wave's direction, exact for parallel faces.

**`src/ray_progress.cpp`** (+3 / −1): the documentation of `acc_dist` says the same (about 1 µm per metre of coordinate).

**`tests/data/wave_twins.tsv`**, rewritten with `QD_TWIN_WRITE` (`md/wave.md` 0):
- 260 of 736 rows changed, in exactly the 5 tables of 1.3, by exactly the measured largest amounts:

  | table | rows changed | largest relative change |
  | --- | --- | --- |
  | *bare tf T* | 84 / 84 | 4.02e-3 |
  | *flat branch H* | 56 / 136 | 8.83e-3 |
  | *rare sheet F* | 48 / 60 | 6.34e-3 |
  | *scalar closed face C* | 12 / 18 | 5.85e-4 |
  | *sheet tf H* | 60 / 268 | 1.32e-3 |

- Keys and row order are unchanged. The file is byte-identical to the one the what-if wrote (`p1/tw_w.tsv`). The old
  file is kept as `p2/wave_twins_old.tsv`.

The implementation follows the what-if's arithmetic, except that types 2 and 5 add the offset directly where the what-if
multiplied by `cdn/cdn`. No printed line differs because of it (2.2).

### 2.2 The suite

Built without warnings (`cmake --build build_linux --parallel 12`):

| run | result |
| --- | --- |
| `[wave]` | **all pass, 165594 assertions in 181 cases** |
| `[acoustic]` | **all pass, 42391 assertions in 60 cases** |
| whole suite | **all pass, 1317534 assertions in 1148 cases** |

- **The built library equals the corrected what-if, line for line.** The whole suite with `--durations yes`, before the
  twins were rewritten, differs from the what-if's run (`p1/d_w.txt`) in one line only: the binary's name in Catch2's
  banner. After the rewrite, the differences are the failure reports of the 5 twin cases, now passing, and the summary
  line (1874 lines removed, 1 added, durations aside).
- **Printed rows move only in the 76 cases of 1.3.** Against today's library on the same tests (`p1/d_new.txt`),
  `cases.py` finds 76 cases with different rows, 11304 lines: the same set of cases and the same line count per case as
  the what-if (`p2/moves_built.txt`).
- **The 13 cases that failed on today's library pass:** the 10 restated `[wave]` fixtures and the 3 *Far from the origin
  - …* cases. No case fails.
- **`python3 tools/acoustic/check_docs.py` fails one check.** Its case index finds the two new `[acoustic]` cases of
  phase 1 not cited in `md/acoustic.md`. This has been so since phase 1; I did not run the check then. The citations are
  part of the phase 3 documents. Every other check passes.

### 2.3 Run time

CPU time (user + sys) of single-threaded runs, today's library (`p2/tb_today`) against the built one, the same test
files, alternated, 4 pairs under `flock /tmp/bench.lock`, load 2.6–3.9 (`p2/cputime.sh`):

| spec | today, median | built, median | paired ratios | median change |
| --- | --- | --- | --- | --- |
| `[#test_wave_progress]` | 12.21 s | 12.21 s | 1.003, 0.979, 1.014, 1.000 | +0.2 % |
| `[#test_wave_state_update_chain]` | 9.66 s | 9.50 s | 0.989, 0.989, 0.979, 0.979 | −1.6 % |
| `[#test_acoustic_state_update]` | 0.48 s | 0.47 s | 0.959, 1.000, 0.979, 0.979 | −2.1 % |

Today's binary fails 13 cases and prints their reports, which costs it a little. There is no measurable cost of the
change; phase 0 measured +0.3 to +0.9 % for W1 + W2 + W3″.

### 2.4 For phase 3

- **`md/wave.md` 8:** replace the "1e-6 m" note with the decided line of limits.
  - The leg is measured from the hit point at any distance.
  - A body thinner than the nudge (about 1 µm per metre of coordinate) is skipped.
  - Rays within about 3° of grazing can be lost in float or AVX2.
  - Add the rule of use: build scenes in local coordinates near the origin; mm-wave through `ray_progress` within about
    100 m; never raw georeferenced coordinates.
- **`md/acoustic.md` 7:**
  - the finding closed: the 89.9° row was the `wall_g` fixture, and the 0.2 dB was the nudge, now carried by the ledger;
  - the float bend test in mass-law media;
  - the same rule of use and limits;
  - cite the two new `[acoustic]` cases, which makes `check_docs.py` pass.
- **The documentation blocks of `ray_mesh_interact`:** the `origN` relaunch offset, and the limit line where the docs
  state precision.
- **Regenerate the three baselines from the built library**, with the counts of 2.2.

## 3. Phase 3 — documents (2026-10-20)

Phase 2 accepted (the manager checked the twin diff and reran `[wave]` and `[acoustic]`). No new open items.

### 3.1 What changed

**`md/wave.md`** (+24 / −7):

- **0:** the three counts are now `[wave]` 165594 / 181, `[acoustic]` 42391 / 60, whole suite 1317534 / 1148.
- **8:** the note "a traced leg is short by the relaunch offset (1e-6 m …)" is replaced by what is true now.
  - The nudge is 8·2⁻²³ of the hit point's largest coordinate (`qd_relaunch_offset`), about 1 µm per metre.
  - The in-medium ledger counts it, so a traced leg is exact in double at any distance and within float rounding in
    float. The refracted ledger is exact for parallel faces.
  - Before the fix, 4 mm glass at 28 GHz was off by 4.8° at 100 m and 48° at 1 km. The two cases that assert the fix
    are named.
  - The co-located pair's anchoring sentence stays.
  - **One rule of use:** local coordinates near the origin, mm-wave through `ray_progress` within about 100 m, never raw
    georeferenced coordinates.
  - **One line of limits:**
    - a body thinner than the nudge is skipped (a 4 mm pane is lost at 10 km);
    - rays within about 3° of grazing can be lost in float or with AVX2;
    - float position rounding leaves about 3.8° at 28 GHz at 1 km.
  - The large turned faces in float: one line, recorded, no item.
- **11 (test instruments):** `bf::walk_far`, `qd_leg_start` (a slab's thickness from the leg's start, never from
  `origN`) and `qd_relaunch_nudge`.

**`md/acoustic.md`** (+47 / −9):

- **Start:** the `[acoustic]` count is now 42391 / 60.
- **7:** the precision finding is closed with its numbers.
  - The relaunch offset was missing from the leg. 4 mm glass passed 0.2 dB more at 100 m (−31.454 against −31.658 dB at
    500 Hz) and 2.4 dB more at 1 km. Now it passes what it passes at the origin, to 5.4e-11 dB up to 1 km.
  - The 89.9° row was `wall_g`'s first segment ending inside the wall. That is repaired, and *Finite plate - …* now
    traces its solid to 89.9°.
  - **The false −26 dB of a float mass-law wall is closed**, with its numbers: 16 of 72 rows at 100 m; through
    `ray_progress` only on segments of a few metres.
  - **The rule of use and the limits** follow, as in `md/wave.md` 8.
- **10:**
  - **"Taken (Stephan, 2026-10-19), after the precision study"**: W1 with the corrected rule and the twin move; W3″; W2
    dropped; `wall_g` repaired; one rule of use and one line of limits, no new items.
  - **"Built (2026-10-20)"**: `src/ray_mesh_interact.cpp` 5211 lines against 5159; the `acc_dist` documentation; three
    new cases; the restated fixtures; the twins (260 rows); the counts; the run equal to the what-if; 76 cases with
    moved rows; CPU time unchanged.
  - The grazing decision's "a separate investigation (7)" now says "closed 2026-10-20, below".
- **Appendix:** the two new `[acoustic]` cases, under sections 7 and 10.

### 3.2 Checks

- The three baselines were regenerated with the 12-thread commands of `md/wave.md` 0:

  | baseline | result |
  | --- | --- |
  | `test_wave.txt` | All tests passed (165594 assertions in 181 test cases) |
  | `test_acoustic.txt` | All tests passed (42391 assertions in 60 test cases) |
  | `test.txt` | All tests passed (1317534 assertions in 1148 test cases) |

- Each equals the phase 2 run of the built library line for line (durations aside).
- One other `test_bin` process was on the machine while the baselines ran. Their output is deterministic and no timing
  was taken from them.
- **`python3 tools/acoustic/check_docs.py` passes every check:**
  - tables;
  - 102 cited keys (the two known gaps of the review index as before);
  - 60 `[acoustic]` cases, all named in `md/acoustic.md`;
  - the four quoted counts against the baselines;
  - spelling.

### 3.3 The round, in sum

| phase | content |
| --- | --- |
| 0 | the two findings reproduced and explained: the relaunch nudge missing from the leg, and a fixture whose first segment ended inside the wall; three more mechanisms found; W1 / W2 / W3″ measured as what-ifs |
| 1 | the fixtures restated to the hit point, `wall_g` repaired, three new cases |
| 2 | W1 (with the rule corrected in 1.4) and W3″ built; the twins rewritten |
| 3 | the documents |

- Not committed: the source, the tests, the twins, both documents and this report are in the working tree for
  Stephan's commit, with `md/acoustic_manager.md` (the manager's edits) and the brief.
