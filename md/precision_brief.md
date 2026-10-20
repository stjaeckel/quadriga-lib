# Brief — precision of a traced leg through a solid (far from the origin, near grazing)

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/precision_report.md` (create it), one section per phase, so the next session can pick
them up.

## Why

Found while building the finite-panel fixtures (`md/grazing_report.md` 1.3, side findings; recorded in `md/acoustic.md`
7), not investigated, cause unknown:

- The same 4 mm traced glass pane (a solid of two faces, the mass column set) passes **0.2 dB more** at normal
  incidence when the ray crosses it at y = −100 m than at y = 0.37 m: −31.454 against −31.658 dB at 500 Hz.
- A 152 mm traced concrete wall crossed at **89.9°** (an 87 m path inside the wall) misses the library's own
  infinite-plate rule by −0.007 / −0.027 / −0.10 / −1.17 / −3.42 dB at 31.5 / 63 / 125 / 500 / 2000 Hz; at 89.5° it
  holds to 1e-4 dB.

The fixtures avoid it (`r1::wall_g` in `tests/catch2_tests/test_acoustic_state_update.cpp` centres the crossing near
the origin and stops the traced solid at 89.5°). A real scene does not: buildings are modelled in site coordinates
hundreds of metres or kilometres from the origin, and a ray tracer meets walls at every angle. The question is
**where the precision is lost, how large the error is over the ranges a scene meets, what it does to EM (the
library's primary use) and to acoustics, and what a fix would cost**. Stephan's rules apply: EM must be correct;
acoustic fixes may be approximate but must not harm EM; the per-ray state does not grow.

## Facts at the start

- Last commit de60194 (the finite panel). `git status` should be clean apart from this brief and
  `md/acoustic_manager.md`; if anything under `src/`, `tests/` or `md/acoustic.md` is modified, stop and ask Stephan.
- Baselines (`md/wave.md` 0, with the 12-thread commands): `[wave]` 165258 assertions in 180 cases, `[acoustic]` 41523
  in 58, whole suite 1316330 in 1145. `src/ray_mesh_interact.cpp` 5159 lines. `python3 tools/acoustic/check_docs.py`
  passes.
- Where the leg is formed (find by name; line numbers drift):
  - `src/ray_triangle_intersect.cpp`: its documentation says "Internal computations always use single precision for
    AVX2 and CUDA kernels; only GENERIC has `double` support"; `use_kernel` 0 = auto (CUDA from 500 rays where
    available, else AVX2, else GENERIC), 1 = GENERIC, 2 = AVX2. The first and second hit points (FBS, SBS) come from
    there. Kernels: `ray_triangle_intersect_avx2.cpp`, `_generic.cpp`, `_cuda.cu`.
  - `src/ray_mesh_interact.cpp`: `ray_mesh_interact` (the event at a face, the in-medium distances from the hit
    points), `ray_state_update` (the accumulated distances `acc_dist_*`, `DIST_ADD` / `DIST_SET`, the cosines), and
    `Material::medium_gain`, where the mass term takes the surface-mass surrogate `dist_geo · cos θ` (the plate's
    thickness reconstructed from the leg's length and cosine) and EM's in-medium loss and phase take the distance.
  - `src/ray_progress.cpp`: the ray state between events; the accumulated distances are stored as `float`
    (`arma::fvec acc_geo`, `acc_geoN`). The per-ray state must not grow (about 150 M rays; every flag bit is spent).
- How sensitive the mass term is: 20 dB per decade of the thickness, so a 1 % thickness error is 0.086 dB; the 0.2 dB
  above is a 2.3 % error on 4 mm, about 0.1 mm. Near grazing the thickness is `dist · cos θ` with `cos 89.9°` 0.0017,
  so a relative error in the cosine or the length becomes the same relative error in the thickness.
- The probe instruments: the test file's walker (`bf::walk` in `tests/catch2_tests/ray_test_helpers.hpp`,
  `diag_ray_at`), `r1::wall` / `r1::wall_g`, `ub::build` (faces of 400 m); `tools/acoustic/band_probe.cpp` (a band
  through `ray_progress`); a scratch probe that includes a test file (`md/wave.md` 11; give it a test spec).

## Hard rules

- Phase 0 is analysis only: no edits outside `build_linux/scratch/prec` and `md/precision_report.md`. Source, tests
  and documents only after Stephan's decision.
- EM first: every effect is measured for EM types (0, 1, 2; TE and TM) as well as for the scalar types, and a fix must
  not move an EM row without a reason stated and measured. `[wave]` and `tests/data/wave_twins.tsv` must not move
  unless Stephan decides a wave change.
- The per-ray state does not grow (no new per-ray arrays in `ray_progress`, no wider types for existing ones, unless
  Stephan decides it after seeing the cost in bytes per ray).
- Measure, don't compute by hand: every number from a run or a script; say which. Distinguish the kernel (GENERIC,
  AVX2; CUDA only if a GPU is free: `nvidia-smi`, `flock /tmp/gpu.lock`), the template (float, double) and the caller
  (`ray_mesh_interact` / `ray_state_update` directly, or `ray_progress`).
- **Resources** (another coding session shares this machine): at most 12 threads, compiler and tests alike
  (`cmake --build build_linux --parallel 12`, `OMP_NUM_THREADS=12` or fewer; numpy pools `OMP_NUM_THREADS=1` per
  worker, at most 12 workers); never two suites at once; ASAN, AFL, sanitizers and other heavy tools only when the
  task justifies them, never by default; timings as CPU time or alternated runs with the load stated, without waiting
  for an idle machine. `timeout` on every probe.
- Do not commit; Stephan commits. Each phase ends with a report and a stop.

## Phase 0 — analysis (scratch and the report only)

Fresh scratch folder `build_linux/scratch/prec`. A plain what-if copy of `src/ray_mesh_interact.cpp` (and of
`ray_triangle_intersect` if needed) first: it must reproduce the baselines.

1. **Reproduce** the two findings with a scratch probe, the same walls and rays, and confirm the numbers.
2. **Localize.** Follow one ray through the chain and print every intermediate in double next to what the library
   carries: the hit points from each kernel (GENERIC, AVX2) against an exact double intersection; the leg length
   and the cosine `ray_mesh_interact` and `ray_state_update` form; the accumulated distances; the thickness
   surrogate `dist_geo · cos θ` and the mass term. Which step loses the digits, and by how much (absolute and
   relative)?
3. **Map the error** over what a scene meets, for the leg's length, the reconstructed thickness, the mass term
   (acoustic), and for EM the in-medium loss (dB) and the in-medium phase (degrees at 1, 10, 28 and 60 GHz):
   - the distance of the crossing from the origin: 0, 1, 10, 100, 1000, 10000 m;
   - the incidence angle: 0° to 89.99° (fine near grazing);
   - the thickness: 4 mm, 12.5 mm, 152 mm, 300 mm;
   - the face size (the mesh's 400 m faces against a wall-sized triangle) where it matters;
   - the template (float, double), the kernel, and the caller (directly and through `ray_progress`).
   Also the free-space segment between events, for EM path length and phase, in the same coordinate ranges, so the
   in-medium error can be set against what every EM path already carries (say whether that is documented).
4. **What it does in practice.** For EM: a traced dielectric slab (a wall of the EM material table, `data/materials_rf.csv`)
   at 10 / 28 GHz far from the origin, its transmission and the coherent sum of its internal bounces (the Airy factor
   depends on the in-slab phase). For acoustics: the 13 reference constructions' walls as traced solids at 100 m and
   1 km from the origin, diffuse average and per angle. Say where the error exceeds the guideline (±3 dB transmission,
   ±0.05 in absorption, a phase error that changes a coherent sum by more than about 0.1 dB).
5. **Fix candidates**, each with its cost and its effect on the numbers of 3 and 4, measured as a what-if where cheap:
   for example (a) the in-medium leg formed from double hit points recomputed against the hit triangle's plane, (b)
   the thickness taken from the distance between the two faces' planes along the normal instead of `length · cos θ`,
   (c) a local origin per event (positions relative to the hit triangle), (d) the GENERIC double kernel for the
   second hit only, (e) documenting the limit and a rule of use (scenes near the origin). Say what each needs at the
   event, whether it touches the per-ray state, `ray_triangle_intersect`, `ray_progress` or other public APIs, its
   lines, its run-time cost (CPU time, alternated), and which `[wave]` and `[acoustic]` rows it moves.
6. **Recommendation** with the reason (EM correctness first, then less code, then a variant that closes the item over
   one that leaves a mode open), and the decisions Stephan has to take. Stop.

## Phase 1 — fixtures (after Stephan's decision)

- Cases that state the decided behavior: a traced wall crossed far from the origin and near grazing against the same
  wall at the origin (acoustic and EM), with references written in the test from closed forms; the fixtures that
  centre their crossings (`r1::wall_g`) restated if the fix makes that unnecessary. Run against today's library: the
  new cases fail with the expected deviations, every other line of the three baselines unchanged. Stop.

## Phase 2 — source

- The decided fix. Whole suite: every case passes; `[wave]` identical (or exactly the decided rows) and the twins
  unchanged; every moved row accounted for; run time as CPU time. Line counts. Stop.

## Phase 3 — documents

- `md/acoustic.md` 7 (the finding closed or recorded as a limitation with its numbers), `md/wave.md` where an EM
  statement changes, the public documentation blocks in the source if an API's behavior or precision statement
  changes; regenerate the baselines; `check_docs.py` passes. Stop.

## Traps

- The traps of `md/wave.md` 11 and `md/acoustic_manager.md` 9: run binaries from the repository root, one suite at a
  time, `pkill -f` kills its own shell, `python3 -I`, a probe that includes a test file runs every case of it unless
  given a test spec.
- The auto kernel picks CUDA from 500 rays where a GPU is present and AVX2 otherwise: fix the kernel explicitly in
  every probe and say which.
- A float printed with `%.6f` hides the digits in question; print hit points and distances with `%.17g`.
- `ub::build` makes 400 m faces: a crossing far along the face and a crossing far from the origin are two different
  tests; separate them.
- Scratch is not kept: the report states every number it relies on.
