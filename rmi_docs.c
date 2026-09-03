/*!MD
# ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media.
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`).
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule); front-side hit with
  FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air; FBS=SBS with opposing normals → media-to-media.
- With `compact = true` (default), rays with `fbs_ind = 0` (no interaction) are omitted from output,
  so `n_rayN ≤ n_ray`; with `compact = false` they are kept as transparent pass-throughs (`n_rayN = n_ray`).
- Output direction encoding (spherical/Cartesian) matches input `tridir` format.
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves).
- Thin sheets: a material with `thk > 0` marks its faces as a **sheet** — a single face carrying an
  authored thickness instead of a traced interior. Each crossing is one atomic entry + traversal + exit,
  with the internal multi-bounce series resolved in closed form. `thk = 0`, or no `thk` column, keeps the
  classic behavior where thickness comes from the geometry and the [[ray_state_update]] state machine.
  Modeling notes:
  - Two coincident sheets of the *same* material are collapsed to a single crossing. Exporters emit
    back-to-back face pairs to make geometry double-sided, and treating such a pair as a two-layer
    laminate would double both the thickness and the loss. Author a genuine laminate as two *different*
    materials, or as one material with the summed `thk`.
  - Stack depth is capped at two faces, because the FBS/SBS intersector returns two. A third layer is
    hit as a separate FBS after the relaunch and concatenates incoherently, and `solid + sheet + solid`
    is already three faces. Scenes have to be designed around this.
  - A closed solid built from a sheet material is legal, not an error: it becomes a hollow box, with
    sheet loss on entry, the embedding medium inside, and sheet loss on exit.
- Types 3–5 (scalar) use a single TE-only coefficient (acoustic simulation with impedance-mapped
  materials, `ε` derived from `Z`); total internal reflection is handled as in the EM path
  (`snell·sinθ ≥ 1` ⇒ unit reflection, refraction collapses to undeviated transmission).
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Declaration:
```
void quadriga_lib::ray_mesh_interact(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *mesh,
    const arma::uvec *mtl_ind,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::u32_vec *fbs_ind,
    const arma::u32_vec *sbs_ind,
    const arma::Mat<dtype> *trivec = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    arma::Mat<dtype> *fbsN = nullptr,
    arma::Mat<dtype> *sbsN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Col<dtype> *fbs_angleN = nullptr,
    arma::Col<dtype> *thicknessN = nullptr,
    arma::Col<dtype> *edge_lengthN = nullptr,
    arma::Mat<dtype> *normal_vecN = nullptr,
    std::vector<uint8_t> *out_typeN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    bool compact = false,
    arma::u32_vec *ray_indN = nullptr);
```

## Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`.
  0 = face has no material (air). NULL → all faces treated as air.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]);
  each value has length `n_mtl`. NULL → air defaults used.
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`compact`** *(optional)* — If `true`, no-hit rays are dropped and `n_rayN ≤ n_ray`. If
  `false` (default), all rays are kept (`n_rayN = n_ray`) and no-hit rays are written as a transparent pass-through
  (gain 1, identity `xprmat`, `out_type = 0`).

## Outputs:
- **`origN`** — New origins after interaction, nudged off the face along the travel direction by ~8 float ULP
  at the interaction-point coordinate magnitude; `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`gainN`** — Linear power gain of the interaction. Derived from `xprmatN`, not an independent factor; `[n_rayN]`
- **`xprmatN`** — Polarization transfer matrix; not normalized: the entries are amplitude coefficients
  that already carry the interaction gain. For types 0–2: interleaved complex, col-major
  `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, i.e. the Jones matrix `M = [VV VH; HV HH]` acting as
  `e_out = M · e_in`; the incoming V/H basis is built from the incident direction, the outgoing basis from
  `path_dirN` (`eH = ẑ × k̂` normalized, `eV = eH × k̂`); `[8, n_rayN]`. For types 3–5 (scalar): `[Re Im]`
  where `Re + jIm` is the scalar pressure coefficient; `[2, n_rayN]`.
  - *Included:* TE/TM interface coefficients (with the lumped `interface_gain` folded in for the transmission
    classes 1/2/4/5) and the incidence-plane orientation.
  - *Excluded:* in-medium attenuation and excess phase — added by [[ray_state_update]] — and FSPL / spreading
    loss, which is never applied here or downstream.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`fbs_angleN`** — Incidence angle at FBS in rad; `[n_rayN]`
- **`thicknessN`** — Geometric FBS-to-SBS distance; `[n_rayN]`. Unrelated to the material `thk` column:
  this is what the mesh measured, `thk` is what the material table authored.
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (`qd::bits<uint8_t>`); `[n_rayN]`<br><br>
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag — meaning depends on bits 2 and 6, see below            |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |
   |   6  | Thin-sheet flag (at least one of FBS/SBS is a `thk > 0` material)       |
   Bit 1 by context:<br><br>
   | Context                                     | Bit 1 = 1 means                               |
   | ------------------------------------------- | --------------------------------------------- |
   | Solid (bit 6 = 0)                           | Front-side hit on FBS (o→i, or M2 hit first)  |
   | Isolated sheet (bit 6 = 1, bit 2 = 0)       | Front-side hit on FBS; informational only     |
   | Sheet glued to a solid (exactly one sheet)  | Front-side hit on the **solid** face          |
   | Two co-located sheets                       | FBS traversed first (0 = SBS traversed first) |
   Reachable composite values (add 32 for the total-reflection variant):<br><br>
   | Code  |  TIR  | Description                                                                        |
   | :---: | :---: | ---------------------------------------------------------------------------------- |
   |   0   |   —   | No hit                                                                             |
   |   1   |  33   | Single hit, inside→outside (exit)                                                  |
   |   3   |  35   | Single hit, outside→inside (entry)                                                 |
   |   5   |  37   | Media-to-media, M1 (current, back) hit first                                       |
   |   7   |  39   | Media-to-media, M2 (next, front) hit first                                         |
   |  13   |  45   | Overlapping faces, inside-inside→outside                                           |
   |  15   |  47   | Overlapping faces, outside→inside-inside                                           |
   |  21   |  53   | Corner hit, inside→outside→inside                                                  |
   |  23   |  55   | Corner hit, outside→inside→outside                                                 |
   |  29   |  61   | Corner hit, inside-inside→outside                                                  |
   |  31   |  63   | Corner hit, outside→inside-inside                                                  |
   |  65   |   —   | Isolated sheet, back-side hit; or a sheet at a corner with SBS dropped             |
   |  67   |   —   | Isolated sheet, front-side hit; or a sheet at a corner with SBS dropped            |
   |  69   |  101  | Glued exit (solid→sheet→air); or a stack with SBS first; or a collapsed duplicate  |
   |  71   |  103  | Glued entry (air→sheet→solid); or a stack with FBS first; or a collapsed duplicate |
   |  79   |   —   | Two-sheet stack with an ill-posed order (geometric FBS→SBS fallback was used)      |
   Codes 69 and 71 are shared by three cases, deliberately: [[ray_state_update]] builds its own material
   set and holds both `mtl_ind_fbs` and `mtl_ind_sbs`, so it can re-derive sheet-ness per face and material
   identity on the pair. The full case is `(composite code, is_sheet(FBS), is_sheet(SBS), same material)`.
   Bit 6 alone is a hint meaning *at least one face is a sheet*; the re-derivation is authoritative. Total
   reflection can only co-occur with bit 6 for the glued cases, where the media either side of the sheet
   differ. Bit 7 is unused.
- **`path_dirN`** — Refraction-correct path direction: mirror for types 0/3, Snell direction for types 1/2/4/5; `[n_rayN, 3]`.
  For undeviated transmission (types 1/4) this is the *refracted* direction, which differs from the geometric continuation
  (along the incoming ray) used for `origN`/`destN`; it lets downstream code recover the true transmission angle.
- **`ray_indN`** — 0-based input ray index for each output ray (inverse of the internal compaction map; order-preserving); `[n_rayN]`

## See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [[obj_file_read]] (for loading `mesh` and `mtl_prop` from OBJ file)
- [[ray_state_update]] (inside/outside state machine)
- [[icosphere]] (for generating beams)
- [[ray_triangle_intersect]] (for computing FBS and SBS positions)
- [[ray_point_intersect]] (for calculating beam interactions with sampling points)
MD!*/
