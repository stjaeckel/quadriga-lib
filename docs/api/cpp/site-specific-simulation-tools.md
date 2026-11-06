# Site-specific simulation tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-site-specific-simulation-tools

| Function | Summary |
| --- | --- |
| [calc_diffraction_gain](#calc_diffraction_gain) | Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh |
| [colormap](#colormap) | Generate a colormap matrix with RGB values |
| [combine_irs_coord](#combine_irs_coord) | Combine path interaction coordinates for IRS-assisted TX → RX channels |
| [coord2path](#coord2path) | Convert path interaction coordinates into FBS/LBS positions, path length, and angles |
| [cube](#cube) | Construct a triangulated cube mesh |
| [generate_diffraction_paths](#generate_diffraction_paths) | Generate elliptic propagation paths and weights for diffraction gain estimation |
| [icosphere](#icosphere) | Construct a geodesic polyhedron from recursive icosahedron subdivision |
| [interface_gain](#interface_gain) | Linear gain of the lumped penetration loss of entering a material |
| [medium_gain](#medium_gain) | Linear gain of a ray traversing a homogeneous lossy medium |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Write a triangular mesh to a Mitsuba 3 XML scene file |
| [obj_file_read](#obj_file_read) | Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials |
| [obj_file_write](#obj_file_write) | Write a triangulated Wavefront .obj (and .mtl) file |
| [obj_overlap_test](#obj_overlap_test) | Detect overlapping 3D objects in a triangular mesh |
| [path_to_tube](#path_to_tube) | Convert a 3D path into a tube surface mesh for visualization |
| [plane](#plane) | Construct a triangulated plane mesh |
| [point_cloud_aabb](#point_cloud_aabb) | Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud |
| [point_cloud_segmentation](#point_cloud_segmentation) | Reorganize a point cloud into spatial sub-clouds for efficient processing |
| [point_cloud_split](#point_cloud_split) | Split a point cloud into two sub-clouds along a spatial axis |
| [point_inside_mesh](#point_inside_mesh) | Test whether 3D points are inside a triangle mesh using raycasting |
| [ray_commit](#ray_commit) | Commit the paths of a launch configuration that reach a receiver |
| [ray_init](#ray_init) | Seed a sphere of rays from a point source |
| [ray_mesh_interact](#ray_mesh_interact) | Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces |
| [ray_point_intersect](#ray_point_intersect) | Calculate intersections of ray beams with points in 3D space |
| [ray_progress](#ray_progress) | Advance a ray set by one interaction, spawning reflected, transmitted, and subdivided rays |
| [ray_state_update](#ray_state_update) | Per-ray medium state machine with closed-form thin-slab (Fabry-Pérot) resolution |
| [ray_subdivide_flag](#ray_subdivide_flag) | Compute the per-ray subdivision decision from the beam-tube footprint at the first bounce |
| [ray_triangle_intersect](#ray_triangle_intersect) | Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm |
| [refractive_index](#refractive_index) | Real refractive index of a homogeneous medium |
| [subdivide_rays](#subdivide_rays) | Subdivide ray beams into four smaller sub-beams |
| [subdivide_triangles](#subdivide_triangles) | Subdivide triangles into smaller triangles |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing |
| [triangle_mesh_split](#triangle_mesh_split) | Split a 3D triangular mesh into two sub-meshes along a given axis |
| [write_png](#write_png) | Write a data matrix to a color-coded PNG file |
| [xpr_update](#xpr_update) | Thread a polarization transfer (Jones) matrix along a ray path |

---
<a id="calc_diffraction_gain"></a>
## calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#calc_diffraction_gain) · [Python](../python/site-specific-simulation-tools.md#calc_diffraction_gain)

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients, generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping triangles whose bounding box does not intersect the TX-RX path

### Declaration:
```cpp
void calc_diffraction_gain(
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &dest,
    const arma::Mat<dtype> &mesh,
    const arma::uvec &mtl_ind,
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    dtype center_frequency,
    int lod = 2,
    arma::Col<dtype> *gain = nullptr,
    arma::Mat<dtype> *xprmat = nullptr,
    arma::Cube<dtype> *coord = nullptr,
    int verbose = 0,
    const arma::u32_vec *sub_mesh_index = nullptr,
    int use_kernel = 0,
    int gpu_id = 0,
    bool scalar_mode = false,
    double thin_slab_threshold = 0.0,
    bool apply_fspl = false,
    bool apply_phase = false);
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face, 0 = no material (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl`
- **`center_frequency`** — Center frequency in [Hz]. In EM mode the electromagnetic frequency. In scalar (acoustic)
  mode the EM-equivalent frequency `f_eq = f_acoustic · c_light / c_sound` (c_light = 299 792 458 m/s,
  c_sound = 342.76271 m/s), so that `c_light / f_eq` is the acoustic wavelength; this is the frequency unit of
  [ray_mesh_interact](#ray_mesh_interact), [ray_progress](#ray_progress) and [generate_diffraction_paths](#generate_diffraction_paths) in scalar mode. Passing the acoustic
  frequency itself gives a wavelength 874 636 times too long. The MATLAB and Python wrappers take the acoustic
  frequency and convert it.
- **`lod`** — Level of detail (0–7), controls `n_path` and `n_seg`; 0 = direct ray only (`n_path = 1`, `n_seg = 1`);
  1–7 as in the table of [generate_diffraction_paths](#generate_diffraction_paths)
- **`verbose`** — Verbosity level
- **`sub_mesh_index`** — 0-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation); `[n_mesh]`
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). Selects
  interaction type passed to [ray_mesh_interact](#ray_mesh_interact) (4 vs. 1).
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold on the round-trip in-slab amplitude `ρ`: a series is summed where `ρ ≥ eps`; 0 = resolve always (default); `eps ≥ 1` re-emits every series whose `ρ` is below one, and a round trip at or above one is always summed; see [ray_state_update](#ray_state_update)
- **`apply_fspl`** — If `true`, folds the geometric spreading loss into `gain`, so that `gain` becomes an
  absolute propagation gain instead of a value relative to FSPL. Default `false`. The reference distance is
  the direct TX-RX distance, matching the `path_length` definition used by [qrt_file_read](channel-functions.md#qrt_file_read) for a transmission-only 
  path. In EM mode the loss is `32.45 + 20·log10(f_GHz) + 20·log10(d_m)` [dB]; in scalar mode it is 
  `20·log10(d_m) + α(f)·d_m` [dB], with `α` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm, evaluated at the
  acoustic frequency `f = center_frequency · c_sound / c_light`.
- **`apply_phase`** — If `true`, multiplies `xprmat` by the propagation phase term `exp(-j·2π·d/λ)`, using the
  same direct TX-RX distance `d`. Default `false`. The wavelength is `λ = c_light / center_frequency` in both
  modes; in scalar mode this is the acoustic wavelength because `center_frequency` is the EM-equivalent
  frequency (see above). Since the term has unit
  magnitude, `xprmat` stays normalized. This is the phase contribution that
  [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) would otherwise add from `path_length`; enabling it here gives the complete
  omni-radiator coefficient without running the channel generator.

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `[n_pos]`. Relative to FSPL by default,
  i.e. bounded by 0 and 1; absolute propagation gain when `apply_fspl = true`.
- **`xprmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase. The `gain` output holds the power.
  For EM mode: interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx), the row order
  of `M` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical); `[8, n_pos]`;
  For scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`.
  With `apply_phase = true`, the propagation phase `exp(-j·2π·d/λ)` is applied on top; the normalization
  is unaffected because that term has unit magnitude.
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]` with `n_seg` from `lod`
  (e.g. `[3, 2, n_pos]` for lod 2, `[3, 9, n_pos]` for lod 7, `[3, 0, n_pos]` for lod 0)

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)
- [ray_mesh_interact](#ray_mesh_interact) (used for media interactions)
- [qrt_file_read](channel-functions.md#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

---
<a id="colormap"></a>
## colormap
Generate a colormap matrix with RGB values

Not in: MATLAB, Python

- Returns a `[64, 3]` or `[256, 3]` matrix of unsigned char RGB values (range 0–255)
- Available maps: `"jet"`, `"parula"`, `"winter"`, `"hot"`, `"turbo"`, `"copper"`, `"spring"`, `"cool"`, `"gray"`, `"autumn"`, `"summer"`

### Declaration:
```cpp
arma::uchar_mat quadriga_lib::colormap(std::string map, bool high_res = false);
```

### Inputs:
- **`map`** — Name of the colormap
- **`high_res`** *(optional)* — If true, returns 256 rows instead of 64

### Returns:
- RGB colormap matrix; `[64, 3]` or `[256, 3]`

---
<a id="combine_irs_coord"></a>
## combine_irs_coord
Combine path interaction coordinates for IRS-assisted TX → RX channels

Not in: MATLAB, Python

- Merges two propagation segments (TX → IRS and IRS → RX) into complete path interaction coordinate sequences
- Interaction coordinates use a compressed format: `no_interact` counts interactions per path, `interact_coord` stores all coordinates sequentially in path order
- Each combined path appends segment 1 coordinates (optionally reversed) then the IRS position then segment 2 coordinates (optionally reversed); reversing affects coordinate order only, not endpoint positions
- Output contains at most `n_path_1 × n_path_2` paths; `active_path` (typically the return value of [get_channels_irs](channel-generation-functions.md#get_channels_irs)) reduces this to active combinations only
- Typically used after [get_channels_irs](channel-generation-functions.md#get_channels_irs) to produce interaction data for path visualization (e.g. in Blender) via [coord2path](#coord2path)

### Declaration:
```cpp
void quadriga_lib::combine_irs_coord(
    dtype Ix, dtype Iy, dtype Iz,
    const arma::u32_vec *no_interact_1,
    const arma::Mat<dtype> *interact_coord_1,
    const arma::u32_vec *no_interact_2,
    const arma::Mat<dtype> *interact_coord_2,
    arma::u32_vec *no_interact,
    arma::Mat<dtype> *interact_coord,
    bool reverse_segment_1 = false,
    bool reverse_segment_2 = false,
    const std::vector<bool> *active_path = nullptr);
```

### Inputs:
- **`Ix, Iy, Iz`** — IRS position in Cartesian coordinates
- **`no_interact_1`** — Number of interaction points per path for segment 1 (TX → IRS); `[n_path_1]`
- **`interact_coord_1`** — Interaction coordinates for segment 1; `[3, sum(no_interact_1)]`; any other size throws
  `std::invalid_argument` (an empty matrix is accepted when `sum(no_interact_1)` is 0)
- **`no_interact_2`** — Number of interaction points per path for segment 2 (IRS → RX); `[n_path_2]`
- **`interact_coord_2`** — Interaction coordinates for segment 2; `[3, sum(no_interact_2)]`; same check as for
  `interact_coord_1`
- **`reverse_segment_1`** *(optional)* — If `true`, reverses interaction coordinate order within segment 1
- **`reverse_segment_2`** *(optional)* — If `true`, reverses interaction coordinate order within segment 2
- **`active_path`** *(optional)* — Boolean mask selecting path combinations to include; pass the return value of [get_channels_irs](channel-generation-functions.md#get_channels_irs) directly; `[n_path_1 × n_path_2]`

### Outputs:
- **`no_interact`** — Number of interaction points per combined path; `[n_path_irs]`
- **`interact_coord`** — Combined interaction coordinates for all output paths; `[3, sum(no_interact)]`

### See also:
- [get_channels_irs](channel-generation-functions.md#get_channels_irs) (generates `active_path` and channel coefficients for IRS channels)
- [coord2path](#coord2path) (consumes interaction coordinates to compute angles and path geometry)

---
<a id="coord2path"></a>
## coord2path
Convert path interaction coordinates into FBS/LBS positions, path length, and angles

Not in: MATLAB, Python

- `no_interact` is a vector of length `n_path` with the number of interactions per path
- `interact_coord` stores all coordinates concatenated in path order, size `[3, sum(no_interact)]`
- LOS paths (`no_interact[i] == 0`) get a virtual FBS/LBS at the midpoint between TX and RX
- Set `reverse_path = true` to swap TX/RX and reverse all interaction sequences

### Declaration:
```cpp
void quadriga_lib::coord2path(
    dtype Tx, dtype Ty, dtype Tz,
    dtype Rx, dtype Ry, dtype Rz,
    const arma::u32_vec *no_interact,
    const arma::Mat<dtype> *interact_coord,
    arma::Col<dtype> *path_length = nullptr,
    arma::Mat<dtype> *fbs_pos = nullptr,
    arma::Mat<dtype> *lbs_pos = nullptr,
    arma::Mat<dtype> *path_angles = nullptr,
    std::vector<arma::Mat<dtype>> *path_coord = nullptr,
    bool reverse_path = false);
```

### Inputs:
- **`Tx, Ty, Tz`** — Transmitter position in Cartesian coordinates
- **`Rx, Ry, Rz`** — Receiver position in Cartesian coordinates
- **`no_interact`** — Number of interactions per path (0 = LOS); must not be null; `[n_path]`
- **`interact_coord`** — Interaction coordinates in path order; must not be null, must have 3 rows; `[3, sum(no_interact)]`
- **`reverse_path`** (optional) — If `true`, swaps TX/RX and reverses interaction sequences

### Outputs:
- **`path_length`** (optional) — Absolute path length TX to RX; `[n_path]`
- **`fbs_pos`** (optional) — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** (optional) — Last-bounce scatterer positions; `[3, n_path]`
- **`path_angles`** (optional) — Departure and arrival angles {AOD, EOD, AOA, EOA}; `[n_path, 4]`
- **`path_coord`** (optional) — Full path coordinates including TX and RX; vector of `n_path` matrices, each `[3, n_interact+2]`

---
<a id="cube"></a>
## cube
Construct a triangulated cube mesh

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#cube) · [Python](../python/site-specific-simulation-tools.md#cube)

- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1) before scaling
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 * n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with [obj_file_write](#obj_file_write); a negative scale mirrors
  the cube, and when the product of the scales is negative the winding is swapped so that the normals stay outward
  (as Blender's "Apply Scale")
- Scale, Euler rotation, and translation are applied in that order (scale -> rotate -> translate)

### Declaration:
```cpp
arma::Mat<dtype> quadriga_lib::cube(
    const arma::vec &scale = {1.0},
    const arma::vec &rotation = {0.0, 0.0, 0.0},
    const arma::vec &location = {0.0, 0.0, 0.0},
    const arma::uword n_div = 1);
```

### Inputs:
- **`scale`** — Length 1 scales all axes uniformly; length 3 scales `{x,y,z}` independently; empty = 1 (no scaling)
- **`rotation`** — Euler angles in [rad] about {x,y,z}, applied as `R = Rz*Ry*Rx` (Blender XYZ);
  length 3 or empty (no rotation)
- **`location`** — Translation {x,y,z} in [m]; length 3 or empty (origin)
- **`n_div`** — Number of subdivisions per edge; results in `12 * n_div^2` triangles

### Returns:
- Triangle mesh; each row holds `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[12 * n_div^2, 9]`

### See also:
- [icosphere](#icosphere)
- [subdivide_triangles](#subdivide_triangles)
- [obj_file_write](#obj_file_write)

---
<a id="generate_diffraction_paths"></a>
## generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#generate_diffraction_paths) · Not in: Python

- Generates inputs required by [calc_diffraction_gain](#calc_diffraction_gain): elliptic-arc paths sampling the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so `sum(prod(weights,3),2) = 1`

### Declaration:
```cpp
void generate_diffraction_paths(
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &dest,
    dtype center_frequency,
    int lod,
    arma::Cube<dtype> &ray_x,
    arma::Cube<dtype> &ray_y,
    arma::Cube<dtype> &ray_z,
    arma::Cube<dtype> &weight);
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail; controls `n_path` and `n_seg`:
   | `lod` | `n_path` | `n_seg` | Note  |
   | ----- | -------- | ------- | ----- |
   | 1     | 7        | 3       | -     |
   | 2     | 19       | 3       | -     |
   | 3     | 37       | 4       | -     |
   | 4     | 61       | 5       | -     |
   | 5     | 1        | 2       | debug |
   | 6     | 2        | 2       | debug |
   | 7     | 1        | 10      | direct line in 10 segments |

### Outputs:
- **`ray_x`** — x-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`ray_y`** — y-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`ray_z`** — z-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`weight`** — Per-segment weights; `[n_pos, n_path, n_seg]`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (consumes the output of this function)

---
<a id="icosphere"></a>
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#icosphere) · [Python](../python/site-specific-simulation-tools.md#icosphere)

- Produces 20 × n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Declaration:
```cpp
arma::uword quadriga_lib::icosphere(
    arma::uword n_div,
    dtype radius,
    arma::Mat<dtype> *center,
    arma::Col<dtype> *length = nullptr,
    arma::Mat<dtype> *vert = nullptr,
    arma::Mat<dtype> *direction = nullptr,
    bool direction_xyz = false);
```

### Inputs:
- **`n_div`** — Number of subdivisions; generates 20 × n_div² faces
- **`radius`** — Radius of icosphere in meters; finite and positive
- **`direction_xyz`** (optional) — Output directions in Cartesian (true) or spherical azimuth/elevation (false)

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward from origin with magnitude equal to the inradius of the face; `[n_faces, 3]`
- **`length`** (optional) — Distance from origin to face plane; equals the magnitude of each `center` vector; `[n_faces]`
- **`vert`** (optional) — Vertex offsets from face center [x1,y1,z1,x2,y2,z2,x3,y3,z3]; `[n_faces, 9]`
- **`direction`** (optional) — Directions of the vertex rays; spherical [az1,el1,az2,el2,az3,el3] or Cartesian [x1,y1,z1,x2,y2,z2,x3,y3,z3] per `direction_xyz` flag; `[n_faces, 6]` or `[n_faces, 9]`

### Returns:
- Number of generated triangular faces (20 × n_div²)

---
<a id="interface_gain"></a>
## interface_gain
Linear gain of the lumped penetration loss of entering a material

Not in: MATLAB, Python

- Computes `g = 10^(-A/10)` with `A = att·(f/fRef)^attB` [dB], the loss applied once where a ray enters material
  `iM`, on top of the Fresnel term of the face (e.g. a building-entry loss of 3GPP TR 38.901). `att` must be
  `≥ 0` (the material table is checked when it is read), so `g` is in `[0, 1]`.
- Only `att`, `attB` and `fRef` are read; the face reflection is computed by [ray_mesh_interact](#ray_mesh_interact) and
  [ray_state_update](#ray_state_update), the distance-dependent loss by [medium_gain](#medium_gain). Columns the material model does not know
  are ignored.

### Declaration:
```cpp
dtype quadriga_lib::interface_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype center_frequency);
```

### Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl`
- **`iM`** — 1-based material index (0 = no material / air)
- **`center_frequency`** — Center frequency in [Hz]

### Returns:
- Linear interface gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the interface

### See also:
- [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md) (the formulas and the columns)
- [medium_gain](#medium_gain) (for the distance-dependent in-medium loss)
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
<a id="medium_gain"></a>
## medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

Not in: MATLAB, Python

- Computes `g = 10^(-A/10)` for a path of length `dist` inside material `iM`, where `A` [dB] sums:
  - the conductivity loss of the ITU-R P.2040 model, `8.686·k₀·|Im √(ε_r·μ_r)|` dB/m, with
    `ε_r = a·(f/fRef)^b − j·17.98·σ/f[GHz]`, `σ = c·(f/fRef)^d`, `μ_r` from `e`, `f`, `g`, `h` in the same form,
    and for a material without the mass column the permittivity resonance (`resF`, `resQ`, `resS`);
  - the distance absorption `alpha·(f/fRef)^alphaB` dB/m (excess loss not captured by `σ`, e.g. foliage);
  - for a mass-law material (`m > 0`, the acoustic switch) the mass term, added once and not per meter where
    `dist` exceeds 1.5 mm: `20·log10((f/fRef)·dist)` dB with `dist` as the normal thickness, the resonance and
    coincidence columns acting on it, zero where it would be negative. With a panel size `side > 0` (√area in m,
    3.29 where the column is absent) Davy's radiation efficiency of a finite panel enters: a panel small in
    wavelengths insulates more than the infinite plate (`side = 0`).
- One path length at normal incidence: the angle dependence of the mass term, the bonded plate of several
  bodies and the refracted path inside a traced solid are applied by [ray_state_update](#ray_state_update), not here.
- The penetration-loss columns `att`, `attB` are not used here ([interface_gain](#interface_gain)).

### Declaration:
```cpp
dtype quadriga_lib::medium_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype dist,
    dtype center_frequency);
```

### Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl`
- **`iM`** —  1-based material index (0 = no material / air)
- **`dist`** — Path length of the ray inside the medium in [m]
- **`center_frequency`** — Center frequency in [Hz]

### Returns:
- Linear in-medium gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the medium

### See also:
- [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md) (the formulas and the columns)
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
<a id="mitsuba_xml_file_write"></a>
## mitsuba_xml_file_write
Write a triangular mesh to a Mitsuba 3 XML scene file

Also in: [Python](../python/site-specific-simulation-tools.md#mitsuba_xml_file_write) · Not in: MATLAB

- Converts quadriga-lib mesh data structures to Mitsuba 3 XML format, loadable by NVIDIA Sionna RT for
  differentiable radio-propagation simulations
- Supports grouping faces into named objects with per-face material assignments
- Optionally maps material names to ITU-defined presets used by Sionna RT
- Creates a subdirectory `<stem>_meshes/` next to the XML file and writes one binary PLY file per object into it;
  both the XML and the mesh folder must be distributable together; the XML refers to each PLY file by its path
  relative to the XML file (`<stem>_meshes/<object>.ply`)
- Objects whose faces reference more than one material are automatically split into sub-objects (one per material)
  and renamed `<obj_name>_<mtl_name>`; the effective object count in the output may therefore exceed the length of `obj_names`

### Declaration:
```cpp
void quadriga_lib::mitsuba_xml_file_write(
    const std::string &fn,
    const arma::Mat<dtype> &vert_list,
    const arma::umat &face_ind,
    const arma::uvec &obj_ind,
    const arma::uvec &mtl_ind,
    const std::vector<std::string> &obj_names,
    const std::vector<std::string> &mtl_names,
    const arma::Mat<dtype> &bsdf = {},
    bool map_to_itu_materials = false);
```

### Inputs:
- **`fn`** — Output file path including `.xml` extension
- **`vert_list`** — Vertex coordinates (x, y, z); `[n_vert, 3]`
- **`face_ind`** — Triangle definitions as 0-based vertex indices; `[n_mesh, 3]`
- **`obj_ind`** — 0-based object index per triangle; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle (0 = no material; every face must reference a 
  material — 0 is rejected); `[n_mesh]`
- **`obj_names`** — Object names; length must equal `max(obj_ind)+1`
- **`mtl_names`** — Material names; length must equal `max(mtl_ind)` (mtl_ind is 1-based)
- **`bsdf`** *(optional)* — BSDF material parameters per material; ignored by Sionna RT, used only by Mitsuba renderer; see [obj_file_read](#obj_file_read) for field definitions; `[mtl_names.size(), 17]`
- **`map_to_itu_materials`** *(optional)* — If `true`, maps material names to ITU presets recognised by Sionna RT

### See also:
- [obj_file_read](#obj_file_read) (source for mesh data and BSDF field layout)

---
<a id="obj_file_read"></a>
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#obj_file_read) · [Python](../python/site-specific-simulation-tools.md#obj_file_read)

- Parses a triangulated `.obj`; quads and n-gons are rejected. Face references may be `v`, `v/vt`, `v//vn` or
  `v/vt/vn`; a negative (relative) index counts back from the last vertex defined before the face line (`-1` = that
  vertex). Index 0, or an index beyond the vertices defined before the face, throws `std::invalid_argument`. The file is
  parsed and validated for any combination of requested outputs, also when no output is requested.
- Lines may end in LF or CRLF (`.obj`, `.mtl` and material CSV).
- Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`). Unmatched names
  throw when `csv_strict = true`; otherwise they map to index 0 (no material). The two index spaces are
  decoupled, so several visual materials (e.g. `wall.001`, `wall.002`) may resolve to a single EM material.
- `usemtl (null)` (written by [obj_file_write](#obj_file_write)) marks the following faces as having no material: index 0 in `mtl_ind`
  and `csv_ind` (also with `csv_strict`); `(null)` is not listed in `mtl_names` or `bsdf`.
- Geometry indices (`face_ind`, `obj_ind`) are 0-based. Material indices (`mtl_ind`, `csv_ind`) are
  1-based, with 0 reserved for the outside / no-material state (faces with no assigned material;
  for `csv_ind` in non-strict mode, also materials absent from the table).
- With an empty `fn_obj`, geometry and `.mtl` outputs are empty and only the table (`csv_names`, `csv_prop`) 
  is populated — useful for inspecting a CSV or the default library. If `fn_csv` is also empty, the built-in default table is returned.
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Declaration:
```cpp
arma::uword quadriga_lib::obj_file_read(
    const std::string &fn_obj = "",
    arma::Mat<dtype> *mesh = nullptr,
    arma::Mat<dtype> *vert_list = nullptr,
    arma::umat *face_ind = nullptr,
    arma::uvec *obj_ind = nullptr,
    std::vector<std::string> *obj_names = nullptr,
    arma::uvec *mtl_ind = nullptr,
    std::vector<std::string> *mtl_names = nullptr,
    arma::Mat<dtype> *bsdf = nullptr,
    const std::string &fn_csv = "",
    arma::uvec *csv_ind = nullptr,
    std::vector<std::string> *csv_names = nullptr,
    std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr,
    bool csv_strict = false);
```

### Inputs:
- **`fn_obj`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** — Path to an EM/acoustic material CSV; must contain a `name` column. Unmatched faces map
  to index 0 (no material) unless `csv_strict` is set. Empty uses the built-in ITU-R P.2040 default table.
- **`csv_strict`** — If `true`, throw when a `usemtl` material is absent from the table; otherwise map to index 0 (no material)

### Outputs:
- **`mesh`** — Triangle vertex coordinates `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 0-based vertex indices into `vert_list` per triangle; `[n_mesh, 3]`
- **`obj_ind`** — 0-based object index per triangle; `[n_mesh]`
- **`obj_names`** — Object names; length `max(obj_ind)+1`
- **`mtl_ind`** — 1-based visual-material index per triangle (0 = no material); `[n_mesh]`
- **`mtl_names`** — Visual material names (raw `usemtl`); length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[no_mtl, 17]`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle (0 = no material); `[n_mesh]`
- **`csv_names`** — Material names from the table; length `n_csv` (the full table)
- **`csv_prop`** — Material properties keyed by CSV column name (excluding `name`); each value has
  length `n_csv`. Columns absent from the table are defaulted by consumers; empty cells parse as 0.

### Returns:
- Number of triangular mesh elements (`n_mesh`)

### See also:
- [obj_file_write](#obj_file_write) (for writing OBJ files)
- [obj_overlap_test](#obj_overlap_test) (for testing mesh geometry)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (used to calculate indexed mesh for faster processing)
- [ray_mesh_interact](#ray_mesh_interact) (calculating interactions between rays and the triangular mesh)
- [mitsuba_xml_file_write](#mitsuba_xml_file_write) (for exporting to Mitsuba scene file format)

---
<a id="obj_file_write"></a>
## obj_file_write
Write a triangulated Wavefront .obj (and .mtl) file

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#obj_file_write) · [Python](../python/site-specific-simulation-tools.md#obj_file_write)

- Supply geometry as either `mesh`, or as `vert_list` + `face_ind`; giving both, or neither, is an error
- With `mesh`: `vert_list_out` + `face_ind_out` are derived from it, merging vertices of the same object that
  are closer than `threshold` (no merging across objects). With `vert_list`/`face_ind`: data is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind`/`obj_names`: a single object named `object` is written
- With `split_loose_parts`: objects are separated into connected components ("separate by loose parts"); connectivity
  follows the welded vertex list, so unwelded input geometry yields one part per face
- Without `mtl_ind`: no `usemtl` tags and no `.mtl` file are written. With `mtl_ind`, each face carries a
  1-based material index (0 = no material, leaving that face unassigned); pass `mtl_ind = nullptr` to omit materials entirely
- A face with `mtl_ind = 0` that follows a face with a material in the same object (or part) is preceded by
  `usemtl (null)`, which [obj_file_read](#obj_file_read) reads back as 0; the `.mtl` gets no `(null)` entry. Face order and objects
  are unchanged. A material named `(null)` that is used by a face is therefore rejected (`std::invalid_argument`);
  unused entries of `mtl_names` may carry the name
- The `.mtl` (named after the `.obj`) lists each used material; values default to a gray material when `bsdf` is omitted
- Duplicate entries in `mtl_names` are merged into one `.mtl` entry if their `bsdf` rows are identical (or if `bsdf`
  is omitted); duplicates with differing `bsdf` rows are disambiguated as `name.001`, `name.002`, ...
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the `.obj`):
  columns follow a fixed canonical order, then any extra `csv_prop` columns (alphabetical); `csv_write_defaults`
  additionally emits canonical columns absent from `csv_prop`, filled with their defaults (`a`, `e`, `fRef` = 1,
  `side` = 3.29, else 0)

### Declaration:
```cpp
void obj_file_write(
    const std::string &fn = "",
    const arma::Mat<dtype> *mesh = nullptr,
    const arma::uvec *obj_ind = nullptr,
    const arma::uvec *mtl_ind = nullptr,
    const std::vector<std::string> *obj_names = nullptr,
    const std::vector<std::string> *mtl_names = nullptr,
    arma::Mat<dtype> *vert_list_out = nullptr,
    arma::umat *face_ind_out = nullptr,
    const arma::Mat<dtype> *vert_list = nullptr,
    const arma::umat *face_ind = nullptr,
    const arma::Mat<dtype> *bsdf = nullptr,
    const dtype threshold = 0.001,
    const arma::uvec *csv_ind = nullptr,
    const std::vector<std::string> *csv_names = nullptr,
    const std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr,
    bool csv_write_defaults = false,
    bool split_loose_parts = false);
```

### Inputs:
- **`fn`** — Output path; must end in `.obj`; if empty, no files are written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{x1,y1,z1,...,x3,y3,z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list`/`face_ind`
- **`obj_ind`** — 0-based object index per face; `[n_mesh]`; each object must be a contiguous block
- **`mtl_ind`** — 1-based material index per face (0 = no material); `[n_mesh]`; omit (`nullptr`) for no materials
- **`obj_names`** — Object names; length > `max(obj_ind)`; required if `obj_ind` is given
- **`mtl_names`** — Material names; length ≥ `max(mtl_ind)` (1-based); required if `mtl_ind` is given
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only with `face_ind`, written unchanged
- **`face_ind`** — 0-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF for the `.mtl`; `[n_mtl, 17]`; see [obj_file_read](#obj_file_read) for columns
- **`threshold`** — Vertex co-location distance for merging within an object; default 1 mm
- **`csv_ind`** — 1-based EM/acoustic-material index per face (0 = no material); `[n_mesh]`; optional, validated if given
- **`csv_names`** — EM/acoustic material names (the full table); writing the `.csv` requires this
- **`csv_prop`** — Material properties keyed by column name; each vector must have one value per `csv_names` entry
- **`csv_write_defaults`** — If `true`, also write canonical columns absent from `csv_prop`, using their defaults
- **`split_loose_parts`** — If `true`, split each object into connected components (faces sharing a vertex); parts of a split object are named `name.001`, `name.002`, ...

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)
- [mitsuba_xml_file_write](#mitsuba_xml_file_write) (for exporting to Mitsuba scene file format)

---
<a id="obj_overlap_test"></a>
## obj_overlap_test
Detect overlapping 3D objects in a triangular mesh

Not in: MATLAB, Python

- Returns the IDs (as given in `obj_ind`) of all objects that intersect at least one other object
- Touching faces or edges are not considered overlapping
- Checks for intersecting triangle faces and vertices/edges penetrating another object's bounding volume
- Overlaps smaller than `tolerance` are ignored to account for numerical imprecision
- Does not modify or repair the mesh

### Declaration:
```cpp
arma::uvec quadriga_lib::obj_overlap_test(
    const arma::Mat<dtype> *mesh,
    const arma::uvec *obj_ind,
    std::vector<std::string> *reason = nullptr,
    dtype tolerance = 0.0005);
```

### Inputs:
- **`mesh`** — Triangular mesh; each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** — Object ID of each triangle; any values, e.g. the 0-based `obj_ind` output of [obj_file_read](#obj_file_read); `[n_mesh]`
- **`reason`** *(optional)* — Human-readable overlap description per overlapping object: the finding of its first
  overlapping pair, pairs taken in the order of the sorted object IDs; independent of the number of threads; `[n_overlap]`
- **`tolerance`** *(optional)* — Geometric tolerance; intersections smaller than this are ignored

### Returns:
- `arma::uvec`: IDs of all overlapping objects as given in `obj_ind`, unique and ascending; empty for an empty mesh; `[n_overlap]`

### See also:
- [obj_file_read](#obj_file_read) (reads mesh data from files and generates `obj_ind` input)

---
<a id="path_to_tube"></a>
## path_to_tube
Convert a 3D path into a tube surface mesh for visualization

Not in: MATLAB, Python

- Converts an ordered sequence of 3D points into a tubular quad mesh with circular cross-sections
- Where the path turns by more than 160° (the two segments enclose an angle below 20°), the tube is split and an
  extra vertex ring is inserted to avoid intersection
- Cross-section orientation uses continuous frame alignment between segments to minimize twisting
- Output `faces` indices are directly usable in `.obj` or `.ply` export

### Declaration:
```cpp
void quadriga_lib::path_to_tube(
    const arma::Mat<dtype> *path_coord,
    arma::Mat<dtype> *vert,
    arma::umat *faces,
    dtype radius = 1.0,
    arma::uword n_edges = 5);
```

### Inputs:
- **`path_coord`** — Ordered 3D path coordinates; `[3, n_coord]`
- **`radius`** *(optional)* — Tube cross-section radius
- **`n_edges`** *(optional)* — Number of vertices per circular cross-section; must be ≥ 3

### Outputs:
- **`vert`** — Tube vertex positions; `[3, (n_coord + n_split) × n_edges]` where `n_split` is the number of turns > 160°
- **`faces`** — Quad face indices into `vert`, 4 indices per quad; `[4, (n_coord - 1) × n_edges]`

---
<a id="plane"></a>
## plane
Construct a triangulated plane mesh

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#plane) · [Python](../python/site-specific-simulation-tools.md#plane)

- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices at +/-1, z = 0) before scaling
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 * n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with [obj_file_write](#obj_file_write); when the product
  of the x and y scales is negative (mirrored plane), the winding is swapped so that the normals still point in +z
  direction before the rotation
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, Euler rotation, and translation are applied in that order (scale -> rotate -> translate)

### Declaration:
```cpp
arma::Mat<dtype> quadriga_lib::plane(
    const arma::vec &scale = {1.0},
    const arma::vec &rotation = {0.0, 0.0, 0.0},
    const arma::vec &location = {0.0, 0.0, 0.0},
    const arma::uword n_div = 1);
```

### Inputs:
- **`scale`** — Length 1 scales x and y uniformly; length 2 scales `{x,y}` independently; length 3 is accepted for
  compatibility with [cube](#cube), but the third element is ignored; empty = 1 (no scaling)
- **`rotation`** — Euler angles in [rad] about {x,y,z}, applied as `R = Rz*Ry*Rx` (Blender XYZ);
  length 3 or empty (no rotation)
- **`location`** — Translation {x,y,z} in [m]; length 3 or empty (origin)
- **`n_div`** — Number of subdivisions per edge; results in `2 * n_div^2` triangles

### Returns:
- Triangle mesh; each row holds `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[2 * n_div^2, 9]`

### See also:
- [cube](#cube)
- [icosphere](#icosphere)
- [subdivide_triangles](#subdivide_triangles)
- [obj_file_write](#obj_file_write)

---
<a id="point_cloud_aabb"></a>
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#point_cloud_aabb) · [Python](../python/site-specific-simulation-tools.md#point_cloud_aabb)

- Each row of the output contains `[x_min, x_max, y_min, y_max, z_min, z_max]` for one sub-cloud
- If `sub_cloud_index` is `nullptr` or empty, the entire input is treated as a single cloud; last index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Declaration:
```cpp
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(
    const arma::Mat<dtype> *points,
    const arma::u32_vec *sub_cloud_index = nullptr,
    arma::uword vec_size = 1);
```

### Inputs:
- **`points`** — 3D point coordinates; `[n_points, 3]`
- **`sub_cloud_index`** *(optional)* — Row indices marking the start of each sub-cloud; use [point_cloud_segmentation](#point_cloud_segmentation) to generate; `[n_sub]`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16)

### Returns:
- Bounding box matrix; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

### See also:
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud indices)
- [point_cloud_split](#point_cloud_split) (split point cloud)
- [ray_point_intersect](#ray_point_intersect) (use AABBs for intersection)

---
<a id="point_cloud_segmentation"></a>
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#point_cloud_segmentation) · [Python](../python/site-specific-simulation-tools.md#point_cloud_segmentation)

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint.
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center.
- Produces a reorganized point array and index maps to track reordering.

### Declaration:
```cpp
arma::uword quadriga_lib::point_cloud_segmentation(
    const arma::Mat<dtype> *points,
    arma::Mat<dtype> *pointsR,
    arma::u32_vec *sub_cloud_index,
    arma::uword target_size = 1024,
    arma::uword vec_size = 1,
    arma::u32_vec *forward_index = nullptr,
    arma::u32_vec *reverse_index = nullptr);
```

### Inputs:
- **`points`** — Original 3D point cloud; `[n_points, 3]`
- **`target_size`** *(optional)* — Maximum points per sub-cloud before padding
- **`vec_size`** *(optional)* — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`;
  [ray_point_intersect](#ray_point_intersect) accepts any value, padding (e.g. 8 for AVX2) only improves its speed

### Outputs:
- **`pointsR`** — Reorganized point cloud with points grouped by sub-cloud; `[n_pointsR, 3]`
- **`sub_cloud_index`** — 0-based starting index of each sub-cloud within `pointsR`; `[n_sub]`
- **`forward_index`** *(optional)* — 1-based row in `points` of each row of `pointsR`; padding entries are `0`; `[n_pointsR]`
- **`reverse_index`** *(optional)* — 0-based row in `pointsR` of each row of `points`; `[n_points]`

### Returns:
- Number of generated sub-clouds `n_sub`

### See also:
- [point_cloud_aabb](#point_cloud_aabb) (bounding box computation)
- [point_cloud_split](#point_cloud_split) (related spatial splitting)
- [ray_point_intersect](#ray_point_intersect) (downstream use case)

---
<a id="point_cloud_split"></a>
## point_cloud_split
Split a point cloud into two sub-clouds along a spatial axis

Not in: MATLAB, Python

- Splits at the bounding box midpoint along the chosen axis (not the statistical median); 
  the split may be unbalanced if points are non-uniformly distributed.
- If `axis == 0`, the longest bounding box extent is used.
- Returns a negative axis value if the split failed (all points on one side); outputs are not modified in that case.

### Declaration:
```cpp
int quadriga_lib::point_cloud_split(
    const arma::Mat<dtype> *points,
    arma::Mat<dtype> *pointsA,
    arma::Mat<dtype> *pointsB,
    int axis = 0,
    arma::Col<int> *split_ind = nullptr);
```

### Inputs:
- **`points`** — Input point cloud; `[n_points, 3]`
- **`axis`** *(optional)* — Split axis: `0` = longest extent, `1` = x, `2` = y, `3` = z

### Outputs:
- **`pointsA`** — First sub-cloud; `[n_pointsA, 3]`
- **`pointsB`** — Second sub-cloud; `[n_pointsB, 3]`
- **`split_ind`** *(optional)* — Per-point destination: `1` = pointsA, `2` = pointsB, `0` = error; `[n_points]`

### Returns:
- Axis used: `1` = x, `2` = y, `3` = z; negative (`-1`, `-2`, `-3`) if split failed

### See also:
- [point_cloud_aabb](#point_cloud_aabb) (bounding box computation)
- [point_cloud_segmentation](#point_cloud_segmentation) (recursive partitioning using this function)
- [ray_point_intersect](#ray_point_intersect) (downstream use case)

---
<a id="point_inside_mesh"></a>
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#point_inside_mesh) · [Python](../python/site-specific-simulation-tools.md#point_inside_mesh)

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided, returns the 1-based enclosing object index instead of binary 0/1

### Declaration:
```cpp
arma::uvec quadriga_lib::point_inside_mesh(
    const arma::Mat<dtype> *points,
    const arma::Mat<dtype> *mesh,
    const arma::uvec *obj_ind = nullptr,
    dtype distance = 0.0);
```

### Inputs:
- **`points`** — 3D coordinates of test points; `[n_points, 3]`
- **`mesh`** — Triangle faces in row-major vertex format  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** *(optional)* — 0-based object index per mesh element; enables per-object output; `[n_mesh]`
- **`distance`** *(optional)* — Surface proximity threshold; points within this distance
  of the mesh surface are classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1);
  range: 0–20 m (default: 0)

### Returns:
- `arma::uvec`, size `[n_points]`; `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object index (with `obj_ind`)

### See also:
- [obj_file_read](#obj_file_read) (for reading `mesh` and `obj_ind` from an .obj file)

---
<a id="ray_commit"></a>
## ray_commit
Commit the paths of a launch configuration that reach a receiver

Not in: MATLAB, Python

- Intersects the current beam set with the receive point cloud ([ray_point_intersect](#ray_point_intersect)) and appends one
  [path](channel-functions.md#path) object to `paths_commit` for every ray-point pair that survives the gates below.
- A pair is committed when the receiver is not shaded by the ray's own first-bounce face, the total path
  length stays below `max_path_length`, and the gain at `center_frequency[0]` stays above `min_gain_dB`.
- Shading is a half-space test against the plane of the first-bounce face: the receiver must lie on the
  same side of that plane as the ray origin. The face is treated as an infinite plane, so a receiver just
  past the edge of a small face is conservatively dropped.
- Rays flagged in `subdiv_flag_in` are skipped entirely — they reappear as sub-beams in the next
  generation and would otherwise be committed twice. Subdivision is not detected internally.
- A ray travelling inside a medium is committed with the in-layer attenuation of the final leg folded into
  its coefficients, per frequency, via [medium_gain](#medium_gain) on `mtl_ind_current & 0x3FFF`.
- In EM mode a receive-side mirror (`VV = 1`, `HH = -1`) is applied per frequency; in SCALAR mode only the gain is applied.
- The receiver index is written to the committed path's `iC`, 0-based. Without `point_index` it indexes
  `points` as passed in; with `point_index` it indexes the caller's original, unsegmented point list, so a
  cloud reordered by [point_cloud_segmentation](#point_cloud_segmentation) needs no remapping afterwards.
- Padding points inserted by [point_cloud_segmentation](#point_cloud_segmentation) for SIMD alignment sit at the center of their
  sub-cloud AABB and are therefore hit often. They are dropped when `point_index` is supplied and committed
  as if they were receivers when it is not.
- `min_no_segments` suppresses commits from rays that have not interacted often enough. The launch-sphere
  beams of the first generation carry `nSEG = 0` and are far too wide for a point-in-tube test to be
  meaningful, so a beam-traced run passes 1 here.
- The committed path is not extended by a segment: the receiver is not an interaction point, so `nSEG` and
  `length` are those of the in-flight ray and the caller recovers the total with `path::calc_length`.

### Declaration:
```cpp
arma::uword quadriga_lib::ray_commit(
    const std::vector<quadriga_lib::path> &paths,
    std::vector<quadriga_lib::path> &paths_commit,
    const arma::fmat &mesh,
    const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
    const arma::fvec &center_frequency,
    const arma::fmat &orig,
    const arma::u32_vec &fbs_ind,
    const arma::fmat &trivec,
    const arma::fmat &tridir,
    const arma::Col<short> &mtl_ind_current,
    const arma::fmat &rx_points,
    const arma::u32_vec *sub_cloud_index = nullptr,
    const arma::u32_vec *point_index = nullptr,
    const std::vector<bool> *subdiv_flag_in = nullptr,
    float max_path_length = 10e3,
    float min_gain_dB = -140.0f,
    uint8_t min_no_segments = 0,
    bool ignore_direct_path = false);
```

### Inputs:
- **`paths`** — In-flight per-ray [path](channel-functions.md#path) objects; `[n_ray]`. Frequency count and layout define those of the committed paths; not modified
- **`mesh`** — Faces of the triangular mesh; each row `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`. Only the
  faces named by `fbs_ind` are read, for the shading plane
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read));
  each value has length `n_mtl` (max 16383)
- **`center_frequency`** — Center frequencies in [Hz]; `[n_freq]`, 1 to 127 entries, must match the layout of
  `paths`. `center_frequency[0]` is the reference frequency for the gain gate
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`. Defines `n_ray`; must be non-empty
- **`fbs_ind`** — 1-based index of the first intersected mesh element, 0 = no hit; `[n_ray]`. Obtained from
  [ray_triangle_intersect](#ray_triangle_intersect) for the same launch configuration
- **`trivec`** — Beam wavefront triangle vertices relative to the ray origin; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, Cartesian; `[n_ray, 9]`. Need not be unit length
- **`mtl_ind_current`** — Current medium state word, 0 = outside (bit-masked: `mat = w & 0x3FFF`,
  flags in bits 14 and 15); `[n_ray]`
- **`rx_points`** — Receive points in 3D space; `[n_point, 3]`
- **`sub_cloud_index`** *(optional)* — Sub-cloud partition offsets for the point cloud (see
  [point_cloud_segmentation](#point_cloud_segmentation)); `[n_sub]`. NULL → no partitioning
- **`point_index`** *(optional)* — 1-based map from each row of `points` back to the caller's original point
  list, 0 = padding inserted for SIMD alignment; `[n_point]`. Pass the `forward_index` output of
  [point_cloud_segmentation](#point_cloud_segmentation) for the same cloud — not `reverse_index`, which runs the other way and has
  length `n_point` of the *unsegmented* cloud. Padded rows are skipped and `iC` is written in the original
  indexing. NULL → `points` is used as passed in and `iC` indexes it directly
- **`subdiv_flag_in`** *(optional)* — Rays that will be split in the next generation and must not be
  committed now; `[n_ray]`, indexed in the full ray set. Pass the output of [ray_subdivide_flag](#ray_subdivide_flag) for the
  same launch configuration. NULL / empty → no ray is excluded on these grounds
- **`max_path_length`** *(optional)* — Maximum total path length including the leg to the receiver [m]
- **`min_gain_dB`** *(optional)* — Gain at `center_frequency[0]` below which a path is not committed, in dB;
  evaluated with free-space path loss and the in-medium loss of the final leg included
- **`min_no_segments`** *(optional)* — Minimum number of interaction segments (`nSEG`) a ray must carry
  before it can commit, 0 to 255. 0 = no restriction. 1 excludes the launch-sphere beams of the first
  generation, including the sub-beams they spawn, which keep `nSEG = 0` until their first interaction
- **`ignore_direct_path`** *(optional)* — Drop every path that arrives by transmission only
  (`nREF == 0 && nSCT == 0`, which includes pure LOS); these are covered by [calc_diffraction_gain](#calc_diffraction_gain).
  The test is unconditional: under refraction the traced path is longer than the straight line, so a
  length-based test would let it through and double-count against the diffraction model

### Output:
- **`paths_commit`** — Committed paths, appended to whatever the vector already holds; extended by
  `n_commit` entries. Each carries the 0-based receiver index in `iC`, the interaction history of its ray, and the
  transfer coefficients with the receive-side mirror and any in-medium loss applied. Existing entries are
  not modified; if the vector is non-empty its layout must match `paths`

### Returns:
- Number of newly committed paths, `n_commit`

### See also:
- [ray_point_intersect](#ray_point_intersect) (ray-point intersection, produces the pair list)
- [ray_subdivide_flag](#ray_subdivide_flag) (produces `subdiv_flag_in`)
- [ray_progress](#ray_progress) (advance the launch configuration to the next generation)
- [point_cloud_segmentation](#point_cloud_segmentation) (generate `sub_cloud_index` and `point_index`)
- [calc_diffraction_gain](#calc_diffraction_gain) (covers the paths removed by `ignore_direct_path`)
- [path](channel-functions.md#path) (the per-ray storage object)

---
<a id="ray_init"></a>
## ray_init
Seed a sphere of rays from a point source

Not in: MATLAB, Python

- Launches `n_ray` rays from a point source `O` onto an [icosphere](#icosphere) tessellation, giving near-uniform
  angular coverage of the full sphere (4π sr).
- `n_ray` is quantized to the icosphere grid: `n_div = round(sqrt(n_ray_target / 20))` (min 1) and
  `n_ray = 20 · n_div²`, so the returned count is the closest tessellation to `n_ray_target`, not exact.
- Ray origins sit on a small launch sphere of radius `r0` centered at `O`, not at `O` itself, so the beam
  triangles (`trivec`) have finite extent from the first segment.
- When `mesh` or `rx_points` is supplied, `r0` is auto-sized to 0.8× the distance to the nearest obstacle
  (found along a coarse probe sphere) or receive point, whichever is closer, clamped to ≥ 0.01 m. A receive
  point inside the launch sphere would never be hit by any ray, so it bounds `r0` the same way an obstacle
  does. If neither is within `max_path_length`, or with both omitted, `r0 = 0.01 m`.
- Emits the per-ray medium-state words and distance accumulators consumed by [ray_state_update](#ray_state_update), all
  initialized to the outside-air / zero-distance start state. `path_dir_prev` and the refracted half of
  `acc_dist` are tracked per frequency, since the Snell bend that defines them depends on the material at
  the evaluation frequency. A launched ray is outside a medium, so the initial direction is the geometric
  one and is replicated across all `n_freq` slices of `path_dir_prev`.
- Beam wavefront (`trivec`) and directions (`tridir`, Cartesian) match the [ray_mesh_interact](#ray_mesh_interact) input format.

### Declaration:
```cpp
arma::uword ray_init(
    arma::uword n_ray_target,
    arma::uword n_freq,
    float Ox, float Oy, float Oz,
    float max_path_length,
    arma::fmat *orig = nullptr,
    arma::fmat *dest = nullptr,
    arma::fmat *trivec = nullptr,
    arma::fmat *tridir = nullptr,
    arma::Col<short> *mtl_ind_prev = nullptr,
    arma::Col<short> *mtl_ind_current = nullptr,
    arma::Col<short> *mtl_ind_buffer = nullptr,
    arma::fcube *path_dir_prev = nullptr,
    arma::fmat *acc_dist = nullptr,
    std::vector<quadriga_lib::path> *paths = nullptr,
    const arma::fmat *mesh = nullptr,
    const arma::u32_vec *sub_mesh_index = nullptr,
    const arma::fmat *aabb = nullptr,
    const arma::fmat *rx_points = nullptr,
    bool scalar_mode = false);
```

### Inputs:
- **`n_ray_target`** — Desired ray count; quantized to the nearest icosphere grid (see above)
- **`n_freq`** — Number of frequency bins allocated per path in `paths`; must be ≥ 1 (throws if 0; the
  [path](channel-functions.md#path) layout supports 1-127)
- **`Ox`**, **`Oy`**, **`Oz`** — Point-source (transmitter) position in GCS [m]
- **`max_path_length`** — Maximum ray length [m]; sets `dest` and bounds the launch-sphere probe (floored at 0.01 m)
- **`mesh`** *(optional)* — Triangle mesh faces; see [obj_file_read](#obj_file_read); `[n_mesh, 9]`. Used only to auto-size
  the launch sphere `r0`. NULL → the mesh does not bound `r0`.
- **`sub_mesh_index`** *(optional)* — Sub-mesh partition offsets for the accelerated intersect; 0-based,
  strictly increasing, first entry 0; passed to [ray_triangle_intersect](#ray_triangle_intersect); `[n_sub]`. NULL → no partitioning
- **`aabb`** *(optional)* — Axis-aligned bounding box per sub-mesh; `[n_sub, 6]`. Requires `sub_mesh_index`.
  Unlike [ray_progress](#ray_progress), boxes are *not* computed internally when omitted: the probe is a fixed 2880 rays,
  so building them here would usually cost more than it saves. Pass the [triangle_mesh_aabb](#triangle_mesh_aabb) result the
  caller already holds for the tracing loop.
- **`rx_points`** *(optional)* — Receive points in 3D space; `[n_point, 3]`. Used only to bound the launch
  sphere `r0`, so that no receiver starts inside it. NULL → receivers are not considered.
- **`scalar_mode`** — Path storage layout passed to `paths`: `true` = SCALAR (acoustic, one pressure
  coefficient per frequency), `false` = EM (2×2 Jones matrix per frequency)

### Outputs:
- **`orig`** — Ray origins on the launch sphere, `O + r0·d̂`; `[n_ray, 3]`
- **`dest`** — Ray destinations at `max_path_length` from `O`; `[n_ray, 3]`
- **`trivec`** — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, matches [ray_mesh_interact](#ray_mesh_interact)
- **`tridir`** — Per-vertex ray directions, Cartesian; `[n_ray, 9]`
- **`mtl_ind_prev`**, **`mtl_ind_current`**, **`mtl_ind_buffer`** — Initial medium-state words for
  [ray_state_update](#ray_state_update), all zeroed (outside air, no flags); `[n_ray]`
- **`path_dir_prev`** — Initial physical ray direction (unit vectors from `O`), one slice per frequency;
  `[n_ray, 3, n_freq]`. Every slice is seeded with the same launch direction. Slice `f` is contiguous, so a
  consumer can alias it as an `[n_ray, 3]` matrix without copying
- **`acc_dist`** — Accumulated in-layer distance, zeroed; `[n_ray, 1 + n_freq]`; col 1 = geometric distance
  (frequency-independent, shared), col `1+f` = refracted distance at frequency `f`
- **`paths`** — Per-ray [path](channel-functions.md#path) objects, one per ray; each reinitialized to 0 segments with `n_freq`
  frequency bins in the `scalar_mode` layout, and `length` seeded to `orig_length` (the `O`-to-origin offset);
  `n_ray` entries

### Returns:
- Number of rays generated, `n_ray = 20 · n_div²`

### See also:
- [icosphere](#icosphere) (generates the ray fan, beam wavefront, and directions)
- [ray_triangle_intersect](#ray_triangle_intersect) (launch-sphere sizing and per-segment intersection)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [triangle_mesh_aabb](#triangle_mesh_aabb) (generates `aabb`)
- [ray_mesh_interact](#ray_mesh_interact) (consumes `orig` / `dest` / `trivec` / `tridir`)
- [ray_state_update](#ray_state_update) (consumes the medium-state words, `path_dir_prev`, and `acc_dist`)
- [ray_commit](#ray_commit) (consumes the same `points` when committing paths that reach a receiver)
- [path](channel-functions.md#path) (the per-ray storage object populated in `paths`)

---
<a id="ray_mesh_interact"></a>
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#ray_mesh_interact) · Not in: Python

- One face event per ray: from the first and second hit (FBS, SBS) of [ray_triangle_intersect](#ray_triangle_intersect), the interface
  coefficients between homogeneous isotropic media (TE/TM Fresnel; one scalar pressure coefficient for types 3–5,
  the acoustic simulation with impedance-mapped materials), the outgoing direction and the relaunch segment, and
  optionally the ray tube (`trivec`, `tridir`; the output format matches the input).
- Face sides follow the vertex order: counter-clockwise is the front (right-hand rule). A front-side hit with
  FBS ≠ SBS is air-to-medium, a back-side hit medium-to-air; co-located FBS and SBS with opposing normals are a
  medium-to-medium boundary. Overlapping geometry must be avoided.
- The function sees faces only and assumes air on every side the mesh does not name; [ray_state_update](#ray_state_update) knows
  the medium a ray is in and corrects the result.
- Thin sheets: a material with `thk > 0` makes its faces sheets, a face standing for a layer of that thickness.
  A crossing is one event with the internal multiple-reflection series summed in closed form. Two co-located
  sheets of different materials are a stack, traversed in the order their normals give (the layer hit on its
  front side first); two coincident sheets of one material are one sheet (a double-sided export). At most two
  co-located faces form a stack; a third layer is a separate event. A closed body of a sheet material is a
  hollow box. `thk = 0`, or no `thk` column, gives the traced solid.
- Past the critical angle into a lossless medium (a lossy one has none) the face reflects fully where no `tf` is
  set; with `tf` it passes the leak of its pad. With a ray tube and types 2 or 5, a tube with any vertex past the
  critical angle passes undeviated as a whole.
- The loss inside a traced solid is not applied here ([ray_state_update](#ray_state_update) applies it where the leg closes); a sheet's
  composite includes the loss and the mass term of its layer. No spreading loss is applied.
- The material model, its columns, the rules of use and the limits: the companion page under See also; the CSV format
  of the material table: [Data Formats](../../formats.md)

### Declaration:
```cpp
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

### Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [obj_file_read](#obj_file_read); `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`.
  0 = face has no material (air). NULL → all faces treated as air.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read));
  each value has length `n_mtl`. NULL → air defaults used.
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`compact`** *(optional)* — If `true`, no-hit rays are dropped and `n_rayN ≤ n_ray`. If
  `false` (default), all rays are kept (`n_rayN = n_ray`) and no-hit rays are written as a transparent pass-through
  (gain 1, identity `xprmat`, `out_type = 0`).

### Outputs:
- **`origN`** — New origins, nudged off the face along the travel direction by 8 float ULP of the hit point's
  coordinate magnitude (past the SBS of a co-located pair, except for reflection); `[n_rayN, 3]`
- **`destN`** — New destinations along the outgoing direction; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_rayN, 3]`
- **`gainN`** — Linear power gain of the interaction, derived from `xprmatN`; `[n_rayN]`
- **`xprmatN`** — Polarization transfer matrix of amplitude coefficients that carry the interaction gain. Types 0–2:
  interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, the Jones matrix `M = [VV VH; HV HH]`
  acting as `e_out = M · e_in`, the incoming V/H basis from the incident direction and the outgoing one from
  `path_dirN` (`eH = ẑ × k̂` normalized, `eV = eH × k̂`); `[8, n_rayN]`. Types 3–5: `[Re Im]` of the scalar pressure
  coefficient; `[2, n_rayN]`. A transmitted coefficient includes the entered material's `interface_gain`.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`fbs_angleN`** — Incidence angle at FBS in rad, the angle to the face (≥ 0 for a front-side hit); `[n_rayN]`
- **`thicknessN`** — Geometric FBS-to-SBS distance; `[n_rayN]`. Unrelated to the material `thk` column:
  this is what the mesh measured, `thk` is what the material table authored.
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (`qd::bits<uint8_t>`); `[n_rayN]`
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag — meaning depends on bits 2 and 6, see below            |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |
   |   6  | Thin-sheet flag (at least one of FBS/SBS is a `thk > 0` material)       |

   Bit 1 by context:
   | Context                                     | Bit 1 = 1 means                               |
   | ------------------------------------------- | --------------------------------------------- |
   | Solid (bit 6 = 0)                           | Front-side hit on FBS (o→i, or M2 hit first)  |
   | Isolated sheet (bit 6 = 1, bit 2 = 0)       | Front-side hit on FBS; informational only     |
   | Sheet glued to a solid (exactly one sheet)  | Front-side hit on the **solid** face          |
   | Two co-located sheets                       | FBS traversed first (0 = SBS traversed first) |

   Reachable composite values (add 32 for the total-reflection variant):
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

   Codes 69 and 71 each stand for three cases, which [ray_state_update](#ray_state_update) tells apart from the materials of the two
   faces. Total reflection occurs with bit 6 only for a glued pair. Bit 7 is unused.
- **`path_dirN`** — Refraction-correct path direction: mirror for types 0/3, Snell direction for types 1/2/4/5; `[n_rayN, 3]`.
  For undeviated transmission (types 1/4) this is the *refracted* direction, which differs from the geometric continuation
  (along the incoming ray) used for `origN`/`destN`; it lets downstream code recover the true transmission angle.
  No ray bends into a mass-law medium (`m > 0`). The wave in it is held: into air or another mass-law body it keeps its
  direction, into a medium without the mass column it is refracted as a wave from air.
- **`ray_indN`** — 0-based input ray index for each output ray (inverse of the internal compaction map; order-preserving); `[n_rayN]`

### See also:
- [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md) (companion document)
- [obj_file_read](#obj_file_read) (for loading `mesh` and `mtl_prop` from OBJ file)
- [ray_state_update](#ray_state_update) (inside/outside state machine)
- [icosphere](#icosphere) (for generating beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for computing FBS and SBS positions)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)

---
<a id="ray_point_intersect"></a>
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#ray_point_intersect) · [Python](../python/site-specific-simulation-tools.md#ray_point_intersect)

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin, enabling energy spread simulation.
- Reports, for each point, the list of 0-based ray indices whose beam intersects that point.
- The primary output is a flat CSR pair (`hit_index`, `hit_offset`); the ray indices of point `i` are `hit_index[hit_offset[i] .. hit_offset[i+1]-1]`.
- All outputs are optional and are only computed when a non-NULL pointer is passed.
- All internal computations use single precision.

### Declaration:
```cpp
void quadriga_lib::ray_point_intersect(
    const arma::Mat<dtype> &points,
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &trivec,
    const arma::Mat<dtype> &tridir,
    std::vector<unsigned> *hit_index = nullptr,
    arma::u32_vec *hit_offset = nullptr,
    arma::u32_vec *hit_count = nullptr,
    std::vector<arma::u32_vec> *hits_per_point = nullptr,
    const arma::u32_vec *sub_cloud_index = nullptr,
    int use_kernel = 0,
    int gpu_id = 0);
```

### Inputs:
- **`points`** — 3D point cloud coordinates; `[n_points, 3]`
- **`orig`** — Ray origin positions in global Cartesian coordinates; `[n_ray, 3]`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order `[v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]`; `[n_ray, 9]`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates (need not be normalized), order `[d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z]`; `[n_ray, 9]`
- **`sub_cloud_index`** *(optional)* — Segment boundary indices for the point cloud (see [point_cloud_segmentation](#point_cloud_segmentation)); sub-clouds
  need not be aligned to the SIMD vector size (padding with `vec_size` is optional and only improves speed); `[n_sub]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable and for other values; auto mode selects CUDA when `n_points >= 500` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

### Optional outputs:
- **`hit_index`** — Flat list of 0-based ray indices, grouped by point; written by the compute kernel without an intermediate copy; `[n_hit]`
- **`hit_offset`** — Start of each point's block within `hit_index`, last element is `n_hit`; `[n_points + 1]`
- **`hit_count`** — Number of rays intersecting each point, equals `hit_offset[i+1] - hit_offset[i]`; `[n_points]`
- **`hits_per_point`** — Per-point list of 0-based ray indices; allocates one `arma::u32_vec` per non-empty point, so only request it when the split form is actually needed; `[n_points]`

### See also:
- [icosphere](#icosphere) (generate ray beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud index)
- [subdivide_rays](#subdivide_rays) (subdivide beams into sub-beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (ray–triangle intersection)
- [ray_mesh_interact](#ray_mesh_interact) (beam–mesh interaction)

---
<a id="ray_progress"></a>
## ray_progress
Advance a ray set by one interaction, spawning reflected, transmitted, and subdivided rays

Not in: MATLAB, Python

- Consumes a launch configuration (origins, destinations, per-ray medium state, and [path](channel-functions.md#path) storage) and
  returns the next iteration: for every ray that hits the mesh, its reflected and/or transmitted
  continuation(s), plus the four sub-beams of any ray flagged for subdivision. Rays that miss, fall below
  `min_gain_dB`, or reach an interaction, reflection or transmission limit are terminated — unless they
  are flagged for subdivision, which takes precedence and is evaluated first.
- The full pipeline per call is: intersect ([ray_triangle_intersect](#ray_triangle_intersect)) → subdivision flag ([ray_subdivide_flag](#ray_subdivide_flag))
  → subdivision ([subdivide_rays](#subdivide_rays)) → compaction → interaction ([ray_mesh_interact](#ray_mesh_interact)) → state resolve
  ([ray_state_update](#ray_state_update)) for a reflection pass and a transmission/refraction pass → assembly.
- The subdivision decision is taken on the full ray set, before the interaction is evaluated and before the
  launch configuration is compacted. A flagged ray is split and does not take part in the interaction passes
  this generation; its sub-beams are re-intersected in the next one.
- The function returns per-stage counts (see Returns); the new configuration holds `n_out = 4·n_subdiv + n_reflect + n_transmit`
  rays, which may exceed or fall short of the `n_ray` passed in.
- An empty returned launch configuration (`orig.n_rows == 0`) signals end of trace; a subsequent call with an empty orig throws.
- Memory is sized for the worst case but committed lazily: the output is built in a worst-case-sized buffer
  committed on first write, the launch configuration is compacted once — on the intersect result and the
  subdivision flag together — before the expensive passes, and dead intermediates are released as the
  function proceeds, so peak footprint stays close to one generation.
  Designed for `n_ray` up to ~10^8; the ray index is 32-bit, so `n_ray` is capped at 2^32-1.
- Geometry is traced once, at the reference frequency `center_frequency[0]`. For the remaining frequencies
  only the polarization/gain coefficient is recomputed and folded into each [path](channel-functions.md#path); the per-frequency
  refracted direction and in-medium distance are approximated by the reference-frequency values (see the
  note on `center_frequency`).
- Beam subdivision and beam-front updates are active only when `trivec` and `tridir` are supplied;
  otherwise rays are traced as infinitesimal.
- The ray-mesh intersection can be delegated: pass `no_interact_in`, `fbs_ind_in` and `sbs_ind_in`
  to reuse a [ray_triangle_intersect](#ray_triangle_intersect) result. When the intersection is delegated, `sub_mesh_index` and `aabb` are unused —
  they only accelerate the internal intersector — validation is skipped in this case and the parameters may be omitted.
- The subdivision decision can likewise be delegated via `subdiv_flag_in`, which a shading pass needs so it
  does not commit beams that reappear as sub-beams. When supplied it is the authority on what gets split;
  see the input description for what that overrides.
- [ray_subdivide_flag](#ray_subdivide_flag) never flags a ray with `fbs_ind` = 0, because there is no face to project the beam
  tube onto. The internal decision therefore never subdivides a ray that missed the mesh; only a delegated
  flag reaches that case.

### Declaration:
```cpp
std::array<unsigned, 4> quadriga_lib::ray_progress(
    const arma::fmat &mesh,
    const arma::uvec &mtl_ind,
    const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
    const arma::fvec &center_frequency,
    float Ox, float Oy, float Oz,
    arma::fmat &orig,
    arma::fmat &dest,
    arma::Col<short> &mtl_ind_prev,
    arma::Col<short> &mtl_ind_current,
    arma::Col<short> &mtl_ind_buffer,
    arma::fcube &path_dir_prev,
    arma::fmat &acc_dist,
    std::vector<quadriga_lib::path> &paths,
    arma::fmat *trivec = nullptr,
    arma::fmat *tridir = nullptr,
    const arma::u32_vec *sub_mesh_index = nullptr,
    const arma::fmat *aabb = nullptr,
    uint8_t max_no_interactions = 20,
    uint8_t max_no_reflections = 10,
    uint8_t max_no_transmissions = 10,
    uint8_t max_no_subdivisions = 2,
    float min_gain_dB = -140.0f,
    float subdivision_tolerance_m = 3.0f,
    float thin_slab_threshold = 0.15f,
    bool refraction_mode = true,
    bool scalar_mode = false,
    const arma::u32_vec *no_interact_in = nullptr,
    const arma::u32_vec *fbs_ind_in = nullptr,
    const arma::u32_vec *sbs_ind_in = nullptr,
    const std::vector<bool> *subdiv_flag_in = nullptr);
```

### Inputs:
- **`mesh`** — Triangle mesh faces; see [obj_file_read](#obj_file_read); `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`. 0 = air
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl` (max 16383)
- **`center_frequency`** — Center frequencies in [Hz]; `[n_freq]`, 1 to 127 entries. `center_frequency[0]` is the reference frequency that defines the traced geometry
- **`Ox`**, **`Oy`**, **`Oz`** — Point-source (transmitter) position in GCS [m]; used to recompute path length at new sub-beam origins
- **`sub_mesh_index`** *(optional)* — Sub-mesh partition offsets for the accelerated intersect; 0-based, strictly increasing,
  first entry 0; passed to [ray_triangle_intersect](#ray_triangle_intersect); `[n_sub]`. NULL → no partitioning
- **`aabb`** *(optional)* — Axis-aligned bounding box per sub-mesh; `[n_sub, 6]`. Requires `sub_mesh_index`.
  NULL with a partition present → boxes are computed internally via [triangle_mesh_aabb](#triangle_mesh_aabb)
- **`max_no_interactions`** — Total interactions (segments) per ray before termination, 0 to 255. 0 disables tracing (returns 0 rays)
- **`max_no_reflections`** — Reflections per ray, 0 to 255. 0 skips the reflection pass
- **`max_no_transmissions`** — Transmissions / refractions per ray, 0 to 255. 0 skips the transmission pass
- **`max_no_subdivisions`** — Beam subdivisions per ray, 0 to 255. 0 (or no beam mode) disables subdivision.
  Not applied when `subdiv_flag_in` is given — the caller owns termination in that case, and the per-path
  counter `nSUB` saturates at 255 instead of wrapping
- **`min_gain_dB`** — Path gain below which a continuation is not launched, in dB (linear-power threshold applied to the accumulated per-path gain × interaction gain)
- **`subdivision_tolerance_m`** — Maximum beam-tube edge length before a ray is subdivided, in [m]; must be > 0
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold forwarded to [ray_state_update](#ray_state_update) as its `eps`; see there. Default 0.15
- **`refraction_mode`** — `true` = refraction (Snell-bent transmission), `false` = straight-path transmission
- **`scalar_mode`** — `true` = scalar (acoustic) layout, `false` = EM (2×2 Jones). Must match the layout of `paths`
- **`no_interact_in`** *(optional)* — Externally computed intersection count per ray from [ray_triangle_intersect](#ray_triangle_intersect); `[n_ray]`.
  When given, the internal intersector is skipped and `fbs_ind_in` and `sbs_ind_in` become mandatory
- **`fbs_ind_in`** *(optional)* — Externally computed 1-based index of the first intersected mesh element, 0 = none; `[n_ray]`.
  Must be non-zero wherever `no_interact_in` is non-zero.
- **`sbs_ind_in`** *(optional)* — Externally computed 1-based index of the second intersected mesh element, 0 = none; `[n_ray]`
- **`subdiv_flag_in`** *(optional)* — Flags that mark rays for subdivision; `[n_ray]`, indexed in the full ray
  set (unlike everything else, which is compacted first). When supplied it is the authority on what gets
  subdivided: a flagged ray is split regardless of whether it hit the mesh, how often it has already been
  subdivided, or whether it is travelling inside a medium, and `subdivision_tolerance_m` is unused. Pass the
  output of [ray_subdivide_flag](#ray_subdivide_flag) to reproduce the internal decision exactly; a hand-built list overrides it.
  Requires beam mode; conflicts with `max_no_subdivisions` = 0

### In/out (launch configuration, updated in place; [n_ray, …] on entry, [n_out, …] on return):
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`. Defines `n_ray`; must be non-empty
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mtl_ind_prev`**, **`mtl_ind_current`**, **`mtl_ind_buffer`** — Medium-state words (bit-masked: `mat = w & 0x3FFF`, flags in bits 14 and 15, see [ray_state_update](#ray_state_update)); `[n_ray]`
- **`path_dir_prev`** — Physical ray direction entering the current segment (unit vectors), one slice per
  frequency; `[n_ray, 3, n_freq]`. The Snell bend that defines this direction depends on the material at the
  evaluation frequency, so a dispersive medium gives a different direction in every slice. Outside a medium
  all slices hold the same geometric direction
- **`acc_dist`** — Accumulated in-layer distance; `[n_ray, 1 + n_freq]`; col 0 = geometric distance,
  col `1+f` = refracted distance at frequency `f`. The geometric column is shared because the traced polyline
  is the base-frequency geometry by construction; only the refracted length varies with frequency. Inside a
  medium both include the relaunch offset of the last event (a few float ulp of the coordinate, about 1 µm
  per metre), so the leg the next event measures from `orig` adds up to the leg from the hit point
- **`paths`** — Per-ray [path](channel-functions.md#path) objects; `n_ray` entries. Frequency count and layout must match
  `center_frequency` and `scalar_mode`. Terminated paths are freed; surviving paths carry the appended
  interaction segment and the updated polarization product
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`. Must be supplied together with `tridir`; empty / NULL disables beam tracing
- **`tridir`** *(optional)* — Vertex-ray directions, Cartesian; `[n_ray, 9]`

### Returns:
- Per-stage ray counts `{n_interact, n_subdiv, n_reflect, n_transmit}`:
  - **`n_interact`** — rays that hit the mesh, i.e. `no_interact != 0`, counted before any compaction
  - **`n_subdiv`** — rays flagged for subdivision, each expanded into 4 sub-beams. Not a subset of
    `n_interact`: a delegated flag may mark rays that missed the mesh, so the two counts are independent
  - **`n_reflect`** — reflected continuations launched
  - **`n_transmit`** — transmitted / refracted continuations launched
- Only rays that hit the mesh *and* were not subdivided reach the interaction passes, so
  `n_reflect + n_transmit` is bounded by twice that number, not by `2·n_interact`.

### See also:
- [ray_init](#ray_init) (produces the initial launch configuration this function advances)
- [ray_triangle_intersect](#ray_triangle_intersect) (first/second interaction points)
- [ray_mesh_interact](#ray_mesh_interact) (per-interaction Fresnel/Jones result)
- [ray_state_update](#ray_state_update) (inside/outside state machine and thin-slab resolution)
- [ray_subdivide_flag](#ray_subdivide_flag) (per-ray subdivision decision, delegated via `subdiv_flag_in`)
- [subdivide_rays](#subdivide_rays) (adaptive beam-tube refinement)
- [path](channel-functions.md#path) (per-ray storage object accumulated across generations)

---
<a id="ray_state_update"></a>
## ray_state_update
Per-ray medium state machine with closed-form thin-slab (Fabry-Pérot) resolution

Not in: MATLAB, Python

- Corrects the per-event `gainN` / `xprmatN` of [ray_mesh_interact](#ray_mesh_interact) with the medium each ray is known to be in,
  and carries that state from event to event in three state words per ray (table below) and the in-medium path
  lengths (`acc_dist_*`):
  - the face is re-evaluated at the incidence of the direction the wave has (`path_dir_prev`) instead of the
    traced segment (the virtual hit point, VBS), with total reflection and the continuation direction;
  - the in-medium attenuation and excess phase of the leg the ray leaves are applied;
  - the internal bounce series of a traced parallel layer is summed in closed form, `S = 1/(1 − r_near·r_far·φ²)`
    per polarization, where its round-trip amplitude `ρ = |r_near·r_far|·|φ|²` is at least `eps`; otherwise the
    bounces are left to the tracer as further events. A body's verdict is taken once, at the first event that
    closes a leg in it;
  - a thin-sheet crossing is re-resolved against the media the state reports (a sheet inside a medium has that
    medium on both sides; [ray_mesh_interact](#ray_mesh_interact) assumed air).
- Called twice per event by the tracer ([ray_progress](#ray_progress), [calc_diffraction_gain](#calc_diffraction_gain)): for the reflect pass
  (`interaction_type` 0 or 3) and for the transmit pass (1, 2, 4, 5), each with the outputs of
  [ray_mesh_interact](#ray_mesh_interact) for that type.
- Without state words every ray starts in air and each event is corrected on its own.

### Declaration:
```cpp
void quadriga_lib::ray_state_update(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbsN,
    const arma::Mat<dtype> *sbsN,
    const arma::u32_vec *no_interact,
    const arma::Col<dtype> *fbs_angleN,
    const arma::Mat<dtype> *normal_vecN,
    const std::vector<uint8_t> *out_typeN,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbsN,
    const arma::Col<short> *mtl_ind_sbsN,
    const arma::Col<short> *mtl_ind_prev_in = nullptr,
    const arma::Col<short> *mtl_ind_current_in = nullptr,
    const arma::Col<short> *mtl_ind_buffer_in = nullptr,
    const arma::Mat<dtype> *path_dir_prev = nullptr,
    const arma::Col<dtype> *acc_dist_geometric = nullptr,
    const arma::Col<dtype> *acc_dist_refracted = nullptr,
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    arma::Col<dtype> *acc_dist_geometricN = nullptr,
    arma::Col<dtype> *acc_dist_refractedN = nullptr,
    std::vector<uint8_t> *resolved_typeN = nullptr,
    const arma::u32_vec *ray_indN = nullptr,
    double eps = 0.15,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    const std::vector<uint8_t> *verdict_inN = nullptr,
    std::vector<uint8_t> *verdict_outN = nullptr);
```

### Inputs:
- **`interaction_type`** — 0 EM reflection, 1 EM transmission, 2 EM refraction, 3 scalar reflection, 4 scalar transmission, 5 scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin, destination, full ray set; `[n_ray, 3]`, read at `g = ray_indN[i]`
- **`fbsN`**, **`sbsN`**  — First and second interaction points, compact set; `[n_rayN, 3]`
- **`no_interact`** — Mesh-hit count per ray, full ray set; `[n_ray]`
- **`fbs_angleN`** — Incidence angle at FBS (the angle to the face, as [ray_mesh_interact](#ray_mesh_interact) writes it), compact set; `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normals `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`, compact set; `[n_rayN, 6]`; required.
  The plane of the VBS, and the test whether the two faces of an entry are parallel.
- **`out_typeN`** — Interaction type code from [ray_mesh_interact](#ray_mesh_interact), compact set; `[n_rayN]`. Its total-reflection
  bit is re-evaluated at the VBS angle; a thin-sheet code takes the sheet path.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); at most
  16383 materials.
- **`mtl_ind_fbsN`**, **`mtl_ind_sbsN`** — Material indices of the FBS / SBS faces, compact set; `[n_rayN]` (0 = air).
  The sheet case of a hit is re-derived from the code and the two faces' materials.
- **`mtl_ind_prev_in`**, **`mtl_ind_current_in`**, **`mtl_ind_buffer_in`** — State words, full ray set; `[n_ray]`,
  read at `g`, never written. NULL reads as state `0` (in air, no flags). Pass the words an earlier call wrote
  through unchanged.
   | Word                 | Bits 0-13                          | Bit 14                                        | Bit 15                    |
   | -------------------- | ---------------------------------- | --------------------------------------------- | ------------------------- |
   | `mtl_ind_current`    | medium the ray is in, 0 = air      | bent: the leg follows a bend traced with type 2 or 5; zero in air | the leg's series was summed |
   | `mtl_ind_prev`       | medium the leg was entered from, or the sheet crossed or reflected off | a sheet in this word lies inside the current medium; with a solid, the face into the current medium was evaluated at a real-invariant cosine | the leg's faces are not parallel |
   | `mtl_ind_buffer`     | deferred medium                    | followed, with a zero index: the leg began at a reflection that left the series to the tracer. With an index: the layer of an ordered stack behind the ray that the previous-medium word does not name | clear |
   | `mtl_ind_buffer`     | thickness code, bits 0-14          | (part of the code)                            | set: mass-crossed         |

   Under bit 15 the buffer word is the normal thickness, as a logarithmic code, of a mass-law leg closed on the way
   into the current one (bonded mass-law bodies are one plate); it is exempt from the material range check.
- **`path_dir_prev`** — Physical ray direction entering this segment, full ray set; `[n_ray, 3]`
- **`acc_dist_geometric`** — Accumulated geometric in-layer distance carried into this call, full ray set; `[n_ray]`.
  NULL reads as 0. Frequency-independent, so a multi-frequency caller keeps one instance.
- **`acc_dist_refracted`** — Accumulated refracted (VBS) in-layer distance carried into this call, full ray set; `[n_ray]`.
  NULL reads as 0. Frequency-dependent (the bend follows the material at `center_frequency`), so a
  multi-frequency caller keeps one instance per frequency.
- **`ray_indN`** — Compact-to-full ray index map; `[n_rayN]` to `[n_ray]`; NULL = identity (`n_ray == n_rayN`)
- **`eps`** *(optional)* — Threshold on the round-trip amplitude `ρ` (`ρ ≥ 0`, not bounded by one: next to a lossy or
  evanescent layer a face can reflect more than one in amplitude): a series is summed where `ρ ≥ eps` and left to
  the tracer where `ρ < eps`. `eps = 0` always sums (for callers that cannot re-emit); `eps ≥ 1` leaves every series
  with `ρ < eps` to the tracer. A layer whose faces are known not to be parallel is left to the tracer for any
  `eps > 0`. A thin sheet's own series is always summed (there is no interior to trace); `eps` gates the host layer
  a sheet lies in. Default `0.15`.
- **`verdict_inN`** *(optional)* — The verdict of each event as an earlier call on the same events wrote it to
  `verdict_outN`, compact set; `[n_rayN]`. Where given, a series is summed exactly where the entry is nonzero and
  `eps` is not read: a multi-frequency caller hands the base frequency's verdicts to the other frequencies, whose
  traced paths are the base frequency's. NULL: this call decides by `eps`.
- **`tridir`** *(optional)* — Vertex-ray directions of the beam entering this segment, Cartesian, as handed to
  [ray_mesh_interact](#ray_mesh_interact); `[n_ray, 9]`. Given together with `tridirN`; requires `origN` and `destN`

### Outputs:
- **`mtl_ind_prev_outN`**, **`mtl_ind_current_outN`**, **`mtl_ind_buffer_outN`** — Updated state words (layout
  above), compact set; `[n_rayN]`. The previous-medium word may hold a sheet: the boundary behind the ray, never a
  medium it travels in. NULL skips the write.
- **`gainN`** *(in/out)* — Per-interaction linear power gain, updated in place; `[n_rayN]`. Kept consistent
  with `xprmatN`. A killed ray (`resolved_typeN == 0`) has `gainN = 0` and an all-zero `xprmatN`.
- **`xprmatN`** *(in/out)* — Polarization transfer matrix, updated in place; `[8, n_rayN]` for EM mode,
  `[2, n_rayN]` for scalar mode, layout and basis as in [ray_mesh_interact](#ray_mesh_interact). On return it also carries the
  in-medium attenuation and excess phase of the leg the ray leaves and, where the series was summed, the factor
  `S`. Spreading loss stays excluded, so the matrix is the complete per-event Jones factor. Where the series is
  left to the tracer the matrix carries the first pass only, and the bounces arrive as further events. At a
  thin-sheet crossing the matrix is replaced by the composite for the media the state reports.
- **`path_dirN`** *(in/out)* — Continuation direction, corrected in place by the VBS construction, compact set; `[n_rayN, 3]`
- **`origN`**, **`destN`** *(optional, in/out)* — The relaunch segment [ray_mesh_interact](#ray_mesh_interact) built, compact set;
  `[n_rayN, 3]`, given together. With a refracting type (2, 5) that segment was bent at a real angle in the
  medium the ray leaves. Where this function writes the direction of a wave out of a lossy medium whose
  in-medium cosine it recovered (a bare exit, a glued exit, a transition into another solid), both points
  are put on one line along `path_dirN`, through the SBS of a co-located pair and the FBS otherwise. Every
  other ray, and every other type, is left untouched. A multi-frequency caller passes them on the call that
  defines the geometry only.
- **`tridirN`** *(optional, in/out)* — Vertex-ray directions leaving, Cartesian, compact set; `[n_rayN, 9]`.
  Corrected together with `origN` and `destN`: each vertex ray gets the construction of the spine from its
  own direction in `tridir`. If one of the three cannot be formed, the ray keeps the geometry it had.
- **`verdict_outN`** *(optional)* — Per event, 1 where this call summed a bounce series and 0 where it left the
  bounces to the tracer, compact set; `[n_rayN]`. NULL skips the write.
- **`acc_dist_geometricN`** — Accumulated geometric distance leaving this call, compact set; `[n_rayN]`. NULL skips the write.
  Where the ray goes on inside a medium it includes the relaunch offset of [ray_mesh_interact](#ray_mesh_interact) (`origN` lies that
  far past the point the ray left at), so the next leg, measured from `origN`, adds up to the leg from the hit point.
- **`acc_dist_refractedN`** — Accumulated refracted (VBS) distance leaving this call, compact set; `[n_rayN]`. NULL skips the write.
  It includes the relaunch offset the same way, projected onto the direction the wave has (exact for parallel faces).
- **`resolved_typeN`** *(optional)* — Resolved interaction-type code, bit-encoded (`qd::bits<uint8_t>`),
  compact set; `[n_rayN]`. 0 = ray killed. NULL skips the write.
   | Bit  | Flag        | Meaning                                                                        |
   | :--: | :---------: | ------------------------------------------------------------------------------ |
   |   0  | ok          | OK flag (0 = a deferred medium is pending in the buffer word)                  |
   |   1  | vbs         | VBS correction (gain/xprmat corrected at the VBS instead of FBS/SBS)           |
   |   2  | resolve     | Internal multi-bounce series accounted for (expect no further internal bounces) |
   |   3  | inside      | Inside-object flag (1 = ray continues inside, 0 = continues outside)           |
   |   4  | fix         | Fix flag (the exit face names another medium than the state, or a sheet event met a deferred medium) |
   |   5  | tir         | Total-reflection flag (also set when a transmission factor forced reflection)  |
   |   6  | trans/sheet | See the qualification below                                                    |
   |   7  | refl        | Reflection flag (0 = transmission/refraction, 1 = reflection/scattering)       |

   Bit 6 is qualified by bit 2 (a transparent interface sums no series, and scattering is not implemented):
   | Bit 6 | Bit 2 | Meaning                                                                       |
   | :---: | :---: | ----------------------------------------------------------------------------- |
   |   1   |   0   | Transparent interface (transmission) / scatter (reflection)                   |
   |   1   |   1   | **Thin-sheet crossing**                                                       |

   In the thin-sheet region bit 1 carries the glued verdict instead of the VBS flag (a sheet is always
   re-resolved at the VBS angle):
   | Bit 1 (sheet region) | Meaning                                                                |
   | :------------------: | ---------------------------------------------------------------------- |
   |          0           | Isolated — the same medium bounds both sides of the sheet              |
   |          1           | Glued — the two bounding media differ                                  |

   The sheet region reports the resolved state, not the sheet topology. The codes 69 and 71 below are unrelated
   to the `out_typeN` codes of the same value.
   Reachable composite values for transmission / refraction:
   |  Dec | Hex  | FIX  |   TIR   | Flags set                  | Meaning                                            |
   | :--: | :--: | :--: | :-----: | -------------------------- | -------------------------------------------------- |
   |    9 | 0x09 |   —  |    41   | inside, ok                 | o-i entry, OR i-i transition (refr. 2/5, FBS==VBS) |
   |    8 | 0x08 |   —  |    40   | inside                     | o-i entry, deferred buffer set (overlap/edge)      |
   |   11 | 0x0B |  27  |  43, 59 | inside, vbs, ok            | i-i transition (undev. 1/4, VBS relocated)         |
   |   13 | 0x0D |  29  |  45, 61 | inside, resolve, ok        | i-i transition + slab series (refr. 2/5, FBS==VBS) |
   |   15 | 0x0F |  31  |  47, 63 | inside, resolve, vbs, ok   | i-i transition + slab series (undev. 1/4, VBS)     |
   |    1 | 0x01 |  17  |  33, 49 | ok                         | i-o exit (refr. 2/5, FBS==VBS)                     |
   |    3 | 0x03 |  19  |  35, 51 | vbs, ok                    | i-o exit (undev. 1/4, VBS relocated)               |
   |    5 | 0x05 |  21  |  37, 53 | resolve, ok                | i-o exit + slab series (refr. 2/5, FBS==VBS)       |
   |    7 | 0x07 |  23  |  39, 55 | resolve, vbs, ok           | i-o exit + slab series (undev. 1/4, VBS)           |
   |   73 | 0x49 |  89  |     —   | trans, inside, ok          | ignore-hit / same-medium pass, no gain change      |
   |   72 | 0x48 |  88  |     —   | trans, inside              | nested pass-through, buffer deferred               |
   |   65 | 0x41 |   —  |     —   | trans, ok                  | advance to ray destination, identity interface     |
   |   69 | 0x45 |   85 |   101   | sheet, resolve, ok         | isolated sheet crossed in air                      |
   |   77 | 0x4D |   93 |   109   | sheet, resolve, inside, ok | isolated sheet crossed inside a medium (embedded)  |
   |   71 | 0x47 |   87 |   103   | sheet, glued, resolve, ok  | glued exit, solid \| sheet \| air                  |
   |   79 | 0x4F |   95 |   111   | sheet, glued, resolve, inside, ok | glued entry, host \| sheet \| solid         |

   Reachable composite values for reflection:
   |  Dec | Hex  |  FIX |   TIR   | Flags set                       | Meaning                                                      |
   | :--: | :--: | :--: | :-----: | ------------------------------- |------------------------------------------------------------- |
   |  129 | 0x81 |    — |   161   | refl, ok                        | eager front reflection (R0), outside (FBS==VBS)              |
   |  137 | 0x89 |  153 | 169,185 | refl, inside, ok                | internal back-reflection (incoming refr. 2/5, FBS==VBS)      |
   |  139 | 0x8B |  155 | 171,187 | refl, inside, vbs, ok           | internal back-reflection (incoming undev. 1/4, VBS)          |
   |  141 | 0x8D |  157 | 173,189 | refl, inside, resolve, ok       | internal back-reflection + slab series (incoming refr. 2/5)  |
   |  143 | 0x8F |  159 | 175,191 | refl, inside, resolve, vbs, ok  | internal back-reflection + slab series (incoming undev. 1/4) |
   |  197 | 0xC5 |  213 |   229   | refl, sheet, resolve, ok        | reflection off an isolated sheet, ray outside                |
   |  205 | 0xCD |  221 |   237   | refl, sheet, resolve, inside, ok | reflection off an isolated sheet, ray inside a medium       |
   |  199 | 0xC7 |  215 |   231   | refl, sheet, glued, resolve, ok | reflection off a glued sheet from the host side              |
   |  207 | 0xCF |  223 |   239   | refl, sheet, glued, resolve, inside, ok | reflection off a glued sheet from the solid side    |
   |  193 | 0xC1 |    — |    —    | refl, scatter, ok               | point scatterer of a GBSM path ([qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers)); ray-traced scattering is not implemented |

   A stack of two different sheets is crossed as one event with the codes of a sheet; a singular stack kills the ray.

### See also:
- [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md) (companion document)
- [ray_mesh_interact](#ray_mesh_interact) (computes the per-interaction Fresnel/Jones result this function corrects)
- [calc_diffraction_gain](#calc_diffraction_gain) (uses this function for its media)
- [ray_progress](#ray_progress) (the tracer that calls it)

---
<a id="ray_subdivide_flag"></a>
## ray_subdivide_flag
Compute the per-ray subdivision decision from the beam-tube footprint at the first bounce

Not in: MATLAB, Python

- Projects the three vertex rays of each beam tube onto the plane of the first-bounce face and
  measures the longest edge of the resulting wavefront triangle
- Flags a ray when that edge exceeds `subdivision_tolerance_m` and the ray is still eligible to be
  split, i.e. it travels outside a medium and has not reached its subdivision or interaction limit
- The decision is purely geometric: the vertex origins on the face are the same for reflection,
  transmission and refraction, so no material data, second-bounce index or frequency is needed
- This is the single source of truth for the subdivision decision. [ray_progress](#ray_progress) consumes the
  result when it is passed in and calls this function itself otherwise, so a caller that needs to
  know the outcome in advance — a shading pass that must not commit beams which will reappear as
  sub-beams — gets exactly the set that will actually be split

### Declaration:
```cpp
std::vector<bool> quadriga_lib::ray_subdivide_flag(
    const arma::fmat &mesh,
    const arma::fmat &orig,
    const arma::fmat &dest,
    const arma::u32_vec &fbs_ind,
    const arma::fmat &trivec,
    const arma::fmat &tridir,
    const std::vector<quadriga_lib::path> &paths,
    const arma::Col<short> &mtl_ind_current,
    uint8_t max_no_interactions = 20,
    uint8_t max_no_subdivisions = 2,
    float subdivision_tolerance_m = 3.0f);
```

### Inputs:
- **`mesh`** — Faces of the triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`. Defines `n_ray`; must be non-empty
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`fbs_ind`** — 1-based index of the first intersected mesh element, 0 = no hit; `[n_ray]`. Obtained from [ray_triangle_intersect](#ray_triangle_intersect) for the same `orig` / `dest` pair
- **`trivec`** — Beam wavefront triangle vertices relative to the ray origin; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, Cartesian; `[n_ray, 9]`. Need not be unit length
- **`paths`** — Per-ray [path](channel-functions.md#path) objects; `n_ray` entries. Only the subdivision counter `nSUB` and the segment count are read
- **`mtl_ind_current`** — Current medium state word, 0 = outside; `[n_ray]`. A ray inside a medium is
  never split: the sub-beams restart their in-layer accumulator and recompute their direction
  geometrically, which is only valid outside
- **`max_no_interactions`** *(optional)* — Total interactions per ray, 0 to 255. A ray that has already reached the limit 
  is not split, so it does not expand into four sub-beams that all terminate in the next generation
- **`max_no_subdivisions`** *(optional)* — Number of subdivisions per ray, 0 to 255. 0 disables subdivision and 
  the returned flags are all `false`
- **`subdivision_tolerance_m`** *(optional)* — Maximum beam-tube edge length before a ray is split, in
  metres; must be greater than 0

### Output:
- **`subdiv_flag`** — `true` where the ray must be split; `n_ray` entries. Rays that miss the mesh
  (`fbs_ind = 0`) are always `false`. A beam whose tube only partially covers the face — a vertex ray
  running parallel to it, pointing away from it, or intersecting absurdly far away — is treated as
  having an infinite edge and is always flagged

### See also:
- [ray_progress](#ray_progress) (advance one generation of a beam-traced ray set)
- [ray_triangle_intersect](#ray_triangle_intersect) (compute `fbs_ind`)
- [subdivide_rays](#subdivide_rays) (split the flagged beams into sub-beams)
- [ray_mesh_interact](#ray_mesh_interact) (reports the same edge length as `edge_lengthN`)

---
<a id="ray_triangle_intersect"></a>
## ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#ray_triangle_intersect) · [Python](../python/site-specific-simulation-tools.md#ray_triangle_intersect)

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

### Declaration:
```cpp
void quadriga_lib::ray_triangle_intersect(
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *mesh,
    arma::Mat<dtype> *fbs = nullptr,
    arma::Mat<dtype> *sbs = nullptr,
    arma::u32_vec *no_interact = nullptr,
    arma::u32_vec *fbs_ind = nullptr,
    arma::u32_vec *sbs_ind = nullptr,
    const arma::u32_vec *sub_mesh_index = nullptr,
    const arma::Mat<dtype> *aabb = nullptr,
    int use_kernel = 0,
    int gpu_id = 0);
```

### Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `[n_mesh, 9]`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; `[n_sub]`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; if `nullptr`, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

### Outputs:
- **`fbs`** (optional) — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** (optional) — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`no_interact`** (optional) — Total number of intersections per ray between `orig` and `dest`; `[n_ray]`
- **`fbs_ind`** (optional) — 1-based index of first intersected mesh element; 0 = none; `[n_ray]`
- **`sbs_ind`** (optional) — 1-based index of second intersected mesh element; 0 = none; `[n_ray]`
- `fbs` and `sbs` may be the same objects as `orig` or `dest` (e.g. to advance rays in place)

### See also:
- [obj_file_read](#obj_file_read) (load mesh from OBJ file)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (compute sub-mesh indices and AABBs)
- [ray_point_intersect](#ray_point_intersect) (beam interactions with sampling points)
- [icosphere](#icosphere) (generate ray beams)
- [subdivide_rays](#subdivide_rays) (split ray beams into sub-beams)

---
<a id="refractive_index"></a>
## refractive_index
Real refractive index of a homogeneous medium

Not in: MATLAB, Python

- Returns `n = Re √(ε_r·μ_r)` of material `iM`, with the ITU-R P.2040 model `ε_r = a·(f/fRef)^b − j·17.98·σ/f[GHz]`,
  `σ = c·(f/fRef)^d`, and `μ_r` from `e`, `f`, `g`, `h` in the same form. For a material without the mass column
  the permittivity resonance (`resF`, `resQ`, `resS`) is part of `ε_r`.
- For a mass-law material (`m > 0`) `ε_r` and `μ_r` are an impedance mapping and `n` is not a propagation index.
- Air (`iM = 0`) returns `1`.

### Declaration:
```cpp
dtype quadriga_lib::refractive_index(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype center_frequency);
```

### Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl`
- **`iM`** — 1-based material index (0 = no material / air)
- **`center_frequency`** — Center frequency in [Hz]

### Returns:
- Real refractive index of the medium relative to air

### See also:
- [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md) (the formulas and the columns)
- [medium_gain](#medium_gain) (for the distance-dependent in-medium loss)
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
<a id="subdivide_rays"></a>
## subdivide_rays
Subdivide ray beams into four smaller sub-beams

Not in: MATLAB, Python

- Each triangular beam is split into 4 sub-beams, `n_rayN = 4 * n_subdiv` rays are written to the output
- Rays can be selected by `index` (0-based list), all rays are subdivided if it is not given
- `n_subdiv` is the number of selected rays
- `tridir` format is auto-detected: spherical `[n_ray, 6]` or Cartesian `[n_ray, 9]`, output matches the input format
- Pre-allocated outputs that can hold all new rays are reused as they are and must have the same number of rows,
  the new rays are written to the first `n_rayN` rows, leaving the remaining rows untouched. Smaller buffers are
  re-allocated to that size, discarding their content.
- Internal math is done in double precision, the new origins stay within 1 ULP of the wavefront plane
  spanned by `orig` and `trivec`, no offset is applied along the propagation direction
- The direction values of the 3 original vertices are passed through unchanged, only the 3 new edge-midpoint
  directions are calculated, hence repeated subdivision does not accumulate rounding errors at the corners
- If `transposed_output` is true, all outputs are written transposed, i.e. the rays are in the columns
  and the components in the rows, e.g. `origN` becomes `[3, n_rayN]`

### Declaration:
```cpp
arma::uword quadriga_lib::subdivide_rays(
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &trivec,
    const arma::Mat<dtype> &tridir,
    const arma::Mat<dtype> *dest = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    const arma::u32_vec *index = nullptr);
```

### Inputs:
- **`orig`** — Ray origin points in GCS; `[n_ray, 3]`
- **`trivec`** — Vectors from origin to the wavefront vertices, columns `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, spherical `[v1az v1el v2az v2el v3az v3el]` or Cartesian
  `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 6]` or `[n_ray, 9]`
- **`dest`** (optional) — Ray destination points, ignored if empty; `[n_ray, 3]`
- **`index`** (optional) — 0-based indices of the rays that should be subdivided, may repeat indices and
  determines the output order; Invalid indices raise an exception after the loop has finished,
  outputs may be partially written by then; `[n_subdiv]`
- **`transposed_output`** (optional) — If true, the outputs are written transposed with the rays in the
  columns, e.g. `origN` becomes `[3, n_rayN]`; default = `false`

### Outputs:
- **`origN`** — Subdivided ray origins, centroids of the sub-beam wavefronts; `[n_rayN, 3]` or `[3, n_rayN]` for `transposed_output`
- **`trivecN`** — Subdivided wavefront vectors, relative to `origN`; `[n_rayN, 9]` or `[9, n_rayN]` for `transposed_output`
- **`tridirN`** — Subdivided vertex-ray directions, same format as `tridir`; `[n_rayN, 6]` or `[n_rayN, 9]` or `[6/9, n_rayN]` for `transposed_output`
- **`destN`** — Subdivided destinations, left untouched if `dest` was `nullptr` or empty; `[n_rayN, 3]` or `[3, n_rayN]` for `transposed_output`

### Returns:
- `n_rayN` — Number of rays written to the output, `4 * n_subdiv`

### See also:
- [icosphere](#icosphere) (generate initial beams)
- [ray_point_intersect](#ray_point_intersect) (beam-sample-point interaction)
- [ray_triangle_intersect](#ray_triangle_intersect) (beam-triangle interaction)

---
<a id="subdivide_triangles"></a>
## subdivide_triangles
Subdivide triangles into smaller triangles

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#subdivide_triangles) · Not in: Python

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in x n_div x n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles
- `triangles_out` may be the same object as `triangles_in`, and `mtl_ind_out` the same as `mtl_ind` (in place)
- Throws `std::invalid_argument` when `n_div` is so large that the number of output values exceeds 64 bits

### Declaration:
```cpp
arma::uword quadriga_lib::subdivide_triangles(
    arma::uword n_div,
    const arma::Mat<dtype> *triangles_in,
    arma::Mat<dtype> *triangles_out,
    const arma::uvec *mtl_ind = nullptr,
    arma::uvec *mtl_ind_out = nullptr);
```

### Inputs:
- **`n_div`** — Number of subdivisions per edge
- **`triangles_in`** — Mesh vertices as `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_triangles_in, 9]`
- **`mtl_ind`** — Material indices per triangle (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_triangles_in]`

### Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_ind_out`** — Material indices for subdivided triangles; `[n_triangles_out]`

### Returns:
- `n_triangles_out` — Number of generated triangles

---
<a id="triangle_mesh_aabb"></a>
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#triangle_mesh_aabb) · [Python](../python/site-specific-simulation-tools.md#triangle_mesh_aabb)

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Declaration:
```cpp
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(
    const arma::Mat<dtype> *mesh,
    const arma::u32_vec *sub_mesh_index = nullptr,
    arma::uword vec_size = 1);
```

### Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `[n_triangles, 9]`
- **`sub_mesh_index`** *(optional)* — 0-based start indices of sub-meshes; if omitted, the AABB of the entire mesh is returned; `[n_sub]`
- **`vec_size`** *(optional)* — Alignment size for SIMD/CUDA padding (e.g., `8` for AVX2, `32` for CUDA)

### Returns:
- `arma::Mat<dtype>` of shape `[n_sub_aligned, 6]`, one AABB per sub-mesh row

### See also:
- [ray_triangle_intersect](#ray_triangle_intersect) (consumer of the output)

---
<a id="triangle_mesh_segmentation"></a>
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

Also in: [MATLAB](../mex/site-specific-simulation-tools.md#triangle_mesh_segmentation) · [Python](../python/site-specific-simulation-tools.md#triangle_mesh_segmentation)

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark padding entries
- If `mtl_ind` is provided, material indices are reordered and padded the same way (padding uses index 0)

### Declaration:
```cpp
arma::uword triangle_mesh_segmentation(
    const arma::Mat<dtype> *mesh,
    arma::Mat<dtype> *meshR,
    arma::u32_vec *sub_mesh_index,
    arma::uword target_size = 1024,
    arma::uword vec_size = 1,
    const arma::uvec *mtl_ind = nullptr,
    arma::uvec *mtl_ind_out = nullptr,
    arma::u32_vec *mesh_index = nullptr);
```

### Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`target_size`** *(optional)* — Target triangle count per sub-mesh; for best performance set near `sqrt(n_mesh)`
- **`vec_size`** *(optional)* — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count rounded up to a multiple of this value
- **`mtl_ind`** *(optional)* — Material indices per triangle (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`

### Outputs:
- **`meshR`** — Reordered and padded triangle vertices; `[n_meshR, 9]`
- **`sub_mesh_index`** — 0-based start indices of sub-meshes in `meshR`; `[n_sub]`
- **`mtl_ind_out`** *(optional)* — Reordered and padded material indices (padding = 0); `[n_meshR]`
- **`mesh_index`** *(optional)* — 1-based row in `mesh` of each row of `meshR` (0 = padding); `[n_meshR]`

### Returns:
- Number of created sub-meshes `n_sub`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (uses `sub_mesh_index` for acceleration)
- [obj_file_read](#obj_file_read) (defines `mtl_ind` / `csv_ind`)

---
<a id="triangle_mesh_split"></a>
## triangle_mesh_split
Split a 3D triangular mesh into two sub-meshes along a given axis

Not in: MATLAB, Python

- Splits at the bounding box center of the selected axis; triangles where all vertices lie within the 
  lower half go to `meshA`; any triangle with at least one vertex exceeding the threshold goes to `meshB`
- `axis = 0` selects the axis with the longest bounding box extent automatically
- On failure (all triangles fall to one side), `meshA` and `meshB` are left unchanged and the return value is negative
- Used internally by [triangle_mesh_segmentation](#triangle_mesh_segmentation)

### Declaration:
```cpp
int triangle_mesh_split(
    const arma::Mat<dtype> *mesh,
    arma::Mat<dtype> *meshA,
    arma::Mat<dtype> *meshB,
    int axis = 0,
    arma::Col<int> *split_ind = nullptr);
```

### Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`axis`** *(optional)* — Split axis: 0 = longest extent, 1 = x, 2 = y, 3 = z

### Outputs:
- **`meshA`** — Triangles with all vertices within the lower half of the bounding box; `[n_meshA, 9]`
- **`meshB`** — Triangles with at least one vertex exceeding the split threshold; `[n_meshB, 9]`
- **`split_ind`** *(optional)* — Per-triangle assignment: 1 = meshA, 2 = meshB, 0 = unassigned (failure); `[n_mesh]`

### Returns:
- Axis used for the split (1, 2, or 3); negative (-1, -2, or -3) on failure

### See also:
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (calls this function recursively)

---
<a id="write_png"></a>
## write_png
Write a data matrix to a color-coded PNG file

Also in: [MATLAB](../mex/miscellaneous-tools.md#write_png) · [Python](../python/miscellaneous-tools.md#write_png)

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NAN`
- The auto-detected range ignores `NaN` and `+-Inf`; `+Inf` and `-Inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value; also an infinite `min_val` or `max_val`) maps every value to the first colour
- Throws `std::invalid_argument` for an empty `data` matrix and for `min_val > max_val` (both given)
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

### Declaration:
```cpp
void quadriga_lib::write_png(
    const arma::Mat<dtype> &data,
    std::string fn,
    std::string colormap = "jet",
    dtype min_val = NAN,
    dtype max_val = NAN,
    bool log_transform = false);
```

### Inputs:
- **`data`** — Input data matrix
- **`fn`** — Output `.png` file path
- **`colormap`** *(optional)* — Colormap name; see [colormap](#colormap) for valid values
- **`min_val`** *(optional)* — Lower clipping bound; auto-detected if `NAN`
- **`max_val`** *(optional)* — Upper clipping bound; auto-detected if `NAN`
- **`log_transform`** *(optional)* — Apply 10*log10(data) before mapping; non-positive values map to the minimum color

---
<a id="xpr_update"></a>
## xpr_update
Thread a polarization transfer (Jones) matrix along a ray path

Not in: MATLAB, Python

- Left-multiplies each per-ray state matrix by a new interaction matrix, vectorized across up to
  hundreds of millions of rays, then optionally normalizes the touched columns and re-imposes a
  separately tracked scalar gain — all in a single pass
- One entry point covers every operation needed while threading a chain: initialization, updating,
  gain extraction, normalization, and gain application. They are selected independently through
  `initialize`, `update`, `normalize`, `gain`, and `apply_gain`, so a call can do any combination
  (see Usage)
- An optional `ray_index` lets the update set be a subset of the global state, scattered into
  `xprmat` by column. Without it the mapping is 1:1 and `n_ray == n_rayU`
- Storage is interleaved complex, column-major, matching the `xprmat` output of
  [calc_diffraction_gain](#calc_diffraction_gain). EM mode carries the full 2x2 Jones matrix; scalar mode carries a
  single complex pressure coefficient

### Declaration:
```cpp
void xpr_update(
    arma::Mat<dtype> &xprmat,
    const arma::Mat<dtype> *update = nullptr,
    arma::Col<dtype> *gain = nullptr,
    bool initialize = false,
    bool normalize = false,
    bool apply_gain = false,
    const arma::uvec *ray_index = nullptr);
```

### Arguments:
- **`xprmat`** — Global XPR state, updated in place. EM mode `[8, n_ray]`, scalar mode
  `[2, n_ray]`. If empty on entry it is sized from `update` and seeded to unity (see Usage). A
  pre-sized (non-empty) buffer is treated as accumulated state unless `initialize` is set. When
  `ray_index` is given, `xprmat` must already exist, because the extent of the global set cannot be
  inferred from a subset.
- **`update`** — Interaction matrix left-multiplied onto each ray of the update set. Layout must
  match `xprmat` (8 or 2 rows). Shapes: `[8 or 2, n_rayU]` applies one matrix per update; `[8 or 2, 1]`
  broadcasts one matrix to every update; `nullptr` or empty applies the identity (no multiply).
- **`gain`** — Dual-purpose, length `n_rayU`; direction set by `apply_gain`:
  - `apply_gain == false` (output): resized to `[n_rayU]` and filled with each ray's gain *before*
    normalization. With `normalize == false` this measures the gain without modifying the matrix
    part of `xprmat` (gain extraction); with `normalize == true` it reports the gain that was
    removed. A value of 0 flags a degenerate (zero / non-finite) matrix the caller should prune.
  - `apply_gain == true` (input): read, not resized; each touched column is multiplied by the
    corresponding `gain` value *after* normalization. Must be non-null and length `n_rayU`.
- **`initialize`** — If `true`, each touched column is set to unity (identity for EM, `[1,0]` for
  scalar) before the update. With an `update` this is a copy-through (the column becomes the
  update, prior contents discarded); without an `update` it is a pure reset. Use this to
  (re)initialize a pre-sized or reused buffer without reallocating.
- **`normalize`** — If `true`, each touched column is scaled to unit gain after the update.
  Untouched columns of `xprmat` are left alone.
- **`apply_gain`** — Flips `gain` from output to input (see above). Applying a gain requires a `gain` vector.
- **`ray_index`** — Optional map from update index (`0..n_rayU-1`) to global column of `xprmat`.
  Length `n_rayU`. Default is the identity map, which requires `n_ray == n_rayU`. Values must be
  `< n_ray`. Duplicate values are allowed: the updates of a column are then applied one after the other
  in update-index order (each with the full initialize -> update -> normalize -> apply_gain sequence),
  so the result is the same as a single-threaded run for any thread count. With duplicates the call
  runs single-threaded; unique indices keep the parallel loop.

### Usage:
The flags are composable in one call; the per-column order is always
initialize -> update -> normalize -> apply_gain. Argument order is
`(xprmat, update, gain, initialize, normalize, apply_gain, ray_index)`.
```cpp
arma::Mat<dtype> M;                                 // ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH (EM), labels <out><in>

// (1) Initialization
xpr_update(M, &U0);                                 // empty M: sized from U0, seeded, M := U0
xpr_update(M, nullptr, nullptr, true);              // pre-sized M: reset every column to unity
xpr_update(M, &U0, nullptr, true);                  // pre-sized M: copy-through, M := U0 (ignores old)

// (2) Updating (accumulate onto existing state)
xpr_update(M, &U);                                  // M := U * M
xpr_update(M, &Usub, nullptr, false, false, false, &idx);   // scatter: M[:,idx[k]] := Usub[:,k] * M[:,idx[k]]

// (3) Gain extraction (measure, leave the matrix part untouched)
arma::Col<dtype> g;
xpr_update(M, nullptr, &g);                         // g[k] = gain( M[:,k] )
xpr_update(M, &U, &g);                              // apply U, then measure into g (no normalize)

// (4) Normalizing (scale each column to unit gain; optionally update first)
xpr_update(M, nullptr, nullptr, false, true);       // M[:,k] := M[:,k] / gain( M[:,k] )
xpr_update(M, &U, nullptr, false, true);            // update, then normalize, one pass

// (5) Normalizing + gain application (strip the matrix's own magnitude, impose a tracked scalar).
//     'g' is read here (apply_gain = true):
xpr_update(M, nullptr, &g, false, true, true);      // M[:,k] := g[k] * ( M[:,k] / gain( M[:,k] ) )
xpr_update(M, &U, &g, false, true, true, &idx);     // full chain step: scatter+update+normalize+apply
```

### Storage layout (column-major, interleaved complex):
- **EM mode**, 8 values per ray: `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, so column 0 of the
  2x2 is `[VV; HV]` (response to a V input) and column 1 is `[VH; HH]` (response to an H input).
  The labels are `<out><in>`: the first letter is the output (receive) polarisation, the second the input
  (transmit) polarisation, so rows 0-1 = (V out, V in), rows 2-3 = (H out, V in), rows 4-5 = (V out, H in),
  rows 6-7 = (H out, H in). This is the row order of `M` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (rows = RX, columns = TX).
- **Scalar mode**, 2 values per ray: `[Re Im]`.
- The update is `M_out = M_update * M_state` (new interaction left-multiplies), consistent with
  `E_out = M * E_in` for column vectors and Armadillo/MATLAB column-major storage.

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (produces `xprmat` in this exact layout)
