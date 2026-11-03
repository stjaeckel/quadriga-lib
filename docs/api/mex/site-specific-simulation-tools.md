# Site-specific simulation tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-site-specific-simulation-tools

| Function | Summary |
| --- | --- |
| [calc_diffraction_gain](#calc_diffraction_gain) | Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh |
| [cube](#cube) | Construct a triangulated cube mesh |
| [generate_diffraction_paths](#generate_diffraction_paths) | Generate elliptic propagation paths and weights for diffraction gain estimation |
| [icosphere](#icosphere) | Construct a geodesic polyhedron from recursive icosahedron subdivision |
| [obj_file_read](#obj_file_read) | Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials |
| [obj_file_write](#obj_file_write) | Write a Wavefront .obj file |
| [plane](#plane) | Construct a triangulated plane mesh |
| [point_cloud_aabb](#point_cloud_aabb) | Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud |
| [point_cloud_segmentation](#point_cloud_segmentation) | Reorganize a point cloud into spatial sub-clouds for efficient processing |
| [point_inside_mesh](#point_inside_mesh) | Test whether 3D points are inside a triangle mesh using raycasting |
| [ray_mesh_interact](#ray_mesh_interact) | Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces |
| [ray_point_intersect](#ray_point_intersect) | Calculate intersections of ray beams with points in 3D space |
| [ray_triangle_intersect](#ray_triangle_intersect) | Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm |
| [subdivide_triangles](#subdivide_triangles) | Subdivide triangles into smaller triangles |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing |

---
<a id="calc_diffraction_gain"></a>
## calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

Also in: [C++](../cpp/site-specific-simulation-tools.md#calc_diffraction_gain) · [Python](../python/site-specific-simulation-tools.md#calc_diffraction_gain)

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided
  into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping
  triangles whose bounding box does not intersect the TX-RX path
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```matlab
[ gain, xprmat, coord ] = quadriga_lib.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, ...
    center_freq, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode, thin_slab_threshold, ...
    apply_fspl, apply_phase );
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output of
  [obj_file_read](#obj_file_read)); each field holds a vector of length `n_mtl`
- **`center_freq`** — Center frequency in Hz. In scalar mode this is the true acoustic frequency;
  it is converted internally to the EM-equivalent frequency used by the core
- **`lod`** — Level of detail (0–7), controls `n_path` and `n_seg`; 0 = direct ray only (`n_path = 1`, `n_seg = 1`);
  1–7 as in the table of [generate_diffraction_paths](#generate_diffraction_paths); default: 2
- **`verbose`** — Verbosity level; default: 0 (no output)
- **`sub_mesh_index`** — 1-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation);  `[n_mesh, 1]`;
  default: `[]` (not using sub-meshes)
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0
- **`scalar_mode`** — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode).
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold; 0 = resolve always (default), 
  1 = resolve never
- **`apply_fspl`** — If `true`, folds the geometric spreading loss into `gain`, making it an absolute
  propagation gain instead of a value relative to FSPL. The reference distance is the direct TX-RX
  distance, matching the `path_length` used by [qrt_file_read](channel-functions.md#qrt_file_read) for a
  transmission-only path. EM mode: `32.45 + 20*log10(f_GHz) + 20*log10(d_m)` dB; scalar mode:
  `20*log10(d_m) + alpha(f)*d_m` dB, with `alpha` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm; default: `false`
- **`apply_phase`** — If `true`, multiplies `xprmat` by the propagation phase `exp(-j*2*pi*d/lambda)`
  using the same direct TX-RX distance. The wavelength follows from `center_freq` and the medium wave
  speed (speed of light in EM mode, speed of sound in scalar mode). The term has unit magnitude, so
  `xprmat` stays normalized. This is the phase [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) would otherwise add from
  `path_length`; default: `false`

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `[n_pos, 1]`. Relative to FSPL by
  default (bounded by 0 and 1), absolute propagation gain when `apply_fspl = true`
- **`xprmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase; `gain` holds the power. EM mode:
  interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx), the row order
  of `M` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical); `[8, n_pos]`; scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]` with `n_seg` from `lod`
  (e.g. `[3, 2, n_pos]` for lod 2, `[3, 9, n_pos]` for lod 7, `[3, 0, n_pos]` for lod 0)

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines `mtl_prop` format)
- [qrt_file_read](channel-functions.md#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

---
<a id="cube"></a>
## cube
Construct a triangulated cube mesh

Also in: [C++](../cpp/site-specific-simulation-tools.md#cube) · [Python](../python/site-specific-simulation-tools.md#cube)

### Description:
- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1)
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 · n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with obj_file_write
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- A negative scale mirrors the cube; when the product of the scales is negative, the winding is swapped so that
  the normals stay outward (as Blender's "Apply Scale")

### Usage:
```matlab
mesh = quadriga_lib.cube( scale, rotation, location, n_div );
```

### Inputs:
- **`scale`** *(optional)* — Length 1 scales all axes uniformly; length 3 scales {x,y,z}
  independently; empty or omitted = 1 (no scaling)
- **`rotation`** *(optional)* — Euler angles about {x,y,z}, applied as R = Rz·Ry·Rx (Blender
  XYZ); length 3; empty or omitted = no rotation
- **`location`** *(optional)* — Translation {x,y,z}; length 3; empty or omitted = origin
- **`n_div`** *(optional)* — Number of subdivisions per edge; yields 12 · n_div^2 triangles;
  default: 1

### Outputs:
- **`mesh`** — Triangle mesh; each row holds {x1,y1,z1,x2,y2,z2,x3,y3,z3}; `[12 · n_div^2, 9]`

### See also:
- [icosphere](#icosphere)
- [obj_file_write](#obj_file_write)

---
<a id="generate_diffraction_paths"></a>
## generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

Also in: [C++](../cpp/site-specific-simulation-tools.md#generate_diffraction_paths) · Not in: Python

- Generates inputs required by [calc_diffraction_gain](#calc_diffraction_gain): elliptic-arc paths sampling
  the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the
  semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so
  sum(prod(weights,3),2) = 1

### Usage:
```matlab
[ rays, weights ] = quadriga_lib.generate_diffraction_paths( orig, dest, center_frequency, lod );
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`center_frequency`** — Center frequency in Hz
- **`lod`** — Level of detail; controls `n_path` and `n_seg`:
   | `lod` | `n_path` | `n_seg` | Note  |
   | :---: | -------: | ------: | ----: |
   | 1     | 7        | 3       | -     |
   | 2     | 19       | 3       | -     |
   | 3     | 37       | 4       | -     |
   | 4     | 61       | 5       | -     |
   | 5     | 1        | 2       | debug |
   | 6     | 2        | 2       | debug |
   | 7     | 1        | 10      | direct line in 10 segments |

### Outputs:
- **`rays`** — Coordinates of path waypoints (x, y, z stacked along the 4th dimension, endpoints
  excluded); `[n_pos, n_path, n_seg-1, 3]`
- **`weights`** — Per-segment weights; `[n_pos, n_path, n_seg]`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (consumes the output of this function)

---
<a id="icosphere"></a>
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

Also in: [C++](../cpp/site-specific-simulation-tools.md#icosphere) · [Python](../python/site-specific-simulation-tools.md#icosphere)

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Usage:
```matlab
[ center, length, vert, direction ] = quadriga_lib.icosphere( n_div, radius, direction_xyz );
```

### Inputs:
- **`n_div`** — Number of subdivisions; generates 20 · n_div² faces; default: 1
- **`radius`** — Radius of icosphere in meters; finite and positive; default: 1
- **`direction_xyz`** — Output directions in Cartesian (true) or spherical azimuth/elevation (false); default: false

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `[n_faces, 3]`
- **`length`** — Distance from origin to face plane; equals the magnitude of each `center` vector; `[n_faces]`
- **`vert`** — Vertex offsets from face center `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_faces, 9]`
- **`direction`** — Directions of the vertex rays; spherical `{az1,el1,az2,el2,az3,el3}` or Cartesian
  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per `direction_xyz` flag; `[n_faces, 6]` or `[n_faces, 9]`

---
<a id="obj_file_read"></a>
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

Also in: [C++](../cpp/site-specific-simulation-tools.md#obj_file_read) · [Python](../python/site-specific-simulation-tools.md#obj_file_read)

### Description:
- Parses a triangulated `.obj`; quads and n-gons are rejected. Face references may be `v`, `v/vt`, `v//vn` or
  `v/vt/vn`; a negative (relative) index counts back from the last vertex defined before the face line (`-1` = that
  vertex). Index 0, or an index beyond the vertices defined before the face, raises an error, for any requested outputs
- Lines may end in LF or CRLF (`.obj`, `.mtl` and material CSV)
- Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- `usemtl (null)` (written by [obj_file_write](#obj_file_write)) marks the following faces as having no material: index 0 in
  `mtl_ind` and `csv_ind` (also with `csv_strict`); `(null)` is not listed in `mtl_names` or `bsdf`
- Unmatched names throw when `csv_strict` is true; otherwise they map to index 0 (no material)
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```matlab
[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read( fn, fn_csv, csv_strict );
```

### Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** *(optional)* — Path to an EM/acoustic material CSV; must contain a `name` column;
  unmatched faces map to index 0 (no material) unless `csv_strict` is set; empty uses the built-in default table
- **`csv_strict`** *(optional)* — If true, throw when a `usemtl` material is absent from the table;
  otherwise map to index 0 (no material); default: false

### Outputs:
- **`mesh`** — Triangle vertex coordinates `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 1-based vertex indices into `vert_list` per triangle; uint64; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; uint64; `[n_mesh]`
- **`obj_names`** — Object names; cell array of strings; length `max(obj_ind)`
- **`mtl_ind`** — 1-based visual-material index per triangle (0 = no material); uint64; `[n_mesh]`
- **`mtl_names`** — Visual material names (raw `usemtl`); cell array of strings; length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[no_mtl, 17]`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle (0 = no material); uint64; `[n_mesh]`
- **`csv_names`** — Material names from the table; cell array of strings; length `n_csv`
- **`csv_prop`** — Material properties as a struct; each field is one CSV column (excluding `name`)
  holding a column vector of length `n_csv`

### See also:
- [obj_file_write](#obj_file_write) (for writing OBJ files)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (used to calculate indexed mesh for faster processing)
- [ray_mesh_interact](#ray_mesh_interact) (calculating interactions between rays and the triangular mesh)

---
<a id="obj_file_write"></a>
## obj_file_write
Write a Wavefront .obj file

Also in: [C++](../cpp/site-specific-simulation-tools.md#obj_file_write) · [Python](../python/site-specific-simulation-tools.md#obj_file_write)

- Supply geometry as either `mesh`, or as `vert_list` and `face_ind`; giving both, or neither, is  an error
- With `mesh`, `vert_list_out` and `face_ind_out` are derived from it, merging vertices of the same
  object that are closer than `threshold` (no merging across objects)
- With `vert_list` and `face_ind`, the geometry is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind` and `obj_names`, a single object named `object` is written
- With `split_loose_parts`, each object is split into connected components ("separate by loose parts");
  faces sharing a vertex belong to the same part, and parts of a split object are named `name.001`, `name.002`, ...
- Without `mtl_ind`, no `usemtl` tags and no `.mtl` file are written
- A face with `mtl_ind = 0` that follows a face with a material in the same object (or part) is preceded by
  `usemtl (null)`, which [obj_file_read](#obj_file_read) reads back as 0; the `.mtl` gets no `(null)` entry. Face order and objects
  are unchanged; a material named `(null)` that is used by a face is rejected with an error (unused entries of
  `mtl_names` may carry the name)
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray
  material when `bsdf` is omitted
- Duplicate entries in `mtl_names` are merged into one `.mtl` entry if their `bsdf` rows are identical (or if
  `bsdf` is omitted); duplicates with differing `bsdf` rows are written as `name.001`, `name.002`, ...
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the
  `.obj`); columns follow a fixed canonical order then any extra `csv_prop` fields; `csv_write_defaults` also
  emits canonical columns absent from `csv_prop`, filled with their defaults (`a`, `e`, `fRef` = 1, `side` = 3.29,
  else 0)

### Usage:
```matlab
[ vert_list_out, face_ind_out ] = quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, ...
    mtl_names, vert_list, face_ind, bsdf, threshold, csv_ind, csv_names, csv_prop, csv_write_defaults, ...
    split_loose_parts );
```

### Inputs:
- **`fn`** — Path to the output `.obj` file; must end in `.obj`; if empty, no file is written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{X1,Y1,Z1,...,X3,Y3,Z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list` and `face_ind`
- **`obj_ind`** — 1-based object index per face; `[n_mesh]`; each object must form a contiguous block
- **`mtl_ind`** — 1-based material index per face (0 = no material); `[n_mesh]`; omit or pass `[]` for no materials
- **`obj_names`** — Object names; cell array of strings; length >= max(obj_ind); required if `obj_ind` is given
- **`mtl_names`** — Material names; cell array of strings; length >= max(mtl_ind); required if `mtl_ind` is given
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only valid with `face_ind`; written unchanged
- **`face_ind`** — 1-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF values for the `.mtl` file; `[n_mtl, 17]`; see [obj_file_read](#obj_file_read) for the column layout
- **`threshold`** — Vertex co-location distance for merging within an object; default: 0.001 (1 mm)
- **`csv_ind`** *(optional)* — 1-based EM/acoustic-material index per face (0 = no material); `[n_mesh]`; validated if given
- **`csv_names`** *(optional)* — EM/acoustic material names; cell array of strings; required to write the `.csv`
- **`csv_prop`** *(optional)* — Material properties as a struct; each field is one CSV column holding a vector of length `numel(csv_names)`
- **`csv_write_defaults`** *(optional)* — If true, also write canonical columns absent from `csv_prop` using their defaults; default: false
- **`split_loose_parts`** *(optional)* — If true, split each object into connected components; default: false

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 1-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)

---
<a id="plane"></a>
## plane
Construct a triangulated plane mesh

Also in: [C++](../cpp/site-specific-simulation-tools.md#plane) · [Python](../python/site-specific-simulation-tools.md#plane)

### Description:
- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices
  at +/-1, z = 0)
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 · n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with obj_file_write
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- When the product of the x and y scales is negative (mirrored plane), the winding is swapped so that the normals
  still point in +z direction before the rotation; the z scale is ignored

### Usage:
```matlab
mesh = quadriga_lib.plane( scale, rotation, location, n_div );
```

### Inputs:
- **`scale`** *(optional)* — Length 1 scales x and y uniformly; length 2 scales {x,y}
  independently; length 3 is accepted for compatibility with `cube`, but the third element is
  ignored; empty or omitted = 1 (no scaling)
- **`rotation`** *(optional)* — Euler angles about {x,y,z}, applied as R = Rz·Ry·Rx (Blender
  XYZ); length 3; empty or omitted = no rotation
- **`location`** *(optional)* — Translation {x,y,z}; length 3; empty or omitted = origin
- **`n_div`** *(optional)* — Number of subdivisions per edge; yields 2 · n_div^2 triangles;
  default: 1

### Outputs:
- **`mesh`** — Triangle mesh; each row holds {x1,y1,z1,x2,y2,z2,x3,y3,z3}; `[2 · n_div^2, 9]`

### See also:
- [cube](#cube)
- [icosphere](#icosphere)
- [obj_file_write](#obj_file_write)

---
<a id="point_cloud_aabb"></a>
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

Also in: [C++](../cpp/site-specific-simulation-tools.md#point_cloud_aabb) · [Python](../python/site-specific-simulation-tools.md#point_cloud_aabb)

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_index` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Usage:
```matlab
aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index, vec_size );
```

### Inputs:
- **`points`** — 3D point coordinates; `[n_points, 3]`
- **`sub_cloud_index`** — Row indices marking the start of each sub-cloud; use [point_cloud_segmentation](#point_cloud_segmentation) 
  to generate; uint32; `[n_sub]`; default: `[]` (not using sub-clouds)
- **`vec_size`** — SIMD alignment padding factor (e.g. 4, 8, 16); default: 1

### Outputs:
- **`aabb`** — Bounding box matrix; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

### See also:
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud indices)
- [ray_point_intersect](#ray_point_intersect) (use AABBs for intersection)

---
<a id="point_cloud_segmentation"></a>
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

Also in: [C++](../cpp/site-specific-simulation-tools.md#point_cloud_segmentation) · [Python](../python/site-specific-simulation-tools.md#point_cloud_segmentation)

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes
  at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are
  placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

### Usage:
```matlab
[ pointsR, sub_cloud_index, forward_index, reverse_index ] = ...
    quadriga_lib.point_cloud_segmentation( points, target_size, vec_size );
```

### Inputs:
- **`points`** — Original 3D point cloud; `[n_points, 3]`
- **`target_size`** — Maximum points per sub-cloud before padding; default: 1024
- **`vec_size`** — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`;
  [ray_point_intersect](#ray_point_intersect) accepts any value, padding (e.g. 8 for AVX2) only improves its speed; default: 1

### Outputs:
- **`pointsR`** — Reorganized point cloud with points grouped by sub-cloud; `[n_pointsR, 3]`
- **`sub_cloud_index`** — 1-based starting index of each sub-cloud within `pointsR`; `[n_sub]`
- **`forward_index`** — 1-based row in `points` of each row of `pointsR`; padding entries are `0`; `[n_pointsR]`
- **`reverse_index`** — 1-based row in `pointsR` of each row of `points`; `[n_points]`

---
<a id="point_inside_mesh"></a>
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

Also in: [C++](../cpp/site-specific-simulation-tools.md#point_inside_mesh) · [Python](../python/site-specific-simulation-tools.md#point_inside_mesh)

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided, returns the 1-based enclosing object index instead of binary 0/1

### Usage:
```matlab
result = quadriga_lib.point_inside_mesh( points, mesh, obj_ind, distance );
```

### Inputs:
- **`points`** — 3D coordinates of test points; `[n_points, 3]`
- **`mesh`** — Triangle faces in row-major vertex format `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** — Object index per mesh element; 1-based; enables per-object output; `[n_mesh]`
- **`distance`** — Surface proximity threshold; points within this distance of the mesh surface are
  classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1); range: 0–20 m; Default: 0

### Output:
- `**result**`— Indicator: `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object
  index (with `obj_ind`); size `[n_points]`

---
<a id="ray_mesh_interact"></a>
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

Also in: [C++](../cpp/site-specific-simulation-tools.md#ray_mesh_interact) · Not in: Python

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`)
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule);
  front-side hit with FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air;
  FBS=SBS with opposing normals → media-to-media
- With `compact = true` (default), rays with `fbs_ind = 0` (no interaction) are dropped, so
  `n_rayN ≤ n_ray`; with `compact = false` all rays are kept and no-hit rays pass through unchanged
- Output direction encoding (spherical/Cartesian) matches input `tridir` format
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves)
- Types 3–5 (scalar) use a single TE-only coefficient, suitable for acoustic simulation with
  impedance-mapped material parameters (ε derived from Z); total internal reflection is handled as
  in the EM path
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```matlab
[ origN, destN, fbsN, sbsN, gainN, xprmatN, trivecN, tridirN, fbs_angleN, thicknessN, ...
    edge_lengthN, normal_vecN, out_typeN, path_dirN, ray_indN ] = ...
    quadriga_lib.ray_mesh_interact( interaction_type, center_frequency, orig, dest, mesh, ...
    mtl_ind, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, compact );
```

### Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction,
  3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_frequency`** — Center frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [obj_file_read](#obj_file_read); `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `csv_ind` output of
  [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output
  of [obj_file_read](#obj_file_read)); each field holds a vector of length `n_mtl`
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); uint32; `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; order
  `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 9]`; default: `[]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical
  `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian; default: `[]`
- **`compact`** *(optional)* — If true, rays with no interaction (`fbs_ind = 0`) are dropped so
  `n_rayN <= n_ray`; if false, all rays are kept (`n_rayN = n_ray`) and no-hit rays are returned as a
  transparent pass-through; logical; default: true

### Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale); averaged over TE/TM polarizations for types 0–2,
  single TE coefficient for types 3–5; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex, column-major
  `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`; includes interaction gain, TE/TM coefficients,
  incidence plane orientation; excludes in-medium attenuation and FSPL; `[8, n_rayN]`. For types
  3–5 (scalar): `[Re Im]` where Re+jIm is the scalar pressure coefficient; `[2, n_rayN]`
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if
  `trivec`/`tridir` not provided
- **`fbs_angleN`** — Incidence angle at FBS; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (Inf if partial hit);
  requires the `trivec` and `tridir` inputs, returns all zeros without them; `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (uint32); `[n_rayN]`
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag (1 = front: o→i or M2 hit first; 0 = back: i→o or M1)   |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |

   Reachable composite values (add 32 for the total-reflection variant):
   | Code  |  TIR  | Description                                         |
   | :---: | :---: | --------------------------------------------------- |
   |   0   |   —   | No hit                                              |
   |   1   |  33   | Single hit, inside→outside (exit)                   |
   |   3   |  35   | Single hit, outside→inside (entry)                  |
   |   5   |  37   | Media-to-media, M1 (current, back) hit first        |
   |   7   |  39   | Media-to-media, M2 (next, front) hit first          |
   |  13   |  45   | Overlapping faces, inside-inside→outside            |
   |  15   |  47   | Overlapping faces, outside→inside-inside            |
   |  21   |  53   | Corner hit, inside→outside→inside                   |
   |  23   |  55   | Corner hit, outside→inside→outside                  |
   |  29   |  61   | Corner hit, inside-inside→outside                   |
   |  31   |  63   | Corner hit, outside→inside-inside                   |
- **`path_dirN`** — Refraction-correct continuation direction: mirror for types 0/3, Snell direction
  for types 1/2/4/5; for undeviated transmission (types 1/4) this is the refracted direction, which
  differs from the geometric continuation used for `origN`/`destN`; `[n_rayN, 3]`
- **`ray_indN`** — 1-based input ray index for each output ray (inverse of the compaction map,
  order-preserving); equals `1:n_ray` when `compact = false`; uint32; `[n_rayN]`

### See also:
- [obj_file_read](#obj_file_read) (for loading `mesh` and `mtl_prop` from OBJ file)
- [icosphere](#icosphere) (for generating beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for computing FBS and SBS positions)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)

---
<a id="ray_point_intersect"></a>
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

Also in: [C++](../cpp/site-specific-simulation-tools.md#ray_point_intersect) · [Python](../python/site-specific-simulation-tools.md#ray_point_intersect)

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the
  origin, enabling energy spread simulation
- Returns, for each point, the list of ray indices whose beam intersects that point
- All internal computations use single precision

### Usage:
```matlab
[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, ...
    sub_cloud_index, use_kernel, gpu_id );
```

### Inputs:
- **`orig`** — Ray origin positions in global Cartesian coordinates; `[n_ray, 3]`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `[n_ray, 9]`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `[n_ray, 9]`
- **`points`** — 3D point cloud coordinates; `[n_points, 3]`
- **`sub_cloud_index`** — Segment boundary indices for the point cloud; use [point_cloud_segmentation](#point_cloud_segmentation) to generate;
  sub-clouds need not be aligned to the SIMD vector size (padding with `vec_size` is optional and only improves speed);
  uint32; `[n_sub]`; default: `[]` (not using sub-clouds)
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable
  and for other values; auto mode selects CUDA when `n_points >= 500` and CUDA is available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

### Outputs:
- **`hit_count`** — Number of beams intersecting each point; `[n_points, 1]`
- **`ray_ind`** — Per-point list of 1-based ray indices that intersected that point; zero-padded to
  a regular 2D array (zero entries indicate unused slots); uint32; `[max_hits, n_points]`

### See also:
- [icosphere](#icosphere) (generate ray beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud index)
- [ray_triangle_intersect](#ray_triangle_intersect) (ray–triangle intersection)
- [ray_mesh_interact](#ray_mesh_interact) (beam–mesh interaction)

---
<a id="ray_triangle_intersect"></a>
## ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

Also in: [C++](../cpp/site-specific-simulation-tools.md#ray_triangle_intersect) · [Python](../python/site-specific-simulation-tools.md#ray_triangle_intersect)

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

### Usage:
```matlab
[ fbs, sbs, no_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( ...
    orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id );
```

### Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `[n_mesh, 9]`
- **`sub_mesh_index`** — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; 
  use [triangle_mesh_segmentation](#triangle_mesh_segmentation) to generate; `[n_sub]`; default: `[]` (not using sub-meshes)
- **`aabb`** — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; 
  if empty or omitted, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable 
  and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

### Outputs:
- **`fbs`** — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`no_interact`** — Total number of intersections per ray between `orig` and `dest`; uint32; `[n_ray]`
- **`fbs_ind`** — 1-based index of first intersected mesh element; 0 = none; uint32; `[n_ray]`
- **`sbs_ind`** — 1-based index of second intersected mesh element; 0 = none; uint32; `[n_ray]`

### See also:
- [obj_file_read](#obj_file_read) (load mesh from OBJ file)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (compute sub-mesh indices)
- [triangle_mesh_aabb](#triangle_mesh_aabb) (compute AABBs)
- [ray_point_intersect](#ray_point_intersect) (beam interactions with sampling points)
- [icosphere](#icosphere) (generate ray beams)

---
<a id="subdivide_triangles"></a>
## subdivide_triangles
Subdivide triangles into smaller triangles

Also in: [C++](../cpp/site-specific-simulation-tools.md#subdivide_triangles) · Not in: Python

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in · n_div · n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles

### Usage:
```matlab
[ triangles_out, mtl_ind_out ] = quadriga_lib.subdivide_triangles( triangles_in, n_div, mtl_ind_in );
```

### Inputs:
- **`triangles_in`** — Mesh vertices as `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; `[n_triangles_in, 9]`
- **`n_div`** — Number of subdivisions per edge
- **`mtl_ind_in`** — 1-based material index per triangle (the `csv_ind` output of [obj_file_read](#obj_file_read));
  0 = no material; `[n_triangles_in]`; passed through unchanged; default: `[]`

### Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_ind_out`** — Material indices for the subdivided triangles, 1-based; uint64; `[n_triangles_out]`; only
  populated if `mtl_ind_in` is given

---
<a id="triangle_mesh_aabb"></a>
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

Also in: [C++](../cpp/site-specific-simulation-tools.md#triangle_mesh_aabb) · [Python](../python/site-specific-simulation-tools.md#triangle_mesh_aabb)

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding
  non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Usage:
```matlab
aabb = quadriga_lib.triangle_mesh_aabb( mesh, sub_mesh_index, vec_size );
```

### Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `[n_triangles, 9]`
- **`sub_mesh_index`** — Start indices of sub-meshes in `mesh`; use [triangle_mesh_segmentation](#triangle_mesh_segmentation) to generate; 
  `[n_sub]` or empty; default: `[]` (returns AABB of the entire mesh)
- **`vec_size`** — Alignment size for SIMD/CUDA padding (e.g., 8 for AVX2); default: 1

### Output:
- **`aabb`** — Axis-aligned bounding boxes, one row per sub-mesh; `[n_sub_aligned, 6]`

### See also:
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (for calculating sub-meshes)

---
<a id="triangle_mesh_segmentation"></a>
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

Also in: [C++](../cpp/site-specific-simulation-tools.md#triangle_mesh_segmentation) · [Python](../python/site-specific-simulation-tools.md#triangle_mesh_segmentation)

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark
  padding entries
- If `mtl_ind_in` is provided, material indices are reordered and padded in the same way

### Usage:
```matlab
[ triangles_out, sub_mesh_index, mesh_index, mtl_ind_out ] = ...
    quadriga_lib.triangle_mesh_segmentation( triangles_in, target_size, vec_size, mtl_ind_in );
```

### Inputs:
- **`triangles_in`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`target_size`** — Target triangle count per sub-mesh; for best performance set near sqrt(n_mesh); default: 1024
- **`vec_size`** — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count is rounded
  up to a multiple of this value; default: 1
- **`mtl_ind_in`** — 1-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read));
  0 = no material; `[n_mesh]` or empty; passed through unchanged; default: `[]`

### Outputs:
- **`triangles_out`** — Reordered and padded triangle vertices; `[n_meshR, 9]`
- **`sub_mesh_index`** — 1-based start indices of sub-meshes in `triangles_out`; uint32; `[n_sub]`
- **`mesh_index`** — 1-based row in `triangles_in` of each row of `triangles_out` (0 = padding); uint32; `[n_meshR]`
- **`mtl_ind_out`** — Reordered and padded material indices, 1-based; padding entries get 0 (no material); uint64;
  `[n_meshR]`; only populated if `mtl_ind_in` is given
