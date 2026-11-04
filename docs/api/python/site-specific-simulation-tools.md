# Site-specific simulation tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-site-specific-simulation-tools

| Function | Summary |
| --- | --- |
| [calc_diffraction_gain](#calc_diffraction_gain) | Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh |
| [cube](#cube) | Construct a triangulated cube mesh |
| [icosphere](#icosphere) | Construct a geodesic polyhedron from recursive icosahedron subdivision |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Write a triangular mesh to a Mitsuba 3 XML scene file |
| [obj_file_read](#obj_file_read) | Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials |
| [obj_file_write](#obj_file_write) | Write a Wavefront .obj file |
| [plane](#plane) | Construct a triangulated plane mesh |
| [point_cloud_aabb](#point_cloud_aabb) | Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud |
| [point_cloud_segmentation](#point_cloud_segmentation) | Reorganize a point cloud into spatial sub-clouds for efficient processing |
| [point_inside_mesh](#point_inside_mesh) | Test whether 3D points are inside a triangle mesh using raycasting |
| [ray_point_intersect](#ray_point_intersect) | Calculate intersections of ray beams with points in 3D space |
| [ray_triangle_intersect](#ray_triangle_intersect) | Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing |

---
<a id="calc_diffraction_gain"></a>
## calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

Also in: [C++](../cpp/site-specific-simulation-tools.md#calc_diffraction_gain) · [MATLAB](../mex/site-specific-simulation-tools.md#calc_diffraction_gain)

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided
  into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping
  triangles whose bounding box does not intersect the TX-RX path
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```python
# Output as tuple
data = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, \
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode, \
    thin_slab_threshold, apply_fspl, apply_phase )

# Unpacked outputs
gain, xprmat, coord = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, \
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode, \
    thin_slab_threshold, apply_fspl, apply_phase )
```

### Inputs:
- **`orig`** — TX positions; `(n_pos, 3)`
- **`dest`** — RX positions; `(n_pos, 3)`
- **`mesh`** — Triangle vertices, each row `[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]`; `(n_mesh, 9)`
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `csv_ind` output of [obj_file_read](#obj_file_read)); `(n_mesh,)`
- **`mtl_prop`** — Material properties as a `dict`; each key is one column (the `csv_prop` output of [obj_file_read](#obj_file_read)) mapping to a 1D array of length `n_mtl`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail (0–7), controls `n_path` and `n_seg`: 0 = direct ray only (1 path, 1 segment);
  1 / 2 = 7 / 19 paths with 3 segments; 3 = 37 paths, 4 segments; 4 = 61 paths, 5 segments; 5 / 6 = 1 / 2 paths
  with 2 segments (debug); 7 = direct line in 10 segments; default: 2
- **`verbose`** — Verbosity level; default: 0
- **`sub_mesh_index`** — 0-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation); `(n_mesh,)` or `None`; default: `None`
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0
- **`scalar_mode`** — If `True`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging; default: `False`
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold; 0 = resolve always, 1 = resolve
  never; default: 0
- **`apply_fspl`** — If `True`, folds the geometric spreading loss into `gain`, making it an absolute
  propagation gain instead of a value relative to FSPL. The reference distance is the direct TX-RX
  distance, matching the `path_length` used by [qrt_file_read](channel-functions.md#qrt_file_read) for a
  transmission-only path. EM mode: `32.45 + 20*log10(f_GHz) + 20*log10(d_m)` dB; scalar mode:
  `20*log10(d_m) + alpha(f)*d_m` dB, with `alpha` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm.;
  default: `False`
- **`apply_phase`** — If `True`, multiplies `xprmat` by the propagation phase `exp(-j*2*pi*d/lambda)`
  using the same direct TX-RX distance. The wavelength follows from `center_frequency` and the medium
  wave speed (speed of light in EM mode, speed of sound in scalar mode). The term has unit magnitude,
  so `xprmat` stays normalized. This is the phase [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) would otherwise add from
  `path_length`; default: `False`

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `(n_pos,)`. Relative to FSPL by
  default (bounded by 0 and 1), absolute propagation gain when `apply_fspl = True`
- **`xprmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase; `gain` holds the power. EM mode:
  interleaved complex, column-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx), the row
  order of `M` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical), `(8, n_pos)`; scalar
  mode: scalar pressure coefficient `[Re Im]`, `(2, n_pos)`
- **`coord`** — Diffracted path coordinates excluding endpoints; `(3, n_seg-1, n_pos)` with `n_seg` from `lod`
  (e.g. `(3, 2, n_pos)` for lod 2, `(3, 9, n_pos)` for lod 7, `(3, 0, n_pos)` for lod 0)

### See also:
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines the material format)
- [qrt_file_read](channel-functions.md#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

---
<a id="cube"></a>
## cube
Construct a triangulated cube mesh

Also in: [C++](../cpp/site-specific-simulation-tools.md#cube) · [MATLAB](../mex/site-specific-simulation-tools.md#cube)

- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1)
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 · n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with obj_file_write
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- A negative scale mirrors the cube; when the product of the scales is negative, the winding is swapped so that
  the normals stay outward (as Blender's "Apply Scale")

### Usage:
```python
mesh = quadriga_lib.RTtools.cube( scale, rotation, location, n_div )
```

### Inputs:
- **`scale`** — Length 1 scales all axes uniformly; length 3 scales (x, y, z) independently;
  `None` or empty = 1 (no scaling); default: `None`
- **`rotation`** — Euler angles about (x, y, z), applied as R = Rz·Ry·Rx (Blender XYZ); shape `(3,)`;
  `None` or empty = no rotation; default: `None`
- **`location`** — Translation (x, y, z); shape `(3,)`; `None` or empty = origin; default: `None`
- **`n_div`** — Number of subdivisions per edge; yields 12 · n_div^2 triangles; default: 1

### Outputs:
- **`mesh`** — Triangle mesh; each row holds (x1, y1, z1, x2, y2, z2, x3, y3, z3); `(12 · n_div^2, 9)`

---
<a id="icosphere"></a>
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

Also in: [C++](../cpp/site-specific-simulation-tools.md#icosphere) · [MATLAB](../mex/site-specific-simulation-tools.md#icosphere)

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Usage:
```python
# Output as tuple
data = quadriga_lib.RTtools.icosphere( no_div, radius, direction_xyz )

# Unpacked outputs
center, length, vert, direction = quadriga_lib.RTtools.icosphere( no_div, radius, direction_xyz )
```

### Inputs:
- **`n_div`** — Number of subdivisions; generates 20 · n_div² faces; default: 1
- **`radius`** — Radius of icosphere in meters; finite and positive; default: 1
- **`direction_xyz`** *(optional)* — Output directions in Cartesian (true) or spherical
  azimuth/elevation (false); default: false

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `(n_faces, 3)`
- **`length`** — Distance from origin to face plane; equals the magnitude of each
  `center` vector; `(n_faces,)`
- **`vert`** — Vertex offsets from face center `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_faces, 9)`
- **`direction`** — Directions of the vertex rays; spherical `{az1,el1,az2,el2,az3,el3}` or Cartesian
  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per `direction_xyz` flag; `(n_faces, 6)` or `[n_faces, 9)`

---
<a id="mitsuba_xml_file_write"></a>
## mitsuba_xml_file_write
Write a triangular mesh to a Mitsuba 3 XML scene file

Also in: [C++](../cpp/site-specific-simulation-tools.md#mitsuba_xml_file_write) · Not in: MATLAB

- Converts quadriga-lib mesh data structures to Mitsuba 3 XML format, loadable by NVIDIA Sionna RT for
  differentiable radio-propagation simulations
- Supports grouping faces into named objects with per-face material assignments
- Optionally maps material names to ITU-defined presets used by Sionna RT
- Creates a subdirectory `<stem>_meshes/` next to the XML file and writes one binary PLY file per object into it;
  both the XML and the mesh folder must be distributable together; the XML refers to each PLY file by its path
  relative to the XML file (`<stem>_meshes/<object>.ply`)
- Objects whose faces reference more than one material are automatically split into sub-objects (one per material)
  and renamed `<obj_name>_<mtl_name>`; the effective object count in the output may therefore exceed the length of `obj_names`
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```python
quadriga_lib.RTtools.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf, map_to_itu )
```

### Input Arguments:
- **`fn`** — Output file path including `.xml` extension
- **`vert_list`** — Vertex coordinates (x, y, z); `(n_vert, 3)`
- **`face_ind`** — Triangle definitions as 0-based vertex indices; uint64; `(n_mesh, 3)`
- **`obj_ind`** — 0-based object index per triangle; length `obj_names` must equal `max(obj_ind) + 1`; uint64; `(n_mesh,)`
- **`mtl_ind`** — 1-based material index per triangle (0 = no material); length `mtl_names` must be ≥ `max(mtl_ind)`; uint64; `(n_mesh,)`
- **`obj_names`** — Object names; list of strings; length must equal `max(obj_ind) + 1`
- **`mtl_names`** — Material names; list of strings; length must be ≥ `max(mtl_ind)` (1-based, so material `w` → `mtl_names[w-1]`)
- **`bsdf`** *(optional)* — BSDF material parameters per material; ignored by Sionna RT, used only by Mitsuba renderer; see [obj_file_read](#obj_file_read) for field definitions; `(mtl_names.size(), 17)`
- **`map_to_itu`** *(optional)* — If `true`, maps material names to ITU presets recognised by Sionna RT

### See also:
- [obj_file_read](#obj_file_read) (source for mesh data and BSDF field layout)

---
<a id="obj_file_read"></a>
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

Also in: [C++](../cpp/site-specific-simulation-tools.md#obj_file_read) · [MATLAB](../mex/site-specific-simulation-tools.md#obj_file_read)

- Parses a triangulated `.obj`; quads and n-gons are rejected. Face references may be `v`, `v/vt`, `v//vn` or
  `v/vt/vn`; a negative (relative) index counts back from the last vertex defined before the face line (`-1` = that
  vertex). Index 0, or an index beyond the vertices defined before the face, raises a `ValueError`, for any requested outputs
- Lines may end in LF or CRLF (`.obj`, `.mtl` and material CSV)
- Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- `usemtl (null)` (written by [obj_file_write](#obj_file_write)) marks the following faces as having no material: index 0 in
  `mtl_ind` and `csv_ind` (also with `csv_strict`); `(null)` is not listed in `mtl_names` or `bsdf`
- Unmatched names raise an error when `csv_strict = True`; otherwise they map to index 0 (no material)
- Geometry indices (`face_ind`, `obj_ind`) are 0-based; material indices (`mtl_ind`, `csv_ind`) are
  1-based, with 0 reserved for the outside / no-material state
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```python
mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop = \
    quadriga_lib.RTtools.obj_file_read( fn, fn_csv, csv_strict )
```

### Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** — Path to an EM/acoustic material CSV; must contain a `name` column. Unmatched faces map
  to index 0 (no material) unless `csv_strict` is set; empty uses the built-in default table; default: `""`
- **`csv_strict`** — If `True`, raise when a `usemtl` material is absent from the table; otherwise map to 
  index 0 (no material); default: `False`

### Outputs:
- **`mesh`** — Triangle vertex coordinates `[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]` per row; `(n_mesh, 9)`
- **`vert_list`** — All vertex positions in the file; `(n_vert, 3)`
- **`face_ind`** — 0-based vertex indices into `vert_list` per triangle; `(n_mesh, 3)`
- **`obj_ind`** — 0-based object index per triangle; `(n_mesh,)`
- **`obj_names`** — Object names; list of `str`; length `max(obj_ind) + 1`
- **`mtl_ind`** — 1-based visual-material index per triangle (0 = no material); `(n_mesh,)`
- **`mtl_names`** — Visual material names (raw `usemtl`); list of `str`; length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `(no_mtl, 17)`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle (0 = no material); `(n_mesh,)`
- **`csv_names`** — Material names from the table; list of `str`; length `n_csv`
- **`csv_prop`** — Material properties as a `dict`; each key is one CSV column (excluding `name`)
  mapping to a 1D array of length `n_csv`

### See also:
- [obj_file_write](#obj_file_write) (for writing OBJ files)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (used to calculate indexed mesh for faster processing)

---
<a id="obj_file_write"></a>
## obj_file_write
Write a Wavefront .obj file

Also in: [C++](../cpp/site-specific-simulation-tools.md#obj_file_write) · [MATLAB](../mex/site-specific-simulation-tools.md#obj_file_write)

- Supply geometry as either `mesh`, or as `vert_list` and `face_ind`; giving both, or neither, is an error
- With `mesh`, `vert_list_out` and `face_ind_out` are derived from it, merging vertices of the same object
  that are closer than `threshold` (no merging across objects)
- With `vert_list` and `face_ind`, the geometry is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- With `split_loose_parts`, each object is split into connected components ("separate by loose parts");
  faces sharing a vertex belong to the same part, and parts of a split object are named `name.001`, `name.002`, ...
- Without `mtl_ind`, no `usemtl` tags and no `.mtl` file are written
- A face with `mtl_ind = 0` that follows a face with a material in the same object (or part) is preceded by
  `usemtl (null)`, which [obj_file_read](#obj_file_read) reads back as 0; the `.mtl` gets no `(null)` entry. Face order and objects
  are unchanged; a material named `(null)` that is used by a face is rejected with a `ValueError` (unused entries of
  `mtl_names` may carry the name)
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray material when `bsdf` is omitted
- Duplicate entries in `mtl_names` are merged into one `.mtl` entry if their `bsdf` rows are identical (or if `bsdf`
  is omitted); duplicates with differing `bsdf` rows are written as `name.001`, `name.002`, ...
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the
  `.obj`); columns follow a fixed canonical order then any extra `csv_prop` fields (alphabetical);
  `csv_write_defaults` also emits canonical columns absent from `csv_prop`, filled with their defaults
  (`a`, `e`, `fRef` = 1, `side` = 3.29, else 0)
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](../../formats.md)

### Usage:
```python
vert_list_out, face_ind_out = quadriga_lib.RTtools.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, \
    mtl_names, vert_list, face_ind, bsdf, threshold, csv_ind, csv_names, csv_prop, csv_write_defaults, \
    split_loose_parts )
```

### Inputs:
- **`fn`** — Path to the output `.obj` file; must end in `.obj`; if empty, no file is written (outputs are still computed); default: `""`
- **`mesh`** — Triangle coordinates `{X1,Y1,Z1,...,X3,Y3,Z3}` per row; `(n_mesh, 9)`; mutually exclusive with `vert_list` and `face_ind`; default: None
- **`obj_ind`** — 0-based object index per face; `(n_mesh,)`; each object must form a contiguous block; default: None
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `mtl_ind`/`csv_ind` output of [obj_file_read](#obj_file_read)); `(n_mesh,)`; omit (None) for no materials; default: None
- **`obj_names`** — Object names; list of str; length > max(obj_ind); required if `obj_ind` is given; default: None
- **`mtl_names`** — Material names; list of str; length > max(mtl_ind); required if `mtl_ind` is given; default: None
- **`vert_list`** — Vertex positions; `(n_vert, 3)`; only valid with `face_ind`; written unchanged; default: None
- **`face_ind`** — 0-based vertex indices per face; `(n_mesh, 3)`; required with `vert_list`; default: None
- **`bsdf`** — Principled BSDF values for the `.mtl` file; `(n_mtl, 17)`; see [obj_file_read](#obj_file_read) for the column layout; default: None
- **`threshold`** — Vertex co-location distance for merging within an object; default: 0.001 (1 mm)
- **`csv_ind`** — 1-based EM/acoustic-material index per face (0 = no material); `(n_mesh,)`; validated if given; default: None
- **`csv_names`** — EM/acoustic material names (the full table); list of str; required to write the `.csv`; default: None
- **`csv_prop`** — Material properties as a `dict`; each key is one CSV column mapping to a 1D array of length `len(csv_names)`; default: None
- **`csv_write_defaults`** — If True, also write canonical columns absent from `csv_prop` using their defaults; default: False
- **`split_loose_parts`** — If True, split each object into connected components; default: False

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `(n_vert, 3)`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `(n_mesh, 3)`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)
- [mitsuba_xml_file_write](#mitsuba_xml_file_write) (for exporting to Mitsuba scene file format)

---
<a id="plane"></a>
## plane
Construct a triangulated plane mesh

Also in: [C++](../cpp/site-specific-simulation-tools.md#plane) · [MATLAB](../mex/site-specific-simulation-tools.md#plane)

- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices at +/-1, z = 0)
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 · n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with obj_file_write
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- When the product of the x and y scales is negative (mirrored plane), the winding is swapped so that the normals
  still point in +z direction before the rotation; the z scale is ignored

### Usage:
```python
mesh = quadriga_lib.RTtools.plane( scale, rotation, location, n_div )
```

### Inputs:
- **`scale`** — Length 1 scales x and y uniformly; length 2 scales (x, y) independently; length 3 is
  accepted for compatibility with `cube`, but the third element is ignored; `None` or empty = 1
  (no scaling); default: `None`
- **`rotation`** — Euler angles about (x, y, z), applied as R = Rz·Ry·Rx (Blender XYZ); shape `(3,)`;
  `None` or empty = no rotation; default: `None`
- **`location`** — Translation (x, y, z); shape `(3,)`; `None` or empty = origin; default: `None`
- **`n_div`** — Number of subdivisions per edge; yields 2 · n_div^2 triangles; default: 1

### Outputs:
- **`mesh`** — Triangle mesh; each row holds (x1, y1, z1, x2, y2, z2, x3, y3, z3); `(2 · n_div^2, 9)`

---
<a id="point_cloud_aabb"></a>
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

Also in: [C++](../cpp/site-specific-simulation-tools.md#point_cloud_aabb) · [MATLAB](../mex/site-specific-simulation-tools.md#point_cloud_aabb)

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_ind` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Usage:
```python
aabb = quadriga_lib.RTtools.point_cloud_aabb( points, sub_cloud_ind, vec_size )
```

### Input Arguments:
- **`points`** — 3D point coordinates; `(n_points, 3)`
- **`sub_cloud_ind`** *(optional)* — 0-based row indices marking the start of each sub-cloud;
  use [point_cloud_segmentation](#point_cloud_segmentation) to generate; uint32; `(n_sub,)`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16); default: 1

### Output Argument:
- **`aabb`** — Bounding box matrix; `(n_out, 6)` where `n_out` is `n_sub` padded to a multiple of `vec_size`

### See also:
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud indices)
- [ray_point_intersect](#ray_point_intersect) (use AABBs for intersection)

---
<a id="point_cloud_segmentation"></a>
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

Also in: [C++](../cpp/site-specific-simulation-tools.md#point_cloud_segmentation) · [MATLAB](../mex/site-specific-simulation-tools.md#point_cloud_segmentation)

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

### Usage:
```python
# Output as tuple
data = quadriga_lib.RTtools.point_cloud_segmentation( points, target_size, vec_size )

# Unpacked outputs
points_out, sub_cloud_ind, forward_ind, reverse_ind = \
    quadriga_lib.RTtools.point_cloud_segmentation( points, target_size, vec_size )
```

### Inputs:
- **`points`** — Original 3D point cloud; `(n_points, 3)`
- **`target_size`** *(optional)* — Maximum points per sub-cloud before padding; default: 1024
- **`vec_size`** *(optional)* — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of
  this value; no padding when `1`; `ray_point_intersect` accepts any value, padding (e.g. 8 for AVX2) only
  improves its speed; default: 1

### Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `(n_points_out, 3)`
- **`sub_cloud_index`** — 0-based starting index of each sub-cloud within `points_out`; uint32; `(n_sub,)`
- **`forward_index`** *(optional)* — 1-based row in `points` of each row of `points_out`; padding entries are `0`; uint32; `(n_points_out,)`
- **`reverse_index`** *(optional)* — 0-based row in `points_out` of each row of `points`; uint32; `(n_points,)`

---
<a id="point_inside_mesh"></a>
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

Also in: [C++](../cpp/site-specific-simulation-tools.md#point_inside_mesh) · [MATLAB](../mex/site-specific-simulation-tools.md#point_inside_mesh)

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided (0-based), returns the enclosing object as a 1-based index (so 0 stays reserved for "outside")

### Usage:
```python
result = quadriga_lib.RTtools.point_inside_mesh( points, mesh, obj_ind, distance )
```

### Input Arguments:
- **`points`** — 3D coordinates of test points; `(n_points, 3)`
- **`mesh`** — Triangle faces in row-major vertex format `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_mesh, 9)`
- **`obj_ind`** *(optional)* — 0-based object index per mesh element; enables per-object output; `(n_mesh,)`
- **`distance`** *(optional)* — Surface proximity threshold; points within this distance
  of the mesh surface are classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1);
  range: 0–20 m; Default: 0

### Output Arguments:
- `**result**`— Indicator: `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object
  index (with `obj_ind`); int64; size `(n_points,)`

See also:
- [obj_file_read](#obj_file_read) (for reading `mesh` and `obj_ind` from an .obj file)

---
<a id="ray_point_intersect"></a>
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

Also in: [C++](../cpp/site-specific-simulation-tools.md#ray_point_intersect) · [MATLAB](../mex/site-specific-simulation-tools.md#ray_point_intersect)

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin,
  enabling energy spread simulation
- Reports, for each point, the list of ray indices whose beam intersects that point
- Results use a compressed sparse row layout: `hit_index` holds the ray indices grouped by point and
  `hit_offset` marks where each point's block starts
- The rays hitting point `i` are `hit_index[hit_offset[i]:hit_offset[i+1]]`, a zero-copy numpy view
- `numpy.split(hit_index, hit_offset[1:-1])` produces a per-point list of arrays if that form is needed
- All internal computations use single precision

### Usage:
```python
hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect( orig, trivec, tridir,
    points, sub_cloud_ind, use_kernel, gpu_id )
```

### Inputs:
- **`orig`** — Ray origin positions in global Cartesian coordinates; `(n_ray, 3)`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `(n_ray, 9)`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `(n_ray, 9)`
- **`points`** — 3D point cloud coordinates; `(n_points, 3)`
- **`sub_cloud_ind`** *(optional)* — 0-based segment boundary indices for the point cloud
  (see `quadriga_lib.point_cloud_segmentation`); sub-clouds need not be aligned to the SIMD vector size (padding with
  `vec_size` is optional and only improves speed); uint32; `(n_sub,)`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2,
  3 = CUDA; throws if unavailable and for other values; auto mode selects CUDA when `n_points >= 500` and CUDA is
  available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA; default: 0

### Outputs:
- **`hit_count`** — Number of beams intersecting each point; equals `numpy.diff(hit_offset)`; uint32;
  `(n_points,)`
- **`hit_index`** — Flat list of 0-based ray indices, grouped by point; `(n_hit,)`
- **`hit_offset`** — Start of each point's block within `hit_index`; the last element equals `n_hit`;
  uint32; `(n_points + 1,)`

### See also:
- [icosphere](#icosphere) (for generating beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (for generating point cloud segments)
- [ray_triangle_intersect](#ray_triangle_intersect) (for calculating intersection of rays and triangles)

---
<a id="ray_triangle_intersect"></a>
## ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

Also in: [C++](../cpp/site-specific-simulation-tools.md#ray_triangle_intersect) · [MATLAB](../mex/site-specific-simulation-tools.md#ray_triangle_intersect)

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

### Usage:
```python
# Output as tuple
data = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id )

# Unpacked outputs
fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id )
```

### Inputs:
- **`orig`** — Ray origins in GCS; `(n_ray, 3)`
- **`dest`** — Ray destinations in GCS; `(n_ray, 3)`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `(n_mesh, 9)`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; 0-based; uint32; `(n_sub)`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row:
  `{x_min x_max y_min y_max z_min z_max}`; if empty or omitted, AABBs are computed from `mesh`; `(n_sub, 6)`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA;
  throws if unavailable and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2,
  else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

### Outputs:
- **`fbs`** — First-bounce intersection points in GCS; `(n_ray, 3)`
- **`sbs`** — Second-bounce intersection points in GCS; `(n_ray, 3)`
- **`no_interact`** — Total number of intersections per ray between `orig` and `dest`; uint32; `(n_ray,)`
- **`fbs_ind`** — 1-based index of first intersected mesh element; 0 = none; uint32; `(n_ray,)`
- **`sbs_ind`** — 1-based index of second intersected mesh element; 0 = none; uint32; `(n_ray,)`

### See also:
- [obj_file_read](#obj_file_read) (load mesh from OBJ file)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (compute sub-mesh indices)
- [triangle_mesh_aabb](#triangle_mesh_aabb) (compute AABBs)
- [ray_point_intersect](#ray_point_intersect) (beam interactions with sampling points)
- [icosphere](#icosphere) (generate ray beams)

---
<a id="triangle_mesh_aabb"></a>
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

Also in: [C++](../cpp/site-specific-simulation-tools.md#triangle_mesh_aabb) · [MATLAB](../mex/site-specific-simulation-tools.md#triangle_mesh_aabb)

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Usage:
```python
aabb = quadriga_lib.RTtools.triangle_mesh_aabb( triangles, sub_mesh_index, vec_size )
```

### Inputs:
- **`triangles`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `(n_triangles, 9)`
- **`sub_mesh_index`** *(optional)* — 1-based start indices of sub-meshes; if omitted, the AABB
  of the entire mesh is returned; uint32; `(n_sub,)`
- **`vec_size`** *(optional)* — Alignment size for SIMD/CUDA padding (e.g., `8` for AVX2, `32` for CUDA); default: 1

### Output:
- **`aabb`** — Axis-aligned bounding boxes, one row per sub-mesh; `(n_sub_aligned, 6)`

### See also:
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (for calculating sub-meshes)

---
<a id="triangle_mesh_segmentation"></a>
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

Also in: [C++](../cpp/site-specific-simulation-tools.md#triangle_mesh_segmentation) · [MATLAB](../mex/site-specific-simulation-tools.md#triangle_mesh_segmentation)

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; real entries in `mesh_index` are
  1-based, with 0 reserved to mark padding entries
- If `mtl_ind` is provided, material indices are reordered and padded in the same way; padding
  entries get index 0

### Usage:
```python
# Output as tuple
data = quadriga_lib.RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_ind )

# Unpacked outputs
triangles_out, sub_mesh_index, mesh_index, mtl_ind_out = \
    quadriga_lib.RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_ind )
```

### Inputs:
- **`triangles`** — Triangle vertices, each row `[x1,y1,z1,x2,y2,z2,x3,y3,z3]`; `(n_mesh, 9)`
- **`target_size`** — Target triangle count per sub-mesh; for best performance set near sqrt(n_mesh); default: 1024
- **`vec_size`** — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count
  is rounded up to a multiple of this value; default: 1
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `mtl_ind`/`csv_ind` output of
  [obj_file_read](#obj_file_read)); `(n_mesh,)` or `None`; default: `None`

### Outputs:
- **`triangles_out`** — Reordered and padded triangle vertices; `(n_triangles_out, 9)`
- **`sub_mesh_index`** — 0-based start indices of sub-meshes in `triangles_out`; uint32; `(n_sub,)`
- **`mesh_index`** — 1-based row in the input mesh of each output triangle (0 = padding); uint32; `(n_triangles_out,)`
- **`mtl_ind_out`** — Reordered and padded material indices; `(n_triangles_out,)`; empty if `mtl_ind` is not provided
