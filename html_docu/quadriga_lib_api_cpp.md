---
title: "C++ API Documentation for Quadriga-Lib v0.11.6"
author: "Stephan Jaeckel"
date: "29.05.2026"
lang: en-US
---

# General usage notes
- Each function has a 1-line short description, optional detailed notes, a Declaration block, and Inputs/Outputs/Returns sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions and classes live in the `quadriga_lib` namespace.
- Default include: `#include "quadriga_lib.hpp"`.
- Template parameter `dtype` is `float` or `double` unless stated.
- Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.
- Pointer arguments: `nullptr` skips optional outputs; required inputs throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- Invalid inputs (shape/domain) cause a `std::invalid_argument`; I/O failures a `std::runtime_error`.
- Index conventions: 0-based unless the field is explicitly called "1-based" (which applies to `obj_ind`, `mtl_ind`, `fbs_ind`, `sbs_ind`, and QDANT `id`).
- Units: angles in radians (degrees only where stated, e.g. `*_deg`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Polarization transfer matrix `M`: 8 rows per path, interleaved real/imaginary, order `[ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]`. A 2-row form `[ReVV, ImVV]` is used for scalar (acoustic) fields.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).
- Kernel-selection parameters (`use_kernel`): `0` = auto (CUDA if available and problem large enough, else AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2, `3` = CUDA. Throws if the requested kernel is unavailable.
- `gpu_id` is only read when `use_kernel` resolves to CUDA.

# Function Index

| Function | Section | Line |
| --- | --- | --- |
| [arrayant](#arrayant) | Array antenna class | 133 |
| [.append](#append) | Array antenna class | 184 |
| [.calc_beamwidth_deg](#calc_beamwidth_deg) | Array antenna class | 206 |
| [.calc_directivity_dBi](#calc_directivity_dbi) | Array antenna class | 238 |
| [.combine_pattern](#combine_pattern) | Array antenna class | 259 |
| [.copy_element](#copy_element) | Array antenna class | 284 |
| [.export_obj_file](#export_obj_file) | Array antenna class | 304 |
| [.interpolate](#interpolate) | Array antenna class | 334 |
| [.is_valid](#is_valid) | Array antenna class | 386 |
| [.qdant_write](#qdant_write) | Array antenna class | 404 |
| [.remove_zeros](#remove_zeros) | Array antenna class | 431 |
| [.rotate_pattern](#rotate_pattern) | Array antenna class | 447 |
| [.set_size](#set_size) | Array antenna class | 483 |
| [arrayant_combine_pattern_multi](#arrayant_combine_pattern_multi) | Array antenna functions | 509 |
| [arrayant_concat_multi](#arrayant_concat_multi) | Array antenna functions | 543 |
| [arrayant_copy_element_multi](#arrayant_copy_element_multi) | Array antenna functions | 577 |
| [arrayant_interpolate_multi](#arrayant_interpolate_multi) | Array antenna functions | 617 |
| [arrayant_is_valid_multi](#arrayant_is_valid_multi) | Array antenna functions | 676 |
| [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) | Array antenna functions | 703 |
| [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) | Array antenna functions | 735 |
| [generate_arrayant_3GPP](#generate_arrayant_3gpp) | Array antenna functions | 761 |
| [generate_arrayant_custom](#generate_arrayant_custom) | Array antenna functions | 811 |
| [generate_arrayant_dipole](#generate_arrayant_dipole) | Array antenna functions | 836 |
| [generate_arrayant_half_wave_dipole](#generate_arrayant_half_wave_dipole) | Array antenna functions | 853 |
| [generate_arrayant_multibeam](#generate_arrayant_multibeam) | Array antenna functions | 870 |
| [generate_arrayant_omni](#generate_arrayant_omni) | Array antenna functions | 924 |
| [generate_arrayant_ula](#generate_arrayant_ula) | Array antenna functions | 941 |
| [generate_arrayant_xpol](#generate_arrayant_xpol) | Array antenna functions | 969 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 986 |
| [qdant_read](#qdant_read) | Array antenna functions | 1061 |
| [qdant_read_multi](#qdant_read_multi) | Array antenna functions | 1087 |
| [qdant_write_multi](#qdant_write_multi) | Array antenna functions | 1114 |
| [channel](#channel) | Channel class | 1143 |
| [.add_paths](#add_paths) | Channel class | 1190 |
| [.calc_effective_path_gain](#calc_effective_path_gain) | Channel class | 1228 |
| [.write_paths_to_obj_file](#write_paths_to_obj_file) | Channel class | 1247 |
| [any_type_id](#any_type_id) | Channel functions | 1288 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 1331 |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Channel functions | 1374 |
| [baseband_freq_response_vec](#baseband_freq_response_vec) | Channel functions | 1422 |
| [get_HDF5_version](#get_hdf5_version) | Channel functions | 1460 |
| [hdf5_create](#hdf5_create) | Channel functions | 1472 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 1502 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 1534 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 1570 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 1604 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 1630 |
| [hdf5_write](#hdf5_write) | Channel functions | 1658 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1698 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 1737 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 1785 |
| [qrt_read_cache_init](#qrt_read_cache_init) | Channel functions | 1877 |
| [quantize_delays](#quantize_delays) | Channel functions | 1914 |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generation functions | 1976 |
| [get_channels_irs](#get_channels_irs) | Channel generation functions | 2057 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 2151 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 2219 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 2287 |
| [acdf](#acdf) | Channel statistics | 2363 |
| [calc_angular_spreads_sphere](#calc_angular_spreads_sphere) | Channel statistics | 2398 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 2440 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 2492 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 2525 |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | 2560 |
| [fast_acos](#fast_acos) | Math functions | 2593 |
| [fast_asin](#fast_asin) | Math functions | 2612 |
| [fast_atan2](#fast_atan2) | Math functions | 2631 |
| [fast_cart2geo](#fast_cart2geo) | Math functions | 2652 |
| [fast_geo2cart](#fast_geo2cart) | Math functions | 2685 |
| [fast_sincos](#fast_sincos) | Math functions | 2730 |
| [fast_slerp](#fast_slerp) | Math functions | 2751 |
| [interp_2D](#interp_2d) | Math functions | 2786 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 2850 |
| [colormap](#colormap) | Site-specific simulation tools | 2900 |
| [combine_irs_coord](#combine_irs_coord) | Site-specific simulation tools | 2919 |
| [coord2path](#coord2path) | Site-specific simulation tools | 2962 |
| [generate_diffraction_paths](#generate_diffraction_paths) | Site-specific simulation tools | 3000 |
| [icosphere](#icosphere) | Site-specific simulation tools | 3044 |
| [medium_gain](#medium_gain) | Site-specific simulation tools | 3078 |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Site-specific simulation tools | 3113 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 3154 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 3302 |
| [obj_overlap_test](#obj_overlap_test) | Site-specific simulation tools | 3352 |
| [path_to_tube](#path_to_tube) | Site-specific simulation tools | 3383 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 3411 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 3440 |
| [point_cloud_split](#point_cloud_split) | Site-specific simulation tools | 3479 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 3515 |
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | 3551 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 3636 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 3679 |
| [subdivide_rays](#subdivide_rays) | Site-specific simulation tools | 3727 |
| [subdivide_triangles](#subdivide_triangles) | Site-specific simulation tools | 3771 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 3801 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 3829 |
| [triangle_mesh_split](#triangle_mesh_split) | Site-specific simulation tools | 3870 |
| [write_png](#write_png) | Site-specific simulation tools | 3905 |

---

# Array antenna class

---
## arrayant
Class for storing and manipulating array antenna models

- Represents a multi-element antenna array; each element has a position relative to the array phase-center
- Elements may be inter-coupled via a complex coupling matrix
- Field pattern cubes `e_theta_re/im`, `e_phi_re/im` must all be `[n_elevation, n_azimuth, n_elements]`
- `element_pos` is optional (empty = all elements at origin); `coupling_re/im` are optional (empty = identity)

### Attributes:
| Attribute                         | Size                                   | Description                                       |
| --------------------------------- | -------------------------------------- | ------------------------------------------------- |
| `arma::Cube<dtype> e_theta_re`    | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, real part               |
| `arma::Cube<dtype> e_theta_im`    | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, imaginary part          |
| `arma::Cube<dtype> e_phi_re`      | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, real part               |
| `arma::Cube<dtype> e_phi_im`      | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, imaginary part          |
| `arma::Col<dtype> azimuth_grid`   | `[n_azimuth]`                          | Azimuth angles in rad, in [-pi, pi], sorted       |
| `arma::Col<dtype> elevation_grid` | `[n_elevation]`                        | Elevation angles in rad, in [-pi/2, pi/2], sorted |
| `arma::Mat<dtype> element_pos`    | `[3, n_elements]` or empty             | Element positions in local Cartesian coords       |
| `arma::Mat<dtype> coupling_re`    | `[n_elements, n_ports]`                | Coupling matrix, real part                        |
| `arma::Mat<dtype> coupling_im`    | `[n_elements, n_ports]`                | Coupling matrix, imaginary part                   |
| `dtype center_frequency`          | scalar                                 | Center frequency                                  |
| `std::string name`                | string                                 | Name of the array antenna object                  |

### Simple member functions:
| Function         | Description                                       |
| ---------------- | ------------------------------------------------- |
| `.n_elevation()` | Number of elevation angles                        |
| `.n_azimuth()`   | Number of azimuth angles                          |
| `.n_elements()`  | Number of antenna elements                        |
| `.n_ports()`     | Number of ports (columns of coupling matrix)      |
| `.copy()`        | Returns a deep copy of the arrayant object        |
| `.reset()`       | Clears all data, resetting size to zero           |
| `.is_valid()`    | Returns `""` if valid, or an error message string |

### Complex member functions:
| Function                  | Description                                                       |
| ------------------------- | ----------------------------------------------------------------- |
| .[append](#append)               | Append elements of another arrayant                               |
| .[calc_beamwidth_deg](#calc_beamwidth_deg)   | Calculate the beam width of an antenna element in degree          |
| .[calc_directivity_dBi](#calc_directivity_dBi) | Calculate the directivity in dBi of a single array element        |
| .[combine_pattern](#combine_pattern)      | Compute effective patterns from elements, positions, and coupling |
| .[copy_element](#copy_element)         | Copy a single element to one or more destination slots            |
| .[export_obj_file](#export_obj_file)      | Export pattern geometry to Wavefront OBJ                          |
| .[interpolate](#interpolate)          | Interpolate field patterns at given azimuth/elevation angles      |
| .[qdant_write](#qdant_write)          | Write arrayant to QDANT file                                      |
| .[remove_zeros](#remove_zeros)         | Remove zero-valued entries from pattern data                      |
| .[rotate_pattern](#rotate_pattern)       | Rotate pattern and/or polarization via Euler angles               |
| .[set_size](#set_size)             | Resize the arrayant to new dimensions                             |
| .[is_valid](#is_valid)             | Validate arrayant integrity                                       |

---
## .append
Append elements of another arrayant to the current one

- Both arrays must share identical sampling grids; throws otherwise
- Coupling is block-diagonal (see [arrayant_concat_multi](#arrayant_concat_multi) diagram); center_frequency is taken from this

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::append(
    const arrayant<dtype> *new_arrayant) const;
```

### Inputs:
- **`new_arrayant`** — Array whose elements are appended; sampling grid must match

### Returns:
- New `arrayant` containing all elements from both arrays

### See also:
- [arrayant_concat_multi](#arrayant_concat_multi) (multi-freq counterpart)

---
## .calc_beamwidth_deg
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in each direction than the antenna sampling grid)
- Ignores element coupling

### Declaration:
```
void calc_beamwidth_deg(arma::uword i_element,
    dtype threshold_dB = 3.0,
    dtype *beamwidth_az = nullptr,
    dtype *beamwidth_el = nullptr,
    dtype *z_point_ang = nullptr,
    dtype *el_point_ang = nullptr) const;
```

### Inputs:
- **`i_element`** — Element index; 0-based
- **`threshold_dB`** — Threshold in dB; 3 dB = FWHM

### Outputs:
- **`beamwidth_az`** — Azimuth beamwidth in degree
- **`beamwidth_el`** — Elevation beamwidth in degree
- **`az_point_ang`** — Azimuth pointing angle for the main beam in degree
- **`el_point_ang`** — Elevation pointing angle for the main beam in degree

### See also:
- .[calc_directivity_dBi](#calc_directivity_dBi) (directivity in dBi of a single array element)

---
## .calc_directivity_dBi
Calculate the directivity in dBi of a single array element

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Ignores element coupling

### Declaration:
```
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const;
```

### Inputs:
- **`i_element`** — Element index, 0-based

### Returns:
- Directivity of the specified element in dBi

### See also:
- .[combine_pattern](#combine_pattern) (the per-port directivity is a typical follow-up)

---
## .combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` to produce one output element per port (column) of the coupling matrix
- Useful for beamforming and MIMO channel computation speedup

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::combine_pattern(
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr) const;
```

### Inputs:
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], sorted; defaults to input grid
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], sorted; defaults to input grid

### Returns:
- New `arrayant` with `n_ports` elements (= number of columns in `coupling_re/im`), each holding the combined effective pattern for that port

### See also:
- .[interpolate](#interpolate) (used internally to compute effective radiation patterns)
- .[rotate_pattern](#rotate_pattern) (useful for orienting array antenna patterns)

---
## .copy_element
Copy a single antenna element to one or more destination slots

- Array is resized if any destination index exceeds the current number of elements
- Coupling matrix for added elements is set to identity; if not existing, it gets initialized to identity

### Declaration:
```
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uword destination);
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uvec destination);
```

### Inputs:
- **`source`** — Index of the element to copy, 0-based
- **`destination`** — Target index or indices, 0-based; array resizes to fit the maximum index

### See also:
- [arrayant_copy_element_multi](#arrayant_copy_element_multi) (multi-freq counterpart)

---
## .export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives finer mesh

### Declaration:
```
void quadriga_lib::arrayant<dtype>::export_obj_file(
    std::string fn,
    dtype directivity_range = 30.0,
    std::string colormap = "jet",
    dtype object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    arma::uvec i_element = {}) const;
```

### Inputs:
- **`fn`** — Output OBJ filename; must not be empty; filename must end in .obj
- **`directivity_range`** *(optional)* — Dynamic range of the visualized directivity pattern in dB
- **`colormap`** *(optional)* — Colormap name; see [colormap](#colormap) for supported options
- **`object_radius`** *(optional)* — Radius of the exported object
- **`icosphere_n_div`** *(optional)* — Icosphere subdivision count; higher = finer mesh, see [icosphere](#icosphere)
- **`i_element`** *(optional)* — 0-based element indices to export; `{}` exports all elements

### See also:
- [colormap](#colormap) (Used for setting the colormap)
- [icosphere](#icosphere) (Used internally to generate icosphere primitive)
- .[write_paths_to_obj_file](#write_paths_to_obj_file) (function of Channel class to export propagation paths to OBJ 3D visualization)

---
## .interpolate
Interpolate polarimetric antenna field patterns for given azimuth/elevation angles

- Outputs complex e-theta (V) and e-phi (H) field components at requested angles
- `n_out` equals `n_elements` when `i_element` is omitted; equals `len(i_element)` otherwise
- Azimuth input supports planar wave mode (`[1, n_ang]`) or per-element spherical wave mode (`[n_out, n_ang]`)
- Output matrices are resized automatically if dimensions do not match; this invalidates existing data pointers

### Declaration:
```
void quadriga_lib::arrayant<dtype>::interpolate(
    const arma::Mat<dtype> *azimuth,
    const arma::Mat<dtype> *elevation,
    arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
    arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
    arma::uvec i_element = {},
    const arma::Cube<dtype> *orientation = nullptr,
    const arma::Mat<dtype> *element_pos_i = nullptr,
    arma::Mat<dtype> *dist = nullptr,
    arma::Mat<dtype> *azimuth_loc = nullptr,
    arma::Mat<dtype> *elevation_loc = nullptr,
    arma::Mat<dtype> *gamma = nullptr) const;
```

### Inputs:
- **`azimuth`** — Azimuth angles in rad, in [-pi, pi]; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in rad, in [-pi/2, pi/2]; `[1, n_ang]` or `[n_out, n_ang]`
- **`i_element`** *(optional)* — Element indices (0-based) to interpolate; duplicates allowed; defaults to all elements; `[n_out]` or `{}`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) in rad; `nullptr`; `[3, 1]`; `[3, n_out]`; `[3, 1, n_ang]`, or `[3, n_out, n_ang]`
- **`element_pos_i`** *(optional)* — Override element positions in m; `nullptr` uses `arrayant.element_pos`; `[3, n_out]`

### Outputs:
- **`V_re`** / **`V_im`** — Real/imaginary e-theta (vertical) field component; `[n_out, n_ang]`
- **`H_re`** / **`H_im`** — Real/imaginary e-phi (horizontal) field component; `[n_out, n_ang]`
- **`dist`** *(optional)* — Distance from the wavefront plane (normal to the incident ray direction) to each element; `nullptr` or `[n_out, n_ang]`
- **`azimuth_loc`** *(optional)* — Azimuth angles in local (rotated) element frame in rad; `nullptr` or `[n_out, n_ang]`
- **`elevation_loc`** *(optional)* — Elevation angles in local element frame in rad; `nullptr` or `[n_out, n_ang]`
- **`gamma`** *(optional)* — Polarization rotation angles in rad; `nullptr` or `[n_out, n_ang]`

### Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
arma::mat azimuth = {0.0, 0.5*pi, -0.5*pi, pi};
arma::mat elevation(1, azimuth.n_elem);  // zeros
arma::mat V_re, V_im, H_re, H_im;
ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);
```

### See also:
- [arrayant_interpolate_multi](#arrayant_interpolate_multi) (multi-freq counterpart)

---
## .is_valid
Validate the integrity of an arrayant object

### Declaration:
```
std::string quadriga_lib::arrayant<dtype>::is_valid(bool quick_check = true) const;
```

### Inputs:
- **`quick_check`** *(optional)* — `true` for fast structural check; `false` for full data validation; full check additionally verifies data values

### Returns:
- Empty string if valid; error message string if invalid

### See also:
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (multi-freq counterpart)

---
## .qdant_write
Write arrayant data to a QDANT (XML) file

- Multiple antennas can be stored in the same file using distinct `id` values
- If `id = 0` and the file exists, the antenna is appended with `id = max_existing_id + 1`; if file does not exist, `id = 1` gets assigned

### Declaration:
```
unsigned quadriga_lib::arrayant<dtype>::qdant_write(
    std::string fn,
    unsigned id = 0,
    arma::u32_mat layout = {}) const;
```

### Inputs:
- **`fn`** — Output QDANT filename; must not be empty
- **`id`** *(optional)* — Target ID in file; `0` appends with auto-assigned ID
- **`layout`** *(optional)* — Matrix organizing multiple antenna IDs within the file; must reference only IDs present in the file

### Returns:
- ID assigned to the written antenna within the file

### See also:
- [qdant_read](#qdant_read) (read back QDANT files)
- [qdant_write_multi](#qdant_write_multi) (multi-freq counterpart)

---
## .remove_zeros
Remove zero-valued entries from antenna pattern data, reducing its size

- Modifies in-place when `output = nullptr`; otherwise writes to `*output`
- A column or row is removed if the total power summed over all field components and array elements does not exceed 1e-12;
  immediate neighbors of retained entries are also kept unless the local grid spacing is strongly asymmetric (factor > 1.5)

### Declaration:
```
void quadriga_lib::arrayant<dtype>::remove_zeros(arrayant<dtype> *output = nullptr);
```

### Inputs:
- **`output`** *(optional)* — Target arrayant to write result to; `nullptr` modifies in-place

---
## .rotate_pattern
Rotate antenna radiation patterns around the principal axes using Euler rotations

- Rotates pattern and/or polarization around x (bank), y (tilt), z (heading) axes in degrees
- Modifies in-place when `output = nullptr`; otherwise writes to `*output`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::rotate_pattern(
    dtype x_deg = 0.0,
    dtype y_deg = 0.0,
    dtype z_deg = 0.0,
    unsigned usage = 0,
    unsigned element = -1,
    arrayant<dtype> *output = nullptr);
```

### Inputs:
- **`x_deg`** *(optional)* — Rotation around x-axis (bank) in degrees
- **`y_deg`** *(optional)* — Rotation around y-axis (tilt) in degrees
- **`z_deg`** *(optional)* — Rotation around z-axis (heading) in degrees
- **`usage`** *(optional)* — Rotation mode:
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`element`** *(optional)* — 0-based element index to rotate; `-1` rotates all elements (implemented as wrap-around to UINT_MAX)
- **`output`** *(optional)* — Target arrayant; `nullptr` modifies in-place

### See also:
- [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) (multi-freq counterpart)

---
## .set_size
Resize an arrayant object to new dimensions

- No-op if current dimensions already match
- After resize: element_pos is zeroed, coupling_re set to identity, coupling_im zeroed; all other field data is undefined

### Declaration:
```
void quadriga_lib::arrayant<dtype>::set_size(
    arma::uword n_elevation,
    arma::uword n_azimuth,
    arma::uword n_elements,
    arma::uword n_ports);
```

### Inputs:
- **`n_elevation`** — Number of elevation samples
- **`n_azimuth`** — Number of azimuth samples
- **`n_elements`** — Number of antenna elements
- **`n_ports`** — Number of ports (columns of coupling matrix)

---

# Array antenna functions

---
## arrayant_combine_pattern_multi
Combine element patterns, positions, and coupling weights into effective radiation patterns (multi-frequency)

- Multi-frequency counterpart to .[combine_pattern](#combine_pattern)
- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` across all entries to produce one output element per port (column of the coupling matrix) at each requested output frequency
- Output length = `freq_grid_new->n_elem` if provided; otherwise one entry per input arrayant
- Field interpolation across frequency is delegated to [arrayant_interpolate_multi](#arrayant_interpolate_multi) (SLERP with linear-interpolation fallback)
- Coupling matrices are SLERP-interpolated between bracketing input entries; out-of-range output frequencies are clamped to the nearest input entry
- Each output arrayant has identity coupling and zero element positions (patterns are pre-combined)

### Declaration:
```
std::vector<arrayant<dtype>> arrayant_combine_pattern_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    const arma::Col<dtype> *freq_grid_new = nullptr);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects (must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi))
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], sorted; defaults to `arrayant_vec[0].azimuth_grid`
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], sorted; defaults to `arrayant_vec[0].elevation_grid`
- **`freq_grid_new`** *(optional)* — Alternative frequency grid in Hz; defaults to per-entry `center_frequency`

### Returns:
- Vector of arrayant objects (length = `n_freq_out`), each with `n_ports` elements equal to the number of columns in the coupling matrix

### See also:
- .[combine_pattern](#combine_pattern) (single-frequency counterpart)
- [arrayant_interpolate_multi](#arrayant_interpolate_multi) (used internally for spatial+frequency field interpolation)
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (input validation)

---
## arrayant_concat_multi
Concatenate two multi-frequency arrayant vectors into a single multi-element model

- Both inputs must have equal entry counts, identical angular grids, and matching `center_frequency` values at each index.
- Per frequency entry: pattern cubes are joined along the element (slice) dimension; `element_pos` matrices are horizontally concatenated (empty positions treated as zeros).
- Both inputs are validated with [arrayant_is_valid_multi](#arrayant_is_valid_multi) before processing; each output entry is validated before returning.
- Output inherits name, azimuth/elevation grids, and `center_frequency` from `arrayant_vec1`.

### Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::arrayant_concat_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec1,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec2);
```

### Inputs:
- **`arrayant_vec1`** — First validated, mutually consistent arrayant vector
- **`arrayant_vec2`** — Second arrayant vector; must match entry count, grids, and center frequencies of `arrayant_vec1`

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports per entry
- Coupling matrices are assembled block-diagonally — elements from `vec1` connect only to ports from `vec1` and vice versa:
   | Element \ Port      | P1…Pp1 (vec1) | Pp1+1…Pp1+p2 (vec2) |
   | ------------------- | :-----------: | :-----------------: |
   | E1…En1 (vec1)       |   C1 block    |          0          |
   | En1+1…En1+n2 (vec2) |       0       |      C2 block       |

### See also:
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (validation called on both inputs)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (position drivers before concatenating)
- [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) (rotate elements after concatenating)
- [qdant_write_multi](#qdant_write_multi) (persist the combined model)

---
## arrayant_copy_element_multi
Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector

- Calls .[copy_element](#copy_element) on every entry in the vector with the same source and destination indices.
- If any destination index exceeds the current element count, all entries are enlarged; new elements receive an identity coupling entry.
- Source and destination indices are 0-based.

### Declaration:
```
void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uvec destination);

void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uword destination);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects; modified in-place
- **`source`** — Index of the element to copy from; must be within current element count
- **`destination`** — Index or indices of target elements; enlarges all entries if any index exceeds current count

### Example:
```
arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
auto driver = quadriga_lib::generate_speaker<double>(
    "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
    0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
quadriga_lib::arrayant_copy_element_multi(driver, 0, arma::uvec{1, 2, 3});
```

### See also:
- .[copy_element](#copy_element) (per-entry operation called internally)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions after copying)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multiple arrayant vectors)

---
## arrayant_interpolate_multi
Interpolate multi-frequency arrayant patterns at arbitrary angles and frequencies

- For each requested frequency, finds the two bracketing `center_frequency` entries, runs spatial interpolation on both via `qd_arrayant_interpolate`, then blends results in the frequency dimension.
- Frequency blending uses SLERP of complex field values with automatic fallback to linear interpolation when phase difference exceeds a threshold.
- Out-of-range frequencies are clamped to the nearest entry (no extrapolation).
- Consecutive frequency requests sharing the same bracketing entries reuse cached spatial interpolation results; sort `frequency` ascending or descending for best cache utilization.
- If `validate_input` is true, calls [arrayant_is_valid_multi](#arrayant_is_valid_multi) once before processing; set to `false` in performance-critical loops after initial validation.

### Declaration:
```
void quadriga_lib::arrayant_interpolate_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> *azimuth,
        const arma::Mat<dtype> *elevation,
        const arma::Col<dtype> *frequency,
        arma::Cube<dtype> *V_re,
        arma::Cube<dtype> *V_im,
        arma::Cube<dtype> *H_re,
        arma::Cube<dtype> *H_im,
        arma::uvec i_element = {},
        const arma::Cube<dtype> *orientation = nullptr,
        const arma::Mat<dtype> *element_pos_i = nullptr,
        bool validate_input = true);
```

### Inputs:
- **`arrayant_vec`** — Multi-frequency arrayant vector; entries need not be sorted by frequency
- **`azimuth`** — Azimuth angles in rad; must not be NULL, `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in rad; must not be NULL; size must match `azimuth`
- **`frequency`** — Target frequencies in Hz; must not be NULL or empty; `[n_freq]`
- **`i_element`** *(optional)* — Element indices to interpolate; if empty, all elements are used (`n_out = n_elements`)
- **`orientation`** *(optional)* — Antenna orientation (bank, tilt, heading) in rad, applied at all frequencies; `[3,1,1]`; `[3,n_out,1]`; `[3,1,n_ang]`, or `[3,n_out,n_ang]`
- **`element_pos_i`** *(optional)* — Override element positions; if `nullptr`, positions from freq index 0 are used; `[3, n_out]`
- **`validate_input`** *(optional)* — If `true`, validates `arrayant_vec` with [arrayant_is_valid_multi](#arrayant_is_valid_multi) before processing

### Outputs:
- **`V_re`** — Real part of interpolated e-theta field; must not be NULL; `[n_out, n_ang, n_freq]`
- **`V_im`** — Imaginary part of interpolated e-theta field; must not be NULL; `[n_out, n_ang, n_freq]`
- **`H_re`** — Real part of interpolated e-phi field; must not be NULL; `[n_out, n_ang, n_freq]`
- **`H_im`** — Imaginary part of interpolated e-phi field; must not be NULL; `[n_out, n_ang, n_freq]`

### Example:
```
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
arma::mat el(1, 4, arma::fill::zeros);
arma::vec qf = {250.0, 1500.0, 8000.0};
arma::cube V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate_multi(speaker, &az, &el, &qf, &V_re, &V_im, &H_re, &H_im);
```

### See also:
- .[interpolate](#interpolate) (single-frequency spatial interpolation)
- [arrayant_concat_multi](#arrayant_concat_multi) (build multi-element/multi-frequency models)
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (validation called when validate_input is true)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
## arrayant_is_valid_multi
Validate a vector of arrayant objects for multi-frequency consistency

- Each entry is validated individually via its `is_valid` member; `quick_check` is forwarded to that call.
- Cross-entry checks (all vs. entry 0): azimuth/elevation grid sizes and values, number of elements, element positions, coupling_re shape, and coupling_im presence and size.
- Pattern data, `center_frequency`, and coupling matrix values are not compared (expected to vary).
- Stops at first error and returns a message identifying the failing entry and property.

### Declaration:
```
std::string quadriga_lib::arrayant_is_valid_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        bool quick_check = true);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects to validate
- **`quick_check`** *(optional)* — If `true`, uses fast pointer-based per-entry validation; if `false`, performs full deep validation

### Returns:
- Empty string if valid; otherwise a message such as `"Entry 3: Azimuth grid values do not match entry 0."`

### See also:
- .[is_valid](#is_valid) (per-entry validation called internally)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
## arrayant_rotate_pattern_multi
Apply Euler rotations to all entries in a multi-frequency arrayant vector

- Calls .[rotate_pattern](#rotate_pattern) on every entry with grid adjustment always disabled (required for uniform-grid consistency across frequencies).
- If `i_element` is empty, all elements are rotated; otherwise only the specified indices are affected.
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects.

### Declaration:
```
void quadriga_lib::arrayant_rotate_pattern_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        dtype x_deg = 0.0,
        dtype y_deg = 0.0,
        dtype z_deg = 0.0,
        unsigned usage = 0,
        arma::uvec i_element = arma::uvec())
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`x_deg`** *(optional)* — Bank angle in degrees
- **`y_deg`** *(optional)* — Tilt angle in degrees
- **`z_deg`** *(optional)* — Heading angle in degrees
- **`usage`** *(optional)* — Rotation mode: `0` = pattern + polarization, `1` = pattern only, `2` = polarization only
- **`i_element`** *(optional)* — Indices of elements to rotate; if empty, all elements are rotated

### See also:
- .[rotate_pattern](#rotate_pattern) (per-entry operation called internally)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multi-frequency vectors before rotating)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions in multi-frequency vectors)

---
## arrayant_set_element_pos_multi
Set element positions for all entries in a multi-frequency arrayant vector

- Updates `element_pos` in-place on every entry in the vector identically.
- If `i_element` is empty, all positions are replaced and `element_pos` must have `n_elements` columns.
- If `i_element` is provided, only those indexed columns are updated; `element_pos` column count must match `i_element` length.
- All entries must have the same element count; uninitialized `element_pos` fields are zero-initialized before update.

### Declaration:
```
void quadriga_lib::arrayant_set_element_pos_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        arma::uvec i_element = arma::uvec());
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`element_pos`** — New (x, y, z) positions; `[3, n_update]`
- **`i_element`** *(optional)* — Indices of elements to update; if empty, all elements are replaced

### See also:
- [arrayant_copy_element_multi](#arrayant_copy_element_multi) (replicate elements before setting positions)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
## generate_arrayant_3GPP
Generate a 3GPP-NR compliant antenna array model

- Supports vertical (M) and horizontal (N) element stacking within panels, and multi-panel arrays (Mg × Ng).
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.
- Electrical downtilt (`tilt`) applies only to `pol` modes 4, 5, and 6.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(
    arma::uword M = 1, 
    arma::uword N = 1, 
    dtype center_freq = 299792458.0,
    unsigned pol = 1, 
    dtype tilt = 0.0, 
    dtype spacing = 0.5, 
    arma::uword Mg = 1,
    arma::uword Ng = 1, 
    dtype dgv = 0.5, 
    dtype dgh = 0.5,
    const quadriga_lib::arrayant<dtype> *pattern = nullptr, 
    dtype res = 1.0);
```

### Inputs:
- **`M`** *(optional)* — Number of vertical elements per panel
- **`N`** *(optional)* — Number of horizontal elements per panel
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:
   | `pol` | Description                          | Elements |
   | ----- | ------------------------------------ | -------- |
   | 1     | Vertical polarization                | NM       |
   | 2     | H/V polarization                     | 2NM      |
   | 3     | ±45° polarization                    | 2NM      |
   | 4     | Vertical, vertical elements combined | N        |
   | 5     | H/V, vertical elements combined      | 2N       |
   | 6     | ±45°, vertical elements combined     | 2N       |
- **`tilt`** *(optional)* — Electrical downtilt in degrees; applies to `pol` 4–6
- **`spacing`** *(optional)* — Inter-element spacing within a panel in wavelengths
- **`Mg`** *(optional)* — Number of vertically stacked panels
- **`Ng`** *(optional)* — Number of horizontally stacked panels
- **`dgv`** *(optional)* — Panel spacing in vertical direction in wavelengths
- **`dgh`** *(optional)* — Panel spacing in horizontal direction in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default 3GPP element pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

### Returns:
- `quadriga_lib::arrayant<dtype>` — 3GPP-NR antenna array object

---
## generate_arrayant_custom
Generate an antenna with custom 3dB beamwidth

- Returns a single-element antenna with independently configurable azimuth and elevation 3dB (FWHM) beamwidths.
- Rear-side gain is controlled by a linear front-to-back ratio; `0.0` means no rear radiation.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(
    dtype az_3dB = 90.0,
    dtype el_3dB = 90.0, 
    dtype rear_gain_lin = 0.0, 
    dtype res = 1.0);
```

### Inputs:
- **`az_3dB`** *(optional)* — Azimuth 3dB beamwidth in degrees
- **`el_3dB`** *(optional)* — Elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna object with specified beamwidth and rear gain

---
## generate_arrayant_dipole
Generate a short dipole antenna with vertical polarization

- Returns a single-element short dipole antenna pattern with vertical polarization.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized short dipole antenna object

---
## generate_arrayant_half_wave_dipole
Generate a half-wave dipole antenna with vertical polarization

- Returns a single-element half-wave dipole antenna pattern with vertical polarization.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized half-wave dipole antenna object

---
## generate_arrayant_multibeam
Generate a planar multi-element antenna array with multiple beam directions

- Returns an M×N planar array with beamforming weights computed via maximum-ratio transmission (MRT).
- MRT is optimal for a single beam; approximate when multiple beams are specified.
- Weights control relative beam contribution; only their ratios matter, not absolute values.
- If `separate_beams = true`, each angle pair produces an independent beam (weights ignored).
- If `apply_weights = true`, beamforming weights are baked into the element coupling matrix.
- Per-element pattern shape is controlled by `az_3dB`, `el_3dB`, and `rear_gain_lin`; see [generate_arrayant_custom](#generate_arrayant_custom).

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_multibeam(
    arma::uword M = 1,
    arma::uword N = 1,
    arma::Col<dtype> az = {0.0},
    arma::Col<dtype> el = {0.0},
    arma::Col<dtype> weight = {1.0},
    dtype center_freq = 299792458.0,
    unsigned pol = 1,
    dtype spacing = 0.5,
    dtype az_3dB = 120.0,
    dtype el_3dB = 120.0,
    dtype rear_gain_lin = 0.0,
    dtype res = 1.0,
    bool separate_beams = false,
    bool apply_weights = false);
```

### Inputs:
- **`M`** *(optional)* — Number of vertical (row) elements
- **`N`** *(optional)* — Number of horizontal (column) elements
- **`az`** *(optional)* — Azimuth beam angles in degrees; `[n_beams]`
- **`el`** *(optional)* — Elevation beam angles in degrees; `[n_beams]`
- **`weight`** *(optional)* — Per-beam scaling factors (normalized to sum = 1); `[n_beams]`
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:
   | `pol` | Description           | Elements |
   | ----- | --------------------- | -------- |
   | 1     | Vertical polarization | NM       |
   | 2     | H/V polarization      | 2NM      |
   | 3     | ±45° polarization     | 2NM      |
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`az_3dB`** *(optional)* — Per-element azimuth 3dB beamwidth in degrees
- **`el_3dB`** *(optional)* — Per-element elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Per-element front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees
- **`separate_beams`** *(optional)* — If `true`, generate one independent beam per angle pair
- **`apply_weights`** *(optional)* — If `true`, bake beamforming weights into the coupling matrix

### Returns:
- `quadriga_lib::arrayant<dtype>` — Multibeam planar array antenna object

---
## generate_arrayant_omni
Generate an isotropic radiator with vertical polarization

- Returns a single-element antenna array with omnidirectional pattern and vertical polarization.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Isotropic radiator antenna object

---
## generate_arrayant_ula
Generate a uniform linear array (ULA)

- Returns a horizontally stacked linear array of N elements with half-wavelength spacing by default.
- Default per-element pattern is a vertically polarized isotropic radiator.
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_ula(
    arma::uword N = 1, 
    dtype center_freq = 299792458.0, 
    dtype spacing = 0.5,
    const quadriga_lib::arrayant<dtype> *pattern = nullptr, 
    dtype res = 1.0);
```

### Inputs:
- **`N`** *(optional)* — Number of elements
- **`center_freq`** *(optional)* — Center frequency
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default isotropic pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

### Returns:
- `quadriga_lib::arrayant<dtype>` — ULA antenna array object

---
## generate_arrayant_xpol
Generate a cross-polarized isotropic radiator

- Returns a two-element antenna array with omnidirectional patterns in vertical and horizontal polarization.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Cross-polarized isotropic radiator antenna object

---
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns one [arrayant](#arrayant) object per frequency sample; each has a single element with the real-valued 
  directivity pattern in `e_theta_re` and `center_frequency` set to the corresponding frequency.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining results 
  via `append` and `element_pos`; crossover behavior emerges from overlapping bandpass responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`, 
  where `n = slope_dB_per_octave / 6`; −3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity − 85) / 20)`.
- If `frequencies` is empty, third-octave band center frequencies are auto-generated from one band below 
  `lower_cutoff` to one band above `upper_cutoff`, clipped to 20–20000 Hz.
- Speed of sound assumed to be 344 m/s.
- **Driver models** (`driver_type`):
  - `piston` — circular piston in baffle, `D(θ) = 2·J1(ka·sinθ)/(ka·sinθ)`, rotationally symmetric, narrows with increasing `ka`
  - `horn` — separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`
  - `omni` — frequency-independent omnidirectional pattern.
- **Enclosure models** (`radiation_type`):
  - `monopole` — no modification
  - `hemisphere` — sealed box with baffle-step transition, `f_baffle = c/(π·sqrt(W·H))`
  - `dipole` — figure-8, `R = abs(cos(θ_off))` with sign inversion in rear hemisphere
  - `cardioid` — `R = 0.5·(1+cos(θ_off))`
- For `"horn"`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2π·radius)`.

### Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::generate_speaker(
        std::string driver_type = "piston",
        dtype radius = 0.05,
        dtype lower_cutoff = 80.0,
        dtype upper_cutoff = 12000.0,
        dtype lower_rolloff_slope = 12.0,
        dtype upper_rolloff_slope = 12.0,
        dtype sensitivity = 85.0,
        std::string radiation_type = "hemisphere",
        dtype hor_coverage = 0.0,
        dtype ver_coverage = 0.0,
        dtype horn_control_freq = 0.0,
        dtype baffle_width = 0.15,
        dtype baffle_height = 0.25,
        arma::Col<dtype> frequencies = arma::Col<dtype>(),
        dtype angular_resolution = 5.0);
```

### Inputs:
- **`driver_type`** *(optional)* — Driver directivity model: `"piston"`, `"horn"`, or `"omni"`
- **`radius`** *(optional)* — Effective radiating radius; cone/dome radius for piston, mouth radius for horn
- **`lower_cutoff`** *(optional)* — Lower −3 dB bandpass frequency
- **`upper_cutoff`** *(optional)* — Upper −3 dB bandpass frequency
- **`lower_rolloff_slope`** *(optional)* — Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth)
- **`upper_rolloff_slope`** *(optional)* — High-frequency rolloff in dB/octave
- **`sensitivity`** *(optional)* — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude
- **`radiation_type`** *(optional)* — Enclosure radiation model: `"monopole"`, `"hemisphere"`, `"dipole"`, or `"cardioid"`
- **`hor_coverage`** *(optional)* — Horn horizontal coverage angle in degrees; `0` defaults to 90°
- **`ver_coverage`** *(optional)* — Horn vertical coverage angle in degrees; `0` defaults to 60°
- **`horn_control_freq`** *(optional)* — Horn pattern control frequency; `0` auto-derives from `radius`
- **`baffle_width`** *(optional)* — Baffle width; used by `"hemisphere"` model
- **`baffle_height`** *(optional)* — Baffle height; used by `"hemisphere"` model
- **`frequencies`** *(optional)* — Frequency sample points; auto-generated third-octave bands if empty; `[n_freq]`
- **`angular_resolution`** *(optional)* — Azimuth and elevation sampling grid resolution in degrees

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One arrayant per frequency sample with directivity in `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180° phase inversion

### Example:
```
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
               12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
auto horn = quadriga_lib::generate_speaker<double>("horn");
auto sub = quadriga_lib::generate_speaker<double>("omni", 0.13, 30.0, 200.0,
               12.0, 24.0, 92.0, "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, {30.,50.,80.,120.,200.}, 10.0);
```

---
## qdant_read
Read an arrayant object from a QDANT file

- Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the arrayant for the given ID.

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(
        std::string fn,
        unsigned id = 1,
        arma::u32_mat *layout = nullptr);
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`id`** *(optional)* — 1-based ID of the antenna entry to read
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix of element IDs

### Returns:
- `quadriga_lib::arrayant<dtype>` constructed from the specified entry in the file

### See also:
- .[qdant_write](#qdant_write) (write a single arrayant)
- [qdant_write_multi](#qdant_write_multi) (write multiple arrayants with sequential IDs)

---
## qdant_read_multi
Read all arrayant objects from a QDANT file into a vector

- Reads all entries from a QDANT file by probing ID 1 to obtain the layout, then reading each unique non-zero ID in order of first appearance (column-major scan).
- Each unique ID is read exactly once regardless of how many times it appears in the layout.
- Counterpart to [qdant_write_multi](#qdant_write_multi); primary mechanism for loading frequency-dependent models where `center_frequency` on each entry identifies the corresponding frequency.

### Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read_multi(
        const std::string &fn,
        arma::u32_mat *layout = nullptr);
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix; non-zero values are entry IDs

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per unique ID, ordered by first appearance in the layout

### See also:
- [qdant_read](#qdant_read) (read a single entry by ID)
- [qdant_write_multi](#qdant_write_multi) (write a vector of arrayants)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---
## qdant_write_multi
Write a vector of arrayant objects to a single QDANT file

- Writes each entry in `arrayant_vec` to a QDANT file with sequential 1-based IDs using .[qdant_write](#qdant_write).
- Auto-generates a `[n_entries, 1]` layout matrix with entries `1, 2, ..., n_entries`.
- Deletes any existing file before writing; all entries are validated first.
- Primary use case: frequency-dependent models where each arrayant holds a pattern at one frequency via `center_frequency`.

### Declaration:
```
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Inputs:
- **`fn`** — Path of the QDANT file to write; must not be empty
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects to store

### See also:
- .[qdant_write](#qdant_write) (per-object write used internally)
- [qdant_read](#qdant_read) (read back individual entries by ID)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---

# Channel class

---
## channel
Class for storing and managing MIMO channel data and metadata across multiple snapshots

- Represents path-level MIMO channel data between antenna arrays over multiple time snapshots
- Each snapshot may have a different number of propagation paths `n_path`
- Unstructured metadata supported via `par_names` / `par_data`

### Attributes:
| Attribute                                         | Size                                                            | Description                                                 |
| ------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| `std::string name`                                | —                                                               | Name of the channel object                                  |
| `arma::Col<dtype> center_frequency`               | `[1]`; `[n_snap]`, or `[]`                                      | Center frequency                                            |
| `arma::Mat<dtype> tx_pos`                         | `[3, n_snap]` or `[3, 1]` = static                              | Transmitter positions                                       |
| `arma::Mat<dtype> rx_pos`                         | `[3, n_snap]` or `[3, 1]` = static                              | Receiver positions                                          |
| `arma::Mat<dtype> tx_orientation`                 | `[3, n_snap]`; `[3, 1]` = static, or `[]` = no rotation         | Transmitter orientation (Euler angles)                      |
| `arma::Mat<dtype> rx_orientation`                 | `[3, n_snap]`; `[3, 1]` = static, or `[]` = no rotation         | Receiver orientation (Euler angles)                         |
| `std::vector<arma::Cube<dtype>> coeff_re`         | per snap `[n_rx, n_tx, n_path]`                                 | Channel coefficients, real part                             |
| `std::vector<arma::Cube<dtype>> coeff_im`         | per snap `[n_rx, n_tx, n_path]`                                 | Channel coefficients, imaginary part                        |
| `std::vector<arma::Cube<dtype>> delay`            | per snap `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` = broadcast | Path delays in seconds                                      |
| `std::vector<arma::Col<dtype>> path_gain`         | per snap `[n_path]`                                             | Path gains before antenna pattern                           |
| `std::vector<arma::Col<dtype>> path_length`       | per snap `[n_path]`                                             | Path lengths TX to RX                                       |
| `std::vector<arma::Mat<dtype>> path_polarization` | per snap `[8, n_path]`                                          | Interleaved polarization transfer matrices                  |
| `std::vector<arma::Mat<dtype>> path_angles`       | per snap `[n_path, 4]`                                          | Angles {AOD, EOD, AOA, EOA} in rad                          |
| `std::vector<arma::Mat<dtype>> path_fbs_pos`      | per snap `[3, n_path]`                                          | First-bounce scatterer positions                            |
| `std::vector<arma::Mat<dtype>> path_lbs_pos`      | per snap `[3, n_path]`                                          | Last-bounce scatterer positions                             |
| `std::vector<arma::Col<unsigned>> no_interact`    | per snap `[n_path]`                                             | Number of interactions per path                             |
| `std::vector<arma::Mat<dtype>> interact_coord`    | per snap `[3, sum(no_interact)]`                                | Interaction point coordinates                               |
| `std::vector<std::string> par_names`              | —                                                               | Names of unstructured metadata fields                       |
| `std::vector<std::any> par_data`                  | —                                                               | Unstructured metadata values (string, scalar, matrix, etc.) |
| `int initial_position`                            | scalar                                                          | 0-based index of the reference snapshot                     |

### Simple member functions:
| Method        | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `.n_snap()`   | Returns the number of snapshots                               |
| `.n_rx()`     | Returns number of receive antennas; 0 if coefficients absent  |
| `.n_tx()`     | Returns number of transmit antennas; 0 if coefficients absent |
| `.n_path()`   | Returns number of paths per snapshot as a vector              |
| `.empty()`    | Returns true if the object contains no channel data           |
| `.is_valid()` | Returns empty string if valid, otherwise an error message     |

### Complex member functions:
- .[add_paths](#add_paths)
- .[calc_effective_path_gain](#calc_effective_path_gain)
- .[write_paths_to_obj_file](#write_paths_to_obj_file)

---
## .add_paths
Append new propagation paths to an existing channel snapshot

- Adds path-level data to snapshot `i_snap` in a `channel` object; does not modify `tx_pos`, `rx_pos`, or orientation fields
- All provided fields must have consistent length `n_path_add` and match existing snapshot structure

### Declaration:
```
void quadriga_lib::channel<dtype>::add_paths(
    arma::uword i_snap,
    const arma::Cube<dtype> *coeff_re_add = nullptr,
    const arma::Cube<dtype> *coeff_im_add = nullptr,
    const arma::Cube<dtype> *delay_add = nullptr,
    const arma::u32_vec *no_interact_add = nullptr,
    const arma::Mat<dtype> *interact_coord_add = nullptr,
    const arma::Col<dtype> *path_gain_add = nullptr,
    const arma::Col<dtype> *path_length_add = nullptr,
    const arma::Mat<dtype> *path_polarization_add = nullptr,
    const arma::Mat<dtype> *path_angles_add = nullptr,
    const arma::Mat<dtype> *path_fbs_pos_add = nullptr,
    const arma::Mat<dtype> *path_lbs_pos_add = nullptr);
```

### Inputs:
- **`i_snap`** — 0-based snapshot index to append paths to
- **`coeff_re_add`** *(optional)* — Real part of channel coefficients; `[n_rx, n_tx, n_path_add]`
- **`coeff_im_add`** *(optional)* — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_add]`
- **`delay_add`** *(optional)* — Propagation delays in seconds; `[n_rx, n_tx, n_path_add]` or `[1, 1, n_path_add]`
- **`no_interact_add`** *(optional)* — Number of interaction points per path; `[n_path_add]`
- **`interact_coord_add`** *(optional)* — Interaction point coordinates; `[3, sum(no_interact)]`
- **`path_gain_add`** *(optional)* — Path gains before antenna effects; `[n_path_add]`
- **`path_length_add`** *(optional)* — Path lengths from TX to RX phase center; `[n_path_add]`
- **`path_polarization_add`** *(optional)* — Interleaved polarization transfer matrices; `[8, n_path_add]`
- **`path_angles_add`** *(optional)* — Departure/arrival angles {AOD, EOD, AOA, EOA} in rad; `[n_path_add, 4]`
- **`path_fbs_pos_add`** *(optional)* — First-bounce scatterer positions; `[3, n_path_add]`
- **`path_lbs_pos_add`** *(optional)* — Last-bounce scatterer positions; `[3, n_path_add]`

---
## .calc_effective_path_gain
Calculate the effective path gain per snapshot in linear scale

- Sums power over all paths and TX/RX antenna pairs to produce one gain value per snapshot
- Uses `coeff_re`/`coeff_im` if available; falls back to `path_polarization` assuming ideal XPOL antennas
- Throws if neither coefficients nor polarization data are present

### Declaration:
```
arma::Col<dtype> quadriga_lib::channel<dtype>::calc_effective_path_gain(bool assume_valid = false) const;
```

### Inputs:
- **`assume_valid`** *(optional)* — Skip internal consistency checks for performance in trusted contexts

### Returns:
- Effective path gains in linear scale, one entry per snapshot; `[n_snap]`

---
## .write_paths_to_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for Blender)
- Tubes are color-coded by path gain using a selected colormap; radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits total count

### Declaration:
```
void quadriga_lib::channel<dtype>::write_paths_to_obj_file(
    std::string fn,
    arma::uword max_no_paths = 0,
    dtype gain_max = -60.0,
    dtype gain_min = -140.0,
    std::string colormap = "jet",
    arma::uvec i_snap = {},
    dtype radius_max = 0.05,
    dtype radius_min = 0.01,
    arma::uword n_edges = 5) const;
```

### Inputs:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** *(optional)* — Max paths to export; `0` includes all paths above `gain_min`
- **`gain_max`** *(optional)* — Upper gain threshold in dB for color/radius mapping; higher values are clipped
- **`gain_min`** *(optional)* — Lower gain threshold in dB; paths below this are excluded
- **`colormap`** *(optional)* — Colormap name; see [colormap](#colormap) for supported options
- **`i_snap`** *(optional)* — 0-based snapshot indices to include; empty exports all snapshots
- **`radius_max`** *(optional)* — Tube radius at maximum gain
- **`radius_min`** *(optional)* — Tube radius at minimum gain
- **`n_edges`** *(optional)* — Vertices per tube cross-section; must be ≥ 3

### See also:
- [path_to_tube](#path_to_tube) (generates tube geometry from path data)
- [colormap](#colormap) (colormap lookup used for coloring)

---

# Channel functions

---
## any_type_id
Get type ID and raw access from a `std::any` object

- Inspects a `std::any` object and returns an integer type identifier for its contents
- Optionally retrieves dimensions (rows, columns, slices) for Armadillo matrix/cube/vector types; for `std::string`, `dims[0]` is the string length, `dims[1]`/`dims[2]` are zero
- Optionally retrieves a raw `void*` to the internal data — not type-safe, bypasses `const` protection; use with caution

### Declaration:
```
int quadriga_lib::any_type_id(
    const std::any *data,
    unsigned long long *dims = nullptr,
    void **dataptr = nullptr);
```

### Inputs:
- **`data`** — Pointer to the `std::any` object to inspect

### Outputs:
- **`dims`** *(optional)* — Array of 3 values filled with `[rows, cols, slices]` of the contained Armadillo object
- **`dataptr`** *(optional)* — Receives a raw pointer to the object's internal data

### Returns:
- Integer type ID of the contained value:
  | ID  | Type                      | ID  | Type                   | ID  | Type                      |
  | --- | ------------------------- | --- | ---------------------- | --- | ------------------------- |
  | -2  | `no value`                | -1  | `unsupported type`     | 9   | `std::string`             |
  | 10  | `float`                   | 11  | `double`               | 12  | `unsigned long long int`  |
  | 13  | `long long int`           | 14  | `unsigned int`         | 15  | `int`                     |
  | 20  | `arma::Mat<float>`        | 21  | `arma::Mat<double>`    | 22  | `arma::Mat<arma::uword>`  |
  | 23  | `arma::Mat<arma::sword>`  | 24  | `arma::Mat<unsigned>`  | 25  | `arma::Mat<int>`          |
  | 30  | `arma::Cube<float>`       | 31  | `arma::Cube<double>`   | 32  | `arma::Cube<arma::uword>` |
  | 33  | `arma::Cube<arma::sword>` | 34  | `arma::Cube<unsigned>` | 35  | `arma::Cube<int>`         |
  | 40  | `arma::Col<float>`        | 41  | `arma::Col<double>`    | 42  | `arma::Col<arma::uword>`  |
  | 43  | `arma::Col<arma::sword>`  | 44  | `arma::Col<unsigned>`  | 45  | `arma::Col<int>`          |
  | 50  | `arma::Row<float>`        | 51  | `arma::Row<double>`    | 52  | `arma::Row<arma::uword>`  |
  | 53  | `arma::Row<arma::sword>`  | 54  | `arma::Row<unsigned>`  | 55  | `arma::Row<int>`          |

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (uses `any_type_id` to read dataset from HDF5 file)
- [hdf5_write_dset](#hdf5_write_dset) (HDF5 dataset writer)

---
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

- Computes the frequency-domain channel matrix `H` at given sub-carrier positions via DFT over time-domain
  path coefficients and delays
- `delay` supports broadcasting: shape `[1, 1, n_path]` applies the same delays to all RX/TX pairs
- `pilot_grid` values are normalized to bandwidth: `0.0` = center frequency, `1.0` = center + bandwidth
- Internal arithmetic is single-precision; uses AVX2 for 8-carrier parallel computation; double inputs are
  narrowed to float internally, results widened back
- Safe to call in a loop over snapshots and parallelize with OpenMP

### Declaration:
```
void quadriga_lib::baseband_freq_response(
    const arma::Cube<dtype> *coeff_re,
    const arma::Cube<dtype> *coeff_im,
    const arma::Cube<dtype> *delay,
    const arma::Col<dtype> *pilot_grid,
    const double bandwidth,
    arma::Cube<dtype> *hmat_re,
    arma::Cube<dtype> *hmat_im,
    arma::Cube<std::complex<dtype>> *hmat = nullptr);
```

### Inputs:
- **`coeff_re`** — Real part of time-domain channel coefficients; `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of time-domain channel coefficients; `[n_rx, n_tx, n_path]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`
- **`pilot_grid`** — Normalized sub-carrier positions in range `[0.0, 1.0]`; `[n_carriers]`
- **`bandwidth`** — Total baseband bandwidth

### Outputs:
- **`hmat_re`** *(optional)* — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carriers]`
- **`hmat_im`** *(optional)* — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carriers]`
- **`hmat`** *(optional)* — Complex-valued frequency-domain channel matrix; `[n_rx, n_tx, n_carriers]`

### See also:
- [baseband_freq_response_vec](#baseband_freq_response_vec) (vectorized version)
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-freq counterpart)
- [get_channels_planar](#get_channels_planar) (for generating coeff and delay)
- [get_channels_spherical](#get_channels_spherical) (for generating coeff and delay)

---
## baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

- Interpolates complex channel coefficients from a coarse input frequency grid (`freq_in`) to a dense
  output grid (`freq_out`) using SLERP: magnitude and unwrapped phase are each interpolated linearly along the shortest arc
- Applies delay-induced phase rotation `exp(-j·2·pi·freq_out·delay)` per output carrier in double
  precision to preserve accuracy at high carrier frequencies
- Only `delay[0]` is used; all entries in the `delay` vector should be identical
  (path geometry is frequency-independent)
- `delay` cube supports `[1, 1, n_path]` (planar wave) or `[n_rx, n_tx, n_path]` (spherical wave)
- Output frequencies outside the range of `freq_in` use constant extrapolation from the nearest endpoint
- At least one of `hmat_re`/`hmat_im` or `hmat` must be non-null

### Declaration:
```
void quadriga_lib::baseband_freq_response_multi(
    const std::vector<arma::Cube<dtype>> &coeff_re,
    const std::vector<arma::Cube<dtype>> &coeff_im,
    const std::vector<arma::Cube<dtype>> &delay,
    const arma::Col<dtype> &freq_in,
    const arma::Col<dtype> &freq_out,
    arma::Cube<dtype> *hmat_re = nullptr,
    arma::Cube<dtype> *hmat_im = nullptr,
    arma::Cube<std::complex<dtype>> *hmat = nullptr,
    bool remove_delay_phase = true);
```

### Inputs:
- **`coeff_re`** — Real part of channel coefficients at each input frequency, vector of `n_freq_in` cubes `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of channel coefficients at each input frequency, same structure as `coeff_re`
- **`delay`** — Path delays in seconds, vector of `n_freq_in` cubes; only `delay[0]` is used; shape `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`
- **`freq_in`** — Input sample frequencies, sorted ascending; `[n_freq_in]`
- **`freq_out`** — Output carrier frequencies (absolute); `[n_carrier]`
- **`remove_delay_phase`** *(optional)* — Removes baked-in `exp(-j·2π·freq_in[f]·delay)` before SLERP and
  re-applies analytically at output frequencies; must be `true` for output from
  [get_channels_multifreq](#get_channels_multifreq) or [get_channels_spherical](#get_channels_spherical), `false` for pure envelope coefficients

### Outputs:
- **`hmat_re`** *(optional)* — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
- **`hmat_im`** *(optional)* — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
- **`hmat`** *(optional)* — Complex-valued frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-snapshot narrowband version)
- [baseband_freq_response_vec](#baseband_freq_response_vec) (batched narrowband version)
- [get_channels_multifreq](#get_channels_multifreq) (produces the multi-frequency input coefficients)

---
## baseband_freq_response_vec
Compute the baseband frequency response of multiple MIMO channels

- Batch wrapper around [baseband_freq_response](#baseband_freq_response), applying it across snapshots in parallel via OpenMP
- Each element of the input vectors is a cube of shape `[n_rx, n_tx, n_path]`; `delay` supports broadcasting to `[1, 1, n_path]`
- Output vectors have length `n_out`: either `n_snap` (all snapshots) or `length(i_snap)` (subset)
- Internal arithmetic is single-precision

### Declaration:
```
void quadriga_lib::baseband_freq_response_vec(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    const arma::Col<dtype> *pilot_grid,
    const double bandwidth,
    std::vector<arma::Cube<dtype>> *hmat_re = nullptr,
    std::vector<arma::Cube<dtype>> *hmat_im = nullptr,
    const arma::uvec *i_snap = nullptr);
```

### Inputs:
- **`coeff_re`** — Real part of time-domain channel coefficients, vector of `n_snap` cubes `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of time-domain channel coefficients, same structure as `coeff_re`
- **`delay`** — Path delays in seconds, same structure as `coeff_re`; each cube broadcastable to `[1, 1, n_path]`
- **`pilot_grid`** — Normalized sub-carrier positions in range `[0.0, 1.0]`; `[n_carriers]`
- **`bandwidth`** — Total baseband bandwidth
- **`i_snap`** *(optional)* — Snapshot indices to process; if omitted, all `n_snap` snapshots are processed; `[n_out]`

### Outputs:
- **`hmat_re`** *(optional)* — Real part of frequency-domain channel matrices, vector of `n_out` cubes `[n_rx, n_tx, n_carriers]`
- **`hmat_im`** *(optional)* — Imaginary part of frequency-domain channel matrices, same structure as `hmat_re`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-snapshot variant)
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-freq counterpart)

---
## get_HDF5_version
Return the HDF5 version string as defined by the compile-time header macros

### Declaration:
```
std::string quadriga_lib::get_HDF5_version();
```

### Returns:
- Version string in the format `"x.y.z"`, e.g., `"1.12.2"`

---
## hdf5_create
Create a new HDF5 channel file with a defined storage layout

- Initializes a new HDF5 file for storing wireless channel data; overwrites existing files.
- Defines a 4D layout (x, y, z, w) where each index combination maps to one channel storage slot.
- Typical dimension mapping: x = BS, y = UE, z = frequency, w = scenario/repetition.
- Layout can be reshaped later with [hdf5_reshape_layout](#hdf5_reshape_layout) if the total slot count stays constant.

### Declaration:
```
void quadriga_lib::hdf5_create(
    std::string fn,
    unsigned nx = 65536,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

### Inputs:
- **`fn`** — Path and filename of the HDF5 file to create
- **`nx`** *(optional)* — Size of x-dimension
- **`ny`** *(optional)* — Size of y-dimension
- **`nz`** *(optional)* — Size of z-dimension
- **`nw`** *(optional)* — Size of w-dimension

### See also:
- [hdf5_reshape_layout](#hdf5_reshape_layout) (change layout dimensions of an existing file)
- [hdf5_write](#hdf5_write) (write channel data into a slot)

---
## hdf5_read_channel
Read a channel object from an HDF5 file at a specified 4D index

- Returns an empty channel object (`no_snapshots == 0`) if the slot contains no valid data.
- Structured data is stored in single precision in the file and converted to `dtype` on read.
- Unstructured fields (`std::any`) retain their original stored type without conversion.

### Declaration:
```
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(
    std::string fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0);
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension

### Returns:
- Channel object at the specified slot; empty if no data is present

### See also:
- [hdf5_write](#hdf5_write) (write a channel object to a slot)
- [hdf5_read_layout](#hdf5_read_layout) (inspect slot occupancy before reading)

---
## hdf5_read_dset
Read an unstructured dataset from an HDF5 file at a specified 4D index

- Reads a user-defined dataset stored under `prefix + par_name` (e.g., `"par_carrier_frequency"`).
- Returns an empty `std::any` if the dataset does not exist at the specified slot or name.
- Use [any_type_id](#any_type_id) to determine the contained type and obtain a raw pointer.

### Declaration:
```
std::any quadriga_lib::hdf5_read_dset(
    std::string fn,
    std::string par_name,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_name`** — Dataset name without prefix, e.g., `"carrier_frequency"`
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Dataset name prefix prepended before `par_name`

### Returns:
- `std::any` containing the dataset, or empty `std::any` if not found

### See also:
- [hdf5_write_dset](#hdf5_write_dset) (write an unstructured dataset)
- [any_type_id](#any_type_id) (inspect the type held in a `std::any`)

---
## hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`; returned names exclude the prefix.

### Declaration:
```
arma::uword quadriga_lib::hdf5_read_dset_names(
    std::string fn,
    std::vector<std::string> *par_names,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_names`** — Pointer to vector receiving dataset names (without prefix)
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Prefix used to identify unstructured datasets

### Returns:
- Number of datasets found at the specified slot

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (read a dataset by name)
- [hdf5_write_dset](#hdf5_write_dset) (write an unstructured dataset)

---
## hdf5_read_layout
Read the storage layout of an HDF5 channel file

- Returns `{nx, ny, nz, nw}` describing the 4D slot grid of the file.
- Returns `{0, 0, 0, 0}` if the file does not exist; throws if the file exists but is not a valid HDF5 file.
- `channelID` entries are `0` for empty slots; length equals `nx × ny × nz × nw` (serialized linear index).

### Declaration:
```
arma::u32_vec quadriga_lib::hdf5_read_layout(
    std::string fn,
    arma::u32_vec *channelID = nullptr);
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`channelID`** *(optional)* — Pointer to vector receiving the serialized slot occupancy list; `[nx·ny·nz·nw]`

### Returns:
- Four-element vector `{nx, ny, nz, nw}` describing the layout dimensions

### See also:
- [hdf5_create](#hdf5_create) (create a file with a defined layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (change layout dimensions of an existing file)

---
## hdf5_reshape_layout
Reshape the 4D storage layout of an existing HDF5 channel file

- Updates `{nx, ny, nz, nw}` of an existing file; total slot count `nx × ny × nz × nw` must remain unchanged.
- Throws if the new layout violates the total-count constraint.

### Declaration:
```
void quadriga_lib::hdf5_reshape_layout(
    std::string fn,
    unsigned nx,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`nx`** — New size of x-dimension
- **`ny`** *(optional)* — New size of y-dimension
- **`nz`** *(optional)* — New size of z-dimension
- **`nw`** *(optional)* — New size of w-dimension

### See also:
- [hdf5_create](#hdf5_create) (set initial layout at file creation)
- [hdf5_read_layout](#hdf5_read_layout) (query current layout)

---
## hdf5_write
Write a channel object to an HDF5 file at a specified 4D index

- Writes a `quadriga_lib::channel<dtype>` object to slot `(ix, iy, iz, iw)` in the HDF5 file.
- Creates the file with default layout `(65536 × 1 × 1 × 1)` if it does not exist; appends to existing files.
- Overwrites slot content if the index already contains data.
- Throws if the index was not reserved during [hdf5_create](#hdf5_create).
- Structured data is always stored in single precision; input may be float or double.
- Unstructured data: supported types are string, double, float, (u)int32, (u)int64; up to 3D; storage order preserved.
- Set `assume_valid = true` to skip integrity validation (faster but unsafe for potentially corrupted data).

### Declaration:
```
int quadriga_lib::hdf5_write(
    const quadriga_lib::channel<dtype> *ch,
    std::string fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    bool assume_valid = false);
```

### Inputs:
- **`ch`** — Pointer to the channel object to write
- **`fn`** — Path to the HDF5 file
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`assume_valid`** *(optional)* — Skip channel integrity validation before writing

### Returns:
- `0` if a new dataset was created, `1` if an existing dataset was overwritten or extended

### See also:
- [hdf5_create](#hdf5_create) (create file and reserve layout)
- [hdf5_read_channel](#hdf5_read_channel) (read channel data from a slot)

---
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file at a specified 4D index

- Dataset is stored under `prefix + par_name`; name must contain only alphanumeric characters and underscores.
- Supported scalar types: `std::string`, `unsigned`, `int`, `long long`, `unsigned long long`, `float`, `double`
- Supported Armadillo types: `arma::Col`, `arma::Row`, `arma::Mat`, `arma::Cube` with element types `float`, `double`, `int`, `unsigned`, `sword`, `uword`, `unsigned long long`
- `arma::Row` is converted to `arma::Col` before writing.
- Throws for unsupported types.
- Throws if a dataset with the same name already exists at the specified slot; no overwrite/update is supported.

### Declaration:
```
void quadriga_lib::hdf5_write_dset(
    std::string fn,
    std::string par_name,
    const std::any *par_data,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_name`** — Dataset name without prefix; alphanumeric and underscores only
- **`par_data`** — Pointer to the data to write; type must be supported (see above)
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Prefix prepended to `par_name` in the HDF5 file

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (read an unstructured dataset by name)
- [hdf5_read_dset_names](#hdf5_read_dset_names) (list available dataset names at a slot)
- [any_type_id](#any_type_id) (inspect the type held in a `std::any`)

---
## qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR offsets, names, positions, orientations, and file version.
- All output arguments are optional; pass `nullptr` to skip any.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and `cir_offset` reflect this.

### Declaration:
```
void quadriga_lib::qrt_file_parse(
    const std::string &fn,
    arma::uword *no_cir = nullptr,
    arma::uword *no_orig = nullptr,
    arma::uword *no_dest = nullptr,
    arma::uword *no_freq = nullptr,
    arma::uvec *cir_offset = nullptr,
    std::vector<std::string> *orig_names = nullptr,
    std::vector<std::string> *dest_names = nullptr,
    int *version = nullptr,
    arma::fvec *freq = nullptr,
    arma::fmat *cir_pos = nullptr,
    arma::fmat *cir_orientation = nullptr,
    arma::fmat *orig_pos = nullptr,
    arma::fmat *orig_orientation = nullptr,
    std::ifstream *file = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally

### Outputs:
- **`no_cir`** *(optional)* — Number of channel snapshots per origin point
- **`no_orig`** *(optional)* — Number of origin points (TX)
- **`no_dest`** *(optional)* — Number of destination points (RX)
- **`no_freq`** *(optional)* — Number of frequency bands
- **`cir_offset`** *(optional)* — CIR offset per destination; `[no_dest]`
- **`orig_names`** *(optional)* — Names of origin points; `[no_orig]`
- **`dest_names`** *(optional)* — Names of destination points; `[no_dest]`
- **`version`** *(optional)* — QRT file version number
- **`freq`** *(optional)* — Frequencies as stored in the file; usually in GHz; `[no_freq]`
- **`cir_pos`** *(optional)* — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** *(optional)* — CIR orientations as Euler angles; `[no_cir, 3]`
- **`orig_pos`** *(optional)* — Origin (TX) positions in Cartesian coordinates; `[no_orig, 3]`
- **`orig_orientation`** *(optional)* — Origin (TX) orientations as Euler angles; `[no_orig, 3]`

---
## qrt_file_read
Read ray-tracing CIR data from a QRT file

- Reads channel impulse response data for a specific snapshot index and origin point.
- All output arguments are optional; pass `nullptr` to skip any.
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped.
- For tight-loop performance, pass a pre-opened `std::ifstream` and a [qrt_read_cache_init](#qrt_read_cache_init)-populated cache; reduces per-call I/O to 2 seeks and 4 reads.
- `fn` is ignored when both `file` and `cache` are provided.

### Declaration:
```
void quadriga_lib::qrt_file_read(
    const std::string &fn,
    arma::uword i_cir = 0,
    arma::uword i_orig = 0,
    bool downlink = true,
    arma::Col<dtype> *center_frequency = nullptr,
    arma::Col<dtype> *tx_pos = nullptr,
    arma::Col<dtype> *tx_orientation = nullptr,
    arma::Col<dtype> *rx_pos = nullptr,
    arma::Col<dtype> *rx_orientation = nullptr,
    arma::Mat<dtype> *fbs_pos = nullptr,
    arma::Mat<dtype> *lbs_pos = nullptr,
    arma::Mat<dtype> *path_gain = nullptr,
    arma::Col<dtype> *path_length = nullptr,
    arma::Cube<dtype> *M = nullptr,
    arma::Col<dtype> *aod = nullptr,
    arma::Col<dtype> *eod = nullptr,
    arma::Col<dtype> *aoa = nullptr,
    arma::Col<dtype> *eoa = nullptr,
    std::vector<arma::Mat<dtype>> *path_coord = nullptr,
    int normalize_M = 1,
    arma::u32_vec *no_int = nullptr,
    arma::fmat *coord = nullptr,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are provided
- **`i_cir`** — Snapshot index, 0-based
- **`i_orig`** — Origin index, 0-based
- **`downlink`** — If `true`, origin=TX, destination=RX; if `false`, roles are swapped
- **`normalize_M`** *(optional)* — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
  | `normalize_M` | `M`                   | `path_gain`                      |
  | ------------- | --------------------- | -------------------------------- |
  | 0             | As stored in QRT file | -PL                              |
  | 1             | Max column power = 1  | -PL minus material losses        |
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; left open on return
- **`cache`** *(optional)* — Pre-populated cache from [qrt_read_cache_init](#qrt_read_cache_init)

### Outputs:
- **`center_frequency`** *(optional)* — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** *(optional)* — Transmitter position in Cartesian coordinates; `[3]`
- **`tx_orientation`** *(optional)* — Transmitter orientation (bank, tilt, heading); `[3]`
- **`rx_pos`** *(optional)* — Receiver position in Cartesian coordinates; `[3]`
- **`rx_orientation`** *(optional)* — Receiver orientation (bank, tilt, heading); `[3]`
- **`fbs_pos`** *(optional)* — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** *(optional)* — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** *(optional)* — Path gain on linear scale; `[n_path, n_freq]`
- **`path_length`** *(optional)* — Absolute path length TX to RX phase center; `[n_path]`
- **`M`** *(optional)* — Polarization transfer matrix; `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** *(optional)* — Departure azimuth angles; `[n_path]`
- **`eod`** *(optional)* — Departure elevation angles; `[n_path]`
- **`aoa`** *(optional)* — Arrival azimuth angles; `[n_path]`
- **`eoa`** *(optional)* — Arrival elevation angles; `[n_path]`
- **`path_coord`** *(optional)* — Interaction coordinates per path; vector of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** *(optional)* — Number of mesh interactions per path; 0 indicates LOS; `[n_path]`
- **`coord`** *(optional)* — Interaction coordinates; `[3, sum(no_int)]`

### Example:
```
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);
arma::vec center_freq, tx_pos, rx_pos, path_length;
arma::mat path_gain; arma::cube M;
for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            nullptr, nullptr, &path_gain, &path_length, &M,
            nullptr, nullptr, nullptr, nullptr, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```

### See also:
- [qrt_read_cache_init](#qrt_read_cache_init) (populate cache for fast repeated reads)
- [qrt_file_parse](#qrt_file_parse) (extract file metadata without reading CIR data)

---
## qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

- Reads all fixed metadata from a QRT file into a `quadriga_lib::qrt_read_cache` struct.
- Pre-computes byte offsets so subsequent [qrt_file_read](#qrt_file_read) calls need only 2 seeks and 4 reads instead of re-parsing the header.
- Populate once, then pass the cache and a shared `std::ifstream` to [qrt_file_read](#qrt_file_read) for tight-loop performance.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.

### Declaration:
```
quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(
    const std::string &fn,
    std::ifstream *file = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally

### Returns:
- Populated `quadriga_lib::qrt_read_cache` struct with the following members:
  | Member             | Type         | Description                                                      |
  | ------------------ | ------------ | ---------------------------------------------------------------- |
  | `version`          | `int`        | QRT file version                                                 |
  | `no_orig`          | `unsigned`   | Number of origin (TX) positions                                  |
  | `no_cir`           | `unsigned`   | Number of CIRs per origin                                        |
  | `no_dest`          | `unsigned`   | Number of destinations (RX)                                      |
  | `no_freq`          | `unsigned`   | Number of frequency bands                                        |
  | `freq`             | `arma::fvec` | Frequency in GHz; `[no_freq]`                                    |
  | `cir_pos`          | `arma::fmat` | CIR positions; `[no_cir, 3]`                                     |
  | `cir_orientation`  | `arma::fmat` | CIR orientations (Euler); `[no_cir, 3]`                          |
  | `orig_pos_all`     | `arma::fmat` | Origin positions; `[no_orig, 3]`                                 |
  | `orig_orientation` | `arma::fmat` | Origin orientations (Euler); `[no_orig, 3]`                      |
  | `orig_index`       | `arma::uvec` | Byte offsets from BOF to each origin data block; `[no_orig]`     |
  | `path_data_offset` | `arma::uvec` | Absolute offset to path_data_index array per origin; `[no_orig]` |

---
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1-delta)^alpha and delta^alpha,
  where delta is the fractional offset within the bin and alpha is `power_exponent`; this avoids discontinuities when 
  delays cross tap boundaries
- Use `power_exponent=1.0` for narrowband (linear interpolation) or `0.5` for wideband (incoherent power preservation)
- If all fractional offsets are below 0.01 or above 0.99, weight computation is skipped but tap-selection logic still applies
- Input `delay` may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`; shared delays are expanded 
  internally when `fix_taps` is 0 or 3, and output delays remain shared `[1, 1, n_taps]` when `fix_taps` is 1 or 2
- `n_rx` and `n_tx` must be identical across all snapshots; `n_path_s` may differ per snapshot

### Declaration:
```
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    std::vector<arma::Cube<dtype>> *coeff_re_quant,
    std::vector<arma::Cube<dtype>> *coeff_im_quant,
    std::vector<arma::Cube<dtype>> *delay_quant,
    dtype tap_spacing = (dtype)5.0e-9,
    arma::uword max_no_taps = 48,
    dtype power_exponent = (dtype)1.0,
    int fix_taps = 0);
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]`
- **`coeff_im`** — Channel coefficients, imaginary part; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`
- **`tap_spacing`** *(optional)* — Delay bin spacing in seconds; 5 ns corresponds to 200 MHz sampling rate
- **`max_no_taps`** *(optional)* — Maximum number of output taps; 0 means unlimited
- **`power_exponent`** *(optional)* — Interpolation exponent alpha; 1.0 = narrowband, 0.5 = wideband power-preserving
- **`fix_taps`** *(optional)* — Delay grid sharing mode:
  | Value | Meaning                                                                                                                               |
  | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps]`                                                                     |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps]`, identical for every snapshot                 |
  | 2     | Per snapshot; output delays `[1, 1, n_taps]`, but each snapshot has its own independent tap grid — taps do not align across snapshots |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps]`                                                             |

### Outputs:
- **`coeff_re_quant`** — Output coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_taps]`
- **`coeff_im_quant`** — Output coefficients, imaginary part; same layout as `coeff_re_quant`
- **`delay_quant`** — Output delays in seconds; each cube `[n_rx, n_tx, n_taps]` or `[1, 1, n_taps]` depending on `fix_taps`

### Example:
```
std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
cre[0].set_size(1,1,3); cim[0].set_size(1,1,3); dl[0].set_size(1,1,3);
cre[1].set_size(1,1,2); cim[1].set_size(1,1,2); dl[1].set_size(1,1,2);
dl[0](0,0,1) = 12.5e-9; dl[0](0,0,2) = 33.4e-9; dl[1](0,0,1) = 10.0e-9;
std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);
```

---

# Channel generation functions

---
## get_channels_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions
- 2D model: azimuth angles and planar motion only, no elevation
- Supported channel types: `A, B, C, D, E, F` (TGn definitions)
- MU-MIMO supported (`n_users > 1`) with per-user distances/floors and optional angle offsets per TGac
- Time-evolving channels via `observation_time`, `update_rate`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Default KF (linear): A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0; applied to first tap only; breakpoint ignored when `KF_linear >= 0`
- Default XPR NLOS: 2 (3 dB); default SF LOS: 3 dB; default SF NLOS: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB
- Default breakpoint distance: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- Floor floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz
- NAN or negative value for any override parameter restores the model default

### Declaration:
```
std::vector<quadriga_lib::channel<double>> quadriga_lib::get_channels_ieee_indoor(
    const quadriga_lib::arrayant<double> &ap_array,
    const quadriga_lib::arrayant<double> &sta_array,
    std::string ChannelType,
    double CarrierFreq_Hz = 5.25e9,
    double tap_spacing_s = 10.0e-9,
    arma::uword n_users = 1,
    double observation_time = 0.0,
    double update_rate = 1.0e-3,
    double speed_station_kmh = 0.0,
    double speed_env_kmh = 1.2,
    arma::vec Dist_m = {4.99},
    arma::uvec n_floors = {0},
    bool uplink = false,
    arma::mat offset_angles = {},
    arma::uword n_subpath = 20,
    double Doppler_effect = 50.0,
    arma::sword seed = -1,
    double KF_linear = NAN,
    double XPR_NLOS_linear = NAN,
    double SF_std_dB_LOS = NAN,
    double SF_std_dB_NLOS = NAN,
    double dBP_m = NAN,
    arma::uvec n_walls = {0},
    double wall_loss = 5.0);
```

### Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see [arrayant](#arrayant)
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see [arrayant](#arrayant)
- **`ChannelType`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`
- **`observation_time`** *(optional)* — Channel observation time in seconds
- **`update_rate`** *(optional)* — Channel update interval in seconds; relevant only when `observation_time > 0`
- **`speed_station_kmh`** *(optional)* — Station speed in km/h; movement direction is `AoA_offset`; relevant only when `observation_time > 0`
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h; use `0.089` for TGac; relevant only when `observation_time > 0`
- **`Dist_m`** *(optional)* — TX-to-RX distance(s); `[n_users]` or `[1]`
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models; `[n_users]` or `[1]`
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; empty uses TGac auto-defaults for `n_users > 1`; `[4, n_users]`
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); `0.0` disables
- **`seed`** *(optional)* — RNG seed for repeatability; `-1` uses the system random device
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN or negative restores model default
- **`SF_std_dB_LOS`** *(optional)* — Overrides LOS shadow fading std in dB (applied when d < dBP); NAN restores model default
- **`SF_std_dB_NLOS`** *(optional)* — Overrides NLOS shadow fading std in dB (applied when d >= dBP); NAN restores model default
- **`dBP_m`** *(optional)* — Overrides breakpoint distance; NAN or negative restores model default
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `[n_users]` or `[1]`
- **`wall_loss`** *(optional)* — Penetration loss for a single wall; TGax defines 5.0 (default) or 7.0

### Returns:
- `std::vector<quadriga_lib::channel<double>>` of length `n_users`; each entry is one user's channel realization with direction set by `uplink`

### See also:
- [get_channels_planar](#get_channels_planar) (used internally to compute MIMO coefficients per user)
- [arrayant](#arrayant) (antenna array type for ap_array and sta_array)
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>

---
## get_channels_irs
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2` (interleaved Re/Im for VV, VH, HV, HH components)
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- Optional `irs_array_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- Setting `center_frequency = 0.0` disables phase computation

### Declaration:
```
std::vector<bool> quadriga_lib::get_channels_irs(
    const quadriga_lib::arrayant<dtype> *tx_array,
    const quadriga_lib::arrayant<dtype> *rx_array,
    const quadriga_lib::arrayant<dtype> *irs_array,
    dtype Tx, dtype Ty, dtype Tz,
    dtype Tb, dtype Tt, dtype Th,
    dtype Rx, dtype Ry, dtype Rz,
    dtype Rb, dtype Rt, dtype Rh,
    dtype Ix, dtype Iy, dtype Iz,
    dtype Ib, dtype It, dtype Ih,
    const arma::Mat<dtype> *fbs_pos_1,
    const arma::Mat<dtype> *lbs_pos_1,
    const arma::Col<dtype> *path_gain_1,
    const arma::Col<dtype> *path_length_1,
    const arma::Mat<dtype> *M_1,
    const arma::Mat<dtype> *fbs_pos_2,
    const arma::Mat<dtype> *lbs_pos_2,
    const arma::Col<dtype> *path_gain_2,
    const arma::Col<dtype> *path_length_2,
    const arma::Mat<dtype> *M_2,
    arma::Cube<dtype> *coeff_re,
    arma::Cube<dtype> *coeff_im,
    arma::Cube<dtype> *delay,
    arma::uword i_irs = 0,
    dtype threshold_dB = -140.0,
    dtype center_frequency = 0.0,
    bool use_absolute_delays = false,
    arma::Cube<dtype> *aod = nullptr,
    arma::Cube<dtype> *eod = nullptr,
    arma::Cube<dtype> *aoa = nullptr,
    arma::Cube<dtype> *eoa = nullptr,
    const quadriga_lib::arrayant<dtype> *irs_array_2 = nullptr,
    const std::vector<bool> *active_path = nullptr);
```

### Inputs:
- **`tx_array`** — Transmit antenna array with `n_tx` elements; see [arrayant](#arrayant)
- **`rx_array`** — Receive antenna array with `n_rx` elements; see [arrayant](#arrayant)
- **`irs_array`** — IRS antenna array (TX-facing side) with `n_irs` elements; see [arrayant](#arrayant)
- **`Tx, Ty, Tz`** — Transmitter position in Cartesian coordinates
- **`Tb, Tt, Th`** — Transmitter orientation as Euler angles (bank, tilt, heading)
- **`Rx, Ry, Rz`** — Receiver position in Cartesian coordinates
- **`Rb, Rt, Rh`** — Receiver orientation as Euler angles (bank, tilt, heading)
- **`Ix, Iy, Iz`** — IRS position in Cartesian coordinates
- **`Ib, It, Ih`** — IRS orientation as Euler angles (bank, tilt, heading)
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center; `[n_path_1]`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; `[n_path_2]`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path_2]`
- **`i_irs`** *(optional)* — IRS codebook port index
- **`threshold_dB`** *(optional)* — Gain threshold in dB; path combinations below this are discarded
- **`center_frequency`** *(optional)* — Center frequency; set to `0` to skip phase computation
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component
- **`irs_array_2`** *(optional)* — Second IRS antenna array for the RX-facing side; enables asymmetric IRS patterns; see [arrayant](#arrayant)
- **`active_path`** *(optional)* — Bitmask selecting active path pairs; overrides `threshold_dB`; `[n_path_1 * n_path_2]`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_irs]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path_irs]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path_irs]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path_irs]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path_irs]`

### Returns:
- Boolean mask of length `n_path_1 * n_path_2` indicating which path combinations were included in the output

### See also:
- [combine_irs_coord](#combine_irs_coord) (coordinate setup for IRS geometry)
- [get_channels_spherical](#get_channels_spherical) (single-segment spherical-wave channel)
- [get_channels_planar](#get_channels_planar) (single-segment planar-wave channel)
- [arrayant](#arrayant) (antenna array class)

---
## get_channels_multifreq
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [get_channels_spherical](#get_channels_spherical) with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (from `tx_array[i].center_frequency`), RX array, input samples (`freq_in`), and output (`freq_out`)
- TX/RX patterns interpolated per output frequency via SLERP with linear fallback (same as [arrayant_interpolate_multi](#arrayant_interpolate_multi))
- `path_gain` interpolated linearly; `M` interpolated via SLERP per complex entry pair to preserve phase
- Extrapolation clamps to nearest frequency entry on all four grids
- `propagation_speed` supports EM (speed of light, default) and acoustic (~343 m/s) simulations
- `M` accepts 8 rows (full polarimetric: ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) or 2 rows (scalar pressure: ReVV, ImVV only)
- Coupling matrices interpolated across frequencies per complex entry (SLERP for complex pairs), identical to antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`

### Declaration:
```
void quadriga_lib::get_channels_multifreq(
    const std::vector<arrayant<dtype>> &tx_array,
    const std::vector<arrayant<dtype>> &rx_array,
    dtype Tx, dtype Ty, dtype Tz,
    dtype Tb, dtype Tt, dtype Th,
    dtype Rx, dtype Ry, dtype Rz,
    dtype Rb, dtype Rt, dtype Rh,
    const arma::Mat<dtype> &fbs_pos,
    const arma::Mat<dtype> &lbs_pos,
    const arma::Mat<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Cube<dtype> &M,
    const arma::Col<dtype> &freq_in,
    const arma::Col<dtype> &freq_out,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    dtype propagation_speed = dtype(299792458.0));
```

### Inputs:
- **`tx_array`** — Multi-frequency TX arrayant vector; all entries must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi)
- **`rx_array`** — Multi-frequency RX arrayant vector; all entries must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi)
- **`Tx, Ty, Tz`** — TX position in Cartesian coordinates
- **`Tb, Tt, Th`** — TX orientation, Euler angles (bank, tilt, heading)
- **`Rx, Ry, Rz`** — RX position in Cartesian coordinates
- **`Rb, Rt, Rh`** — RX orientation, Euler angles  (bank, tilt, heading)
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Linear-scale path gains; `[n_path, n_freq_in]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq_in]` (full pol) or `[2, n_path, n_freq_in]` (scalar)
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `[n_freq_in]`
- **`freq_out`** — Target output frequencies; `[n_freq_out]`
- **`use_absolute_delays`** *(optional)* — Include LOS delay in all paths if true
- **`add_fake_los_path`** *(optional)* — Add zero-power LOS path if none detected
- **`propagation_speed`** *(optional)* — Wave speed [m/s]; use ~343.0 for acoustics

### Outputs:
- **`coeff_re`** — Real part of coefficients; vector length `n_freq_out`, each cube `[n_rx_ports, n_tx_ports, n_path_out]`
- **`coeff_im`** — Imaginary part; same structure as `coeff_re`
- **`delay`** — Propagation delays [s]; same structure as `coeff_re`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency equivalent)
- [arrayant_interpolate_multi](#arrayant_interpolate_multi) (underlying pattern interpolation)
- [arrayant_concat_multi](#arrayant_concat_multi) (building multi-frequency arrays)
- [generate_speaker](#generate_speaker) (acoustic source construction)

---
## get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX element pairs across `n_path` propagation paths.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based (angles ignored).
- Polarization transfer matrix `M` must be normalized; rows are interleaved real/imag components.
- If `add_fake_los_path` is true, a zero-power LOS path is appended, making output size `n_path+1`.
- Setting `center_frequency = 0` disables phase calculation (delays still computed).
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance from all path lengths before converting to delay.

### Declaration:
```
void quadriga_lib::get_channels_planar(
    const quadriga_lib::arrayant<dtype> *tx_array,
    const quadriga_lib::arrayant<dtype> *rx_array,
    dtype Tx, dtype Ty, dtype Tz,
    dtype Tb, dtype Tt, dtype Th,
    dtype Rx, dtype Ry, dtype Rz,
    dtype Rb, dtype Rt, dtype Rh,
    const arma::Col<dtype> *aod,
    const arma::Col<dtype> *eod,
    const arma::Col<dtype> *aoa,
    const arma::Col<dtype> *eoa,
    const arma::Col<dtype> *path_gain,
    const arma::Col<dtype> *path_length,
    const arma::Mat<dtype> *M,
    arma::Cube<dtype> *coeff_re,
    arma::Cube<dtype> *coeff_im,
    arma::Cube<dtype> *delay,
    dtype center_frequency = dtype(0.0),
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    arma::Col<dtype> *rx_Doppler = nullptr);
```

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` elements
- **`rx_array`** — Receive antenna array; `n_rx` elements
- **`Tx, Ty, Tz`** — Transmitter position
- **`Tb, Tt, Th`** — Transmitter orientation: bank, tilt, heading
- **`Rx, Ry, Rz`** — Receiver position
- **`Rb, Rt, Rh`** — Receiver orientation: bank, tilt, heading
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path]`
- **`path_length`** — Path lengths from TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix, row order: ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH; `[8, n_path]`
- **`center_frequency`** *(optional)* — Center frequency; 0 disables phase calculation
- **`use_absolute_delays`** *(optional)* — Include LOS delay offset in all paths
- **`add_fake_los_path`** *(optional)* — Append a zero-power LOS path when no LOS is present

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** *(optional)* — Doppler weights for moving RX; positive = moving toward path, negative = away; `[n_path(+1)]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (spherical wave variant accounting for per-element angle differences)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (for generating IEEE compliant channels using `get_channels_planar` internally)
- [baseband_freq_response](#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed grid)
- [arrayant](#arrayant) (antenna array class)

---
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths, using spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH components).
- If `center_frequency == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == false`, the minimum delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == true`, a zero-power LOS path is prepended when no LOS path is detected.

### Declaration:
```
void quadriga_lib::get_channels_spherical(
    const quadriga_lib::arrayant<dtype> *tx_array,
    const quadriga_lib::arrayant<dtype> *rx_array,
    dtype Tx, dtype Ty, dtype Tz,
    dtype Tb, dtype Tt, dtype Th,
    dtype Rx, dtype Ry, dtype Rz,
    dtype Rb, dtype Rt, dtype Rh,
    const arma::Mat<dtype> *fbs_pos,
    const arma::Mat<dtype> *lbs_pos,
    const arma::Col<dtype> *path_gain,
    const arma::Col<dtype> *path_length,
    const arma::Mat<dtype> *M,
    arma::Cube<dtype> *coeff_re,
    arma::Cube<dtype> *coeff_im,
    arma::Cube<dtype> *delay,
    dtype center_frequency = dtype(0.0),
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    arma::Cube<dtype> *aod = nullptr,
    arma::Cube<dtype> *eod = nullptr,
    arma::Cube<dtype> *aoa = nullptr,
    arma::Cube<dtype> *eoa = nullptr,
    bool use_avx2 = false);
```

### Inputs:
- **`tx_array`** — Transmit antenna array with `n_tx` elements; see [arrayant](#arrayant)
- **`rx_array`** — Receive antenna array with `n_rx` elements; see [arrayant](#arrayant)
- **`Tx, Ty, Tz`** — Transmitter position in Cartesian coordinates
- **`Tb, Tt, Th`** — Transmitter orientation as Euler angles (bank, tilt, heading)
- **`Rx, Ry, Rz`** — Receiver position in Cartesian coordinates
- **`Rb, Rt, Rh`** — Receiver orientation as Euler angles (bank, tilt, heading)
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path]`
- **`path_length`** — Total path lengths from TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path]`
- **`center_frequency`** *(optional)* — Center frequency; set to `0` to skip phase computation
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component
- **`add_fake_los_path`** *(optional)* — If `true`, prepends a zero-power LOS path when none is present
- **`use_avx2`** *(optional)* — If `true`, use AVX2 for antenna interpolation; faster, but less accurate; ignored when not supported

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path]`

### See also:
- [get_channels_planar](#get_channels_planar) (planar wave variant)
- [get_channels_multifreq](#get_channels_multifreq) (multi-freq counterpart)
- [get_channels_irs](#get_channels_irs) (for IRS-assisted communication)
- [arrayant](#arrayant) (antenna array class)
- [baseband_freq_response](#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed grid)

---

# Channel statistics

---
## acdf
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each column CDF are averaged, 
  then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` points to an empty vector, equally spaced bins spanning the data range are generated and stored back; 
  if non-empty, those bin centers are used; if `nullptr`, bins are auto-generated internally

### Declaration:
```
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
    arma::Col<dtype> *bins = nullptr,
    arma::Mat<dtype> *cdf_per_set = nullptr,
    arma::Col<dtype> *cdf_avg = nullptr,
    arma::Col<dtype> *mu = nullptr,
    arma::Col<dtype> *sig = nullptr,
    arma::uword n_bins = 201);
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins`** *(optional)* — Bin centers; auto-generated and stored back if pointing to empty vector, 
  used as-is if non-empty, ignored if `nullptr`; `[n_bins]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when non-empty bins are provided

### Outputs:
- **`cdf_per_set`** *(optional)* — Individual CDFs, one per column of data; `[n_bins, n_sets]`
- **`cdf_avg`** *(optional)* — Averaged CDF via quantile-space averaging across data sets; `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across data sets, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across data sets, `[9]`

---
## calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles; each CIR may have a different number of paths.
- RMS spread formula: `sqrt(sum(pw .* d^2))` where `d` are wrapped deviations from the circular mean (3GPP TR 38.901 second-moment definition).
- Mean direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies on the equator before computing spreads, avoiding pole singularity artifacts.
- When `calc_bank_angle = true`, an optimal bank angle maximizing azimuth spread is derived analytically from eigenvectors of the 2x2 power-weighted covariance matrix of centered angles.
- When `disable_wrapping = true`, spreads are computed directly from raw angles; `orientation` will be zero and `phi`/`theta` equal the input `az`/`el`.
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed before computing spreads.

### Declaration:
```
void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<dtype>> &az,
    const std::vector<arma::Col<dtype>> &el,
    const std::vector<arma::Col<dtype>> &powers,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    std::vector<arma::Col<dtype>> *phi = nullptr,
    std::vector<arma::Col<dtype>> *theta = nullptr,
    bool disable_wrapping = false,
    bool calc_bank_angle = true,
    dtype quantize = (dtype)0);
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_cir]` vector, each element of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; same structure as `az`
- **`powers`** — Path powers in [W]; same structure as `az`
- **`disable_wrapping`** *(optional)* — If true, skips spherical rotation and computes spreads from raw angles
- **`calc_bank_angle`** *(optional)* — If true, computes optimal bank angle analytically; only used when `disable_wrapping = false`
- **`quantize`** *(optional)* — Angular quantization step in [deg]; paths within this distance are grouped; 0 disables grouping

### Outputs:
- **`azimuth_spread`** *(optional)* — RMS azimuth spread; `[n_cir]`
- **`elevation_spread`** *(optional)* — RMS elevation spread; `[n_cir]`
- **`orientation`** *(optional)* — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `[3, n_cir]`
- **`phi`** *(optional)* — Rotated azimuth angles; `[n_cir]` vector, each element of length `n_path`
- **`theta`** *(optional)* — Rotated elevation angles; same structure as `phi`

---
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio.
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis via Jones matrix transform
  `M_circ = T · M_lin · T^-1`.
- LOS paths are identified by comparing path length against direct TX-RX distance `dTR`; paths with
  `path_length < dTR + window_size` are excluded by default (`include_los = false`).
- Polarization transfer matrix `M` is stored column-major with interleaved real/imaginary parts,
  8 rows per path: `[Re(M_vv), Im(M_vv), Re(M_vh), Im(M_vh), Re(M_hv), Im(M_hv), Re(M_hh), Im(M_hh)]`.
- Normalization of `M` does not affect XPR (cancels in ratio) but does affect `pg`.
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR is set to 0.
- TX/RX positions may be fixed `[3, 1]` or mobile `[3, n_cir]`.

### Declaration:
```
void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &M,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr = nullptr,
    arma::Col<dtype> *pg = nullptr,
    bool include_los = false,
    dtype window_size = 0.01);
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_cir]` vector, each element of length `n_path`
- **`M`** — Polarization transfer matrices; `[n_cir]` vector, each element of size `[8, n_path]`
- **`path_length`** — Absolute TX-to-RX path lengths; same structure as `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`include_los`** *(optional)* — If true, includes LOS and near-LOS paths in the XPR calculation
- **`window_size`** *(optional)* — LOS exclusion window; paths within `dTR + window_size` are excluded when `include_los = false`

### Outputs:
- **`xpr`** *(optional)* — XPR on linear scale; `[n_cir, 6]`; columns:
   | Col | Description                                                     |
   | --- | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_hv)^2)                      |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_vh)^2)                      |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_RL)^2)                   |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_LR)^2)                   |
- **`pg`** *(optional)* — Total path gain summed over all paths (including LOS) as
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir]`

---
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; default threshold of 100 dB effectively includes all paths.
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their powers summed before computing the spread; function recurses on the binned profile.

### Declaration:
```
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    dtype threshold = 100.0,
    dtype granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

### Inputs:
- **`delays`** — Delays in [s] per CIR; `[n_cir]` vector, each element a column vector of length `n_path`
- **`powers`** — Path powers in linear scale [W]; same structure as `delays`
- **`threshold`** *(optional)* — Power threshold in [dB] relative to strongest path; paths below threshold are excluded
- **`granularity`** *(optional)* — Bin width in [s] for grouping paths in the delay domain; 0 disables grouping

### Outputs:
- **`mean_delay`** *(optional)* — Mean delay in [s] per CIR; `[n_cir]`

### Returns:
- RMS delay spread in [s] for each CIR; `[n_cir]`

### See also:
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed tap grid)
- [calc_rician_k_factor](#calc_rician_k_factor) (for calculating K-factor)

---
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

- KF = LOS power / NLOS power; LOS paths are those with length ≤ `dTR + window_size`, where `dTR` is the direct TX-RX distance.
- If total NLOS power is zero, KF is set to `HUGE_VAL`; if total LOS power is zero, KF is set to 0.
- TX/RX positions may be fixed `[3, 1]` (reused for all snapshots) or mobile `[3, n_cir]`.

### Declaration:
```
void quadriga_lib::calc_rician_k_factor(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Col<dtype> *kf = nullptr,
    arma::Col<dtype> *pg = nullptr,
    dtype window_size = 0.01);
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_cir]` vector, each element of length `n_path`
- **`path_length`** — Absolute TX-to-RX path lengths; same structure as `powers`
- **`tx_pos`** — Transmitter position in Cartesian coordinates [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`rx_pos`** — Receiver position in Cartesian coordinates [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`window_size`** *(optional)* — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS

### Outputs:
- **`kf`** *(optional)* — Rician K-Factor on linear scale; `[n_cir]`
- **`pg`** *(optional)* — Total path gain (sum of all path powers) in [W]; `[n_cir]`

---

# Math functions

---
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- Internally uses double precision regardless of `dtype`

### Declaration:
```
arma::Cube<dtype> quadriga_lib::calc_rotation_matrix(
    const arma::Cube<dtype> &orientation,
    bool invert_y_axis = false, 
    bool transposeR = false);

arma::Mat<dtype> quadriga_lib::calc_rotation_matrix(
    const arma::Mat<dtype> &orientation,
    bool invert_y_axis = false, 
    bool transposeR = false);

arma::Col<dtype> quadriga_lib::calc_rotation_matrix(
    const arma::Col<dtype> &orientation,
    bool invert_y_axis = false, 
    bool transposeR = false);
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, head); `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`invert_y_axis`** *(optional)* — Flips the sign of the tilt angle, i.e. applies `-tilt` instead of `tilt`; use when the input convention defines positive tilt as downward
- **`transposeR`** *(optional)* — Returns the transpose of the rotation matrix

### Returns:
- Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9]`

---
## fast_acos
Compute elementwise approximate arc-cosine of a vector

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN

### Declaration:
```
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &c);
void quadriga_lib::fast_acos(const arma::vec &x,  arma::fvec &c);
```

### Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`

### Outputs:
- **`c`** — acos(x); `[n_elem]`

---
## fast_asin
Compute elementwise approximate arc-sine of a vector

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN

### Declaration:
```
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &s);
void quadriga_lib::fast_asin(const arma::vec &x,  arma::fvec &s);
```

### Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`

### Outputs:
- **`s`** — asin(x); `[n_elem]`

---
## fast_atan2
Compute elementwise approximate two-argument arc-tangent of two vectors

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Returns angles in (-pi, pi]; max error ~3 ULP (~3.6e-7)
- atan2(0, 0) returns 0; atan2(±0, -0) returns ±0 (not ±pi)

### Declaration:
```
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &a);
void quadriga_lib::fast_atan2(const arma::vec &y,  const arma::vec &x,  arma::fvec &a);
```

### Inputs:
- **`y`** — Y-coordinates (numerator); `[n_elem]`
- **`x`** — X-coordinates (denominator); `[n_elem]`

### Outputs:
- **`a`** — atan2(y, x); `[n_elem]`

---
## fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Conversion: len = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / len, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- z/len is clamped to [-1, 1] before asin to guard against len == 0 and FMA rounding artefacts pushing abs(z/len) slightly above 1
- All inputs must have the same length
- AVX2 kernel computes internally in single precision (double outputs are cast back from float); GENERIC kernel preserves full `dtype` precision

### Declaration:
```
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el, arma::fvec *len = nullptr, int use_kernel = 0);

void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::vec &az, arma::vec &el, arma::vec *len = nullptr, int use_kernel = 0);
```

### Inputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); default `0`

### Outputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`

### See also:
- [fast_geo2cart](#fast_geo2cart) (inverse conversion)

---
## fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: x = cos(el)*cos(az)*len, y = cos(el)*sin(az)*len, z = sin(el)*len
- Optional pointer outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; pass `nullptr` to skip
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Precision: GENERIC kernel uses full `dtype` precision (double or float).
- AVX2 kernel always computes in single precision internally — for `dtype=double`, inputs are narrowed to 
  float and results widened back. Use `use_kernel=1` to force GENERIC if full double precision is required.

### Declaration:
```
void fast_geo2cart(
    const arma::Col<dtype> &az,
    const arma::Col<dtype> &el,
    arma::Col<dtype> &x,
    arma::Col<dtype> &y,
    arma::Col<dtype> &z,
    arma::Col<dtype> *sAZ = nullptr,
    arma::Col<dtype> *cAZ = nullptr,
    arma::Col<dtype> *sEL = nullptr,
    arma::Col<dtype> *cEL = nullptr,
    const arma::Col<dtype> *len = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable)

### Outputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`sAZ`** *(optional)* — sin(az); `[n_elem]` or `nullptr`
- **`cAZ`** *(optional)* — cos(az); `[n_elem]` or `nullptr`
- **`sEL`** *(optional)* — sin(el); `[n_elem]` or `nullptr`
- **`cEL`** *(optional)* — cos(el); `[n_elem]` or `nullptr`

### See also:
- [fast_cart2geo](#fast_cart2geo) (inverse conversion)

---
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- For x in [-pi, pi]: max absolute error = 2^(-22.1); for x in [-500, 500]: 2^(-16.0)
- Either `s` or `c` may be `nullptr` to skip that computation

### Declaration:
```
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
void quadriga_lib::fast_sincos(const arma::vec &x,  arma::fvec *s = nullptr, arma::fvec *c = nullptr);
```

### Inputs:
- **`x`** — Input angles; `[n_elem]`

### Outputs:
- **`s`** *(optional)* — sin(x); `[n_elem]` or `nullptr`
- **`c`** *(optional)* — cos(x); `[n_elem]` or `nullptr`

---
## fast_slerp
Compute elementwise approximate SLERP interpolation between two complex-valued vectors

- Interpolates phase via SLERP on normalized directions; amplitudes are linearly interpolated
- Weight `w=0` returns A, `w=1` returns B; per-element weights in [0, 1]
- Near-antipodal inputs (phase difference close to pi) fall back to linear interpolation smoothly
- If both input amplitudes are negligible, output is zero
- Max error vs. double-precision reference: ~5 ULP
- AVX2-optimized (8 complex pairs/lane); scalar fallback without AVX2

### Declaration:
```
void quadriga_lib::fast_slerp(const arma::fvec &Ar, const arma::fvec &Ai,
                              const arma::fvec &Br, const arma::fvec &Bi,
                              const arma::fvec &w,
                              arma::fvec &Xr, arma::fvec &Xi);

void quadriga_lib::fast_slerp(const arma::vec &Ar, const arma::vec &Ai,
                              const arma::vec &Br, const arma::vec &Bi,
                              const arma::vec &w,
                              arma::fvec &Xr, arma::fvec &Xi);
```

### Inputs:
- **`Ar`** — Real part of source A; `[n_elem]`
- **`Ai`** — Imaginary part of source A; `[n_elem]`
- **`Br`** — Real part of source B; `[n_elem]`
- **`Bi`** — Imaginary part of source B; `[n_elem]`
- **`w`** — Per-element interpolation weight in [0, 1]; `[n_elem]`

### Outputs:
- **`Xr`** — Real part of interpolated result; `[n_elem]`
- **`Xi`** — Imaginary part of interpolated result; `[n_elem]`

---
## interp_2D
Perform linear interpolation (1D or 2D) on single or multiple data sets

- Interpolates given input data at specified output points.
- Supports single and multiple data sets.
- Returns interpolated results either directly or through reference argument.
- Data types (`dtype`): `float` or `double`

### Declarations:
```
void interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
               const arma::Col<dtype> &xo, const arma::Col<dtype> &yo, arma::Cube<dtype> &output);

arma::Cube<dtype> interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                            const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_2D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                           const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_1D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);

arma::Col<dtype> interp_1D(const arma::Col<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);
```

### Arguments:
- `input`: Input data array/matrix (size details below)
- `xi`: Input x-axis sampling points, vector of length `nx`
- `yi`: Input y-axis sampling points (for 2D only), vector of length `ny`
- `xo`: Output x-axis sampling points, vector of length `mx`
- `yo`: Output y-axis sampling points (for 2D only), vector of length `my`
- `output`: Interpolated data cube (modified in-place for one variant)

### Input / Output size details:
- 2D interpolation of multiple datasets (`arma::Cube`): input: `[ny, nx, ne]`; output: `[my, mx, ne]`
- 2D interpolation of single dataset (`arma::Mat`): input: `[ny, nx]`; output: `[my, mx]`
- 1D interpolation of multiple datasets (`arma::Mat`): input: `[nx, ne]`; output: `[mx, ne]`
- 1D interpolation of single dataset (`arma::Col`): input: `[nx]`, output: `[mx]`

### Examples:
- 2D interpolation example:
```
arma::cube input(5, 5, 2, arma::fill::randu); // example input data
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec yi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);
arma::vec yo = arma::linspace(0, 4, 10);

arma::cube output;
quadriga_lib::interp_2D(input, xi, yi, xo, yo, output);
```
- 1D interpolation example:
```
arma::vec input = arma::linspace(0, 1, 5);
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);

auto output = quadriga_lib::interp_1D(input, xi, xo);
```

---

# Site-specific simulation tools

---
## calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients, generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping triangles whose bounding box does not intersect the TX-RX path

### Declaration:
```
void calc_diffraction_gain(
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *mesh,
    const arma::Mat<dtype> *mtl_prop,
    dtype center_frequency,
    int lod = 2,
    arma::Col<dtype> *gain = nullptr,
    arma::Cube<dtype> *coord = nullptr,
    int verbose = 0,
    const arma::u32_vec *sub_mesh_index = nullptr,
    int use_kernel = 0,
    int gpu_id = 0);
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; see [obj_file_read](#obj_file_read); `[n_mesh, n_param]`
- **`center_frequency`** — Center frequency
- **`lod`** *(optional)* — Level of detail (0–6), controls `n_path` and `n_seg`; see [generate_diffraction_paths](#generate_diffraction_paths)
- **`verbose`** *(optional)* — Verbosity level
- **`sub_mesh_index`** *(optional)* — 0-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation); `[n_mesh]`
- **`use_kernel`** *(optional)* — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable
- **`gpu_id`** *(optional)* — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** *(optional)* — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). Selects
  interaction type passed to [ray_mesh_interact](#ray_mesh_interact) (4 vs. 1).

### Outputs:
- **`gain`** *(optional)* — Diffraction gain per TX-RX pair, linear scale; `[n_pos]`
- **`coord`** *(optional)* — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]`

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)
- [ray_mesh_interact](#ray_mesh_interact) (used for media interactions)

---
## colormap
Generate a colormap matrix with RGB values

- Returns a `[64, 3]` or `[256, 3]` matrix of unsigned char RGB values (range 0–255)
- Available maps: `"jet"`, `"parula"`, `"winter"`, `"hot"`, `"turbo"`, `"copper"`, `"spring"`, `"cool"`, `"gray"`, `"autumn"`, `"summer"`

### Declaration:
```
arma::uchar_mat quadriga_lib::colormap(std::string map, bool high_res = false);
```

### Inputs:
- **`map`** — Name of the colormap
- **`high_res`** *(optional)* — If true, returns 256 rows instead of 64

### Returns:
- RGB colormap matrix; `[64, 3]` or `[256, 3]`

---
## combine_irs_coord
Combine path interaction coordinates for IRS-assisted TX → RX channels

- Merges two propagation segments (TX → IRS and IRS → RX) into complete path interaction coordinate sequences
- Interaction coordinates use a compressed format: `no_interact` counts interactions per path, `interact_coord` stores all coordinates sequentially in path order
- Each combined path appends segment 1 coordinates (optionally reversed) then the IRS position then segment 2 coordinates (optionally reversed); reversing affects coordinate order only, not endpoint positions
- Output contains at most `n_path_1 × n_path_2` paths; `active_path` (typically the return value of [get_channels_irs](#get_channels_irs)) reduces this to active combinations only
- Typically used after [get_channels_irs](#get_channels_irs) to produce interaction data for path visualization (e.g. in Blender) via [coord2path](#coord2path)

### Declaration:
```
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
- **`interact_coord_1`** — Interaction coordinates for segment 1; `[3, sum(no_interact_1)]`
- **`no_interact_2`** — Number of interaction points per path for segment 2 (IRS → RX); `[n_path_2]`
- **`interact_coord_2`** — Interaction coordinates for segment 2; `[3, sum(no_interact_2)]`
- **`reverse_segment_1`** *(optional)* — If `true`, reverses interaction coordinate order within segment 1
- **`reverse_segment_2`** *(optional)* — If `true`, reverses interaction coordinate order within segment 2
- **`active_path`** *(optional)* — Boolean mask selecting path combinations to include; pass the return value of [get_channels_irs](#get_channels_irs) directly; `[n_path_1 × n_path_2]`

### Outputs:
- **`no_interact`** — Number of interaction points per combined path; `[n_path_irs]`
- **`interact_coord`** — Combined interaction coordinates for all output paths; `[3, sum(no_interact)]`

### See also:
- [get_channels_irs](#get_channels_irs) (generates `active_path` and channel coefficients for IRS channels)
- [coord2path](#coord2path) (consumes interaction coordinates to compute angles and path geometry)

---
## coord2path
Convert path interaction coordinates into FBS/LBS positions, path length, and angles

- `no_interact` is a vector of length `n_path` with the number of interactions per path
- `interact_coord` stores all coordinates concatenated in path order, size `[3, sum(no_interact)]`
- LOS paths (`no_interact[i] == 0`) get a virtual FBS/LBS at the midpoint between TX and RX
- Set `reverse_path = true` to swap TX/RX and reverse all interaction sequences

### Declaration:
```
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
## generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by [calc_diffraction_gain](#calc_diffraction_gain): elliptic-arc paths sampling the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so `sum(prod(weights,3),2) = 1`

### Declaration:
```
void generate_diffraction_paths(
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    dtype center_frequency,
    int lod,
    arma::Cube<dtype> *ray_x,
    arma::Cube<dtype> *ray_y,
    arma::Cube<dtype> *ray_z,
    arma::Cube<dtype> *weight);
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

### Outputs:
- **`ray_x`** — x-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`ray_y`** — y-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`ray_z`** — z-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`weight`** — Per-segment weights; `[n_pos, n_path, n_seg]`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (consumes the output of this function)

---
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

- Produces 20 × n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Declaration:
```
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
- **`radius`** — Radius of icosphere in meters
- **`direction_xyz`** (optional) — Output directions in Cartesian (true) or spherical azimuth/elevation (false)

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward from origin with magnitude equal to the inradius of the face; `[n_faces, 3]`
- **`length`** (optional) — Distance from origin to face plane; equals the magnitude of each `center` vector; `[n_faces]`
- **`vert`** (optional) — Vertex offsets from face center [x1,y1,z1,x2,y2,z2,x3,y3,z3]; `[n_faces, 9]`
- **`direction`** (optional) — Edge directions; spherical [az1,el1,az2,el2,az3,el3] or Cartesian [x1,y1,z1,x2,y2,z2,x3,y3,z3] per `direction_xyz` flag; `[n_faces, 6]` or `[n_faces, 9]`

### Returns:
- Number of generated triangular faces (20 × n_div²)

---
## medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

- Computes `g = 10^(-A/10)`, where `A` [dB] is the total attenuation accumulated over a path
  of length `dist` inside the medium. The per-meter loss combines two contributions:
  - Conductivity-based loss from the complex permittivity model of ITU-R P.2040-1: `ε_r = a·(f/fRef)^b`,
    `σ = c·(f/fRef)^d`. These give an gain distance `Δ` and a per-meter power loss `8.686 / Δ` dB/m.
  - Distance absorption of the form `α·(f/fRef)^αB` dB/m, intended to model excess loss not captured
    by `σ` (e.g. foliage, scattering media).
- The penetration-loss columns (`att`, `attB`) of `mtl_prop` are not used — they describe
  thin-slab transmission loss, not propagation through a finite-thickness medium.

### Declaration:
```
dtype quadriga_lib::medium_gain(
        const arma::Mat<dtype> &mtl_prop,
        arma::uword iM,
        dtype dist,
        dtype fGHz);
```

### Inputs:
- **`mtl_prop`** — Material properties; see [obj_file_read](#obj_file_read) for the column layout; `[n_mesh, 9]`
- **`iM`** — Row index selecting the material from `mtl_prop`
- **`dist`** — Path length of the ray inside the medium
- **`center_frequency`** — Center frequency; Hu

### Returns:
- Linear in-medium gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the medium

### See also:
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
## mitsuba_xml_file_write
Write a triangular mesh to a Mitsuba 3 XML scene file

- Converts quadriga-lib mesh data structures to Mitsuba 3 XML format, loadable by NVIDIA Sionna RT for 
  differentiable radio-propagation simulations
- Supports grouping faces into named objects with per-face material assignments
- Optionally maps material names to ITU-defined presets used by Sionna RT
- Creates a subdirectory `<stem>_meshes/` next to the XML file and writes one binary PLY file per object into it; 
  both the XML and the mesh folder must be distributable together
- Objects whose faces reference more than one material are automatically split into sub-objects (one per material) 
  and renamed `<obj_name>_<mtl_name>`; the effective object count in the output may therefore exceed the length of `obj_names`

### Declaration:
```
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
- **`obj_ind`** — 1-based object index per triangle; length must match `obj_names`; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle; length must match `mtl_names`; `[n_mesh]`
- **`obj_names`** — Object names; length must equal `max(obj_ind)`
- **`mtl_names`** — Material names; length must equal `max(mtl_ind)`
- **`bsdf`** *(optional)* — BSDF material parameters per material; ignored by Sionna RT, used only by Mitsuba renderer; see [obj_file_read](#obj_file_read) for field definitions; `[mtl_names.size(), 17]`
- **`map_to_itu_materials`** *(optional)* — If `true`, maps material names to ITU presets recognised by Sionna RT

### See also:
- [obj_file_read](#obj_file_read) (source for mesh data and BSDF field layout)

---
## obj_file_read
Read a Wavefront .obj file and extract geometry and material information

- Parses a triangulated Wavefront `.obj` file; quads and n-gons are not supported
- Materials applied per triangle via `usemtl` tag; unknown/missing materials default to `"vacuum"` (all
  parameters at their defaults: ε_r = 1, σ = 0, all loss and resonance terms disabled)
- Material name matching is case-sensitive
- Default materials follow ITU-R P.2040-3 Table 3 (1–40 GHz; ground materials limited to 1–10 GHz)
- Default material tag syntax: `usemtl itu_concrete` (or `itu_brick`, `itu_wood`, etc.)
- Custom material tag syntax: `usemtl Name::a:b:c:d:att:attB:alpha:alphaB:fRef:m:resF:resQ:resS:coiF:coiQ:coiA`
  - Trailing fields are optional; any omitted field falls back to its default (see the parameter table below)
  - Example (only ε and conductivity): `usemtl Glass::6.31:0:0.0036:1.3394`
- A material row has between 1 and 16 columns. Only column 0 (`a`) is required; every other column may be
  omitted and is then substituted with its default. The columns split into three roles:
  - **Interface reflection** (`a`, `b`, `c`, `d`, `resF`, `resQ`, `resS`) — set the complex permittivity ε, which
    fixes the Fresnel reflection coefficient and therefore the room-side absorption `1 − abs(R)²`.
    Applied once per surface hit, independent of path length.
  - **Interface transmission** (`att`, `attB`, `coiF`, `coiQ`, `coiA`) — a lumped through-surface loss in dB,
    applied once per transmission, independent of path length.
  - **In-medium attenuation** (`c`, `d` via ε, `alpha`, `alphaB`, `m`) — loss accumulated along the path
    traversed inside a body; depends on the in-medium distance.
- Frequency laws (`f` in GHz; `f/fRef` is the relative frequency, but `resF` and `coiF` are absolute GHz):
  | Parameter  | Formula                                                      | Unit   | Meaning                                |
  | ---------- | ------------------------------------------------------------ | ------ | -------------------------------------- |
  | ε(f)       | `a·(f/fRef)^b + resS·resF² / (resF² − f² − i·(resF/resQ)·f)` | —      | relative permittivity (complex)        |
  | σ(f)       | `c·(f/fRef)^d`                                               | [S/m]  | conductivity                           |
  | att(f)     | `att·(f/fRef)^attB + coiA / (1 + (coiQ·(f − coiF)/coiF)²)`   | [dB]   | per-interface transmission loss        |
  | α(f)       | `alpha·(f/fRef)^alphaB`                                      | [dB/m] | in-medium loss × in-medium path length |
  | mass(f, L) | `max(0, m·log10((f/fRef)·L))`                                | [dB]   | in-medium, L = path length in metres   |
- **Permittivity resonance** (`resF`, `resQ`, `resS`): a Lorentz pole that adds a peak to absorption (acoustic α) and a feature to reflection near `resF`; `resQ` sets sharpness (higher = narrower). Active only when `resF > 0` and `resS ≠ 0`. Models resonant dielectrics / frequency-selective media (EM) and Helmholtz / membrane absorbers (acoustic).
- **Coincidence term** (`coiF`, `coiQ`, `coiA`): a Lorentzian added to the transmission loss at `coiF`. Negative `coiA` produces a transmission dip (acoustic coincidence / pass-band); positive `coiA` produces a stop-band. Total loss is clamped to ≥ 0. Active only when `coiF > 0` and `coiA ≠ 0`.
- **Mass-law term** (`m`): a transmission loss that is logarithmic in both frequency and in-medium path length. `m = 20` reproduces the acoustic mass law (+6 dB/octave and +6 dB per thickness doubling). Default 0 (EM through-loss is the linear `alpha` term). The imaginary sign of the ε resonance follows the library's loss convention (consistent with σ).

### Declaration:
```
arma::uword quadriga_lib::obj_file_read(
    std::string fn,
    arma::Mat<dtype> *mesh = nullptr,
    arma::Mat<dtype> *mtl_prop = nullptr,
    arma::Mat<dtype> *vert_list = nullptr,
    arma::umat *face_ind = nullptr,
    arma::uvec *obj_ind = nullptr,
    arma::uvec *mtl_ind = nullptr,
    std::vector<std::string> *obj_names = nullptr,
    std::vector<std::string> *mtl_names = nullptr,
    arma::Mat<dtype> *bsdf = nullptr,
    const std::string &materials_csv = "",
    bool trim = true);
```

### Inputs:
- **`fn`** — Path to the `.obj` file
- **`materials_csv`** — Path to CSV file with custom material properties.
  Required column: `a`, plus `name`. Optional columns (any order, any subset):
  `b`, `c`, `d`, `att`, `attB`, `alpha`, `alphaB`, `fRef`, `m`, `resF`, `resQ`, `resS`, `coiF`, `coiQ`, `coiA`.
  Missing optional columns and empty cells fall back to the per-column defaults below (`a` → 1, `fRef` → 1, everything else → 0).
  If empty, ITU-R P.2040-3 defaults are used.
- **`trim`** — If `true`, remove non-default columns from `mtl_prop`

### Outputs:
- **`mesh`** — Triangle vertex coordinates as `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per row; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; `[n_mesh, n_cols]` with `1 ≤ n_cols ≤ 16`. The width is the minimum that captures all non-default parameters in the scene; consumers substitute defaults for absent columns. Columns:
  | Index | Symbol | Property                             | Units    | Default |
  | :---: | :----: | ------------------------------------ | :------: | :-----: |
  | 0     | a      | ε_r at fRef                          | —        | 1.0     |
  | 1     | b      | Frequency exponent for ε_r           | —        | 0       |
  | 2     | c      | σ at fRef                            | S/m      | 0       |
  | 3     | d      | Frequency exponent for σ             | —        | 0       |
  | 4     | att    | Penetration loss at fRef             | dB       | 0       |
  | 5     | attB   | Frequency exponent for att           | —        | 0       |
  | 6     | alpha  | In-medium absorption at fRef         | dB/m     | 0       |
  | 7     | alphaB | Frequency exponent for alpha         | —        | 0       |
  | 8     | fRef   | Reference frequency                  | GHz      | 1.0     |
  | 9     | m      | Mass-law transmission slope          | dB/decade| 0       |
  | 10    | resF   | Permittivity resonance frequency     | GHz      | 0       |
  | 11    | resQ   | Permittivity resonance quality factor| —        | 0       |
  | 12    | resS   | Permittivity resonance strength      | —        | 0       |
  | 13    | coiF   | Coincidence frequency                | GHz      | 0       |
  | 14    | coiQ   | Coincidence quality factor           | —        | 0       |
  | 15    | coiA   | Coincidence loss amplitude           | dB       | 0       |
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 0-based indices into `vert_list` per triangle; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle; `[n_mesh]`
- **`obj_names`** — Object names; length = `max(obj_ind)`
- **`mtl_names`** — Material names; length = `max(mtl_ind)`
- **`bsdf`** — Principled BSDF values from the `.mtl` file; `[n_mtl, 17]` (unchanged; see below)

### Returns:
- Number of triangular mesh elements (`n_mesh`)

### Default material table:
- All built-in materials use only columns 0–4 (`a`, `b`, `c`, `d`, `att`); `attB = alpha = alphaB = 0`, `fRef = 1 GHz`, and all extended parameters (`m`, `resF`, `resQ`, `resS`, `coiF`, `coiQ`, `coiA`) = 0. A scene using only built-in materials therefore yields a 5-column `mtl_prop` (4 columns if no material sets `att`; only `irr_glass` does).
  | Name                  | a     | b      | c       | d      | att  | max fGHz |
  | --------------------- | ----: | -----: | ------: | -----: | ---: | -------: |
  | vacuum / air          | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100      |
  | textiles              | 1.5   | 0.0    | 5e-5    | 0.62   | 0.0  | 100      |
  | plastic               | 2.44  | 0.0    | 2.33e-5 | 1.0    | 0.0  | 100      |
  | ceramic               | 6.5   | 0.0    | 0.0023  | 1.32   | 0.0  | 100      |
  | sea_water             | 80.0  | -0.25  | 4.0     | 0.58   | 0.0  | 100      |
  | sea_ice               | 3.2   | -0.022 | 1.1     | 1.5    | 0.0  | 100      |
  | water                 | 80.0  | -0.18  | 0.6     | 1.52   | 0.0  | 20       |
  | water_ice             | 3.17  | -0.005 | 5.6e-5  | 1.7    | 0.0  | 20       |
  | itu_concrete          | 5.24  | 0.0    | 0.0462  | 0.7822 | 0.0  | 100      |
  | itu_brick             | 3.91  | 0.0    | 0.0238  | 0.16   | 0.0  | 40       |
  | itu_plasterboard      | 2.73  | 0.0    | 0.0085  | 0.9395 | 0.0  | 100      |
  | itu_wood              | 1.99  | 0.0    | 0.0047  | 1.0718 | 0.0  | 100      |
  | itu_glass             | 6.31  | 0.0    | 0.0036  | 1.3394 | 0.0  | 100      |
  | itu_ceiling_board     | 1.48  | 0.0    | 0.0011  | 1.075  | 0.0  | 100      |
  | itu_chipboard         | 2.58  | 0.0    | 0.0217  | 0.78   | 0.0  | 100      |
  | itu_plywood           | 2.71  | 0.0    | 0.33    | 0.0    | 0.0  | 40       |
  | itu_marble            | 7.074 | 0.0    | 0.0055  | 0.9262 | 0.0  | 60       |
  | itu_floorboard        | 3.66  | 0.0    | 0.0044  | 1.3515 | 0.0  | 100      |
  | itu_metal             | 1.0   | 0.0    | 1.0e7   | 0.0    | 0.0  | 100      |
  | itu_very_dry_ground   | 3.0   | 0.0    | 0.00015 | 2.52   | 0.0  | 10       |
  | itu_medium_dry_ground | 15.0  | -0.1   | 0.035   | 1.63   | 0.0  | 10       |
  | itu_wet_ground        | 30.0  | -0.4   | 0.15    | 1.3    | 0.0  | 10       |
  | itu_vegetation        | 1.0   | 0.0    | 1.0e-4  | 1.1    | 0.0  | 100      |
  | irr_glass             | 6.27  | 0.0    | 0.0043  | 1.1925 | 23.0 | 100      |

### See also:
- [obj_file_write](#obj_file_write) (for writing OBJ files)
- [obj_overlap_test](#obj_overlap_test) (for testing mesh geometry)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (used to calculate indexed mesh for faster processing)
- [ray_mesh_interact](#ray_mesh_interact) (calculating interactions between rays and the triangular mesh)
- [mitsuba_xml_file_write](#mitsuba_xml_file_write) (for exporting to Mitsuba scene file format)

<a target="_blank" rel="noopener noreferrer" href=""></a>

### Background and references:
Base EM model (columns 0–8, reflection and σ-loss):
- <a target="_blank" rel="noopener noreferrer" href="https://www.itu.int/rec/R-REC-P.2040">ITU-R P.2040</a> (recommendation
  defining the (a, b, c, d) permittivity/conductivity model and the Fresnel coefficients)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Fresnel_equations">Fresnel equations</a> (interface reflection/transmission from ε)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Relative_permittivity">Relative permittivity (a, b)</a>
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity">Electrical resistivity and conductivity (c, d)</a>
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Dielectric_loss">Dielectric loss / loss tangent</a> (in-medium σ loss)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Attenuation">Attenuation</a> (alpha, linear in path length)

Acoustic mechanism mapping (the analogy the parameters approximate):
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Acoustic_impedance">Acoustic impedance</a> (basis for deriving `a` from an impedance ratio)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Acoustic_transmission">Acoustic transmission</a> (mass-law transmission loss (m) and the coincidence effect)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Soundproofing">Soundproofing</a> (mass law, coincidence, partition behaviour)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Sound_transmission_class">Sound transmission class</a> (single-number TL rating context)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Absorption_(acoustics)">Absorption (acoustics)</a> (porous absorption: alpha, alphaB)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Helmholtz_resonance">Helmholtz resonance</a> (resonant absorbers (resF/resQ/resS, coiF/coiQ/coiA))

---
## obj_file_write
Write a triangulated Wavefront .obj (and .mtl) file

- Supply geometry as either `mesh`, or as `vert_list` + `face_ind`; giving both, or neither, is an error
- With `mesh`: `vert_list_out` + `face_ind_out` are derived from it, merging vertices of the same object that
  are closer than `threshold` (no merging across objects). With `vert_list`/`face_ind`: data is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind`/`obj_names`: a single object named `object` is written
- Without `mtl_ind` (or if all entries are `0`): no `usemtl` tags and no `.mtl` file are written;
  `mtl_ind = 0` marks an individual face as unassigned
- The `.mtl` (named after the `.obj`) lists each used material; values default to a gray material when `bsdf` is omitted

### Declaration:
```
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
    const dtype threshold = 0.001);
```

### Inputs:
- **`fn`** — Output path; must end in `.obj`; if empty, no files are written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{x1,y1,z1,...,x3,y3,z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list`/`face_ind`
- **`obj_ind`** — 1-based object index per face; `[n_mesh]`; each object must be a contiguous block
- **`mtl_ind`** — 1-based material index per face (`0` = none); `[n_mesh]`
- **`obj_names`** — Object names; length ≥ `max(obj_ind)`; required if `obj_ind` is given
- **`mtl_names`** — Material names; length ≥ `max(mtl_ind)`; required if `mtl_ind` has nonzero entries
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only with `face_ind`, written unchanged
- **`face_ind`** — 0-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF for the `.mtl`; `[n_mtl, 17]`; see [obj_file_read](#obj_file_read) for columns
- **`threshold`** — Vertex co-location distance for merging within an object; default 1 mm

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)
- [mitsuba_xml_file_write](#mitsuba_xml_file_write) (for exporting to Mitsuba scene file format)

---
## obj_overlap_test
Detect overlapping 3D objects in a triangular mesh

- Returns 1-based indices of all objects that intersect at least one other object
- Touching faces or edges are not considered overlapping
- Checks for intersecting triangle faces and vertices/edges penetrating another object's bounding volume
- Overlaps smaller than `tolerance` are ignored to account for numerical imprecision
- Does not modify or repair the mesh

### Declaration:
```
arma::uvec quadriga_lib::obj_overlap_test(
    const arma::Mat<dtype> *mesh,
    const arma::uvec *obj_ind,
    std::vector<std::string> *reason = nullptr,
    dtype tolerance = 0.0005);
```

### Inputs:
- **`mesh`** — Triangular mesh; each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** — 1-based object index mapping triangles to objects; output of [obj_file_read](#obj_file_read); `[n_mesh]`
- **`reason`** *(optional)* — Human-readable overlap descriptions per overlapping object; `[n_overlap]`
- **`tolerance`** *(optional)* — Geometric tolerance; intersections smaller than this are ignored

### Returns:
- `arma::uvec`: Unique 1-based object indices of all overlapping objects; `[n_overlap]`

### See also:
- [obj_file_read](#obj_file_read) (reads mesh data from files and generates `obj_ind` input)

---
## path_to_tube
Convert a 3D path into a tube surface mesh for visualization

- Converts an ordered sequence of 3D points into a tubular quad mesh with circular cross-sections
- At bends steeper than 10°, the tube is split and an extra vertex ring is inserted to avoid intersection
- Cross-section orientation uses continuous frame alignment between segments to minimize twisting
- Output `faces` indices are directly usable in `.obj` or `.ply` export

### Declaration:
```
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
- **`vert`** — Tube vertex positions; `[3, (n_coord + n_split) × n_edges]` where `n_split` is the number of bends > 10°
- **`faces`** — Quad face indices into `vert`, 4 indices per quad; `[4, (n_coord - 1) × n_edges]`

---
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `[x_min, x_max, y_min, y_max, z_min, z_max]` for one sub-cloud
- If `sub_cloud_index` is `nullptr` or empty, the entire input is treated as a single cloud; last index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Declaration:
```
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
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint.
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center.
- Produces a reorganized point array and index maps to track reordering.

### Declaration:
```
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
- **`vec_size`** *(optional)* — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`

### Outputs:
- **`pointsR`** — Reorganized point cloud with points grouped by sub-cloud; `[n_pointsR, 3]`
- **`sub_cloud_index`** — 0-based starting index of each sub-cloud within `pointsR`; `[n_sub]`
- **`forward_index`** *(optional)* — 1-based index map from `points` to `pointsR`; padding entries are `0`; `[n_pointsR]`
- **`reverse_index`** *(optional)* — 0-based index map from `pointsR` back to `points`; `[n_points]`

### Returns:
- Number of generated sub-clouds `n_sub`

### See also:
- [point_cloud_aabb](#point_cloud_aabb) (bounding box computation)
- [point_cloud_split](#point_cloud_split) (related spatial splitting)
- [ray_point_intersect](#ray_point_intersect) (downstream use case)

---
## point_cloud_split
Split a point cloud into two sub-clouds along a spatial axis

- Splits at the bounding box midpoint along the chosen axis (not the statistical median); 
  the split may be unbalanced if points are non-uniformly distributed.
- If `axis == 0`, the longest bounding box extent is used.
- Returns a negative axis value if the split failed (all points on one side); outputs are not modified in that case.

### Declaration:
```
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
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided, returns the 1-based enclosing object index instead of binary 0/1

### Declaration:
```
arma::u32_vec quadriga_lib::point_inside_mesh(
    const arma::Mat<dtype> *points,
    const arma::Mat<dtype> *mesh,
    const arma::u32_vec *obj_ind = nullptr,
    dtype distance = 0.0);
```

### Inputs:
- **`points`** — 3D coordinates of test points; `[n_points, 3]`
- **`mesh`** — Triangle faces in row-major vertex format  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** *(optional)* — 1-based object index per mesh element; enables per-object output; `[n_mesh]`
- **`distance`** *(optional)* — Surface proximity threshold; points within this distance
  of the mesh surface are classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1);
  range: 0–20 m (default: 0)

### Returns:
- `arma::u32_vec`, size `[n_points]`; `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object index (with `obj_ind`)

### See also:
- [obj_file_read](#obj_file_read) (for reading `mesh` and `obj_ind` from an .obj file)

---
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media.
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`).
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule);
  front-side hit with FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air;
  FBS=SBS with opposing normals → media-to-media.
- Rays with `fbs_ind = 0` (no interaction) are omitted from output, so `n_rayN ≤ n_ray`.
- Output direction encoding (spherical/Cartesian) matches input `tridir` format.
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves).
- Types 3–4 (scalar) use TE-only reflection with no total internal reflection, suitable for acoustic
  simulation with impedance-mapped material parameters (ε derived from Z).

### Declaration:
```
void quadriga_lib::ray_mesh_interact(
    int interaction_type, dtype center_frequency,
    const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbs, const arma::Mat<dtype> *sbs,
    const arma::Mat<dtype> *mesh, const arma::Mat<dtype> *mtl_prop,
    const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
    const arma::Mat<dtype> *trivec = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    const arma::Col<dtype> *orig_length = nullptr,
    arma::Mat<dtype> *origN = nullptr, arma::Mat<dtype> *destN = nullptr,
    arma::Col<dtype> *gainN = nullptr, arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr, arma::Mat<dtype> *tridirN = nullptr,
    arma::Col<dtype> *orig_lengthN = nullptr,
    arma::Col<dtype> *fbs_angleN = nullptr,
    arma::Col<dtype> *thicknessN = nullptr,
    arma::Col<dtype> *edge_lengthN = nullptr,
    arma::Mat<dtype> *normal_vecN = nullptr,
    arma::s32_vec *out_typeN = nullptr);
```

### Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
- **`center_frequency`** — Center frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`fbs`**, **`sbs`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; ee [obj_file_read](#obj_file_read); `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; see [obj_file_read](#obj_file_read); `[n_mesh, n_param]`
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`orig_length`** *(optional)* — Accumulated path length at origin; `[n_ray]`, default 0

### Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale, includes in-medium attenuation, excludes FSPL); averaged over TE/TM polarizations for types 0–2, TE-only for types 3–4; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex `[ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH]`; includes interaction gain, TE/TM coefficients, incidence plane orientation, in-medium attenuation (excludes FSPL); `[n_rayN, 8]`. For types 3–4 (scalar): `[Re Im 0 0 0 0 0 0]` where Re+jIm is the scalar pressure coefficient including in-medium attenuation; `[n_rayN, 8]`.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`orig_lengthN`** — Path length from `orig` to `origN`, added to input `orig_length` if given; `[n_rayN]`
- **`fbs_angleN`** — Incidence angle at FBS in rad; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code; `[n_rayN]`
   | Code  | Description                                         |
   | :---: | --------------------------------------------------- |
   |   1   | Single hit, outside→inside                          |
   |   2   | Single hit, inside→outside                          |
   |   3   | Single hit, inside→outside, total reflection        |
   |   4   | Media-to-media, M2 hit first                        |
   |   5   | Media-to-media, M1 hit first                        |
   |   6   | Media-to-media, M1 hit first, total reflection      |
   |   7   | Overlapping faces, outside→inside                   |
   |   8   | Overlapping faces, inside→outside                   |
   |   9   | Overlapping faces, inside→outside, total reflection |
   |  10   | Edge hit, outside→inside→outside                    |
   |  11   | Edge hit, inside→outside→inside                     |
   |  12   | Edge hit, inside→outside→inside, total reflection   |
   |  13   | Edge hit, outside→inside                            |
   |  14   | Edge hit, inside→outside                            |
   |  15   | Edge hit, inside→outside, total reflection          |

### See also:
- [obj_file_read](#obj_file_read) (for loading `mesh` and `mtl_prop` from OBJ file)
- [icosphere](#icosphere) (for generating beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for computing FBS and SBS positions)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)

---
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin, enabling energy spread simulation.
- Returns, for each point, the list of 0-based ray indices whose beam intersects that point.
- All internal computations use single precision.

### Declaration:
```
std::vector<arma::u32_vec> quadriga_lib::ray_point_intersect(
    const arma::Mat<dtype> *points,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *trivec,
    const arma::Mat<dtype> *tridir,
    const arma::u32_vec *sub_cloud_index = nullptr,
    arma::u32_vec *hit_count = nullptr,
    int use_kernel = 0,
    int gpu_id = 0);
```

### Inputs:
- **`points`** — 3D point cloud coordinates; `[n_points, 3]`
- **`orig`** — Ray origin positions in global Cartesian coordinates; `[n_ray, 3]`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order `[v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]`; `[n_ray, 9]`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates (need not be normalized), order `[d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z]`; `[n_ray, 9]`
- **`sub_cloud_index`** *(optional)* — Segment boundary indices for the point cloud (see [point_cloud_segmentation](#point_cloud_segmentation)); `[n_sub]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_points >= 500` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

### Optional output:
- **`hit_count`** — Number of rays intersecting each point; `[n_points]`

### Returns:
- `std::vector<arma::u32_vec>` — Per-point list of 0-based ray indices that intersected that point; length `n_points`

### See also:
- [icosphere](#icosphere) (generate ray beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud index)
- [subdivide_rays](#subdivide_rays) (subdivide beams into sub-beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (ray–triangle intersection)
- [ray_mesh_interact](#ray_mesh_interact) (beam–mesh interaction)

---
## ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- Internal computations always use single precision for AVX2 and CUDA kernels; only GENERIC has `double` support

### Declaration:
```
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
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; `[n_sub]`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; if `nullptr`, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

### Outputs:
- **`fbs`** (optional) — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** (optional) — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`no_interact`** (optional) — Total number of intersections per ray between `orig` and `dest`; `[n_ray]`
- **`fbs_ind`** (optional) — 1-based index of first intersected mesh element; 0 = none; `[n_ray]`
- **`sbs_ind`** (optional) — 1-based index of second intersected mesh element; 0 = none; `[n_ray]`

### See also:
- [obj_file_read](#obj_file_read) (load mesh from OBJ file)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (compute sub-mesh indices and AABBs)
- [ray_point_intersect](#ray_point_intersect) (beam interactions with sampling points)
- [icosphere](#icosphere) (generate ray beams)
- [subdivide_rays](#subdivide_rays) (split ray beams into sub-beams)

---
## subdivide_rays
Subdivide ray beams into four smaller sub-beams

- Each triangular beam is split into 4 sub-beams; output size is `4 × n_ray` or `4 × n_ind` when `index` is provided
- `tridir` format auto-detected: spherical `[n_ray, 6]` or Cartesian `[n_ray, 9]`; output matches input format

### Declaration:
```
arma::uword quadriga_lib::subdivide_rays(
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *trivec,
    const arma::Mat<dtype> *tridir,
    const arma::Mat<dtype> *dest = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    const arma::u32_vec *index = nullptr,
    const double ray_offset = 0.0);
```

### Inputs:
- **`orig`** — Ray origin points in GCS; `[n_ray, 3]`
- **`trivec`** — Vectors from origin to triangle vertices, columns `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, spherical `[v1az v1el v2az v2el v3az v3el]` or Cartesian `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 6]` or `[n_ray, 9]`
- **`dest`** (optional) — Ray destination points; `[n_ray, 3]`
- **`index`** (optional) — 0-based indices of rays to subdivide; `[n_ind]`
- **`ray_offset`** (optional) — Origin offset along propagation direction

### Outputs:
- **`origN`** — Subdivided ray origins; `[n_rayN, 3]`
- **`trivecN`** — Subdivided triangle vectors; `[n_rayN, 9]`
- **`tridirN`** — Subdivided vertex-ray directions, same format as `tridir`; `[n_rayN, 6]` or `[n_rayN, 9]`
- **`destN`** — Subdivided destinations, empty if `dest` was `nullptr` or empty; `[n_rayN, 3]`

### Returns:
- `n_rayN` — Number of output rays

### See also:
- [icosphere](#icosphere) (generate initial beams)
- [ray_point_intersect](#ray_point_intersect) (beam–sample-point interaction)
- [ray_triangle_intersect](#ray_triangle_intersect) (beam–triangle interaction)

---
## subdivide_triangles
Subdivide triangles into smaller triangles

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in x n_div x n_div`
- Material properties are duplicated from parent triangle to all sub-triangles

### Declaration:
```
arma::uword quadriga_lib::subdivide_triangles(
    arma::uword n_div,
    const arma::Mat<dtype> *triangles_in,
    arma::Mat<dtype> *triangles_out,
    const arma::Mat<dtype> *mtl_prop = nullptr,
    arma::Mat<dtype> *mtl_prop_out = nullptr);
```

### Inputs:
- **`n_div`** — Number of subdivisions per edge
- **`triangles_in`** — Mesh vertices as `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_triangles_in, 9]`
- **`mtl_prop`** *(optional)* — Material properties; see [obj_file_read](#obj_file_read); `[n_triangles_in, n_param]`

### Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_prop_out`** *(optional)* — Material properties for subdivided triangles; `[n_triangles_out, n_param]`

### Returns:
- `n_triangles_out` — Number of generated triangles

---
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Declaration:
```
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(
    const arma::Mat<dtype> *mesh,
    const arma::u32_vec *sub_mesh_index = nullptr,
    arma::uword vec_size = 1);
```

### Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; `[n_triangles, 9]`
- **`sub_mesh_index`** *(optional)* — 0-based start indices of sub-meshes; if omitted, the AABB of the entire mesh is returned; `[n_sub]`
- **`vec_size`** *(optional)* — Alignment size for SIMD/CUDA padding (e.g., `8` for AVX2, `32` for CUDA)

### Returns:
- `arma::Mat<dtype>` of shape `[n_sub_aligned, 6]`, one AABB per sub-mesh row

### See also:
- [ray_triangle_intersect](#ray_triangle_intersect) (consumer of the output)

---
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark padding entries
- If `mtl_prop` is provided, material rows are reordered and padded in the same way

### Declaration:
```
arma::uword triangle_mesh_segmentation(
    const arma::Mat<dtype> *mesh,
    arma::Mat<dtype> *meshR,
    arma::u32_vec *sub_mesh_index,
    arma::uword target_size = 1024,
    arma::uword vec_size = 1,
    const arma::Mat<dtype> *mtl_prop = nullptr,
    arma::Mat<dtype> *mtl_propR = nullptr,
    arma::u32_vec *mesh_index = nullptr);
```

### Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`target_size`** *(optional)* — Target triangle count per sub-mesh; for best performance set near `sqrt(n_mesh)`
- **`vec_size`** *(optional)* — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count rounded up to a multiple of this value
- **`mtl_prop`** *(optional)* — Material properties; see [obj_file_read](#obj_file_read); `[n_mesh, n_param]`

### Outputs:
- **`meshR`** — Reordered and padded triangle vertices; `[n_meshR, 9]`
- **`sub_mesh_index`** — 0-based start indices of sub-meshes in `meshR`; `[n_sub]`
- **`mtl_propR`** *(optional)* — Reordered and padded material properties; `[n_meshR, n_param]`
- **`mesh_index`** *(optional)* — 1-based mapping from original to reorganized mesh (0 = padding); `[n_meshR]`

### Returns:
- Number of created sub-meshes `n_sub`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (uses `sub_mesh_index` for acceleration)
- [obj_file_read](#obj_file_read) (defines `mtl_prop` format)

---
## triangle_mesh_split
Split a 3D triangular mesh into two sub-meshes along a given axis

- Splits at the bounding box center of the selected axis; triangles where all vertices lie within the 
  lower half go to `meshA`; any triangle with at least one vertex exceeding the threshold goes to `meshB`
- `axis = 0` selects the axis with the longest bounding box extent automatically
- On failure (all triangles fall to one side), `meshA` and `meshB` are left unchanged and the return value is negative
- Used internally by [triangle_mesh_segmentation](#triangle_mesh_segmentation)

### Declaration:
```
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
## write_png
Write a data matrix to a color-coded PNG file

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NAN`
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

### Declaration:
```
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

