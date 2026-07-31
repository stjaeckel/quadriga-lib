---
title: "C++ API Documentation for Quadriga-Lib v0.12.0"
author: "Stephan Jaeckel"
date: "31.07.2026"
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
| [arrayant](#arrayant) | Array antenna class | 145 |
| [.append](#append) | Array antenna class | 196 |
| [.calc_beamwidth_deg](#calc_beamwidth_deg) | Array antenna class | 218 |
| [.calc_directivity_dBi](#calc_directivity_dbi) | Array antenna class | 250 |
| [.combine_pattern](#combine_pattern) | Array antenna class | 271 |
| [.copy_element](#copy_element) | Array antenna class | 296 |
| [.export_obj_file](#export_obj_file) | Array antenna class | 316 |
| [.interpolate](#interpolate) | Array antenna class | 346 |
| [.is_valid](#is_valid) | Array antenna class | 398 |
| [.qdant_write](#qdant_write) | Array antenna class | 416 |
| [.remove_zeros](#remove_zeros) | Array antenna class | 443 |
| [.rotate_pattern](#rotate_pattern) | Array antenna class | 459 |
| [.set_size](#set_size) | Array antenna class | 495 |
| [arrayant_combine_pattern_multi](#arrayant_combine_pattern_multi) | Array antenna functions | 521 |
| [arrayant_concat_multi](#arrayant_concat_multi) | Array antenna functions | 555 |
| [arrayant_copy_element_multi](#arrayant_copy_element_multi) | Array antenna functions | 589 |
| [arrayant_interpolate_multi](#arrayant_interpolate_multi) | Array antenna functions | 629 |
| [arrayant_is_valid_multi](#arrayant_is_valid_multi) | Array antenna functions | 688 |
| [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) | Array antenna functions | 715 |
| [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) | Array antenna functions | 747 |
| [generate_arrayant_3GPP](#generate_arrayant_3gpp) | Array antenna functions | 773 |
| [generate_arrayant_custom](#generate_arrayant_custom) | Array antenna functions | 823 |
| [generate_arrayant_dipole](#generate_arrayant_dipole) | Array antenna functions | 848 |
| [generate_arrayant_half_wave_dipole](#generate_arrayant_half_wave_dipole) | Array antenna functions | 865 |
| [generate_arrayant_multibeam](#generate_arrayant_multibeam) | Array antenna functions | 882 |
| [generate_arrayant_omni](#generate_arrayant_omni) | Array antenna functions | 936 |
| [generate_arrayant_ula](#generate_arrayant_ula) | Array antenna functions | 953 |
| [generate_arrayant_xpol](#generate_arrayant_xpol) | Array antenna functions | 981 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 998 |
| [qdant_read](#qdant_read) | Array antenna functions | 1073 |
| [qdant_read_multi](#qdant_read_multi) | Array antenna functions | 1099 |
| [qdant_write_multi](#qdant_write_multi) | Array antenna functions | 1126 |
| [channel](#channel) | Channel class | 1155 |
| [.add_paths](#add_paths) | Channel class | 1202 |
| [.calc_effective_path_gain](#calc_effective_path_gain) | Channel class | 1240 |
| [.write_paths_to_obj_file](#write_paths_to_obj_file) | Channel class | 1259 |
| [any_type_id](#any_type_id) | Channel functions | 1300 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 1343 |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Channel functions | 1386 |
| [baseband_freq_response_vec](#baseband_freq_response_vec) | Channel functions | 1434 |
| [get_HDF5_version](#get_hdf5_version) | Channel functions | 1472 |
| [hdf5_create](#hdf5_create) | Channel functions | 1484 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 1514 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 1546 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 1582 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 1616 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 1642 |
| [hdf5_write](#hdf5_write) | Channel functions | 1670 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1710 |
| [path](#path) | Channel functions | 1749 |
| [qrt_file_append](#qrt_file_append) | Channel functions | 1870 |
| [qrt_file_init](#qrt_file_init) | Channel functions | 1906 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 1944 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 1992 |
| [qrt_file_read_raw](#qrt_file_read_raw) | Channel functions | 2089 |
| [qrt_read_cache_init](#qrt_read_cache_init) | Channel functions | 2126 |
| [quantize_delays](#quantize_delays) | Channel functions | 2165 |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generation functions | 2227 |
| [get_channels_irs](#get_channels_irs) | Channel generation functions | 2308 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 2402 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 2470 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 2538 |
| [acdf](#acdf) | Channel statistics | 2614 |
| [calc_angular_spreads_sphere](#calc_angular_spreads_sphere) | Channel statistics | 2649 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 2691 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 2743 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 2776 |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | 2811 |
| [fast_acos](#fast_acos) | Math functions | 2844 |
| [fast_asin](#fast_asin) | Math functions | 2863 |
| [fast_atan2](#fast_atan2) | Math functions | 2882 |
| [fast_cart2geo](#fast_cart2geo) | Math functions | 2903 |
| [fast_geo2cart](#fast_geo2cart) | Math functions | 2936 |
| [fast_sincos](#fast_sincos) | Math functions | 2981 |
| [fast_slerp](#fast_slerp) | Math functions | 3002 |
| [interp_2D](#interp_2d) | Math functions | 3037 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 3101 |
| [colormap](#colormap) | Site-specific simulation tools | 3159 |
| [combine_irs_coord](#combine_irs_coord) | Site-specific simulation tools | 3178 |
| [coord2path](#coord2path) | Site-specific simulation tools | 3221 |
| [cube](#cube) | Site-specific simulation tools | 3259 |
| [generate_diffraction_paths](#generate_diffraction_paths) | Site-specific simulation tools | 3292 |
| [icosphere](#icosphere) | Site-specific simulation tools | 3336 |
| [interface_gain](#interface_gain) | Site-specific simulation tools | 3370 |
| [medium_gain](#medium_gain) | Site-specific simulation tools | 3406 |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Site-specific simulation tools | 3443 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 3485 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 3552 |
| [obj_overlap_test](#obj_overlap_test) | Site-specific simulation tools | 3613 |
| [path_to_tube](#path_to_tube) | Site-specific simulation tools | 3644 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 3672 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 3701 |
| [point_cloud_split](#point_cloud_split) | Site-specific simulation tools | 3740 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 3776 |
| [ray_init](#ray_init) | Site-specific simulation tools | 3812 |
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | 3886 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 4008 |
| [ray_progress](#ray_progress) | Site-specific simulation tools | 4056 |
| [ray_state_update](#ray_state_update) | Site-specific simulation tools | 4194 |
| [ray_subdivide_flag](#ray_subdivide_flag) | Site-specific simulation tools | 4327 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 4388 |
| [refractive_index](#refractive_index) | Site-specific simulation tools | 4436 |
| [subdivide_rays](#subdivide_rays) | Site-specific simulation tools | 4468 |
| [subdivide_triangles](#subdivide_triangles) | Site-specific simulation tools | 4526 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 4556 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 4584 |
| [triangle_mesh_split](#triangle_mesh_split) | Site-specific simulation tools | 4625 |
| [write_png](#write_png) | Site-specific simulation tools | 4660 |
| [xpr_update](#xpr_update) | Site-specific simulation tools | 4686 |

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
## path
Class for storing and managing a single propagation path with a compact fixed-size header

- Represents one ray from origin to destination through a sequence of `nSEG` interaction points
- A 64-byte header holds metadata, the first 6 interaction type codes, and the frequency-0
  transfer coefficients; a variable-length heap buffer holds coordinates, the remaining
  frequency coefficients, and the overflow interaction codes
- Two layout modes selected at initialization: EM carries a full 2x2 Jones matrix per frequency;
  SCALAR carries a single complex pressure coefficient per frequency
- Copyable and movable; the moved-from object is left in the valid empty state (`nSEG == 0`, `nFRQ == 1`, EM)

### Attributes:
| Attribute            | Size   | Description                                                       |
| -------------------- | ------ | ----------------------------------------------------------------- |
| `unsigned iC`        | scalar | Channel ID: the channel to which the path belongs                 |
| `unsigned iR`        | scalar | Ray index: relative index in the launch configuration             |
| `uint8_t nREF`       | scalar | Number of reflections (interaction type codes 128-255)            |
| `uint8_t nTRA`       | scalar | Number of transmissions / refractions (type codes 1-127)          |
| `uint8_t nSUB`       | scalar | Number of subdivisions                                            |
| `uint8_t nSCT`       | scalar | Number of scattering events                                       |
| `float length`       | scalar | Accumulated path length, origin through the last interaction (m)  |

### Data buffer layout (EM mode):
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Jones         | `8 * (nFRQ - 1)`  | Transfer matrices for frequencies 1..nFRQ-1, col-major  |
| Interactions  | `(nSEG - 3) / 4`  | Overflow type codes past the first 6, packed 4 per float, present for `nSEG >= 7` |

### Data buffer layout (SCALAR mode):
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Coeff         | `2 * (nFRQ - 4)`  | Pressure coefficients for frequencies 4..nFRQ-1         |
| Interactions  | `(nSEG - 3) / 4`  | Overflow type codes past the first 6, packed 4 per float, present for `nSEG >= 7` |

### Simple member functions:
| Method            | Description                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| `.n_freq()`       | Returns the number of frequencies (1-127)                                       |
| `.n_seg()`        | Returns the number of segments                                                  |
| `.is_scalar()`    | Returns true for SCALAR layout, false for EM                                    |
| `.init()`         | (Re)initializes storage to a given segment / frequency layout                   |
| `.free()`         | Releases the data buffer and resets to the valid empty state                    |
| `.coord()`        | Returns a pointer to a segment's `[x, y, z]` coordinates                        |
| `.xpr_coeff()`    | Returns a pointer to a frequency's transfer coefficients                        |
| `.operator()`     | Returns a last-segment coordinate by index (0=x, 1=y, 2=z)                      |
| `.interaction_type_codes()` | Returns the full interaction type sequence as a vector                |
| `.set_interaction_type_codes()` | Sets the interaction type sequence; requires one code per segment |

### Function 'calc_length':
Calculates the total length of the path

- The stored member `length` spans origin through the last interaction; it never includes the leg to the destination
- The read-only overload adds only the final last-to-D leg to the stored member; the origin overloads recompute every leg and overwrite the member with the origin-to-last total
- Returns the full length including the leg to `D`; on an empty path (`nSEG == 0`) the read-only form returns NaN while the origin form returns the straight origin-to-destination distance
```
float calc_length(float Dx, float Dy, float Dz) const;                         // member length + last to D distance
float calc_length(float Dx, float Dy, float Dz, float Ox, float Oy, float Oz); // calculate full path length, update member
float calc_length(const float *D, const float *O = nullptr);                   // 3-element overload
```
- **`Dx`, `Dy`, `Dz`, `*D`** — Destination coordinates
- **`Ox`, `Oy`, `Oz`, `*O`** — Origin coordinates (optional); if provided, accumulated path length is updated

### Function 'calc_gain':
Calculates the path gain (linear power) from the transfer coefficients at one frequency

- EM mode returns the maximum column power of the 2x2 Jones matrix; SCALAR mode returns the squared magnitude of the pressure coefficient
- A degenerate coefficient set (all-zero, underflowed, or containing NaN) returns 0, flagging the path for pruning
- With `fGHz > 0`, free-space path loss is folded in: `(lambda / (4 pi d))^2` in EM mode, `1 / d^2` (frequency-independent) in SCALAR mode
- The distance `d` is the stored `length` by default; pass `len > 0` to override it, e.g. to match a total length from the
  read-only `calc_length` before the member is updated
```
float calc_gain(float fGHz = 0.0f, size_t freq = 0, float len = 0.0f) const;
```
- **`fGHz`** — Frequency in GHz; `> 0` applies path loss, `0` returns polarization power only. In SCALAR mode any
  positive value applies spherical spreading; the magnitude is ignored
- **`freq`** — Frequency index into the coefficient store, valid range `0` to `nFRQ - 1`
- **`len`** — Path length override in meters; `> 0` replaces the stored `length` for the path-loss term, `<= 0` uses the stored value

### Function 'xpr_update':
Left-multiplies the ray's transfer matrix by an interaction matrix, applies a power gain, and returns the resulting gain

- Updates the coefficient slot for one frequency in place: `M := coeff_update * M`, then scales by `sqrt(gain_update)` in amplitude
- `coeff_update == nullptr` applies only the gain; `gain_update == 1.0` applies only the multiply; both defaults leave the matrix unchanged and just measure it
- The update reads all old values before writing, so `coeff_update` may alias the slot; the return value uses the same gain definition as `calc_gain`
- Must be called after `extend` closes the path, so the stored `length` is final when path loss is requested
```
float xpr_update(const float *coeff_update = nullptr, float gain_update = 1.0f, size_t freq = 0, float fGHz = 0.0f, float len = 0.0f);
```
- **`coeff_update`** — Interaction matrix, length 8 (EM) or 2 (SCALAR); `nullptr` skips the multiply
- **`gain_update`** — Power gain applied to the slot (converted to an amplitude factor internally); `1.0` skips the scaling
- **`freq`** — Frequency index into the coefficient store, valid range `0` to `nFRQ - 1`
- **`fGHz`** — Frequency in GHz for the returned gain; `> 0` applies path loss as in `calc_gain`, `0` returns polarization power only
- **`len`** — Path length override in meters; `> 0` replaces the stored `length` for the path-loss term, `<= 0` uses the stored value

### Function 'duplicate';
Copies the path into an existing target object and returns its length

- Performs a deep copy: the target receives an independent data buffer and all metadata
- The target's previous contents are released; the source is left unchanged
```
float duplicate(path &target) const;
```
- **`target`** — Destination path, overwritten with a deep copy of the source

### Function 'extend':
Copies the path into a target and appends one new segment, returning the new total length

- The new coordinate is appended after the existing segments; the stored `length` grows by the origin-to-new-point distance of the appended leg
- The interaction `type` is recorded for the new segment, counters nREF, nTRA, nSUB, nSCT are not updated (left to the caller)
- The target receives an independent buffer sized for the extra segment; the source is left unchanged
- Throws if the source already holds the maximum of 255 segments
```
float extend(path &target, float x, float y, float z, uint8_t type = 0) const;
```
- **`target`** — Destination path, overwritten with the extended copy
- **`x`, `y`, `z`** — Coordinates of the appended interaction point
- **`type`** — Interaction type code for the new segment; drives the reflection / transmission counters

---
## qrt_file_append
Append one transmitter's path data to an existing QRT file

- Writes the path data for a single TX into the next free slot reserved by [qrt_file_init](#qrt_file_init), and records the TX name, position, and orientation.
- Paths are grouped by their CIR index (`path::iC`) and stored per CIR as: interaction counts, polarization coefficients, interaction coordinates, and interaction type codes.
- Validates every path before writing: `iC` must be within the file's CIR count, and each path's frequency count and layout mode (EM/scalar) must match the file exactly.
- Throws when all TX slots are already filled — the file holds at most the `no_orig` slots reserved at init.
- Only v5/6 files can be appended to. Positional data is converted to `float` on write.
- Returns the total number of paths written.

### Declaration:
```
template <typename dtype>
size_t quadriga_lib::qrt_file_append(
    const std::string &fn,
    const std::vector<quadriga_lib::path> &path_data,
    const arma::Col<dtype> &orig_pos,
    const arma::Col<dtype> &orig_orientation = {0.0, 0.0, 0.0},
    const std::string &orig_name = "TX");
```

### Inputs:
- **`fn`** — Path to an existing QRT file created by [qrt_file_init](#qrt_file_init)
- **`path_data`** — Paths to write; each carries its CIR index in `iC` and must match the file's frequency count and layout mode
- **`orig_pos`** — Transmitter position; `[3]`
- **`orig_orientation`** — Transmitter orientation (bank, tilt, head) in rad; `[3]`
- **`orig_name`** — Transmitter name; at most 255 characters

### Returns:
- Number of paths written across all CIRs.

### See also:
- [qrt_file_init](#qrt_file_init) (create the file and reserve TX slots)
- [qrt_file_read](#qrt_file_read) (read the appended data back)

---
## qrt_file_init
Create a new QRT file and write its metadata header

- Writes a v5 (EM) or v6 (scalar) header: frequencies, CIR positions/orientations, RX (MT) metadata, and a reserved BS (TX) region.
- Writes no path data. The BS position/orientation rows and the BS index table are reserved as zeros; each [qrt_file_append](#qrt_file_append) call fills the next free slot.
- CIR orientation is stored compressed: only the angle columns that carry a nonzero value are written, encoded in a per-file format byte.
- `no_orig` fixes the number of TX slots; the file can hold at most that many appended TX blocks.
- Positional data is converted to `float` on write regardless of `dtype`.

### Declaration:
```
template <typename dtype>
void quadriga_lib::qrt_file_init(
    const std::string &fn,
    const arma::Col<dtype> &freq,
    const arma::Mat<dtype> &cir_pos,
    const arma::Mat<dtype> &cir_orientation,
    const std::vector<std::string> &dest_names = {"RX"},
    const arma::u32_vec &cir_offset = {0},
    unsigned no_orig = 1,
    bool scalar_mode = false);
```

### Inputs:
- **`fn`** — Path to the QRT file to create (truncated if it exists)
- **`freq`** — Frequencies in GHz (EM) or Hz (scalar); `[n_freq]`, 1 to 127 entries
- **`cir_pos`** — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles (bank, tilt, head); `[no_cir, 3]` or empty for none
- **`dest_names`** — Receiver (MT) names; `[no_dest]`
- **`cir_offset`** — CIR offset for each receiver, 0-based; `[no_dest]`, must equal `dest_names` in length
- **`no_orig`** — Number of origin (TX) slots to reserve; at least 1
- **`scalar_mode`** — `true` writes a v6 scalar-layout file, `false` a v5 EM file

### See also:
- [qrt_file_append](#qrt_file_append) (write one TX block into a reserved slot)
- [qrt_file_read](#qrt_file_read) (read CIR data back)

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
- Reading a TX slot that was reserved by [qrt_file_init](#qrt_file_init) but not yet written by [qrt_file_append](#qrt_file_append) returns
  empty path outputs with zeroed positions, rather than throwing; an out-of-range index still throws.

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
    std::vector<uint8_t> *interact_type = nullptr,
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
- **`center_frequency`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3]`
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gain on linear scale; `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_coord`** — Interaction coordinates per path; vector of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; `[n_path]`
- **`coord`** — Interaction coordinates; `[3, sum(no_int)]`
- **`interact_type`** — Interaction type codes, concatenated per path and segmented by `no_int`; `[sum(no_int)]`.
  Empty for v4 legacy files.

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
## qrt_file_read_raw
Read raw ray-tracing path data from a QRT file into path objects

- Reassembles the stored path data for a single origin (TX) into a vector of [path](#path) objects — the inverse of [qrt_file_append](#qrt_file_append).
- Unlike [qrt_file_read](#qrt_file_read), which returns processed CIR data (channel matrices, angles, path loss), this returns the raw per-path storage:
  coordinates, polarization coefficients, and interaction type codes.
- Each returned path has `iC` set to its CIR index and `iR` to a running index within the origin. Counters not stored in the file
  (`nREF`, `nTRA`, `nSUB`, `nSCT`) and `length` are left at their defaults.
- Only v05/v06 files can be read as raw paths; v4 legacy files are rejected.
- An origin slot that was reserved by [qrt_file_init](#qrt_file_init) but never written by [qrt_file_append](#qrt_file_append) returns an empty vector.
- For tight loops over many origins, pass a pre-opened `std::ifstream` and a [qrt_read_cache_init](#qrt_read_cache_init)-populated cache to avoid
  re-parsing the header on each call.

### Declaration:
```
std::vector<quadriga_lib::path> quadriga_lib::qrt_file_read_raw(
    const std::string &fn,
    arma::uword i_orig = 0,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are supplied
- **`i_orig`** — Origin index to read (for downlink, origin = TX); must be less than `no_orig`
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally
- **`cache`** *(optional)* — Pre-parsed metadata from [qrt_read_cache_init](#qrt_read_cache_init); pass `nullptr` to parse the header on this call

### Returns:
- Vector of [path](#path) objects for the requested origin, one per stored path across all CIRs; empty if the origin slot is unwritten.

### See also:
- [qrt_file_append](#qrt_file_append) (write path data — the inverse operation)
- [qrt_read_cache_init](#qrt_read_cache_init) (populate cache for fast repeated reads)
- [qrt_file_read](#qrt_file_read) (read processed CIR data instead of raw paths)

---
## qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

- Reads all fixed metadata from a QRT file into a `quadriga_lib::qrt_read_cache` struct.
- Pre-computes byte offsets so subsequent [qrt_file_read](#qrt_file_read) calls need only 2 seeks and 4 reads instead of re-parsing the header.
- Populate once, then pass the cache and a shared `std::ifstream` to [qrt_file_read](#qrt_file_read) for tight-loop performance.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- For a TX slot reserved by [qrt_file_init](#qrt_file_init) but not yet written, `orig_index` and `path_data_offset`
  are `0`; [qrt_file_read](#qrt_file_read) treats that as an empty (unwritten) origin.

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
  | Member             | Type         | Description                                                                                     |
  | ------------------ | ------------ | ----------------------------------------------------------------------------------------------- |
  | `version`          | `int`        | QRT file version                                                                                |
  | `no_orig`          | `unsigned`   | Number of origin (TX) positions                                                                 |
  | `no_cir`           | `unsigned`   | Number of CIRs per origin                                                                       |
  | `no_dest`          | `unsigned`   | Number of destinations (RX)                                                                     |
  | `no_freq`          | `unsigned`   | Number of frequency bands                                                                       |
  | `freq`             | `arma::fvec` | Frequency in GHz; `[no_freq]`                                                                   |
  | `cir_pos`          | `arma::fmat` | CIR positions; `[no_cir, 3]`                                                                    |
  | `cir_orientation`  | `arma::fmat` | CIR orientations (Euler); `[no_cir, 3]`                                                         |
  | `orig_pos_all`     | `arma::fmat` | Origin positions; `[no_orig, 3]`                                                                |
  | `orig_orientation` | `arma::fmat` | Origin orientations (Euler); `[no_orig, 3]`                                                     |
  | `orig_index`       | `arma::uvec` | Byte offsets from BOF to each origin data block; `[no_orig]`                                    |
  | `path_data_offset` | `arma::uvec` | Absolute offset to path_data_index array per origin; 0 if the TX slot is unwritten; `[no_orig]` |

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
    double thin_slab_threshold = 0.0);
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face, 0 = no material (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail (0–6), controls `n_path` and `n_seg`; see [generate_diffraction_paths](#generate_diffraction_paths)
- **`verbose`** — Verbosity level
- **`sub_mesh_index`** — 0-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation); `[n_mesh]`
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). Selects
  interaction type passed to [ray_mesh_interact](#ray_mesh_interact) (4 vs. 1).
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold; 0 = resolve always (default), 1 = resolve never,  see [ray_state_update](#ray_state_update)

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear scale; `[n_pos]`
- **`xprmat`** — For EM mode: polarization transfer matrix excluding FSPL, interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`; `[8, n_pos]`;
  For scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`.
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]`

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
## cube
Construct a triangulated cube mesh

- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1) before scaling
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 * n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with [obj_file_write](#obj_file_write)
- Scale, Euler rotation, and translation are applied in that order (scale -> rotate -> translate)

### Declaration:
```
arma::Mat<dtype> quadriga_lib::cube(
    const arma::vec &scale = {1.0},
    const arma::vec &rotation = {0.0, 0.0, 0.0},
    const arma::vec &location = {0.0, 0.0, 0.0},
    const arma::uword n_div = 1);
```

### Inputs:
- **`scale`** — Length 1 scales all axes uniformly; length 3 scales `{x,y,z}` independently; empty = 1 (no scaling)
- **`rotation`** — Euler angles in [rad] about {x,y,z}, applied as R = Rz*Ry*Rx (Blender XYZ); length 3 or empty (no rotation)
- **`location`** — Translation {x,y,z} in [m]; length 3 or empty (origin)
- **`n_div`** — Number of subdivisions per edge; results in `12 * n_div^2` triangles

### Returns:
- Triangle mesh; each row holds `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[12 * n_div^2, 9]`

### See also:
- [icosphere](#icosphere)
- [subdivide_triangles](#subdivide_triangles)
- [obj_file_write](#obj_file_write)

---
## generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by [calc_diffraction_gain](#calc_diffraction_gain): elliptic-arc paths sampling the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so `sum(prod(weights,3),2) = 1`

### Declaration:
```
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
## interface_gain
Linear gain of a wave crossing a thin interface (lumped penetration loss)

- Computes `g = 10^(-A/10)`, where `A` [dB] is the lumped transmission loss applied once when a
  ray enters a material (the air-to-material or material-to-material front-side crossing). It is
  independent of path length and is applied on top of the Fresnel interface term `1 - abs(R)²`:
  - Power-law penetration loss `att·(f/fRef)^attB` (e.g. 3GPP TR 38.901 building-entry loss).
  - An optional Lorentzian coincidence feature `coiA / (1 + (coiQ·(f - coiF)/coiF)²)`, active only
    when `coiF > 0` and `coiA != 0`; negative `coiA` is a transmission dip (acoustic coincidence),
    positive `coiA` a stop-band. The total is clamped to `>= 0`.
- The reflection / conductivity columns (`a`, `b`, `c`, `d`) and the in-medium columns
  (`alpha`, `alphaB`, `m`) of `mtl_prop` are not used here — the Fresnel reflection is handled by
  the caller and the distance-dependent loss by [medium_gain](#medium_gain).

### Declaration:
```
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
- [medium_gain](#medium_gain) (for the distance-dependent in-medium loss)
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
## medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

- Computes `g = 10^(-A/10)`, where `A` [dB] is the total attenuation accumulated over a path
  of length `dist` inside the medium. The loss combines three contributions:
  - Conductivity-based loss from the complex permittivity model of ITU-R P.2040-1: `ε_r = a·(f/fRef)^b`,
    `σ = c·(f/fRef)^d`. These give an attenuation length `Δ` and a per-meter power loss `8.686 / Δ` dB/m.
  - Distance absorption of the form `α·(f/fRef)^αB` dB/m, intended to model excess loss not captured
    by `σ` (e.g. foliage, scattering media).
  - An acoustic mass-law term `m·log10((f/fRef)·dist)` dB, added once (not per meter) when `m > 0`,
    `dist` exceeds ~1.5 mm, and the term is positive; `m` is the mass-law slope in dB/decade.
- The penetration-loss columns (`att`, `attB`) of `mtl_prop` are not used — they describe
  thin-slab transmission loss, not propagation through a finite-thickness medium.

### Declaration:
```
dtype quadriga_lib::medium_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype dist,
    dtype center_frequency);
```

### Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl`
- **`iM`** —  1-based material index (0 = no material / air)
- **`dist`** — Path length of the ray inside the medium
- **`center_frequency`** — Center frequency in [Hz]

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
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

- Parses a triangulated `.obj`; quads and n-gons are rejected. Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by name with a trailing Blender
  `.NNN` suffix removed. Unmatched names throw when `csv_strict = true`; otherwise they map to index 0
  (no material). The two index spaces are decoupled, so several visual
  materials (e.g. `wall.001`, `wall.002`) may resolve to a single EM material.
- Geometry indices (`face_ind`, `obj_ind`) are 0-based. Material indices (`mtl_ind`, `csv_ind`) are
  1-based, with 0 reserved for the outside / no-material state (faces with no assigned material;
  for `csv_ind` in non-strict mode, also materials absent from the table).
- With an empty `fn_obj`, geometry and `.mtl` outputs are empty and only the table (`csv_names`, `csv_prop`) 
  is populated — useful for inspecting a CSV or the default library. If `fn_csv` is also empty, the built-in default table is returned.
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Declaration:
```
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
## obj_file_write
Write a triangulated Wavefront .obj (and .mtl) file

- Supply geometry as either `mesh`, or as `vert_list` + `face_ind`; giving both, or neither, is an error
- With `mesh`: `vert_list_out` + `face_ind_out` are derived from it, merging vertices of the same object that
  are closer than `threshold` (no merging across objects). With `vert_list`/`face_ind`: data is written unchanged
- Faces are written grouped by object; the faces of each object must form a contiguous block in `obj_ind`
- Without `obj_ind`/`obj_names`: a single object named `object` is written
- Without `mtl_ind`: no `usemtl` tags and no `.mtl` file are written. With `mtl_ind`, each face carries a
  1-based material index (0 = no material, leaving that face unassigned); pass `mtl_ind = nullptr` to omit materials entirely
- The `.mtl` (named after the `.obj`) lists each used material; values default to a gray material when `bsdf` is omitted
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the `.obj`):
  columns follow a fixed canonical order, then any extra `csv_prop` columns (alphabetical); `csv_write_defaults`
  additionally emits canonical columns absent from `csv_prop`, filled with their defaults (`a`, `e`, `fRef` = 1, else 0)

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
    const dtype threshold = 0.001,
    const arma::uvec *csv_ind = nullptr,
    const std::vector<std::string> *csv_names = nullptr,
    const std::unordered_map<std::string, std::vector<dtype>> *csv_prop = nullptr,
    bool csv_write_defaults = false);
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
## ray_init
Seed a sphere of rays from a point source

- Launches `n_ray` rays from a point source `O` onto an [icosphere](#icosphere) tessellation, giving near-uniform
  angular coverage of the full sphere (4π sr).
- `n_ray` is quantized to the icosphere grid: `n_div = round(sqrt(n_ray_target / 20))` (min 1) and
  `n_ray = 20 · n_div²`, so the returned count is the closest tessellation to `n_ray_target`, not exact.
- Ray origins sit on a small launch sphere of radius `r0` centered at `O`, not at `O` itself, so the beam
  triangles (`trivec`) have finite extent from the first segment. 
- When `mesh` is supplied, `r0` is auto-sized to 0.8× the nearest obstacle distance along a coarse probe
  sphere (clamped to ≥ 0.01 m); if no obstacle is hit within `max_path_length`, or without `mesh`, `r0 = 0.01 m`.
- Emits the per-ray medium-state words and distance accumulators consumed by [ray_state_update](#ray_state_update), all
  initialized to the outside-air / zero-distance start state.
- Beam wavefront (`trivec`) and directions (`tridir`, Cartesian) match the [ray_mesh_interact](#ray_mesh_interact) input format.

### Declaration:
```
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
    arma::fmat *path_dir_prev = nullptr,
    arma::fmat *acc_dist = nullptr,
    std::vector<quadriga_lib::path> *paths = nullptr,
    const arma::fmat *mesh = nullptr,
    const arma::u32_vec *sub_mesh_index = nullptr,
    bool scalar_mode = false);
```

### Inputs:
- **`n_ray_target`** — Desired ray count; quantized to the nearest icosphere grid (see above)
- **`n_freq`** — Number of frequency bins allocated per path in `paths`; must be ≥ 1 (throws if 0; the
  [path](#path) layout supports 1-127)
- **`Ox`**, **`Oy`**, **`Oz`** — Point-source (transmitter) position in GCS [m]
- **`max_path_length`** — Maximum ray length [m]; sets `dest` and bounds the launch-sphere probe (floored at 0.01 m)
- **`mesh`** *(optional)* — Triangle mesh faces; see [obj_file_read](#obj_file_read); `[n_mesh, 9]`. Used only to auto-size
  the launch sphere `r0`. NULL → `r0 = 0.01 m`.
- **`sub_mesh_index`** *(optional)* — Sub-mesh partition offsets for the accelerated intersect; passed to
  [ray_triangle_intersect](#ray_triangle_intersect). NULL → no partitioning.
- **`scalar_mode`** — Path storage layout passed to `paths`: `true` = SCALAR (acoustic, one pressure
  coefficient per frequency), `false` = EM (2×2 Jones matrix per frequency)

### Outputs:
- **`orig`** — Ray origins on the launch sphere, `O + r0·d̂`; `[n_ray, 3]`
- **`dest`** — Ray destinations at `max_path_length` from `O`; `[n_ray, 3]`
- **`trivec`** — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, matches [ray_mesh_interact](#ray_mesh_interact)
- **`tridir`** — Per-vertex ray directions, Cartesian; `[n_ray, 9]`
- **`mtl_ind_prev`**, **`mtl_ind_current`**, **`mtl_ind_buffer`** — Initial medium-state words for
  [ray_state_update](#ray_state_update), all zeroed (outside air, no flags); `[n_ray]`
- **`path_dir_prev`** — Initial physical ray direction (unit vectors from `O`); `[n_ray, 3]`
- **`acc_dist`** — Accumulated in-layer distance, zeroed; `[n_ray, 2]`; col 1 = refracted distance, col 2 = geometric distance
- **`paths`** — Per-ray [path](#path) objects, one per ray; each reinitialized to 0 segments with `n_freq`
  frequency bins in the `scalar_mode` layout, and `length` seeded to `orig_length` (the `O`-to-origin offset);
  `n_ray` entries

### Returns:
- Number of rays generated, `n_ray = 20 · n_div²`

### See also:
- [icosphere](#icosphere) (generates the ray fan, beam wavefront, and directions)
- [ray_triangle_intersect](#ray_triangle_intersect) (launch-sphere sizing and per-segment intersection)
- [ray_mesh_interact](#ray_mesh_interact) (consumes `orig` / `dest` / `trivec` / `tridir` / `orig_length`)
- [ray_state_update](#ray_state_update) (consumes the medium-state words, `path_dir_prev`, and `acc_dist`)
- [path](#path) (the per-ray storage object populated in `paths`)

---
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media.
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`).
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule); front-side hit with
  FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air; FBS=SBS with opposing normals → media-to-media.
- With `compact = true` (default), rays with `fbs_ind = 0` (no interaction) are omitted from output,
  so `n_rayN ≤ n_ray`; with `compact = false` they are kept as transparent pass-throughs (`n_rayN = n_ray`).
- Output direction encoding (spherical/Cartesian) matches input `tridir` format.
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves).
- Types 3–5 (scalar) use a single TE-only coefficient (acoustic simulation with impedance-mapped
  materials, `ε` derived from `Z`); total internal reflection is handled as in the EM path
  (`snell·sinθ ≥ 1` ⇒ unit reflection, refraction collapses to undeviated transmission).
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Declaration:
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
  - *Excluded:* in-medium attenuation and excess phase — added by [ray_state_update](#ray_state_update) — and FSPL / spreading
    loss, which is never applied here or downstream.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`fbs_angleN`** — Incidence angle at FBS in rad; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (`qd::bits<uint8_t>`); `[n_rayN]`
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
- **`path_dirN`** — Refraction-correct path direction: mirror for types 0/3, Snell direction for types 1/2/4/5; `[n_rayN, 3]`.
  For undeviated transmission (types 1/4) this is the *refracted* direction, which differs from the geometric continuation
  (along the incoming ray) used for `origN`/`destN`; it lets downstream code recover the true transmission angle.
- **`ray_indN`** — 0-based input ray index for each output ray (inverse of the internal compaction map; order-preserving); `[n_rayN]`

### See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [obj_file_read](#obj_file_read) (for loading `mesh` and `mtl_prop` from OBJ file)
- [ray_state_update](#ray_state_update) (inside/outside state machine)
- [icosphere](#icosphere) (for generating beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for computing FBS and SBS positions)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)

---
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin, enabling energy spread simulation.
- Reports, for each point, the list of 0-based ray indices whose beam intersects that point.
- The primary output is a flat CSR pair (`hit_index`, `hit_offset`); the ray indices of point `i` are `hit_index[hit_offset[i] .. hit_offset[i+1]-1]`.
- All outputs are optional and are only computed when a non-NULL pointer is passed.
- All internal computations use single precision.

### Declaration:
```
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
- **`sub_cloud_index`** *(optional)* — Segment boundary indices for the point cloud (see [point_cloud_segmentation](#point_cloud_segmentation)); `[n_sub]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_points >= 500` and CUDA is available, else AVX2, else GENERIC.
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
## ray_progress
Advance a ray set by one interaction, spawning reflected, transmitted, and subdivided rays

- Consumes a launch configuration (origins, destinations, per-ray medium state, and [path](#path) storage) and
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
  only the polarization/gain coefficient is recomputed and folded into each [path](#path); the per-frequency
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
```
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
    arma::fmat &path_dir_prev,
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
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read)); each value has length `n_mtl` (max 32767)
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
- **`mtl_ind_prev`**, **`mtl_ind_current`**, **`mtl_ind_buffer`** — Medium-state words (bit-masked: `mat = w & 0x7FFF`, `flag = w & 0x8000`); `[n_ray]`
- **`path_dir_prev`** — Physical ray direction entering the current segment (unit vectors); `[n_ray, 3]`
- **`acc_dist`** — Accumulated in-layer distance; `[n_ray, 2]`; col 1 = refracted distance, col 2 = geometric distance
- **`paths`** — Per-ray [path](#path) objects; `n_ray` entries. Frequency count and layout must match
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
- [path](#path) (per-ray storage object accumulated across generations)

---
## ray_state_update
Batched inside/outside ray-state machine with analytic thin-slab (Fabry-Perot) resolution

- Corrects the per-interaction `gainN` / `xprmatN` produced by [ray_mesh_interact](#ray_mesh_interact) using a tracked
  per-ray medium state, and carries that state forward. Three signed-`short` words per ray hold the
  current medium, the previous medium, and a one-slot next-transition buffer (bit-masked: `mat = w &
  0x7FFF`, `flag = w & 0x8000`).
- Implements the inside/outside state machine and overlays a closed-form thin-slab factor `S` (the Airy
  sum) so a single coefficient captures the full internal multiple-reflection series of a parallel slab
  thin enough to matter, instead of relying on the tracer to follow every internal bounce.
- Called twice per interaction by the ray tracer: once for the reflection pass (`interaction_type` 0
  or 3) and once for the transmission/refraction pass (`interaction_type` 1, 2, 4, 5). With `S`
  suppressed (the survival gate re-emits) the transmission/refraction path reproduces [calc_diffraction_gain](#calc_diffraction_gain)

### Declaration:
```
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
    const arma::Mat<dtype> *acc_dist_in = nullptr,
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    arma::Mat<dtype> *acc_dist_outN = nullptr,
    std::vector<uint8_t> *resolved_typeN = nullptr,
    const arma::u32_vec *ray_indN = nullptr,
    double eps = 0.15);
```

### Inputs:
- **`interaction_type`** — 0 EM reflection, 1 EM transmission, 2 EM refraction, 3 scalar reflection, 4 scalar transmission, 5 scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin, destination, full ray set; `[n_ray, 3]`, read at `g = ray_indN[i]`
- **`fbsN`**, **`sbsN`**  — First and second interaction points, compact set; `[n_rayN, 3]`
- **`no_interact`** — Mesh-hit count per ray, full ray set; `[n_ray]`
- **`fbs_angleN`** — Incidence angle at FBS (ITU convention), compact set; `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normals `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`, compact set; `[n_rayN, 6]`.
  The VBS plane normal for the Snell corrections; currently also gates the parallelism (wedge) test.
  NULL disables the wedge test.
- **`out_typeN`** — Interaction type code from [ray_mesh_interact](#ray_mesh_interact), compact set; `[n_rayN]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [obj_file_read](#obj_file_read))
- **`mtl_ind_fbsN`**, **`mtl_ind_sbsN`** — Material indices M1 / M2 of the FBS / SBS faces, compact set; `[n_rayN]` (0 = air)
- **`mtl_ind_prev_in`**, **`mtl_ind_current_in`**, **`mtl_ind_buffer_in`** — State words, full ray set; `[n_ray]`,
  read at `g`, never written. NULL reads as state `0` (outside, no flags).
- **`path_dir_prev`** — Physical ray direction entering this segment, full ray set; `[n_ray, 3]`
- **`acc_dist_in`** — Accumulated in-layer distance carried into this call, full ray set; `[n_ray, 2]`; col 1 = refracted distance; col 2 = geometric distance
- **`ray_indN`** — Compact-to-full ray index map; `[n_rayN]` to `[n_ray]`; NULL = identity (`n_ray == n_rayN`)
- **`eps`** *(optional)* — Thin-slab (Fabry-Pérot) resolve threshold on the round-trip in-slab
  amplitude `ρ` (`ρ ∈ [0, 1]`): the closed-form Airy factor is applied when `ρ ≥ eps` and the series
  is re-emitted to the tracer when `ρ < eps` (weak / fast-decaying slabs). `eps = 0` always resolves
  (for callers that cannot re-emit, e.g. [calc_diffraction_gain](#calc_diffraction_gain)); `eps ≥ 1` always re-emits
  (resolution disabled). A near-pole `S` or a known non-parallel slab re-emits regardless of `eps`.
  Default `0.15`.

### Outputs:
- **`mtl_ind_prev_outN`**, **`mtl_ind_current_outN`**, **`mtl_ind_buffer_outN`** — Updated state words,
  compact set; `[n_rayN]`. NULL skips the write. Passing all six state args NULL disables tracking —
  each interaction is corrected on its own (entry loss, TR kill, single-hit air-gap `S`); cross-interaction slab `S` and
  reflection-bounce `S` need the tracked medium.
- **`gainN`** *(in/out)* — Per-interaction linear power gain, updated in place; `[n_rayN]`. Kept consistent
  with `xprmatN` at every write. A killed ray (`resolved_typeN == 0`) yields `gainN = 0` together with
  an all-zero `xprmatN`.
- **`xprmatN`** *(in/out)* — Polarization transfer matrix, updated in place; `[8, n_rayN]` for EM mode,
  `[2, n_rayN]` for scalar mode. Same layout and basis convention as in [ray_mesh_interact](#ray_mesh_interact), but with the
  medium closed out: on return it additionally contains the in-medium attenuation and excess phase of the
  traversed segment and, when the resolve bit of `resolved_typeN` is set, the closed-form thin-slab (Airy)
  factor `S` covering the full internal multiple-reflection series. FSPL / spreading loss remains excluded,
  so the matrix is the complete per-interaction Jones factor and can be left-multiplied directly into a
  per-path product. When the series is re-emitted instead (`ρ < eps`, resolve bit clear), the matrix carries
  only the first-pass coefficient and the remaining bounces arrive as separate interactions.
- **`path_dirN`** *(in/out)* — Continuation direction, corrected in place by the VBS construction, compact set; `[n_rayN, 3]`
- **`acc_dist_outN`** — Accumulated VBS distance leaving this call, compact set; `[n_rayN, 2]`
- **`resolved_typeN`** *(optional)* — Resolved interaction-type code, bit-encoded (`qd::bits<uint8_t>`),
  compact set; `[n_rayN]`. 0 = ray killed. NULL skips the write.
   | Bit  | Flag        | Meaning                                                                        |
   | :--: | :---------: | ------------------------------------------------------------------------------ |
   |   0  | ok          | OK flag (0 = a deferred degenerate-resolve buffer is pending)                  |
   |   1  | vbs         | VBS correction (gain/xprmat corrected at the VBS instead of FBS/SBS)           |
   |   2  | resolve     | Slab-resolve flag (an internal multi-bounce series was resolved analytically)  |
   |   3  | inside      | Inside-object flag (1 = ray continues inside, 0 = continues outside)           |
   |   4  | fix         | Fix flag (resolved-false-outside, or entry/exit material mismatch)             |
   |   5  | tir         | Total-reflection flag (also set when a transmission factor forced reflection)  |
   |   6  | trans       | Transmission: transparent-interface flag; reflection: scatter flag             |
   |   7  | refl        | Reflection flag (0 = transmission/refraction, 1 = reflection/scattering)       |
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
   Reachable composite values for reflection:
   |  Dec | Hex  |  FIX |   TIR   | Flags set                       | Meaning                                                      |
   | :--: | :--: | :--: | :-----: | ------------------------------- |------------------------------------------------------------- |
   |  129 | 0x81 |    — |   161   | refl, ok                        | eager front reflection (R0), outside (FBS==VBS)              |
   |  137 | 0x89 |  153 | 169,185 | refl, inside, ok                | internal back-reflection (incoming refr. 2/5, FBS==VBS)      |
   |  139 | 0x8B |  155 | 171,187 | refl, inside, vbs, ok           | internal back-reflection (incoming undev. 1/4, VBS)          |
   |  141 | 0x8D |  157 | 173,189 | refl, inside, resolve, ok       | internal back-reflection + slab series (incoming refr. 2/5)  |
   |  143 | 0x8F |  159 | 175,191 | refl, inside, resolve, vbs, ok  | internal back-reflection + slab series (incoming undev. 1/4) |
   |  192+| 0xC0+|    — |    —    | refl, trans, ...                | reserved: scattering not implemented                         |

### See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [ray_mesh_interact](#ray_mesh_interact) (computes the per-interaction Fresnel/Jones result this function corrects)
- [calc_diffraction_gain](#calc_diffraction_gain) (the reference state machine this function ports)

---
## ray_subdivide_flag
Compute the per-ray subdivision decision from the beam-tube footprint at the first bounce

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
```
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
- **`paths`** — Per-ray [path](#path) objects; `n_ray` entries. Only the subdivision counter `nSUB` and the segment count are read
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
## refractive_index
Real refractive index of a homogeneous medium

- Returns `n = Re(sqrt(ε_r · μ_r))`, the real part of the complex refractive index, using the
  ITU-R P.2040-1 permittivity model `ε_r = a·(f/fRef)^b` together with the relative permeability `μ_r`.
- Only the bulk (base) permittivity is used. The coincidence / resonance features
  (`coiF`, `coiQ`, `coiA`, `resF`, ...) are excluded, since they model a thin-interface surface
  effect, not bulk propagation, and must not enter the geometric refraction index.
- Air (`iM = 0`) returns `1`.

### Declaration:
```
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
- [medium_gain](#medium_gain) (for the distance-dependent in-medium loss)
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
## subdivide_rays
Subdivide ray beams into four smaller sub-beams

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
```
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
## subdivide_triangles
Subdivide triangles into smaller triangles

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in x n_div x n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles

### Declaration:
```
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
- If `mtl_ind` is provided, material indices are reordered and padded the same way (padding uses index 0)

### Declaration:
```
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
- **`mesh_index`** *(optional)* — 1-based mapping from original to reorganized mesh (0 = padding); `[n_meshR]`

### Returns:
- Number of created sub-meshes `n_sub`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (uses `sub_mesh_index` for acceleration)
- [obj_file_read](#obj_file_read) (defines `mtl_ind` / `csv_ind`)

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

---
## xpr_update
Thread a polarization transfer (Jones) matrix along a ray path

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
```
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
  `< n_ray`.

### Usage:
The flags are composable in one call; the per-column order is always
initialize -> update -> normalize -> apply_gain. Argument order is
`(xprmat, update, gain, initialize, normalize, apply_gain, ray_index)`.
```
arma::Mat<dtype> M;                                 // ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH (EM)

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
- **Scalar mode**, 2 values per ray: `[Re Im]`.
- The update is `M_out = M_update * M_state` (new interaction left-multiplies), consistent with
  `E_out = M * E_in` for column vectors and Armadillo/MATLAB column-major storage.

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (produces `xprmat` in this exact layout)

