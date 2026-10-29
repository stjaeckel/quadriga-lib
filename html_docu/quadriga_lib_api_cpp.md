---
title: "C++ API Documentation for Quadriga-Lib v0.12.4"
author: "Stephan Jaeckel"
date: "29.10.2026"
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
| [arrayant](#arrayant) | Array antenna class | 170 |
| [.append](#append) | Array antenna class | 221 |
| [.calc_beamwidth_deg](#calc_beamwidth_deg) | Array antenna class | 244 |
| [.calc_directivity_dBi](#calc_directivity_dbi) | Array antenna class | 282 |
| [.combine_pattern](#combine_pattern) | Array antenna class | 304 |
| [.copy_element](#copy_element) | Array antenna class | 334 |
| [.export_obj_file](#export_obj_file) | Array antenna class | 355 |
| [.interpolate](#interpolate) | Array antenna class | 385 |
| [.is_valid](#is_valid) | Array antenna class | 438 |
| [.qdant_write](#qdant_write) | Array antenna class | 456 |
| [.remove_zeros](#remove_zeros) | Array antenna class | 495 |
| [.rotate_pattern](#rotate_pattern) | Array antenna class | 513 |
| [.set_size](#set_size) | Array antenna class | 551 |
| [arrayant_combine_pattern_multi](#arrayant_combine_pattern_multi) | Array antenna functions | 577 |
| [arrayant_concat_multi](#arrayant_concat_multi) | Array antenna functions | 615 |
| [arrayant_copy_element_multi](#arrayant_copy_element_multi) | Array antenna functions | 649 |
| [arrayant_interpolate_multi](#arrayant_interpolate_multi) | Array antenna functions | 689 |
| [arrayant_is_valid_multi](#arrayant_is_valid_multi) | Array antenna functions | 748 |
| [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) | Array antenna functions | 775 |
| [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) | Array antenna functions | 807 |
| [generate_arrayant_3GPP](#generate_arrayant_3gpp) | Array antenna functions | 833 |
| [generate_arrayant_custom](#generate_arrayant_custom) | Array antenna functions | 883 |
| [generate_arrayant_dipole](#generate_arrayant_dipole) | Array antenna functions | 908 |
| [generate_arrayant_half_wave_dipole](#generate_arrayant_half_wave_dipole) | Array antenna functions | 925 |
| [generate_arrayant_multibeam](#generate_arrayant_multibeam) | Array antenna functions | 942 |
| [generate_arrayant_omni](#generate_arrayant_omni) | Array antenna functions | 996 |
| [generate_arrayant_ula](#generate_arrayant_ula) | Array antenna functions | 1013 |
| [generate_arrayant_xpol](#generate_arrayant_xpol) | Array antenna functions | 1041 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 1058 |
| [qdant_read](#qdant_read) | Array antenna functions | 1133 |
| [qdant_read_multi](#qdant_read_multi) | Array antenna functions | 1165 |
| [qdant_write_multi](#qdant_write_multi) | Array antenna functions | 1192 |
| [channel](#channel) | Channel class | 1222 |
| [.add_paths](#add_paths) | Channel class | 1269 |
| [.calc_effective_path_gain](#calc_effective_path_gain) | Channel class | 1308 |
| [.write_paths_to_obj_file](#write_paths_to_obj_file) | Channel class | 1327 |
| [any_type_id](#any_type_id) | Channel functions | 1368 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 1411 |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Channel functions | 1456 |
| [baseband_freq_response_vec](#baseband_freq_response_vec) | Channel functions | 1508 |
| [hdf5_create](#hdf5_create) | Channel functions | 1546 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 1581 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 1621 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 1664 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 1699 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 1726 |
| [hdf5_write](#hdf5_write) | Channel functions | 1756 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1804 |
| [path](#path) | Channel functions | 1849 |
| [qrt_file_append](#qrt_file_append) | Channel functions | 1969 |
| [qrt_file_init](#qrt_file_init) | Channel functions | 2005 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 2045 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 2094 |
| [qrt_file_read_raw](#qrt_file_read_raw) | Channel functions | 2193 |
| [qrt_read_cache_init](#qrt_read_cache_init) | Channel functions | 2231 |
| [quantize_delays](#quantize_delays) | Channel functions | 2271 |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generation functions | 2350 |
| [get_channels_irs](#get_channels_irs) | Channel generation functions | 2437 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 2541 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 2633 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 2715 |
| [acdf](#acdf) | Channel statistics | 2809 |
| [calc_angular_spreads_sphere](#calc_angular_spreads_sphere) | Channel statistics | 2847 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 2893 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 2959 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 2995 |
| [conf_file_list](#conf_file_list) | Geometry-based stochastic model | 3060 |
| [conf_file_read](#conf_file_read) | Geometry-based stochastic model | 3107 |
| [conf_file_write](#conf_file_write) | Geometry-based stochastic model | 3189 |
| [los_probability](#los_probability) | Geometry-based stochastic model | 3276 |
| [lsf_draw](#lsf_draw) | Geometry-based stochastic model | 3378 |
| [lsf_init_phases](#lsf_init_phases) | Geometry-based stochastic model | 3471 |
| [lsp_generate](#lsp_generate) | Geometry-based stochastic model | 3565 |
| [o2i_loss](#o2i_loss) | Geometry-based stochastic model | 3688 |
| [path_loss](#path_loss) | Geometry-based stochastic model | 3757 |
| [scenpar_validate](#scenpar_validate) | Geometry-based stochastic model | 3881 |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | 3948 |
| [fast_acos](#fast_acos) | Math functions | 3981 |
| [fast_asin](#fast_asin) | Math functions | 4000 |
| [fast_atan2](#fast_atan2) | Math functions | 4019 |
| [fast_cart2geo](#fast_cart2geo) | Math functions | 4041 |
| [fast_geo2cart](#fast_geo2cart) | Math functions | 4077 |
| [fast_sincos](#fast_sincos) | Math functions | 4123 |
| [fast_slerp](#fast_slerp) | Math functions | 4147 |
| [interp_2D](#interp_2d) | Math functions | 4184 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 4252 |
| [colormap](#colormap) | Site-specific simulation tools | 4341 |
| [combine_irs_coord](#combine_irs_coord) | Site-specific simulation tools | 4360 |
| [coord2path](#coord2path) | Site-specific simulation tools | 4405 |
| [cube](#cube) | Site-specific simulation tools | 4443 |
| [generate_diffraction_paths](#generate_diffraction_paths) | Site-specific simulation tools | 4478 |
| [icosphere](#icosphere) | Site-specific simulation tools | 4523 |
| [interface_gain](#interface_gain) | Site-specific simulation tools | 4557 |
| [medium_gain](#medium_gain) | Site-specific simulation tools | 4590 |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Site-specific simulation tools | 4631 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 4674 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 4749 |
| [obj_overlap_test](#obj_overlap_test) | Site-specific simulation tools | 4821 |
| [path_to_tube](#path_to_tube) | Site-specific simulation tools | 4853 |
| [plane](#plane) | Site-specific simulation tools | 4882 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 4920 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 4949 |
| [point_cloud_split](#point_cloud_split) | Site-specific simulation tools | 4989 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 5025 |
| [ray_commit](#ray_commit) | Site-specific simulation tools | 5061 |
| [ray_init](#ray_init) | Site-specific simulation tools | 5166 |
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | 5259 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 5404 |
| [ray_progress](#ray_progress) | Site-specific simulation tools | 5453 |
| [ray_state_update](#ray_state_update) | Site-specific simulation tools | 5598 |
| [ray_subdivide_flag](#ray_subdivide_flag) | Site-specific simulation tools | 5801 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 5862 |
| [refractive_index](#refractive_index) | Site-specific simulation tools | 5916 |
| [subdivide_rays](#subdivide_rays) | Site-specific simulation tools | 5948 |
| [subdivide_triangles](#subdivide_triangles) | Site-specific simulation tools | 6006 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 6038 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 6066 |
| [triangle_mesh_split](#triangle_mesh_split) | Site-specific simulation tools | 6107 |
| [write_png](#write_png) | Site-specific simulation tools | 6142 |
| [xpr_update](#xpr_update) | Site-specific simulation tools | 6173 |
| [sos_acf_2d](#sos_acf_2d) | Sum-of-sinusoids random number generator | 6420 |
| [sos_acf_approx](#sos_acf_approx) | Sum-of-sinusoids random number generator | 6455 |
| [sos_acf_estimate](#sos_acf_estimate) | Sum-of-sinusoids random number generator | 6496 |
| [sos_acfi](#sos_acfi) | Sum-of-sinusoids random number generator | 6556 |
| [sos_calc_mse](#sos_calc_mse) | Sum-of-sinusoids random number generator | 6586 |
| [sos_generate](#sos_generate) | Sum-of-sinusoids random number generator | 6647 |
| [sos_init_phases](#sos_init_phases) | Sum-of-sinusoids random number generator | 6727 |
| [sos_map](#sos_map) | Sum-of-sinusoids random number generator | 6766 |
| [sos_preset](#sos_preset) | Sum-of-sinusoids random number generator | 6814 |
| [sos_rand](#sos_rand) | Sum-of-sinusoids random number generator | 6864 |
| [sos_randi](#sos_randi) | Sum-of-sinusoids random number generator | 6925 |
| [sos_randn](#sos_randn) | Sum-of-sinusoids random number generator | 6989 |
| [sos_remove_bias](#sos_remove_bias) | Sum-of-sinusoids random number generator | 7049 |
| [sos_val](#sos_val) | Sum-of-sinusoids random number generator | 7135 |

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
| .[calc_directivity_dBi](#calc_directivity_dbi) | Calculate the directivity in dBi of a single array element        |
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
- An empty `coupling_re` counts as identity (one port per element), an empty `coupling_im` as zeros

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
- The width is that of the main lobe: starting at the maximum, the search goes outwards until the power drops below the
  threshold; side lobes above the threshold are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360° when
  the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°
- Ignores element coupling

### Declaration:
```
void calc_beamwidth_deg(arma::uword i_element,
    dtype threshold_dB = 3.0,
    dtype *beamwidth_az = nullptr,
    dtype *beamwidth_el = nullptr,
    dtype *az_point_ang = nullptr,
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
- .[calc_directivity_dBi](#calc_directivity_dbi) (directivity in dBi of a single array element)

---
## .calc_directivity_dBi
Calculate the directivity in dBi of a single array element

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Independent of the scale of the pattern; an element whose pattern is zero everywhere returns 0 dBi
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
- The element positions add the phase `-2 pi f d / propagation_speed` with `f = center_frequency` and `d` the distance
  of the element from the wavefront plane through the origin; the default speed of light applies to radio waves, use
  about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))
- Useful for beamforming and MIMO channel computation speedup

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::combine_pattern(
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    dtype propagation_speed = 299792458.0) const;
```

### Inputs:
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], sorted; defaults to input grid
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], sorted; defaults to input grid
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

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
- Throws `std::invalid_argument` for a read-only object or an empty `destination`

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
- NaN or infinite angles give NaN in all outputs for that angle (and element); no exception is thrown
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
- Without `layout` (empty), the layout in the file is replaced by a `[1, n]` row of all IDs in the file in file
  order; a custom layout written before is lost. Pass the layout again with every write that should keep it.
- The file has single-precision-equivalent accuracy relative to the element peak, also for double-precision data:
  pattern magnitudes (in dB) and phases (in degree) get the fewest decimals (0 to 6) that keep the error of each
  complex sample at or below 6e-8 times the peak amplitude of its element (over e_theta and e_phi). Samples near
  the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the
  peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at
  a low level is quantized relative to its own peak. Element positions and coupling (magnitude, phase) are written
  as the shortest text that reads back as the same float (at most 9 significant digits); sampling grids are rounded
  to 0.001 degree
- All non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element that is
  zero in both field components is written as an all `-inf` `EthetaMag` node.

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
- A column or row is removed if the total power summed over all field components and array elements does not exceed
  1e-12 times the peak power of a single sample (summed over field components and elements); the result does not
  depend on the scale of the pattern; an all-zero pattern is left unchanged
- Immediate neighbors of retained entries are also kept unless the local grid spacing is strongly asymmetric (factor > 1.5)

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
- Coupling, `center_frequency` and `name` are kept (also in `*output`); when a single `element` is written to
  `*output`, the output holds only that element and its coupling is identity

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
- The element positions add the phase `-2 pi f d / propagation_speed` at each output frequency `f`; the default speed
  of light applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Declaration:
```
std::vector<arrayant<dtype>> arrayant_combine_pattern_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    const arma::Col<dtype> *freq_grid_new = nullptr,
    dtype propagation_speed = 299792458.0);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects (must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi))
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], sorted; defaults to `arrayant_vec[0].azimuth_grid`
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], sorted; defaults to `arrayant_vec[0].elevation_grid`
- **`freq_grid_new`** *(optional)* — Alternative frequency grid in Hz; defaults to per-entry `center_frequency`
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

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
- **`weight`** *(optional)* — Per-beam scaling factors (not normalized); `[n_beams]`
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:
   | `pol` | Description           | Elements |
   | ----- | --------------------- | -------- |
   | 1     | Vertical polarization | NM       |
   | 2     | H/V polarization      | 2NM      |
   | 3     | ±45° polarization     | 2NM      |
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`az_3dB`** *(optional)* — Per-element azimuth 3dB beamwidth in degrees; values <= 0 select the default 120
- **`el_3dB`** *(optional)* — Per-element elevation 3dB beamwidth in degrees; values <= 0 select the default 120
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
- Throws `std::invalid_argument` if the file has no entry with the given ID. The first entry of a file is ID 1 only
  if it has no `id` attribute (files written by other tools); otherwise ID 1 must exist.
- Malformed files throw `std::invalid_argument`. The number of elements (`NoElements`) must be backed by data in the
  file (element positions, coupling, or pattern nodes of the last element). Elements without pattern nodes read as
  zeros; their zero patterns, and the identity coupling of a file without `CouplingAbs`, are limited to 2^24 values
  each (e.g. 256 elements without pattern data on a 1-degree grid, 4096 elements without coupling).

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
- All entries are validated first; an existing file is deleted only after the validation passed, then rewritten.
  Invalid input throws and leaves an existing file unchanged.
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
- The first provided field sets `n_path_add`, also when it holds 0 paths; a field with a different path count throws

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
- The phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to one period
  before sin / cos are evaluated in single precision, so long (absolute) delays keep their accuracy; coefficients and
  sums are single precision; uses AVX2 for 8-carrier parallel computation; double inputs are narrowed to float
  internally, results widened back
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
- **`remove_delay_phase`** *(optional)* — Removes baked-in `exp(-j·2π·freq_in[f]·delay)` before SLERP and re-applies
  analytically at output frequencies; must be `true` for output from [get_channels_multifreq](#get_channels_multifreq) or
  [get_channels_spherical](#get_channels_spherical), `false` for pure envelope coefficients. These channels must be generated with
  `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also when `delay` holds
  relative delays. With relative delays the phase of the TX-RX distance stays in the coefficients and is interpolated
  across the `freq_in` grid, which gives wrong phases (magnitudes stay right) unless the `freq_in` spacing is far
  below `c / (TX-RX distance)`

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
- Same arithmetic as [baseband_freq_response](#baseband_freq_response): phase in double precision, sin / cos and sums in single precision

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
## hdf5_create
Create a new HDF5 channel file with a defined storage layout

- Initializes a new HDF5 file for storing wireless channel data; overwrites existing files.
- Defines a 4D layout (x, y, z, w) where each index combination maps to one channel storage slot.
- Typical dimension mapping: x = BS, y = UE, z = frequency, w = scenario/repetition.
- Layout can be reshaped later with [hdf5_reshape_layout](#hdf5_reshape_layout) if the total slot count stays constant.
- The total slot count `nx × ny × nz × nw` must not exceed 4294967295 (2^32 - 1); larger layouts throw before
  the file is created.
- Files use the quadriga-lib channel file format version 3, an HDF5 file readable by any HDF5 reader (h5py, MATLAB,
  HDFView). Modify files only with the quadriga-lib functions: files changed by other HDF5 tools are not supported
  and may be rejected.

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

- Returns an empty channel object (`n_snap() == 0`) if the slot is empty or outside the layout.
- Structured data is stored in single precision in the file and converted to `dtype` on read.
- Unstructured fields (`std::any`) retain their original stored type without conversion; see [hdf5_read_dset](#hdf5_read_dset)
  for the returned types.
- Round trip with [hdf5_write](#hdf5_write): every per-snapshot field comes back with one entry per snapshot, also for
  snapshots without paths (empty entries keep their shape); a channel written without `center_frequency` comes back
  with an empty `center_frequency`; empty `tx_orientation` / `rx_orientation` come back as `[3, 1]` zeros.
- Throws `std::invalid_argument` if the file is malformed (e.g. a dataset does not have the size given by the
  attributes of the channel, or a required dataset or attribute is missing) or is not a channel file of version 3
  written by quadriga-lib: files written by quadriga-lib 0.12.4 and earlier (version 2) are rejected; files modified
  with other HDF5 tools are not supported and may be rejected.

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
- Returned types: a scalar dataset gives a scalar; a 1D dataset an `arma::Col`, also when it has a single element;
  a 2D dataset an `arma::Mat`; a 3D dataset an `arma::Cube`. An `arma::Row` written with [hdf5_write_dset](#hdf5_write_dset)
  therefore comes back as an `arma::Col`.
- Element types: float, double, 32- and 64-bit signed and unsigned integers.
- Strings are returned without trailing NUL characters (an empty string written by [hdf5_write_dset](#hdf5_write_dset) comes back
  empty).
- Throws if the file does not exist or is not a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).

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
- Throws if the file does not exist or is not a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).

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
- Returns `{0, 0, 0, 0}` if the file does not exist; throws if the file exists but is not a channel file of
  version 3 (see [hdf5_read_channel](#hdf5_read_channel)).
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
- Throws if the new layout violates the total-count constraint or has more than 4294967295 (2^32 - 1) slots.
- Throws if the file does not exist or is not a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).
- One writer per file at a time; files are not locked.

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
- Creates the file with default layout `(65536 × 1 × 1 × 1)` if it does not exist; an existing file must be a
  channel file of version 3 written by quadriga-lib (files of quadriga-lib 0.12.4 and earlier are rejected, see
  [hdf5_read_channel](#hdf5_read_channel)).
- If the slot already holds data, the slot is replaced completely: the old structured data and all unstructured
  datasets of the slot (`par_*`, including those added with [hdf5_write_dset](#hdf5_write_dset)) are removed, and only the new
  channel is stored there. The space of the removed data is not reused, so the file does not shrink; to reclaim
  it, copy the channels into a new file ([hdf5_read_channel](#hdf5_read_channel), [hdf5_write](#hdf5_write)).
- Throws if the index was not reserved during [hdf5_create](#hdf5_create).
- Structured data is always stored in single precision; input may be float or double.
- Unstructured data: supported types are string, double, float, (u)int32, (u)int64; up to 3D; storage order preserved.
- Set `assume_valid = true` to skip integrity validation (faster but unsafe for potentially corrupted data).
- The channel name is stored with at most 255 characters.
- One writer per file at a time; files are not locked. Reading is safe while no one writes; different files can be
  read and written from parallel threads.

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
- `0` if the slot was empty, `1` if existing data in the slot was replaced

### See also:
- [hdf5_create](#hdf5_create) (create file and reserve layout)
- [hdf5_read_channel](#hdf5_read_channel) (read channel data from a slot)

---
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file at a specified 4D index

- Dataset is stored under `prefix + par_name`; name must contain only alphanumeric characters and underscores.
- Adds the dataset to the slot and keeps everything else stored there (channel data and other datasets); an empty
  slot is reserved for it. A later [hdf5_write](#hdf5_write) to the same slot replaces the slot completely, including the
  datasets written with this function.
- Supported scalar types: `std::string`, `unsigned`, `int`, `long long`, `unsigned long long`, `float`, `double`
- Supported Armadillo types: `arma::Col`, `arma::Row`, `arma::Mat`, `arma::Cube` with element types `float`, `double`, `int`, `unsigned`, `sword`, `uword`, `unsigned long long`
- `arma::Row` is converted to `arma::Col` before writing.
- Throws for unsupported types and invalid names, before the file is changed.
- Throws if a dataset with the same name already exists at the specified slot; no overwrite/update is supported.
- One writer per file at a time; files are not locked.
- Creates the file with the default layout of [hdf5_create](#hdf5_create) if it does not exist; an existing file must be
  a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).

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
- A 56-byte header holds metadata, the first 2 interaction type codes, and the frequency-0
  transfer coefficients; a variable-length heap buffer holds coordinates, the remaining
  frequency coefficients, and the overflow interaction codes
- Two layout modes selected at initialization: EM carries a full 2x2 Jones matrix per frequency;
  SCALAR carries a single complex pressure coefficient per frequency
- Copyable and movable; the moved-from object is left in the valid empty state (`nSEG == 0`, `nFRQ == 1`, EM)

### Attributes:
| Attribute            | Size   | Description                                                       |
| -------------------- | ------ | ----------------------------------------------------------------- |
| `unsigned iC`        | scalar | Channel ID: the channel to which the path belongs                 |
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
| Interactions  | `(nSEG + 1) / 4`  | Overflow type codes past the first 2, packed 4 per float, present for `nSEG > 2` |

### Data buffer layout (SCALAR mode):
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Coeff         | `2 * (nFRQ - 4)`  | Pressure coefficients for frequencies 4..nFRQ-1         |
| Interactions  | `(nSEG + 1) / 4`  | Overflow type codes past the first 2, packed 4 per float, present for `nSEG > 2` |

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
- CIR orientation is stored compressed: only the angle columns that carry a nonzero value after conversion to `float` are written,
  encoded in a per-file format byte. [qrt_file_append](#qrt_file_append) reads this byte from the file.
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
- **`fn`** — Path to the QRT file to create (truncated if it exists; all inputs are checked before the file is
  opened, so an invalid input throws and leaves an existing file unchanged)
- **`freq`** — Frequencies in GHz (EM) or Hz (scalar); `[n_freq]`, 1 to 127 entries
- **`cir_pos`** — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles (bank, tilt, head); `[no_cir, 3]` or empty for none
- **`dest_names`** — Receiver (MT) names, at most 255 characters each; `[no_dest]`
- **`cir_offset`** — CIR offset for each receiver, 0-based, the first entry 0; `[no_dest]`, must equal `dest_names` in length
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
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.

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
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.

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
  Other values throw `std::invalid_argument`.
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
- Each returned path has `iC` set to its CIR index. Counters not stored in the file
  (`nREF`, `nTRA`, `nSUB`, `nSCT`) and `length` are left at their defaults.
- Only v05/v06 files can be read as raw paths; v4 legacy files are rejected.
- An origin slot that was reserved by [qrt_file_init](#qrt_file_init) but never written by [qrt_file_append](#qrt_file_append) returns an empty vector.
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.
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
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.

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
  internally when `fix_taps` is 0 or 3, and output delays remain shared `[1, 1, n_taps]` when `fix_taps` is 1 or 2;
  per-antenna input always gives `[n_rx, n_tx, n_taps]` output delays (identical for all pairs when `fix_taps` is 1 or 2)
- `n_rx` and `n_tx` must be identical across all snapshots; `n_path_s` may differ per snapshot
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0 (there is no tap below 0 to
  interpolate with; the error is below `tap_spacing / 2`, as for nearest-tap rounding). Delays at or below
  `-tap_spacing / 2` throw `std::invalid_argument`, as do NaN, Inf and delays of 2^31 taps or more
  (`delay / tap_spacing < 2^31`, e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](#get_channels_spherical), [get_channels_multifreq](#get_channels_multifreq) and
  [get_channels_planar](#get_channels_planar)) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values
- Outputs may alias inputs, also across arguments (e.g. `delay_quant == coeff_re`); all inputs are read before the
  outputs are written. The three outputs must be different objects.

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
- **`delay`** — Path delays in seconds; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`;
  non-negative with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** *(optional)* — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate
- **`max_no_taps`** *(optional)* — Maximum number of output taps; 0 means unlimited
- **`power_exponent`** *(optional)* — Interpolation exponent alpha; 1.0 = narrowband, 0.5 = wideband power-preserving
- **`fix_taps`** *(optional)* — Delay grid sharing mode:
  | Value | Meaning                                                                                                                               |
  | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps]`                                                                     |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps]` for shared delay input (`[n_rx, n_tx, n_taps]` for per-antenna input), identical for every snapshot |
  | 2     | Per snapshot; output delays `[1, 1, n_taps]` for shared delay input (`[n_rx, n_tx, n_taps]` for per-antenna input), but each snapshot has its own independent tap grid — taps do not align across snapshots |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps]`                                                             |

### Outputs:
- **`coeff_re_quant`** — Output coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_taps]`
- **`coeff_im_quant`** — Output coefficients, imaginary part; same layout as `coeff_re_quant`
- **`delay_quant`** — Output delays in seconds; each cube `[n_rx, n_tx, n_taps]`, or `[1, 1, n_taps]` for shared delay input with
  `fix_taps` 1 or 2

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
- Floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz; floors add attenuation
  (and the TGah shadow fading below 1 GHz) but no delay; the station is placed 3 m per floor above the AP
- Taps with zero power in the model tables are removed; the number of paths does not depend on distance, floors or walls
- Time evolution: each sub-path is rotated by `exp(+j 2 pi f_D t)`, i.e. a station moving towards the AP or a scatterer
  sees a positive Doppler shift `f_D`, as the channel of [get_channels_spherical](#get_channels_spherical) along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- NAN or negative value for any override parameter restores the model default, except `XPR_NLOS_linear` (see below)

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
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency (finite)
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`
- **`observation_time`** *(optional)* — Channel observation time in seconds; must be finite
- **`update_rate`** *(optional)* — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`
- **`speed_station_kmh`** *(optional)* — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`
- **`Dist_m`** *(optional)* — TX-to-RX distance(s) (finite); `[n_users]` or `[1]`
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `[n_users]` or `[1]`
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; empty uses TGac auto-defaults for `n_users > 1`; `[4, n_users]`
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); `0.0` disables
- **`seed`** *(optional)* — RNG seed for repeatability; `-1` uses the system random device
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN restores model default; must otherwise be finite and > 0 (0, negative and Inf throw)
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
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)
- Each segment is computed by [get_channels_spherical](#get_channels_spherical): NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`
- A segment without paths (all its path inputs empty) gives empty outputs `[n_rx, n_tx, 0]`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- Optional `irs_array_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- Setting `center_frequency = 0.0` disables phase computation
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

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
- **`path_length_1`** — Total path lengths from TX to IRS phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `[n_path_1]`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `[n_path_2]`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `[8, n_path_2]`
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
- `M` accepts 8 rows (full polarimetric: the 2x2 Jones matrix, rows = RX polarisation V/H, columns = TX polarisation
  V/H, column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm
- Coupling matrices interpolated across frequencies per complex entry (SLERP for complex pairs), identical to antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 0 holds the LOS path
  (zero power if no LOS path was found); a LOS path found in the input is moved to slot 0 and leaves a zero-power path
  in its original slot (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]` per output frequency; with
  `add_fake_los_path = true` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [quantize_delays](#quantize_delays) rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [baseband_freq_response_multi](#baseband_freq_response_multi) with `remove_delay_phase = true` removes the phase of `delay` before it
  interpolates across `freq_in`, so it needs channels generated with `use_absolute_delays = true`; with relative
  delays, the phase of the TX-RX distance stays in the coefficients, is interpolated across the `freq_in` grid and
  gives wrong phases

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
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `[n_path]`
- **`M`** — Polarization transfer matrix, row order as above; `[8, n_path, n_freq_in]` (full pol) or
  `[2, n_path, n_freq_in]` (scalar)
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `[n_freq_in]`
- **`freq_out`** — Target output frequencies; `[n_freq_out]`
- **`use_absolute_delays`** *(optional)* — Include LOS delay in all paths if true
- **`add_fake_los_path`** *(optional)* — Always add one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot
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
- If `add_fake_los_path` is true, one output path is always added (output size `n_path+1`): slot 0 holds the LOS path,
  with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the TX-RX
  distance; it is moved to slot 0 and its original slot (input index + 1) holds a zero-power path. All other paths
  keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- Polarization coupling: `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation V/H),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- Setting `center_frequency = 0` disables phase calculation (delays still computed).
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance (between the phase centres) from all path
  lengths before converting to delay; per-element delays of pairs that are closer to each other than the phase
  centres are then negative, down to about -(half the TX array size + half the RX array size) / speed of light; [quantize_delays](#quantize_delays) rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.

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
- **`M`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`
- **`center_frequency`** *(optional)* — Center frequency; 0 disables phase calculation
- **`use_absolute_delays`** *(optional)* — Include LOS delay offset in all paths
- **`add_fake_los_path`** *(optional)* — Always add one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot

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
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [coord2path](#coord2path) or [qrt_file_read](#qrt_file_read) with more than two interactions) is added to the FBS-LBS segment. Delays and
  phases follow this length; the per-element offsets follow the FBS / LBS geometry (spherical waves).
- If `center_frequency == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [quantize_delays](#quantize_delays) maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == true`, one output path is always added (`n_path + 1` outputs): slot 0 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 0 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair.

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
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `[n_path]`
- **`M`** — Polarization transfer matrix, 2x2 Jones matrix column-major, Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx); `[8, n_path]`
- **`center_frequency`** *(optional)* — Center frequency; set to `0` to skip phase computation
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component
- **`add_fake_los_path`** *(optional)* — If `true`, always adds one output path; slot 0 holds the LOS path (zero power
  if none was found); a LOS path found in the input leaves a zero-power path in its original slot
- **`use_avx2`** *(optional)* — If `true`, use AVX2 for antenna interpolation; faster, but less accurate; ignored when not supported

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path(+1)]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path(+1)]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path(+1)]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path(+1)]`

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
- A data set without any finite value gets a NaN column in `cdf_per_set` and is left out of `cdf_avg`, `mu` and `sig`;
  if no set has finite values (possible only with user-supplied bins), `cdf_avg`, `mu` and `sig` are NaN
- If `bins` points to an empty vector, equally spaced bins spanning the data range are generated and stored back
  (data without any finite value then throws); if non-empty, those bin centers are used and must be finite and
  strictly ascending, else `std::invalid_argument` is thrown; if `nullptr`, bins are auto-generated internally

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
  used as-is if non-empty (finite, strictly ascending), ignored if `nullptr`; `[n_bins]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when non-empty bins are provided

### Outputs:
- **`cdf_per_set`** *(optional)* — Individual CDFs, one per column of data (NaN for a set without finite values); `[n_bins, n_sets]`
- **`cdf_avg`** *(optional)* — Averaged CDF via quantile-space averaging across the data sets with finite values; `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across the data sets with finite values, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values, `[9]`

---
## calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles; each CIR may have a different number of paths.
- RMS spread formula: `sqrt(sum(pw .* d^2))` where `d` are wrapped deviations from the circular mean (3GPP TR 38.901 second-moment definition).
- With wrapping (`disable_wrapping = false`), the mean direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies on the equator before computing spreads, avoiding pole singularity artifacts.
- When `calc_bank_angle = true` (with wrapping), an optimal bank angle maximizing azimuth spread is derived analytically from eigenvectors of the 2x2 power-weighted covariance matrix of centered angles.
- When `disable_wrapping = true` (the default), spreads are computed directly from raw angles; `orientation` will be zero and `phi`/`theta` equal the input `az`/`el`.
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed before computing spreads;
  `quantize` must be 0 or a finite value >= 1e-3 deg (negative, NaN, Inf and smaller positive values throw).

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
    bool disable_wrapping = true,
    bool calc_bank_angle = false,
    dtype quantize = (dtype)0);
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_cir]` vector, each element of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; same structure as `az`
- **`powers`** — Path powers in [W]; same structure as `az`
- **`disable_wrapping`** *(optional)* — If true, skips spherical rotation and computes spreads from raw angles; default:
  `true` (wrapping off, as in the MATLAB and Python wrappers)
- **`calc_bank_angle`** *(optional)* — If true, computes optimal bank angle analytically; only used when
  `disable_wrapping = false`; default: `false`
- **`quantize`** *(optional)* — Angular quantization step in [deg]; paths within this distance are grouped; 0 disables grouping;
  otherwise it must be finite and >= 1e-3 deg, else `std::invalid_argument` is thrown

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
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis.
- LOS paths are identified by comparing path length against direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default (`include_los = false`).
- Polarization transfer matrix `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with interleaved real/imaginary parts, 8 rows per path: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its
  receive and transmit polarisation: `M_vv` = rows 0-1, `M_hv` = rows 2-3, `M_vh` = rows 4-5, `M_hh` = rows 6-7
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `M` as [get_channels_spherical](#get_channels_spherical) couples antenna patterns,
  `c(r, t) = r^T M t` (plain transpose), with the same circular vectors at both ends, each in that antenna's own
  (e_theta, e_phi) frame: L = (V, H) = (1, +j) / sqrt(2) (LHCP), R = (1, -j) / sqrt(2) (RHCP). The names follow the
  IEEE definition (rotation seen along the propagation direction) for the time convention exp(+j 2 pi f t), which the
  library's coefficients exp(-j 2 pi f tau) imply, and e_theta x e_phi = the outward direction of the pattern.
  `M_<rx><tx>`: `M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2`, `M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2`,
  `M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2` (receive L, transmit R), `M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2`.
  Co-polar means the same circular vector at both ends: the LOS matrix `diag(1, -1)` of [get_channels_spherical](#get_channels_spherical) is
  co-polar (handedness kept), `diag(1, 1)` (one mirrored component, as by a single reflection) is cross-polar.
- LHCP-XPR and RHCP-XPR are ratios per receive polarisation, like V-XPR and H-XPR.
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
- **`window_size`** *(optional)* — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded
  when `include_los = false`

### Outputs:
- **`xpr`** *(optional)* — XPR on linear scale; `[n_cir, 6]`; columns:
   | Col | Description                                                     |
   | --- | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_vh)^2) (receive V)          |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_hv)^2) (receive H)          |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_LR)^2) (receive L)       |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_RL)^2) (receive R)       |
- **`pg`** *(optional)* — Total path gain summed over all paths (including LOS) as
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir]`

---
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; default threshold of 100 dB effectively includes all paths.
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their powers summed before computing the spread; function recurses on the binned profile.
  The bins are placed on a grid common to all CIRs; only occupied bins are stored, so memory does not grow with the delay span.
  All delays must then be finite, and the global delay span divided by `granularity` must be below 2^53.

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
- **`granularity`** *(optional)* — Bin width in [s] for grouping paths in the delay domain; 0 disables grouping; must be finite
  and non-negative; when > 0, non-finite delays or a span / granularity of 2^53 or more throw `std::invalid_argument`

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

# Geometry-based stochastic model

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the
*scenario* (e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The
functions of this section implement the large-scale part of the model of 3GPP TR 38.901, clause 7.5 steps 1 to 4:
path loss, shadow fading, LOS probability, O2I penetration loss and the large-scale parameters (LSPs) of each link,
i.e. the spreads of delays and angles, the K-factor and the cross-polarization ratio.

The statistics of a scenario are read from a conf file by [conf_file_read](#conf_file_read). The folder `data/config` holds 23
files for TR 38.901 v19.5.0; [scenpar_validate](#scenpar_validate) completes a parameter set from another source,
[conf_file_list](#conf_file_list) finds the files and [conf_file_write](#conf_file_write) writes them.

Per drop (layout): [o2i_loss](#o2i_loss) gives the penetration loss and the indoor distance of each indoor UT; then
[los_probability](#los_probability) evaluates the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS
state and the indoor state with [sos_rand](#sos_rand) / [sos_randn](#sos_randn) at the decorrelation distances of the keys
`LOS_state_lambda` and `IO_state_lambda`, and the O2I variables at the distances of TR 38.901 clause 7.6.3.3 (see
[o2i_loss](#o2i_loss)). The two state distances apply to all links of a scenario and are stated once in the TR 38.901 files:
`LOS_state_lambda` in the NLOS file, `IO_state_lambda` in the O2I files; 0 in the other files means "not stated
here". Per BS site and scenario (builder): [lsf_init_phases](#lsf_init_phases) draws the random
phases once; [lsf_draw](#lsf_draw) evaluates them at the links (columns 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR,
8 absolute TOA: Normal; 9 ground permittivity, 10 UMa environment height: Uniform; each at the decorrelation
distance of its key); [path_loss](#path_loss) returns the path loss and the SF standard deviation; [lsp_generate](#lsp_generate) turns the
random values into the LSPs of each link and frequency.

Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default; QuaDRiGa's behaviour is available
through an argument where it is still wanted (3D correlation, symmetric square root of the LSP
cross-correlation matrix, no clipping of the spreads, outdoor path-loss distance of O2I links). Where
QuaDRiGa's behaviour was an error, only the corrected rule is implemented.

Units: positions in m, frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, angular
spreads in degrees.

---
## conf_file_list
List the scenarios defined by the conf files in a set of directories

- Lists the regular files and symbolic links to regular files whose name ends in `.conf` (case-sensitive) and
  is longer than `.conf`, not recursive. The scenario name is the file name without this extension, whatever
  characters it has.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order of their
  names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first directory
  wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = true` each file is read and one entry is added right after the file for every
  line `ShortName = name` that is not a comment. Other lines are not examined; a UTF-8 byte order mark at the
  start of the file is skipped.
- Throws `std::invalid_argument` if a short name equals an earlier entry, if a file name equals an earlier
  short name, or if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`, `_`
- Throws `std::runtime_error` if a listed file cannot be opened for reading the short names
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty string is the
  current directory
- A failed call leaves the outputs untouched

### Declaration:
```
arma::uword quadriga_lib::conf_file_list(
    const std::vector<std::string> &dirs,
    std::vector<std::string> &scenarios,
    std::vector<std::string> *file_names = nullptr,
    std::vector<std::string> *file_dirs = nullptr,
    bool parse_shortnames = true);
```

### Inputs:
- **`dirs`** — Directories, searched in this order; can be empty
- **`parse_shortnames`** *(optional)* — `true`: read the files and add their short names; `false`: list the
  files only, which is faster

### Outputs:
- **`scenarios`** — Scenario names and short names; previous content is discarded; `[n_scen]`
- **`file_names`** *(optional)* — File name including `.conf` for each entry; `[n_scen]`
- **`file_dirs`** *(optional)* — Directory of the file for each entry, exactly as given in `dirs`; `[n_scen]`

### Returns:
- Number of entries, `n_scen`

### See also:
- [conf_file_read](#conf_file_read) (read one of the files)

---
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank lines
  are allowed; line ends can be LF or CRLF; space and tab are white space; a UTF-8 byte order mark at the start
  of the file is skipped
- Keys are case-sensitive names of the characters `A-Z`, `a-z`, `0-9` and `_`; there are four kinds:
  - Scenario parameters: `n_key = 171` numeric keys with one number each, returned in `scenpar`. Keys that
    are not in the file have their default value. The names and their order are returned in `key_order`.
  - `SubpathMethod`: one of the words `legacy`, `Laplacian`, `mmMAGIC`; default `legacy`
  - Path-loss parameters: every key that starts with `PL_`. `PL_model` has one name as value. All others
    have one or more numbers separated by white space and are returned in `plpar` without the prefix.
  - `ShortName`: an alias for the scenario, one name; can occur several times with different values
- Any other key is ignored and reported in `ignored_keys`; its value is not examined
- A key that is stated more than once has its last value
- Numbers are decimal with an optional sign, an optional decimal point and an optional exponent (`e` or
  `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted in any case
- A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the smallest
  subnormal reads as `0` (or `-0`)
- Numbers are read independent of the global locale (the decimal point is always `.`)
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 171 keys are 138 numeric keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys` like any other unknown key; the permittivity is complex and is set in the simulation
- The 28 correlation keys (`asD_ds` to `xpr_esa`) define the symmetric matrix `lsp_xcorr` with unit diagonal
- Throws `std::runtime_error` if the file cannot be opened or is a directory
- Throws `std::invalid_argument` with file name and line number for a line that is not blank, not a comment
  and not a well-formed assignment: no `=`, more or less than one name before the `=`, a missing value, a
  word where a number is required or the reverse, a second value or other text after the value, `NaN`,
  complex numbers, a number out of range, a `PL_` key without a name, a `ShortName` value stated twice
- With `validate = true`, a value that fails the checks of [scenpar_validate](#scenpar_validate) is an error as well
- A correlation matrix that is not positive definite is not an error; it is reported by the return value
- The file is parsed and checked completely before any output is written; a failed call leaves all
  outputs untouched

### Declaration:
```
bool quadriga_lib::conf_file_read(
    const std::string &fn,
    std::unordered_map<std::string, double> &scenpar,
    std::string *subpath_method = nullptr,
    std::string *pl_model = nullptr,
    std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    std::vector<std::string> *pl_order = nullptr,
    std::vector<std::string> *short_names = nullptr,
    std::vector<std::string> *ignored_keys = nullptr,
    bool validate = true);
```

### Inputs:
- **`fn`** — Path to the conf file
- **`validate`** *(optional)* — `true`: apply the value checks of [scenpar_validate](#scenpar_validate); `false`: only the line
  grammar is checked

### Outputs:
- **`scenpar`** — Numeric scenario parameters; previous content is discarded; `n_key = 171` entries
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`; `legacy` if not in the file
- **`pl_model`** *(optional)* — Value of `PL_model`; empty if not in the file
- **`plpar`** *(optional)* — Path-loss parameters; keys without the prefix `PL_`; `PL_model` is not included;
  empty if the file has no such key; `n_pl` entries with one or more values each
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 33 new keys; `[n_key]`
- **`pl_order`** *(optional)* — Keys of `plpar` in the order of their first occurrence in the file; `[n_pl]`
- **`short_names`** *(optional)* — Values of all `ShortName` lines in file order; `[n_short]`
- **`ignored_keys`** *(optional)* — Unknown keys, each once, in the order of their first occurrence; `[n_ignored]`

### Returns:
- `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be generated from it

### See also:
- [scenpar_validate](#scenpar_validate) (value checks, defaults and the list of validation rules)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)
- [lsf_init_phases](#lsf_init_phases) (first step of the large-scale model with the scenario parameters)
- [path_loss](#path_loss) (evaluates the path-loss model and its parameters)

---
## conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files shipped with
  QuaDRiGa, so that [conf_file_read](#conf_file_read) returns the same values bit for bit
- Header: `% Config file for scenario "name"` with the file name without the extension `.conf`, then the
  reference `ref`, the validity range and one line `ShortName = name` per short name
- Up to 10 sections follow, each with a header and, if the entry of `ref_sec` is not empty, a line
  `% See: ref; ref_sec`. Index in `ref_sec` and write order:
  - 0: Large scale distributions, with an overview formula per LSP; 1: Large-Scale fading decorrelation distances;
    2: Inter-parameter correlations, with the correlation matrix as comment; 3: Model parameters;
    8: EoD offset; 4: Decorrelation distance for the small-scale fading spatial consistency;
    5: Parameters for the ground reflection extension; 6: Absolute time of arrival offset;
    9: Optional modelling components; 7: Path-loss model
  - The indices 0 to 7 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- Within a section, keys are grouped into paragraphs separated by a blank line; correlations are written in the
  row order of the matrix, all other keys in the order of `key_order`; `SubpathMethod` stands before
  `LOS_scatter_radius`
- The overview formula of an LSP shows its mean and STD with the frequency, distance, height and elevation terms;
  with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`; the ESD line also has the terms of
  `ES_D_mu_A` to `ES_D_mu_D` and, if `ES_D_mu_min` is not `-Inf`, the lower limit of the mean; the DS line has
  the term `log10( a · V/S + b )` when `DS_VS > 0` (a = `DS_VS_a`, b = `DS_VS_b`; `V/S` stays symbolic)
- Each line is `name = value % comment`, with the name padded to 19 and the value to 13 characters. The comment
  states the unit; for the mean and STD of an LSP it adds the reference point of the frequency, distance, height
  and elevation terms, e.g. `@ 1 GHz`.
- The path-loss section has the formula blocks of the model (`logdist`, `dual_slope`, `nlos`, `tripple_slope`),
  the line `PL_model`, then the keys of `plpar` with the prefix `PL_`; known keys have a comment that can depend on
  the model; several values of one key are separated by a space
- Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back to the same
  double in [conf_file_read](#conf_file_read), independent of the global locale; `Inf` and `-Inf` are written as such, `-0` is
  kept
- With `write_defaults = false` a key is written only if its value differs from the default, compared bit for bit;
  `SubpathMethod` only if it is not `legacy`; a section without a written key is left out
- With `write_defaults = true` all 171 keys and `SubpathMethod` are written
- Validation before the file is opened: the file name has no line break; the value checks of [scenpar_validate](#scenpar_validate)
  (unknown keys, `NaN`, classes, `subpath_method`); `pl_model` empty or one name; `plpar` keys are names other
  than `model`, each with at least one value and no `NaN`; `pl_order` names each key of `plpar` once; short names
  are names and distinct; `ref` and `ref_sec` have no line break; `ref_sec` has at most 10 entries; `valid_range`
  is empty or `[4, 2]`
- The text is written to a temporary file in the same directory, which must be writable, and then renamed to
  `fn`; a symbolic link `fn` is written through
- A failed call does not create a file and leaves an existing file unchanged; an existing file is replaced and
  keeps its permissions
- Throws `std::invalid_argument` for invalid input; `std::runtime_error` if the file cannot be opened or written,
  or if `fn` exists and is not a regular file that can be opened for writing
- Every file written passes [conf_file_read](#conf_file_read) with `validate = true` and without ignored keys

### Declaration:
```
void quadriga_lib::conf_file_write(
    const std::string &fn,
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &subpath_method = "legacy",
    const std::string &pl_model = "",
    const std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,
    const std::vector<std::string> *pl_order = nullptr,
    const std::vector<std::string> *short_names = nullptr,
    const std::string &ref = "",
    const std::vector<std::string> *ref_sec = nullptr,
    const arma::mat *valid_range = nullptr,
    bool write_defaults = false);
```

### Inputs:
- **`fn`** — Path of the conf file to write
- **`scenpar`** — Numeric scenario parameters; any subset of the `n_key = 171` keys of [conf_file_read](#conf_file_read); missing
  keys have their default
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`: `legacy`, `Laplacian` or `mmMAGIC`
- **`pl_model`** *(optional)* — Value of `PL_model`; one name; empty: not written
- **`plpar`** *(optional)* — Path-loss parameters; keys without the prefix `PL_`, one or more values each; `[n_pl]`
- **`pl_order`** *(optional)* — Order in which the keys of `plpar` are written; must name each key once; omitted:
  sorted by name; `[n_pl]`
- **`short_names`** *(optional)* — Values of the `ShortName` lines, one name each
- **`ref`** *(optional)* — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; one line
- **`ref_sec`** *(optional)* — Reference per section, indexed as listed above; one line each; an empty or missing
  entry writes no reference; up to `n_sec = 10` entries
- **`valid_range`** *(optional)* — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, MT height in m; columns: lower and upper end, one value is written if both are equal; a row with `NaN`
  is not written; empty: no validity range; `[4, 2]`
- **`write_defaults`** *(optional)* — `true`: write all keys; `false`: write only keys that differ from their default

### See also:
- [conf_file_read](#conf_file_read) (read the file back)
- [scenpar_validate](#scenpar_validate) (value checks and defaults)

---
## los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

- Evaluates the probability that a link is in line-of-sight (LOS) for each link, from its 2D distance and the BS and
  UT heights, and optionally the soft LOS state of TR 38.901 eq. (7.6-18)
- Notation: `d = d2d_out(l)`, `hB = h_bs(l)`, `hU = h_ut(l)` in m, `lg = log10`, `ln` the natural logarithm. All
  results are limited to [0, 1]
- TR 38.901 v19.5.0 Table 7.4.2-1:
  - `38.901_RMa`: 1 for `d <= 10`, else `exp(-(d - 10)/1000)`
  - `38.901_UMi`: 1 for `d <= 18`, else `18/d + exp(-d/36)·(1 - 18/d)`
  - `38.901_UMa`: 1 for `d <= 18`, else `(18/d + exp(-d/63)·(1 - 18/d))·(1 + C'·5/4·(d/100)³·exp(-d/150))`
    with `C' = 0` for `hU <= 13`, `((hU - 13)/10)^1.5` for `13 < hU <= 23` and 1 for `hU > 23`; for `hU > 13`
    the formula exceeds 1 slightly just above 18 m (e.g. up to 1.0060 for `hU = 22.5`), where the limit gives 1
  - `38.901_InH_Mixed` (mixed office): 1 for `d <= 1.2`, `exp(-(d - 1.2)/4.7)` for `1.2 < d < 6.5`,
    `exp(-(d - 6.5)/32.6)·0.32` for `d >= 6.5`
  - `38.901_InH_Open` (open office): 1 for `d <= 5`, `exp(-(d - 5)/70.8)` for `5 < d <= 49`,
    `exp(-(d - 49)/211.7)·0.54` for `d > 49`
  - `38.901_InF_SL`, `38.901_InF_DL`: `exp(-d·kinv)` with `kinv = -ln(1 - r)/d_clutter`
  - `38.901_InF_SH`, `38.901_InF_DH`: with `kinv = -ln(1 - r)/d_clutter`: 1 for `hU >= h_c` (UT above the
    clutter); `exp(-d·kinv·(h_c - hU)/(hB - hU))` for `hB > hU` (TR: `k = -d_clutter/ln(1 - r)·(hB - hU)/(h_c - hU)`);
    `exp(-d·kinv)` (the rule of SL and DL) for `hB <= hU < h_c`, where the TR formula has a BS above the UT and
    would give `k <= 0`. `r = 0` gives 1
  - `38.901_InF_HH`: 1
  - `38.901_SMa` (TR v19): 1 for `d <= 10` or `hB <= hU`, else `exp(-d·(k_com + k_res + k_veg))` with
    `k_x = -ln(1 - r_x)/d_clutter·max(h_x - hU, 0)/(hB - hU)`, heights `h_com = 20`, `h_res = 8`, `h_veg = 15` m,
    densities `r_com = 0.02`, `r_res = 0.18`, `r_veg = r_clutter`; the TR writes `max(., 0)` only for the
    residential term, which gives the same values for `hU < 15` m (the TR range)
- Defaults of the clutter arguments (NaN), TR 38.901 Table 7.2-4 and Table 7.8-7 for InF, Table 7.4.2-1 for SMa:
  - `InF_SL`: `r = 0.2`, `d_clutter = 10` m; `InF_DL`: `r = 0.6`, `d_clutter = 2` m
  - `InF_SH`: `r = 0.2`, `d_clutter = 10` m, `h_c = 2` m; `InF_DH`: `r = 0.6`, `d_clutter = 2` m, `h_c = 6` m
  - `SMa`: `d_clutter = 30` m, `r_veg = 0` (no vegetation; 0.1 sparse, 0.2 dense vegetation); `h_clutter` is not used
  - The other models ignore the clutter arguments (they are validated for every model)
- TR 38.811 (via QuaDRiGa), `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`: the LOS
  probability in percent over the elevation `atan2(hB, d)` in degrees (absolute BS height), linear interpolation
  over the grid -90, 0, 10, ..., 90 deg, times 0.01:
  - DenseUrban: 0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1
  - Urban: 0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2
  - Suburban and Rural: 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8
- QuaDRiGa, `QuaDRiGa_Industrial`: `min(1, exp(-(d + 20)/80)·exp(min(hB, 8)/5 - 1))`
- `LOS`: 1, `NLOS`: 0
- Soft LOS state (TR 38.901 eq. 7.6-18), computed when `los_soft` is requested:
  `los_soft(l, k) = 1/2 + atan(sqrt(20/lambda_k)·(rnd_g(l) + F(p_los(l))))/pi`, `lambda_k = c/fc(k)` with
  `c = 299792458` m/s and `F(p) = sqrt(2)·erfinv(2p - 1)` the inverse of the standard normal CDF (accurate to
  `1e-10·max(1, abs(F))` for `p` in [1e-12, 1 - 1e-12]). `p = 1` gives 1, `p = 0` gives 0. `los_soft > 0.5`
  exactly when `rnd_g > -F(p)`, which has the probability `p` for `rnd_g` from N(0,1)
- The LOS state itself (a draw against `p_los`) and its spatial correlation are the caller's: e.g. a spatially
  consistent uniform value from [sos_rand](#sos_rand), or `rnd_g` from [sos_randn](#sos_randn), at the decorrelation distance of the
  scenario key `LOS_state_lambda` (TR 38.901 Table 7.6.3.1-2). The key applies to all links of a scenario and is
  stated only in its NLOS conf file (0 in the LOS and O2I files means "not stated here"). The `LOS_state_lambda` of
  the InF conf files is `d_clutter / 2` of the TR defaults; it does not follow a `d_clutter` argument, so a caller
  who changes `d_clutter` passes its own correlation distance
- O2I UTs: Table 7.4.2-1 uses the outdoor distance `d2d_out = d2D - d2D-in` with `d2D-in` from [o2i_loss](#o2i_loss), which
  therefore runs first; the caller clamps `d2d_out` at 0 (negative distances are rejected). Indoor scenarios
  (InH, InF) pass the 2D distance
- Outputs must not share memory with the inputs or with each other
- Validation, in this order: the model name; `d2d_out` finite and >= 0; `h_bs` and `h_ut` with 1 or `n_link`
  elements and finite; `d_clutter` NaN or finite and > 0, `r_clutter` NaN or in [0, 1), `h_clutter` NaN or finite
  and >= 0 (for every model); if `los_soft` is requested: `rnd_g` and `fc` present, `rnd_g` with `n_link`
  finite elements, `fc` non-empty, finite and > 0. Without `los_soft`, `rnd_g` and `fc` are ignored. All checks
  run before any output is written

### Declaration:
```
void quadriga_lib::los_probability(
    const std::string &model,
    const arma::vec &d2d_out,
    const arma::vec &h_bs,
    const arma::vec &h_ut,
    arma::vec &p_los,
    double d_clutter = NAN,
    double r_clutter = NAN,
    double h_clutter = NAN,
    const arma::vec *rnd_g = nullptr,
    const arma::vec *fc = nullptr,
    arma::mat *los_soft = nullptr);
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in [m] (indoor scenarios: 2D distance), finite and >= 0; `[n_link]`, 0 allowed
- **`h_bs`** — BS height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`h_ut`** — UT height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`d_clutter`** *(optional)* — Clutter size in [m] of the InF and SMa models; NaN = TR default of the model
- **`r_clutter`** *(optional)* — Clutter density of the InF models, vegetation density `r_veg` of SMa, in [0, 1);
  NaN = default
- **`h_clutter`** *(optional)* — Clutter height in [m] of InF-SH and InF-DH; NaN = default
- **`rnd_g`** *(optional)* — Normal(0,1) value per link for the soft LOS state (G of eq. 7.6-18); required with
  `los_soft`; `[n_link]`
- **`fc`** *(optional)* — Carrier frequencies in [Hz] for the soft LOS state; required with `los_soft`; `[n_freq]`

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link]`
- **`los_soft`** *(optional)* — Soft LOS state in [0, 1]; `[n_link, n_freq]`

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos_rand](#sos_rand), [sos_randn](#sos_randn) (spatially consistent LOS-state draw and `rnd_g`)
- [conf_file_read](#conf_file_read) (key `LOS_state_lambda`)

---
## lsf_draw
Spatially correlated normalized random values of the large-scale generators at a set of links

- Evaluates the 11 large-scale fading generators of [lsf_init_phases](#lsf_init_phases) at the links: column 0 DS, 1 KF, 2 SF,
  3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal(0,1)), 9 ground permittivity, 10 UMa environment height
  hE (Uniform(0,1)); each generator with the sum-of-sinusoids frequencies of `preset` at its own decorrelation
  distance (a distance of 0 is used as 0.1 m)
- `scenpar` is completed and validated by [scenpar_validate](#scenpar_validate) as in [lsf_init_phases](#lsf_init_phases); its messages pass
  through unchanged
- Single mobility (`dual_mobility = false`): column g is the field of generator g at the Rx position,
  `x_g = s_g(rx)` with `s_g(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and the phases of slice 0;
  the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 1, i.e. the field of the mirror generator `m(g)`
  at the Tx position (ASD and ASA, ESD and ESA exchanged). Self-mirrored generators (DS, KF, SF, XPR, TOA, 9, 10)
  share one map: `x_g = ( s_g(rx) + s_g(tx) ) / sqrt( 2 + 2·A_g(D) )`, with the Tx-Rx distance `D` and the ACF
  `A_g` of the preset, linearly interpolated. The pairs are two maps: `x_g = ( s_g(rx) + s_m(tx) ) / sqrt(2)`.
  Columns 3 to 6 thus stay Normal(0,1), and ASD/ASA and ESD/ESA have the correlation
  `( A_g(D) + A_m(D) ) / 2` at the Tx-Rx distance, 1 at `D = 0` (then the two columns are equal)
- Swap reciprocity in dual mobility: swapping `tx_pos` and `rx_pos` leaves columns 0, 1, 2, 7, 8, 9, 10
  unchanged and swaps column 3 with 4 and column 5 with 6, exactly (the same evaluations in swapped roles), also
  for unequal decorrelation distances and with bias-removed phases, when both calls use the same phase cube
- Uniform columns 9 and 10: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = true`: the positions are projected on the x-y plane (z = 0) and the values do not depend on z;
  `false`: 3D positions. `D` is measured in the same space
- 2D and 3D mode give the same per-link statistics. In 3D mode the height separates the floors: the LSPs of UTs on
  different floors are correlated by the ACF of their height difference, while TR 38.901 (Table 7.5-6 note 3)
  assumes no correlation between floors. `floor_id` is therefore not accepted with `horizontal = false`
- Floors: links with different floor IDs see different fields, links with equal IDs share one field. Floor `q`
  adds the offsets `2·pi · k · n / N + psi_b(n)` to the phases of all generators on both sides, with
  `N = n_coeff`, `k = q mod N`, `b = q div N`, `n = 0 ... N-1`, `psi_0 = 0` and `psi_b(n) = 2·pi · u_n` for
  `b >= 1`, where `u_n` are the uniform numbers of a `std::mt19937_64` seeded with `b` (the upper 53 bits of each
  raw output times 2^-53). Floor 0 and `floor_id = nullptr` give the same values
- Fields of two floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3): exactly for
  two IDs in the same block of `N` IDs, with a random residual of standard deviation `1/sqrt(2N)` for IDs in
  different blocks (0.044 for 256 sinusoids). At other horizontal lags the cross-correlation of two floors is not
  zero (up to about 0.16 for 256 sinusoids), the same level as for random offsets. Mirror relations and swap
  reciprocity hold per floor
- `floor_id` is the caller's floor index of the UT: for the UT heights `hUT = 3·(n_fl - 1) + 1.5` of TR 38.901
  Table 7.2-1 it is `n_fl - 1`; outdoor UTs use 0
- Dual mobility with floors: `floor_id` is one value per link and its offsets are added on both sides, i.e. both
  devices of a link are on the link's floor. A Tx device on another floor than its Rx (e.g. a D2D link between two
  floors) is evaluated on the floor of the Rx, so the same Tx device sees different fields on links to Rx devices
  on different floors
- The value of a link does not depend on the other links of the call
- Precision: the sinusoids are evaluated by [sos_val](#sos_val) with the AVX2 kernel where available (sine and cosine in
  single precision, accumulation in double); the values deviate from a double-precision evaluation by less than
  `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [lsf_init_phases](#lsf_init_phases). A wrong preset with the same number of sinusoids
  (e.g. `Exp256` for `Comb256`) cannot be detected
- `n_link = 0` returns `rnd` with size `[0, 11]`. `rnd` may share memory with `tx_pos` or `rx_pos` (the positions
  are copied before `rnd` is written)
- Validation order: positions (3 columns, rows of `tx_pos`, finite values), `preset`, `scenpar`, size of `phase`,
  finite `phase`, length of `floor_id`, `floor_id` with `horizontal = false`

### Declaration:
```
void quadriga_lib::lsf_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    const arma::uvec *floor_id = nullptr,
    bool dual_mobility = false,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters as returned by [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate); missing keys
  have their default value; only the 10 decorrelation distances are used
- **`phase`** — Phases in [rad] from [lsf_init_phases](#lsf_init_phases); slice 0 = Rx side, slice 1 = Tx side; must be finite;
  `[n_coeff, 11, 2]`, `n_coeff` of `preset`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; not used in single mobility (but validated);
  `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`floor_id`** *(optional)* — Floor index per link, 0-based; omitted = all links on floor 0; in dual mobility
  the floor of both devices; only with `horizontal = true`; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move, the Tx side is evaluated as a second device
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); must match [lsf_init_phases](#lsf_init_phases)
- **`horizontal`** *(optional)* — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); must match [lsf_init_phases](#lsf_init_phases)

### Output:
- **`rnd`** — Random values; columns 0 to 8 Normal(0,1), columns 9 and 10 Uniform(0,1); `[n_link, 11]`

### See also:
- [lsf_init_phases](#lsf_init_phases) (draws the phases)
- [path_loss](#path_loss) (column 10 as `rnd_he`)
- [lsp_generate](#lsp_generate) (the large-scale parameters from `rnd`)
- [sos_val](#sos_val) (evaluation of a sum-of-sinusoids generator)

---
## lsf_init_phases
Draw the random phases of the large-scale generators of one BS site and scenario

- The large-scale fading (LSF) of a site is described by 11 spatially correlated sum-of-sinusoids generators, one
  column each: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal fields), 9 ground
  permittivity, 10 UMa environment height hE (Uniform fields). Their decorrelation distances are the scenario
  parameters `DS_lambda`, `KF_lambda`, `SF_lambda`, `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`,
  `XPR_lambda`, `absTOA_lambda`, `SC_lambda` and again `SF_lambda` for hE; a decorrelation distance of 0 is used
  as 0.1 m. The generator of column 8 exists also when the absolute TOA is disabled
- This function draws the phases; [lsf_draw](#lsf_draw) evaluates them at the links. The caller stores `phase` and
  evaluates the same fields again along tracks or on maps
- `scenpar` is completed with the defaults of missing keys and validated by [scenpar_validate](#scenpar_validate); its messages
  (e.g. for an unknown key or a negative decorrelation distance) pass through unchanged. An invalid value of a key
  that is not used here throws as well
- Slice 0 (Rx side) holds the phases of the 11 generators. Without bias removal it equals
  `sos_init_phases(n_coeff, 11, seed)` of [sos_init_phases](#sos_init_phases), with `n_coeff` the number of sinusoids of
  `preset`. The same seed gives the same phases
- Slice 1 (Tx side) is always the mirror of slice 0: `slice1.col(g) = slice0.col(m(g))` with the mirror
  generator `m(g)`: ASD and ASA (3, 4) and ESD and ESA (5, 6) are exchanged, all others mirror themselves. For dual
  mobility the departure spreads at the Tx are thus the arrival spreads of the Tx as a second device
- Bias removal (`remove_bias = true`, `rx_pos` given, `n_link > 2`): one realisation of a correlated field has a
  non-zero mean over a drop. For each Normal generator (columns 0 to 8) one common phase offset is applied with
  [sos_remove_bias](#sos_remove_bias), so that the sample mean of its field over the positions is exactly 0. Single mobility: over
  the Rx positions; `tx_pos` is not used. Dual mobility: over the Rx and Tx positions together (`2·n_link`
  positions), because the field of a generator is shared by both devices (the Tx side of ASD is the ASA field);
  `tx_pos` is then required. The Uniform generators 9 and 10 are never corrected
- The correction is applied to slice 0 and the Tx side is mirrored afterwards, so the mirror relation holds
  exactly with and without bias removal. Swapping `tx_pos` and `rx_pos` in dual mobility gives the same phases up to
  rounding (the sums over the positions are formed in another order)
- Consequences for [lsf_draw](#lsf_draw) at the same positions: in single mobility the link mean of columns 0 to 8 is 0; in
  dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have the same Tx-Rx distance (the
  normalisation is then the same for all links), while ASD/ASA and ESD/ESA have a zero link mean only
  approximately (their two devices see different fields, each corrected over the union of the positions)
- The positions are used as in [lsf_draw](#lsf_draw): projected on the x-y plane (z = 0) for `horizontal = true`, in 3D
  for `horizontal = false`. `preset` and `horizontal` must be the same in [lsf_draw](#lsf_draw)
- 2D mode: all positions in `rx_pos` are treated as one field (floor 0): one phase correction per generator cannot
  zero the means of several floor fields at once. For a drop with UTs on several floors pass the positions of one
  floor (e.g. the links of floor 0) or set `remove_bias = false`; the fields of the other floors keep their sample
  mean. 3D mode (`horizontal = false`): there are no floor IDs (the height separates the floors) and the correction
  covers all given positions as one 3D field
- 2D and 3D mode give the same per-link statistics. In 3D mode the LSPs of UTs on different floors are correlated by
  the ACF of their height difference, while TR 38.901 (Table 7.5-6 note 3) assumes no correlation between floors
- The correction removes variance: the field is conditioned on a zero mean over the positions, which costs little
  only when the positions spread over many decorrelation distances (see [sos_remove_bias](#sos_remove_bias))
- With fewer than 3 positions, without `rx_pos` or with `remove_bias = false` no correction is made
- TR 38.901 clause 7.4.4 literally (exponential ACF, no bias removal): `preset = "Exp256"` and
  `remove_bias = false`; the defaults `Comb256` and the bias removal are QuaDRiGa choices
- Reason for `Comb256`: the TR gives the exponential ACF (eq. 7.4-5) as a fit "with sufficient accuracy",
  not as an exact requirement; its kink at zero distance makes the values evolve roughly along a track.
  `Comb256` (Gaussian near zero, exponential tail, same decorrelation distance) evolves smoothly
- `phase` must not share memory with `tx_pos` or `rx_pos`
- Validation order: `seed`, `preset`, `scenpar`, then the positions (3 columns, rows of `tx_pos` 1 or `n_link`
  when both are given, finite values), then the requirement of `tx_pos` for the bias removal in dual mobility

### Declaration:
```
void quadriga_lib::lsf_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    arma::cube &phase,
    arma::sword seed = -1,
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    bool dual_mobility = false,
    bool remove_bias = true,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters as returned by [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate); missing keys
  have their default value; only the 10 decorrelation distances are used
- **`seed`** *(optional)* — Seed of the random phases, must be >= -1; -1 = system random device, a new set of
  phases on each call
- **`tx_pos`** *(optional)* — Tx positions in [m]; used only for the bias removal with dual mobility, required
  there; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** *(optional)* — Rx positions of the drop in [m] (one floor in 2D mode, all floors in 3D mode) for
  the bias removal; omitted or fewer than 3 rows = no bias removal; `[n_link, 3]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move, the Tx is a second device on the same maps
- **`remove_bias`** *(optional)* — `true`: remove the sample mean of the Normal fields over the positions
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); sets `n_coeff`
- **`horizontal`** *(optional)* — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour);
  `false`: 3D correlation (legacy QuaDRiGa behaviour)

### Output:
- **`phase`** — Phases in [rad]; slice 0 = Rx side, slice 1 = Tx side (mirror of slice 0); all values in
  [-pi, pi); `[n_coeff, 11, 2]`

### See also:
- [conf_file_read](#conf_file_read) (the scenario parameters)
- [lsf_draw](#lsf_draw) (evaluates the phases at the links)
- [sos_init_phases](#sos_init_phases) (the raw phases)
- [sos_remove_bias](#sos_remove_bias) (the bias removal of one generator)

---
## lsp_generate
Large-scale parameters of a set of links from normalized random values

- Generates the 8 large-scale parameters (LSPs) of TR 38.901 clause 7.5 step 4 per link and frequency: delay
  spread DS, Ricean K-factor KF, shadow fading SF, azimuth spreads of departure and arrival ASD, ASA, elevation
  spreads of departure and arrival ESD, ESA and cross-polarization ratio XPR. The random values `rnd` are the
  normalized, spatially correlated values of [lsf_draw](#lsf_draw) (columns 0 to 7 Normal(0,1) for the 8 LSPs, column 8
  Normal(0,1) for the absolute TOA, column 9 Uniform(0,1) for the ground permittivity; column 10 is not used here)
- `scenpar` is completed with the defaults of missing keys and validated by [scenpar_validate](#scenpar_validate); its messages pass
  through unchanged. An invalid value of a key that is not used here throws as well
- Mean and standard deviation in the log domain, per link `l` and frequency `k`, with the key prefix `P` of the
  LSP (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), the 2D distance `d2D`, the Tx height
  `hBS = tx_pos.z` and the Rx height `hUT = rx_pos.z`:
  `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h / d)` (elevation in rad),
  `f' = max(fc / 1e9, LSP_fc_min)` (frequency in GHz, limited from below by `LSP_fc_min` for all LSP terms);
  `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`;
  `s = max(0, P_sigma + P_delta·lg(P_omega + f'))`;
  `sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`. The standard deviation is limited at 0 after
  all terms, since a negative value would flip the sign of the realization and its cross-correlations. SF
  has no mean keys (mean 0). A frequency term with the coefficient 0 contributes 0; an active one
  (`P_gamma` or `P_delta` not 0) requires `P_omega + f' > 0` at every frequency. The multiplying coefficients
  (`P_sigma`, `P_gamma`, `P_delta`, `P_epsilon`, `P_zeta`, `P_alpha`, `P_kappa`, `P_tau`, `P_beta`,
  `ES_D_mu_A` to `ES_D_mu_D`, `P_omega` of an active frequency term) must be finite; `P_mu` and `ES_D_mu_min` may
  be infinite
- DS with the V/S term of TR 38.901 Table 7.5-6 Part-3 (InF): if `DS_VS > 0`,
  `mu += lg(DS_VS_a·DS_VS + DS_VS_b)`. The InF files set `DS_VS` for the hall of their calibration scenario; a
  caller with another hall sets `DS_VS` (V/S in m) in its copy of `scenpar`
- ESD with the terms of TR 38.901 Tables 7.5-7 to 7.5-12: `mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT +
  ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)`, then the floor `mu = max(mu, ES_D_mu_min)`
- Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF and XPR in dB
- `sf_sigma` (e.g. from the path-loss model) replaces the SF standard deviation per link and frequency, also in
  `lsp_sigma`; it must be finite and >= 0 (a negative value would flip the sign of the shadow fading)
- Cross-correlation: `x = Q·s` with `s = rnd(l, 0..7)` and a root `Q` of the LSP cross-correlation matrix `R` of
  `scenpar` (`Q·Q^T = R`). `cholesky = true` (TR 38.901 clause 7.5 step 4): Cholesky factorization in the TR order
  [SF, KF, DS, ASD, ASA, ESD, ESA], XPR appended last; the SF value is passed through (`x_SF = s_SF`).
  `cholesky = false`: the symmetric square root of `R` (principal root, legacy QuaDRiGa behaviour); it does not
  depend on the order of the parameters. Both throw if `R` is not positive definite
- The same `x` is used for all frequencies of a link (TR 38.901 clause 7.6.5)
- Linear values: `lsp = 10^(mu + sigma·x)` for DS, ASD, ASA, ESD, ESA; `10^(0.1·(mu + sigma·x))` for KF, SF, XPR.
  `clip_spreads = true` (TR 38.901 clause 7.5 step 4): ASD and ASA are limited to 104 deg, ESD and ESA to 52 deg;
  `lsp_mu` and `lsp_sigma` are not affected
- `rnd = zeros(n_link, 11)` gives the median LSPs; `lsp_mu` and `lsp_sigma` are the mean and std without any
  randomness
- Absolute TOA offset (TR 38.901 clause 7.6.9): active if `absTOA_mu > -30`, then
  `abs_toa = min(10^(absTOA_mu + absTOA_sigma·rnd(l, 8)), 2·absTOA_Lmax / c)` with `c = 299792458 m/s`, without
  the bound if `absTOA_Lmax = 0` (`absTOA_sigma` must be finite); 0 if not active. The InF and InH NLOS files set
  `absTOA_Lmax` to the largest dimension of the hall or office of their calibration scenario; a caller with another
  hall sets the key; in the TR 38.901 files the offset is active in all NLOS and O2I files (an O2I link
  has no LOS ray) and off in the LOS files
- Ground permittivity with `U = rnd(l, 9)` and `f = fc / 1e9` (not limited by `LSP_fc_min`): dry ground
  `3 + j·0.003·f^1.34`, medium `30.4·f^-0.47 + j·0.18·f^1.05`, wet `31.3·f^-0.48 + j·0.63·f^0.77`; linear
  interpolation from dry (`U = 0`) to medium (`U = 0.5`) to wet (`U = 1`). The imaginary part is >= 0;
  TR 38.901 eq. (7.6-40) writes the complex permittivity with the opposite sign of the imaginary part, so the TR
  reflection formulas take the complex conjugate `gr_eps_re - j·gr_eps_im`. Column 9 of `rnd` must be in [0, 1]
  when `gr_eps_re` or `gr_eps_im` is requested; it is not checked otherwise
- The three ground types are QuaDRiGa's fits (as in QuaDRiGa's `gen_lsf_parameters`), not the models of TR 38.901
  Table 7.6.8-1: very dry ground agrees, medium and wet ground differ (e.g. at 3.5 GHz medium `16.87 + j·0.67` here,
  `13.23 + j·1.39` for medium dry ground of the table; wet `j·1.65` vs. `j·3.93`). A caller who needs the table
  values computes them with eq. (7.6-40) to (7.6-42)
- EoD offset in degrees, positive = up (the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12), with
  `f'`, `d`, `h` as above: `e = EoD_offset_e + EoD_offset_e_gamma·lg f'`; if `EoD_offset_b > 0`:
  `e += 10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)`; if `EoD_offset_hcl > 0` and `hUT < EoD_offset_hcl`:
  `e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi`. The coefficients `EoD_offset_e`, `_e_gamma`,
  `_a`, `_a_gamma`, `_c`, `_c_gamma`, `_c_hUT` must be finite
- Optional outputs that are not requested are not computed. `n_link = 0` gives outputs with 0 rows
- Outputs must not share memory with the inputs or with each other
- Validation order: positions, `fc`, `rnd`, `sf_sigma`, `scenpar`, positive definiteness, column 9 of `rnd`, the
  coefficients and frequency terms of the 8 LSPs, the `EoD_offset_` coefficients (if `eod_offset_deg` is
  requested) and `absTOA_sigma` (if `abs_toa` is requested and active) finite; all checks run before any output is
  written

### Declaration:
```
void quadriga_lib::lsp_generate(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &fc,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::mat &rnd,
    arma::cube &lsp,
    const arma::mat *sf_sigma = nullptr,
    arma::vec *abs_toa = nullptr,
    arma::mat *gr_eps_re = nullptr,
    arma::mat *gr_eps_im = nullptr,
    arma::mat *eod_offset_deg = nullptr,
    arma::cube *lsp_mu = nullptr,
    arma::cube *lsp_sigma = nullptr,
    bool cholesky = true,
    bool clip_spreads = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters as returned by [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate); missing keys
  have their default value; the LSP cross-correlation matrix must be positive definite
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Normalized random values from [lsf_draw](#lsf_draw), finite; columns 0 to 7 for the 8 LSPs, column 8 for the
  absolute TOA, column 9 in [0, 1] for the ground permittivity, column 10 not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB], finite and >= 0; replaces the value from
  `scenpar`; omitted = the scenario value; `[n_link, n_freq]`
- **`cholesky`** *(optional)* — `true`: Cholesky root in the TR order; `false`: symmetric square root (legacy)
- **`clip_spreads`** *(optional)* — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in [s], 1 KF (linear), 2 SF (linear power ratio), 3 ASD,
  4 ASA, 5 ESD, 6 ESA in [deg], 7 XPR (linear); `[n_link, n_freq, 8]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s], 0 if disabled; `[n_link]`
- **`gr_eps_re`** *(optional)* — Relative permittivity of the ground, real part; `[n_link, n_freq]`
- **`gr_eps_im`** *(optional)* — Relative permittivity of the ground, imaginary part, >= 0; `[n_link, n_freq]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** *(optional)* — Mean in the log domain (lg for DS, ASD, ASA, ESD, ESA; dB for KF, SF, XPR);
  `[n_link, n_freq, 8]`
- **`lsp_sigma`** *(optional)* — Standard deviation in the log domain, >= 0; `[n_link, n_freq, 8]`

### See also:
- [lsf_draw](#lsf_draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [lsf_init_phases](#lsf_init_phases) (the phases of the random generators)
- [scenpar_validate](#scenpar_validate) (the scenario parameters and the cross-correlation matrix)

---
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency,
  from four normalized random values per UT; `f = fc / 1e9` in GHz, `lg = log10`
- Material losses of TR 38.901 v19.5.0 Table 7.4.3-1 in dB: `L_glass = 2 + 0.2 f`, `L_IRR = 25.4 + 0.11 f` (IRR
  glass, Release 19 value), `L_concrete = 5 + 4 f`, `L_plywood = 1.03 + 0.17 f`; external wall
  `tw(p1, L1, p2, L2) = 5 - 10·lg(p1·10^(-L1/10) + p2·10^(-L2/10))`
- Models, with `N = rnd(u, 3)` and the indoor 2D distance `d2din`:
  - `low` (low-loss, Table 7.4.3-2): `tw(0.3, L_glass, 0.7, L_concrete) + 0.5·d2din + 4.4·N`
  - `high` (high-loss, Table 7.4.3-2): `tw(0.7, L_IRR, 0.3, L_concrete) + 0.5·d2din + 6.5·N`
  - `low_A` (low-loss A, Table 7.4.3-2): `tw(0.3, L_glass, 0.7, L_plywood) + 0.5·d2din + 4.4·N`
  - `legacy_6GHz` (backward compatible model for single-frequency simulations below 6 GHz, Table 7.4.3-3):
    `20 + 0.5·d2din`, no deviation; it goes with a shadow-fading std of 7 dB (the O2I conf files of UMa and UMi
    have `SF_sigma = 7`)
  - `car` (clause 7.4.3.2): `9 + 5·N`; `car_metal` (metallized car windows): `20 + 5·N`; no indoor distance
- Columns of `rnd`: 0 = Uniform(0,1) for the model choice, 1 and 2 = Uniform(0,1) for the indoor distance,
  3 = Normal(0,1) for the deviation (sigma_P of Table 7.4.3-2, the car loss)
- Model choice per UT: `Ccum_j = (w_0 + ... + w_j)/sum(w)`; the model is the smallest `j` with
  `rnd(u, 0) < Ccum_j`, or, if there is none, the last `j` with `w_j > 0`. A model with weight 0 is never chosen
- Indoor distance: `low`, `high`, `low_A`: `d2din = min(rnd(u, 1), rnd(u, 2))·d2d_in_max`, the minimum of two
  uniforms of clause 7.4.3.1; `legacy_6GHz`: `d2din = rnd(u, 1)·d2d_in_max` (one uniform, Table 7.4.3-3);
  `car`, `car_metal`: 0
- The caller draws `rnd` with [sos_rand](#sos_rand) and [sos_randn](#sos_randn) at the correlation distances of TR 38.901 clause
  7.6.3.3 (building type 50 m for column 0, d2D-in 25 m for columns 1 and 2, sigma_P 10 m for column 3) and adds
  `loss` to the path loss of the indoor UT. `d2d_in` gives the outdoor distance `d2D - d2D-in` for
  [los_probability](#los_probability) (call `o2i_loss` first) and is the input of the legacy outdoor distance of [path_loss](#path_loss)
- Applicability (Table 7.4.3-2): RMa uses `low` only, InF `high` only; UMa, UMi: `low` and `high`; SMa: `low`,
  `high` and `low_A`. TR maximum indoor distance: 25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and
  SMa residential buildings (conf-file key `O2I_d2D_in_max`). The TR SMa layout has 90 % residential and 10 %
  commercial buildings (Table 7.2-5): split the UTs by building type and call `o2i_loss` once per type
- Outputs must not share memory with the inputs or with each other
- Validation, in this order: `fc` non-empty, finite and > 0; `rnd` with 4 columns, columns 0 to 2 in [0, 1],
  column 3 finite; `models` non-empty, each a known name; `weights` with one finite element >= 0 per model and a
  sum of 1 within 1e-6; `d2d_in_max` finite and >= 0. All checks run before any output is written

### Declaration:
```
void quadriga_lib::o2i_loss(
    const arma::vec &fc,
    const arma::mat &rnd,
    const std::vector<std::string> &models,
    const arma::vec &weights,
    arma::mat &loss,
    double d2d_in_max = 25.0,
    arma::vec *d2d_in = nullptr,
    arma::uvec *model_index = nullptr);
```

### Inputs:
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`rnd`** — Random values per UT: Uniform(0,1) model choice, Uniform(0,1) d2D-in #1, Uniform(0,1) d2D-in #2,
  Normal(0,1) deviation; `[n_ut, 4]`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; `[n_model]`
- **`weights`** — Probability of each model, >= 0, sum 1; `[n_model]`
- **`d2d_in_max`** *(optional)* — Maximum indoor 2D distance in [m], finite and >= 0 (conf-file key
  `O2I_d2D_in_max`)

### Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m], 0 for the car models; `[n_ut]`
- **`model_index`** *(optional)* — Chosen model per UT, 0-based index into `models`; `[n_ut]`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)
- [sos_rand](#sos_rand), [sos_randn](#sos_randn) (spatially consistent random values for `rnd`)

---
## path_loss
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier
  frequency. `pl_model` and `plpar` are the values of `PL_model` and the `PL_` keys (without the prefix) as returned
  by [conf_file_read](#conf_file_read); the TR 38.901 v19.5.0 conf files in `data/config` reproduce Table 7.4.1-1 with them
- Notation: `f = fc / 1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance in m; `hBS`,
  `hUT` the z coordinates of Tx and Rx (no ground model); `[X]` the value of key `X`, or its default if the key is
  missing (`X = v` in the lists below); `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std
- Models, required keys and defaults:
  - `constant` (A): `pl = A`
  - `logdist` (A, B, C; SF optional): `pl = A·lg(d3) + B + C·lg(f)`; `sf = [SF]` if present
  - `logdist_simple` (A, B): `pl = A·lg(d3) + B`
  - `winner_los` (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`,
    `hm = max(hUT, 1.5)` of each link; `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`,
    `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`, breakpoint `bp = 10^((H - G)/(A1 - A2))`;
    `d2 <= bp`: `pl = A1·lg d2 + G + F1·hm`, `sf = sig1`; else `pl = A2·lg d2 + H + F2·hm`, `sf = sig2`
    (continuous at `bp` for `F1 = F2`); `A1` and `A2` must differ
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`;
    `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm`; `pl = L_1` for `f < 1.5`,
    `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`
  - `winner_pathloss` (A, B, C, X; SF optional): `pl = A·lg d3 + B + C·lg(f/5) + X`; `sf = [SF]` if present
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2 optional):
    `hb = max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`; `dBP = E·f·(hb - hE)·(hm - hE)`;
    `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]`; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = [sig2]`
    (each only if present), with the 3D breakpoint distance `dBP3 = sqrt(dBP² + (hb - hm)²)`. This is Table 7.4.1-1
    for UMa, UMi, InH and InF; for RMa and SMa LOS the TR uses the 2D breakpoint distance in PL2, which differs by
    at most 0.034 dB over the TR ranges
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3
    optional): `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)`, `hm` the same; `dBPj = Ej·f·(hb - hEj)·(hm - hEj)`,
    `dBP1 = min(dBP1, dBP2)`, `dBPj3 = sqrt(dBPj² + (hb - hm)²)`; `LBP1 = PL1(dBP13)`,
    `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`; `d2 <= dBP1`: `PL1(d3)`, `sf = [sig1]`; `dBP1 < d2 <= dBP2`:
    `LBP1 + A2·lg(d3/dBP13)`, `sf = [sig2]`; `d2 > dBP2`: `LBP2 + A3·lg(d3/dBP23)`, `sf = [sig3]`
  - `nlos` (An, Bn; the `dual_slope` keys if `LOSmax != 0`; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0,
    G2n = 1, LOSmax = 1, sig optional; Ax, Bx, Cx = 0): `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`;
    `PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb + En·lg hm +
    E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; `pl = PLn`;
    if `[LOSmax] != 0`: `pl = max(pl, PL_LOS)` with the `dual_slope` loss of the same parameters (incl. the
    random hE); if `Ax`, `Bx` or `Cx` is present: `pl = max(pl, Ax·lg d3 + Bx + Cx·lg f)` (e.g. the InF-SL term of
    InF-DL); `sf = [sig]` if present
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS / d3, 1)` with the absolute Tx height;
    `pl = +Inf` if `sa <= 0`; else `pl = A·lg d3 + B + C·lg f + D·lg(asin(sa)) + PLa` with `PLa = Z(f) / sa` if
    `usePLa != 0`, else 0; `Z(f)` is the zenith attenuation by atmospheric gases (93 values from 1 to 100 GHz,
    linear interpolation, constant beyond both ends)
  - empty `pl_model`: `pl = 0`
- New keys of the TR 38.901 v19 files: `hE_rand` (non-zero: random environment height of UMa), `LOSmax` (0: no
  maximum with the LOS term in `nlos`), `Ax`, `Bx`, `Cx` (additional term in the maximum of `nlos`)
- Random environment height (TR 38.901 Table 7.4.1-1, note 1), in `dual_slope` and the LOS term of `nlos` when
  `[hE_rand] != 0` and `rnd_he` is given, per link with `u = rnd_he(l)`, `d = d2`, raw `hUT`:
  `g = 0` for `d <= 18`, else `1.25·(d/100)³·exp(-d/150)`; `C' = 0` for `hUT < 13`, `((hUT - 13)/10)^1.5` for
  `hUT <= 23`, else 1; `C = C'·g`, `p1 = 1/(1 + C)`; `n` = size of the set {12, 15, ..., hUT - 1.5};
  `hE = 1` if `C = 0`, `n = 0`, `u < p1` or `p1` rounds to 1 (C below 1.1e-16, e.g. beyond about 7.5 km at
  `hUT = 20`), else `hE = 12 + 3·min(floor((u - p1)/(1 - p1)·n), n - 1)`.
  The UT height is then clamped to `hm = max(hUT, hE + 0.1, 0.1)` instead of `1.1·hE`: for `hE = 1` this is the
  fixed-hE clamp (also below `hUT = 1.1`), and the drawn heights 12, 15, ... are not affected (`hUT - hE >= 1.5`).
  Without `rnd_he` the fixed `[hE]` is used. `rnd_he` is column 10 of [lsf_draw](#lsf_draw), a spatially consistent uniform
  value: TR 38.901 draws hE independently per link and gives no decorrelation distance; the library makes it
  spatially consistent in the sense of clause 7.6.3 with the decorrelation distance `SF_lambda`, so nearby UTs of
  a site share hE. Along a track hE switches between its discrete values where `u` crosses a threshold, and the
  loss beyond the breakpoint steps there; evaluate `rnd_he` once per track segment to avoid this
- Distances: `d3`, `d2` are the geometric distances (TR 38.901 eq. 7.4-1: for an indoor UT `d3D-out + d3D-in`).
  If `d2d_in` is given (legacy mode), a link with `d2d_in(l) > 0` uses the outdoor distance instead:
  `din = min(d2d_in, d2)`, `d3in = d3·din/d2` (`d3` if `d2 = 0`), limited to `d3`; if `d3 - d3in < 10`:
  `d3in = d3 - 10` (minimum outdoor distance 10 m), then `max(d3in, 0)`; `p = d3in/d3` (0 if `d3 = 0`);
  `d3 = d3 - d3in`, `d2 = d2·(1 - p)`. A link with `d2d_in(l) = 0` is an outdoor link and keeps the geometric
  distances (no 10 m rule). Then `d3 = max(d3, 0.1)`, `d2 = max(d2, 0.1)`. The O2I penetration loss is not added;
  add the output of [o2i_loss](#o2i_loss)
- Default shadow-fading std, where a model does not set it: the SF std of the LSP formula of [lsp_generate](#lsp_generate)
  with the geometric distance, `sf = max(0, max(0, SF_sigma + SF_delta·lg(SF_omega + f')) + SF_kappa·lg(d) +
  SF_tau·lg(h) + SF_beta·lg(atan(h/d)))` with `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`,
  `f' = max(f, LSP_fc_min)`; for scenario files without such terms this is `SF_sigma`. Passing `sf_sigma` to
  [lsp_generate](#lsp_generate) is then neutral. With `sf_sigma` requested, `SF_sigma`, `SF_delta`, `SF_kappa`, `SF_tau`,
  `SF_beta` must be finite, and a frequency term with `SF_delta != 0` requires a finite `SF_omega` and
  `SF_omega + f' > 0`
- Validation: positions, `fc`, the lengths of `d2d_in` and `rnd_he`, their values, the model name, the
  parameters of the model in the order the model reads them (a required key that is missing throws
  `Path-loss model '<model>' requires the parameter '<key>'.`, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every key that the model reads must have exactly one value, which must be finite; keys not read are ignored),
  the breakpoint factors `E`, `E1`, `E2` and `G2n` of `nlos` (> 0) and `A1 != A2` of `winner_los`, then `scenpar`
  ([scenpar_validate](#scenpar_validate), its messages pass through), then the SF coefficients and frequency term if `sf_sigma` is
  requested. All checks run before any output is written
- The models are evaluated as written for any positive frequency, also outside their validity range
- Outputs must not share memory with the inputs or with each other

### Declaration:
```
void quadriga_lib::path_loss(
    const std::string &pl_model,
    const std::unordered_map<std::string, std::vector<double>> &plpar,
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &fc,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &pl,
    arma::mat *sf_sigma = nullptr,
    const arma::vec *d2d_in = nullptr,
    const arma::vec *rnd_he = nullptr);
```

### Inputs:
- **`pl_model`** — Path-loss model: `logdist`, `logdist_simple`, `constant`, `winner_los`, `winner_nlos`,
  `winner_pathloss`, `dual_slope`, `tripple_slope`, `nlos`, `satellite`, or empty (zero loss)
- **`plpar`** — Path-loss parameters, keys without "PL_", one value each, as returned by [conf_file_read](#conf_file_read)
- **`scenpar`** — Scenario parameters (SF std terms, `LSP_fc_min`) as returned by [conf_file_read](#conf_file_read) or
  [scenpar_validate](#scenpar_validate); missing keys have their default value
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from [o2i_loss](#o2i_loss)), finite
  and >= 0; given = legacy mode with the outdoor distance; 0 = outdoor link; `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height, column 10 of
  [lsf_draw](#lsf_draw), in [0, 1]; `[n_link]`

### Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** *(optional)* — Shadow-fading standard deviation in [dB], >= 0 for the default; `[n_link, n_freq]`

### See also:
- [conf_file_read](#conf_file_read) (the model and its parameters)
- [lsf_draw](#lsf_draw) (the random values, column 10 for `rnd_he`)
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss and indoor distance)

---
## scenpar_validate
Validate scenario parameters and complete them with default values

- Checks a set of QuaDRiGa scenario parameters that did not come from a conf file and adds every missing key
  with its default value, so that `scenpar` has all `n_key = 171` entries on return
- An empty `scenpar` returns the complete default parameter set
- A set with the 138 numeric keys of QuaDRiGa is completed with the 33 keys for 3GPP TR 38.901 v19.5.0, which
  have no effect at their default of 0
- A name in `scenpar` that is not one of the 171 keys is an error, also with `validate = false`; this includes
  the QuaDRiGa key `GR_epsilon`, which is not supported
- Value checks with `validate = true`; `NaN` is an error for every key:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`, `PerClusterES_D`,
    `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`, `EoD_offset_b`,
    `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`, `NF_alpha`, `NF_beta`,
    `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`, `LOS_state_lambda`,
    `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`, `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
  - `subpath_method` must be `legacy`, `Laplacian` or `mmMAGIC`
- With `validate = false` no value is checked: `NaN` and any `subpath_method` are accepted
- The symmetric matrix `lsp_xcorr` is built from the 28 correlation keys and has a unit diagonal. It is
  tested by a Cholesky factorization. A matrix that is not positive definite is not an error.
- Throws `std::invalid_argument`; a failed call leaves `scenpar` and the outputs untouched

### Declaration:
```
bool quadriga_lib::scenpar_validate(
    std::unordered_map<std::string, double> &scenpar,
    const std::string &subpath_method = "legacy",
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    bool validate = true);
```

### Inputs:
- **`scenpar`** — Numeric scenario parameters; updated in-place: given values are kept, missing keys are
  added with their default; `n_key = 171` entries on return
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`, the only parameter that is not numeric
- **`validate`** *(optional)* — `false`: only complete with defaults, no value checks

### Outputs:
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 33 new keys; `[n_key]`

### Returns:
- `true` if `lsp_xcorr` is positive definite (all 8 pivots of the factorization are > 0); `false` also if it
  contains `NaN`

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

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
- Returns angles in [-pi, pi]; max error ~3 ULP (~3.6e-7)
- Special values as std::atan2 (both kernels): atan2(±0, +0) = ±0, atan2(±0, -0) = ±pi, atan2(±0, x < 0) = ±pi,
  atan2(±inf, ±inf) = ±pi/4 or ±3pi/4; NaN in x or y gives NaN

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
- AVX2 kernel computes the angles in single precision (double outputs are cast back from float); for `dtype=double`,
  `len` and z / len are computed in double precision, so very small or very large vectors keep their direction;
  GENERIC kernel preserves full `dtype` precision
- Azimuth in [-pi, pi] (atan2 special values as in [fast_atan2](#fast_atan2)); NaN input gives NaN in `az`, `el` and `len`

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
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw; default `0`

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
- AVX2 kernel computes sine and cosine in single precision (max. error about 1.3e-7 for angles up to 1e10;
  `dtype=double` angles are reduced in double precision first), results widened back. Use `use_kernel=1` to force
  GENERIC if full double precision is required.

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
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw

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
- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for |x| <= 1e10
  (both kernels; the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; inf and NaN give NaN
- The AVX2 kernel is fastest for |x| <= 2^22 (4.2e6); larger arguments take about 1.5 times as long
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
- If both input amplitudes are negligible (below 1.19e-7), output is zero
- Max relative error vs. double-precision reference about 5e-7 for phase differences up to 2 rad, 1.2e-5 up to
  3 rad; larger towards the near-antipodal transition to linear interpolation
- The result of an element does not depend on the other elements
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
- Outside the range of the input sample points, the nearest end value is returned (constant extrapolation);
  output sample points that are NaN give NaN.
- The input sample points need not be sorted (they are sorted internally); repeated values are allowed;
  they must be finite.
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
  the direct TX-RX distance, matching the `path_length` definition used by [qrt_file_read](#qrt_file_read) for a transmission-only 
  path. In EM mode the loss is `32.45 + 20·log10(f_GHz) + 20·log10(d_m)` [dB]; in scalar mode it is 
  `20·log10(d_m) + α(f)·d_m` [dB], with `α` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm, evaluated at the
  acoustic frequency `f = center_frequency · c_sound / c_light`.
- **`apply_phase`** — If `true`, multiplies `xprmat` by the propagation phase term `exp(-j·2π·d/λ)`, using the
  same direct TX-RX distance `d`. Default `false`. The wavelength is `λ = c_light / center_frequency` in both
  modes; in scalar mode this is the acoustic wavelength because `center_frequency` is the EM-equivalent
  frequency (see above). Since the term has unit
  magnitude, `xprmat` stays normalized. This is the phase contribution that
  [get_channels_spherical](#get_channels_spherical) would otherwise add from `path_length`; enabling it here gives the complete
  omni-radiator coefficient without running the channel generator.

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `[n_pos]`. Relative to FSPL by default,
  i.e. bounded by 0 and 1; absolute propagation gain when `apply_fspl = true`.
- **`xprmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase. The `gain` output holds the power.
  For EM mode: interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx), the row order
  of `M` in [get_channels_spherical](#get_channels_spherical); `[8, n_pos]`;
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
- [qrt_file_read](#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

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
- **`interact_coord_1`** — Interaction coordinates for segment 1; `[3, sum(no_interact_1)]`; any other size throws
  `std::invalid_argument` (an empty matrix is accepted when `sum(no_interact_1)` is 0)
- **`no_interact_2`** — Number of interaction points per path for segment 2 (IRS → RX); `[n_path_2]`
- **`interact_coord_2`** — Interaction coordinates for segment 2; `[3, sum(no_interact_2)]`; same check as for
  `interact_coord_1`
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
- Triangle winding is consistent (outward-facing normals), compatible with [obj_file_write](#obj_file_write); a negative scale mirrors
  the cube, and when the product of the scales is negative the winding is swapped so that the normals stay outward
  (as Blender's "Apply Scale")
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
   | 7     | 1        | 10      | direct line in 10 segments |

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
## interface_gain
Linear gain of the lumped penetration loss of entering a material

- Computes `g = 10^(-A/10)` with `A = att·(f/fRef)^attB` [dB], the loss applied once where a ray enters material
  `iM`, on top of the Fresnel term of the face (e.g. a building-entry loss of 3GPP TR 38.901). `att` must be
  `≥ 0` (the material table is checked when it is read), so `g` is in `[0, 1]`.
- Only `att`, `attB` and `fRef` are read; the face reflection is computed by [ray_mesh_interact](#ray_mesh_interact) and
  [ray_state_update](#ray_state_update), the distance-dependent loss by [medium_gain](#medium_gain). Columns the material model does not know
  are ignored.

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
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> (the formulas and the columns)
- [medium_gain](#medium_gain) (for the distance-dependent in-medium loss)
- [ray_mesh_interact](#ray_mesh_interact) (for complex ray-material interactions)
- [obj_file_read](#obj_file_read) (defines mtl_prop format)

---
## medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

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
- **`dist`** — Path length of the ray inside the medium in [m]
- **`center_frequency`** — Center frequency in [Hz]

### Returns:
- Linear in-medium gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the medium

### See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> (the formulas and the columns)
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
  both the XML and the mesh folder must be distributable together; the XML refers to each PLY file by its path
  relative to the XML file (`<stem>_meshes/<object>.ply`)
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
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

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
## obj_overlap_test
Detect overlapping 3D objects in a triangular mesh

- Returns the IDs (as given in `obj_ind`) of all objects that intersect at least one other object
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
- **`obj_ind`** — Object ID of each triangle; any values, e.g. the 0-based `obj_ind` output of [obj_file_read](#obj_file_read); `[n_mesh]`
- **`reason`** *(optional)* — Human-readable overlap description per overlapping object: the finding of its first
  overlapping pair, pairs taken in the order of the sorted object IDs; independent of the number of threads; `[n_overlap]`
- **`tolerance`** *(optional)* — Geometric tolerance; intersections smaller than this are ignored

### Returns:
- `arma::uvec`: IDs of all overlapping objects as given in `obj_ind`, unique and ascending; empty for an empty mesh; `[n_overlap]`

### See also:
- [obj_file_read](#obj_file_read) (reads mesh data from files and generates `obj_ind` input)

---
## path_to_tube
Convert a 3D path into a tube surface mesh for visualization

- Converts an ordered sequence of 3D points into a tubular quad mesh with circular cross-sections
- Where the path turns by more than 160° (the two segments enclose an angle below 20°), the tube is split and an
  extra vertex ring is inserted to avoid intersection
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
- **`vert`** — Tube vertex positions; `[3, (n_coord + n_split) × n_edges]` where `n_split` is the number of turns > 160°
- **`faces`** — Quad face indices into `vert`, 4 indices per quad; `[4, (n_coord - 1) × n_edges]`

---
## plane
Construct a triangulated plane mesh

- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices at +/-1, z = 0) before scaling
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 * n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with [obj_file_write](#obj_file_write); when the product
  of the x and y scales is negative (mirrored plane), the winding is swapped so that the normals still point in +z
  direction before the rotation
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, Euler rotation, and translation are applied in that order (scale -> rotate -> translate)

### Declaration:
```
arma::Mat<dtype> quadriga_lib::plane(
    const arma::vec &scale = {1.0},
    const arma::vec &rotation = {0.0, 0.0, 0.0},
    const arma::vec &location = {0.0, 0.0, 0.0},
    const arma::uword n_div = 1);
```

### Inputs:
- **`scale`** — Length 1 scales x and y uniformly; length 2 scales `{x,y}` independently; length 3 is accepted for
  compatibility with [cube](#cube), but the third element is ignored; empty = 1 (no scaling)
- **`rotation`** — Euler angles in [rad] about {x,y,z}, applied as R = Rz*Ry*Rx (Blender XYZ); length 3 or empty (no rotation)
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
## ray_commit
Commit the paths of a launch configuration that reach a receiver

- Intersects the current beam set with the receive point cloud ([ray_point_intersect](#ray_point_intersect)) and appends one
  [path](#path) object to `paths_commit` for every ray-point pair that survives the gates below.
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
```
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
- **`paths`** — In-flight per-ray [path](#path) objects; `[n_ray]`. Frequency count and layout define those of the committed paths; not modified
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
- [path](#path) (the per-ray storage object)

---
## ray_init
Seed a sphere of rays from a point source

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
  [path](#path) layout supports 1-127)
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
- **`paths`** — Per-ray [path](#path) objects, one per ray; each reinitialized to 0 segments with `n_freq`
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
- [path](#path) (the per-ray storage object populated in `paths`)

---
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

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
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

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
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> (companion document)
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
Per-ray medium state machine with closed-form thin-slab (Fabry-Pérot) resolution

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
   |  192+| 0xC0+|    — |    —    | refl, scatter, resolve clear    | reserved: scattering not implemented                         |
   A stack of two different sheets is crossed as one event with the codes of a sheet; a singular stack kills the ray.

### See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> (companion document)
- [ray_mesh_interact](#ray_mesh_interact) (computes the per-interaction Fresnel/Jones result this function corrects)
- [calc_diffraction_gain](#calc_diffraction_gain) (uses this function for its media)
- [ray_progress](#ray_progress) (the tracer that calls it)

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
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

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
## refractive_index
Real refractive index of a homogeneous medium

- Returns `n = Re √(ε_r·μ_r)` of material `iM`, with the ITU-R P.2040 model `ε_r = a·(f/fRef)^b − j·17.98·σ/f[GHz]`,
  `σ = c·(f/fRef)^d`, and `μ_r` from `e`, `f`, `g`, `h` in the same form. For a material without the mass column
  the permittivity resonance (`resF`, `resQ`, `resS`) is part of `ε_r`.
- For a mass-law material (`m > 0`) `ε_r` and `μ_r` are an impedance mapping and `n` is not a propagation index.
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
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> (the formulas and the columns)
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
- `triangles_out` may be the same object as `triangles_in`, and `mtl_ind_out` the same as `mtl_ind` (in place)
- Throws `std::invalid_argument` when `n_div` is so large that the number of output values exceeds 64 bits

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
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `[n_triangles, 9]`
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
- **`mesh_index`** *(optional)* — 1-based row in `mesh` of each row of `meshR` (0 = padding); `[n_meshR]`

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
- The auto-detected range ignores `NaN` and `+-Inf`; `+Inf` and `-Inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value; also an infinite `min_val` or `max_val`) maps every value to the first colour
- Throws `std::invalid_argument` for an empty `data` matrix and for `min_val > max_val` (both given)
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
  `< n_ray`. Duplicate values are allowed: the updates of a column are then applied one after the other
  in update-index order (each with the full initialize -> update -> normalize -> apply_gain sequence),
  so the result is the same as a single-threaded run for any thread count. With duplicates the call
  runs single-threaded; unique indices keep the parallel loop.

### Usage:
The flags are composable in one call; the per-column order is always
initialize -> update -> normalize -> apply_gain. Argument order is
`(xprmat, update, gain, initialize, normalize, apply_gain, ray_index)`.
```
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
  rows 6-7 = (H out, H in). This is the row order of `M` in [get_channels_spherical](#get_channels_spherical) (rows = RX, columns = TX).
- **Scalar mode**, 2 values per ray: `[Re Im]`.
- The update is `M_out = M_update * M_state` (new interaction left-multiplies), consistent with
  `E_out = M * E_in` for column vectors and Armadillo/MATLAB column-major storage.

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (produces `xprmat` in this exact layout)

---

# Sum-of-sinusoids random number generator

These functions generate **spatially correlated random numbers**: values that are random, but similar at
nearby positions. Two positions that are much closer than the *decorrelation distance* get almost the same
value, two positions that are far apart get independent values. The decorrelation distance is the distance
at which the correlation has fallen to `exp(-1)`, about 0.37. Typical uses are shadow fading maps and the
spatially consistent parameters of the 3GPP TR 38.901 channel model.

The values are a sum of sinusoids, `s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n )`. The frequencies
`f_n` set how the correlation falls with the distance, i.e. the autocorrelation function (ACF). The random
phases `phi_n` select one realisation of the random field. The generator needs no map and no grid:
positions can be anywhere, in 1, 2 or 3 dimensions, and the same frequencies and phases always give the
same field.

**Quick start**
[sos_randn](#sos_randn) sets up the generators, draws the phases and returns the values in one call. [sos_rand](#sos_rand)
and [sos_randi](#sos_randi) do the same for Uniform and integer values.

```
#include "quadriga_lib.hpp"

// 1000 positions; one row per position, columns are {x, y, z} in meters
arma::mat pos(1000, 3, arma::fill::randu);
pos *= 100.0;

// One independent generator per decorrelation distance: 10 m and 50 m
arma::vec dist_decorr = {10.0, 50.0};

// Normal distributed values with zero mean and unit variance, size [1000, 2]
arma::mat val;
quadriga_lib::sos_randn<double>(dist_decorr, pos, val, nullptr, nullptr, "Comb256", 42);

// Uniform values between 0 and 1, and integers from 1 to 6
arma::mat uni, dice;
quadriga_lib::sos_rand<double>(dist_decorr, pos, uni, nullptr, nullptr, "Comb256", 42);
quadriga_lib::sos_randi<double>(dist_decorr, pos, 6, dice, nullptr, nullptr, "Comb256", 42);
```

The last argument is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field.

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.
**Keeping a generator**
A generator is two matrices: the frequencies from [sos_preset](#sos_preset) and the phases from [sos_init_phases](#sos_init_phases).
[sos_val](#sos_val) evaluates them. Each column of the phase matrix is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```
// Frequencies of a predefined generator for a decorrelation distance of 10 m
arma::mat freq;
arma::vec acf, dist;
quadriga_lib::sos_preset<double>("Comb256", freq, &acf, &dist, 10.0);

// Phases of 20 generators; each column is one independent random field
arma::mat phase;
quadriga_lib::sos_init_phases<double>(freq.n_rows, 20, phase, 42);

// Values of all 20 generators at the positions, size [1000, 20]
arma::mat val_all;
quadriga_lib::sos_val(freq, phase, pos, val_all);

// The same fields at other positions, at any later time
arma::mat pos_new = pos + 2.5, val_new;
quadriga_lib::sos_val(freq, phase, pos_new, val_new);

// Uniform distributed output
arma::mat val_uni;
quadriga_lib::sos_val<double>(freq, phase, pos, val_uni, nullptr, nullptr, nullptr, nullptr, 1.0, true);

// Map on a regular grid, one cube of size [n_y, n_x, 1] per generator
arma::vec x = arma::regspace(0.0, 0.5, 250.0), y = arma::regspace(0.0, 0.5, 100.0);
std::vector<arma::cube> grid;
quadriga_lib::sos_map<double>(freq, phase, x, grid, &y);
```

- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `"Exp512"`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [sos_calc_mse](#sos_calc_mse) reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `"Comb256"` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves.

```
// Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
arma::mat pos_tx = pos, pos_rx(1000, 3, arma::fill::randu), val_link;
pos_rx *= 100.0;
quadriga_lib::sos_val<double>(freq, phase, pos_tx, val_link, &pos_rx, nullptr, &acf, &dist);
```

- **Mean over an area:** the values at positions that lie within a few decorrelation distances of each
  other are similar, so their average is not zero for a single realisation. This is how a correlated
  field behaves. [sos_remove_bias](#sos_remove_bias) can force the average to zero if an application needs it. This
  removes variance: for 200 positions in a square with an edge of 20 decorrelation distances 2.5 %, for
  an edge of 5 decorrelation distances 14 %, for 2 already 44 %. It is meant for positions that spread
  over many decorrelation distances.
- **Coordinates:** use local coordinates. The argument `f_n·x` of a sinusoid is computed in double
  precision and is known to about 1e-16 of its size, so the values lose accuracy with the distance from
  the origin counted in wavelengths. Up to 1e9 wavelengths (more than 1e9 m for a decorrelation distance
  of 10 m) the results stay within the tolerance stated in [sos_val](#sos_val). The `float` versions of the
  functions read positions in single precision, i.e. rounded to about 1 mm at a distance of 10 km.
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [sos_generate](#sos_generate), which runs on one thread. No result depends on the number of threads.
  [sos_val](#sos_val) computes in its output: for `double` it needs no further memory of that size for one device
  and one more array of that size for two devices; for `float` one array in double precision is added.

**Own autocorrelation functions**
[sos_generate](#sos_generate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos_calc_mse](#sos_calc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```
// Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
arma::vec dist_own = arma::regspace(0.0, 0.125, 24.875);
arma::vec acf_own = arma::exp(-arma::square(dist_own / 5.0));

// 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
arma::mat freq_own;
quadriga_lib::sos_generate<double>(acf_own, dist_own, 256, freq_own, nullptr, nullptr, 3, true, 300, true,
                                   0.01, 0.0, 1);

// Approximation error in dB, larger is better; about 35 dB for this example
double mse = quadriga_lib::sos_calc_mse(freq_own, acf_own, dist_own);
```

The result is used like a preset: draw phases with [sos_init_phases](#sos_init_phases) and evaluate with [sos_val](#sos_val). To
change the decorrelation distance afterwards, divide the frequencies and multiply the distances by the
same factor.

**Reference**
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
<a target="_blank" rel="noopener noreferrer" href="https://ieeexplore.ieee.org/document/8644265">IEEE Xplore</a> (Original version),
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.pdf">PDF version</a>,
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.md">Markdown version</a>

---
## sos_acf_2d
Target ACF on a 2-D grid

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the distance,
  on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample point;
  `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 2` (0-based)
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos_acf_approx](#sos_acf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on the regular
  axis; for long single-precision tables the rounding of the sample points themselves is allowed for
- `dist_2d` may be the same object as `acf` or `dist`

### Declaration:
```
void quadriga_lib::sos_acf_2d(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    arma::Mat<dtype> &acf_2d,
    arma::Col<dtype> *dist_2d = nullptr);
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; `[n_acf]`

### Outputs:
- **`acf_2d`** — ACF on the grid; `[n_grid, n_grid]`
- **`dist_2d`** *(optional)* — Grid axis in [m]; `[n_grid]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
## sos_acf_approx
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [sos_val](#sos_val) over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [sos_acf_2d](#sos_acf_2d): from `-2·Dmax` to `2·Dmax` with the spacing of `dist`,
  `n_grid = 4·n_acf - 3`
- `n_dim = 1`: along the axis; `[n_grid, 1, 1]`
- `n_dim = 2`: on the x-y plane; `[n_grid, n_grid, 1]`; element `(iy, ix)`
- `n_dim = 3`: on the x-y, x-z and y-z plane (slices 0, 1, 2); `[n_grid, n_grid, 3]`; the row index belongs
  to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center. One half is computed and the other
  half is its mirror image, so the symmetry is exact
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)
- `dist_2d` may be the same object as `dist`
- `acf_approx` cannot share memory with `freq` or `dist`; such a call throws before anything is written

### Declaration:
```
void quadriga_lib::sos_acf_approx(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &dist,
    arma::Cube<dtype> &acf_approx,
    arma::Col<dtype> *dist_2d = nullptr);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

### Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** *(optional)* — Grid axis in [m]; `[n_grid]`

### See also:
- [sos_acf_2d](#sos_acf_2d) (target ACF on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
## sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at random
  positions, without using the frequencies other than through [sos_val](#sos_val)
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin, evaluates
  the generator there, and for each distance bin computes the correlation coefficient between the two values
  of all position pairs whose distance falls into the bin
- Positions: a matrix `[2·n_pos, 3]` is filled column by column (all x, then all y, then all z) with
  `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in [sos_init_phases](#sos_init_phases)
- Bin `n` covers the distances from the midpoint between `bins(n-1)` and `bins(n)` (excluded) to the midpoint
  between `bins(n)` and `bins(n+1)` (included). The first bin starts at 1e-5 m, the last bin ends half a mean
  bin spacing above `bins(end)`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the first device,
  the second `n_pos` positions the second device, and the value of a link is [sos_val](#sos_val) with both positions.
  Two links form a pair if the distance between their first devices falls into the bin and the distance
  between their second devices does not exceed the upper limit of the bin; requires `acf` and `dist`
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- The pairs are searched in parallel and added in a fixed order; the result does not depend on the number
  of threads. Run time and memory: `2·n_pos^2` pairs for one device, at most 36 MB for the pairs of a block
  of rows
- The outputs may be the same objects as inputs

### Declaration:
```
void quadriga_lib::sos_acf_estimate(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &phase,
    const arma::Col<dtype> &bins,
    arma::uword n_pos,
    arma::Col<dtype> &acf_est,
    arma::Col<dtype> *dist_est = nullptr,
    arma::Col<dtype> *acf_est_dual = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    arma::sword seed = -1);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, 3]`
- **`phase`** — Phases of the generator in [rad]; `[n_coeff]`
- **`bins`** — Bin centers in [m]; must be >= 0, increase strictly and be finite; at least 2 elements;
  `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`acf`** *(optional)* — Target ACF; required for `acf_est_dual`, must then be > -1; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`seed`** *(optional)* — Seed for the positions; must be >= -1; `-1` seeds from the system random device

### Outputs:
- **`acf_est`** — Estimated ACF; `[n_bin]`
- **`dist_est`** *(optional)* — Mean distance of the pairs in each bin in [m]; `[n_bin]`
- **`acf_est_dual`** *(optional)* — Estimated ACF for two devices on the same map; `[n_bin]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF of the sinusoids in closed form)
- [sos_val](#sos_val) (evaluation of the generator)

---
## sos_acfi
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [sos_val](#sos_val) and [sos_remove_bias](#sos_remove_bias) for two devices on the same map
- An empty `d` returns an empty `val`
- `val` may be the same object as `acf`, `dist` or `d`

### Declaration:
```
void quadriga_lib::sos_acfi(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    const arma::Col<dtype> &d,
    arma::Col<dtype> &val);
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances in [m]; must be >= 0; `[n_d]`

### Output:
- **`val`** — ACF at the distances `d`; `[n_d]`

### See also:
- [sos_preset](#sos_preset) (ACF tables of the predefined models)

---
## sos_calc_mse
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- Return value: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`. A direction and its opposite
  give the same ACF
- `mse_core`: error on the grid of [sos_acf_approx](#sos_acf_approx) against [sos_acf_2d](#sos_acf_2d), for the grid points within a
  radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error is averaged over
  the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and the return value equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`. The
  target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core` and `mse_all` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as +Inf dB; this happens for a target that
  the sinusoids reproduce exactly at all sample points, e.g. sinusoids at zero frequency and a target of 1
- The test directions are evaluated in parallel; the result does not depend on the number of threads
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)
- `acf_tail` and `test_dir` may be the same objects as inputs

### Declaration:
```
dtype quadriga_lib::sos_calc_mse(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    dtype *mse_core = nullptr,
    dtype *mse_all = nullptr,
    arma::Col<dtype> *acf_tail = nullptr,
    arma::Mat<dtype> *test_dir = nullptr,
    arma::uword n_dir = 10);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** *(optional)* — Number of test directions; must be > 0; ignored for `n_dim = 1`

### Outputs:
- **`mse_core`** *(optional)* — Error within the core radius in [dB]
- **`mse_all`** *(optional)* — Error on the complete grid in [dB]
- **`acf_tail`** *(optional)* — Approximated ACF beyond the fitted range; `[4]`
- **`test_dir`** *(optional)* — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}`
  for `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### Returns:
- Error along the test directions in [dB]

### See also:
- [sos_generate](#sos_generate) (finds the frequencies; minimises the error returned here)
- [sos_acf_approx](#sos_acf_approx), [sos_acf_2d](#sos_acf_2d) (the two grids that are compared)

---
## sos_generate
Find the frequencies of a sum-of-sinusoids generator for a given ACF

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = true` places them regularly (equally spaced azimuth, spiral lattice, as the test directions
  of [sos_calc_mse](#sos_calc_mse)), `false` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the residual
  between the target and the contribution of all other sinusoids along its own direction: coarse search
  over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The
  new frequency is kept only if the mean squared error over the test directions decreases. The iteration
  stops after a pass that changes nothing or lowers the error by less than a factor of 1e-6, or after
  1000 passes
- `random_init = true` starts from uniformly random frequencies; `false` builds the set up sinusoid by
  sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is not
  controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [sos_calc_mse](#sos_calc_mse)). To prevent this, set `freq_min`; the tables of [sos_preset](#sos_preset) use
  `0.05 / d` with the decorrelation distance `d`
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse in
  others; the tables of [sos_preset](#sos_preset) were made with 300
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each sinusoid;
  0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- The search grid has `( freq_max - freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are stored for
  all sample points. It must not have more than 100000 points (the default has 151), which limits
  `freq_max - freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- `ase` and `test_dir` may be the same objects as `acf` or `dist`
- `test_dir` and the reported error use the same test directions as [sos_calc_mse](#sos_calc_mse) with the same `n_dir`
- All computations are done in double precision for both instantiations
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject decisions, so a
  difference in the last bit of the input or of the math library can lead to another table of equivalent
  quality

### Declaration:
```
void quadriga_lib::sos_generate(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    arma::uword n_coeff,
    arma::Mat<dtype> &freq,
    arma::Col<dtype> *ase = nullptr,
    arma::Mat<dtype> *test_dir = nullptr,
    arma::uword n_dim = 3,
    bool uniform_smp = true,
    arma::uword n_dir = 10,
    bool random_init = true,
    dtype freq_min = 0.0,
    dtype freq_max = 0.0,
    arma::sword seed = -1);
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_dim`** *(optional)* — Number of dimensions; 1, 2 or 3
- **`uniform_smp`** *(optional)* — Regular (`true`) or random (`false`) directions of the sinusoids
- **`n_dir`** *(optional)* — Number of test directions; must be > 0; ignored for `n_dim = 1`
- **`random_init`** *(optional)* — Start from random frequencies (`true`) or build up sinusoid by sinusoid
  (`false`)
- **`freq_min`** *(optional)* — Lower limit of the frequency magnitude in [1/m]; must be >= 0; 0 = none
- **`freq_max`** *(optional)* — Upper limit of the frequency magnitude in [1/m]; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; limited by the size of the search grid, see above
- **`seed`** *(optional)* — Seed; must be >= -1; `-1` seeds from the system random device

### Outputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, n_dim]`
- **`ase`** *(optional)* — Mean squared error over the test directions after each pass, linear scale;
  `[n_pass]`
- **`test_dir`** *(optional)* — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}`
  for `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_calc_mse](#sos_calc_mse) (error of a frequency table)
- [sos_preset](#sos_preset) (precomputed tables)

---
## sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [sos_val](#sos_val)
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: `std::mt19937_64` seeded with `seed`; each value is `2·pi · ( u - 0.5 )`, where `u` is the
  upper 53 bits of one generator output divided by 2^53; no distribution class of the standard library is
  used, so a seed >= 0 gives the same phases on all platforms
- The matrix is filled column by column from one stream: for the same seed, column `g` does not depend on
  `n_gen`
- For `dtype = float` the values are rounded from double; a value that rounds to pi is returned as -pi
- A seed stands for a set of phases. Generators that use phases of the same seed with different
  frequencies (another decorrelation distance or preset) produce related fields, which are equal at the
  coordinate origin and correlated around it; draw independent quantities as columns of one call or with
  different seeds

### Declaration:
```
void quadriga_lib::sos_init_phases(
    arma::uword n_coeff,
    arma::uword n_gen,
    arma::Mat<dtype> &phase,
    arma::sword seed = -1);
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0
- **`seed`** *(optional)* — Seed of the random stream; must be >= -1; `-1` seeds from the system random
  device and gives different phases on each call

### Output:
- **`phase`** — Phases in [rad]; `[n_coeff, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluates the generators)
- [sos_preset](#sos_preset) (frequencies that belong to the phases)

---
## sos_map
Spatially correlated random values on a regular grid

- Evaluates the generators of [sos_val](#sos_val) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz)` of a cube belongs to the position `{x(ix), y(iy), z(iz)}`
- One cube per generator, i.e. per column of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; an omitted or empty
  `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [sos_val](#sos_val) at the grid points, bit for bit; kernels, accuracy and the Uniform
  mapping are described there
- Cubes of the correct size keep their memory. If they lie one after the other in memory (one block of
  `n_gen` cubes, as the MATLAB and Python interfaces allocate it), the values are computed in place;
  otherwise they are computed in a matrix `[ny·nx·nz, n_gen]` and copied
- A cube that shares memory with `freq` or `phase` is an error, before anything is written

### Declaration:
```
void quadriga_lib::sos_map(
    const arma::Mat<dtype> &freq,
    const arma::Mat<dtype> &phase,
    const arma::Col<dtype> &x,
    std::vector<arma::Cube<dtype>> &map,
    const arma::Col<dtype> *y = nullptr,
    const arma::Col<dtype> *z = nullptr,
    bool uniform = false,
    int use_kernel = 0);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`x`** — Grid points on the x-axis in [m]; `[nx]`
- **`y`** *(optional)* — Grid points on the y-axis in [m]; requires `n_dim >= 2`; omitted = one sample at 0,
  `ny = 1`; `[ny]`
- **`z`** *(optional)* — Grid points on the z-axis in [m]; requires `n_dim = 3`; omitted = one sample at 0,
  `nz = 1`; `[nz]`
- **`uniform`** *(optional)* — Output distribution: `false` = Normal(0,1), `true` = Uniform between 0 and 1
- **`use_kernel`** *(optional)* — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  throws if the requested kernel is unavailable

### Output:
- **`map`** — Values; vector of length `n_gen` with one cube of size `[ny, nx, nz]` per generator

### See also:
- [sos_val](#sos_val) (evaluation at arbitrary positions, two devices)

---
## sos_preset
Frequencies and ACF of a predefined sum-of-sinusoids generator

- Returns a precomputed frequency table for [sos_val](#sos_val) together with the autocorrelation function (ACF)
  it was fitted to
- Presets: `Exp128`, `Exp256`, `Exp512`, `Exp1024`, `Gauss128`, `Gauss256`, `Gauss512`, `Gauss1024`,
  `Comb128`, `Comb256`, `Comb512`, `Comb1024`. The number is the number of sinusoids, `n_coeff`. The name is
  matched exactly, including upper and lower case
- ACF types, with the distance `D` and the decorrelation distance `d`: `Exp`: `R = exp( -D/d )`;
  `Gauss`: `R = exp( -(D/d)^2 )`; `Comb`: `Gauss` for `D <= d`, `Exp` beyond. All three have `R(d) = exp(-1)`
- The ACF is returned as a table with 200 samples from 0 to `4.975·d` in steps of `0.025·d`
- The tables are stored for `d = 10 m`. For another decorrelation distance the distances are multiplied by
  `d/10` and the frequencies by `10/d`
- All tables are 3-dimensional and can be used with positions of 1, 2 or 3 columns
- The outputs cannot be the same object or share memory with each other; such a call throws before anything
  is written
- How the tables were made: [sos_generate](#sos_generate) with the 200-sample ACF for 10 m, `n_dim = 3`,
  `uniform_smp = true`, `n_dir = 300`, `random_init = true`, `freq_min = 0.005`, `freq_max = 15 / 49.75`,
  and the seeds 39, 3, 2, 5 (Exp), 13, 32, 6, 21 (Gauss), 13, 25, 37, 47 (Comb) for 128, 256, 512 and 1024
  sinusoids. Sinusoid `n` points in direction `n` of the spiral lattice described in [sos_calc_mse](#sos_calc_mse)
- The lower frequency limit means that no sinusoid has a wavelength above 20 decorrelation distances. The
  ACF of the tables therefore stays near zero beyond the fitted range (magnitude below 0.015 at 10, 20, 50
  and 100 decorrelation distances), and the mean of the generator over a large area is not offset by
  sinusoids of nearly zero frequency

### Declaration:
```
void quadriga_lib::sos_preset(
    const std::string &name,
    arma::Mat<dtype> &freq,
    arma::Col<dtype> *acf = nullptr,
    arma::Col<dtype> *dist = nullptr,
    dtype dist_decorr = 10.0);
```

### Inputs:
- **`name`** — Name of the preset, e.g. `"Comb256"`
- **`dist_decorr`** *(optional)* — Decorrelation distance in [m]; must be > 0

### Outputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, 3]`
- **`acf`** *(optional)* — Target ACF; `[200]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; `[200]`

### See also:
- [sos_init_phases](#sos_init_phases) (random phases for the generator)
- [sos_val](#sos_val) (evaluation)
- [sos_generate](#sos_generate) (frequencies for other ACFs)

---
## sos_rand
Spatially correlated Uniform distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [sos_val](#sos_val)
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

### Declaration:
```
void quadriga_lib::sos_rand(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [sos_preset](#sos_preset)
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

### Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_randn](#sos_randn), [sos_randi](#sos_randi) (Normal and integer values)

---
## sos_randi
Spatially correlated integer random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- The Uniform value `u` of [sos_rand](#sos_rand) is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [sos_val](#sos_val), and
  an integer can then differ by 1 where the Uniform value lies at a step
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

### Declaration:
```
void quadriga_lib::sos_randi(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::uword imax,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`imax`** — Largest value; must be > 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [sos_preset](#sos_preset)
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

### Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability; one row per
  position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_randn](#sos_randn), [sos_rand](#sos_rand) (Normal and Uniform values)

---
## sos_randn
Spatially correlated Normal distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [sos_val](#sos_val)
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

### Declaration:
```
void quadriga_lib::sos_randn(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [sos_preset](#sos_preset)
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

### Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row per
  position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_rand](#sos_rand), [sos_randi](#sos_randi) (Uniform and integer values)

---
## sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF) over all
  pairs of positions, which is much larger than `1 / n_pos` when the positions lie within a few
  decorrelation distances of each other. This is a property of any correlated field, not an error of the
  generator
- This function adds one offset to all phases of a generator such that the mean of the Normal distributed
  values of [sos_val](#sos_val) over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`, which
  vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the smaller magnitude is
  applied, so the offset lies between -pi/2 and pi/2
- One device (`pos_b` omitted): one offset per generator
- Two devices on the same map (`pos_b` given, `phase_b` omitted): one offset per generator that zeroes the
  mean of the combined, normalised value; `acf` and `dist` are required
- Two devices on different maps (`pos_b` and `phase_b` given): each device is treated on its own, `phase`
  with `pos` and `phase_b` with `pos_b` and the frequencies `freq · freq_scale_b`; two offsets per generator.
  The mean of the combined value is then zero as well. `phase_b` cannot share memory with `phase`: one
  matrix cannot hold two corrections
- The arguments that are written, `phase`, `phase_b`, `offset` and `mean_before`, cannot share memory with
  another argument, not even in part, and `offset` and `mean_before` cannot be the same object. Such a call
  throws and leaves all arguments unchanged
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions. The
  variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the sample mean and
  `c` the mean ACF between that position and all positions; on average over the positions the loss is at
  least `v`. It is small only if the positions spread over many decorrelation distances. For positions
  within about one decorrelation distance most of the variance is removed
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation distance
  `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for `L = 5·d`, 5 % for
  `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are used
  with other frequencies (another decorrelation distance) or positions
- The mean of the Uniform output of [sos_val](#sos_val) is not exactly 0.5 afterwards
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose `abs(Z)` is
  below `1e-14 · n_coeff` already has a zero mean for every offset and is left unchanged, also if its
  phases lie outside that interval
- Two devices on the same map need an ACF above -1 at all sample points, as in [sos_val](#sos_val)
- The sums over the positions are computed in parallel over the sinusoids; the result does not depend
  on the number of threads

### Declaration:
```
void quadriga_lib::sos_remove_bias(
    const arma::Mat<dtype> &freq,
    arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    const arma::Mat<dtype> *pos_b = nullptr,
    arma::Mat<dtype> *phase_b = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    arma::Mat<dtype> *offset = nullptr,
    arma::Mat<dtype> *mean_before = nullptr,
    dtype freq_scale_b = 1.0);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; updated in-place; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos_b, 1]`, `[n_pos_b, 2]` or `[n_pos_b, 3]` as for `pos`; `n_pos_b = n_pos` for the same map, any
  number for different maps
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or empty =
  same map; updated in-place; no memory shared with `phase`; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** *(optional)* — Frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0; other than 1 requires `phase_b`

### Outputs:
- **`offset`** *(optional)* — Offsets added to the phases in [rad]; `[n_gen, 1]`, or `[n_gen, 2]` with
  `phase_b` (column 0: `phase`, column 1: `phase_b`)
- **`mean_before`** *(optional)* — Sample mean before the correction; `[n_gen, 1]`: mean of the values of
  [sos_val](#sos_val) for the given arguments; with `phase_b` `[n_gen, 2]`: mean of the field of each device on its
  own positions, the mean of the combined value being their sum divided by `sqrt(2)`

### See also:
- [sos_val](#sos_val) (the values whose mean is removed)

---
## sos_val
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of `freq`)
  and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function (ACF)
  `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the frequencies were
  fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The sines and
  cosines of the positions are computed once and reused for all generators
- Positions may have fewer columns than `freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and `pos_b`
  and `A` the target ACF interpolated as in [sos_acfi](#sos_acfi); `acf` and `dist` are required. The value does
  not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `freq · freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same matrix as
  `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- Two devices on the same map need an ACF above -1 at all sample points, because the value is divided by
  `sqrt( 2 + 2·A(D) )`
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and 1
  as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the tail is
  rounded to them, for `dtype = float` every value above 5.4 gives 1
- Kernels: GENERIC computes in double precision. AVX2 reduces the argument `f_n·x` to one period in
  double precision, evaluates sine and cosine in single precision and accumulates in double precision;
  its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by less than 1e-6
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The bound
  between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both kernels lose
  accuracy at the same rate, and from 2^53 periods on every position returns the value at the origin
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor on the
  number of threads
- For `dtype = float` the inputs are converted to double, the computation is the same and the result is
  rounded to single precision
- Memory: for `dtype = double` the values are computed in `val`; two devices need one more array of the
  size of `val`, and `dtype = float` one array of that size in double precision
- `val` cannot be the same object as an input, nor use the memory of one: it is resized and written while
  the inputs are read. Such a call throws and leaves `val` unchanged

### Declaration:
```
void quadriga_lib::sos_val(
    const arma::Mat<dtype> &freq,
    const arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Mat<dtype> *phase_b = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    dtype freq_scale_b = 1.0,
    bool uniform = false,
    int use_kernel = 0);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or empty =
  same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** *(optional)* — Frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0
- **`uniform`** *(optional)* — Output distribution: `false` = Normal(0,1), `true` = Uniform between 0 and 1
- **`use_kernel`** *(optional)* — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  throws if the requested kernel is unavailable

### Output:
- **`val`** — Values; one row per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_preset](#sos_preset) (frequencies and ACF of the predefined models)
- [sos_init_phases](#sos_init_phases) (random phases)
- [sos_map](#sos_map) (evaluation on a grid)
- [sos_randn](#sos_randn) (generator set-up and evaluation in one call)

