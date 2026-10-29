---
title: "MATLAB / Octave API Documentation for Quadriga-Lib v0.12.4"
author: "Stephan Jaeckel"
date: "29.10.2026"
lang: en-US
---

# General usage notes
- Each function has a 1-line short description, optional detailed notes, a Usage block, and Inputs/Outputs sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions live in the `+quadriga_lib` package and are called as `quadriga_lib.function_name(...)`.
- Numeric inputs accept any numeric type and are cast internally; default is `double` unless stated otherwise (e.g. `uint64` for some index arguments).
- MATLAB arrays are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for 3-D arrays; `[rows, cols]` for 2-D matrices; `[n]` for vectors.
- Parameters marked *(optional)* have defaults; all others are required. Output arguments are optional via `nargout`; outputs after the first are computed only when requested.
- Functions operating on arrayant data accept either a single struct (*struct mode*) or the individual fields as separate positional arguments (*split mode*). Split-mode signatures are single-frequency only; multi-frequency input requires struct mode.
- Invalid inputs (shape/domain) and I/O failures raise a MATLAB error; the error identifier indicates the category (`quadriga_lib:invalid_argument`, `quadriga_lib:runtime_error`).
- Index conventions: 1-based unless the field is explicitly called 0-based.
- Units: angles in radians (degrees only where stated, e.g. `*_deg`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Polarization transfer matrix `M`: 8 rows per path, interleaved real/imaginary, order `[ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]`. A 2-row form `[ReVV, ImVV]` is used for scalar (acoustic) fields.
- The canonical arrayant struct is defined in [arrayant_generate](#arrayant_generate). Mandatory fields: `e_theta_re/im`, `e_phi_re/im`, `azimuth_grid`, `elevation_grid`. Optional field defaults: `element_pos = zeros(3, n_elements)`, identity coupling, `center_freq = 299792458`, empty `name`. A struct array represents a frequency-dependent (multi-frequency) model with one entry per frequency.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).
- Kernel-selection parameters (`use_kernel`): `0` = auto (CUDA if available and problem large enough, else AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2, `3` = CUDA. Throws if the requested kernel is unavailable.
- `gpu_id` is only read when `use_kernel` resolves to CUDA.

# Function Index

| Function | Section | Line |
| --- | --- | --- |
| [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) | Array antenna functions | 121 |
| [arrayant_calc_directivity](#arrayant_calc_directivity) | Array antenna functions | 179 |
| [arrayant_combine_pattern](#arrayant_combine_pattern) | Array antenna functions | 218 |
| [arrayant_concat](#arrayant_concat) | Array antenna functions | 274 |
| [arrayant_copy_element](#arrayant_copy_element) | Array antenna functions | 304 |
| [arrayant_export_obj_file](#arrayant_export_obj_file) | Array antenna functions | 332 |
| [arrayant_generate](#arrayant_generate) | Array antenna functions | 361 |
| [arrayant_interpolate](#arrayant_interpolate) | Array antenna functions | 484 |
| [arrayant_qdant_read](#arrayant_qdant_read) | Array antenna functions | 556 |
| [arrayant_qdant_write](#arrayant_qdant_write) | Array antenna functions | 602 |
| [arrayant_rotate_pattern](#arrayant_rotate_pattern) | Array antenna functions | 663 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 716 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 778 |
| [channel_export_obj_file](#channel_export_obj_file) | Channel functions | 850 |
| [hdf5_create_file](#hdf5_create_file) | Channel functions | 886 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 915 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 979 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 1010 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 1038 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 1064 |
| [hdf5_write_channel](#hdf5_write_channel) | Channel functions | 1093 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1136 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 1173 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 1206 |
| [quantize_delays](#quantize_delays) | Channel functions | 1267 |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generation functions | 1324 |
| [get_channels_irs](#get_channels_irs) | Channel generation functions | 1404 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 1479 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 1551 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 1614 |
| [acdf](#acdf) | Channel statistics | 1689 |
| [calc_angular_spread](#calc_angular_spread) | Channel statistics | 1723 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 1764 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 1820 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 1850 |
| [conf_file_list](#conf_file_list) | Geometry-based stochastic model | 1919 |
| [conf_file_read](#conf_file_read) | Geometry-based stochastic model | 1960 |
| [conf_file_write](#conf_file_write) | Geometry-based stochastic model | 2029 |
| [los_probability](#los_probability) | Geometry-based stochastic model | 2103 |
| [lsf_draw](#lsf_draw) | Geometry-based stochastic model | 2207 |
| [lsf_init_phases](#lsf_init_phases) | Geometry-based stochastic model | 2302 |
| [lsp_generate](#lsp_generate) | Geometry-based stochastic model | 2401 |
| [o2i_loss](#o2i_loss) | Geometry-based stochastic model | 2523 |
| [path_loss](#path_loss) | Geometry-based stochastic model | 2590 |
| [scenpar_validate](#scenpar_validate) | Geometry-based stochastic model | 2719 |
| [ssf_clusters](#ssf_clusters) | Geometry-based stochastic model | 2777 |
| [ssf_draw](#ssf_draw) | Geometry-based stochastic model | 2859 |
| [ssf_init_phases](#ssf_init_phases) | Geometry-based stochastic model | 2904 |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | 2944 |
| [cart2geo](#cart2geo) | Math functions | 2966 |
| [fast_sincos](#fast_sincos) | Math functions | 2996 |
| [geo2cart](#geo2cart) | Math functions | 3026 |
| [interp](#interp) | Math functions | 3067 |
| [version](#version) | Miscellaneous / Tools | 3100 |
| [write_png](#write_png) | Miscellaneous / Tools | 3117 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 3146 |
| [cube](#cube) | Site-specific simulation tools | 3215 |
| [generate_diffraction_paths](#generate_diffraction_paths) | Site-specific simulation tools | 3249 |
| [icosphere](#icosphere) | Site-specific simulation tools | 3288 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 3314 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 3368 |
| [plane](#plane) | Site-specific simulation tools | 3425 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 3463 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 3490 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 3518 |
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | 3547 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 3645 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 3686 |
| [subdivide_triangles](#subdivide_triangles) | Site-specific simulation tools | 3731 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 3755 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 3782 |
| [sos_acf_2d](#sos_acf_2d) | Sum-of-sinusoids random number generator | 3949 |
| [sos_acf_approx](#sos_acf_approx) | Sum-of-sinusoids random number generator | 3979 |
| [sos_acf_estimate](#sos_acf_estimate) | Sum-of-sinusoids random number generator | 4013 |
| [sos_acfi](#sos_acfi) | Sum-of-sinusoids random number generator | 4065 |
| [sos_calc_mse](#sos_calc_mse) | Sum-of-sinusoids random number generator | 4089 |
| [sos_generate](#sos_generate) | Sum-of-sinusoids random number generator | 4140 |
| [sos_init_phases](#sos_init_phases) | Sum-of-sinusoids random number generator | 4206 |
| [sos_map](#sos_map) | Sum-of-sinusoids random number generator | 4240 |
| [sos_preset](#sos_preset) | Sum-of-sinusoids random number generator | 4277 |
| [sos_rand](#sos_rand) | Sum-of-sinusoids random number generator | 4316 |
| [sos_randi](#sos_randi) | Sum-of-sinusoids random number generator | 4374 |
| [sos_randn](#sos_randn) | Sum-of-sinusoids random number generator | 4434 |
| [sos_remove_bias](#sos_remove_bias) | Sum-of-sinusoids random number generator | 4491 |
| [sos_val](#sos_val) | Sum-of-sinusoids random number generator | 4567 |

---

# Array antenna functions

---
## arrayant_calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in
  each direction than the antenna sampling grid)
- The width is that of the main lobe, searched outwards from the maximum; side lobes above the threshold
  are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full
  circle; 360° when the whole cut is above the threshold (e.g. a beam pointing to a pole)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles;
  at most 180°
- Calculated per element, not per port; ignores element coupling

### Usage:
```
% Input as struct (struct mode)
[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( arrayant );

[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( arrayant, i_element, threshold_dB );

% Separate inputs (split mode)
[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = ...
    quadriga_lib.arrayant_calc_beamwidth( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid );

[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = ...
    quadriga_lib.arrayant_calc_beamwidth( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, ...
    i_element, threshold_dB );
```

### Inputs (struct mode):
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element index; 1-based; if not provided or empty, all elements are used; uint64; `[n_out]` or empty
- **`threshold_dB`** — Threshold in dB; default: 3 (equivalent to FWHM)

### Inputs (split mode):
- **`e_theta_re`** — e-theta field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, sorted; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, sorted; `[n_elevation]`
- **`i_element`** — Element index; 1-based; if not provided or empty, all elements are used; uint64; `[n_out]` or empty
- **`threshold_dB`** — Threshold in dB; default: 3 (equivalent to FWHM)

### Outputs:
- **`beamwidth_az`** — Azimuth beamwidth in degree; `[n_out, n_freq]`; with `n_out = n_elements` when `i_element` is omitted/empty
- **`beamwidth_el`** — Elevation beamwidth in degree; `[n_out, n_freq]`
- **`az_point_ang`** — Azimuth pointing angle of the main beam in degree; `[n_out, n_freq]`
- **`el_point_ang`** — Elevation pointing angle of the main beam in degree; `[n_out, n_freq]`

### See also:
- [arrayant_combine_pattern](#arrayant_combine_pattern) (to apply element coupling before calculating directivity)
- [arrayant_calc_directivity](#arrayant_calc_directivity) (directivity in dBi of array antenna elements)

---
## arrayant_calc_directivity
Calculates the directivity in dBi of array antenna elements

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi

### Usage:
```
% Input as struct (struct mode)
directivity = quadriga_lib.arrayant_calc_directivity(arrayant);
directivity = quadriga_lib.arrayant_calc_directivity(arrayant, i_element);

% Separate inputs (split mode)
directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid);

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, i_element);
```

### Inputs (struct mode):
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element index; 1-based; if not provided or empty, all elements are used; uint64; `[n_out]` or empty

### Inputs (split mode):
- split mode accepts `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid`
  in place of arrayant; see [arrayant_calc_beamwidth](#arrayant_calc_beamwidth)

### Output:
- **`directivity`** - Directivity of the antenna pattern in dBi; `[n_out, n_freq]`;
  with `n_out = n_elements` when `i_element` is omitted/empty.

### See also:
- [arrayant_combine_pattern](#arrayant_combine_pattern) (to apply element coupling before calculating directivity)
- [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) (calculates the beam width of array antennas)

---
## arrayant_combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Usage:
```
% Input as struct (struct mode)
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in );
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in, center_freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );

% Separate outputs, struct input (single-freq only)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, ...
    coupling_im, center_freq, name] = quadriga_lib.arrayant_combine_pattern( arrayant_in );

% Separate inputs (split-mode, single-freq only); propagation_speed is the 16th input
arrayant_out = quadriga_lib.arrayant_combine_pattern( [], center_freq_new, azimuth_grid_new, elevation_grid_new, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, ...
    coupling_im, center_freq, name, propagation_speed );
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`center_freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is 
  recomputed (interpolated) at each requested frequency. If omitted or empty, the function uses each input 
   entry's `center_freq` unchanged.
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; sorted; defaults to input grid
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; sorted; defaults to input grid
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics;
  5th input in struct mode, 16th input with separate inputs; default: 299792458
- **`e_theta_re`** ... **`elevation_grid`** — Required inputs for split mode
- **`element_pos`** ... **`name`** — Optional inputs for split mode

### Outputs:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [arrayant_generate](#arrayant_generate). Single struct when both input and
  `freq` describe a single frequency; struct array of size `numel(freq)` (or `numel(arrayant_in)`
  when `freq` is omitted) otherwise.
- **`e_theta_re`, ..., `name`** — Separate-field outputs; **only available for single-frequency
  results** (single-struct input with scalar/omitted `freq`, or separate-input mode).

### See also:
- [arrayant_generate](#arrayant_generate) (for field layout in the arrayant struct)
- [arrayant_rotate_pattern](#arrayant_rotate_pattern) (for changing the orientation of elements before combining)
- [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) (calculates the beam width of array antennas)
- [arrayant_calc_directivity](#arrayant_calc_directivity) (directivity in dBi of array antenna elements)

---
## arrayant_concat
Concatenate two arrayant structs into a single one

- Concatenates all elements from `arrayant_in2` onto `arrayant_in1` along the element dimension; 
  `element_pos` matrices are joined horizontally
- Both inputs must share identical azimuth and elevation sampling grids
- Coupling is assembled block-diagonally: elements from `arrayant_in1` connect only to ports from 
  `arrayant_in1`, elements from `arrayant_in2` only to ports from `arrayant_in2`
- `center_freq` and `name` are inherited from `arrayant_in1`
- Supports multi-frequency arrayant models: when both inputs are struct arrays, they must have the 
  same number of entries and matching `center_freq` at each index; concatenation is performed per 
  entry and a struct array of equal size is returned
- Output struct shape matches the input shape (scalar struct -> scalar struct, struct array -> struct array)

### Usage:
```
arrayant_out = quadriga_lib.arrayant_concat( arrayant_in1, arrayant_in2 );
```

### Inputs:
- **`arrayant_in1`** — Struct containing the first arrayant data; field layout as documented in 
  [arrayant_generate](#arrayant_generate); a struct array may contain a frequency-dependent  model
- **`arrayant_in2`** — Struct containing the second arrayant data; must match the sampling grids of
  `arrayant_in1`, and for multi-frequency the entry count and `center_freq` per entry

### Outputs:
- **`arrayant_out`** — Struct containing the combined arrayant data; same field layout as the inputs; 
  a struct array of equal size is returned for multi-frequency input

---
## arrayant_copy_element
Create copies of array antenna elements

- Copies a source element to one or more destination slots within an arrayant
- Array is resized if any destination index exceeds the current number of elements
- Coupling matrix entries for newly added elements are set to identity; existing coupling is preserved
- Supports multi-frequency arrayant models: when `arrayant_in` is a struct array, the same copy is
  applied to every entry and a struct array of equal size is returned
- If `source_element` is a vector, `dest_element` must have the same length; copies are performed
  pairwise as `source_element(i)` to `dest_element(i)`

### Usage:
```
arrayant_out = quadriga_lib.arrayant_copy_element( arrayant_in, source_element, dest_element );
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in
  [arrayant_generate](#arrayant_generate); a struct array may contain a frequency-dependent model
- **`source_element`** — Index of the source element(s); 1-based; uint64; scalar or `[n_copy]`
- **`dest_element`** — Index of the destination element(s); 1-based; uint64; scalar or `[n_copy]`;
  if `source_element` is a vector, must have the same length

### Outputs:
- **`arrayant_out`** — Struct containing the modified arrayant data; same field layout as
  `arrayant_in`; a struct array of equal size is returned for multi-frequency input

---
## arrayant_export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives a finer mesh
- Supports multi-frequency arrayant models: pass a struct array and select the entry to export
  via the `i_freq` argument

### Usage:
```
quadriga_lib.arrayant_export_obj_file( fn, arrayant, directivity_range, colormap,  object_radius, ...
   icosphere_n_div, i_element, i_freq );
```

### Inputs:
- **`fn`** — Output OBJ filename; must not be empty; filename must end in `.obj`
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in
  [arrayant_generate](#arrayant_generate); a struct array may contain a frequency-dependent model
- **`directivity_range`** — Dynamic range of the visualized directivity pattern in dB; default: 30 dB
- **`colormap`** — Colormap name; default: jet; Available: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
- **`object_radius`** — Radius of the exported object; default: 1 m
- **`icosphere_n_div`** — Icosphere subdivision count; higher = finer mesh; see [icosphere](#icosphere); default: 4
- **`i_element`** — Element indices to export; 1-based; uint64; empty = export all elements
- **`i_freq`** — Frequency index to export from a multi-frequency arrayant struct array; 1-based; 
  uint64; default: 1; must satisfy `1 <= i_freq <= n_freq`

### See also:
- [icosphere](#icosphere) (icosphere primitive used internally)

---
## arrayant_generate
Generates predefined array antenna models

- Dispatches to one of several C++ generator functions based on the `type` string
- Supported types: `omni`, `dipole` (or `short-dipole`), `half-wave-dipole`, `xpol`, `custom`,
  `ula`, `3GPP` (or `3gpp`), `multibeam`, `multibeam_sep`
- All positional arguments after `type` are optional and type-specific; use `[]` to skip and fall
  back to defaults

### Usage:
```
% Simple antennas (v-pol)
ant = quadriga_lib.arrayant_generate('omni', res);
ant = quadriga_lib.arrayant_generate('dipole', res);
ant = quadriga_lib.arrayant_generate('half-wave-dipole', res);

% Cross-polarized isotropic
ant = quadriga_lib.arrayant_generate('xpol', res);

% Custom 3dB beamwidth
ant = quadriga_lib.arrayant_generate('custom', res, freq, az_3dB, el_3dB, bfr);

% Uniform linear array (N horizontal elements, half-wavelength spacing by default)
ant = quadriga_lib.arrayant_generate('ula', res, freq, [], [], [], [], N, pol, [], spacing);

% Uniform linear array with custom per-element pattern struct
ant = quadriga_lib.arrayant_generate('ula', res, freq, [], [], [], [], N, [], [], spacing, [], [], [], [], pattern);

% 3GPP-NR array (default 3GPP element pattern)
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [], M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);

% 3GPP-NR array with custom element beamwidth
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, az_3dB, el_3dB, bfr, M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);

% 3GPP-NR array with custom per-element pattern struct
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [], M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh, pattern);

% Multi-beam M×N array (one combined beam / one beam per direction)
ant = quadriga_lib.arrayant_generate('multibeam', res, freq, az_3dB, el_3dB, bfr, M, N, pol, dir, spacing);
ant = quadriga_lib.arrayant_generate('multibeam_sep', res, freq, az_3dB, el_3dB, bfr, M, N, pol, dir, spacing);

% Separate outputs (2 to 11 outputs)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name] = quadriga_lib.arrayant_generate( ... );
```

### Mapping table with defaults:
| Position          | 2   | 3    | 4      | 5      | 6   | 7   | 8   | 9   | 10       | 11      | 12  | 13  | 14  | 15  | 16      |
| ----------------- | --: | ---: | -----: | -----: | --: | --: | --: | --: | -------: | ------: | --: | --: | --: | --: | ------: |
|  **Variable**     | res | freq | az_3dB | el_3dB | bfr |   M |   N | pol | tilt_dir | spacing | Mg  | Ng  | dgv | dgh | pattern |
|  **Type / Unit**  | deg |   Hz |    deg |    deg | lin | int | int | int |      deg |       λ | int | int | dbl | dbl |  struct |
| omni              |  10 | 300k |      — |      — |   — |   — |   — |   — |        — |       — |   — |   — |   — |   — |       — |
| dipole            |   1 | 300k |      — |      — |   — |   — |   — |   — |        — |       — |   — |   — |   — |   — |       — |
| half-wave-dipole  |   1 | 300k |      — |      — |   — |   — |   — |   — |        — |       — |   — |   — |   — |   — |       — |
| xpol              |  10 | 300k |      — |      — |   — |   — |   — |   — |        — |       — |   — |   — |   — |   — |       — |
| custom            |   1 | 300k |     90 |     90 |   0 |   — |   — |   — |        — |       — |   — |   — |   — |   — |       — |
| ula               |   1 | 300k |      — |      — |   — |   — |   1 |   1 |        — |       — |   — |   — |   — |   — |       ✓ |
| 3GPP              |   1 | 300k |     67 |     67 |   0 |   1 |   1 |   1 | tilt = 0 |     0.5 |   1 |   1 | 0.5 | 0.5 |       ✓ |
| multibeam         |   1 | 300k |    120 |    120 |   0 |   1 |   1 |   1 |  [0;0;1] |     0.5 |   — |   — |   — |   — |       — |
| multibeam_sep     |   1 | 300k |    120 |    120 |   0 |   1 |   1 |   1 |  [0;0;1] |     0.5 |   — |   — |   — |   — |       — |

### Inputs (common):
- **`type`** — Antenna model type; string (see usage above)
- **`res`** — Pattern sampling grid resolution in degrees; default: 10 for types omni+xpol, 1 otherwise
- **`freq`** — Center frequency in Hz; default: 299792458; equivalent to λ = 1 m

### Inputs (type custom, 3GPP, multibeam, multibeam_sep):
- **`az_3dB`** — Azimuth 3dB beamwidth in degrees; default: 90 for `custom`, 120 (per element) for `multibeam` and
  `multibeam_sep`; for `3GPP`, 0 or `[]` selects the 3GPP default element pattern
- **`el_3dB`** — Elevation 3dB beamwidth in degrees; same defaults as `az_3dB`
- **`bfr`** — Back-to-front gain ratio (linear); default: 0

### Inputs (type ula, 3GPP, multibeam, multibeam_sep):
- **`M`** — Number of vertical elements per panel; default: 1; ignored for `ula`
- **`N`** — Number of horizontal elements per panel; default: 1
- **`pol`** — Polarization mode; default: 1:
   | `pol` | Description                                              |
   | :---: | -------------------------------------------------------- |
   | 1     | Vertical polarization                                    |
   | 2     | H/V polarization (2NM elements)                          |
   | 3     | ±45° polarization (2NM elements)                         |
   | 4     | Vertical, vertical elements combined (N elements)        |
   | 5     | H/V, vertical elements combined (2N elements)            |
   | 6     | ±45°, vertical elements combined (2N elements)           |
- **`spacing`** — Inter-element spacing in wavelengths; default: 0.5

### Inputs (type 3GPP only):
- **`tilt`** — Electrical downtilt in degrees; applies to `pol = 4/5/6`; default: 0
- **`Mg`** — Number of vertically stacked panels; default: 1
- **`Ng`** — Number of horizontally stacked panels; default: 1
- **`dgv`** — Panel spacing in vertical direction in wavelengths; default: 0.5
- **`dgh`** — Panel spacing in horizontal direction in wavelengths; default: 0.5

### Inputs (type ula, 3GPP):
- **`pattern`** — Custom per-element pattern struct used for 3GPP or ULA; same format as
  outputs; overrides default 3GPP/ULA element pattern; only the radiation pattern is used: element
  positions, coupling and center frequency of the pattern are ignored, as are other struct fields

### Inputs (type multibeam, multibeam_sep):
- **`dir`** — Beam steering angles, `[3, n_beams]`; rows are `[azimuth_deg; elevation_deg; weight]`; 
  `multibeam` combines beams via MRT weighting, `multibeam_sep` produces one independent beam per column (weights ignored)

### Outputs:
- **`ant`** — Struct with fields:
   | Field            | Description                                     | Size                                   |
   | ---------------- | ----------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part              | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`     | e-theta field component, imaginary part         | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`       | e-phi field component, real part                | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`       | e-phi field component, imaginary part           | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, sorted          | `[n_azimuth]`                          |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, sorted    | `[n_elevation]`                        |
   | `element_pos`    | Element (x,y,z) positions in meters             | `[3, n_elements]`                      |
   | `coupling_re`    | Coupling matrix, real part                      | `[n_elements, n_ports]`                |
   | `coupling_im`    | Coupling matrix, imaginary part                 | `[n_elements, n_ports]`                |
   | `center_freq`    | Center frequency in Hz                          | scalar                                 |
   | `name`           | Name of the array antenna object                | string                                 |

   If `ant` is used as an input to other functions, fields `e_theta_re`, `e_theta_im`, `e_phi_re`, 
   `e_phi_im`, `azimuth_grid`, `elevation_grid` are mandatory; remaining fields are optional (defaults: 
   unit coupling, zero positions, 299792458 Hz).

---
## arrayant_interpolate
Interpolate polarimetric array antenna field patterns (single- and multi-frequency)

- Interpolates complex e-theta (V) and e-phi (H) field components at the requested azimuth / elevation angles.
- NaN or infinite angles give NaN in all outputs for that angle (and element); no error is raised
- Single-frequency mode: pass a 1-element struct (or arrayant data via separate inputs); returns up to 8 outputs 
  including the optional `dist`, `azimuth_loc`, `elevation_loc`, and `gamma` as matrices of size `[n_out, n_ang]`
- Multi-frequency mode is selected automatically when `arrayant` is a struct array with more than one 
  element or when `freq` is non-empty; for each target frequency, the two bracketing `center_freq` 
  entries are located and blended via SLERP.
- Separate arrayant inputs are accepted in single-frequency mode only

### Usage:
```
% Single-frequency, struct input
[V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
    quadriga_lib.arrayant_interpolate( arrayant, azimuth, elevation, element, orientation, element_pos );

% Single-frequency, separate arrayant inputs
[V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
    quadriga_lib.arrayant_interpolate( [], azimuth, elevation, element, orientation, element_pos, [], ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid );

% Multi-frequency, struct array input
[V_re, V_im, H_re, H_im] = quadriga_lib.arrayant_interpolate( arrayant_multi, azimuth, elevation, ...
    element, orientation, element_pos, freq );
```

### Inputs:
- **`arrayant`** — Struct (single-frequency) or struct array (multi-frequency) containing the arrayant data; 
  field layout as in [arrayant_generate](#arrayant_generate). Pass `[]` to provide the data via separate inputs (single-frequency only)
- **`azimuth`** — Azimuth angles in rad, in [-π, π]; single or double precision; `[1, n_ang]` for planar-wave mode 
  (same angles for all elements) or `[n_out, n_ang]` for per-element angles (spherical-wave mode)
- **`elevation`** — Elevation angles in rad, in [-π/2, π/2]; single or double; shape must match `azimuth`
- **`element`** — Element indices to interpolate; 1-based; duplicates allowed; defaults to `[1:n_elements]` 
  when empty; `[1, n_out]`, `[n_out, 1]`, or `[]`; uint32 or double
- **`orientation`** — Antenna orientation (bank, tilt, heading) in rad; East is the default broadside; 
  `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]`, `[3, n_out, n_ang]`, or `[]`
- **`element_pos`** — Override element positions in m; `[3, n_out]` or `[]`; falls back to `arrayant.element_pos` 
  (or zeros) when empty
- **`freq`** — Target frequencies in Hz; `[n_freq]`. When passing a struct array, `freq` may be omitted or `[]`, 
  in which case the `center_freq` values of the struct array entries are used  as target frequencies (no 
  interpolation between bands, one output slice per entry).

### Inputs (separate arrayant data, required when arrayant is [], single-frequency only):
- **`e_theta_re`** — e-theta real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth sample grid in rad, sorted, in [-π, π]; `[n_azimuth]`
- **`elevation_grid`** — Elevation sample grid in rad, sorted, in [-π/2, π/2]; `[n_elevation]`

### Outputs:
- **`V_re`** — Real part of the interpolated e-theta (vertical) field component;
  `[n_out, n_ang]` in single-freq mode, `[n_out, n_ang, n_freq]` in multi-freq mode
- **`V_im`** — Imaginary part of the e-theta component; same size as `V_re`
- **`H_re`** — Real part of the interpolated e-phi (horizontal) field component; same size as `V_re`
- **`H_im`** — Imaginary part of the e-phi component; same size as `V_re`
- **`dist`** *(single-frequency only)* — Effective distances between the antenna elements
  projected onto the wavefront plane; used for phase computation; `[n_out, n_ang]`
- **`azimuth_loc`** *(single-frequency only)* — Azimuth angles in the local (rotated) element
  frame in rad; `[n_out, n_ang]`
- **`elevation_loc`** *(single-frequency only)* — Elevation angles in the local element frame in
  rad; `[n_out, n_ang]`
- **`gamma`** *(single-frequency only)* — Polarization rotation angles in rad; `[n_out, n_ang]`

### See also:
- [arrayant_qdant_read](#arrayant_qdant_read) / [arrayant_qdant_write](#arrayant_qdant_write) (load / save arrayant data)
- [arrayant_generate](#arrayant_generate) (arrayant struct layout)
- [generate_speaker](#generate_speaker) (typical multi-frequency struct array source)

---
## arrayant_qdant_read
Reads array antenna data from QDANT files

- The QuaDRiGa array antenna exchange format (QDANT) is an XML format for storing antenna pattern data
- Without `id`, all entries are read: returns a struct array when the file has multiple entries (frequency-dependent model) 
  or a single struct when it has exactly one entry; multiple entries must form a frequency set (same number of elements,
  same azimuth and elevation grids, element positions and coupling size), otherwise
  an error asks to pass `id`
- Differs from Python, where `qdant_read(fn)` reads ID 1 only
- With `id`, a single entry is read; useful for picking one frequency from a multi-frequency file
- Separate-fields output (3 to 12 outputs) is only available when the result is a single entry 
  (i.e. `id` was provided, or the file contains exactly one entry)
- An `id` that is not in the file is an error; the first entry counts as ID 1 only if it has no `id` attribute
- Malformed files are an error, also when they declare more elements (`NoElements`) than the file has data for;
  elements without pattern data read as zeros (at most 2^24 zero values per entry)

### Usage:
```
% Multi-frequency read (struct array, all entries)
[ ant, layout ] = quadriga_lib.arrayant_qdant_read( fn );

% Single-frequency read (struct output)
[ ant, layout ] = quadriga_lib.arrayant_qdant_read( fn, id );

% Single-frequency read (separate fields)
[ e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name, layout ] = quadriga_lib.arrayant_qdant_read( fn, id );
```

### Inputs:
- **`fn`** — Path to the QDANT file; string; must not be empty
- **`id`** — 1-based ID of the antenna entry to read; pass `[]` or omit to read every entry in the file

### Outputs:
- **`ant`** — Arrayant struct (single entry) or struct array (multiple entries); field layout as documented in [arrayant_generate](#arrayant_generate)
- **`layout`** — Matrix of element IDs describing how entries are arranged in the file; datatype: uint32
- **`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid`, `element_pos`, 
  `coupling_re`, `coupling_im`, `center_freq`, `name`** — Separate-field outputs with contents and sizes as 
  in [arrayant_generate](#arrayant_generate); only available when the result is a single entry

### See also:
- [arrayant_qdant_write](#arrayant_qdant_write) (for writing QDANT data)
- [arrayant_generate](#arrayant_generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)

---
## arrayant_qdant_write
Writes array antenna data to QDANT files

- The QuaDRiGa array antenna exchange format (QDANT) is an XML format for storing antenna pattern data
- Multiple array antennas can be stored in the same file using distinct `id` values
- If writing to an existing file without specifying an `id`, the data is appended at the end and
  the returned `id_in_file` identifies its location in the file
- An optional `layout` can be provided to organize the data inside the file
- Without `layout`, the layout in the file is replaced by a row of all IDs in the file in file order;
  a custom layout written before is lost (pass it again with every write that should keep it)
- The file has single-precision-equivalent accuracy relative to the element peak: pattern
  magnitudes (dB) and phases (degree) are rounded so that every sample is within 6e-8 of the peak
  amplitude of its element; samples far below the peak are less accurate relative to their own
  amplitude (about 6e-4 at -80 dB, 6e-2 at -120 dB below the peak); positions and coupling are
  written as float values (at most 9 significant digits); sampling grids are rounded to 0.001
  degree; all non-zero pattern samples are written at any level, exact zeros as `-inf` dB
- Passing a struct array (multiple elements) writes a frequency-dependent model with sequential
  1-based IDs; in this mode appending to an exisiting file is not allowed and will cause an error

### Usage:
```
% Arrayant as struct
id_in_file = quadriga_lib.arrayant_qdant_write( fn, arrayant, id, layout );

% Arrayant as separate inputs
id_in_file = quadriga_lib.arrayant_qdant_write( fn, [], id, layout, e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name );
```

### Inputs:
- **`fn`** — Output QDANT filename; string; must not be empty
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate); 
  pass `[]` to provide the data via separate inputs instead; a struct array writes a frequency-dependent 
  model and requires `id` and `layout` to be omitted
- **`id`** — Target ID of the antenna inside the file; default: max-ID in existing file + 1 (or 1 if the file does not exist);
  ignored for multi-frequency model
- **`layout`** — Matrix organizing multiple antenna IDs within the file; must only reference IDs present in the file; uint32;
  default: a row of all IDs in the file (replaces a layout written before)

### Inputs (separate arrayant data, required when `arrayant` is `[]`):
- **`e_theta_re`** — e-theta field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, sorted; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, sorted; `[n_elevation]`
- **`element_pos`** — Element (x,y,z) positions; `[3, n_elements]`; default: zeros
- **`coupling_re`** — Coupling matrix, real part; `[n_elements, n_ports]`; default: identity
- **`coupling_im`** — Coupling matrix, imaginary part; `[n_elements, n_ports]`; default: zeros
- **`center_freq`** — Center frequency in Hz; default: 299792458
- **`name`** — Name of the array antenna object; string

### Outputs:
- **`id_in_file`** — ID assigned to the antenna in the file after writing; set to 0 in multi-frequency (struct array) mode

### See also:
- [arrayant_qdant_read](#arrayant_qdant_read) (for reading QDANT data)
- [arrayant_generate](#arrayant_generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)

---
## arrayant_rotate_pattern
Rotate antenna radiation patterns around the principal axes using Euler rotations

- Rotates pattern and/or polarization around x (bank), y (tilt), z (heading) axes
- Rotations applied in order x, y, z, composed as Rz·Ry·Rx (intrinsic Tait-Bryan)
- Adjusts the sampling grid for non-uniformly sampled antennas when `usage` is 0 or 1
- Coupling, `center_freq` and `name` are kept, also when the sampling grid is adjusted
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid
  spurious polarization effects

### Usage:
```
% Struct in / struct out
arrayant_out = quadriga_lib.arrayant_rotate_pattern(arrayant_in, x_deg, y_deg, z_deg, usage, i_element);

% Separate-field outputs (single-frequency results only)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name] = quadriga_lib.arrayant_rotate_pattern( ...
    arrayant_in, x_deg, y_deg, z_deg, usage, i_element);

% Separate inputs (single-frequency only)
arrayant_out = quadriga_lib.arrayant_rotate_pattern([], x_deg, y_deg, z_deg, usage, element, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name);
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array represents a frequency-dependent model
- **`x_deg`** — Rotation around x-axis (bank) in degrees; default: 0
- **`y_deg`** — Rotation around y-axis (tilt) in degrees; default: 0
- **`z_deg`** — Rotation around z-axis (heading) in degrees; default: 0
- **`usage`** — Rotation mode; default: 0
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

  Multi-frequency input never adjusts the grid (uniform-grid consistency across frequencies): 0 and 3
  rotate pattern and polarization, 1 and 4 the pattern only, 2 the polarization only (as in Python).
- **`i_element`** — 1-based element indices to rotate; defaults to all elements; `[n]`

### Outputs:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [arrayant_generate](#arrayant_generate). Single struct when input is a
  single struct; struct array of size `numel(arrayant_in)` when input is a struct array.
- **`e_theta_re`, ..., `name`** — Separate-field outputs; **only available for single-frequency
  results** (single-struct input).

---
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns one arrayant per frequency sample; each has a single element with the real-valued
  directivity pattern in `e_theta_re` and `center_frequency` set to the corresponding frequency.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining
  results via `append` and `element_pos`; crossover behavior emerges from overlapping bandpass
  responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity - 85) / 20)`.
- If `frequencies` is empty, third-octave band center frequencies are auto-generated from one band
  below `lower_cutoff` to one band above `upper_cutoff`, clipped to 20-20000 Hz.
- Speed of sound assumed to be 343 m/s.
- Driver models (`driver_type`):
  - `piston` - circular piston in baffle, `D(theta) = 2·J1(ka·sin theta)/(ka·sin theta)`,
    rotationally symmetric, narrows with increasing `ka`
  - `horn` - separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni
    below `horn_control_freq`
  - `omni` - frequency-independent omnidirectional pattern.
- Enclosure models (`radiation_type`):
  - `monopole` - no modification
  - `hemisphere` - sealed box with baffle-step transition, `f_baffle = c/(pi·sqrt(W·H))`
  - `dipole` - figure-8, `R = abs(cos(theta_off))` with sign inversion in rear hemisphere
  - `cardioid` - `R = 0.5·(1+cos(theta_off))`
- For `horn`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2pi·radius)`.

### Usage:
```
arrayant = quadriga_lib.generate_speaker( driver_type, radius, lower_cutoff, upper_cutoff, ...
    lower_rolloff_slope, upper_rolloff_slope, sensitivity, radiation_type, hor_coverage, ...
    ver_coverage, horn_control_freq, baffle_width, baffle_height, frequencies, ...
    angular_resolution );
```

### Inputs:
- **`driver_type`** - Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** - Effective radiating radius in m; cone/dome radius for piston, mouth radius for horn; default: 0.05
- **`lower_cutoff`** - Lower -3 dB bandpass frequency in Hz; default: 80
- **`upper_cutoff`** - Upper -3 dB bandpass frequency in Hz; default: 12000
- **`lower_rolloff_slope`** - Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth); default: 12
- **`upper_rolloff_slope`** - High-frequency rolloff in dB/octave; default: 12
- **`sensitivity`** - On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** - Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`; default: `hemisphere`
- **`hor_coverage`** - Horn horizontal coverage angle in degrees; `0` defaults to 90;  default: 90
- **`ver_coverage`** - Horn vertical coverage angle in degrees; `0` defaults to 60;  default: 60
- **`horn_control_freq`** - Horn pattern control frequency; `0` auto-derives from `radius`; default: 0
- **`baffle_width`** - Baffle width; used by `hemisphere` model; default: 0.15
- **`baffle_height`** - Baffle height; used by `hemisphere` model; default: 0.25
- **`frequencies`** - Frequency sample points; auto-generated third-octave bands if empty; `[n_freq]`
- **`angular_resolution`** - Azimuth and elevation sampling grid resolution in degrees; default: 5

### Outputs:
- **`arrayant`** - Struct array with one arrayant per frequency sample; directivity is stored in
  `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180 degree phase inversion; 
  see [arrayant_generate](#arrayant_generate) for struct fields; `[n_freq]`

---

# Channel functions

---
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

- Three dispatch modes depending on the dimensionality of coeff_re / coeff_im / delay  and the
  presence of `i_snap` — see the table below
- Single-frequency mode: computes the frequency-domain channel matrix at the carrier positions via
  DFT over time-domain path coefficients and delays
- Multi-frequency mode: interpolates complex channel coefficients from the input frequency grid `center_freq` to the
  output grid using SLERP, then applies delay-induced phase rotation per output carrier; only the first frequency slab
  of `delay` is used. The delay-induced phase is removed from the input coefficients before the interpolation, so
  channels from [get_channels_multifreq](#get_channels_multifreq) / [get_channels_spherical](#get_channels_spherical) must be generated with
  `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also when `delay` is
  relative, and with relative delays the phase of the TX-RX distance is interpolated across `center_freq`, which gives
  wrong phases (magnitudes stay right)
- Multi-snapshot mode: applies the single-frequency backend in parallel across snapshots via OpenMP;
  `i_snap` (1-based MATLAB indices) optionally selects a subset
- Carrier positions are specified one of two ways: as a normalized `pilot_grid` paired with `bandwidth`
  (where `0.0` corresponds to `center_freq(1)` and `1.0` to `center_freq(1) + bandwidth`), or as absolute
  frequencies via `carrier_freq`; supplying both pairs is an error
- Single-frequency and multi-snapshot modes with `carrier_freq` (same rules as the Python wrapper): the carriers may
  come in any order and the output keeps that order; the phase reference is `center_freq(1)` when given, else the
  lowest carrier; more than one carrier with a zero span (e.g. duplicates only) is an error
- Multi-frequency inputs always require `center_freq`; if `carrier_freq` is omitted it is derived as
  `center_freq(1) + pilot_grid · bandwidth`
- Multi-snapshot inputs require `center_freq` to be omitted or scalar (no combined multi-freq + multi-snap)
- Single-frequency inputs ignore `center_freq` when `pilot_grid` + `bandwidth` are given
- `delay` supports broadcasting: shape `[1, 1, n_path, ...]` applies the same delays to all RX/TX pairs
- Single-frequency and multi-snapshot modes: the phase `bandwidth · pilot_grid · delay` (in cycles) is computed in
  double precision and reduced to one period before sin / cos are evaluated in single precision; coefficients and sums
  are single precision; uses AVX2 where supported; double inputs are narrowed to float internally, results widened
  back. Multi-frequency mode computes in double precision

### Usage:
```
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, ...
    pilot_grid, bandwidth, center_freq, carrier_freq, i_snap );
```

### Dispatch modes:
| Mode             | Triggered by                             | `coeff_re` shape                  | Output shape                       |
|------------------|------------------------------------------|-----------------------------------|------------------------------------|
| Single-frequency | 3D `coeff_re` and `i_snap` omitted       | `[n_rx, n_tx, n_path]`            | `[n_rx, n_tx, n_carrier]`          |
| Multi-frequency  | 4D `coeff_re` and `i_snap` omitted       | `[n_rx, n_tx, n_path, n_freq]`    | `[n_rx, n_tx, n_carrier]`          |
| Multi-snapshot   | `i_snap` supplied (may be 0 for all)     | `[n_rx, n_tx, n_path, n_snap]`    | `[n_rx, n_tx, n_carrier, n_out]`   |

### Inputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path]` (single-freq) or
  `[n_rx, n_tx, n_path, n_freq]` (multi-freq) or `[n_rx, n_tx, n_path, n_snap]` (multi-snap)
- **`coeff_im`** — Imaginary part of channel coefficients; same shape as `coeff_re`
- **`delay`** — Path delays in seconds; same shape as `coeff_re`, optionally broadcast over RX/TX with
  shape `[1, 1, n_path]` or `[1, 1, n_path, ...]`
- **`pilot_grid`** — Normalized sub-carrier positions; `0.0` = center, `1.0` = center + bandwidth;
  must be paired with `bandwidth`; `[n_carrier, 1]`
- **`bandwidth`** — Total baseband bandwidth in Hz; must be paired with `pilot_grid`
- **`center_freq`** — Input sample frequencies; required for multi-frequency inputs; length must equal
  the 4th dimension of `coeff_re`; for multi-snap must be omitted or scalar; with `carrier_freq` in single-frequency
  and multi-snap mode, `center_freq(1)` is the phase reference (default: the lowest carrier); `[n_freq, 1]`
- **`carrier_freq`** — Absolute output carrier frequencies in Hz; cannot be combined with
  `pilot_grid` + `bandwidth`; `[n_carrier, 1]`
- **`i_snap`** — Triggers multi-snap mode. Scalar `0` processes all snapshots; a positive vector of 1-based
  indices processes the selected subset. Omitting this argument or passing `[]` keeps the function in single/multi-frequency mode.

### Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
  (single/multi-freq) or `[n_rx, n_tx, n_carrier, n_out]` (multi-snap)
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; same shape as `hmat_re`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency channel generator)
- [get_channels_multifreq](#get_channels_multifreq) (multi-frequency channel generator)

---
## channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits the total number of exported paths
- The function takes raw channel data fields directly; no MATLAB channel struct is needed

### Usage:
```
quadriga_lib.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max,  radius_min, ...
    n_edges, rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, i_snap );
```

### Inputs:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** — Max paths to export; 0 includes all paths above `gain_min`; default: 0
- **`gain_max`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5
- **`rx_pos`** — Receiver positions; `[3, n_snap]` or `[3, 1]`
- **`tx_pos`** — Transmitter positions; `[3, n_snap]` or `[3, 1]`
- **`no_interact`** — Number of interaction points of paths with the environment; uint32; `[n_path, n_snap]`
- **`interact_coord`** — Interaction coordinates; `[3, max(sum(no_interact)), n_snap]`
- **`center_freq`** — Center frequency in Hz; `[n_snap]` or scalar
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`i_snap`** — Snapshot indices to include; 1-based; range [1 ... n_snap]; empty exports all

### Outputs:
- This function writes the OBJ file directly to disk and does not return any data

---
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

- Initializes a new HDF5 file for storing wireless channel data
- Defines a 4D layout `(nx, ny, nz, nw)` where each index combination maps to one channel storage slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be altered later, except by reshaping while
  keeping the total slot count constant
- Errors if the target file already exists; delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

### Usage:
```
storage_space = quadriga_lib.hdf5_create_file( fn, storage_dims );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; string
- **`storage_dims`** — Size of the storage layout; vector with 1-4 elements, i.e. `[nx]`, `[nx, ny]`, 
  `[nx, ny, nz]` or `[nx, ny, nz, nw]`; default: `[65536, 1, 1, 1]`

### Output:
- **`storage_space`** — Actual storage dimensions used; `[4]`; uint32

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)

---
## hdf5_read_channel
Read one or more channel objects from an HDF5 file

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (omitted/empty = read full extent along that dimension)
- Slots are visited in column-major order and empty slots are skipped.
- Structured fields are stored in single precision in the file and returned in double.
- Unstructured datasets keep their stored type and shape.
- If no data is found, both outputs are empty `0x0` structs.
- Errors if the file is not a quadriga-lib channel file of version 3: files written by quadriga-lib
  0.12.4 and earlier are rejected; files modified with other HDF5 tools are not supported and may be
  rejected.
- Files of quadriga-lib 0.12.4 and earlier can be converted with the Python script
  `tools/hdf5_convert_v2.py` of the quadriga-lib sources (needs Python with numpy, h5py and
  quadriga-lib), e.g. `system('python tools/hdf5_convert_v2.py old.hdf5 new.hdf5')`.

### Usage:
```
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, snap );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`ix`** — 1-based slot indices along dimension X; scalar or vector; default: `1:nx`
- **`iy`** — 1-based slot indices along dimension Y; scalar or vector; default: `1:ny`
- **`iz`** — 1-based slot indices along dimension Z; scalar or vector; default: `1:nz`
- **`iw`** — 1-based slot indices along dimension W; scalar or vector; default: `1:nw`
- **`snap`** — Snapshot indices to read; 1-based; default: all snapshots. Only allowed
  when the total selection is a single slot.

### Outputs:
- **`chan`** — Struct array of length `N` (number of non-empty slots in the selection) with the channel data. Fields per element:
  | Field              | Description                                                              | Type / Size                         |
  | ------------------ | ------------------------------------------------------------------------ | ----------------------------------- |
  | `name`             | Channel name                                                             | String                              |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `[3, 1]` or `[3, n_snap]`           |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `[3, 1]` or `[3, n_snap]`           |
  | `coeff_re`         | Channel coefficients, real part                                          | `[n_rx, n_tx, n_path, n_snap]`      |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | `[n_rx, n_tx, n_path, n_snap]`      |
  | `delay`            | Propagation delays in seconds                                            | `[n_rx, n_tx, n_path, n_snap]`      |
  | `path_gain`        | Path gain before antenna, linear scale                                   | `[n_path, n_snap]`                  |
  | `path_length`      | Path length in m                                                         | `[n_path, n_snap]`                  |
  | `path_polarization`| Polarization transfer function, interleaved complex                      | `[8, n_path, n_snap]`               |
  | `path_angles`      | Departure and arrival angles [AOD, EOD, AOA, EOA] in rad                 | `[n_path, 4, n_snap]`               |
  | `fbs_pos`          | First-bounce scatterer positions                                         | `[3, n_path, n_snap]`               |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | `[3, n_path, n_snap]`               |
  | `no_interact`      | Number of interaction points per path; uint32                            | `[n_path, n_snap]`                  |
  | `interact_coord`   | Interaction coordinates                                                  | `[3, max(sum(no_interact)), n_snap]`|
  | `center_frequency` | Center Frequency in Hz                                                   | Scalar or `[n_snap]`                |
  | `initial_position` | Index of reference position; 1-based                                     | int32, scalar                       |
- **`par`** — Unstructured datasets as a `1xN` struct array matching `chan`. The field set is the
  **union** of dataset names across all loaded channels (without the `par_` prefix). For channels that
  do not contain a given field, the corresponding element is left empty. If no unstructured data is present anywhere,
  `par` is an empty `0x0` struct.

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout in the file)
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
## hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

- Reads a user-defined dataset stored under `prefix + name` (e.g. `"par_carrier_frequency"`)
- Type and shape of the returned data are determined by the dataset's HDF5 dataspace
- Returns an empty `[]` matrix if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [hdf5_read_channel](#hdf5_read_channel))
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element 
  types: single, double, int32, uint32, int64, uint64

### Usage:
```
dset = quadriga_lib.hdf5_read_dset( fn, location, name, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; string
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; string; default: `'par_'`

### Outputs:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; empty `[]` if the dataset is missing

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
## hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`
- Returned names have the prefix stripped
- Returns an empty cell array if no matching datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [hdf5_read_channel](#hdf5_read_channel))

### Usage:
```
names = quadriga_lib.hdf5_read_dset_names( fn, location, prefix );
```

### Inputs:
- **`fn`** — Path to the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`, `[ix, iy]`,
  `[ix, iy, iz]` or `[ix, iy, iz, iw]`; default:  `[1, 1, 1, 1]`
- **`prefix`** — Prefix used to identify unstructured datasets; string; default: `'par_'`

### Outputs:
- **`names`** — Names of all datasets at the given slot, with the prefix stripped; cell array of strings

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
## hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

- Returns the dimensions of the 4D channel slot grid stored inside an HDF5 file
- Returns `[0, 0, 0, 0]` if the file does not exist; errors if the file exists but is not a
  quadriga-lib channel file of version 3 (files written by quadriga-lib 0.12.4 and earlier are
  rejected; files modified with other HDF5 tools are not supported and may be rejected)
- Also reports which slots already contain data, so callers can locate free slots without scanning the file

### Usage:
```
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

### Input:
- **`fn`** — Filename of the HDF5 file; string

### Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `[4]`;  uint32
- **`has_data`** — Slot occupancy mask; `true` where data exists, `false` otherwise; `[nx, ny, nz, nw]`; logical

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)

---
## hdf5_reshape_layout
Reshape the storage layout inside an existing HDF5 file

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (files
  written by quadriga-lib 0.12.4 and earlier are rejected; files modified with other HDF5 tools are
  not supported and may be rejected)
- One writer per file at a time; files are not locked

### Usage:
```
storage_space = quadriga_lib.hdf5_reshape_layout( fn, storage_dims );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`storage_dims`** — New storage layout; vector with 1-4 elements,
  i.e. `[nx]`, `[nx, ny]`, `[nx, ny, nz]` or `[nx, ny, nz, nw]`; default: `[65536, 1, 1, 1]`

### Outputs:
- **`storage_space`** — New storage dimensions `[nx, ny, nz, nw]`; `[4]`; uint32

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the existing layout)

---
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file with a sensible default layout if it does not yet exist; an existing file must be
  a quadriga-lib channel file of version 3 (files of quadriga-lib 0.12.4 and earlier are rejected,
  see [hdf5_read_channel](#hdf5_read_channel))
- A warning is issued if any selected slot already contains data. That slot is replaced completely:
  its old structured data and all its unstructured datasets (also those written with
  [hdf5_write_dset](#hdf5_write_dset)) are removed. The file does not shrink; copy the channels into a new file to
  reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [hdf5_write_dset](#hdf5_write_dset))
- Each scalar location input is broadcast to all `numel(chan)` channels; each vector input must have exactly `numel(chan)` elements
- If the file does not exist, it is created with layout `[max(numel(chan), max(ix)), max(iy), max(iz), max(iw)]`

### Usage:
```
storage_dims = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`chan`** — Structured channel data; non-empty struct array; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; struct array of the same size as `chan`. Field names become HDF5 dataset
  names per slot (each prefixed with `par_`). Empty fields are skipped. Pass `[]` or omit to disable.
  Field values follow the type rules of [hdf5_write_dset](#hdf5_write_dset) (at most 3 dimensions).
- **`ix`** — 1-based slot index along dimension X; scalar or vector of length `numel(chan)`; default `1:numel(chan)`
- **`iy`** — 1-based slot index along dimension Y; scalar or vector of length `numel(chan)`; default `1`
- **`iz`** — 1-based slot index along dimension Z; scalar or vector of length `numel(chan)`; default `1`
- **`iw`** — 1-based slot index along dimension W; scalar or vector of length `numel(chan)`; default `1`

### Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data)

---
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must contain only alphanumeric characters and underscores
- The file must already exist (use [hdf5_create_file](#hdf5_create_file) first) and be a quadriga-lib channel file of
  version 3 (see [hdf5_read_channel](#hdf5_read_channel))
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [hdf5_write_channel](#hdf5_write_channel) to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element
  types: single, double, int32, uint32, int64, uint64
- Arrays with more than 3 dimensions, sparse and complex arrays are rejected
- Row vectors are stored as given (`[1, n]`)
- One writer per file at a time; files are not locked

### Usage:
```
storage_dims = quadriga_lib.hdf5_write_dset( fn, location, name, data, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; string
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; string; default: `'par_'`

### Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)

---
## qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count,
  CIR offsets, names, positions, orientations, and file version
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```
[ no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, center_freq, ...
     cir_pos, cir_orientation, orig_pos, orig_orientation ] = quadriga_lib.qrt_file_parse( fn );
```

### Input:
- **`fn`** — Path to the QRT file; string

### Outputs:
- **`no_cir`** — Number of channel snapshots per origin point; uint64 scalar
- **`no_orig`** — Number of origin points (TX); uint64 scalar
- **`no_dest`** — Number of destination points (RX); uint64 scalar
- **`no_freq`** — Number of frequency bands; uint64 scalar
- **`cir_offset`** — CIR offset per destination; uint64; `[no_dest]`
- **`orig_names`** — Names of origin points; cell array of strings; `[no_orig]`
- **`dest_names`** — Names of destination points; cell array of strings; `[no_dest]`
- **`version`** — QRT file version number; int32 scalar
- **`center_freq`** — Frequencies as stored in the file; GHz for EM mode (v4/v5), Hz for scalar mode (v6); single; `[no_freq]`
- **`cir_pos`** — CIR positions in Cartesian coordinates; single; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles; single; `[no_cir, 3]`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; single; `[no_orig, 3]`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; single; `[no_orig, 3]`

---
## qrt_file_read
Read ray-tracing CIR data from a QRT file

- Reads channel impulse response data from QRT files
- All output arguments are optional; MATLAB only computes outputs that are requested
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped

### Usage:
```
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, M, aod, eod, aoa, eoa, path_coord, no_int, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
```

### Inputs:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** — Snapshot indices; 1-based; uint64; `[n_out]` or empty; default: `[]` (read all; for a file
  without CIRs, all outputs are empty with `n_out = 0`)
- **`i_orig`** — Origin index; 1-based; uint64; scalar; default: 1
- **`downlink`** — If `true`, origin=TX, destination=RX; if `false`, roles are swapped; logical scalar; default: `true`
- **`normalize_M`** — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |
    Other values are rejected with an error.

### Outputs:
- **`center_freq`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** — First-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions;  Cell of length `n_out`; elements `[3, n_path]`
- **`path_gain`** — Path gain in linear scale; Cell of length `n_out`; elements `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; Cell of length `n_out`; elements `[n_path]`
- **`M`** — Polarization transfer matrix; Cell of length `n_out`;
  elements `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eod`** — Departure elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`aoa`** — Arrival azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eoa`** — Arrival elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`path_coord`** — Interaction coordinates per path; Cell of length `n_out`;
  elements Cell of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; uint32;
  Cell of length `n_out`; elements `[n_path]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single;
  Cell of length `n_out`; elements `[3, sum(no_int)]`
- **`interact_type`** — Interaction type codes (flat, concatenated across paths, matching `coord`); uint8;
  Cell of length `n_out`; elements `[sum(no_int)]`. Empty for v4 legacy files.

### See also:
- [arrayant_generate](#arrayant_generate) (for generating antenna arrays)
- [get_channels_planar](#get_channels_planar) (for embedding antennas using departure and arrival angles)
- [get_channels_spherical](#get_channels_spherical) (for embedding antennas using FBS/LBS positions)
- [get_channels_multifreq](#get_channels_multifreq) (for multi-frequency antenna embedding)

---
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α and δ^α, 
  where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband (incoherent power preservation)
- If all fractional per-tap offsets are below 0.01 or above 0.99, weight computation is skipped 
  (nearest neighor selection) but tap-selection logic still applies
- Input `delay` may be per-antenna `[n_rx, n_tx, n_path, n_snap]` or shared `[1, 1, n_path, n_snap]`; 
  shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0. Delays at or below
  `-tap_spacing / 2` raise an error, as do NaN, Inf and delays of 2^31 taps or more (`delay / tap_spacing < 2^31`,
  e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](#get_channels_spherical), [get_channels_multifreq](#get_channels_multifreq) and
  [get_channels_planar](#get_channels_planar)) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values

### Usage:
```
[ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
    tap_spacing, max_no_taps, power_exponent, fix_taps );
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`; non-negative
  with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate; default: 5e-9
- **`max_no_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; default: 0
  | Value | Meaning                                                                                         |
  | :---: | ----------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps, n_snap]`                       |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps, n_snap]` for shared delay input, `[n_rx, n_tx, n_taps, n_snap]` for per-antenna input |
  | 2     | Per snapshot; output delays `[1, 1, n_taps, n_snap]` for shared delay input, `[n_rx, n_tx, n_taps, n_snap]` for per-antenna input; each snapshot independent |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps, n_snap]`               |

### Outputs:
- **`coeff_re_q`** — Output coefficients, real part; `[n_rx, n_tx, n_taps, n_snap]`
- **`coeff_im_q`** — Output coefficients, imaginary part; `[n_rx, n_tx, n_taps, n_snap]`
- **`delay_q`** — Output delays in seconds; `[n_rx, n_tx, n_taps, n_snap]`, or `[1, 1, n_taps, n_snap]` for shared delay
  input with `fix_taps` 1 or 2

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
- Floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz; floors add attenuation
  (and the TGah shadow fading below 1 GHz) but no delay; the station is placed 3 m per floor above the AP
- Taps with zero power in the model tables are removed; the number of paths does not depend on distance, floors or walls
- Time evolution: each sub-path is rotated by `exp(+j 2 pi f_D t)`, i.e. a station moving towards the AP or a scatterer
  sees a positive Doppler shift `f_D`, as the channel of [get_channels_spherical](#get_channels_spherical) along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- NAN or negative value for any override parameter restores the model default, except `XPR_NLOS_linear` (see below)

### Usage:
```
chan = quadriga_lib.get_channels_ieee_indoor( ap_array, sta_array, channel_type, center_freq, ...
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, ...
   dist_m, n_floors, uplink, offset_angles, n_subpath, doppler_effect, seed, ...
   KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m, n_walls, wall_loss );
```

### Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling; see [arrayant_generate](#arrayant_generate)
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling; see [arrayant_generate](#arrayant_generate)
- **`channel_type`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`center_freq`** — Carrier frequency (finite); default: 5.25e9
- **`tap_spacing_s`** — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** — Channel observation time in seconds; must be finite; default: 0
- **`update_rate`** — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist_m`** — TX-to-RX distance(s) (finite); `[n_users]` or `[1]`; default: 4.99
- **`n_floors`** — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `[n_users]` or `[1]`; default: 0
- **`uplink`** — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: false
- **`offset_angles`** — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `[4, n_users]`; default: [] (auto-generate)
- **`n_subpath`** — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** — RNG seed for repeatability; `-1` uses the system random device; default: -1
- **`KF_linear`** — Overrides model KF (linear scale); default:  A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0; 
  applied to first tap only; breakpoint ignored when `KF_linear >= 0`
- **`XPR_NLOS_linear`** — Overrides NLOS cross-polarization ratio (linear scale); NAN or `[]` restores the default; must otherwise
  be finite and > 0 (0, negative and Inf throw); default: XPR NLOS: 2 (3 dB)
- **`SF_std_dB_LOS`** — Overrides LOS shadow fading std in dB (applied when d < dBP); default: 3 dB
- **`SF_std_dB_NLOS`** — Overrides NLOS shadow fading std in dB (applied when d >= dBP); default: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB
- **`dBP_m`** — Overrides breakpoint distance; default: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- **`n_walls`** — Number of walls per user TGax models; `[n_users]` or `[1]`; default: 0
- **`wall_loss`** — Penetration loss for a single wall; TGax defines 5 or 7; default: 5

### Output:
- **`chan`**
  Struct array of length `n_users` containing the channel data with the following fields:
  | Field              | Description                                                              | Type                                  |
  | ------------------ | ------------------------------------------------------------------------ | ------------------------------------- |
  | `name`             | Channel name                                                             | String                                |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `tx_orientation`   |  Transmitter orientation, Euler angles (AP for downlink, STA for uplink) | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `coeff_re`         | Channel coefficients, real part                                          | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `delay`            | Propagation delays in seconds                                            | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `path_gain`        | Path gain before antenna, linear scale                                   | Size: `[n_path, n_snap]`              |
  | `center_frequency` | Center Frequency in Hz                                                   | Scalar                                |

### See also:
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf">S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025</a>
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data to a HDF5 file)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data to a HDF5 file)

---
## get_channels_irs
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx),
  rows 7-8 = (H rx, H tx)
- Each segment is computed like [get_channels_spherical](#get_channels_spherical): NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths exceeding `threshold_dB`
- A segment without paths (all its path inputs empty) gives empty outputs `[n_rx, n_tx, 0]`
- If `active_path_in` is provided, it overrides `threshold_dB` for path selection
- Optional `ant_irs_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- If `center_freq == 0`, phase calculation is disabled and only delays are computed
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

### Usage:
```
[ coeff_re, coeff_im, delay, active_path_out, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, active_path_in, ant_irs_2 );
```

### Inputs:
- **`ant_tx`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [arrayant_generate](#arrayant_generate)
- **`ant_rx`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [arrayant_generate](#arrayant_generate)
- **`ant_irs`** — IRS antenna array (TX-facing side); `n_irs` = number of ports
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1, 1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center for TX → IRS paths; values shorter than the
  polyline (e.g. 0) use the polyline length; `[n_path_1, 1]`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2, 1]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center for IRS → RX paths; values shorter than the
  polyline (e.g. 0) use the polyline length; `[n_path_2, 1]`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `[8, n_path_2]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`irs_pos`** — IRS position; `[3, 1]`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`i_irs`** — IRS codebook port index, 1-based (one port per column of the IRS coupling matrix); default: `1`
- **`threshold_dB`** — Gain threshold in dB; path combinations below this are discarded; dB; default: `-140`
- **`center_freq`** — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: `0`
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`active_path_in`** — Bitmask selecting active path pairs; overrides `threshold_dB` when non-empty; logical; `[n_path_1 · n_path_2, 1]`
- **`ant_irs_2`** — Second IRS antenna array for the RX-facing side; enables asymmetric IRS patterns

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** — Bitmask indicating which path combinations were included in the output; logical; `[n_path_1 · n_path_2, 1]`
- **`aod`** — Azimuth of departure; `[n_rx, n_tx, n_path_irs]`
- **`eod`** — Elevation of departure; `[n_rx, n_tx, n_path_irs]`
- **`aoa`** — Azimuth of arrival; `[n_rx, n_tx, n_path_irs]`
- **`eoa`** — Elevation of arrival; `[n_rx, n_tx, n_path_irs]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-segment spherical-wave channel)
- [get_channels_planar](#get_channels_planar) (single-segment planar-wave channel)
- [arrayant_generate](#arrayant_generate) (antenna array generator)
- [combine_irs_coord](#combine_irs_coord) (coordinate setup for IRS geometry)

---
## get_channels_multifreq
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [get_channels_spherical](#get_channels_spherical) with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (from each `tx_array.center_freq`), RX array, input samples (`freq_in`), and output (`freq_out`)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `M` is interpolated via SLERP per complex entry pair to preserve phase
- Extrapolation clamps to the nearest frequency entry on all four grids
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `M` accepts 8 rows (full polarimetric: the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx),
  rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry.
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs), identical to antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 1 holds the LOS path
  (zero power if no LOS path was found; LOS = the path whose length is closest to the TX-RX distance, within 0.1 mm);
  a LOS path found in the input is moved to slot 1 and leaves a zero-power path in its original slot
  (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0, n_freq_out]`; with
  `add_fake_los_path = true` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [quantize_delays](#quantize_delays) rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [baseband_freq_response](#baseband_freq_response) in multi-frequency mode removes the phase of `delay` before it interpolates across
  `center_freq`, so it needs channels generated with `use_absolute_delays = true`; with relative delays, the phase of
  the TX-RX distance stays in the coefficients, is interpolated across the frequency grid and gives wrong phases

### Usage:
```
[ coeff_re, coeff_im, delay ] = quadriga_lib.get_channels_multifreq( tx_array, rx_array, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    freq_in, freq_out, use_absolute_delays, add_fake_los_path, propagation_speed );
```

### Inputs:
- **`tx_array`** — Multi-frequency TX arrayant struct array; one entry per input frequency, see [arrayant_generate](#arrayant_generate)
- **`rx_array`** — Multi-frequency RX arrayant struct array; one entry per input frequency, see [arrayant_generate](#arrayant_generate)
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Linear-scale path gains per input frequency; `[n_path, n_freq_in]`
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `[n_path, 1]`
- **`M`** — Polarization transfer matrix, row order as above; `[8, n_path, n_freq_in]` (full pol) or
  `[2, n_path, n_freq_in]` (scalar pressure)
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `[n_freq_in, 1]`
- **`freq_out`** — Target output frequencies; `[n_freq_out, 1]`
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `false`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_out, n_freq_out]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_out, n_freq_out]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_out, n_freq_out]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency equivalent)
- [generate_speaker](#generate_speaker) (acoustic source construction)

---
## get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX element pairs across `n_path` propagation paths.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based (angles ignored).
- Polarization transfer matrix `M` must be normalized; it is the 2x2 Jones matrix (rows = RX polarisation V/H,
  columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- If `add_fake_los_path` is true, one output path is always added (output size `n_path+1`): slot 1 holds the LOS path,
  with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the TX-RX
  distance; it is moved to slot 1 and its original slot (input index + 1) holds a zero-power path. All other paths
  keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- Setting `center_freq = 0` disables phase calculation (delays still computed).
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance (between the phase centres) from all path
  lengths before converting to delay; per-element delays of pairs that are closer to each other than the phase
  centres are then negative, down to about -(half the TX array size + half the RX array size) / speed of light; [quantize_delays](#quantize_delays) rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.

### Usage:
```
[ coeff_re, coeff_im, delay, rx_Doppler ] = quadriga_lib.get_channels_planar( tx_array, rx_array, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    center_freq, use_absolute_delays, add_fake_los_path );
```

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [arrayant_generate](#arrayant_generate)
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [arrayant_generate](#arrayant_generate)
- **`aod`** — Departure azimuth angles; rad; `[n_path, 1]`
- **`eod`** — Departure elevation angles; rad; `[n_path, 1]`
- **`aoa`** — Arrival azimuth angles; rad; `[n_path, 1]`
- **`eoa`** — Arrival elevation angles; rad; `[n_path, 1]`
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`
- **`path_length`** — Total path lengths from TX to RX phase center; `[n_path, 1]`
- **`M`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`center_freq`** — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; Default: `false`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = moving toward path, negative = away; `[1, n_path(+1)]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (spherical wave variant accounting for per-element angle differences)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (for generating IEEE compliant channels using `get_channels_planar` internally)
- [arrayant_generate](#arrayant_generate) (antenna array generator)
- [baseband_freq_response](#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed grid)

---
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [coord2path](#coord2path) or [qrt_file_read](#qrt_file_read) with more than two interactions) is added to the FBS-LBS segment. Delays and
  phases follow this length; the per-element offsets follow the FBS / LBS geometry.
- If `center_freq == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [quantize_delays](#quantize_delays) maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == true`, one output path is always added (`n_path + 1` outputs): slot 1 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 1 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair.

### Usage:
```
[ coeff_re, coeff_im, delay, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( tx_array, rx_array, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    center_freq, use_absolute_delays, add_fake_los_path, use_avx2 );
```

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [arrayant_generate](#arrayant_generate)
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [arrayant_generate](#arrayant_generate)
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `[n_path, 1]`
- **`M`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`center_freq`** — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; Default: `false`
- **`use_avx2`** — If `true`, use AVX2 for antenna interpolation; reduces accuracy to single-precision interpolation, 
  equivalent precision loss to the AVX2 kernel in arrayant_interpolate; default: `false`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`aod`** — Azimuth of departure; `[n_rx, n_tx, n_path(+1)]`
- **`eod`** — Elevation of departure; `[n_rx, n_tx, n_path(+1)]`
- **`aoa`** — Azimuth of arrival; `[n_rx, n_tx, n_path(+1)]`
- **`eoa`** — Elevation of arrival; `[n_rx, n_tx, n_path(+1)]`

### See also:
- [get_channels_planar](#get_channels_planar) (planar wave variant)
- [get_channels_irs](#get_channels_irs) (for IRS-assisted communication)
- [arrayant_generate](#arrayant_generate) (antenna array generator)
- [baseband_freq_response](#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed grid)

---

# Channel statistics

---
## acdf
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each 
  column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- A data set without any finite value gets a NaN column in `cdf_per_set` and is left out of `cdf_avg`, `mu`
  and `sig`; if no set has finite values (possible only with `bins_in`), `cdf_avg`, `mu` and `sig` are NaN
- If `bins` is empty, equally spaced bins spanning the data range are generated

### Usage:
```
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins_in`** — Bin centers; used as-is if non-empty (must be finite and strictly ascending, else an
  error is raised); if empty, equally spaced bins spanning the data range are generated; `[n_bins_in]` or
  empty; default: `[]`
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins_in` 
  are provided; default: 201

### Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data (NaN for a set without finite values); `[n_bins_out, n_sets]`
- **`bins_out`** — Auto-generated bins; copy of `bins_in` when non-empty `bins_in` are provided;
  `[n_bins_out = n_bins]` or `[n_bins_out = n_bins_in]`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across the data sets with finite values; `[n_bins]`
- **`mu`** — Mean of the 0.1–0.9 quantiles across the data sets with finite values; `[9]`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values; `[9]`

---
## calc_angular_spread
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles; each column
  of `az`/`el`/`powers` is one CIR; zero-power paths can be used to pad CIRs with fewer paths
- RMS spread formula: `sqrt(sum(pw .* d^2))` where `d` are wrapped deviations from the
  circular mean (3GPP TR 38.901 second-moment definition)
- When `wrapping = true`, the power-weighted mean direction is computed in Cartesian
  coordinates and all paths are rotated so the centroid lies on the equator before computing
  spreads, avoiding pole singularity artifacts
- When `wrapping = false`, spreads are computed directly from raw angles; `orientation` is
  zero and `phi`/`theta` equal the input `az`/`el`
- When `calc_bank_angle = true`, an optimal bank angle maximizing azimuth spread is derived
  analytically from eigenvectors of the 2x2 power-weighted covariance matrix of centered
  angles; only used when `wrapping = true`
- When `quantize > 0`, paths within that angular distance are grouped and their powers
  summed before computing spreads

### Usage:
```
[ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spread( az, el, powers, ...
    wrapping, calc_bank_angle, quantize );
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_path, n_cir]`
- **`el`** — Elevation angles; range -pi/2 to pi/2; `[n_path, n_cir]`
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`wrapping`** — If true, enables spherical rotation; default: false
- **`calc_bank_angle`** — If true, computes optimal bank angle analytically; only used when `wrapping = true`; default: false
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0 (no quantization);
  otherwise it must be finite and >= 1e-3 deg, else an error is raised

### Outputs:
- **`as`** — RMS azimuth angular spread; `[n_cir]`
- **`es`** — RMS elevation angular spread; `[n_cir]`
- **`orientation`** — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `[3, n_cir]`
- **`phi`** — Rotated azimuth angles; `[n_path, n_cir]`
- **`theta`** — Rotated elevation angles; `[n_path, n_cir]`

---
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method: co-pol 
  and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default
- Polarization transfer matrix `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with interleaved real/imaginary parts, 8 rows per path: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its
  receive and transmit polarisation: `M_vv` = rows 1-2, `M_hv` = rows 3-4, `M_vh` = rows 5-6, `M_hh` = rows 7-8
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `M` as get_channels_spherical couples antenna patterns,
  `c(r, t) = r^T M t` (plain transpose), with the same circular vectors at both ends, each in that antenna's own
  (e_theta, e_phi) frame: L = (V, H) = (1, +j) / sqrt(2) (LHCP), R = (1, -j) / sqrt(2) (RHCP); names after the IEEE
  definition for the time convention exp(+j 2 pi f t) of the library's coefficients exp(-j 2 pi f tau).
  `M_<rx><tx>`: `M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2`, `M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2`,
  `M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2` (receive L, transmit R), `M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2`.
  Co-polar means the same circular vector at both ends: the LOS matrix `diag(1, -1)` is co-polar (handedness
  kept), `diag(1, 1)` (one mirrored component, as by a single reflection) is cross-polar
- LHCP-XPR and RHCP-XPR are ratios per receive polarisation, like V-XPR and H-XPR
- Normalization of `M` does not affect XPR (cancels in the ratio) but does affect `pg`
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR is set to 0
- TX/RX positions may be fixed `[3, 1]` or mobile `[3, n_cir]`

### Usage:
```
[ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los, window_size );
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`M`** — Polarization transfer matrices with interleaved real/imag parts; `[8, n_path, n_cir]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, n_cir]`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`include_los`** — If true, includes LOS and near-LOS paths in the XPR calculation; default: false
- **`window_size`** — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded when
  `include_los = false`; default: 0.01

### Outputs:
- **`xpr`** — XPR on linear scale; `[n_cir, 6]`; columns:
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 1   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 2   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_vh)^2) (receive V)          |
   | 3   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_hv)^2) (receive H)          |
   | 4   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 5   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_LR)^2) (receive L)       |
   | 6   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_RL)^2) (receive R)       |
- **`pg`** — Total path gain summed over all paths (including LOS) as 
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir]`

---
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; the default threshold
  of 100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have
  their powers summed before computing the spread; binning is applied before the spread calculation

### Usage:
```
[ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold, granularity );
```

### Inputs:
- **`delays`** — Delays in [s] per CIR; `[n_path, n_cir]`
- **`powers`** — Path powers on linear scale in [W]; `[n_path, n_cir]`
- **`threshold`** — Power threshold in [dB] relative to strongest path; paths below threshold are excluded; default: 100
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0 (no grouping); must be finite
  and non-negative; when > 0, all delays must be finite and the delay span / granularity must be below 2^53, else an
  error is raised

### Outputs:
- **`ds`** — RMS delay spread in [s] per CIR; `[n_cir]`
- **`mean_delay`** — Mean delay in [s] per CIR; `[n_cir]`

### See also:
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed tap grid)
- [calc_rician_k_factor](#calc_rician_k_factor) (for calculating K-factor)

---
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

- KF = LOS power / NLOS power; LOS paths are those with length ≤ `dTR + window_size`, where
  `dTR` is the direct TX-RX distance
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is
  set to 0
- TX/RX positions may be fixed `[3, 1]` (reused for all snapshots) or mobile `[3, n_cir]`

### Usage:
```
[ kf, pg ] = quadriga_lib.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size );
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, n_cir]`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS; default: 0.01

### Outputs:
- **`kf`** — Rician K-Factor on linear scale; `[n_cir]`
- **`pg`** — Total path gain (sum of all path powers) in [W]; `[n_cir]`

---

# Geometry-based stochastic model

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of
environment, the *scenario* (e.g. urban macro-cell with or without line of sight), instead of from
the geometry of one site. The functions of this section implement the model of 3GPP TR 38.901,
clause 7.5 steps 1 to 7: path loss, shadow fading, LOS probability, O2I penetration loss, the
large-scale parameters (LSPs) of each link, i.e. the spreads of delays and angles, the K-factor and
the cross-polarization ratio, and the clusters.

The statistics of a scenario are read from a conf file by [conf_file_read](#conf_file_read). The folder
`data/config` holds 23 files for TR 38.901 v19.5.0; [scenpar_validate](#scenpar_validate) completes a parameter set
from another source, [conf_file_list](#conf_file_list) finds the files and [conf_file_write](#conf_file_write) writes them.

Per drop (layout): [o2i_loss](#o2i_loss) gives the penetration loss and the indoor distance of each indoor
UT; then [los_probability](#los_probability) evaluates the LOS probability at the outdoor distance `d2D - d2D-in`.
The caller draws the LOS state and the indoor state with [sos_rand](#sos_rand) / [sos_randn](#sos_randn) at the
decorrelation distances of the fields `LOS_state_lambda` and `IO_state_lambda`, and the O2I
variables at the distances of TR 38.901 clause 7.6.3.3 (see [o2i_loss](#o2i_loss)). The two state distances
apply to all links of a scenario and are stated once in the TR 38.901 files: `LOS_state_lambda` in
the NLOS file, `IO_state_lambda` in the O2I files; 0 in the other files means "not stated here".
Per BS site and scenario (builder): [lsf_init_phases](#lsf_init_phases) draws the random phases once; [lsf_draw](#lsf_draw)
evaluates them at the links (columns 1 DS, 2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA, 8 XPR, 9 absolute TOA: Normal;
10 ground permittivity, 11 UMa environment height: Uniform; each at the decorrelation distance of
its field); [path_loss](#path_loss) returns the path loss and the SF standard deviation; [lsp_generate](#lsp_generate)
turns the random values into the LSPs of each link and frequency.

The small-scale part (clause 7.5 steps 5 to 7, cluster level): [ssf_init_phases](#ssf_init_phases) draws the phases
of the cluster random values per site, scenario and frequency from the LSF seed, [ssf_draw](#ssf_draw)
evaluates them at the links, and [ssf_clusters](#ssf_clusters) turns the LSPs and random values into cluster
delays, powers and angles with the method `'38.901'` (clause 7.5), `'38.901_MF'` (clause 7.6.5.1) or
`'quadriga'` (spatially consistent, frequency-independent). The ray level (steps 8 to 12) follows in
a later version.

Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default; QuaDRiGa's behaviour is available
through an argument where it is still wanted (3D correlation, symmetric square root of the LSP
cross-correlation matrix, no clipping of the spreads, outdoor path-loss distance of O2I links). Where
QuaDRiGa's behaviour was an error, only the corrected rule is implemented.

Units: positions in m, frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power
ratios, angular spreads in degrees.

---
## conf_file_list
List the scenarios defined by the conf files in a set of directories

- Lists the regular files and symbolic links to regular files whose name ends in `.conf`
  (case-sensitive) and is longer than `.conf`, not recursive. The scenario name is the file name
  without this extension, whatever characters it has.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order
  of their names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first
  directory wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = true` each file is read and one entry is added right after the file for
  every line `ShortName = name` that is not a comment. Other lines are not examined; a UTF-8 byte
  order mark at the start of the file is skipped.
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`,
  `_`, or if a listed file cannot be opened for reading the short names
- A directory that does not exist, is not a directory or cannot be listed is skipped
- An empty string `''` is the current directory

### Usage:
```
[ scenarios, file_names, file_dirs ] = quadriga_lib.conf_file_list( dirs, parse_shortnames );
```

### Inputs:
- **`dirs`** — Directories, searched in this order; one string or a cell array of strings; an empty
  cell array lists nothing
- **`parse_shortnames`** — `true`: read the files and add their short names; `false`: list the files
  only, which is faster; default: true

### Outputs:
- **`scenarios`** — Scenario names and short names; cell array of strings; `[n_scen, 1]`
- **`file_names`** — File name including `.conf` for each entry; cell array of strings;
  `[n_scen, 1]`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs`; cell array of
  strings; `[n_scen, 1]`

### See also:
- [conf_file_read](#conf_file_read) (read one of the files)

---
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank
  lines are allowed; line ends can be LF or CRLF; space and tab are white space; a UTF-8 byte order
  mark at the start of the file is skipped
- Keys are case-sensitive names of the characters `A-Z`, `a-z`, `0-9` and `_`; there are four kinds:
  - Scenario parameters: 171 numeric keys with one number each. Keys that are not in the file have
    their default value.
  - `SubpathMethod`: one of the words `legacy`, `Laplacian`, `mmMAGIC`; default `legacy`
  - Path-loss parameters: every key that starts with `PL_`. `PL_model` has one name as value. All
    others have one or more numbers separated by white space.
  - `ShortName`: an alias for the scenario, one name; can occur several times with different values
- Any other key is ignored and reported in `ignored_keys`; its value is not examined
- A key that is stated more than once has its last value
- Numbers are decimal with an optional sign, an optional decimal point and an optional exponent
  (`e` or `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted
  in any case
- A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the
  smallest subnormal reads as `0` (or `-0`)
- Numbers are read independent of the locale (the decimal point is always `.`)
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 171 numeric keys are 138 keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys`; the permittivity is complex and is set in the simulation
- An error is raised if the file cannot be opened or is a directory, and, with file name and line
  number, for a line that is not blank, not a comment and not a well-formed assignment: no `=`, more
  or less than one name before the `=`, a missing value, a word where a number is required or the
  reverse, a second value or other text after the value, `NaN`, complex numbers, a number out of
  range, a `PL_` key without a name, a `ShortName` value stated twice
- With `validate = true`, a value that fails the checks of [scenpar_validate](#scenpar_validate) is an error as well
- A correlation matrix that is not positive definite is not an error; see `lsp_xcorr_chk`

### Usage:
```
[ scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys ] = ...
    quadriga_lib.conf_file_read( fn, validate );
```

### Inputs:
- **`fn`** — Path to the conf file; string
- **`validate`** — `true`: apply the value checks of [scenpar_validate](#scenpar_validate); `false`: only the line
  grammar is checked; default: true

### Outputs:
- **`scenpar`** — Scenario parameters; struct with 172 fields in the QuaDRiGa field order: the 138
  numeric keys of QuaDRiGa with `SubpathMethod` (string) as field 107, then the 33 new keys; all
  numeric fields are scalars
- **`plpar`** — Path-loss parameters; struct with the field `model` (string, only if the file states
  `PL_model`) followed by one field per `PL_` key in file order, named without the prefix `PL_`;
  each is a row vector with one or more values; `[]` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; logical
- **`short_names`** — Values of all `ShortName` lines in file order; cell array of strings;
  `[n_short, 1]`
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; cell array
  of strings; `[n_ignored, 1]`

### See also:
- [scenpar_validate](#scenpar_validate) (value checks, defaults and the list of validation rules)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)
- [lsf_init_phases](#lsf_init_phases) (first step of the large-scale model with the scenario parameters)
- [path_loss](#path_loss) (evaluates the path-loss model and its parameters)

---
## conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files
  shipped with QuaDRiGa, so that [conf_file_read](#conf_file_read) returns the same values bit for bit
- Header: `% Config file for scenario "name"` with the file name without the extension `.conf`,
  then the reference `ref`, the validity range and one line `ShortName = name` per short name
- Up to 10 sections follow, each with a header and, if the entry of `ref_sec` is not empty, a line
  `% See: ref; ref_sec`. Entry in `ref_sec` and write order:
  - 1: Large scale distributions, with an overview formula per LSP; 2: Large-Scale fading
    decorrelation distances; 3: Inter-parameter correlations, with the correlation matrix as
    comment; 4: Model parameters; 9: EoD offset; 5: Decorrelation distance for the small-scale
    fading spatial consistency; 6: Parameters for the ground reflection extension; 7: Absolute
    time of arrival offset; 10: Optional modelling components; 8: Path-loss model
  - The entries 1 to 8 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- The overview formula of an LSP shows its mean and STD with the frequency, distance, height and
  elevation terms; with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`; the
  ESD line also has the terms of `ES_D_mu_A` to `ES_D_mu_D` and, if `ES_D_mu_min` is not `-Inf`,
  the lower limit of the mean; the DS line has the term `log10( a · V/S + b )` when `DS_VS > 0`
  (a = `DS_VS_a`, b = `DS_VS_b`; `V/S` stays symbolic)
- Each line is `name = value % comment`, aligned in columns; the comment states the unit
- The path-loss section has the formula blocks of the model (`logdist`, `dual_slope`, `nlos`,
  `tripple_slope`), the line `PL_model`, then the fields of `plpar` in struct order with the prefix
  `PL_`; several values of one field are written on one line
- Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back
  to the same double in [conf_file_read](#conf_file_read); `Inf` and `-Inf` are written as such
- With `write_defaults = false` a field is written only if its value differs from the default;
  `SubpathMethod` only if it is not `legacy`; a section without a written field is left out
- With `write_defaults = true` all 171 numeric fields and `SubpathMethod` are written
- The inputs are checked with the rules of [scenpar_validate](#scenpar_validate) before the file is opened: unknown
  fields (including `GR_epsilon`), `NaN`, values outside their range and a wrong `SubpathMethod`
  are errors. Further errors: a line break in the file name, a `model` that is not one name, an
  empty or `NaN` path-loss value, short names that are not one name or not distinct, line breaks
  in `ref` or `ref_sec`, more than 10 entries in `ref_sec`, a `valid_range` that is not empty and
  not `[4, 2]`.
- The text is written to a temporary file in the same directory, which must be writable, and then
  renamed to `fn`; a symbolic link `fn` is written through
- A failed call does not create a file and leaves an existing file unchanged; an existing file is
  replaced and keeps its permissions
- Every file written passes [conf_file_read](#conf_file_read) with `validate = true` and without ignored keys

### Usage:
```
quadriga_lib.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, ...
    write_defaults );
```

### Inputs:
- **`fn`** — Path of the conf file to write; string
- **`scenpar`** — Scenario parameters; struct with any subset of the 172 fields returned by
  [conf_file_read](#conf_file_read); numeric fields are real scalars of class double, single, int32, uint32,
  int64, uint64 or logical; `SubpathMethod` is a string; missing fields have their default; `[]`
  writes the default set
- **`plpar`** *(optional)* — Path-loss parameters; struct with the field `model` (string) and one
  real vector (row or column) of class double, single, int32, uint32, int64 or uint64 per further
  field, named without the prefix `PL_`, as returned by [conf_file_read](#conf_file_read); a matrix is an error;
  written in struct order; if omitted or `[]`, no path-loss section
- **`short_names`** *(optional)* — Short names of the scenario; cell array of strings or one string;
  each one name of the characters `A-Z`, `a-z`, `0-9`, `_`; distinct
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; string; default: no reference
- **`ref_sec`** *(optional)* — Reference per section, numbered as listed above; cell array of up to
  10 strings; an empty or missing entry writes no reference
- **`valid_range`** *(optional)* — Validity range, written as comment; rows: frequency in GHz,
  distance, BS height, MT height in m; columns: lower and upper end, one value is written if both
  are equal; a row with `NaN` is not written; `[4, 2]`
- **`write_defaults`** — `true`: write all fields; `false`: write only fields that differ from their
  default; default: false

### See also:
- [conf_file_read](#conf_file_read) (read the file back)
- [scenpar_validate](#scenpar_validate) (value checks and defaults)

---
## los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

- Evaluates the probability that a link is in line-of-sight (LOS) for each link, from its 2D
  distance and the BS and UT heights, and optionally the soft LOS state of TR 38.901 eq. (7.6-18)
- Notation: `d = d2d_out(l)`, `hB = h_bs(l)`, `hU = h_ut(l)` in m, `lg = log10`, `ln` the natural
  logarithm. All results are limited to [0, 1]
- TR 38.901 v19.5.0 Table 7.4.2-1:
  - `38.901_RMa`: 1 for `d <= 10`, else `exp(-(d - 10)/1000)`
  - `38.901_UMi`: 1 for `d <= 18`, else `18/d + exp(-d/36)·(1 - 18/d)`
  - `38.901_UMa`: 1 for `d <= 18`, else
    `(18/d + exp(-d/63)·(1 - 18/d))·(1 + C'·5/4·(d/100)³·exp(-d/150))` with `C' = 0` for `hU <= 13`,
    `((hU - 13)/10)^1.5` for `13 < hU <= 23` and 1 for `hU > 23`; for `hU > 13` the formula exceeds
    1 slightly just above 18 m (e.g. up to 1.0060 for `hU = 22.5`), where the limit gives 1
  - `38.901_InH_Mixed` (mixed office): 1 for `d <= 1.2`, `exp(-(d - 1.2)/4.7)` for `1.2 < d < 6.5`,
    `exp(-(d - 6.5)/32.6)·0.32` for `d >= 6.5`
  - `38.901_InH_Open` (open office): 1 for `d <= 5`, `exp(-(d - 5)/70.8)` for `5 < d <= 49`,
    `exp(-(d - 49)/211.7)·0.54` for `d > 49`
  - `38.901_InF_SL`, `38.901_InF_DL`: `exp(-d·kinv)` with `kinv = -ln(1 - r)/d_clutter`
  - `38.901_InF_SH`, `38.901_InF_DH`: with `kinv = -ln(1 - r)/d_clutter`: 1 for `hU >= h_c` (UT
    above the clutter); `exp(-d·kinv·(h_c - hU)/(hB - hU))` for `hB > hU` (TR:
    `k = -d_clutter/ln(1 - r)·(hB - hU)/(h_c - hU)`); `exp(-d·kinv)` (the rule of SL and DL) for
    `hB <= hU < h_c`, where the TR formula has a BS above the UT and would give `k <= 0`. `r = 0`
    gives 1
  - `38.901_InF_HH`: 1
  - `38.901_SMa` (TR v19): 1 for `d <= 10` or `hB <= hU`, else `exp(-d·(k_com + k_res + k_veg))`
    with `k_x = -ln(1 - r_x)/d_clutter·max(h_x - hU, 0)/(hB - hU)`, heights `h_com = 20`,
    `h_res = 8`, `h_veg = 15` m, densities `r_com = 0.02`, `r_res = 0.18`, `r_veg = r_clutter`; the
    TR writes `max(., 0)` only for the residential term, which gives the same values for `hU < 15` m
    (the TR range)
- Defaults of the clutter arguments (NaN or empty), TR 38.901 Table 7.2-4 and Table 7.8-7 for InF,
  Table 7.4.2-1 for SMa:
  - `InF_SL`: `r = 0.2`, `d_clutter = 10` m; `InF_DL`: `r = 0.6`, `d_clutter = 2` m
  - `InF_SH`: `r = 0.2`, `d_clutter = 10` m, `h_c = 2` m; `InF_DH`: `r = 0.6`, `d_clutter = 2` m,
    `h_c = 6` m
  - `SMa`: `d_clutter = 30` m, `r_veg = 0` (no vegetation; 0.1 sparse, 0.2 dense vegetation);
    `h_clutter` is not used
  - The other models ignore the clutter arguments (they are validated for every model)
- TR 38.811 (via QuaDRiGa), `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`:
  the LOS probability in percent over the elevation `atan2(hB, d)` in degrees (absolute BS height),
  linear interpolation over the grid -90, 0, 10, ..., 90 deg, times 0.01:
  - DenseUrban: 0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1
  - Urban: 0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2
  - Suburban and Rural: 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8
- QuaDRiGa, `QuaDRiGa_Industrial`: `min(1, exp(-(d + 20)/80)·exp(min(hB, 8)/5 - 1))`
- `LOS`: 1, `NLOS`: 0
- Soft LOS state (TR 38.901 eq. 7.6-18), computed when `rnd_g` and `fc` are both given and
  `los_soft` is requested: `los_soft(l, k) = 1/2 + atan(sqrt(20/lambda_k)·(rnd_g(l) +
  F(p_los(l))))/pi`, `lambda_k = c/fc(k)` with `c = 299792458` m/s and
  `F(p) = sqrt(2)·erfinv(2p - 1)` the inverse of the standard normal CDF (accurate to
  `1e-10·max(1, abs(F))` for `p` in [1e-12, 1 - 1e-12]). `p = 1` gives 1, `p = 0` gives 0.
  `los_soft > 0.5` exactly when `rnd_g > -F(p)`, which has the probability `p` for `rnd_g` from
  N(0,1)
- The LOS state itself (a draw against `p_los`) and its spatial correlation are the caller's: e.g. a
  spatially consistent uniform value from [sos_rand](#sos_rand), or `rnd_g` from [sos_randn](#sos_randn), at the
  decorrelation distance of the scenario field `LOS_state_lambda` (TR 38.901 Table 7.6.3.1-2). The
  field applies to all links of a scenario and is stated only in its NLOS conf file (0 in the LOS and
  O2I files means "not stated here"). The `LOS_state_lambda` of the InF conf files is
  `d_clutter / 2` of the TR defaults; it does not follow a `d_clutter` argument, so a caller who
  changes `d_clutter` passes its own correlation distance
- O2I UTs: Table 7.4.2-1 uses the outdoor distance `d2d_out = d2D - d2D-in` with `d2D-in` from
  [o2i_loss](#o2i_loss), which therefore runs first; the caller clamps `d2d_out` at 0 (negative distances are
  rejected). Indoor scenarios (InH, InF) pass the 2D distance
- Validation, in this order: the model name; `d2d_out` finite and >= 0; `h_bs` and `h_ut` with 1 or
  `n_link` elements and finite; `d_clutter` NaN or finite and > 0, `r_clutter` NaN or in [0, 1),
  `h_clutter` NaN or finite and >= 0 (for every model); if `los_soft` is computed: `rnd_g` with
  `n_link` finite elements, `fc` finite and > 0. Without `los_soft`, `rnd_g` and `fc` are ignored

### Usage:
```
[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, ...
    d_clutter, r_clutter, h_clutter, rnd_g, fc );
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`,
  `38.901_InH_Open`, `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`,
  `38.901_InF_HH`, `38.901_SMa`, `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`,
  `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`; string
- **`d2d_out`** — 2D outdoor distance in [m] (indoor scenarios: 2D distance), finite and >= 0;
  `[n_link]`
- **`h_bs`** — BS height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`h_ut`** — UT height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`d_clutter`** — Clutter size in [m] of the InF and SMa models; default: NaN = TR default of the
  model
- **`r_clutter`** — Clutter density of the InF models, vegetation density `r_veg` of SMa, in [0, 1);
  default: NaN = TR default of the model
- **`h_clutter`** — Clutter height in [m] of InF-SH and InF-DH; default: NaN = TR default of the
  model
- **`rnd_g`** *(optional)* — Normal(0,1) value per link for the soft LOS state (G of eq. 7.6-18);
  `[n_link]`
- **`fc`** *(optional)* — Carrier frequencies in [Hz] for the soft LOS state; `[n_freq]`

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link, 1]`
- **`los_soft`** — Soft LOS state in [0, 1]; `[n_link, n_freq]`; `[]` if `rnd_g` or `fc` is omitted
  or empty

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos_rand](#sos_rand), [sos_randn](#sos_randn) (spatially consistent LOS-state draw and `rnd_g`)
- [conf_file_read](#conf_file_read) (field `LOS_state_lambda`)

---
## lsf_draw
Spatially correlated normalized random values of the large-scale generators at a set of links

- Evaluates the 11 large-scale fading generators of [lsf_init_phases](#lsf_init_phases) at the links: column 1 DS,
  2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA, 8 XPR, 9 absolute TOA (Normal(0,1)), 10 ground
  permittivity, 11 UMa environment height hE (Uniform(0,1)); each generator with the
  sum-of-sinusoids frequencies of `preset` at its own decorrelation distance (a distance of 0 is
  used as 0.1 m)
- `scenpar` is completed and validated as in [lsf_init_phases](#lsf_init_phases); the messages of
  [scenpar_validate](#scenpar_validate) pass through unchanged
- Single mobility (`dual_mobility = false`): column g is the field of generator g at the Rx
  position, `x_g = s_g(rx)` with `s_g(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and
  the phases of slice 1; the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 2, i.e. the field of the mirror
  generator `m(g)` at the Tx position (ASD and ASA, ESD and ESA exchanged). Self-mirrored generators
  (DS, KF, SF, XPR, TOA, columns 10 and 11) share one map:
  `x_g = ( s_g(rx) + s_g(tx) ) / sqrt( 2 + 2·A_g(D) )`, with the Tx-Rx distance `D` and the ACF
  `A_g` of the preset, linearly interpolated. The pairs are two maps:
  `x_g = ( s_g(rx) + s_m(tx) ) / sqrt(2)`. Columns 4 to 7 thus stay Normal(0,1), and ASD/ASA and
  ESD/ESA have the correlation `( A_g(D) + A_m(D) ) / 2` at the Tx-Rx distance, 1 at `D = 0` (then
  the two columns are equal)
- Swap reciprocity in dual mobility: swapping `tx_pos` and `rx_pos` leaves columns 1, 2, 3, 8, 9,
  10, 11 unchanged and swaps column 4 with 5 and column 6 with 7, exactly (the same evaluations in
  swapped roles), also for unequal decorrelation distances and with bias-removed phases, when both
  calls use the same `phase`
- Uniform columns 10 and 11: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = true`: the positions are projected on the x-y plane (z = 0) and the values do not
  depend on z; `false`: 3D positions. `D` is measured in the same space
- 2D and 3D mode give the same per-link statistics. In 3D mode the height separates the floors: the
  LSPs of UTs on different floors are correlated by the ACF of their height difference, while
  TR 38.901 (Table 7.5-6 note 3) assumes no correlation between floors. `floor_id` is therefore not
  accepted with `horizontal = false`
- Floors: links with different floor IDs see different fields, links with equal IDs share one
  field. Floor `q` adds the offsets `2·pi · k · n / N + psi_b(n)` to the phases of all generators
  on both sides, with `N = n_coeff`, `k = q mod N`, `b = q div N`, `n = 0 ... N-1`, `psi_0 = 0` and
  `psi_b(n) = 2·pi · u_n` for `b >= 1`, where `u_n` are the uniform numbers of a `std::mt19937_64`
  seeded with `b` (the upper 53 bits of each raw output times 2^-53). Floor 0 and an omitted
  `floor_id` give the same values
- Fields of two floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6
  note 3): exactly for two IDs in the same block of `N` IDs, with a random residual of standard
  deviation `1/sqrt(2N)` for IDs in different blocks (0.044 for 256 sinusoids). At other horizontal
  lags the cross-correlation of two floors is not zero (up to about 0.16 for 256 sinusoids), the
  same level as for random offsets. Mirror relations and swap reciprocity hold per floor
- `floor_id` is the caller's floor index of the UT, a label (no index shift): for the UT heights
  `hUT = 3·(n_fl - 1) + 1.5` of TR 38.901 Table 7.2-1 it is `n_fl - 1`; outdoor UTs use 0. The IDs
  are read as double, so they must be below 2^53 (larger values would not be exact)
- Dual mobility with floors: `floor_id` is one value per link and its offsets are added on both
  sides, i.e. both devices of a link are on the link's floor. A Tx device on another floor than its
  Rx (e.g. a D2D link between two floors) is evaluated on the floor of the Rx, so the same Tx device
  sees different fields on links to Rx devices on different floors
- The value of a link does not depend on the other links of the call
- Precision: the sinusoids are evaluated by [sos_val](#sos_val) with the AVX2 kernel where available (sine
  and cosine in single precision, accumulation in double); the values deviate from a
  double-precision evaluation by less than `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [lsf_init_phases](#lsf_init_phases). A wrong preset with the same
  number of sinusoids (e.g. `Exp256` for `Comb256`) cannot be detected
- Validation order: positions (3 columns, rows of `tx_pos`, finite values), `preset`, `scenpar`,
  size of `phase`, finite `phase`, length of `floor_id`, `floor_id` with `horizontal = false`

### Usage:
```
rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, ...
    preset, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [conf_file_read](#conf_file_read) or
  [scenpar_validate](#scenpar_validate) (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; only the 10 decorrelation distances are used
- **`phase`** — Phases in [rad] from [lsf_init_phases](#lsf_init_phases); slice 1 = Rx side, slice 2 = Tx side;
  must be finite; `[n_coeff, 11, 2]`, `n_coeff` of `preset`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; not used in single mobility
  (but validated); `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`floor_id`** *(optional)* — Floor index per link, a label starting at 0 (no index shift);
  non-negative integers below 2^53; omitted or empty = all links on floor 0; in dual mobility the
  floor of both devices; only with `horizontal = true`; `[n_link]`
- **`dual_mobility`** — `true`: Tx and Rx both move, the Tx side is evaluated as a second device;
  default: false
- **`preset`** — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); must match [lsf_init_phases](#lsf_init_phases);
  default: `'Comb256'`
- **`horizontal`** — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); must match [lsf_init_phases](#lsf_init_phases); default: true

### Output:
- **`rnd`** — Random values; columns 1 to 9 Normal(0,1), columns 10 and 11 Uniform(0,1);
  `[n_link, 11]`

### See also:
- [lsf_init_phases](#lsf_init_phases) (draws the phases)
- [path_loss](#path_loss) (column 11 as `rnd_he`)
- [lsp_generate](#lsp_generate) (the large-scale parameters from `rnd`)
- [sos_val](#sos_val) (evaluation of a sum-of-sinusoids generator)

---
## lsf_init_phases
Draw the random phases of the large-scale generators of one BS site and scenario

- The large-scale fading (LSF) of a site is described by 11 spatially correlated sum-of-sinusoids
  generators, one column each: 1 DS, 2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA, 8 XPR, 9 absolute TOA
  (Normal fields), 10 ground permittivity, 11 UMa environment height hE (Uniform fields)
- Their decorrelation distances are the scenario parameters `DS_lambda`, `KF_lambda`, `SF_lambda`,
  `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`, `XPR_lambda`, `absTOA_lambda`,
  `SC_lambda` and again `SF_lambda` for hE; a decorrelation distance of 0 is used as 0.1 m. The
  generator of column 9 exists also when the absolute TOA is disabled
- This function draws the phases; [lsf_draw](#lsf_draw) evaluates them at the links. The caller stores
  `phase` and evaluates the same fields again along tracks or on maps
- `scenpar` is completed with the defaults of missing fields and validated as in
  [scenpar_validate](#scenpar_validate); its messages (e.g. for an unknown field or a negative decorrelation
  distance) pass through unchanged. An invalid value of a field that is not used here throws as well
- Slice 1 (Rx side) holds the phases of the 11 generators. Without bias removal it equals
  `quadriga_lib.sos_init_phases( n_coeff, 11, seed )` of [sos_init_phases](#sos_init_phases), with `n_coeff` the
  number of sinusoids of `preset`. The same seed gives the same phases
- Slice 2 (Tx side) is always the mirror of slice 1:
  `phase(:, :, 2) = phase(:, [1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11], 1)`, i.e. ASD and ASA and ESD and
  ESA are exchanged, all others mirror themselves. For dual mobility the departure spreads at the Tx
  are thus the arrival spreads of the Tx as a second device
- Bias removal (`remove_bias = true`, `rx_pos` given, `n_link > 2`): one realisation of a correlated
  field has a non-zero mean over a drop. For each Normal generator (columns 1 to 9) one common phase
  offset is applied with [sos_remove_bias](#sos_remove_bias), so that the sample mean of its field over the
  positions is exactly 0. Single mobility: over the Rx positions; `tx_pos` is not used. Dual
  mobility: over the Rx and Tx positions together (`2·n_link` positions), because the field of a
  generator is shared by both devices (the Tx side of ASD is the ASA field); `tx_pos` is then
  required. The Uniform generators 10 and 11 are never corrected
- The correction is applied to slice 1 and the Tx side is mirrored afterwards, so the mirror
  relation holds exactly with and without bias removal. Swapping `tx_pos` and `rx_pos` in dual
  mobility gives the same phases up to rounding (the sums over the positions are formed in another
  order)
- Consequences for [lsf_draw](#lsf_draw) at the same positions: in single mobility the link mean of columns
  1 to 9 is 0; in dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have
  the same Tx-Rx distance (the normalisation is then the same for all links), while ASD/ASA and
  ESD/ESA have a zero link mean only approximately (their two devices see different fields, each
  corrected over the union of the positions)
- The positions are used as in [lsf_draw](#lsf_draw): projected on the x-y plane (z = 0) for
  `horizontal = true`, in 3D for `horizontal = false`. `preset` and `horizontal` must be the same in
  [lsf_draw](#lsf_draw)
- 2D mode: all positions in `rx_pos` are treated as one field (floor 0): one phase correction per
  generator cannot zero the means of several floor fields at once. For a drop with UTs on several
  floors pass the positions of one floor (e.g. the links of floor 0) or set `remove_bias = false`;
  the fields of the other floors keep their sample mean. 3D mode (`horizontal = false`): there are no
  floor IDs (the height separates the floors) and the correction covers all given positions as one
  3D field
- 2D and 3D mode give the same per-link statistics. In 3D mode the LSPs of UTs on different floors
  are correlated by the ACF of their height difference, while TR 38.901 (Table 7.5-6 note 3) assumes
  no correlation between floors
- The correction removes variance: the field is conditioned on a zero mean over the positions, which
  costs little only when the positions spread over many decorrelation distances (see
  [sos_remove_bias](#sos_remove_bias))
- With fewer than 3 positions, without `rx_pos` or with `remove_bias = false` no correction is made
- TR 38.901 clause 7.4.4 literally (exponential ACF, no bias removal): `preset = 'Exp256'` and
  `remove_bias = false`; the defaults `Comb256` and the bias removal are QuaDRiGa choices
- Reason for `Comb256`: the TR gives the exponential ACF (eq. 7.4-5) as a fit "with sufficient accuracy",
  not as an exact requirement; its kink at zero distance makes the values evolve roughly along a track.
  `Comb256` (Gaussian near zero, exponential tail, same decorrelation distance) evolves smoothly
- Validation order: `seed`, `preset`, `scenpar`, then the positions (3 columns, rows of `tx_pos` 1
  or `n_link` when both are given, finite values), then the requirement of `tx_pos` for the bias
  removal in dual mobility

### Usage:
```
phase = quadriga_lib.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, ...
    remove_bias, preset, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [conf_file_read](#conf_file_read) or
  [scenpar_validate](#scenpar_validate) (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; only the 10 decorrelation distances are used
- **`seed`** — Seed of the random phases; integer from -1 to 2^63-1; -1 = system random device, a
  new set of phases on each call; default: -1
- **`tx_pos`** *(optional)* — Tx positions in [m]; used only for the bias removal with dual
  mobility, required there; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** *(optional)* — Rx positions of the drop in [m] (one floor in 2D mode, all floors in
  3D mode) for the bias removal; omitted, empty or fewer than 3 rows = no bias removal; `[n_link, 3]`
- **`dual_mobility`** — `true`: Tx and Rx both move, the Tx is a second device on the same maps;
  default: false
- **`remove_bias`** — `true`: remove the sample mean of the Normal fields over the positions;
  default: true
- **`preset`** — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); sets `n_coeff`; default: `'Comb256'`
- **`horizontal`** — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); default: true

### Output:
- **`phase`** — Phases in [rad]; slice 1 = Rx side, slice 2 = Tx side (mirror of slice 1); all
  values in [-pi, pi); `[n_coeff, 11, 2]`

### See also:
- [conf_file_read](#conf_file_read) (the scenario parameters)
- [lsf_draw](#lsf_draw) (evaluates the phases at the links)
- [ssf_init_phases](#ssf_init_phases) (the small-scale sets, seeds derived from this seed)
- [sos_init_phases](#sos_init_phases) (the raw phases)
- [sos_remove_bias](#sos_remove_bias) (the bias removal of one generator)

---
## lsp_generate
Large-scale parameters of a set of links from normalized random values

- Generates the 8 large-scale parameters (LSPs) of TR 38.901 clause 7.5 step 4 per link and
  frequency: delay spread DS, Ricean K-factor KF, shadow fading SF, azimuth spreads of departure and
  arrival ASD, ASA, elevation spreads of departure and arrival ESD, ESA and cross-polarization ratio
  XPR. The random values `rnd` are the normalized, spatially correlated values of [lsf_draw](#lsf_draw)
  (columns 1 to 8 Normal(0,1) for the 8 LSPs, column 9 Normal(0,1) for the absolute TOA, column 10
  Uniform(0,1) for the ground permittivity; column 11 is not used here)
- `scenpar` is completed with the defaults of missing fields and validated as in
  [scenpar_validate](#scenpar_validate); its messages pass through unchanged. An invalid value of a field that is not
  used here throws as well
- Mean and standard deviation in the log domain, per link `l` and frequency `k`, with the field
  prefix `P` of the LSP (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), the 2D distance
  `d2D`, the Tx height `hBS = tx_pos(:, 3)` and the Rx height `hUT = rx_pos(:, 3)`:
  `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h / d)` (elevation in rad),
  `f' = max(fc / 1e9, LSP_fc_min)` (frequency in GHz, limited from below by `LSP_fc_min` for all LSP
  terms); `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`;
  `s = max(0, P_sigma + P_delta·lg(P_omega + f'))`;
  `sigma = max(0, s + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`. The standard deviation is
  limited at 0 after all terms, since a negative value would flip the sign of the realization
  and its cross-correlations. SF has no mean fields (mean 0). A frequency term with the
  coefficient 0 contributes 0; an active one (`P_gamma` or `P_delta` not 0) requires
  `P_omega + f' > 0` at every frequency. The multiplying coefficients (`P_sigma`, `P_gamma`,
  `P_delta`, `P_epsilon`, `P_zeta`, `P_alpha`, `P_kappa`, `P_tau`, `P_beta`, `ES_D_mu_A` to
  `ES_D_mu_D`, `P_omega` of an active frequency term) must be finite; `P_mu` and `ES_D_mu_min` may be
  infinite
- DS with the V/S term of TR 38.901 Table 7.5-6 Part-3 (InF): if `DS_VS > 0`,
  `mu += lg(DS_VS_a·DS_VS + DS_VS_b)`. The InF files set `DS_VS` for the hall of their calibration
  scenario; a caller with another hall sets the field `DS_VS` (V/S in m) of `scenpar`
- ESD with the terms of TR 38.901 Tables 7.5-7 to 7.5-12: `mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT +
  ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)`, then the floor `mu = max(mu, ES_D_mu_min)`
- Units of mean and std: DS in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF and XPR in dB
- `sf_sigma` (e.g. from [path_loss](#path_loss)) replaces the SF standard deviation per link and frequency,
  also in `lsp_sigma`; it must be finite and >= 0 (a negative value would flip the sign of the
  shadow fading)
- Cross-correlation: `x = Q·s` with `s = rnd(l, 1:8)` and a root `Q` of the LSP cross-correlation
  matrix `R` of `scenpar` (`Q·Q^T = R`). `cholesky = true` (TR 38.901 clause 7.5 step 4): Cholesky
  factorization in the TR order [SF, KF, DS, ASD, ASA, ESD, ESA], XPR appended last; the SF value is
  passed through (`x_SF = s_SF`). `cholesky = false`: the symmetric square root of `R` (principal
  root, legacy QuaDRiGa behaviour); it does not depend on the order of the parameters. Both raise an
  error if `R` is not positive definite
- The same `x` is used for all frequencies of a link (TR 38.901 clause 7.6.5)
- Linear values: `lsp = 10^(mu + sigma·x)` for DS, ASD, ASA, ESD, ESA; `10^(0.1·(mu + sigma·x))` for
  KF, SF, XPR. `clip_spreads = true` (TR 38.901 clause 7.5 step 4): ASD and ASA are limited to
  104 deg, ESD and ESA to 52 deg; `lsp_mu` and `lsp_sigma` are not affected
- `rnd = zeros(n_link, 11)` gives the median LSPs; `lsp_mu` and `lsp_sigma` are the mean and std
  without any randomness
- Absolute TOA offset (TR 38.901 clause 7.6.9): active if `absTOA_mu > -30`, then
  `abs_toa = min(10^(absTOA_mu + absTOA_sigma·rnd(l, 9)), 2·absTOA_Lmax / c)` with
  `c = 299792458 m/s`, without the bound if `absTOA_Lmax = 0` (`absTOA_sigma` must be finite); 0 if
  not active. The InF and InH NLOS
  files set `absTOA_Lmax` to the largest dimension of the hall or office of their calibration
  scenario; a caller with another hall sets the field; in the TR 38.901 files the offset is active in all NLOS and O2I files (an O2I link
  has no LOS ray) and off in the LOS files
- Ground permittivity with `U = rnd(l, 10)` and `f = fc / 1e9` (not limited by `LSP_fc_min`): dry
  ground `3 + j·0.003·f^1.34`, medium `30.4·f^-0.47 + j·0.18·f^1.05`, wet
  `31.3·f^-0.48 + j·0.63·f^0.77`; linear interpolation from dry (`U = 0`) to medium (`U = 0.5`) to
  wet (`U = 1`). The imaginary part is >= 0; TR 38.901 eq. (7.6-40) writes the complex permittivity
  with the opposite sign of the imaginary part, so the TR reflection formulas take the complex
  conjugate `gr_eps_re - j·gr_eps_im`
- The three ground types are QuaDRiGa's fits (as in QuaDRiGa's `gen_lsf_parameters`), not the models
  of TR 38.901 Table 7.6.8-1: very dry ground agrees, medium and wet ground differ (e.g. at 3.5 GHz
  medium `16.87 + j·0.67` here, `13.23 + j·1.39` for medium dry ground of the table; wet `j·1.65`
  vs. `j·3.93`). A caller who needs the table values computes them with eq. (7.6-40) to (7.6-42)
- Column 10 of `rnd` must be in [0, 1] when `gr_eps_re` or `gr_eps_im` is requested, i.e. with 3 or
  more outputs; it is not checked otherwise. The error message counts the columns from 0
  ("Column 9 of 'rnd' ...")
- EoD offset in degrees, positive = up (the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to
  7.5-12), with `f'`, `d`, `h` as above: `e = EoD_offset_e + EoD_offset_e_gamma·lg f'`; if
  `EoD_offset_b > 0`: `e += 10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) +
  EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)`; if `EoD_offset_hcl > 0` and
  `hUT < EoD_offset_hcl`: `e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi`. The
  coefficients `EoD_offset_e`, `_e_gamma`, `_a`, `_a_gamma`, `_c`, `_c_gamma`, `_c_hUT` must be finite
- Outputs that are not requested are not computed
- Validation order: positions, `fc`, `rnd`, `sf_sigma`, `scenpar`, positive definiteness, column 10
  of `rnd`, the coefficients and frequency terms of the 8 LSPs, the `EoD_offset_` coefficients and
  `absTOA_sigma` (each only if its output is requested)

### Usage:
```
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ...
    quadriga_lib.lsp_generate( scenpar, fc, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads );
```

### Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [conf_file_read](#conf_file_read) or
  [scenpar_validate](#scenpar_validate) (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; the LSP cross-correlation matrix must be
  positive definite
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Normalized random values from [lsf_draw](#lsf_draw), finite; columns 1 to 8 for the 8 LSPs,
  column 9 for the absolute TOA, column 10 in [0, 1] for the ground permittivity (checked only with
  3 or more outputs), column 11 not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB], finite and >= 0, e.g. from
  [path_loss](#path_loss); replaces the value from `scenpar`; omitted or empty = the scenario value;
  `[n_link, n_freq]`
- **`cholesky`** — `true`: Cholesky root in the TR order; `false`: symmetric square root (legacy);
  default: true
- **`clip_spreads`** — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg; default: true

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 1 DS in [s], 2 KF (linear), 3 SF (linear power
  ratio), 4 ASD, 5 ASA, 6 ESD, 7 ESA in [deg], 8 XPR (linear); `[n_link, n_freq, 8]`
- **`abs_toa`** — Absolute TOA offset in [s], 0 if disabled; `[n_link, 1]`
- **`gr_eps_re`** — Relative permittivity of the ground, real part; `[n_link, n_freq]`
- **`gr_eps_im`** — Relative permittivity of the ground, imaginary part, >= 0; `[n_link, n_freq]`
- **`eod_offset_deg`** — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** — Mean in the log domain (lg for DS, ASD, ASA, ESD, ESA; dB for KF, SF, XPR);
  `[n_link, n_freq, 8]`
- **`lsp_sigma`** — Standard deviation in the log domain, >= 0; `[n_link, n_freq, 8]`

### See also:
- [lsf_draw](#lsf_draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [lsf_init_phases](#lsf_init_phases) (the phases of the random generators)
- [scenpar_validate](#scenpar_validate) (the scenario parameters and the cross-correlation matrix)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)

---
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier
  frequency, from four normalized random values per UT; `f = fc / 1e9` in GHz, `lg = log10`
- Material losses of TR 38.901 v19.5.0 Table 7.4.3-1 in dB: `L_glass = 2 + 0.2 f`,
  `L_IRR = 25.4 + 0.11 f` (IRR glass, Release 19 value), `L_concrete = 5 + 4 f`,
  `L_plywood = 1.03 + 0.17 f`; external wall
  `tw(p1, L1, p2, L2) = 5 - 10·lg(p1·10^(-L1/10) + p2·10^(-L2/10))`
- Models, with `N = rnd(u, 4)` and the indoor 2D distance `d2din`:
  - `low` (low-loss, Table 7.4.3-2): `tw(0.3, L_glass, 0.7, L_concrete) + 0.5·d2din + 4.4·N`
  - `high` (high-loss, Table 7.4.3-2): `tw(0.7, L_IRR, 0.3, L_concrete) + 0.5·d2din + 6.5·N`
  - `low_A` (low-loss A, Table 7.4.3-2): `tw(0.3, L_glass, 0.7, L_plywood) + 0.5·d2din + 4.4·N`
  - `legacy_6GHz` (backward compatible model for single-frequency simulations below 6 GHz,
    Table 7.4.3-3): `20 + 0.5·d2din`, no deviation; it goes with a shadow-fading std of 7 dB (the
    O2I conf files of UMa and UMi have `SF_sigma = 7`)
  - `car` (clause 7.4.3.2): `9 + 5·N`; `car_metal` (metallized car windows): `20 + 5·N`; no indoor
    distance
- Columns of `rnd`: 1 = Uniform(0,1) for the model choice, 2 and 3 = Uniform(0,1) for the indoor
  distance, 4 = Normal(0,1) for the deviation (sigma_P of Table 7.4.3-2, the car loss)
- Model choice per UT: `Ccum_j = (w_1 + ... + w_j)/sum(w)`; the model is the smallest `j` with
  `rnd(u, 1) < Ccum_j`, or, if there is none, the last `j` with `w_j > 0`. A model with weight 0 is
  never chosen
- Indoor distance: `low`, `high`, `low_A`: `d2din = min(rnd(u, 2), rnd(u, 3))·d2d_in_max`, the
  minimum of two uniforms of clause 7.4.3.1; `legacy_6GHz`: `d2din = rnd(u, 2)·d2d_in_max` (one
  uniform, Table 7.4.3-3); `car`, `car_metal`: 0
- The caller draws `rnd` with [sos_rand](#sos_rand) and [sos_randn](#sos_randn) at the correlation distances of
  TR 38.901 clause 7.6.3.3 (building type 50 m for column 1, d2D-in 25 m for columns 2 and 3,
  sigma_P 10 m for column 4) and adds `loss` to the path loss of the indoor UT. `d2d_in` gives the
  outdoor distance `d2D - d2D-in` for [los_probability](#los_probability) (call `o2i_loss` first) and is the input
  of the legacy outdoor distance of [path_loss](#path_loss)
- Applicability (Table 7.4.3-2): RMa uses `low` only, InF `high` only; UMa, UMi: `low` and `high`;
  SMa: `low`, `high` and `low_A`. TR maximum indoor distance: 25 m for UMa, UMi and SMa commercial
  buildings, 10 m for RMa and SMa residential buildings (conf-file field `O2I_d2D_in_max`). The TR
  SMa layout has 90 % residential and 10 % commercial buildings (Table 7.2-5): split the UTs by
  building type and call `o2i_loss` once per type
- Validation, in this order: `fc` non-empty, finite and > 0; `rnd` with 4 columns, columns 1 to 3 in
  [0, 1], column 4 finite; `models` non-empty, each a known name; `weights` with one finite element
  >= 0 per model and a sum of 1 within 1e-6; `d2d_in_max` finite and >= 0. The error messages count
  the columns of `rnd` from 0

### Usage:
```
[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, d2d_in_max );
```

### Inputs:
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`rnd`** — Random values per UT: Uniform(0,1) model choice, Uniform(0,1) d2D-in #1,
  Uniform(0,1) d2D-in #2, Normal(0,1) deviation; `[n_ut, 4]`
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; cell
  array of strings or one string; `[n_model]`
- **`weights`** — Probability of each model, >= 0, sum 1; `[n_model]`
- **`d2d_in_max`** — Maximum indoor 2D distance in [m], finite and >= 0 (conf-file field
  `O2I_d2D_in_max`); default: 25

### Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** — Indoor 2D distance in [m], 0 for the car models; `[n_ut, 1]`
- **`model_index`** — Chosen model per UT, 1-based index into `models`; uint64; `[n_ut, 1]`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)
- [sos_rand](#sos_rand), [sos_randn](#sos_randn) (spatially consistent random values for `rnd`)

---
## path_loss
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and
  carrier frequency. `plpar` is the struct of [conf_file_read](#conf_file_read): its field `model` is the model
  (value of `PL_model`), the other fields are the `PL_` keys without the prefix; the TR 38.901
  v19.5.0 conf files in `data/config` reproduce Table 7.4.1-1 with them
- Notation: `f = fc / 1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance
  in m; `hBS`, `hUT` the z coordinates of Tx and Rx (no ground model); `[X]` the value of field `X`,
  or its default if the field is missing (`X = v` in the lists below);
  `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std
- Models, required fields and defaults:
  - `constant` (A): `pl = A`
  - `logdist` (A, B, C; SF optional): `pl = A·lg(d3) + B + C·lg(f)`; `sf = [SF]` if present
  - `logdist_simple` (A, B): `pl = A·lg(d3) + B`
  - `winner_los` (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2):
    `hb = max(hBS, 1.5)`, `hm = max(hUT, 1.5)` of each link;
    `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`, `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`,
    breakpoint `bp = 10^((H - G)/(A1 - A2))`;
    `d2 <= bp`: `pl = A1·lg d2 + G + F1·hm`, `sf = sig1`; else `pl = A2·lg d2 + H + F2·hm`,
    `sf = sig2` (continuous at `bp` for `F1 = F2`); `A1` and `A2` must differ
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as
    `winner_los`;
    `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm`;
    `pl = L_1` for `f < 1.5`, `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`
  - `winner_pathloss` (A, B, C, X; SF optional): `pl = A·lg d3 + B + C·lg(f/5) + X`; `sf = [SF]` if
    present
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2 optional):
    `hb = max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`; `dBP = E·f·(hb - hE)·(hm - hE)`;
    `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]`; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)`,
    `sf = [sig2]` (each only if present), with the 3D breakpoint distance
    `dBP3 = sqrt(dBP² + (hb - hm)²)`. This is Table 7.4.1-1 for UMa, UMi, InH and InF; for RMa
    and SMa LOS the TR uses the 2D breakpoint distance in PL2, which differs by at most 0.034 dB
    over the TR ranges
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0,
    sig1..sig3 optional): `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)`, `hm` the same;
    `dBPj = Ej·f·(hb - hEj)·(hm - hEj)`, `dBP1 = min(dBP1, dBP2)`,
    `dBPj3 = sqrt(dBPj² + (hb - hm)²)`;
    `LBP1 = PL1(dBP13)`, `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`; `d2 <= dBP1`: `PL1(d3)`, `sf = [sig1]`;
    `dBP1 < d2 <= dBP2`: `LBP1 + A2·lg(d3/dBP13)`, `sf = [sig2]`; `d2 > dBP2`:
    `LBP2 + A3·lg(d3/dBP23)`, `sf = [sig3]`
  - `nlos` (An, Bn; the `dual_slope` fields if `LOSmax != 0`; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n,
    E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, sig optional; Ax, Bx, Cx = 0): `hb = max(hBS, 0.1)`,
    `hm = max(hUT, 0.1)`; `PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb +
    D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm +
    Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; `pl = PLn`; if `[LOSmax] != 0`: `pl = max(pl, PL_LOS)`
    with the `dual_slope` loss of the same parameters (incl. the random hE); if `Ax`, `Bx` or `Cx`
    is present: `pl = max(pl, Ax·lg d3 + Bx + Cx·lg f)` (e.g. the InF-SL term of InF-DL);
    `sf = [sig]` if present
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS / d3, 1)` with the
    absolute Tx height; `pl = +Inf` if `sa <= 0`; else `pl = A·lg d3 + B + C·lg f + D·lg(asin(sa)) +
    PLa` with `PLa = Z(f) / sa` if `usePLa != 0`, else 0; `Z(f)` is the zenith attenuation by
    atmospheric gases (93 values from 1 to 100 GHz, linear interpolation, constant beyond both ends)
  - no model (`plpar` empty or without the field `model`): `pl = 0`
- New fields of the TR 38.901 v19 files: `hE_rand` (non-zero: random environment height of UMa),
  `LOSmax` (0: no maximum with the LOS term in `nlos`), `Ax`, `Bx`, `Cx` (additional term in the
  maximum of `nlos`)
- Random environment height (TR 38.901 Table 7.4.1-1, note 1), in `dual_slope` and the LOS term of
  `nlos` when `[hE_rand] != 0` and `rnd_he` is given, per link with `u = rnd_he(l)`, `d = d2`, raw
  `hUT`: `g = 0` for `d <= 18`, else `1.25·(d/100)³·exp(-d/150)`; `C' = 0` for `hUT < 13`,
  `((hUT - 13)/10)^1.5` for `hUT <= 23`, else 1; `C = C'·g`, `p1 = 1/(1 + C)`; `n` = size of the
  set {12, 15, ..., hUT - 1.5}; `hE = 1` if `C = 0`, `n = 0`, `u < p1` or `p1` rounds to 1 (C
  below 1.1e-16, e.g. beyond about 7.5 km at `hUT = 20`), else
  `hE = 12 + 3·min(floor((u - p1)/(1 - p1)·n), n - 1)`. The UT height is then clamped to
  `hm = max(hUT, hE + 0.1, 0.1)` instead of `1.1·hE`: for `hE = 1` this is the fixed-hE clamp (also
  below `hUT = 1.1`), and the drawn heights 12, 15, ... are not affected (`hUT - hE >= 1.5`).
  Without `rnd_he` the fixed `[hE]` is used. `rnd_he` is column 11 of [lsf_draw](#lsf_draw), a spatially
  consistent uniform value: TR 38.901 draws hE independently per link and gives no decorrelation
  distance; the library makes it spatially consistent in the sense of clause 7.6.3 with the
  decorrelation distance `SF_lambda`, so nearby UTs of a site share hE. Along a track hE switches
  between its discrete values where `u` crosses a threshold, and the loss beyond the breakpoint
  steps there; evaluate `rnd_he` once per track segment to avoid this
- Distances: `d3`, `d2` are the geometric distances (TR 38.901 eq. 7.4-1: for an indoor UT
  `d3D-out + d3D-in`). If `d2d_in` is given (legacy mode), a link with `d2d_in(l) > 0` uses the
  outdoor distance instead: `din = min(d2d_in, d2)`, `d3in = d3·din/d2` (`d3` if `d2 = 0`), limited
  to `d3`; if `d3 - d3in < 10`: `d3in = d3 - 10` (minimum outdoor distance 10 m), then
  `max(d3in, 0)`; `p = d3in/d3` (0 if `d3 = 0`); `d3 = d3 - d3in`, `d2 = d2·(1 - p)`. A link with
  `d2d_in(l) = 0` is an outdoor link and keeps the geometric distances (no 10 m rule). Then
  `d3 = max(d3, 0.1)`, `d2 = max(d2, 0.1)`. The O2I penetration loss is not added; add the output
  of [o2i_loss](#o2i_loss)
- Default shadow-fading std, where a model does not set it: the SF std of the LSP formula of
  [lsp_generate](#lsp_generate) with the geometric distance, `sf = max(0, max(0, SF_sigma +
  SF_delta·lg(SF_omega + f')) + SF_kappa·lg(d) + SF_tau·lg(h) + SF_beta·lg(atan(h/d)))` with
  `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `f' = max(f, LSP_fc_min)`; for scenario files without
  such terms this is `SF_sigma`. Passing `sf_sigma` to [lsp_generate](#lsp_generate) is then neutral. With
  `sf_sigma` requested, `SF_sigma`, `SF_delta`, `SF_kappa`, `SF_tau`, `SF_beta` must be finite, and a
  frequency term with `SF_delta != 0` requires a finite `SF_omega` and `SF_omega + f' > 0`
- Validation: positions, `fc`, the lengths of `d2d_in` and `rnd_he`, their values, the model name,
  the parameters of the model in the order the model reads them (a required field that is missing
  raises `Path-loss model '<model>' requires the parameter '<key>'.`, naming `A1` for "A1 or A" and
  `E1` for "E1 or E"; every field that the model reads must have exactly one value, which must be
  finite; fields not read are ignored), the breakpoint factors `E`, `E1`, `E2` and `G2n` of `nlos`
  (> 0) and `A1 != A2` of `winner_los`, then `scenpar` (as in [scenpar_validate](#scenpar_validate), its messages pass
  through), then the SF coefficients and frequency term if `sf_sigma` is requested
- The models are evaluated as written for any positive frequency, also outside their validity range

### Usage:
```
[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he );
```

### Inputs:
- **`plpar`** — Path-loss parameters; struct as returned by [conf_file_read](#conf_file_read): the field `model`
  (string) and one numeric field per parameter, named without "PL_", one value each; `[]` or no
  field `model` = no model, zero loss
- **`scenpar`** — Scenario parameters (SF std terms, `LSP_fc_min`); struct as returned by
  [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) (the field `SubpathMethod` is ignored), or any subset
  of its fields; missing fields have their default value; `[]` = all defaults
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from
  [o2i_loss](#o2i_loss)), finite and >= 0; given = legacy mode with the outdoor distance; 0 = outdoor link;
  omitted or empty = geometric distance (TR 38.901); `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height,
  column 11 of [lsf_draw](#lsf_draw), in [0, 1]; omitted or empty = fixed `hE`; `[n_link]`

### Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** — Shadow-fading standard deviation in [dB], >= 0 for the default;
  `[n_link, n_freq]`

### See also:
- [conf_file_read](#conf_file_read) (the model and its parameters)
- [lsf_draw](#lsf_draw) (the random values, column 11 for `rnd_he`)
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss and indoor distance)

---
## scenpar_validate
Validate scenario parameters and complete them with default values

- Checks a struct of QuaDRiGa scenario parameters that did not come from a conf file and adds every
  missing field with its default value, so that `scenpar_out` always has all 172 fields
- An empty input (`[]` or a struct without fields) returns the complete default parameter set
- A struct with the fields of QuaDRiGa is completed with the 33 fields for 3GPP TR 38.901 v19.5.0,
  which have no effect at their default of 0
- All fields are real numeric scalars, except `SubpathMethod`, which is a string
- A field that is not one of the 172 names is an error, also with `validate = false`; this includes
  the QuaDRiGa field `GR_epsilon`, which is not supported
- Value checks with `validate = true`; `NaN` is an error for every field:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`,
    `PerClusterES_D`, `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`,
    `LSP_fc_min`, `EoD_offset_b`, `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`,
    `absTOA_Lmax`, `NF_alpha`, `NF_beta`, `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`,
    `PolVar_sigma`, `LOS_state_lambda`, `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`,
    `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other numeric fields accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
  - `SubpathMethod` must be `legacy`, `Laplacian` or `mmMAGIC`
- With `validate = false` no value is checked: `NaN` and any `SubpathMethod` are accepted
- The symmetric matrix `lsp_xcorr` is built from the 28 correlation fields and has a unit diagonal.
  It is tested by a Cholesky factorization. A matrix that is not positive definite is not an error.

### Usage:
```
[ scenpar_out, lsp_xcorr, lsp_xcorr_chk ] = quadriga_lib.scenpar_validate( scenpar_in, validate );
```

### Inputs:
- **`scenpar_in`** *(optional)* — Scenario parameters; struct with any subset of the 172 fields;
  numeric fields are real scalars of class double, single, int32, uint32, int64, uint64 or
  logical; if omitted or empty, the default set is returned
- **`validate`** — `false`: only complete with defaults, no value checks; default: true

### Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing fields have their
  default; struct with 172 fields in the QuaDRiGa field order: the 138 numeric keys of QuaDRiGa with
  `SubpathMethod` (string) as field 107, then the 33 new keys; all numeric fields are double scalars
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite; `false` also if it contains
  `NaN`; logical

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---
## ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

- Turns the LSPs of [lsp_generate](#lsp_generate) and the random values of [ssf_draw](#ssf_draw) into the clusters of each link, by one of
  three methods (a method × property table is in the C++ documentation of `ssf_clusters`):
  - `'38.901'` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalisation); the K-factor in the scaling factors is limited to
    >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` (columns
    N + 1 to 9·N) at the drop or segment start and keep them along the track
  - `'38.901_MF'`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `'quadriga'`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links
- Slots per link: 1 = LOS path, 2 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1`
  NLOS clusters; `n_slot = NumClusters + G`. NLOS cluster `n = 1 ... N` is in slot `1 + G + n` and belongs to the
  columns `b·N + n` of `rnd` (no sorting). Clusters `n > n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and
  the LOS angles. Identify the LOS path by its slot, never by delay or path length: NLOS clusters of the TR methods
  can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot 1 holds the
  geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters
  `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `'38.901'`, `'38.901_MF'` add `abs_toa(l)` to the NLOS
  delays; `'quadriga'` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where
  `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0. Slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`'quadriga'`) with `R² = (abs(R_par)² + abs(R_perp)²)/2` of TR eqs. (7.6-38),
  (7.6-39). The coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios (`K/(K+1) = 1`, also in the ground-reflection
  power), plus the ground reflection if enabled; `rnd`, `rnd_freq` and `lsp_anchor` may be empty
- Swap reciprocity: with mirrored random values (dual mobility in [ssf_draw](#ssf_draw)) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`'38.901'`: `eod_offset_deg` = 0 and no O2I link;
  `'quadriga'`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario (only sizes are checked); `scenpar` is
  validated by [scenpar_validate](#scenpar_validate). At least one link is required

### Usage:
```
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, rx_pos, ...
    lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, o2i, n_cluster );
```

### Inputs:
- **`scenpar`** — Scenario parameters (struct, `SubpathMethod` ignored); used: `NumClusters`, `NumClusters_min`,
  `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m], one row for all links; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate); DS, ASD, ASA, ESD, ESA (slices 1, 4 to 7) > 0, KF (slice 2) >= 0;
  `'38.901'`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [ssf_draw](#ssf_draw) (`set_id = 0`), columns 1 to N in [0, 1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `'38.901'` (default), `'38.901_MF'` or `'quadriga'`
- **`rnd_freq`** *(optional)* — Per-frequency sets of [ssf_draw](#ssf_draw), slice `k` from `set_id = k`; required by
  `'38.901'` and `'38.901_MF'`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz); required by `'38.901_MF'`;
  `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [lsp_generate](#lsp_generate); used by `'38.901'` and
  `'quadriga'` in NLOS scenarios; omitted or empty = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [lsp_generate](#lsp_generate), >= 0; required by `'quadriga'` in
  NLOS scenarios with `absTOA_mu > -30`; omitted or empty = no offset; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im`, real part >= 1,
  imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean ZOA 90 deg), logical or numeric; used by `'38.901'` in NLOS
  scenarios; omitted or empty = none; `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link (a count) in
  `[min(max(1, NumClusters_min - 1), N), N]`; omitted or empty = N; `[n_link]`

### Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for `'38.901_MF'`, else 1
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `[n_link, 1]`

### See also:
- [ssf_draw](#ssf_draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread) (spreads of the output)

---
## ssf_draw
Spatially correlated normalized random values of one small-scale fading generator set at a set of links

- Evaluates a set of [ssf_init_phases](#ssf_init_phases) at the links with the rules of [lsf_draw](#lsf_draw) for single and dual mobility,
  floors, 2D/3D mode and precision; every generator uses the decorrelation distance `SC_lambda` (TR 38.901 Table
  7.6.3.1-2; 0 = spatial consistency off, used as 0.1 m)
- Columns, `N = NumClusters - 1`; block `b = 0 ... 8` holds the columns `b·N + 1` to `(b + 1)·N`, one per NLOS
  cluster:
  - shared set (`set_id = 0`, `9·N` columns): block 0 cluster delay, Uniform(0,1); blocks 1, 2 sign of AOD, AOA;
    3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA; Normal(0,1)
  - per-frequency set (`set_id = k >= 1`, `N` columns): cluster shadowing, Normal(0,1)
- Dual mobility: swapping `tx_pos` and `rx_pos` exchanges the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 and keeps
  the delay and shadowing blocks, exactly (D2D reciprocity of S. Jaeckel et al., "A Spatially Consistent Geometric
  D2D Small-Scale Fading Model for Multiple Frequencies", 2019)
- A column depends neither on the other columns nor on the other links; later versions append blocks and keep the
  values of the existing columns
- `preset`, `horizontal` and the set type must match [ssf_init_phases](#ssf_init_phases); at least one link is required

### Usage:
```
rnd = quadriga_lib.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, ...
    preset, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters as in [ssf_init_phases](#ssf_init_phases); `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases from [ssf_init_phases](#ssf_init_phases); `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m], one row for all links; used in dual mobility; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`set_id`** *(optional)* — `0` = shared cluster set; `k >= 1` = per-frequency set; default: 0
- **`floor_id`** *(optional)* — Floor label per link starting at 0 (no index shift) as in [lsf_draw](#lsf_draw), 2D mode
  only; omitted or empty = floor 0; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: the Tx is a second moving device; default: false
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); default: `'Comb256'`
- **`horizontal`** *(optional)* — `true`: 2D correlation (TR 38.901); `false`: 3D (legacy); default: true

### Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

### See also:
- [ssf_init_phases](#ssf_init_phases) (draws the phases)
- [ssf_clusters](#ssf_clusters) (the clusters from the random values)
- [lsf_draw](#lsf_draw) (the large-scale values; mobility, floor and precision rules)

---
## ssf_init_phases
Draw the random phases of one small-scale fading generator set of a BS site and scenario

- The cluster random values of the small-scale fading (TR 38.901 clause 7.6.3.1) come in two kinds of sets, with
  `N = NumClusters - 1` NLOS clusters: one shared cluster set (`set_id = 0`, `9·N` generators) and one set per
  carrier frequency (`set_id = k`, `N` generators, `k` = position in the caller's frequency list, the same order
  in every call). [ssf_draw](#ssf_draw) evaluates the phases at the links and lists the columns
- The phases are a fixed function of `seed`, the seed of the large-scale set of the same site and scenario
  ([lsf_init_phases](#lsf_init_phases)), and `set_id`, equal on all platforms; the sets do not repeat the large-scale phases and are
  independent of each other. Keep an explicit seed (or the phase arrays) to reproduce the clusters
- Each scenario of a site (LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4 makes the link
  types uncorrelated
- `NumClusters = 1`: both sets are empty

### Usage:
```
phase = quadriga_lib.ssf_init_phases( scenpar, seed, set_id, preset );
```

### Inputs:
- **`scenpar`** — Scenario parameters; struct from [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) or any subset of
  its fields (missing fields take their defaults, `SubpathMethod` is ignored); `NumClusters` sets `N`
- **`seed`** *(optional)* — Seed of the large-scale set; -1 = system random device; default: -1
- **`set_id`** *(optional)* — `0` = shared cluster set; `k >= 1` = set of frequency `k`; default: 0
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); default: `'Comb256'`

### Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 1 = Rx side, slice 2 = Tx side (mirror generators);
  `[n_coeff, n_gen, 2]`, `n_gen = 9·N` (shared set) or `N`

### See also:
- [ssf_draw](#ssf_draw) (evaluates the phases at the links)
- [ssf_clusters](#ssf_clusters) (the clusters from the random values)
- [lsf_init_phases](#lsf_init_phases) (the large-scale set and its seed)

---

# Math functions

---
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, heading) in column-major order (9 elements
  per orientation)
- Single-precision input is cast to double; output is always double

### Usage:
```
rotation = quadriga_lib.calc_rotation_matrix( orientation, invert_y_axis, transpose );
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, heading); `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`invert_y_axis`** — Flips the sign of the tilt angle, i.e. applies `-tilt` instead of
  `tilt`; use when the input convention defines positive tilt as downward; logical; default: false
- **`transpose`** — Returns the transpose of the rotation matrix; logical; default: false

### Outputs:
- **`rotation`** — Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9]`

---
## cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Computes: `len = sqrt(x² + y² + z²)`, `az = atan2(y, x)`, `el = asin(clamp(z / len, -1, 1))`
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- `z/len` is clamped to [-1, 1] before `asin` to guard against `len == 0` and rounding artifacts
  pushing `abs(z/len)` slightly above 1
- Azimuth in [-pi, pi], with the special values of `atan2` (e.g. `atan2(0, -0) = pi`); NaN input gives NaN
  in `az`, `el` and `len`
- The AVX2 kernel computes the angles in single precision; `len` and `z/len` in double precision
- Option to provide a single `[3, n, m]` cube or separate x, y, z `[n, m]` inputs

### Usage:
```
[ az, el, len ] = quadriga_lib.cart2geo( x, y, z, use_kernel );
```

### Inputs:
- **`x`** — X-coordinates or combined input; `[n, m]` or `[3, n, m]`; 
- **`y`** — Y-coordinates; `[n, m]` or empty; ignored for combined input
- **`z`** — Z-coordinates; `[n, m]` or empty; ignored for combined input
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 
  (throws if AVX2 unavailable); other values throw; default: 1

### Outputs:
- **`az`** — Azimuth angles in radians; `[n, m]`
- **`el`** — Elevation angles in radians; `[n, m]`
- **`len`** — Euclidean vector length `sqrt(x² + y² + z²)`; `[n, m]`

---
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for
  abs(x) <= 1e10 (the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; Inf and NaN give NaN
- Either `s` or `c` may be `nullptr` to skip that computation
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision, also for double input
- Request one or two outputs to control which results are returned
- With one output, set the optional `cos_only` flag to `true` to return cosine instead of sine

### Usage:
```
[s, c] = quadriga_lib.fast_sincos(x);
s = quadriga_lib.fast_sincos(x);
c = quadriga_lib.fast_sincos(x, true);
```

### Inputs:
- **`x`** (input) — Input angles; radians; `[n_elem]`
- **`cos_only`** — For a single output: `true` returns `cos(x)`; `false` or `[]` returns `sin(x)`; default: false

### Outputs:
- **`s`** — sin(x); single; `[n_elem]`
- **`c`** — cos(x); single; `[n_elem]`

---
## geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: `x = cos(el) cos(az) len`, `y = cos(el) sin(az) len`, `z = sin(el) len`
- Optional outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; omit from the
  output list to skip their computation
- Defaults to the GENERIC kernel (`use_kernel=1`) to preserve full double precision, matching
  MATLAB's default numeric type
- Set `use_kernel=0` for auto-selection or `use_kernel=2` to force AVX2; the AVX2 kernel
  computes sine and cosine in single precision (angles reduced in double precision first; max. error
  about 1.3e-7 for angles up to 1e10), results widened back

### Usage:
```
split = true;
[ x, y, z, sAZ, cAZ, sEL, cEL ] = quadriga_lib.geo2cart( az, el, len, use_kernel, split );

split = false;
cart = quadriga_lib.geo2cart( az, el, len, use_kernel, split );
```

### Inputs:
- **`az`** — Azimuth angles in radians; `[n, m]`
- **`el`** — Elevation angles in radians; `[n, m]`
- **`len`** — Euclidean vector length sqrt(x^2 + y^2 + z^2); `[n, m]`; default: 1
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); other values throw; default: 1
- **`split`** — If true, return x/y/z and optional sin/cos as separate `[n, m]` matrices. If false, 
  return a single combined `[3, n, m]` cube; sin/cos outputs unavailable in this mode; default: false

### Outputs:
- **`x_or_cart`** — If `split=true`: X-coordinates `[n, m]`. If `split=false`: combined cube with components 
  along the first dim, `[3, n, m]`
- **`y`** — Y-coordinates; `[n, m]` or empty
- **`z`** — Z-coordinates; `[n, m]` or empty
- **`sAZ`** — sin(az); `[n, m]` or empty
- **`cAZ`** — cos(az); `[n, m]` or empty
- **`sEL`** — sin(el); `[n, m]` or empty
- **`cEL`** — cos(el); `[n, m]` or empty

---
## interp
Perform linear interpolation (1D or 2D) on single or multiple data sets

- Interpolates given input data at specified output points.
- Outside the range of the data sampling points, the nearest end value is returned (constant extrapolation);
  output sampling points that are NaN give NaN.
- The data sampling points need not be sorted; repeated values are allowed; they must be finite.
- Supports single and multiple data sets.
- Data types: `single` or `double`

### Usage:
```
dataI = quadriga_lib.interp( x, y, data, xI, yI );      % 2D case

dataI = quadriga_lib.interp( x, [], data, xI );         % 1D case
```

### Inputs:
- **`x`** — Data x-axis sampling points; Length: `[nx]`
- **`y`** — Data y-axis sampling points; Length: `[ny]`
- **`data`** — Input data array/matrix; `[ny, nx, ne]` or `[1, nx, ne]` for 1D case; 3rd dimension
  enables interpolation for multiple datasets simultaneously.
- **`xI`** — Output x-axis sampling points; Length: `[nxI]`
- **`yI`** — Output y-axis sampling points; Length: `[nyI]`; required when `y` is not empty

### Output:
- **`dataI`**  —  Interpolated data `[nyI, nxI, ne]` or `[1, nxI, ne]` for 1D case

---

# Miscellaneous / Tools

---
## version
Returns the quadriga-lib version number

- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  a suffix `_AVX2` is added after the version number
- If Quadriga-Lib was compiled with CUDA support and a CUDA-capable GPU is available,
  a suffix `_CUDA` is added after the version number

### Usage:
```
version = quadriga_lib.version;
```

### Outputs:
- **`version`** — Version string (e.g. "0.11.5_AVX2_CUDA")

---
## write_png
Write a data matrix to a color-coded PNG file

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NaN`
- The auto-detected range ignores `NaN` and `Inf`; `Inf` and `-Inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value; also an infinite `min_val` or `max_val`) maps every value to the first colour
- Raises an error for empty `data` and for `min_val > max_val` (both given)
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

### Usage:
```
quadriga_lib.write_png( fn, data, colormap, min_val, max_val, log_transform );
```

### Inputs:
- **`fn`** — Output `.png` file path; string
- **`data`** — Input data matrix; `[n_rows, n_cols]`
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`min_val`** — Lower clipping bound; auto-detected if `NaN`; default: `NaN`
- **`max_val`** — Upper clipping bound; auto-detected if `NaN`; default: `NaN`
- **`log_transform`** — Apply 10·log10(data) before mapping; non-positive values map to the minimum color; logical; default: false

---

# Site-specific simulation tools

---
## calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided
  into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping
  triangles whose bounding box does not intersect the TX-RX path
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
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
  1 = resolve never, see [ray_state_update](#ray_state_update)
- **`apply_fspl`** — If `true`, folds the geometric spreading loss into `gain`, making it an absolute
  propagation gain instead of a value relative to FSPL. The reference distance is the direct TX-RX
  distance, matching the `path_length` used by [qrt_file_read](#qrt_file_read) for a
  transmission-only path. EM mode: `32.45 + 20*log10(f_GHz) + 20*log10(d_m)` dB; scalar mode:
  `20*log10(d_m) + alpha(f)*d_m` dB, with `alpha` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm; default: `false`
- **`apply_phase`** — If `true`, multiplies `xprmat` by the propagation phase `exp(-j*2*pi*d/lambda)`
  using the same direct TX-RX distance. The wavelength follows from `center_freq` and the medium wave
  speed (speed of light in EM mode, speed of sound in scalar mode). The term has unit magnitude, so
  `xprmat` stays normalized. This is the phase [get_channels_spherical](#get_channels_spherical) would otherwise add from
  `path_length`; default: `false`

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `[n_pos, 1]`. Relative to FSPL by
  default (bounded by 0 and 1), absolute propagation gain when `apply_fspl = true`
- **`xprmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase; `gain` holds the power. EM mode:
  interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx), the row order
  of `M` in [get_channels_spherical](#get_channels_spherical); `[8, n_pos]`; scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]` with `n_seg` from `lod`
  (e.g. `[3, 2, n_pos]` for lod 2, `[3, 9, n_pos]` for lod 7, `[3, 0, n_pos]` for lod 0)

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines `mtl_prop` format)
- [qrt_file_read](#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

---
## cube
Construct a triangulated cube mesh

### Description:
- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1)
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 · n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with obj_file_write
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- A negative scale mirrors the cube; when the product of the scales is negative, the winding is swapped so that
  the normals stay outward (as Blender's "Apply Scale")

### Usage:
```
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
## generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by quadriga_lib.[calc_diffraction_gain](#calc_diffraction_gain): elliptic-arc paths sampling
  the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the
  semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so
  sum(prod(weights,3),2) = 1

### Usage:
```
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
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Usage:
```
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
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

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
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
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
## obj_file_write
Write a Wavefront .obj file

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
```
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
## plane
Construct a triangulated plane mesh

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
```
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
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_index` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Usage:
```
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
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes
  at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are
  placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

### Usage:
```
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

### Usage:
```
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
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

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
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
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
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the
  origin, enabling energy spread simulation
- Returns, for each point, the list of ray indices whose beam intersects that point
- All internal computations use single precision

### Usage:
```
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
- [subdivide_rays](#subdivide_rays) (subdivide beams into sub-beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (ray–triangle intersection)
- [ray_mesh_interact](#ray_mesh_interact) (beam–mesh interaction)

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

### Usage:
```
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
## subdivide_triangles
Subdivide triangles into smaller triangles

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in · n_div · n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles

### Usage:
```
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
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding
  non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Usage:
```
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
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark
  padding entries
- If `mtl_ind_in` is provided, material indices are reordered and padded in the same way

### Usage:
```
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
% 1000 positions; one row per position, columns are [x, y, z] in meters
pos = rand(1000, 3) * 100;

% One independent generator per decorrelation distance: 10 m and 50 m
dist_decorr = [10, 50];

% Normal distributed values with zero mean and unit variance, size [1000, 2]
val = quadriga_lib.sos_randn(dist_decorr, pos, [], [], 'Comb256', 42);

% Uniform values between 0 and 1, and integers from 1 to 6
uni  = quadriga_lib.sos_rand(dist_decorr, pos, [], [], 'Comb256', 42);
dice = quadriga_lib.sos_randi(dist_decorr, pos, 6, [], [], 'Comb256', 42);
```

The last argument is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field. The functions have their own random
stream; the state of `rand` and `randn` in MATLAB or Octave has no effect on them.

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
% Frequencies of a predefined generator for a decorrelation distance of 10 m
[freq, acf, dist] = quadriga_lib.sos_preset('Comb256', 10);

% Phases of 20 generators; each column is one independent random field
phase = quadriga_lib.sos_init_phases(size(freq, 1), 20, 42);

% Values of all 20 generators at the positions, size [1000, 20]
val_all = quadriga_lib.sos_val(freq, phase, pos);

% The same fields at other positions, at any later time
pos_new = pos + 2.5;
val_new = quadriga_lib.sos_val(freq, phase, pos_new);

% Uniform distributed output
val_uni = quadriga_lib.sos_val(freq, phase, pos, [], [], [], [], [], true);

% Map on a regular grid, size [n_y, n_x, 1, 20]
x = 0:0.5:250;  y = 0:0.5:100;
grid = quadriga_lib.sos_map(freq, phase, x, y);
imagesc(x, y, grid(:, :, 1, 1));  axis xy;
```

- **Positions:** one row per position, `[n_pos, 3]`. Results have one row per position and one column per
  generator, `[n_pos, n_gen]`.
- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `'Exp512'`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [sos_calc_mse](#sos_calc_mse) reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `'Comb256'` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves.

```
% Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
pos_tx = pos;
pos_rx = rand(1000, 3) * 100;
val_link = quadriga_lib.sos_val(freq, phase, pos_tx, pos_rx, [], acf, dist);
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
  of 10 m) the results stay within the tolerance stated in [sos_val](#sos_val).
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [sos_generate](#sos_generate), which runs on one thread. No result depends on the number of threads.
  [sos_val](#sos_val) computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**
[sos_generate](#sos_generate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos_calc_mse](#sos_calc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```
% Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
dist_own = 0:0.125:24.875;
acf_own  = exp(-(dist_own / 5).^2);

% 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
freq_own = quadriga_lib.sos_generate(acf_own, dist_own, 256, 3, true, 300, true, 0.01, 0, 1);

% Approximation error in dB, larger is better; about 35 dB for this example
mse = quadriga_lib.sos_calc_mse(freq_own, acf_own, dist_own);
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

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 1`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos_acf_approx](#sos_acf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on
  the regular axis

### Usage:
```
[ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`

### Outputs:
- **`acf_2d`** — ACF on the grid; `[n_grid, n_grid]`
- **`dist_2d`** — Grid axis; column vector; `[n_grid]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
## sos_acf_approx
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [sos_val](#sos_val) over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [sos_acf_2d](#sos_acf_2d): from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- `n_dim = 1`: along the axis; `[n_grid, 1]`
- `n_dim = 2`: on the x-y plane; `[n_grid, n_grid]`; element `(iy, ix)`
- `n_dim = 3`: on the x-y, x-z and y-z plane (pages 1, 2, 3); `[n_grid, n_grid, 3]`; the row index
  belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center, exactly
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)

### Usage:
```
[ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( freq, dist );
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

### Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; column vector; `[n_grid]`

### See also:
- [sos_acf_2d](#sos_acf_2d) (target ACF on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
## sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: a matrix `[2·n_pos, 3]` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [sos_init_phases](#sos_init_phases)
- Bin `n` covers the distances from the midpoint between `bins(n-1)` and `bins(n)` (excluded) to
  the midpoint between `bins(n)` and `bins(n+1)` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins(end)`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is
  [sos_val](#sos_val) with both positions. Two links form a pair if the distance between their first
  devices falls into the bin and the distance between their second devices does not exceed the
  upper limit of the bin; requires `acf` and `dist`
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

### Usage:
```
[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( freq, phase, bins, n_pos, ...
    acf, dist, seed );
```

### Inputs:
- **`freq`** — Frequencies; `[n_coeff, 3]`
- **`phase`** — Phases of the generator; `[n_coeff]`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`acf`** *(optional)* — Target ACF; required for `acf_est_dual`, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`seed`** — Seed for the positions; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Outputs:
- **`acf_est`** — Estimated ACF; column vector; `[n_bin]`
- **`dist_est`** — Mean distance of the pairs in each bin; column vector; `[n_bin]`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; column vector; `[n_bin]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF of the sinusoids in closed form)
- [sos_val](#sos_val) (evaluation of the generator)

---
## sos_acfi
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [sos_val](#sos_val) and [sos_remove_bias](#sos_remove_bias) for two devices on the same map

### Usage:
```
val = quadriga_lib.sos_acfi( acf, dist, d );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances; must be >= 0; `[n_d]`

### Output:
- **`val`** — ACF at the distances `d`; column vector; `[n_d]`

### See also:
- [sos_preset](#sos_preset) (ACF tables of the predefined models)

---
## sos_calc_mse
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core`: error on the grid of [sos_acf_approx](#sos_acf_approx) against [sos_acf_2d](#sos_acf_2d), for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse` equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core` and `mse_all` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as `Inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)

### Usage:
```
[ mse, mse_core, mse_all, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( freq, acf, dist, ...
    n_dir );
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10

### Outputs:
- **`mse`** — Error along the test directions in [dB]; scalar
- **`mse_core`** — Error within the core radius in [dB]; scalar
- **`mse_all`** — Error on the complete grid in [dB]; scalar
- **`acf_tail`** — Approximated ACF beyond the fitted range; column vector; `[4]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

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
  `uniform_smp = true` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [sos_calc_mse](#sos_calc_mse)), `false` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the
  residual between the target and the contribution of all other sinusoids along its own direction:
  coarse search over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The new
  frequency is kept only if the mean squared error over the test directions decreases. The
  iteration stops after a pass that changes nothing or lowers the error by less than a factor of
  1e-6, or after 1000 passes
- `random_init = true` starts from uniformly random frequencies; `false` builds the set up
  sinusoid by sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is
  not controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [sos_calc_mse](#sos_calc_mse)). To prevent this, set `freq_min`
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; 0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- The search grid has `( freq_max - freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are
  stored for all sample points. It must not have more than 100000 points (the default has 151),
  which limits `freq_max - freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [sos_preset](#sos_preset) were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

### Usage:
```
[ freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, n_coeff, n_dim, uniform_smp, ...
    n_dir, random_init, freq_min, freq_max, seed );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `[n_acf]`
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_dim`** — Number of dimensions; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (true) or random (false) directions of the sinusoids; default: true
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (true) or build up sinusoid by sinusoid
  (false); default: true
- **`freq_min`** — Lower limit of the frequency magnitude; must be >= 0; 0 = none;
  default: 0
- **`freq_max`** — Upper limit of the frequency magnitude; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; limited by the size of the search
  grid, see above; default: 0
- **`seed`** — Seed; integer from -1 to 2^63-1; -1 seeds from the system random device; default: -1

### Outputs:
- **`freq`** — Frequencies; `[n_coeff, n_dim]`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale; column
  vector; `[n_pass]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_calc_mse](#sos_calc_mse) (error of a frequency table)
- [sos_preset](#sos_preset) (precomputed tables)

---
## sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [sos_val](#sos_val)
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: 64-bit Mersenne Twister (`std::mt19937_64`) seeded with `seed`; each value is
  `2·pi · ( u - 0.5 )`, where `u` is the upper 53 bits of one generator output divided by 2^53.
  A seed >= 0 gives the same phases on all platforms; MATLAB's own random stream is not used
- The matrix is filled column by column from one stream: for the same seed, column `g` does not
  depend on `n_gen`
- A seed stands for a set of phases. Generators that use phases of the same seed with different
  frequencies (another decorrelation distance or preset) produce related fields, which are equal
  at the coordinate origin and correlated around it; draw independent quantities as columns of
  one call or with different seeds

### Usage:
```
phase = quadriga_lib.sos_init_phases( n_coeff, n_gen, seed );
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0; default: 1
- **`seed`** — Seed of the random stream; integer from -1 to 2^63-1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

### Output:
- **`phase`** — Phases; `[n_coeff, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluates the generators)
- [sos_preset](#sos_preset) (frequencies that belong to the phases)

---
## sos_map
Spatially correlated random values on a regular grid

- Evaluates the generators of [sos_val](#sos_val) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz, g)` belongs to the position
  `{x(ix), y(iy), z(iz)}` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; an omitted or
  empty `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [sos_val](#sos_val) at the grid points, bit for bit; kernels, accuracy and the
  Uniform mapping are described there

### Usage:
```
map = quadriga_lib.sos_map( freq, phase, x, y, z, uniform, use_kernel );
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`x`** — Grid points on the x-axis; `[nx]`
- **`y`** *(optional)* — Grid points on the y-axis; requires `n_dim >= 2`; omitted or empty = one
  sample at 0, `ny = 1`; `[ny]`
- **`z`** *(optional)* — Grid points on the z-axis; requires `n_dim = 3`; omitted or empty = one
  sample at 0, `nz = 1`; `[nz]`
- **`uniform`** — Output distribution: false = Normal(0,1), true = Uniform between 0 and 1;
  default: false
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  default: 0

### Output:
- **`map`** — Values; `[ny, nx, nz, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation at arbitrary positions, two devices)

---
## sos_preset
Frequencies and ACF of a predefined sum-of-sinusoids generator

- Returns a precomputed frequency table for [sos_val](#sos_val) together with the autocorrelation
  function (ACF) it was fitted to
- Presets: 'Exp128', 'Exp256', 'Exp512', 'Exp1024', 'Gauss128', 'Gauss256', 'Gauss512',
  'Gauss1024', 'Comb128', 'Comb256', 'Comb512', 'Comb1024'. The number is the number of
  sinusoids, `n_coeff`. The name is matched exactly, including upper and lower case
- ACF types, with the distance `D` and the decorrelation distance `d`: `Exp`: `R = exp( -D/d )`;
  `Gauss`: `R = exp( -(D/d)^2 )`; `Comb`: `Gauss` for `D <= d`, `Exp` beyond.
  All three have `R(d) = exp(-1)`
- The ACF is returned as a table with 200 samples from 0 to `4.975·d` in steps of `0.025·d`
- The tables are stored for `d = 10 m`. For another decorrelation distance the distances are
  multiplied by `d/10` and the frequencies by `10/d`
- All tables are 3-dimensional and can be used with positions of 1, 2 or 3 columns
- No sinusoid has a wavelength above 20 decorrelation distances. The ACF of the tables stays near
  zero beyond the fitted range, and the mean of the generator over a large area is not offset by
  sinusoids of nearly zero frequency

### Usage:
```
[ freq, acf, dist ] = quadriga_lib.sos_preset( name, dist_decorr );
```

### Inputs:
- **`name`** — Name of the preset, e.g. 'Comb256'; string
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10

### Outputs:
- **`freq`** — Frequencies; `[n_coeff, 3]`
- **`acf`** — Target ACF; `[200]`
- **`dist`** — Sample points of the ACF; `[200]`

### See also:
- [sos_init_phases](#sos_init_phases) (random phases for the generator)
- [sos_val](#sos_val) (evaluation)
- [sos_generate](#sos_generate) (frequencies for other ACFs)

---
## sos_rand
Spatially correlated Uniform distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second
  device on its own map uses column `g` of a second stream,
  `sos_init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62 flipped, i.e.
  `seed + 2^62` for seeds below 2^62 (exact only as `int64`); for `seed = -1` both streams are
  random. The phases of a generator therefore depend neither on the number of generators nor on
  whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos_val](#sos_val). MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

### Usage:
```
val = quadriga_lib.sos_rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [sos_preset](#sos_preset); string; default: 'Comb256'
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

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

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second
  device on its own map uses column `g` of a second stream,
  `sos_init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62 flipped, i.e.
  `seed + 2^62` for seeds below 2^62 (exact only as `int64`); for `seed = -1` both streams are
  random. The phases of a generator therefore depend neither on the number of generators nor on
  whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- The Uniform value `u` of [sos_rand](#sos_rand) is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos_val](#sos_val), and an integer can then differ by 1
  where the Uniform value lies at a step. MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

### Usage:
```
val = quadriga_lib.sos_randi( dist_decorr, pos, imax, pos_b, dist_decorr_b, preset, seed );
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`imax`** — Largest value; must be > 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [sos_preset](#sos_preset); string; default: 'Comb256'
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability;
  returned as double; one row per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_randn](#sos_randn), [sos_rand](#sos_rand) (Normal and Uniform values)

---
## sos_randn
Spatially correlated Normal distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second
  device on its own map uses column `g` of a second stream,
  `sos_init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62 flipped, i.e.
  `seed + 2^62` for seeds below 2^62 (exact only as `int64`); for `seed = -1` both streams are
  random. The phases of a generator therefore depend neither on the number of generators nor on
  whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos_val](#sos_val). MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

### Usage:
```
val = quadriga_lib.sos_randn( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [sos_preset](#sos_preset); string; default: 'Comb256'
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row
  per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_rand](#sos_rand), [sos_randi](#sos_randi) (Uniform and integer values)

---
## sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF)
  over all pairs of positions, which is much larger than `1 / n_pos` when the positions lie within
  a few decorrelation distances of each other. This is a property of any correlated field, not an
  error of the generator
- This function adds one offset to all phases of a generator such that the mean of the Normal
  distributed values of [sos_val](#sos_val) over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`,
  which vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the
  smaller magnitude is applied, so the offset lies between -pi/2 and pi/2
- One device (`pos_b` omitted): one offset per generator
- Two devices on the same map (`pos_b` given, `phase_b_in` omitted): one offset per generator that
  zeroes the mean of the combined, normalised value; `acf` and `dist` are required
- Two devices on different maps (`pos_b` and `phase_b_in` given): each device is treated on its
  own, `phase_in` with `pos` and `phase_b_in` with `pos_b` and the frequencies
  `freq · freq_scale_b`; two offsets per generator. The mean of the combined value is then zero
  as well
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions.
  The variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the
  sample mean and `c` the mean ACF between that position and all positions; on average over the
  positions the loss is at least `v`. It is small only if the positions spread over many
  decorrelation distances. For positions within about one decorrelation distance most of the
  variance is removed
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are
  used with other frequencies (another decorrelation distance) or positions
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation
  distance `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for
  `L = 5·d`, 5 % for `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose
  `abs(Z)` is below `1e-14 · n_coeff` already has a zero mean for every offset and is returned
  unchanged, also if its phases lie outside that interval
- Two devices on the same map need an ACF above -1 at all sample points, as in [sos_val](#sos_val)

### Usage:
```
[ phase_out, phase_b_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( freq, phase_in, ...
    pos, pos_b, phase_b_in, acf, dist, freq_scale_b );
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase_in`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos_b, 1]`, `[n_pos_b, 2]` or `[n_pos_b, 3]` as for `pos`; `n_pos_b = n_pos` for the same map,
  any number for different maps
- **`phase_b_in`** *(optional)* — Phases of the second device; requires `pos_b`; omitted
  or empty = same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be
  > -1; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `phase_b_in`;
  default: 1

### Outputs:
- **`phase_out`** — Corrected phases; `[n_coeff, n_gen]`
- **`phase_b_out`** — Corrected phases of the second device; `[n_coeff, n_gen]`; empty if
  `phase_b_in` was not given
- **`offset`** — Offsets added to the phases; `[n_gen, 1]`, or `[n_gen, 2]` with
  `phase_b_in` (column 1: `phase_in`, column 2: `phase_b_in`)
- **`mean_before`** — Sample mean before the correction; `[n_gen, 1]`: mean of the values of
  [sos_val](#sos_val) for the given arguments; with `phase_b_in` `[n_gen, 2]`: mean of the field of each
  device on its own positions, the mean of the combined value being their sum divided by `sqrt(2)`

### See also:
- [sos_val](#sos_val) (the values whose mean is removed)

---
## sos_val
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of
  `freq`, in cycles per meter) and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function
  (ACF) `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the
  frequencies were fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The
  sines and cosines of the positions are computed once and reused for all generators
- Positions are given as one row per position with the columns `{x, y, z}`; they may have fewer
  columns than `freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and
  `pos_b` and `A` the target ACF interpolated as in [sos_acfi](#sos_acfi); `acf` and `dist` are required.
  The value does not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `freq · freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same
  matrix as `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- Two devices on the same map need an ACF above -1 at all sample points, because the value is
  divided by `sqrt( 2 + 2·A(D) )`
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and
  1 as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the
  tail is rounded to them
- Kernels: GENERIC computes in double precision. AVX2 evaluates sine and cosine in single
  precision; its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by
  less than 1e-6
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The
  bound between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both
  kernels lose accuracy at the same rate
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor
  on the number of threads

### Usage:
```
val = quadriga_lib.sos_val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, ...
    uniform, use_kernel );
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`phase_b`** *(optional)* — Phases of the second device; requires `pos_b`; omitted or
  empty = same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be
  > -1; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; default: 1
- **`uniform`** — Output distribution: false = Normal(0,1), true = Uniform between 0 and 1;
  default: false
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (error if
  unavailable); default: 0

### Output:
- **`val`** — Values; one row per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_preset](#sos_preset) (frequencies and ACF of the predefined models)
- [sos_init_phases](#sos_init_phases) (random phases)
- [sos_map](#sos_map) (evaluation on a grid)
- [sos_randn](#sos_randn) (generator set-up and evaluation in one call)

