---
title: "MATLAB / Octave API Documentation for Quadriga-Lib v0.11.8"
author: "Stephan Jaeckel"
date: "04.06.2026"
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
| [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) | Array antenna functions | 93 |
| [arrayant_calc_directivity](#arrayant_calc_directivity) | Array antenna functions | 145 |
| [arrayant_combine_pattern](#arrayant_combine_pattern) | Array antenna functions | 183 |
| [arrayant_concat](#arrayant_concat) | Array antenna functions | 233 |
| [arrayant_copy_element](#arrayant_copy_element) | Array antenna functions | 263 |
| [arrayant_export_obj_file](#arrayant_export_obj_file) | Array antenna functions | 291 |
| [arrayant_generate](#arrayant_generate) | Array antenna functions | 320 |
| [arrayant_interpolate](#arrayant_interpolate) | Array antenna functions | 441 |
| [arrayant_qdant_read](#arrayant_qdant_read) | Array antenna functions | 512 |
| [arrayant_qdant_write](#arrayant_qdant_write) | Array antenna functions | 552 |
| [arrayant_rotate_pattern](#arrayant_rotate_pattern) | Array antenna functions | 604 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 656 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 718 |
| [channel_export_obj_file](#channel_export_obj_file) | Channel functions | 780 |
| [hdf5_create_file](#hdf5_create_file) | Channel functions | 816 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 844 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 902 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 931 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 957 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 981 |
| [hdf5_version](#hdf5_version) | Channel functions | 1007 |
| [hdf5_write_channel](#hdf5_write_channel) | Channel functions | 1023 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1059 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 1091 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 1124 |
| [quantize_delays](#quantize_delays) | Channel functions | 1181 |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generation functions | 1225 |
| [get_channels_irs](#get_channels_irs) | Channel generation functions | 1298 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 1364 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 1413 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 1464 |
| [acdf](#acdf) | Channel statistics | 1522 |
| [calc_angular_spread](#calc_angular_spread) | Channel statistics | 1553 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 1593 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 1635 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 1663 |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | 1693 |
| [cart2geo](#cart2geo) | Math functions | 1715 |
| [fast_sincos](#fast_sincos) | Math functions | 1742 |
| [geo2cart](#geo2cart) | Math functions | 1770 |
| [interp](#interp) | Math functions | 1810 |
| [version](#version) | Miscellaneous / Tools | 1841 |
| [write_png](#write_png) | Miscellaneous / Tools | 1858 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 1882 |
| [generate_diffraction_paths](#generate_diffraction_paths) | Site-specific simulation tools | 1924 |
| [icosphere](#icosphere) | Site-specific simulation tools | 1962 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 1988 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 2036 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 2075 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 2102 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 2129 |
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | 2158 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 2234 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 2274 |
| [subdivide_triangles](#subdivide_triangles) | Site-specific simulation tools | 2314 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 2338 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 2365 |

---

# Array antenna functions

---
## arrayant_calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in
  each direction than the antenna sampling grid)
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

### Usage:
```
% Input as struct (struct mode)
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in );
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in, center_freq_new, azimuth_grid_new, elevation_grid_new );

% Separate outputs, struct input (single-freq only)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, ...
    coupling_im, center_freq, name] = quadriga_lib.arrayant_combine_pattern( arrayant_in );

% Separate inputs (split-mode, single-freq only)
arrayant_out = quadriga_lib.arrayant_combine_pattern( [], center_freq_new, azimuth_grid_new, elevation_grid_new, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name );
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`center_freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is 
  recomputed (interpolated) at each requested frequency. If omitted or empty, the function uses each input 
   entry's `center_freq` unchanged.
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; sorted; defaults to input grid
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; sorted; defaults to input grid
- **`e_theta_re`** ... **`elevation_grid`** — Required inputs for slit-mode
- **`element_pos`** ... **`name`** — Optional inputs for slit-mode

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
  via the `freq` argument

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
  uint64; default: 1; must satisfy `1 <= freq <= n_freq`

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

% Separate outputs (must request exactly 11)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name] = quadriga_lib.arrayant_generate( ... );
```

### Mapping table with defaults:
| Position          | 2   | 3    | 4      | 5      | 6   | 7   | 8   | 9   | 10       | 11      | 12  | 13  | 14  | 15  | 16      |
| ----------------- | --: | ---: | -----: | -----: | --: | --: | --: | --: | -------: | ------: | --: | --: | --: | --: | ------: |
|  **Varialbe**     | res | freq | az_3dB | el_3dB | bfr |   M |   N | pol | tilt_dir | spacing | Mg  | Ng  | dgv | dgh | pattern |
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
- **`az_3dB`** — Azimuth 3dB beamwidth in degrees; default: 90 for `custom` else triggers 3GPP / multibeam internal defaults
- **`el_3dB`** — Elevation 3dB beamwidth in degrees; default: 90 for `custom`; else triggers 3GPP / multibeam internal defaults
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
  outputs; overrides default 3GPP/ULA element pattern; other struct fields if present are ignored

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
- **`element`** — Element indices to interpolate; duplicates allowed; defaults to `[1:n_elements]` 
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
  or a single struct when it has exactly one entry
- With `id`, a single entry is read; useful for picking one frequency from a multi-frequency file
- Separate-fields output (11 or 12 outputs) is only available when the result is a single entry 
  (i.e. `id` was provided, or the file contains exactly one entry)

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
- **`layout`** — Matrix organizing multiple antenna IDs within the file; must only reference IDs present in the file; uint32

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

  Multi-frequency input accepts `usage` in {0, 1, 2} and never adjusts the grid (internally maps
  0 → 3 and 1 → 4 for uniform-grid consistency across frequencies).
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
- Multi-frequency mode: interpolates complex channel coefficients from the input frequency grid
  `center_freq` to the output grid using SLERP, then applies delay-induced phase rotation per output
  carrier; only the first frequency slab of `delay` is used
- Multi-snapshot mode: applies the single-frequency backend in parallel across snapshots via OpenMP;
  `i_snap` (1-based MATLAB indices) optionally selects a subset
- Carrier positions are specified one of two ways: as a normalized `pilot_grid` paired with `bandwidth`
  (where `0.0` corresponds to `center_freq(1)` and `1.0` to `center_freq(1) + bandwidth`), or as absolute
  frequencies via `carrier_freq`; supplying both pairs is an error
- Multi-frequency inputs always require `center_freq`; if `carrier_freq` is omitted it is derived as
  `center_freq(1) + pilot_grid · bandwidth`
- Multi-snapshot inputs require `center_freq` to be omitted or scalar (no combined multi-freq + multi-snap)
- Single-frequency inputs ignore `center_freq` when `pilot_grid` + `bandwidth` are given
- `delay` supports broadcasting: shape `[1, 1, n_path, ...]` applies the same delays to all RX/TX pairs
- Internal arithmetic is single-precision; uses AVX2 where supported; double inputs are narrowed to
  float internally, results widened back

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
  the 4th dimension of `coeff_re`; for multi-snap must be omitted or scalar; `[n_freq, 1]`
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
- **`i_snap`** — Snapshot indices to include; range [1 ... n_snap]; empty exports all

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
- Returns `[0, 0, 0, 0]` if the file does not exist; errors if the file exists but is not a valid HDF5 file
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
- Errors if the file does not exist or is not a valid HDF5 file

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
- [hdf5_read_layout](#hdf5_read_layout) (for reading the exisiting layout)

---
## hdf5_version
Return the HDF5 library version string

- Reports the HDF5 C library version that quadriga-lib was compiled against, taken from the
  HDF5 header macros at compile time
- Useful for diagnosing binary/library mismatches when loading or writing channel files

### Usage:
```
version = quadriga_lib.hdf5_version;
```

### Outputs:
- **`version`** — HDF5 version string in the format `"x.y.z"` (e.g. `"1.12.2"`); string

---
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file with a sensible default layout if it does not yet exist; appends to existing files otherwise
- A warning is issued if any selected slot already contains data (it is overwritten)
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
- The file must already exist (use [hdf5_create_file](#hdf5_create_file) first)
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element
  types: single, double, int32, uint32, int64, uint64
- Row vectors are stored as column vectors

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
    path_length, M, aod, eod, aoa, eoa, path_coord, no_int, coord ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
```

### Inputs:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** — Snapshot indices; 1-based; uint64; `[n_out]` or empty; default: `[]` (read all)
- **`i_orig`** — Origin index; 1-based; uint64; scalar; default: 1
- **`downlink`** — If `true`, origin=TX, destination=RX; if `false`, roles are swapped; logical scalar; default: `true`
- **`normalize_M`** — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |

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

### Usage:
```
[ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
    tap_spacing, max_no_taps, power_exponent, fix_taps );
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`
- **`tap_spacing`** — Delay bin spacing in seconds; 5 ns corresponds to 200 MHz sampling rate; default: 5e-9
- **`max_no_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; default: 0
  | Value | Meaning                                                                                         |
  | :---: | ----------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps, n_snap]`                       |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps, n_snap]` |
  | 2     | Per snapshot; output delays `[1, 1, n_taps, n_snap]`, each snapshot independent                 |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps, n_snap]`               |

### Outputs:
- **`coeff_re_q`** — Output coefficients, real part; `[n_rx, n_tx, n_taps, n_snap]`
- **`coeff_im_q`** — Output coefficients, imaginary part; `[n_rx, n_tx, n_taps, n_snap]`
- **`delay_q`** — Output delays in seconds; `[n_rx, n_tx, n_taps, n_snap]` or `[1, 1, n_taps, n_snap]` depending on `fix_taps`

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
- Floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz
- NAN or negative value for any override parameter restores the model default

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
- **`center_freq`** — Carrier frequency; default: 5.25e9
- **`tap_spacing_s`** — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** — Channel observation time in seconds; default: 0
- **`update_rate`** — Channel update interval in seconds; relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h; movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** — Environment speed in km/h; use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist_m`** — TX-to-RX distance(s); `[n_users]` or `[1]`; default: 4.99
- **`n_floors`** — Number of floors per user for TGah or TGax models; `[n_users]` or `[1]`; default: 0
- **`uplink`** — Set `true` to generate uplink (reverse) direction; default: false
- **`offset_angles`** — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `[4, n_users]`; default: [] (auto-generate)
- **`n_subpath`** — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** — RNG seed for repeatability; `-1` uses the system random device; default: -1
- **`KF_linear`** — Overrides model KF (linear scale); default:  A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0; 
  applied to first tap only; breakpoint ignored when `KF_linear >= 0`
- **`XPR_NLOS_linear`** — Overrides NLOS cross-polarization ratio (linear scale); default: XPR NLOS: 2 (3 dB)
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
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2` (interleaved Re/Im for VV, VH, HV, HH components)
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths exceeding `threshold_dB`
- If `active_path_in` is provided, it overrides `threshold_dB` for path selection
- Optional `ant_irs_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- If `center_freq == 0`, phase calculation is disabled and only delays are computed
- If `use_absolute_delays == false`, the minimum delay (LOS delay) is subtracted from all paths

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
- **`path_length_1`** — Total path lengths from TX to IRS phase center for TX → IRS paths; `[n_path_1, 1]`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2, 1]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center for IRS → RX paths; `[n_path_2, 1]`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, interleaved complex; `[8, n_path_2]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`irs_pos`** — IRS position; `[3, 1]`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`i_irs`** — IRS codebook port index; default: `0`
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
- `M` accepts 8 rows (full polarimetric: ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) or 2 rows (scalar pressure: ReVV, ImVV only)
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs), identical to antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`

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
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, 1]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq_in]` (full pol, interleaved Re/Im for VV, VH, HV, HH) or `[2, n_path, n_freq_in]` (scalar pressure: ReVV, ImVV only)
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `[n_freq_in, 1]`
- **`freq_out`** — Target output frequencies; `[n_freq_out, 1]`
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`add_fake_los_path`** — If `true`, prepends a zero-power LOS path when none is present; default: `false`
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
- Polarization transfer matrix `M` must be normalized; rows are interleaved real/imag components.
- If `add_fake_los_path` is true, a zero-power LOS path is appended, making output size `n_path+1`.
- Setting `center_freq = 0` disables phase calculation (delays still computed).
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance from all path lengths before
  converting to delay.

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
- **`M`** — Polarization transfer matrix, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`center_freq`** — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** — If `true`, prepends a zero-power LOS path when none is present; Default: `false`

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
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH components).
- If `center_freq == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == false`, the minimum delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == true`, a zero-power LOS path is prepended when no LOS path is detected.

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
- **`path_length`** — Total path lengths from TX to RX phase center; `[n_path, 1]`
- **`M`** — Polarization transfer matrix, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`center_freq`** — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** — If `true`, prepends a zero-power LOS path when none is present; Default: `false`
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
- If `bins` is empty, equally spaced bins spanning the data range are generated

### Usage:
```
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins_in`** — Bin centers; used as-is if non-empty; if empty, equally spaced bins spanning the 
  data range are generated; `[n_bins_in]` or empty; default: `[]`
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins_in` 
  are provided; default: 201

### Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data; `[n_bins_out, n_sets]`
- **`bins_out`** — Auto-generated bins; copy of `bins_in` when non-empty `bins_in` are provided;
  `[n_bins_out = n_bins]` or `[n_bins_out = n_bins_in]`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across data sets; `[n_bins]`
- **`mu`** — Mean of the 0.1–0.9 quantiles across data sets; `[9]`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across data sets; `[9]`

---
## calc_angular_spread
Calculate azimuth and elevation angular spreads with spherical wrapping

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
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0 (no quantization)

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
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis via the Jones matrix transform `M_circ = T * M_lin * T^-1`
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths with 
  `path_length < dTR + window_size` are excluded by default
- Polarization transfer matrix `M` is stored column-major with interleaved real/imaginary parts, 8 rows per path:
  `[Re(M_vv); Im(M_vv); Re(M_vh); Im(M_vh); Re(M_hv); Im(M_hv); Re(M_hh); Im(M_hh)]`
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
- **`window_size`** — LOS exclusion window; paths within `dTR + window_size` are excluded when `include_los = false`; default: 0.01

### Outputs:
- **`xpr`** — XPR on linear scale; `[n_cir, 6]`; columns:
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 1   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 2   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_hv)^2)                      |
   | 3   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_vh)^2)                      |
   | 4   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 5   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_RL)^2)                   |
   | 6   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_LR)^2)                   |
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
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0 (no grouping)

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
  (throws if AVX2 unavailable); default: 1

### Outputs:
- **`az`** — Azimuth angles in radians; `[n, m]`
- **`el`** — Elevation angles in radians; `[n, m]`
- **`len`** — Euclidean vector length `sqrt(x² + y² + z²)`; `[n, m]`

---
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- For x in [-pi, pi]: max absolute error = 2^(-22.1); for x in [-500, 500]: 2^(-16.0)
- Either `s` or `c` may be `nullptr` to skip that computation
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision
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
- **`cos_only`** — Forsingle output: `true` returns `cos(x)`; false returns `sin(x)`; default: false

### Outputs:
- **`s`** — sin(x); `[n_elem]`
- **`c`** — cos(x); `[n_elem]`

---
## geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: `x = cos(el) cos(az) len`, `y = cos(el) sin(az) len`, `z = sin(el) len`
- Optional outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; omit from the
  output list to skip their computation
- Defaults to the GENERIC kernel (`use_kernel=1`) to preserve full double precision, matching
  MATLAB's default numeric type
- Set `use_kernel=0` for auto-selection or `use_kernel=2` to force AVX2; the AVX2 kernel
  computes in single precision internally (inputs narrowed to float, results widened back)

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
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); default: 1
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
- Supports single and multiple data sets.
- Returns interpolated results either directly or through reference argument.
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
  enables interpolation for mutiple datasets simultaneously.
- **`xI`** — Output x-axis sampling points; Length: `[nxI]`
- **`yI`** — Output y-axis sampling points; Length: `[nyI]`

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
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
[ gain, coord ] = quadriga_lib.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, ...
    center_freq, lod, verbose, sub_mesh_index, use_kernel, gpu_id );
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output of
  [obj_file_read](#obj_file_read)); each field holds a vector of length `n_mtl`
- **`center_freq`** — Center frequency
- **`lod`** — Level of detail (0–6), controls `n_path` and `n_seg`; see [generate_diffraction_paths](#generate_diffraction_paths); default: 2
- **`verbose`** — Verbosity level; default: 0 (no output)
- **`sub_mesh_index`** — 1-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation);  `[n_mesh, 1]`;
  default: `[]` (not using sub-meshes)
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear scale; `[n_pos, 1]`
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]`

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines `mtl_prop` format)

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
- **`radius`** — Radius of icosphere in meters; default: 1
- **`direction_xyz`** — Output directions in Cartesian (true) or spherical azimuth/elevation (false); default: false

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `[n_faces, 3]`
- **`length`** — Distance from origin to face plane; equals the magnitude of each `center` vector; `[n_faces]`
- **`vert`** — Vertex offsets from face center `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_faces, 9]`
- **`direction`** — Edge directions; spherical `{az1,el1,az2,el2,az3,el3}` or Cartesian
  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per `direction_xyz` flag; `[n_faces, 6]` or `[n_faces, 9]`

---
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

### Description:
- Parses a triangulated `.obj`; quads and n-gons are rejected
- Parses a triangulated `.obj`; quads and n-gons are rejected. Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- Unmatched names throw when `csv_strict` is true; otherwise they map to row 1 of the table (the transparent fallback)
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read( fn, fn_csv, csv_strict );
```

### Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** *(optional)* — Path to an EM/acoustic material CSV; must contain a `name` column, and
  row 1 is the fallback material (should be transparent, e.g. air); empty uses the built-in default table
- **`csv_strict`** *(optional)* — If true, throw when a `usemtl` material is absent from the table;
  otherwise map to row 1; default: false

### Outputs:
- **`mesh`** — Triangle vertex coordinates `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 1-based vertex indices into `vert_list` per triangle; uint64; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; uint64; `[n_mesh]`
- **`obj_names`** — Object names; cell array of strings; length `max(obj_ind)`
- **`mtl_ind`** — 1-based visual-material index per triangle; uint64; `[n_mesh]`
- **`mtl_names`** — Visual material names (raw `usemtl`); cell array of strings; length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[no_mtl, 17]`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle; uint64; `[n_mesh]`
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
- Without `mtl_ind`, no `usemtl` tags and no `.mtl` file are written
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray
  material when `bsdf` is omitted

### Usage:
```
[ vert_list_out, face_ind_out ] = quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, ...
    obj_names, mtl_names, vert_list, face_ind, bsdf, threshold );
```

### Inputs:
- **`fn`** — Path to the output `.obj` file; must end in `.obj`; if empty, no file is written (outputs are still computed)
- **`mesh`** — Triangle coordinates `{X1,Y1,Z1,...,X3,Y3,Z3}` per row; `[n_mesh, 9]`; mutually exclusive with `vert_list` and `face_ind`
- **`obj_ind`** — 1-based object index per face; `[n_mesh]`; each object must form a contiguous block
- **`mtl_ind`** — 1-based material index per face; `[n_mesh]`; omit or pass `[]` for no materials
- **`obj_names`** — Object names; cell array of strings; length >= max(obj_ind); required if `obj_ind` is given
- **`mtl_names`** — Material names; cell array of strings; length >= max(mtl_ind); required if `mtl_ind` is given
- **`vert_list`** — Vertex positions; `[n_vert, 3]`; only valid with `face_ind`; written unchanged
- **`face_ind`** — 1-based vertex indices per face; `[n_mesh, 3]`; required with `vert_list`
- **`bsdf`** — Principled BSDF values for the `.mtl` file; `[n_mtl, 17]`; see [obj_file_read](#obj_file_read) for the column layout
- **`threshold`** — Vertex co-location distance for merging within an object; default: 0.001 (1 mm)

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 1-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)

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
- **`sub_cloud_index`** — Rrow indices marking the start of each sub-cloud; use [point_cloud_segmentation](#point_cloud_segmentation) 
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
- **`vec_size`** — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`; default: 1

### Outputs:
- **`pointsR`** — Reorganized point cloud with points grouped by sub-cloud; `[n_pointsR, 3]`
- **`sub_cloud_index`** — 1-based starting index of each sub-cloud within `pointsR`; `[n_sub]`
- **`forward_index`** — 1-based index map from `points` to `pointsR`; padding entries are `0`; `[n_pointsR]`
- **`reverse_index`** — 1-based index map from `pointsR` back to `points`; `[n_points]`

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
- **`obj_ind`** — Object index per mesh element; enables per-object output; `[n_mesh]`
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
- Rays with `fbs_ind = 0` (no interaction) are omitted from output, so `n_rayN ≤ n_ray`
- Output direction encoding (spherical/Cartesian) matches input `tridir` format
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves)
- Types 3–4 (scalar) use TE-only reflection with no total internal reflection, suitable for
  acoustic simulation with impedance-mapped material parameters (ε derived from Z)
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
[ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, ...
    normal_vecN, out_typeN ] = quadriga_lib.ray_mesh_interact( interaction_type, center_frequency, ...
    orig, dest, fbs, sbs, mesh, mtl_ind, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, orig_length );
```

### Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
- **`center_frequency`** — Center frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`fbs`**, **`sbs`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see `obj_file_read`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output
  of [obj_file_read](#obj_file_read)); each field holds a vector of length `n_mtl`
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); uint32; `[n_ray]`
- **`trivec`** — Beam wavefront triangle vertices relative to origin; order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 9]`; default: `[]`
- **`tridir`** — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian; default: `[]`
- **`orig_length`** — Accumulated path length at origin; default: 0; `[n_ray]`; default: `[]`

### Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear, includes in-medium attenuation, excludes FSPL);
  averaged over TE/TM polarizations for types 0–2, TE-only for types 3–4; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex `[ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH]`;
  for types 3–4 (scalar): `[Re Im 0 0 0 0 0 0]` where Re+jIm is the scalar pressure coefficient; includes interaction gain,
  TE/TM coefficients, incidence plane orientation, in-medium attenuation (excludes FSPL); `[n_rayN, 8]`
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if `trivec`/`tridir` not provided
- **`orig_lengthN`** — Path length from `orig` to `origN`, added to input `orig_length` if given; `[n_rayN]`
- **`fbs_angleN`** — Incidence angle at FBS; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (Inf if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code (int32); `[n_rayN]`
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
- **`sub_cloud_index`** — Segment boundary indices for the point cloud; use [point_cloud_segmentation](#point_cloud_segmentation) to gnerate;
  uint32; `[n_sub]`; default: `[]` (not using sub-clouds)
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable; 
  auto mode selects CUDA when `n_points >= 10000` and CUDA is available, else AVX2, else GENERIC; default: 0
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
- Internal computations always use single precision for AVX2 and CUDA kernels; only GENERIC has `double` support

### Usage:
```
[ fbs, sbs, no_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( ...
    orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id );
```

### Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`
- **`sub_mesh_index`** — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; 
  use [triangle_mesh_segmentation](#triangle_mesh_segmentation) to generate; `[n_sub]`; default: `[]` (not using sub-meshes)
- **`aabb`** — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; 
  if empty or omitted, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable; 
  auto mode selects CUDA when `n_ray >= 10000` and CUDA is available, else AVX2, else GENERIC; default: 0
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
 `[n_triangles_in]`; default: `[]`

### Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_ind_out`** — Material indices for the subdivided triangles; `[n_triangles_out]`; only
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
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; `[n_triangles, 9]`
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
  `[n_mesh]` or empty; padding entries get index 1; default: `[]`

### Outputs:
- **`triangles_out`** — Reordered and padded triangle vertices; `[n_meshR, 9]`
- **`sub_mesh_index`** — 1-based start indices of sub-meshes in `triangles_out`; uint32; `[n_sub]`
- **`mesh_index`** — 1-based mapping from original to reorganized mesh (0 = padding); uint32; `[n_meshR]`
- **`mtl_ind_out`** — Reordered and padded material indices; `[n_meshR]`; only populated if `mtl_ind_in` is given

