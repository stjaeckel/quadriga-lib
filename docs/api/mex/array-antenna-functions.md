# Array antenna functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-array-antenna-functions

| Function | Summary |
| --- | --- |
| [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) | Calculate the beamwidth and pointing angles of array antenna elements in degrees |
| [arrayant_calc_directivity](#arrayant_calc_directivity) | Calculates the directivity in dBi of array antenna elements |
| [arrayant_combine_pattern](#arrayant_combine_pattern) | Combine element patterns, positions, and coupling weights into effective radiation patterns |
| [arrayant_concat](#arrayant_concat) | Concatenate two arrayant structs into a single one |
| [arrayant_copy_element](#arrayant_copy_element) | Create copies of array antenna elements |
| [arrayant_export_obj_file](#arrayant_export_obj_file) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [arrayant_generate](#arrayant_generate) | Generates predefined array antenna models |
| [arrayant_interpolate](#arrayant_interpolate) | Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles |
| [arrayant_rotate_pattern](#arrayant_rotate_pattern) | Rotate antenna radiation patterns using Euler rotations |
| [generate_speaker](#generate_speaker) | Generate a parametric frequency-dependent loudspeaker directivity model |
| [qdant_read](#qdant_read) | Reads array antenna data from QDANT files |
| [qdant_write](#qdant_write) | Writes array antenna data to QDANT files |

---
<a id="arrayant_calc_beamwidth"></a>
## arrayant_calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

Also in: [C++](../cpp/array-antenna-class.md#.calc_beamwidth_deg) · [Python](../python/array-antenna-functions.md#calc_beamwidth)

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
```matlab
% Input as struct (struct mode)
[ beamwidth_az, beamwidth_el, az_point_ang, ...
    el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( arrayant );

[ beamwidth_az, beamwidth_el, az_point_ang, ...
    el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( arrayant, i_element, threshold_dB );

% Separate inputs (split mode)
[ beamwidth_az, beamwidth_el, az_point_ang, ...
    el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid );

[ beamwidth_az, beamwidth_el, az_point_ang, ...
    el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, i_element, threshold_dB );
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
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, strictly increasing; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, strictly increasing; `[n_elevation]`
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
<a id="arrayant_calc_directivity"></a>
## arrayant_calc_directivity
Calculates the directivity in dBi of array antenna elements

Also in: [C++](../cpp/array-antenna-class.md#.calc_directivity_dBi) · [Python](../python/array-antenna-functions.md#calc_directivity)

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi

### Usage:
```matlab
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
<a id="arrayant_combine_pattern"></a>
## arrayant_combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_combine_pattern) · [Python](../python/array-antenna-functions.md#combine_pattern)

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Usage:
```matlab
% Input as struct (struct mode)
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in );
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in, center_freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );

% Separate outputs, struct input (single-freq only)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, ...
    coupling_im, center_freq, name] = quadriga_lib.arrayant_combine_pattern( arrayant_in );

% Separate inputs (split-mode, single-freq only); propagation_speed is the 16th input
arrayant_out = quadriga_lib.arrayant_combine_pattern( [], center_freq_new, azimuth_grid_new, ...
    elevation_grid_new, e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, ...
    element_pos, coupling_re, coupling_im, center_freq, name, propagation_speed );
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`center_freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is 
  recomputed (interpolated) at each requested frequency. If omitted or empty, the function uses each input 
   entry's `center_freq` unchanged.
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; strictly increasing; defaults to input grid
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; strictly increasing; defaults to input grid
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
<a id="arrayant_concat"></a>
## arrayant_concat
Concatenate two arrayant structs into a single one

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_concat) · [Python](../python/array-antenna-functions.md#concat)

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
```matlab
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
<a id="arrayant_copy_element"></a>
## arrayant_copy_element
Create copies of array antenna elements

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_copy_element) · [Python](../python/array-antenna-functions.md#copy_element)

- Copies a source element to one or more destination slots within an arrayant
- Array is resized if any destination index exceeds the current number of elements
- Coupling matrix entries for newly added elements are set to identity; existing coupling is preserved
- Supports multi-frequency arrayant models: when `arrayant_in` is a struct array, the same copy is
  applied to every entry and a struct array of equal size is returned
- If `source_element` is a vector, `dest_element` must have the same length; copies are performed
  pairwise as `source_element(i)` to `dest_element(i)`

### Usage:
```matlab
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
<a id="arrayant_export_obj_file"></a>
## arrayant_export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_export_obj_file) · [Python](../python/array-antenna-functions.md#export_obj_file)

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives a finer mesh
- Supports multi-frequency arrayant models: pass a struct array and select the entry to export
  via the `i_freq` argument

### Usage:
```matlab
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
- **`icosphere_n_div`** — Icosphere subdivision count; higher = finer mesh; see [icosphere](site-specific-simulation-tools.md#icosphere); default: 4
- **`i_element`** — Element indices to export; 1-based; uint64; empty = export all elements
- **`i_freq`** — Frequency index to export from a multi-frequency arrayant struct array; 1-based; 
  uint64; default: 1; must satisfy `1 <= i_freq <= n_freq`

### See also:
- [icosphere](site-specific-simulation-tools.md#icosphere) (icosphere primitive used internally)

---
<a id="arrayant_generate"></a>
## arrayant_generate
Generates predefined array antenna models

Also in: [C++](../cpp/array-antenna-functions.md#generate_arrayant_3GPP) · [Python](../python/array-antenna-functions.md#generate)

- Dispatches to one of several C++ generator functions based on the `type` string
- Supported types: `omni`, `dipole` (or `short-dipole`), `half-wave-dipole`, `xpol`, `custom`,
  `ula`, `3GPP` (or `3gpp`), `multibeam`, `multibeam_sep`
- All positional arguments after `type` are optional and type-specific; use `[]` to skip and fall
  back to defaults

### Usage:
```matlab
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
ant = quadriga_lib.arrayant_generate('ula', res, freq, [], [], [], [], N, [], [], spacing, [], [], ...
    [], [], pattern);

% 3GPP-NR array (default 3GPP element pattern)
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [], M, N, pol, tilt, spacing, Mg, ...
    Ng, dgv, dgh);

% 3GPP-NR array with custom element beamwidth
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, az_3dB, el_3dB, bfr, M, N, pol, tilt, ...
    spacing, Mg, Ng, dgv, dgh);

% 3GPP-NR array with custom per-element pattern struct
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [], M, N, pol, tilt, spacing, Mg, ...
    Ng, dgv, dgh, pattern);

% Multi-beam M×N array (one combined beam / one beam per direction)
ant = quadriga_lib.arrayant_generate('multibeam', res, freq, az_3dB, el_3dB, bfr, M, N, pol, dir, ...
    spacing);
ant = quadriga_lib.arrayant_generate('multibeam_sep', res, freq, az_3dB, el_3dB, bfr, M, N, pol, ...
    dir, spacing);

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
- **`res`** — Pattern sampling grid resolution in degrees, 0.1 to 90; default: 10 for types omni+xpol, 1 otherwise
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
   | Field            | Description                                               | Size                                   |
   | ---------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part                        | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`     | e-theta field component, imaginary part                   | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`       | e-phi field component, real part                          | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`       | e-phi field component, imaginary part                     | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, strictly increasing       | `[n_azimuth]`                          |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, strictly increasing | `[n_elevation]`                        |
   | `element_pos`    | Element (x,y,z) positions in meters                       | `[3, n_elements]`                      |
   | `coupling_re`    | Coupling matrix, real part                                | `[n_elements, n_ports]`                |
   | `coupling_im`    | Coupling matrix, imaginary part                           | `[n_elements, n_ports]`                |
   | `center_freq`    | Center frequency in Hz                                    | scalar                                 |
   | `name`           | Name of the array antenna object                          | string                                 |

   If `ant` is used as an input to other functions, fields `e_theta_re`, `e_theta_im`, `e_phi_re`, 
   `e_phi_im`, `azimuth_grid`, `elevation_grid` are mandatory; remaining fields are optional (defaults: 
   unit coupling, zero positions, 299792458 Hz).

---
<a id="arrayant_interpolate"></a>
## arrayant_interpolate
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_interpolate) · [Python](../python/array-antenna-functions.md#interpolate)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Single-frequency mode (a 1x1 struct and no `frequency`): up to 8 outputs. Multi-frequency mode (a struct array with
  more than one entry, or `frequency` given): for each target frequency, the two entries whose center frequencies
  bracket it are interpolated and blended; frequencies outside the entries take the nearest entry; up to 4 outputs
- `azimuth` of size `[1, 0]` or `[n_out, 0]` gives empty outputs; `[]` raises an error
- NaN or infinite angles give NaN in all outputs of that angle and output, without an error
- Kernels: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the peak
  field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole of the local frame up to about
  2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most 4e-4 rad) in the local
  elevation; auto selects GENERIC in this version, which may change in a later version

### Usage:
```matlab
% Single-frequency mode
[ V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma ] = quadriga_lib.arrayant_interpolate( ...
    arrayant, azimuth, elevation, element, orientation, element_pos, [], use_kernel );

% Multi-frequency mode
[ V_re, V_im, H_re, H_im ] = quadriga_lib.arrayant_interpolate( arrayant, azimuth, elevation, ...
    element, orientation, element_pos, frequency, use_kernel );
```

### Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate); a struct
  array may contain a frequency-dependent model
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` (same angles for all outputs) or `[n_out, n_ang]` (one angle per
  output)
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements
- **`orientation`** — Euler angles (bank, tilt, heading) of the antenna; `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]` or
  `[3, n_out, n_ang]`; default: no rotation
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; `[3, n_out]`; default: the
  positions of the arrayant
- **`frequency`** — Target frequencies in Hz; selects the multi-frequency mode; `[n_freq]`; default: the `center_freq`
  of the entries
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`; multi-frequency mode:
  `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; same size as `V_re`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the antenna; negative for
  elements closer to the source; `[n_out, n_ang]`; single-frequency mode only
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `[n_out, n_ang]`;
  single-frequency mode only
- **`gamma`** — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`; single-frequency
  mode only

### See also:
- [arrayant_generate](#arrayant_generate) (arrayant struct layout)
- [qdant_read](#qdant_read) (load arrayant data)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency struct arrays)

---
<a id="arrayant_rotate_pattern"></a>
## arrayant_rotate_pattern
Rotate antenna radiation patterns using Euler rotations

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_rotate_pattern) · [Python](../python/array-antenna-functions.md#rotate_pattern)

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [arrayant_interpolate](#arrayant_interpolate) returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); coupling, `center_freq` and `name` are kept
- Usage 0 and 1 adjust the sampling grid for non-uniformly sampled antennas; this needs all elements, so a subset
  in `i_element` is an error when the grid must be adjusted
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid
  spurious polarization effects

### Usage:
```matlab
arrayant_out = quadriga_lib.arrayant_rotate_pattern(arrayant_in, orientation, usage, i_element);
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array represents a frequency-dependent model
- **`orientation`** — Euler angles (bank, tilt, heading) in rad; 3-element vector
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
- **`i_element`** — Element indices to rotate, 1-based, no duplicates; default: all elements; `[n]`

### Outputs:
- **`arrayant_out`** — Arrayant struct, or struct array of size `numel(arrayant_in)` for multi-frequency input;
  field layout as documented in [arrayant_generate](#arrayant_generate)

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [C++](../cpp/array-antenna-functions.md#generate_speaker) · [Python](../python/array-antenna-functions.md#generate_speaker)

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
```matlab
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
- **`angular_resolution`** - Azimuth and elevation sampling grid resolution in degrees, 0.1 to 90; default: 5

### Outputs:
- **`arrayant`** - Struct array with one arrayant per frequency sample; directivity is stored in
  `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180 degree phase inversion; 
  see [arrayant_generate](#arrayant_generate) for struct fields; `[n_freq]`

---
<a id="qdant_read"></a>
## qdant_read
Reads array antenna data from QDANT files

Also in: [C++](../cpp/array-antenna-functions.md#qdant_read) · [Python](../python/array-antenna-functions.md#qdant_read)

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
```matlab
% Multi-frequency read (struct array, all entries)
[ ant, layout ] = quadriga_lib.qdant_read( fn );

% Single-frequency read (struct output)
[ ant, layout ] = quadriga_lib.qdant_read( fn, id );

% Single-frequency read (separate fields)
[ e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name, layout ] = quadriga_lib.qdant_read( fn, id );
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
- [qdant_write](#qdant_write) (for writing QDANT data)
- [arrayant_generate](#arrayant_generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format ([QDANT](../../formats/qdant.md))

---
<a id="qdant_write"></a>
## qdant_write
Writes array antenna data to QDANT files

Also in: [C++](../cpp/array-antenna-functions.md#qdant_write) · [Python](../python/array-antenna-functions.md#qdant_write)

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
```matlab
% Arrayant as struct
id_in_file = quadriga_lib.qdant_write( fn, arrayant, id, layout );

% Arrayant as separate inputs
id_in_file = quadriga_lib.qdant_write( fn, [], id, layout, e_theta_re, e_theta_im, e_phi_re, ...
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
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, strictly increasing; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, strictly increasing; `[n_elevation]`
- **`element_pos`** — Element (x,y,z) positions; `[3, n_elements]`; default: zeros
- **`coupling_re`** — Coupling matrix, real part; `[n_elements, n_ports]`; default: identity
- **`coupling_im`** — Coupling matrix, imaginary part; `[n_elements, n_ports]`; default: zeros
- **`center_freq`** — Center frequency in Hz; default: 299792458
- **`name`** — Name of the array antenna object; string

### Outputs:
- **`id_in_file`** — ID assigned to the antenna in the file after writing; set to 0 in multi-frequency (struct array) mode

### See also:
- [qdant_read](#qdant_read) (for reading QDANT data)
- [arrayant_generate](#arrayant_generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format ([QDANT](../../formats/qdant.md))
