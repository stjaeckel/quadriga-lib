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
| [arrayant_generate](#arrayant_generate) | Generate an array antenna of a given type |
| [arrayant_interpolate](#arrayant_interpolate) | Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles |
| [arrayant_rotate_pattern](#arrayant_rotate_pattern) | Rotate antenna radiation patterns using Euler rotations |
| [generate_speaker](#generate_speaker) | Generate a parametric frequency-dependent loudspeaker directivity model |
| [qdant_read](#qdant_read) | Reads array antenna data from QDANT files |
| [qdant_write](#qdant_write) | Writes array antenna data to QDANT files |

---
<a id="arrayant_calc_beamwidth"></a>
## arrayant_calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_calc_beamwidth) · [Python](../python/array-antenna-functions.md#calc_beamwidth)

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
[ beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg ] = ...
    quadriga_lib.arrayant_calc_beamwidth( arrayant, i_element, threshold_dB );
```

### Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements
- **`threshold_dB`** — Threshold in dB, positive and finite; default: 3 (equivalent to FWHM)

### Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degree; `[n_out, n_freq]`; with `n_out = n_elements` when `i_element` is omitted/empty
- **`beamwidth_el_deg`** — Elevation beamwidth in degree; `[n_out, n_freq]`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degree; `[n_out, n_freq]`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degree; `[n_out, n_freq]`

### See also:
- [arrayant_combine_pattern](#arrayant_combine_pattern) (to apply element coupling before calculating the beamwidth)
- [arrayant_calc_directivity](#arrayant_calc_directivity) (directivity in dBi of array antenna elements)

---
<a id="arrayant_calc_directivity"></a>
## arrayant_calc_directivity
Calculates the directivity in dBi of array antenna elements

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_calc_directivity) · [Python](../python/array-antenna-functions.md#calc_directivity)

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi

### Usage:
```matlab
directivity_dBi = quadriga_lib.arrayant_calc_directivity( arrayant, i_element );
```

### Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements

### Output:
- **`directivity_dBi`** — Directivity in dBi; `[n_out, n_freq]`; with `n_out = n_elements` when `i_element` is omitted/empty

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
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in, freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array may contain a frequency-dependent model
- **`freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is
  recomputed (interpolated) at each requested frequency; default: `[]` (each input entry's `center_freq` unchanged)
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; strictly increasing; default: `[]` (input grid)
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; strictly increasing; default: `[]` (input grid)
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics;
  default: 299792458

### Output:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [arrayant_generate](#arrayant_generate). Single struct when both input and
  `freq_new` describe a single frequency; struct array of size `numel(freq_new)` (or
  `numel(arrayant_in)` when `freq_new` is omitted) otherwise.

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
  [arrayant_generate](#arrayant_generate); a struct array may contain a frequency-dependent model
- **`arrayant_in2`** — Struct containing the second arrayant data; must match the sampling grids of
  `arrayant_in1`, and for multi-frequency the entry count and `center_freq` per entry

### Output:
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
- If `i_source` is a vector, `i_target` must have the same length; copies are performed
  pairwise as `i_source(i)` to `i_target(i)`

### Usage:
```matlab
arrayant_out = quadriga_lib.arrayant_copy_element( arrayant_in, i_source, i_target );
```

### Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in
  [arrayant_generate](#arrayant_generate); a struct array may contain a frequency-dependent model
- **`i_source`** — Index of the source element(s); 1-based; scalar or `[n_copy]`
- **`i_target`** — Index of the destination element(s); 1-based; scalar or `[n_copy]`;
  if `i_source` is a vector, must have the same length

### Output:
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
quadriga_lib.arrayant_export_obj_file( fn, arrayant, directivity_range_dB, colormap, object_radius, ...
    icosphere_n_div, i_element, i_freq );
```

### Inputs:
- **`fn`** — Output OBJ filename; must not be empty; filename must end in `.obj`
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in
  [arrayant_generate](#arrayant_generate); a struct array may contain a frequency-dependent model
- **`directivity_range_dB`** — Dynamic range of the visualized directivity pattern in dB; default: 30 dB
- **`colormap`** — Colormap name; default: jet; Available: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
- **`object_radius`** — Radius of the exported object; default: 1 m
- **`icosphere_n_div`** — Icosphere subdivision count; higher = finer mesh; see [icosphere](site-specific-simulation-tools.md#icosphere); default: 4
- **`i_element`** — Element indices to export; 1-based; default: `[]` (all elements)
- **`i_freq`** — Frequency index to export from a multi-frequency arrayant struct array; 1-based;
  default: 1; must satisfy `1 <= i_freq <= n_freq`

### See also:
- [icosphere](site-specific-simulation-tools.md#icosphere) (icosphere primitive used internally)

---
<a id="arrayant_generate"></a>
## arrayant_generate
Generate an array antenna of a given type

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- One function for all antenna types. The panel types take the parameters of 3GPP TR 38.901 clause 7.3 as tuples: trailing entries may be omitted; 0 (in `panel`, `ports`, `pol`) or NaN (in `spacing`, `steer`, `element`) selects the default of that entry; more entries than listed raise an error.
- An input that a type does not use raises an error, also `separate_beams` or `apply_weights` set to `true`. The result has the given `center_freq`.
- Panel arrays: elements ordered polarization fastest, then `m` (upwards along z), `n` (along +y), `mg`, `ng`, positions centred on the origin; ports in the same order with `m_p`, `n_p`. Each port feeds a sub-array of `M / M_P` × `N / N_P` elements with the weights of TR 38.901 eq. 7.3-1 (`tilt`, and by analogy along y `pan`), stored in the coupling matrix.
- `multibeam`: maximum-ratio-transmission weights for the directions in `beams`; one port per polarization, with `separate_beams` one per polarization and beam.

### Usage:
```matlab
% omni, xpol, dipole, half-wave-dipole
ant = quadriga_lib.arrayant_generate( type, res, center_freq );

% Custom-shape element
ant = quadriga_lib.arrayant_generate( 'custom', res, center_freq, [], [], [], [], [], element );

% Uniform linear array (omni element) and TR 38.901 panel array
ant = quadriga_lib.arrayant_generate( 'ula', res, center_freq, panel, spacing, ports, pol, steer, [], ...
    pattern, [], [], apply_weights );
ant = quadriga_lib.arrayant_generate( '3GPP', res, center_freq, panel, spacing, ports, pol, steer, element, ...
    pattern, [], [], apply_weights );

% Multi-beam panel array
ant = quadriga_lib.arrayant_generate( 'multibeam', res, center_freq, panel, spacing, [], pol, [], element, ...
    [], beams, separate_beams, apply_weights );
```

### Inputs:
- **`type`** — Antenna type and the inputs it takes besides `res` and `center_freq`:
   | `type`                                           | Inputs                                                                             |
   | ------------------------------------------------ | ---------------------------------------------------------------------------------- |
   | `omni`, `xpol`                                   | none                                                                               |
   | `dipole` (or `short-dipole`), `half-wave-dipole` | none                                                                               |
   | `custom`                                         | `element`                                                                          |
   | `ula`                                            | `panel`, `spacing`, `ports`, `pol`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)                               | `panel`, `spacing`, `ports`, `pol`, `steer`, `element`, `pattern`, `apply_weights` |
   | `multibeam`                                      | `panel`, `spacing`, `pol`, `element`, `beams`, `separate_beams`, `apply_weights`  |
- **`res`** — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: default; default: 10 for `omni` and `xpol`, 1 otherwise
- **`center_freq`** — Center frequency in Hz; positive and finite; default: 299792458
- **`panel`** — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; at most 1e7 elements per polarization; default: 1 (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing (centre to centre); >= 0; default: `[0.5, d_V, M d_V, N d_H]`
- **`ports`** — `[M_P, N_P]`: ports per column and per row of a panel and polarization; must divide `M` and `N`; default: `[M, N]` (one port per element)
- **`pol`** — `[polarization, model]`: polarization 1 (vertical), 2 (H/V) or 3 (±45°); model 1 or 2 (TR 38.901 Model-1, Model-2); default: `[1, 1]`
- **`steer`** — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt (positive down) and pan (positive towards +y); `[1, 2]` for all panels or `[Mg Ng, 2]` with one row per panel; one column: pan 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`; default: 0
- **`element`** — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi; default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity
- **`pattern`** — Element struct to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; default: the element of the type
- **`beams`** — `[2 or 3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); default: `[0; 0; 1]`
- **`separate_beams`** — If `true`, one port per beam (weights ignored); default: `false`
- **`apply_weights`** — If `true`, combine each port into one pattern at its sub-array centre; default: `false`

### Outputs:
- **`ant`** — Struct with fields:
   | Field            | Description                                               | Size                                   |
   | ---------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part                        | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`     | e-theta field component, imaginary part                   | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`       | e-phi field component, real part                          | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`       | e-phi field component, imaginary part                     | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, strictly increasing       | `[1, n_azimuth]`                       |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, strictly increasing | `[1, n_elevation]`                     |
   | `element_pos`    | Element (x,y,z) positions in meters                       | `[3, n_elements]`                      |
   | `coupling_re`    | Coupling matrix, real part                                | `[n_elements, n_ports]`                |
   | `coupling_im`    | Coupling matrix, imaginary part                           | `[n_elements, n_ports]`                |
   | `center_freq`    | Center frequency in Hz                                    | scalar                                 |
   | `name`           | Name of the array antenna object                          | string                                 |

   If `ant` is used as an input to other functions, fields `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are mandatory; remaining fields are optional (defaults: unit coupling, zero positions, 299792458 Hz).

---
<a id="arrayant_interpolate"></a>
## arrayant_interpolate
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_interpolate) · [Python](../python/array-antenna-functions.md#interpolate)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Single-frequency mode (a 1x1 struct and no `freq_new`): up to 8 outputs. Multi-frequency mode (a struct array with
  more than one entry, or `freq_new` given): for each target frequency, the two entries whose center frequencies
  bracket it are interpolated and blended; frequencies outside the entries take the nearest entry; up to 4 outputs
- `azimuth` of size `[1, 0]` or `[n_out, 0]` gives empty outputs; `[]` raises an error
- NaN or infinite angles give NaN in all outputs of that angle and output, without an error
- On a pole of the local frame (a direction along the z-axis of the oriented antenna) the local azimuth is chosen so
  that the local and the global theta unit vectors agree: the polarization is not rotated there
- Kernels: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the peak
  field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole of the local frame up to about
  2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most 4e-4 rad) in the local
  elevation; auto selects GENERIC in this version, which may change in a later version

### Usage:
```matlab
% Single-frequency mode
[ V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma ] = quadriga_lib.arrayant_interpolate( ...
    arrayant, azimuth, elevation, i_element, orientation, element_pos, [], use_kernel );

% Multi-frequency mode
[ V_re, V_im, H_re, H_im ] = quadriga_lib.arrayant_interpolate( arrayant, azimuth, elevation, ...
    i_element, orientation, element_pos, freq_new, use_kernel );
```

### Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate); a struct
  array may contain a frequency-dependent model
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` (same angles for all outputs) or `[n_out, n_ang]` (one angle per
  output)
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements
- **`orientation`** — Euler angles (bank, tilt, heading) of the antenna; `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]` or
  `[3, n_out, n_ang]`; default: no rotation
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; `[3, n_out]`; default: the
  positions of the arrayant
- **`freq_new`** — Target frequencies in Hz; selects the multi-frequency mode; `[n_freq]`; default: the `center_freq`
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

### Output:
- **`arrayant_out`** — Arrayant struct, or struct array of size `numel(arrayant_in)` for multi-frequency input;
  field layout as documented in [arrayant_generate](#arrayant_generate)

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [C++](../cpp/array-antenna-functions.md#generate_speaker) · [Python](../python/array-antenna-functions.md#generate_speaker)

- Returns one arrayant per frequency sample; each has a single element with the real-valued
  directivity pattern in `e_theta_re` and `center_freq` set to the corresponding frequency.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining
  results via [arrayant_concat](#arrayant_concat) and `element_pos`; crossover behavior emerges from overlapping bandpass
  responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity_dB - 85) / 20)`.
- If `center_freq` is empty, third-octave band center frequencies are auto-generated from one band
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
- For `horn`, `hor_coverage_deg`, `ver_coverage_deg` and `horn_control_freq` set to NaN (the default) or <= 0 select 90°, 60°
  and `f_ctrl = c/(2pi·radius)`.

### Usage:
```matlab
arrayant = quadriga_lib.generate_speaker( driver_type, radius, lower_cutoff, upper_cutoff, ...
    lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB, radiation_type, hor_coverage_deg, ...
    ver_coverage_deg, horn_control_freq, baffle_width, baffle_height, center_freq, ...
    angular_resolution_deg );
```

### Inputs:
- **`driver_type`** — Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** — Effective radiating radius in m, > 0; cone/dome radius for piston, mouth radius for horn; default: 0.05
- **`lower_cutoff`** — Lower -3 dB bandpass frequency in Hz, > 0; default: 80
- **`upper_cutoff`** — Upper -3 dB bandpass frequency in Hz, larger than `lower_cutoff`; default: 12000
- **`lower_rolloff_slope_dB`** — Low-frequency rolloff in dB/octave, > 0 (12 dB/oct = 2nd-order Butterworth); default: 12
- **`upper_rolloff_slope_dB`** — High-frequency rolloff in dB/octave, > 0; default: 12
- **`sensitivity_dB`** — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`; default: `hemisphere`
- **`hor_coverage_deg`** — Horn horizontal coverage angle in degrees; NaN or <= 0: default 90; default: NaN
- **`ver_coverage_deg`** — Horn vertical coverage angle in degrees; NaN or <= 0: default 60; default: NaN
- **`horn_control_freq`** — Horn pattern control frequency in Hz; NaN or <= 0: default `c/(2pi·radius)`; default: NaN
- **`baffle_width`** — Baffle width in m, > 0; used by `hemisphere` model; default: 0.15
- **`baffle_height`** — Baffle height in m, > 0; used by `hemisphere` model; default: 0.25
- **`center_freq`** — Frequency sample points; `[n_freq]`; default: `[]` (auto-generated third-octave bands)
- **`angular_resolution_deg`** — Azimuth and elevation sampling grid resolution in degrees, 0.1 to 90; default: 5

### Output:
- **`arrayant`** — Struct array with one arrayant per frequency sample; directivity is stored in
  `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180 degree phase inversion; 
  see [arrayant_generate](#arrayant_generate) for struct fields; `[1, n_freq]`

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
- With `id`, a single entry is read; useful for picking one frequency from a multi-frequency file
- An `id` that is not in the file is an error; the first entry counts as ID 1 only if it has no `id` attribute
- Malformed files are an error, also when they declare more elements (`NoElements`) than the file has data for;
  elements without pattern data read as zeros (at most 2^24 zero values per entry)

### Usage:
```matlab
% Multi-frequency read (struct array, all entries)
[ ant, layout ] = quadriga_lib.qdant_read( fn );

% Single-frequency read (struct output)
[ ant, layout ] = quadriga_lib.qdant_read( fn, id );
```

### Inputs:
- **`fn`** — Path to the QDANT file; string; must not be empty
- **`id`** — 1-based ID of the antenna entry to read; pass `[]` or omit to read every entry in the file

### Outputs:
- **`ant`** — Arrayant struct (single entry) or struct array (multiple entries); field layout as documented in [arrayant_generate](#arrayant_generate)
- **`layout`** — Matrix of element IDs describing how entries are arranged in the file; datatype: uint32

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
  1-based IDs; in this mode appending to an existing file is not allowed and will cause an error

### Usage:
```matlab
id_in_file = quadriga_lib.qdant_write( fn, arrayant, id, layout );
```

### Inputs:
- **`fn`** — Output QDANT filename; string; must not be empty
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [arrayant_generate](#arrayant_generate);
  a struct array writes a frequency-dependent model and requires `id` and `layout` to be omitted
- **`id`** — Target ID of the antenna inside the file; default: max-ID in existing file + 1 (or 1 if the file does not exist);
  must be omitted for a multi-frequency model
- **`layout`** — Matrix organizing multiple antenna IDs within the file; must only reference IDs present in the file; uint32;
  default: a row of all IDs in the file (replaces a layout written before)

### Output:
- **`id_in_file`** — ID assigned to the antenna in the file after writing; uint32; set to 0 in multi-frequency (struct array) mode

### See also:
- [qdant_read](#qdant_read) (for reading QDANT data)
- [arrayant_generate](#arrayant_generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format ([QDANT](../../formats/qdant.md))
