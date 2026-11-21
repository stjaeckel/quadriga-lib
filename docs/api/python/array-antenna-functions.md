# Array antenna functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-array-antenna-functions

| Function | Summary |
| --- | --- |
| [calc_beamwidth](#calc_beamwidth) | Calculate the beamwidth and pointing angles of array antenna elements in degrees |
| [calc_directivity](#calc_directivity) | Calculates the directivity in dBi of array antenna elements |
| [combine_pattern](#combine_pattern) | Combine element patterns, positions, and coupling weights into effective radiation patterns |
| [concat](#concat) | Concatenate two array antennas into a single one |
| [copy_element](#copy_element) | Create copies of array antenna elements |
| [export_obj_file](#export_obj_file) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [generate](#generate) | Generate an array antenna of a given type |
| [generate_speaker](#generate_speaker) | Generate a parametric frequency-dependent loudspeaker directivity model |
| [interpolate](#interpolate) | Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles |
| [qdant_read](#qdant_read) | Reads array antenna data from QDANT files |
| [qdant_write](#qdant_write) | Writes array antenna data to QDANT files |
| [rotate_pattern](#rotate_pattern) | Rotate antenna radiation patterns using Euler rotations |

---
<a id="calc_beamwidth"></a>
## calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_calc_beamwidth) · [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_beamwidth)

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern
  (≈100x finer grid in each direction than the antenna sampling grid)
- The width is that of the main lobe, searched outwards from the maximum; side lobes above the threshold
  are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full
  circle; 360° when the whole cut is above the threshold (e.g. a beam pointing to a pole)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles;
  at most 180°
- Calculated per element, not per port; ignores element coupling
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

### Usage:
```python
beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg = \
    quadriga_lib.arrayant.calc_beamwidth( arrayant, *, i_element, threshold_dB )
```

### Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [generate](#generate) for the field layout
- **`i_element`** — Element indices, 0-based, duplicates allowed; empty: all elements; `(n_out,)`; default: None (all elements)
- **`threshold_dB`** — Threshold in dB, positive and finite (3 = FWHM); default: 3.0

### Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degrees; `(n_out,)` for single-frequency input,
  `(n_out, n_freq)` for multi-frequency input; `n_out = n_elements` when `i_element` is None
- **`beamwidth_el_deg`** — Elevation beamwidth in degrees; same shape as `beamwidth_az_deg`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degrees; same shape as `beamwidth_az_deg`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degrees; same shape as `beamwidth_az_deg`

### See also:
- [combine_pattern](#combine_pattern) (apply element coupling before calculating beamwidth)
- [calc_directivity](#calc_directivity) (directivity in dBi of array antenna elements)

---
<a id="calc_directivity"></a>
## calc_directivity
Calculates the directivity in dBi of array antenna elements

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_calc_directivity) · [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_directivity)

- Directivity = 10·log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

### Usage:
```python
directivity_dBi = quadriga_lib.arrayant.calc_directivity( arrayant, *, i_element )
```

### Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [generate](#generate) for the field layout
- **`i_element`** — Element indices, 0-based, duplicates allowed; empty: all elements; `(n_out,)`; default: None (all elements)

### Output:
- **`directivity_dBi`** — Directivity in dBi; `(n_out,)` for a single-frequency input,
  `(n_out, n_freq)` for a multi-frequency input; `n_out = n_elements` when `i_element` is None

### See also:
- [combine_pattern](#combine_pattern) (apply element coupling before calculating directivity)
- [calc_beamwidth](#calc_beamwidth) (calculates the beam width of array antennas)

---
<a id="combine_pattern"></a>
## combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_combine_pattern) · [MATLAB](../mex/array-antenna-functions.md#arrayant_combine_pattern)

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- Accepts a frequency-dependent antenna (4D pattern fields); the pattern is then combined per frequency
- `freq_new` may recompute (interpolate) the combined pattern at one or more requested frequencies
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Usage:
```python
# Single frequency
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant )
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, *, freq_new, azimuth_grid, \
    elevation_grid )

# Multiple frequencies (freq_new as a 1D array, or a frequency-dependent input antenna)
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, *, freq_new, azimuth_grid, \
    elevation_grid )

# Acoustic array (speed of sound)
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, propagation_speed=343.0 )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as documented in [generate](#generate); 4D pattern fields
  require `center_freq` with one value per frequency entry `(n_freq,)`
- **`freq_new`** — Alternative center frequency in Hz; scalar or 1D array; if given, the pattern is
  recomputed at each value; if `None`, each input entry's `center_freq` is used; a scalar `<= 0`
  is treated as not given; default: `None`
- **`azimuth_grid`** — Alternative output azimuth grid in rad; -pi to pi; strictly increasing; `(n_azimuth_out,)`;
  if `None`, the input grid is used; default: `None`
- **`elevation_grid`** — Alternative output elevation grid in rad; -pi/2 to pi/2; strictly increasing;
  `(n_elevation_out,)`; if `None`, the input grid is used; default: `None`
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for
  acoustics; default: `299792458.0`

### Output:
- **`arrayant_out`** — Dict with the combined array antenna data; keys as in [generate](#generate)

### See also:
- [generate](#generate) (for field layout in the arrayant struct)
- [rotate_pattern](#rotate_pattern) (for changing the orientation of elements before combining)
- [calc_beamwidth](#calc_beamwidth) (calculates the beam width of array antennas)
- [calc_directivity](#calc_directivity) (directivity in dBi of array antenna elements)

---
<a id="concat"></a>
## concat
Concatenate two array antennas into a single one

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_concat) · [MATLAB](../mex/array-antenna-functions.md#arrayant_concat)

- Appends all elements of `arrayant2` onto `arrayant1` along the element dimension; the `element_pos`
  matrices are joined horizontally
- Both inputs must share identical azimuth and elevation sampling grids
- Coupling is assembled block-diagonally: `arrayant1`'s elements connect only to its own ports,
  `arrayant2`'s elements only to its own ports
- `center_freq` and `name` are taken from `arrayant1`
- Frequency-dependent input (4D pattern fields): both inputs must have the same number of
  frequency entries with matching `center_freq` at each index; concatenation is done per entry
- Output format matches the input: single-frequency dict for 3D input, frequency-dependent dict
  for 4D input

### Usage:
```python
arrayant_out = quadriga_lib.arrayant.concat( arrayant1, arrayant2 )
```

### Inputs:
- **`arrayant1`** — Dict with the first array antenna; keys as in [generate](#generate); pattern fields
  may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`arrayant2`** — Dict with the second array antenna; must use the same azimuth and elevation
  grids as `arrayant1`, and for 4D input the same number of frequency entries with matching `center_freq`

### Output:
- **`arrayant_out`** — Dict with the concatenated array antenna; same keys and layout as the
  inputs; single-frequency dict for 3D input, frequency-dependent dict for 4D input

---
<a id="copy_element"></a>
## copy_element
Create copies of array antenna elements

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_copy_element) · [MATLAB](../mex/array-antenna-functions.md#arrayant_copy_element)

- Copies a source element to one or more destination slots within an arrayant
- The antenna is resized when a destination index exceeds the current number of elements
- Coupling-matrix entries for newly added elements are set to identity; existing coupling is kept
- Works on single-frequency dicts (3D pattern fields) and frequency-dependent dicts (4D pattern
  fields); the same copy is applied to every frequency and a matching dict is returned
- If `i_source` has one entry, that element is copied to every index in `i_target`
- If `i_source` has several entries, `i_target` must have the same length and copies
  are performed pairwise as `i_source[i]` to `i_target[i]`

### Usage:
```python
# Copy element 0 to position 1
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, 1 )

# Copy element 0 to several positions
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, [2, 3] )

# Pairwise copy — i_source and i_target must have equal length
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, [0, 1], [2, 3] )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`i_source`** — Index of the source element(s); scalar int or 1D list or array of int; `(n_copy,)`
- **`i_target`** — Index of the destination element(s); scalar int or 1D list or array of int; `(n_copy,)`;
  must have the same length as `i_source` unless `i_source` is a single index

### Output:
- **`arrayant_out`** — Dict with the modified array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input

---
<a id="export_obj_file"></a>
## export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_export_obj_file) · [MATLAB](../mex/array-antenna-functions.md#arrayant_export_obj_file)

- The pattern is mapped onto an icosphere; a higher `icosphere_n_div` gives a finer mesh
- The OBJ file is written to `fn`; the function returns nothing
- Accepts a frequency-dependent antenna (4D pattern fields); `i_freq` selects which frequency
  entry is exported

### Usage:
```python
quadriga_lib.arrayant.export_obj_file( fn, arrayant, *, directivity_range_dB, colormap, object_radius, \
    icosphere_n_div, i_element, i_freq )
```

### Inputs:
- **`fn`** — Output OBJ filename; must not be empty and must end in `.obj`
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`directivity_range_dB`** — Dynamic range of the visualized directivity pattern in dB; default: 30.0
- **`colormap`** — Colormap name; default: jet; Available: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
- **`object_radius`** — Radius of the exported object in meters; default: 1.0
- **`icosphere_n_div`** — Icosphere subdivision count; higher gives a finer mesh; see [icosphere](site-specific-simulation-tools.md#icosphere); default: 4
- **`i_element`** — Element indices to export; 1D list or array of int; `None` or empty exports all elements; default: `None`
- **`i_freq`** — Frequency index to export from a frequency-dependent antenna; must satisfy `0 <= i_freq < n_freq`; default: 0

---
<a id="generate"></a>
## generate
Generate an array antenna of a given type

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_generate) · [MATLAB](../mex/array-antenna-functions.md#arrayant_generate)

- Tuples (`element`, `panel`, `spacing`, `pol`, `ports`, `steer`): entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their default, e.g. `panel=[4, 8]` is one panel of 4 rows and 8 columns.
- Panel arrays: elements are ordered polarization fastest, then row `m` (upwards along z), column `n` (along +y), panel row `mg`, panel column `ng`, centred on the origin; ports in the same order with `m_p`, `n_p`. The coupling matrix `[n_elements, n_ports]` holds the weights; `apply_weights` combines each port into one element at its sub-array centre.
- Sub-array weights (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds `K = M / M_P` × `L = N / N_P` elements; element `(k, l)` gets `exp(j 2π ((k - (K-1)/2) d_V sin(tilt) - (l - (L-1)/2) d_H sin(pan))) / sqrt(K L)`; a positive tilt points the beam down, a positive pan towards +y.
- Polarization (TR 38.901 clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45° (element 0 first). Model-1 turns the element by ζ about its boresight (eq. 7.3-3); Model-2 sets F_θ = sqrt(A') cos ζ and F_φ = sqrt(A') sin ζ with A' = |F_θ|² + |F_φ|² of the element (eq. 7.3-4, 7.3-5).
- `multibeam`: maximum-ratio-transmission weights for the directions in `beams` (exact for one beam, approximate for several; only the ratios of the beam weights matter); one port per polarization, with `separate_beams` one per polarization and beam.
- An input that the type does not take raises `ValueError`.

### Usage:
```python
# omni, xpol, dipole, half-wave-dipole
ant = quadriga_lib.arrayant.generate( type, res, *, center_freq )

# Custom-shape element
ant = quadriga_lib.arrayant.generate( 'custom', res, *, center_freq, element )

# Uniform linear array (omni element) and TR 38.901 panel array
ant = quadriga_lib.arrayant.generate( 'ula', res, *, center_freq, panel, spacing, pol, ports, steer, \
    pattern, apply_weights )
ant = quadriga_lib.arrayant.generate( '3GPP', res, *, center_freq, element, panel, spacing, pol, ports, \
    steer, pattern, apply_weights )

# Multi-beam panel array
ant = quadriga_lib.arrayant.generate( 'multibeam', res, *, center_freq, element, panel, spacing, pol, \
    beams, separate_beams, apply_weights )
```

### Inputs:
- **`type`** — Antenna type; str:
   | `type`                       | Antenna                                                                    | Inputs besides `res`, `center_freq`                                                |
   | ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
   | `omni`                       | Isotropic element, vertical polarization                                   | none                                                                               |
   | `xpol`                       | Two isotropic elements, vertical and horizontal polarization               | none                                                                               |
   | `dipole` (or `short-dipole`) | Short dipole, vertical polarization                                        | none                                                                               |
   | `half-wave-dipole`           | Half-wave dipole, vertical polarization                                    | none                                                                               |
   | `custom`                     | One element with given 3 dB beamwidths, peak gain equal to its directivity | `element`                                                                          |
   | `ula`                        | Uniform linear array of omni elements                                      | `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)           | Panel array of 3GPP TR 38.901 clause 7.3                                   | `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights` |
   | `multibeam`                  | Panel array with beams towards given directions                            | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
- **`res`** — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: default; default: 10 for `omni` and `xpol`, 1 otherwise
- **`center_freq`** — Center frequency in Hz; positive and finite; NaN: default; default: 299792458 (wavelength 1 m)
- **`element`** — Element parameters per type; default: `None` (the defaults below):
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi; default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity
- **`panel`** — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; int or `(n,)` of int; at most 1e7 elements per polarization; default: `None` (1; `ula`: `panel=8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing (centre to centre); >= 0; default: `None` (`[0.5, d_V, M d_V, N d_H]`)
- **`pol`** — `[polarization, model]`: polarization 1 (vertical), 2 (H/V) or 3 (±45°); model 1 or 2 (TR 38.901 Model-1, Model-2); int or `(n,)` of int; default: `None` (`[1, 1]`)
- **`ports`** — `[M_P, N_P]`: ports per column and per row of a panel and polarization; int or `(n,)` of int; must divide `M` and `N`; default: `None` (`[M, N]`, one port per element)
- **`steer`** — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt (positive down) and pan (positive towards +y); `(2,)` (a 1D input is one row) for all panels or `(Mg Ng, 2)` with one row per panel; one column: pan 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`; default: `None` (0)
- **`beams`** — `(2, n_beams)` or `(3, n_beams)`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); a 1D input is one beam; default: `None` (`[[0], [0], [1]]`)
- **`pattern`** — Element dict (keys as the output) to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; default: `None` (the element of the type)
- **`separate_beams`** — If `True`, one port per beam (weights ignored); default: `False`
- **`apply_weights`** — If `True`, combine each port into one pattern at its sub-array centre; default: `False`

### Outputs:
- **`ant`** — Dict with keys:
   | Key              | Description                                               | Shape                                  |
   | ---------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part                        | `(n_elevation, n_azimuth, n_elements)` |
   | `e_theta_im`     | e-theta field component, imaginary part                   | `(n_elevation, n_azimuth, n_elements)` |
   | `e_phi_re`       | e-phi field component, real part                          | `(n_elevation, n_azimuth, n_elements)` |
   | `e_phi_im`       | e-phi field component, imaginary part                     | `(n_elevation, n_azimuth, n_elements)` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, strictly increasing       | `(n_azimuth,)`                         |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, strictly increasing | `(n_elevation,)`                       |
   | `element_pos`    | Element (x,y,z) positions in meters                       | `(3, n_elements)`                      |
   | `coupling_re`    | Coupling matrix, real part                                | `(n_elements, n_ports)`                |
   | `coupling_im`    | Coupling matrix, imaginary part                           | `(n_elements, n_ports)`                |
   | `center_freq`    | Center frequency in Hz; inputs accept `(1,)`              | scalar                                 |
   | `name`           | Name of the array antenna object                          | str                                    |

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [C++](../cpp/array-antenna-functions.md#generate_speaker) · [MATLAB](../mex/array-antenna-functions.md#generate_speaker)

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency; for a
  single frequency the dict is a single-frequency arrayant dict (3D pattern fields, scalar `center_freq`)
- Directivity is stored as real-valued data in `e_theta_re`; dipole rear hemisphere is encoded with negative sign for 180° phase inversion
- Multi-driver systems (e.g. two-way) are built by calling this per driver and combining via [concat](#concat);
  crossover behavior emerges from overlapping bandpass responses
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity_dB - 85) / 20)`
- If `center_freq` is None or empty, third-octave band centers are auto-generated from one band below
  `lower_cutoff` to one band above `upper_cutoff`, clipped to 20-20000 Hz
- Speed of sound assumed to be 343 m/s
- Driver models (`driver_type`):
  - `piston` - circular piston in baffle, `D(theta) = 2·J1(ka·sin theta)/(ka·sin theta)`, rotationally symmetric, narrows with increasing `ka`
  - `horn` - separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`
  - `omni` - frequency-independent omnidirectional pattern
- Enclosure models (`radiation_type`):
  - `monopole` - no modification
  - `hemisphere` - sealed box with baffle-step transition, `f_baffle = c/(pi·sqrt(W·H))`
  - `dipole` - figure-8, `R = abs(cos(theta_off))` with sign inversion in rear hemisphere
  - `cardioid` - `R = 0.5·(1+cos(theta_off))`
- For `horn`, `hor_coverage_deg`, `ver_coverage_deg` and `horn_control_freq` set to NaN (the default) or <= 0 select 90°, 60°
  and `f_ctrl = c/(2pi·radius)`

### Usage:
```python
speaker = quadriga_lib.arrayant.generate_speaker( driver_type, *, radius, lower_cutoff, upper_cutoff, \
    lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB, radiation_type, hor_coverage_deg, \
    ver_coverage_deg, horn_control_freq, baffle_width, baffle_height, center_freq, angular_resolution_deg )
```

### Inputs:
- **`driver_type`** — Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** — Effective radiating radius in m, > 0; cone/dome radius for piston, mouth radius for horn;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB bandpass frequency in Hz, > 0; default: 80
- **`upper_cutoff`** — Upper -3 dB bandpass frequency in Hz, larger than `lower_cutoff`;
  default: 12000
- **`lower_rolloff_slope_dB`** — Low-frequency rolloff in dB/octave, > 0 (12 dB/oct = 2nd-order Butterworth);
  default: 12
- **`upper_rolloff_slope_dB`** — High-frequency rolloff in dB/octave, > 0; default: 12
- **`sensitivity_dB`** — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`;
  default: `hemisphere`
- **`hor_coverage_deg`** — Horn horizontal coverage angle in degrees; NaN or <= 0: default 90; horn only;
  default: NaN
- **`ver_coverage_deg`** — Horn vertical coverage angle in degrees; NaN or <= 0: default 60; horn only;
  default: NaN
- **`horn_control_freq`** — Horn pattern control frequency in Hz; NaN or <= 0: default `c/(2pi·radius)`;
  horn only; default: NaN
- **`baffle_width`** — Baffle width in m, > 0; used by `hemisphere` model; default: 0.15
- **`baffle_height`** — Baffle height in m, > 0; used by `hemisphere` model; default: 0.25
- **`center_freq`** — Frequency sample points in Hz; auto-generated third-octave bands if None;
  `(n_freq,)` or None; default: None
- **`angular_resolution_deg`** — Azimuth and elevation grid resolution in degrees, 0.1 to 90; default: 5

### Output:
- **`speaker`** — Multi-frequency arrayant dict with the fields below; for `n_freq = 1` the pattern fields
  are 3D `(n_elevation, n_azimuth, n_elements)` and `center_freq` is a float:
  - `e_theta_re` — e-theta field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_theta_im` — e-theta field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_re` — e-phi field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_im` — e-phi field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `azimuth_grid` — Azimuth angles in rad, -π to π, strictly increasing; `(n_azimuth,)`
  - `elevation_grid` — Elevation angles in rad, -π/2 to π/2, strictly increasing; `(n_elevation,)`
  - `element_pos` — Element (x,y,z) positions in meters; `(3, n_elements)`
  - `coupling_re` — Coupling matrix, real part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `coupling_im` — Coupling matrix, imaginary part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `center_freq` — Frequency samples in Hz; `(n_freq,)`
  - `name` — Name of the array antenna object; str

---
<a id="interpolate"></a>
## interpolate
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_interpolate) · [MATLAB](../mex/array-antenna-functions.md#arrayant_interpolate)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Single-frequency mode (3D pattern arrays and no `freq_new`): outputs `(n_out, n_ang)`; `dist` and `local_angles` add
  outputs, returned only on request since computing them costs extra time and memory. Multi-frequency mode (4D pattern
  arrays, or `freq_new` given): for each target frequency, the two entries whose center frequencies bracket it are
  interpolated and blended; frequencies outside the entries take the nearest entry; outputs `(n_out, n_ang, n_freq)`;
  a 3D pattern is the same at every frequency
- `azimuth` of shape `(1, 0)` or `(n_out, 0)` gives empty outputs; `np.array([])` raises `ValueError`
- NaN or infinite angles give NaN in all outputs of that angle and output, without an error
- On a pole of the local frame (a direction along the z-axis of the oriented antenna) the local azimuth is chosen so
  that the local and the global theta unit vectors agree: the polarization is not rotated there
- Kernels: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the peak
  field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole of the local frame up to about
  2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most 4e-4 rad) in the local
  elevation; auto selects GENERIC in this version, which may change in a later version

### Usage:
```python
# Single-frequency mode
V, H = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, *, i_element, orientation, \
    element_pos, use_kernel )

# Single-frequency mode with the projected distance and the local angles
V, H, dist, azimuth_loc, elevation_loc, gamma = quadriga_lib.arrayant.interpolate( arrayant, azimuth, \
    elevation, ..., dist=True, local_angles=True )

# Multi-frequency mode
V, H = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, ..., freq_new=freqs )
```

### Inputs:
- **`arrayant`** — Dict containing the arrayant data; layout as documented in [generate](#generate); 4D pattern arrays contain
  a frequency-dependent model
- **`azimuth`** — Azimuth angles in GCS; `(1, n_ang)` (same angles for all outputs) or `(n_out, n_ang)` (one angle
  per output)
- **`elevation`** — Elevation angles in GCS; same shape as `azimuth`
- **`i_element`** — Element indices, 0-based, duplicates allowed; `(n_out,)`; default: `None` (all elements)
- **`orientation`** — Euler angles (bank, tilt, heading) of the antenna; `(3, 1)`, `(3, n_out)`, `(3, 1, n_ang)` or
  `(3, n_out, n_ang)`; default: `None` (no rotation)
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; `(3, n_out)`; default: `None`
  (the positions of the arrayant)
- **`freq_new`** — Target frequencies in Hz; selects the multi-frequency mode; `(n_freq,)`; default: `None` (the
  `center_freq` of the entries)
- **`dist`** — Also return the projected distance; single-frequency mode only; default: `False`
- **`local_angles`** — Also return the local angles and the polarization rotation angle; single-frequency mode only;
  default: `False`
- **`fast_access`** — Require the pattern arrays to be float64 and Fortran-contiguous, so that they are read without
  a copy; raises otherwise. With `False`, such arrays are read without a copy as well and others are copied;
  default: `False`
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (raises if unavailable); `int`; default: 0

### Outputs:
- **`V`** — e-theta (vertical) field, complex; `(n_out, n_ang)`; multi-frequency mode: `(n_out, n_ang, n_freq)`
- **`H`** — e-phi (horizontal) field, complex; same shape as `V`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the antenna; negative for
  elements closer to the source; `(n_out, n_ang)`; returned when `dist` is `True`
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `(n_out, n_ang)`;
  returned when `local_angles` is `True`
- **`gamma`** — Polarization rotation angle from the global to the local frame; `(n_out, n_ang)`; returned when
  `local_angles` is `True`

### See also:
- [generate](#generate) (arrayant dict layout)
- [qdant_read](#qdant_read) (load arrayant data)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayants)

---
<a id="qdant_read"></a>
## qdant_read
Reads array antenna data from QDANT files

Also in: [C++](../cpp/array-antenna-functions.md#qdant_read) · [MATLAB](../mex/array-antenna-functions.md#qdant_read)

- QDANT is the QuaDRiGa array antenna exchange format, an XML format for storing antenna patterns
- A QDANT file may hold one or several entries (e.g. a frequency-dependent antenna model)
- `id` reads a single entry by its 1-based ID; the result is a dict with 3D pattern fields
- Without `id` (or with `id = None`), every entry is read: a frequency-dependent dict with 4D pattern fields
  when the file holds multiple entries, or a plain single-entry dict (3D fields) when it holds exactly one;
  multiple entries must form a frequency set (same number of elements, same azimuth and elevation grids, element
  positions and coupling size), otherwise an error asks to pass `id`
- Reading all entries is the inverse of `qdant_write` with a 4D-pattern dict
- An `id` that is not in the file raises an error; the first entry counts as ID 1 only if it has no
  `id` attribute
- Malformed files raise an error, also when they declare more elements (`NoElements`) than the file
  has data for; elements without pattern data read as zeros (at most 2^24 zero values per entry)

### Usage:
```python
# Read every entry (a frequency-dependent arrayant when the file holds several)
ant, layout = quadriga_lib.arrayant.qdant_read( fn )

# Read a single entry
ant, layout = quadriga_lib.arrayant.qdant_read( fn, id )
```

### Inputs:
- **`fn`** — path to the QDANT file; must not be empty
- **`id`** — 1-based ID of the entry to read, >= 1; `None` reads all entries (see above); default: None

### Outputs:
- **`ant`** — Dict with the array antenna data; keys as in [generate](#generate), with these specifics:
  - `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` — Pattern fields;
    `(n_elevation, n_azimuth, n_elements)` for a single entry, or
    `(n_elevation, n_azimuth, n_elements, n_freq)` when multiple entries are read
  - `center_freq` — Center frequency in Hz; scalar for a single entry, `(n_freq,)` when reading multiple entries
  - `coupling_re`, `coupling_im` — Coupling matrices; `(n_elements, n_ports)`, or
    `(n_elements, n_ports, n_freq)` when reading multiple entries with per-entry coupling
- **`layout`** — Matrix of element IDs describing how the entries are arranged in the file; int64

### See also:
- [qdant_write](#qdant_write) (for writing QDANT data)
- [generate](#generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format ([QDANT](../../formats/qdant.md))

---
<a id="qdant_write"></a>
## qdant_write
Writes array antenna data to QDANT files

Also in: [C++](../cpp/array-antenna-functions.md#qdant_write) · [MATLAB](../mex/array-antenna-functions.md#qdant_write)

- QDANT is the QuaDRiGa array antenna exchange format, an XML format for storing antenna patterns
- Single-frequency input (3D pattern fields) writes one entry; `id` places it in the file, and
  several antennas can share one file via distinct IDs
- Frequency-dependent input (4D pattern fields) writes every frequency entry with sequential
  1-based IDs; `id` and `layout` do not apply, and the file must not already exist
- Returns the ID assigned to the written entry (`0` for a frequency-dependent write)
- Without `layout`, the layout in the file is replaced by a row of all IDs in the file in file order;
  a custom layout written before is lost (pass it again with every write that should keep it)
- The file has single-precision-equivalent accuracy relative to the element peak: pattern
  magnitudes (dB) and phases (degree) are rounded so that every sample is within 6e-8 of the peak
  amplitude of its element; samples far below the peak are less accurate relative to their own
  amplitude (about 6e-4 at -80 dB, 6e-2 at -120 dB below the peak); positions and coupling are
  written as float values (at most 9 significant digits); sampling grids are rounded to 0.001
  degree; all non-zero pattern samples are written at any level, exact zeros as `-inf` dB

### Usage:
```python
# Single-frequency write
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant )
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant, id, *, layout )

# Frequency-dependent write (4D patterns) — all entries written sequentially
quadriga_lib.arrayant.qdant_write( fn, arrayant )
```

### Inputs:
- **`fn`** — Output QDANT filename; must not be empty
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`; 4D input
  requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`id`** — Target 1-based ID of the entry inside the file; `0` appends after the highest
  existing ID (or 1 if the file does not exist); ignored for 4D input; default: 0
- **`layout`** — uint32 matrix organizing multiple antenna IDs within the file; must reference
  only IDs present in the file; ignored for 4D input; default: `None` (a row of all IDs in the file,
  which replaces a layout written before)

### Output:
- **`id_in_file`** — ID assigned to the entry in the file after writing; `0` for a frequency-dependent (4D) write

### See also:
- [qdant_read](#qdant_read) (for reading QDANT data)
- [generate](#generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format ([QDANT](../../formats/qdant.md))

---
<a id="rotate_pattern"></a>
## rotate_pattern
Rotate antenna radiation patterns using Euler rotations

Also in: [C++](../cpp/array-antenna-functions.md#arrayant_rotate_pattern) · [MATLAB](../mex/array-antenna-functions.md#arrayant_rotate_pattern)

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [interpolate](#interpolate) returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); coupling, `center_freq` and `name` are kept
- Single-frequency input (3D pattern fields): `usage` 0 and 1 adjust the sampling grid for non-uniformly sampled
  antennas; this needs all elements, so a subset in `i_element` raises an error when the grid must be adjusted
- Frequency-dependent input (4D pattern fields): the rotation is applied to every entry and the grid is never adjusted,
  since all entries must share one grid
- For scalar acoustic fields (pressure stored in `e_theta_re` only) use `usage = 1` to avoid spurious polarization effects

### Usage:
```python
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, orientation, *, usage, i_element )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad; 3 elements
- **`usage`** — Rotation mode; default: 0
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

   for 4D input the grid is never adjusted, so `0`/`3` and `1`/`4` are equivalent
- **`i_element`** — Element indices to rotate, 0-based, no duplicates; `None` or empty rotates all elements;
  default: `None`

### Output:
- **`arrayant_out`** — Dict with the rotated array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input
