---
title: "Python API Documentation for Quadriga-Lib v0.12.4"
author: "Stephan Jaeckel"
date: "27.10.2026"
lang: en-US
---

<!-- PLACEHOLDER: Python API preamble -->
<!-- Edit this section to add pip install instructions, import conventions, and general usage notes for the Python API. -->

# Function Index

| Function | Section | Line |
| --- | --- | --- |
| [calc_beamwidth](#calc_beamwidth) | Array antenna functions | 98 |
| [calc_directivity](#calc_directivity) | Array antenna functions | 140 |
| [combine_pattern](#combine_pattern) | Array antenna functions | 170 |
| [concat](#concat) | Array antenna functions | 219 |
| [copy_element](#copy_element) | Array antenna functions | 249 |
| [export_obj_file](#export_obj_file) | Array antenna functions | 285 |
| [generate](#generate) | Array antenna functions | 310 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 407 |
| [interpolate](#interpolate) | Array antenna functions | 493 |
| [qdant_read](#qdant_read) | Array antenna functions | 573 |
| [qdant_write](#qdant_write) | Array antenna functions | 620 |
| [rotate_pattern](#rotate_pattern) | Array antenna functions | 668 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 719 |
| [channel_export_obj_file](#channel_export_obj_file) | Channel functions | 791 |
| [hdf5_create_file](#hdf5_create_file) | Channel functions | 837 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 866 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 925 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 958 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 985 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 1010 |
| [hdf5_write_channel](#hdf5_write_channel) | Channel functions | 1031 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1075 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 1110 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 1143 |
| [quantize_delays](#quantize_delays) | Channel functions | 1204 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 1280 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 1362 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 1438 |
| [get_ieee_indoor](#get_ieee_indoor) | Channel generation functions | 1539 |
| [acdf](#acdf) | Channel statistics | 1628 |
| [calc_angular_spread](#calc_angular_spread) | Channel statistics | 1662 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 1702 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 1764 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 1794 |
| [conf_file_list](#conf_file_list) | Geometry-based stochastic model | 1860 |
| [conf_file_read](#conf_file_read) | Geometry-based stochastic model | 1900 |
| [conf_file_write](#conf_file_write) | Geometry-based stochastic model | 1964 |
| [los_probability](#los_probability) | Geometry-based stochastic model | 2036 |
| [lsf_draw](#lsf_draw) | Geometry-based stochastic model | 2131 |
| [lsf_init_phases](#lsf_init_phases) | Geometry-based stochastic model | 2223 |
| [lsp_generate](#lsp_generate) | Geometry-based stochastic model | 2316 |
| [o2i_loss](#o2i_loss) | Geometry-based stochastic model | 2439 |
| [path_loss](#path_loss) | Geometry-based stochastic model | 2500 |
| [scenpar_validate](#scenpar_validate) | Geometry-based stochastic model | 2615 |
| [cart2geo](#cart2geo) | Math functions | 2678 |
| [geo2cart](#geo2cart) | Math functions | 2710 |
| [components](#components) | Miscellaneous / Tools | 2748 |
| [version](#version) | Miscellaneous / Tools | 2757 |
| [write_png](#write_png) | Miscellaneous / Tools | 2770 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 2818 |
| [cube](#cube) | Site-specific simulation tools | 2893 |
| [icosphere](#icosphere) | Site-specific simulation tools | 2921 |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Site-specific simulation tools | 2953 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 2988 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 3043 |
| [plane](#plane) | Site-specific simulation tools | 3101 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 3131 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 3158 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 3189 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 3222 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 3268 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 3316 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 3342 |
| [sos.acf_2d](#sosacf_2d) | Sum-of-sinusoids random number generator | 3518 |
| [sos.acf_approx](#sosacf_approx) | Sum-of-sinusoids random number generator | 3548 |
| [sos.acf_estimate](#sosacf_estimate) | Sum-of-sinusoids random number generator | 3583 |
| [sos.acfi](#sosacfi) | Sum-of-sinusoids random number generator | 3638 |
| [sos.calc_mse](#soscalc_mse) | Sum-of-sinusoids random number generator | 3662 |
| [sos.generate](#sosgenerate) | Sum-of-sinusoids random number generator | 3712 |
| [sos.init_phases](#sosinit_phases) | Sum-of-sinusoids random number generator | 3779 |
| [sos.map](#sosmap) | Sum-of-sinusoids random number generator | 3813 |
| [sos.preset](#sospreset) | Sum-of-sinusoids random number generator | 3850 |
| [sos.rand](#sosrand) | Sum-of-sinusoids random number generator | 3889 |
| [sos.randi](#sosrandi) | Sum-of-sinusoids random number generator | 3948 |
| [sos.randn](#sosrandn) | Sum-of-sinusoids random number generator | 4009 |
| [sos.remove_bias](#sosremove_bias) | Sum-of-sinusoids random number generator | 4067 |
| [sos.val](#sosval) | Sum-of-sinusoids random number generator | 4146 |

---

# Array antenna functions

---
## calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

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
```
bw_az, bw_el, az_pt, el_pt = quadriga_lib.arrayant.calc_beamwidth( arrayant )
bw_az, bw_el, az_pt, el_pt = quadriga_lib.arrayant.calc_beamwidth( arrayant, element, threshold_dB )
```

### Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [generate](#generate) for the field layout
- **`element`** — Element indices; if None or empty, all elements are used; `(n_out,)` or None; default: None
- **`threshold_dB`** — Threshold in dB (3 = FWHM); default: 3.0

### Outputs:
- **`bw_az`** — Azimuth beamwidth in degrees; `(n_out,)` for single-frequency input,
  `(n_out, n_freq)` for multi-frequency input; `n_out = n_elements` when `element` is None
- **`bw_el`** — Elevation beamwidth in degrees; same shape as `bw_az`
- **`az_pt`** — Azimuth pointing angle of the main beam in degrees; same shape as `bw_az`
- **`el_pt`** — Elevation pointing angle of the main beam in degrees; same shape as `bw_az`

### See also:
- [combine_pattern](#combine_pattern) (apply element coupling before calculating beamwidth)
- [calc_directivity](#calc_directivity) (directivity in dBi of array antenna elements)

---
## calc_directivity
Calculates the directivity in dBi of array antenna elements

- Directivity = 10·log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

### Usage:
```
directivity = quadriga_lib.arrayant.calc_directivity( arrayant )
directivity = quadriga_lib.arrayant.calc_directivity( arrayant, element )
```

### Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [generate](#generate) for the field layout
- **`element`** — Element indices; if None or empty, all elements are used; `(n_out,)` or None; default: None

### Outputs:
- **`directivity`** — Directivity in dBi; `(n_out,)` for a single-frequency input,
  `(n_out, n_freq)` for a multi-frequency input; `n_out = n_elements` when `element` is None

### See also:
- [combine_pattern](#combine_pattern) (apply element coupling before calculating directivity)
- [calc_beamwidth](#calc_beamwidth) (calculates the beam width of array antennas)

---
## combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- Accepts a frequency-dependent antenna (4D pattern fields); the pattern is then combined per frequency
- `freq` may recompute (interpolate) the combined pattern at one or more requested frequencies
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Usage:
```
# Single frequency
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant )
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, freq, azimuth_grid, elevation_grid )

# Multiple frequencies (freq as a 1D array, or a frequency-dependent input antenna)
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, freq, azimuth_grid, elevation_grid )

# Acoustic array (speed of sound)
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, propagation_speed=343.0 )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as documented in [generate](#generate); 4D pattern fields
  require `center_freq` with one value per frequency entry `(n_freq,)`
- **`freq`** — Alternative center frequency in Hz; scalar or 1D array; if given, the pattern is
  recomputed at each value; if `None`, each input entry's `center_freq` is used; a scalar `<= 0`
  is treated as not given; default: `None`
- **`azimuth_grid`** — Alternative output azimuth grid in rad; -pi to pi; sorted; `(n_azimuth_out,)`;
  if `None`, the input grid is used; default: `None`
- **`elevation_grid`** — Alternative output elevation grid in rad; -pi/2 to pi/2; sorted;
  `(n_elevation_out,)`; if `None`, the input grid is used; default: `None`
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for
  acoustics; default: `299792458.0`

### Outputs:
- **`arrayant_out`** — Dict with the combined array antenna data; keys as in [generate](#generate)

### See also:
- [generate](#generate) (for field layout in the arrayant struct)
- [rotate_pattern](#rotate_pattern) (for changing the orientation of elements before combining)
- [calc_beamwidth](#calc_beamwidth) (calculates the beam width of array antennas)
- [calc_directivity](#calc_directivity) (directivity in dBi of array antenna elements)

---
## concat
Concatenate two array antennas into a single one

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
```
arrayant_out = quadriga_lib.arrayant.concat( arrayant1, arrayant2 )
```

### Inputs:
- **`arrayant1`** — Dict with the first array antenna; keys as in [generate](#generate); pattern fields
  may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`arrayant2`** — Dict with the second array antenna; must use the same azimuth and elevation
  grids as `arrayant1`, and for 4D input the same number of frequency entries with matching `center_freq`

### Outputs:
- **`arrayant_out`** — Dict with the concatenated array antenna; same keys and layout as the
  inputs; single-frequency dict for 3D input, frequency-dependent dict for 4D input

---
## copy_element
Create copies of array antenna elements

- Copies a source element to one or more destination slots within an arrayant
- The antenna is resized when a destination index exceeds the current number of elements
- Coupling-matrix entries for newly added elements are set to identity; existing coupling is kept
- Works on single-frequency dicts (3D pattern fields) and frequency-dependent dicts (4D pattern
  fields); the same copy is applied to every frequency and a matching dict is returned
- If `source_element` has one entry, that element is copied to every index in `dest_element`
- If `source_element` has several entries, `dest_element` must have the same length and copies
  are performed pairwise as `source_element[i]` to `dest_element[i]`

### Usage:
```
# Copy element 0 to position 1
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, 1 )

# Copy element 0 to several positions
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, [2, 3] )

# Pairwise copy — source_element and dest_element must have equal length
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, [0, 1], [2, 3] )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`source_element`** — Index of the source element(s); scalar int or 1D list or array of int; `(n_copy,)`
- **`dest_element`** — Index of the destination element(s); scalar int  1D list or array of int; `(n_copy,)`;
  must have the same length as `source_element` unless `source_element` is a single index

### Outputs:
- **`arrayant_out`** — Dict with the modified array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input

---
## export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- The pattern is mapped onto an icosphere; a higher `icosphere_n_div` gives a finer mesh
- The OBJ file is written to `fn`; the function returns nothing
- Accepts a frequency-dependent antenna (4D pattern fields); `freq_ind` selects which frequency
  entry is exported

### Usage:
```
quadriga_lib.arrayant.export_obj_file( fn, arrayant, directivity_range, colormap, object_radius, \
    icosphere_n_div, element, freq_ind )
```

### Inputs:
- **`fn`** — Output OBJ filename; str; must not be empty and must end in `.obj`
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`directivity_range`** — Dynamic range of the visualized directivity pattern in dB; default: 30.0
- **`colormap`** — Colormap name; default: jet; Available: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
- **`object_radius`** — Radius of the exported object in meters; default: 1.0
- **`icosphere_n_div`** — Icosphere subdivision count; higher gives a finer mesh; see [icosphere](#icosphere); default: 4
- **`element`** — Element indices to export; 1D list or array of int; `None` or empty exports all elements; default: `None`
- **`freq_ind`** — Frequency index to export from a frequency-dependent antenna; must satisfy `0 <= freq_ind < n_freq`; default: 0

---
## generate
Generates predefined array antenna models

- Dispatches to one of several C++ generator functions based on the `type` string
- Supported types: `omni`, `dipole` (or `short-dipole`), `half-wave-dipole`, `xpol`, `custom`,
  `ula`, `3gpp` (or `3GPP`), `multibeam`
- All arguments after `type` are keyword-only and type-specific; defaults are filled where omitted
- For `3gpp` and `ula`, a `pattern` dict can override the default per-element pattern
- `multibeam` combines beams via MRT weighting; set `separate_beams=True` for one beam per direction

### Usage:
```
# Simple antennas (v-pol)
ant = quadriga_lib.arrayant.generate('omni')
ant = quadriga_lib.arrayant.generate('dipole')
ant = quadriga_lib.arrayant.generate('half-wave-dipole')

# Cross-polarized isotropic
ant = quadriga_lib.arrayant.generate('xpol')

# Custom 3dB beamwidth
ant = quadriga_lib.arrayant.generate('custom', az_3dB=90.0, el_3dB=90.0, rear_gain_lin=0.0)

# Uniform linear array
ant = quadriga_lib.arrayant.generate('ula', N=4, freq=2.4e9, spacing=0.7, pol=1)

# 3GPP-NR array (default 3GPP element pattern)
ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9, pol=1, spacing=0.7)

# 3GPP-NR array with custom per-element pattern dict
ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9, pol=1, pattern=my_pattern)

# Multi-beam M×N array
ant = quadriga_lib.arrayant.generate('multibeam', M=6, N=6, freq=3.7e9, pol=1, \
    beam_az=[-30.0, 30.0], beam_el=[0.0, 0.0], separate_beams=False)
```

### Inputs (common):
- **`type`** — Antenna model type; str
- **`res`** — Pattern sampling grid resolution in degrees; default: 10 for `omni` and `xpol`, 1 otherwise
- **`freq`** — Center frequency in Hz; default: 299792458 (λ = 1 m)

### Inputs (custom, 3gpp, multibeam):
- **`az_3dB`** — Azimuth 3dB beamwidth in degrees; default: 90 for `custom`, 67 for `3gpp`, 120 for `multibeam`
- **`el_3dB`** — Elevation 3dB beamwidth in degrees; same defaults as `az_3dB`
- **`rear_gain_lin`** — Back-to-front gain ratio (linear scale); default: 0.0

### Inputs (ula, 3gpp, multibeam):
- **`M`** — Number of vertical elements per panel; default: 1; ignored for `ula`
- **`N`** — Number of horizontal elements per panel; default: 1
- **`pol`** — Polarization mode (1..6); default: 1
   | `pol` | Description                                              |
   | :---: | -------------------------------------------------------- |
   | 1     | Vertical polarization                                    |
   | 2     | H/V polarization (2NM elements)                          |
   | 3     | ±45° polarization (2NM elements)                         |
   | 4     | Vertical, vertical elements combined (N elements)        |
   | 5     | H/V, vertical elements combined (2N elements)            |
   | 6     | ±45°, vertical elements combined (2N elements)           |
- **`spacing`** — Inter-element spacing in wavelengths; default: 0.5

### Inputs (3gpp only):
- **`tilt`** — Electrical downtilt in degrees; applies to `pol = 4/5/6`; default: 0.0
- **`Mg`** — Number of vertically stacked panels; default: 1
- **`Ng`** — Number of horizontally stacked panels; default: 1
- **`dgv`** — Panel spacing in vertical direction in wavelengths; default: 0.5
- **`dgh`** — Panel spacing in horizontal direction in wavelengths; default: 0.5

### Inputs (ula, 3gpp):
- **`pattern`** — Custom per-element pattern dict (same field layout as the output);
  overrides default 3GPP/ULA element pattern; only the radiation pattern is used: element positions,
  coupling and center frequency of the pattern are ignored; default: empty dict (use built-in pattern)

### Inputs (multibeam only):
- **`beam_az`** — Azimuth beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_el`** — Elevation beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_weight`** — Per-beam scaling factor; not normalized; same length as `beam_az`; default: `[1.0, 1.0, ...]` (all-ones)
- **`separate_beams`** — Produce one independent beam per angle pair (weights ignored); default: False
- **`apply_weights`** — Apply the beamforming weights to the output coupling matrix; default: False

### Outputs:
- **`ant`** — Dict with fields:
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
   | `center_freq`    | Center frequency in Hz; inputs accept `(1,)`    | scalar                                 |
   | `name`           | Name of the array antenna object                | string                                 |

---
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency; for a
  single frequency the dict is a single-frequency arrayant dict (3D pattern fields, scalar `center_freq`)
- Directivity is stored as real-valued data in `e_theta_re`; dipole rear hemisphere is encoded with negative sign for 180° phase inversion
- Multi-driver systems (e.g. two-way) are built by calling this per driver and combining via arrayant.[concat](#concat);
  crossover behavior emerges from overlapping bandpass responses
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity - 85) / 20)`
- If `frequencies` is None or empty, third-octave band centers are auto-generated from one band below
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
- For `horn`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2pi·radius)`

### Usage:
```
# Default piston driver (4-inch, 80 Hz – 12 kHz)
speaker = quadriga_lib.arrayant.generate_speaker()

# Horn tweeter with custom coverage
speaker = quadriga_lib.arrayant.generate_speaker(driver_type='horn', radius=0.025, \
    lower_cutoff=1500.0, upper_cutoff=20000.0, hor_coverage=90.0, ver_coverage=60.0)

# Omnidirectional subwoofer with steep rolloff
speaker = quadriga_lib.arrayant.generate_speaker(driver_type='omni', radius=0.165, \
    lower_cutoff=30.0, upper_cutoff=300.0, lower_rolloff_slope=24.0, upper_rolloff_slope=24.0, \
    sensitivity=90.0, radiation_type='monopole')

# Piston driver at specific frequencies
speaker = quadriga_lib.arrayant.generate_speaker( frequencies=np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0]), \
    angular_resolution=5.0)
```

### Inputs:
- **`driver_type`** — Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** — Effective radiating radius in m; cone/dome radius for piston, mouth radius for horn;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB bandpass frequency in Hz; default: 80
- **`upper_cutoff`** — Upper -3 dB bandpass frequency in Hz; default: 12000
- **`lower_rolloff_slope`** — Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth);
  default: 12
- **`upper_rolloff_slope`** — High-frequency rolloff in dB/octave; default: 12
- **`sensitivity`** — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`;
  default: `hemisphere`
- **`hor_coverage`** — Horn horizontal coverage angle in degrees; 0 auto-defaults to 90; horn only;
  default: 0
- **`ver_coverage`** — Horn vertical coverage angle in degrees; 0 auto-defaults to 60; horn only;
  default: 0
- **`horn_control_freq`** — Horn pattern control frequency in Hz; 0 auto-derives from `radius`;
  default: 0
- **`baffle_width`** — Baffle width in m; used by `hemisphere` model; default: 0.15
- **`baffle_height`** — Baffle height in m; used by `hemisphere` model; default: 0.25
- **`frequencies`** — Frequency sample points in Hz; auto-generated third-octave bands if None;
  `(n_freq,)` or None; default: None
- **`angular_resolution`** — Azimuth and elevation grid resolution in degrees; default: 5

### Outputs:
- **`speaker`** — Multi-frequency arrayant dict with the fields below; for `n_freq = 1` the pattern fields
  are 3D `(n_elevation, n_azimuth, n_elements)` and `center_freq` is a float:
  - `e_theta_re` — e-theta field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_theta_im` — e-theta field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_re` — e-phi field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_im` — e-phi field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `azimuth_grid` — Azimuth angles in rad, -π to π, sorted; `(n_azimuth,)`
  - `elevation_grid` — Elevation angles in rad, -π/2 to π/2, sorted; `(n_elevation,)`
  - `element_pos` — Element (x,y,z) positions in meters; `(3, n_elements)`
  - `coupling_re` — Coupling matrix, real part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `coupling_im` — Coupling matrix, imaginary part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `center_freq` — Frequency samples in Hz; `(n_freq,)`
  - `name` — Name of the array antenna object; str

---
## interpolate
Interpolate polarimetric array antenna field patterns (single- and multi-frequency)

- Interpolates the complex e-theta (V) and e-phi (H) field components at the requested azimuth / elevation angles
- NaN or infinite angles give NaN in all outputs for that angle (and element); no error is raised
- Single-frequency input (3D pattern fields) returns 2D outputs `(n_out, n_ang)`; the optional `dist` and
  `local_angles` outputs are available only on this path
- Multi-frequency input (4D pattern fields) interpolates spatially and across frequency; for each
  target frequency the two bracketing `center_freq` entries are blended via SLERP
- Passing `frequency` adds a frequency dimension to the output `(n_out, n_ang, n_freq_out)`; for a
  single-frequency antenna the spatial result is simply replicated across the requested frequencies
- `azimuth` / `elevation` of shape `(1, n_ang)` apply the same angles to all elements (planar
  wave); shape `(n_out, n_ang)` gives per-element angles (spherical wave)

### Usage:
```
# Single-frequency, separate real / imaginary parts
vr, vi, hr, hi = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation )

# Complex-valued output
v, h = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, complex=True )

# Projected distance / local angles (single-frequency only)
vr, vi, hr, hi, dist = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, dist=True )
vr, vi, hr, hi, azimuth_loc, elevation_loc, gamma = quadriga_lib.arrayant.interpolate( arrayant, azimuth, \
    elevation, orientation=ori, local_angles=True )

# Element selection, orientation, element positions
vr, vi, hr, hi = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, element, orientation, element_pos )

# Multi-frequency interpolation — output gains a frequency axis
vr, vi, hr, hi = quadriga_lib.arrayant.interpolate( arrayant, azimuth, elevation, frequency=freqs )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`;
  4D input requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`azimuth`** — Azimuth angles in rad, in [-pi, pi]; `(1, n_ang)` for shared angles (planar
  wave) or `(n_out, n_ang)` for per-element angles (spherical wave)
- **`elevation`** — Elevation angles in rad, in [-pi/2, pi/2]; same shape as `azimuth`
- **`element`** — Element indices to interpolate; duplicates allowed; `(n_out,)` or a
  list of int; `None` or empty uses all elements (`n_out = n_elements`); default: `None`
- **`orientation`** — antenna orientation (bank, tilt, heading) in rad; shape `(3, 1)`,
  `(3, n_out)`, `(3, 1, n_ang)`, or `(3, n_out, n_ang)`; `None` uses East-facing broadside;
  default: `None`
- **`element_pos`** — Alternative element (x,y,z) positions in m; `(3, n_out)`; `None` falls back
  to `arrayant["element_pos"]` (or zeros); default: `None`
- **`frequency`** — Target frequencies in Hz; `(n_freq_out,)` or scalar; adds a frequency axis to
  the output; for 4D input it interpolates between entries, for 3D input it replicates the spatial
  result; if omitted for 4D input the entries' `center_freq` values are used; default: `None`
- **`complex`** — If `True`, return complex-valued arrays instead of separate real / imaginary
  parts; default: `False`
- **`dist`** — If `True`, also return the projected distances; single-frequency input only and
  not allowed together with `frequency`; default: `False`
- **`local_angles`** — If `True`, also return the antenna-local angles; single-frequency input
  only and not allowed together with `frequency`; default: `False`
- **`fast_access`** — If `True`, require the pattern arrays to be double and column-major
  (Fortran-contiguous) so they can be read without a copy; raises if not. With `False`, such arrays are
  read without a copy as well and all others are copied; default: `False`

### Outputs:
Returned as a tuple; its length depends on `complex`, `dist`, and `local_angles`.
- **`vr`, `vi`, `hr`, `hi`** — Real / imaginary parts of the interpolated e-theta (V) and e-phi
  (H) field components; `(n_out, n_ang)`, or `(n_out, n_ang, n_freq_out)` when `frequency` is
  given; returned when `complex` is `False`
- **`v`, `h`** — Complex-valued e-theta (V) and e-phi (H) components; same shape as above;
  returned instead of `vr, vi, hr, hi` when `complex` is `True`
- **`dist`** — Projected distances between elements on the wavefront plane, used for phase
  computation; `(n_out, n_ang)`; returned only when `dist` is `True`
- **`azimuth_loc`, `elevation_loc`, `gamma`** — Azimuth, elevation, and polarization rotation
  angles in the local element frame, in rad; `(n_out, n_ang)` each; returned only when
  `local_angles` is `True`

### See also:
- [qdant_read](#qdant_read) / [qdant_write](#qdant_write) (load / save arrayant data)
- [generate](#generate) (arrayant struct layout)
- [generate_speaker](#generate_speaker) (typical multi-frequency struct array source)

---
## qdant_read
Reads array antenna data from QDANT files

- QDANT is the QuaDRiGa array antenna exchange format, an XML format for storing antenna patterns
- A QDANT file may hold one or several entries (e.g. a frequency-dependent antenna model)
- `id` reads a single entry by its 1-based ID; the result is a dict with 3D pattern fields
- `id = 0` reads every entry: a frequency-dependent dict with 4D pattern fields when the file
  holds multiple entries, or a plain single-entry dict (3D fields) when it holds exactly one; multiple
  entries must form a frequency set (same number of elements, same azimuth and elevation grids, element
  positions and coupling size), otherwise an error asks to pass `id`
- Differs from MATLAB, where `arrayant_qdant_read(fn)` without `id` reads every entry (here: `id = 0`)
- Reading all entries is the inverse of `qdant_write` with a 4D-pattern dict
- An `id` that is not in the file raises an error; the first entry counts as ID 1 only if it has no
  `id` attribute
- Malformed files raise an error, also when they declare more elements (`NoElements`) than the file
  has data for; elements without pattern data read as zeros (at most 2^24 zero values per entry)

### Usage:
```
# Read a single entry (default: ID 1)
data = quadriga_lib.arrayant.qdant_read( fn )
data = quadriga_lib.arrayant.qdant_read( fn, id )

# Read every entry as a frequency-dependent arrayant
data = quadriga_lib.arrayant.qdant_read( fn, id=0 )
```

### Inputs:
- **`fn`** — path to the QDANT file; str; must not be empty
- **`id`** — 1-based ID of the entry to read; `0` reads all entries (see above); default: 1

### Outputs:
- **`data`** — Dict with the array antenna data; keys as in [generate](#generate), with these specifics:
  - `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` — Pattern fields;
    `(n_elevation, n_azimuth, n_elements)` for a single entry, or
    `(n_elevation, n_azimuth, n_elements, n_freq)` when `id = 0` reads multiple entries
  - `center_freq` — Center frequency in Hz; scalar for a single entry, `(n_freq,)` when reading multiple entries
  - `coupling_re`, `coupling_im` — Coupling matrices; `(n_elements, n_ports)`, or
    `(n_elements, n_ports, n_freq)` when reading multiple entries with per-entry coupling
  - `layout` — Matrix of element IDs describing how the entries are arranged in the file; int64

### See also:
- [qdant_write](#qdant_write) (for writing QDANT data)
- [generate](#generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)

---
## qdant_write
Writes array antenna data to QDANT files

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
```
# Single-frequency write
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant )
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant, id )

# Frequency-dependent write (4D patterns) — all entries written sequentially
arrayant.quadriga_lib.qdant_write( fn, arrayant )
```

### Inputs:
- **`fn`** — Output QDANT filename; str; must not be empty
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`; 4D input
  requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`id`** — Target 1-based ID of the entry inside the file; `0` appends after the highest
  existing ID (or 1 if the file does not exist); ignored for 4D input; default: 0
- **`layout`** — uint32 matrix organizing multiple antenna IDs within the file; must reference
  only IDs present in the file; ignored for 4D input; default: `None` (a row of all IDs in the file,
  which replaces a layout written before)

### Outputs:
- **`id_in_file`** — ID assigned to the entry in the file after writing; `0` for a frequency-dependent (4D) write

### See also:
- [qdant_read](#qdant_read) (for reading QDANT data)
- [generate](#generate) (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)

---
## rotate_pattern
Rotate antenna radiation patterns around the principal axes using Euler rotations

- Rotates the pattern and/or polarization of array antenna elements around the x (bank), y (tilt), and z (heading) axes
- Rotations are applied in the order x, y, z, composed as Rz·Ry·Rx (intrinsic Tait-Bryan)
- Single-frequency input (3D pattern fields): `usage` 0 and 1 adjust the sampling grid for non-uniformly sampled antennas
- Coupling, `center_freq` and `name` are kept, also when the sampling grid is adjusted
- Frequency-dependent input (4D pattern fields): the rotation is applied to every entry and the grid is never adjusted,
  since all entries must share one grid
- For scalar acoustic fields (pressure stored in `e_theta_re` only) use `usage = 1` to avoid spurious polarization effects

### Usage:
```
# Rotate all elements by 45 deg bank
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, x_deg=45.0 )

# Rotate only elements 0 and 1 by 90 deg heading
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, z_deg=90.0, element=[0, 1] )

# Frequency-dependent input (4D patterns) — same interface
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, y_deg=10.0 )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [generate](#generate); pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`x_deg`** — Rotation around the x-axis (bank) in degrees; default: 0.0
- **`y_deg`** — Rotation around the y-axis (tilt) in degrees; default: 0.0
- **`z_deg`** — Rotation around the z-axis (heading) in degrees; default: 0.0
- **`usage`** — Rotation mode; default: 0
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

   for 4D input the grid is never adjusted, so `0`/`3` and `1`/`4` are equivalent
- **`element`** — Element indices to rotate; 1D list or array of int; `None` or empty rotates all
  elements; default: `None`

### Outputs:
- **`arrayant_out`** — Dict with the rotated array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input

---

# Channel functions

---
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

- Transforms time-domain channel coefficients into the frequency-domain transfer function H(f)
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel via OpenMP
- Two per-snapshot modes, selected by the ndim of the coefficient arrays:
  - single-freq: 3D items `(n_rx, n_tx, n_path)`; DFT over path delays at the requested carriers (AVX2)
  - multi-freq: 4D items `(n_rx, n_tx, n_path, n_freq_in)`; SLERP interpolation across `freq_in`,
    then delay-induced phase applied per output carrier
- All list items must share ndim, `n_rx`, `n_tx` (and `n_freq_in` for 4D); `n_path` may vary per snapshot
- Coefficients are given either as one complex array `coeff`, or as split `coeff_re` + `coeff_im`;
  supplying both forms is an error
- Output carriers are defined one of two mutually exclusive ways:
  - `bandwidth` paired with `pilot_grid` (or `carriers` for an evenly spaced grid) — single-freq only
  - `freq_in` + `freq_out` as absolute frequencies — required for multi-freq, optional for single-freq
- For single-freq with `freq_in` + `freq_out`, the baseband grid is derived internally; the phase is
  referenced to `freq_in[0]` when given, else to `min(freq_out)`
- `delay` may be broadcast: a `(1, 1, n_path)` item applies the same delays to every RX/TX pair;
  for multi-freq a 3D delay item is reused for all input frequencies
- `snap` selects a 0-based subset of snapshots; omitted processes all snapshots
- single-freq: the phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to
  one period before sin / cos are evaluated in single precision; coefficients and sums are single precision, double
  inputs are narrowed to float, results widened back; multi-freq computes in double precision

### Usage:
```
# Single-freq, bandwidth + evenly spaced carriers
hmat = quadriga_lib.channel.baseband_freq_response( coeff=coeff, delay=delay, bandwidth=100e6, carriers=128 )

# Single-freq, absolute frequencies
hmat = quadriga_lib.channel.baseband_freq_response( coeff=coeff, delay=delay, freq_in=freq_in, freq_out=freq_out )

# Multi-freq, split real / imaginary parts
hmat = quadriga_lib.channel.baseband_freq_response( coeff_re=cre, coeff_im=cim, delay=delay, freq_in=freq_in, freq_out=freq_out )
```

### Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
  (single-freq) or `(n_rx, n_tx, n_path, n_freq_in)` (multi-freq); mutually exclusive with
  `coeff_re` / `coeff_im`; default: `None`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item shaped like the matching
  `coeff` item or broadcast as `(1, 1, n_path)` (with an optional 4th dimension `n_freq_in`)
- **`bandwidth`** — Baseband bandwidth in Hz; paired with `pilot_grid` or `carriers`; single-freq only;
  cannot be combined with `freq_out`; default: `0.0`
- **`carriers`** — Number of evenly spaced carriers on `[0, 1]`; used only when both `pilot_grid` and
  `freq_out` are omitted; default: 128
- **`pilot_grid`** — Normalized sub-carrier positions, `0.0` = center, `1.0` = center + `bandwidth`;
  1D array of length `n_carrier`; default: `None`
- **`snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; omitted processes
  all snapshots; default: `None`
- **`coeff_re`** — Real part of channel coefficients; same list/shape layout as `coeff`; must be
  paired with `coeff_im`; mutually exclusive with `coeff`; default: `None`
- **`coeff_im`** — Imaginary part of channel coefficients; same layout as `coeff_re`; default: `None`
- **`freq_in`** — Input sample frequencies in Hz; 1D array; required for multi-freq input; for
  single-freq input used together with `freq_out` as the phase reference; default: `None`
- **`freq_out`** — Absolute output carrier frequencies in Hz; 1D array of length `n_carrier`;
  required for multi-freq input; may replace `bandwidth` + `pilot_grid` for single-freq; default: `None`
- **`remove_delay_phase`** — If `True`, undo the delay-induced phase baked in by the channel generator before SLERP,
  then re-apply it analytically per output carrier; multi-freq only; needs channels generated with
  `use_absolute_delays = True`: the coefficients of `get_channels_*` carry the phase of the absolute path length also
  when `delay` is relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq_in`,
  which gives wrong phases (magnitudes stay right); default: `True`

### Outputs:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`,
  where `n_out` is the number of processed snapshots

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency channel generator)
- [get_channels_multifreq](#get_channels_multifreq) (multi-frequency channel generator)

---
## channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g. for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` caps the total number of exported paths
- Takes raw channel data fields directly; no channel dict is required
- The per-snapshot fields `no_interact`, `interact_coord` and `coeff` each accept either a `list`
  (one entry per snapshot, ragged) or a single padded `ndarray` (MATLAB-style nD layout)

### Usage:
```
quadriga_lib.channel.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max, radius_min, \
    n_edges, rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff, coeff_re, coeff_im, i_snap )
```

### Inputs:
- **`fn`** — Output `.obj` file path; str
- **`max_no_paths`** — Max paths to export; 0 includes all paths above `gain_min`; default: 0
- **`gain_max`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5
- **`rx_pos`** — Receiver positions; ndarray `(3, n_snap)` or `(3, 1)`
- **`tx_pos`** — Transmitter positions; ndarray `(3, n_snap)` or `(3, 1)`
- **`no_interact`** — Number of interaction points of each path with the environment; uint32;
  either a `list` of `n_snap` arrays of shape `(n_path,)`, or an ndarray `(n_path, n_snap)`
- **`interact_coord`** — Interaction coordinates; either a `list` of `n_snap` arrays of shape
  `(3, sum(no_interact))`, or an ndarray `(3, max(sum(no_interact)), n_snap)`; fewer than `sum(no_interact)`
  columns in a snapshot raise an error
- **`center_freq`** — Center frequency in Hz; ndarray `(n_snap,)` or scalar
- **`coeff`** — Channel coefficients, complex valued; either a `list` of `n_snap` arrays of shape
  `(n_rx, n_tx, n_path)`, or an ndarray `(n_rx, n_tx, n_path, n_snap)`; provide this or
  `coeff_re`/`coeff_im`, not both; default: None
- **`coeff_re`** — Real part of the channel coefficients; same format as `coeff` but real-valued;
  must be paired with `coeff_im`; default: None
- **`coeff_im`** — Imaginary part of the channel coefficients; same format as `coeff_re`;
  must be paired with `coeff_re`; default: None
- **`i_snap`** — Snapshot indices to include; 0-based; `None` or empty exports all; default: None

### Outputs:
- None; the OBJ file is written directly to disk

---
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

- Initializes a new HDF5 file for storing channel data
- Defines a 4D storage layout `(nx, ny, nz, nw)`; each index combination maps to one channel slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be changed afterwards
- Raises an error if the target file already exists; delete it first to recreate it

### Usage:
```
storage_space = quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; str
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

### Outputs:
- **`storage_space`** — Storage dimensions used by the new file, `[nx, ny, nz, nw]`; `(4,)`; int64

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)

---
## hdf5_read_channel
Reads channel data from HDF5 files

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (= None)
- Datasets span `n_rx` RX antennas, `n_tx` TX antennas, `n_path` paths and `n_snap` snapshots;
  snapshots typically index positions along a trajectory or frequencies
- Slots are visited in column-major order and empty slots are skipped.
- Not every dataset spans all dimensions; only datasets present in the file are returned
- Per-snapshot data is returned as a `list` of arrays, one per snapshot, since `n_path` may differ between snapshots
- Structured fields are stored in single precision in the file and returned in double.
- Unstructured datasets keep their stored type and shape.
- `snap` selects a subset of snapshots; if `None`, all snapshots are read

### Usage:
```
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, snap, stack )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — 0-based slot indices along dimension X; scalar or vector; default: `0 ... nx-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector; default: `0 ... ny-1`
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector; default: `0 ... nz-1`
- **`iw`** — 0-based slot indices along dimension W; scalar or vector; default: `0 ... nw-1`
- **`snap`** — Snapshot indices to read; 0-based; default: all snapshots. Only allowed when the total selection is a single slot.
- **`stack`** — If `True`, stack snapshots. Default: `False` (return as list)

### Outputs:
- **`chan`** — List of dicts with the following keys:
  | Key                | Description                                                              | Shape `stack = False`             | Shape `stack = True`                 |
  | ------------------ | ------------------------------------------------------------------------ | --------------------------------- | ------------------------------------ |
  | `name`             | Channel name                                                             | str                               | str                                  |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `coeff`            | Channel coefficients, complex-valued                                     | list of `(n_rx, n_tx, n_path_s)`  | `(n_rx, n_tx, n_path, n_snap)`       |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)`  | `(n_rx, n_tx, n_path, n_snap)`       |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `path_length`      | Path length in m                                                         | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `path_polarization`| Polarization transfer function, complex                                  | list of `(4, n_path_s)`           | `(4, n_path, n_snap)`                |
  | `path_angles`      | Departure and arrival angles (AOD, EOD, AOA, EOA) in rad                 | list of `(n_path_s, 4)`           | `(n_path, 4, n_snap)`                |
  | `fbs_pos`          | First-bounce scatterer positions                                         | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `no_interact`      | Number of interaction points per path; int64                             | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `interact_coord`   | Interaction coordinates                                                  | list of `(3, sum(no_interact_s))` | `(3, max(sum(no_interact)), n_snap)` |
  | `center_frequency` | Center Frequency in Hz                                                   | `(n_snap,)` or scalar             | `(n_snap,)` or scalar                |
  | `initial_position` | Index of reference position; 1-based                                     | int32, scalar                     | int32, scalar                        |

- **`par`** — Dict (single slot) or list of dicts (multiple slots) containing the unstructured data in the file.

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout in the file)
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
## hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

- Reads one user-defined dataset from the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- The dataset is looked up under `'par_' + name` — the `par_` prefix is prepended internally
- The returned type and shape are defined by the dataset's HDF5 dataspace: a scalar dataset gives a Python
  number, a vector a numpy array of shape `(n,)` (also with a single element), a matrix or 3D array a numpy array
  of the stored shape
- Returns `None` if the dataset does not exist at the requested slot
- Supported types: str, scalar, vector, 2D array, and 3D array

### Usage:
```
dset = quadriga_lib.channel.hdf5_read_dset( fn, ix, iy, iz, iw, name )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`name`** — Dataset name without the `par_` prefix, e.g. `'carrier_frequency'`; str

### Outputs:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; `None` if the dataset is missing

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)
- [hdf5_read_channel](#hdf5_read_channel) (for reading structured channel data)

---
## hdf5_read_dset_names
Read the names of unstructured datasets from an HDF5 file

- Lists all unstructured datasets stored at the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- Datasets are identified by the `par_` prefix; the returned names have that prefix stripped
- Returns an empty list if no unstructured datasets are present at the slot

### Usage:
```
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0

### Outputs:
- **`names`** — Names of all unstructured datasets at the slot, with the `par_` prefix stripped; list of str

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
## hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

- Returns the dimensions of the 4D channel-slot grid stored inside an HDF5 file
- Also reports which slots already hold data, so free slots can be found without scanning the file
- Returns `(0, 0, 0, 0)` dimensions if the file does not exist
- Raises an error if the file exists but is not a valid HDF5 file

### Usage:
```
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str

### Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `(4,)`; int64
- **`has_data`** — Slot occupancy mask; 1 where a slot holds data, 0 otherwise; uint32; `(nx, ny, nz, nw)`

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data)

---
## hdf5_reshape_layout
Reshapes the storage layout inside an existing HDF5 file

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a valid HDF5 file

### Usage:
```
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; str
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

---
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

- Writes a list of channel dicts into 4D storage slots (one slot per list entry)
- `chan` may also be a single dict, in which case one channel is written
- Optional unstructured data (`par`) can be passed as a matching list of dicts (or a single dict)
- Creates the file with a default layout if it does not exist; appends to existing files otherwise
- A warning is issued if any selected slot already contains data. That slot is replaced completely: its old
  structured data and all its unstructured datasets (also those written with [hdf5_write_dset](#hdf5_write_dset)) are removed.
  The file does not shrink; use `h5repack` to reclaim the space of the removed data
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape; supported types as in [hdf5_write_dset](#hdf5_write_dset)
- Each scalar index input is broadcast to all channels; each vector index must have one entry per channel
- If the file does not exist, the layout is `[max(len(chan), max(ix)+1), max(iy)+1, max(iz)+1, max(iw)+1]`
- Channel dict field layout matches [hdf5_read_channel](#hdf5_read_channel)
- Per-snapshot fields accept the list model (list of arrays) or the stack model
  (single array with the snapshot index as the last axis), detected per field
- Coefficients may be passed as a complex `coeff`, or as separate real `coeff_re` and `coeff_im` (the two forms are mutually exclusive)
- Slot indices are 0-based

### Usage:
```
storage_dims = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`chan`** — Structured channel data; a channel dict or a list of channel dicts; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; a dict or a list of dicts with the same number of entries as `chan`. Dict keys become HDF5 dataset
  names per slot; `None` values are skipped. Pass `None` to disable; default: `None`
- **`ix`** — 0-based slot indices along dimension X; scalar or vector of length `len(chan)`; default: `0 ... len(chan)-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector of length `len(chan)`; default: 0
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector of length `len(chan)`; default: 0
- **`iw`** — 0-based slot indices along dimension W; scalar or vector of length `len(chan)`; default: 0

### Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `(4,)`; int64

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
- The dataset is added to the slot; everything else stored there is kept. A later [hdf5_write_channel](#hdf5_write_channel) to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: str; Python float (stored as double), int and bool (stored as int64; a bool is stored as the
  integer 0 or 1 and reads back as an int); numpy scalars and 0-d arrays; numpy arrays with 1, 2 or 3 dimensions
  in any memory layout and byte order; numeric element types: float32, float64, int32, uint32, int64, uint64.
  Other types (e.g. int16, numpy bool, complex) raise an error
- 1D arrays are stored as vectors and read back with shape `(n,)`, also with a single element (not as a scalar);
  2D and 3D arrays keep their shape, also `(n, 1)` and `(1, 1)`

### Usage:
```
quadriga_lib.channel.hdf5_write_dset( fn, ix, iy, iz, iw, name, data )
```

### Input Arguments:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; str
- **`data`** — Data to be written; type must be supported (see above); cannot be empty

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)

---
## qrt_file_parse
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR
  offsets, names, positions, orientations, and file version
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```
no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

### Inputs:
- **`fn`** — Path to the QRT file

### Outputs:
- **`no_cir`** — Number of channel snapshots per origin point
- **`no_orig`** — Number of origin points (TX)
- **`no_dest`** — Number of destination points (RX)
- **`no_freq`** — Number of frequency bands
- **`cir_offset`** — CIR offset per destination; `(no_dest,)`
- **`orig_names`** — Names of the origin points (TX); list of strings; length `no_orig`
- **`dest_names`** — Names of the destination points (RX); list of strings; length `no_dest`
- **`version`** — QRT file version number
- **`center_freq`** — Frequencies as stored in the file; GHz for EM mode (v4/v5), Hz for scalar mode (v6); `(no_freq,)`
- **`cir_pos`** — CIR positions in Cartesian coordinates; `(no_cir, 3)`
- **`cir_orientation`** — CIR orientations as Euler angles; `(no_cir, 3)`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; `(no_orig, 3)`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; `(no_orig, 3)`

---
## qrt_file_read
Read ray-tracing CIR data from a QRT file

- Reads channel impulse response data from QRT files
- A file read cache is initialized once and reused across all requested snapshots, which
  significantly speeds up multi-snapshot reads
- File reading and the conversion to numpy arrays are overlapped: a background thread reads the
  next snapshot into a double buffer while the main thread converts the previous one
- If `downlink = True`, origin is TX and destination is RX; if `False`, the roles are swapped
- Per-snapshot outputs are returned as lists with one entry per requested snapshot

### Usage:
```
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, M, aod, eod, aoa, eoa, \
    path_coord, no_int, coord, interact_type = quadriga_lib.channel.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M )
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`i_cir`** — Snapshot indices to read, 0-based; `(n_out,)` or `None`; if `None` or empty, all snapshots are read (for a file
  without CIRs, all outputs are empty with `n_out = 0`); default: `None`
- **`i_orig`** — Origin index, 0-based (origin = TX for downlink); scalar; default: 0
- **`downlink`** — If `True`, origin=TX and destination=RX; if `False`, the roles are swapped; default: `True`
- **`normalize_M`** — Controls `M` and `path_gain` scaling; 0 = values as stored in the QRT file with
  `path_gain` = -PL; 1 = `M` columns scaled to max power 1 and `path_gain` = -PL minus material losses; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |
    Other values raise an error.

### Outputs:
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`
- **`tx_pos`** — Transmitter positions in Cartesian coordinates; `(3, n_out)`
- **`tx_orientation`** — Transmitter orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`rx_pos`** — Receiver positions in Cartesian coordinates; `(3, n_out)`
- **`rx_orientation`** — Receiver orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`fbs_pos`** — First-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`path_gain`** — Path gain in linear scale; list of length `n_out`; entries `(n_path, n_freq)`
- **`path_length`** — Absolute path length from TX to RX phase center; list of length `n_out`; entries `(n_path,)`
- **`M`** — Polarization transfer matrix, stored as interleaved real/imaginary pairs; list of length `n_out`; entries `(8, n_path, n_freq)`, or `(2, n_path, n_freq)` for v6 files
- **`aod`** — Departure azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eod`** — Departure elevation angles; list of length `n_out`; entries `(n_path,)`
- **`aoa`** — Arrival azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eoa`** — Arrival elevation angles; list of length `n_out`; entries `(n_path,)`
- **`path_coord`** — Interaction coordinates per path; list of length `n_out`; each entry is a list of length `n_path` with arrays `(3, n_interact + 2)`
- **`no_int`** — Number of mesh interactions per path (0 indicates LOS); list of length `n_out`; entries `(n_path,)`
- **`coord`** — Interaction coordinates concatenated across paths; list of length `n_out`; entries `(3, sum(no_int))`
- **`interact_type`** — Interaction type codes concatenated across paths (matching `coord`); list of length `n_out`;
  entries `(sum(no_int),)`. Empty for v4 legacy files.

### See also:
- [generate](#generate) (for generating antenna arrays)
- [get_channels_planar](#get_channels_planar) (for embedding antennas using departure and arrival angles)
- [get_channels_spherical](#get_channels_spherical) (for embedding antennas using FBS/LBS positions)
- [get_channels_multifreq](#get_channels_multifreq) (for multi-frequency antenna embedding)

---
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α and δ^α,
  where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband
  (incoherent power preservation)
- If all fractional per-tap offsets are below 0.01 or above 0.99, weight computation is skipped
  (nearest-neighbor selection) but tap-selection logic still applies
- Snapshots are passed as Python lists; each list item is one snapshot and `n_path` may vary per item
- Coefficients are given either as one complex list `coeff`, or as split `coeff_re` + `coeff_im`;
  supplying both forms is an error. The input and output forms are independent
- `complex=True` returns one combined complex coefficient output; `complex=False` (default) returns
  separate real `coeff_re_q` and `coeff_im_q`
- Input `delay` items may be per-antenna `(n_rx, n_tx, n_path)` or shared `(1, 1, n_path)`;
  shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0. Delays at or below
  `-tap_spacing / 2` raise `ValueError`, as do NaN, Inf and delays of 2^31 taps or more (`delay / tap_spacing < 2^31`,
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
# Split real / imaginary output
coeff_re_q, coeff_im_q, delay_q = quadriga_lib.channel.quantize_delays( coeff_re, coeff_im, delay )

# Complex input and / or output (keyword)
coeff_q, delay_q = quadriga_lib.channel.quantize_delays( coeff=coeff, delay=delay, complex=True )
```

### Inputs:
- **`coeff_re`** — Real part of channel coefficients; list of length `n_snap`; each item
  `(n_rx, n_tx, n_path)`; must be paired with `coeff_im`; mutually exclusive with `coeff`;
  default: `None`
- **`coeff_im`** — Imaginary part of channel coefficients; same layout as `coeff_re`; default: `None`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or
  shared `(1, 1, n_path)`; non-negative with a tolerance of `tap_spacing / 2`, finite, below
  `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz
  sampling rate; default: 5e-9
- **`max_no_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; 0 = per tx-rx pair and snapshot, 1 = single shared grid,
  2 = per snapshot, 3 = per tx-rx pair across all snapshots; default: 0
- **`stack`** — If `True`, stack snapshots into a 4D array; if `False`, return a list of per-snapshot
  arrays; default: `False`
- **`complex`** — If `True`, return combined complex coefficients `coeff_q`; if `False`, return
  separate `coeff_re_q` and `coeff_im_q`; default: `False`
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`;
  mutually exclusive with `coeff_re` / `coeff_im`; default: `None`

### Outputs:
- **`coeff_q`** — Combined complex output coefficients; returned when `complex` is `True`; list of
  `(n_rx, n_tx, n_taps)` when `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)`
- **`coeff_re_q`** — Output coefficients, real part; returned when `complex` is `False`; list of
  `(n_rx, n_tx, n_taps)` when `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)`
- **`coeff_im_q`** — Output coefficients, imaginary part; returned when `complex` is `False`; same
  shape as `coeff_re_q`
- **`delay_q`** — Output delays in seconds; list of `(n_rx, n_tx, n_taps)` or `(1, 1, n_taps)` when
  `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)` or `(1, 1, n_taps, n_snap)`; the `(1, 1, ...)`
  shape only for shared delay input with `fix_taps` 1 or 2

---

# Channel generation functions

---
## get_channels_multifreq
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [get_channels_spherical](#get_channels_spherical) with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (`ant_tx['center_freq']`), RX array (`ant_rx['center_freq']`), input samples (`freq_in`), and output (`freq_out`)
- Antenna pattern fields may be 3D (single-frequency, clamped for all output frequencies) or 4D (multi-frequency, 4th axis = frequency)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `M` is interpolated via SLERP per complex entry pair to preserve phase
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs); pass 3D coupling `(n_elem, n_ports, n_freq)` for per-frequency coupling
- Extrapolation clamps to the nearest frequency entry on all four grids
- `M` accepts 8 rows (full polarimetric: the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx),
  rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 0 holds the LOS path
  (zero power if no LOS path was found; LOS = the path whose length is closest to the TX-RX distance, within 0.1 mm);
  a LOS path found in the input is moved to slot 0 and leaves a zero-power path in its original slot
  (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0, n_freq_out)`; with
  `add_fake_los_path = True` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [quantize_delays](#quantize_delays) rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [baseband_freq_response](#baseband_freq_response) with multi-frequency input and `remove_delay_phase = True` removes the phase of `delay`
  before it interpolates across `freq_in`, so it needs channels generated with `use_absolute_delays = True`; with
  relative delays, the phase of the TX-RX distance stays in the coefficients, is interpolated across the frequency
  grid and gives wrong phases
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path

### Usage:
```
coeff_re, coeff_im, delay = quadriga_lib.arrayant.get_channels_multifreq( ant_tx, ant_rx, fbs_pos, \
    lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, freq_in, \
    freq_out, use_absolute_delays, add_fake_los_path, propagation_speed )

coeff, delay = quadriga_lib.arrayant.get_channels_multifreq( ..., complex=True )
```

### Inputs:
- **`ant_tx`** — Multi-frequency TX arrayant dict; pattern fields 3D `(n_el, n_az, n_elem)` or 4D `(n_el, n_az, n_elem, n_freq)`; see [generate](#generate);
  4D input requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`ant_rx`** — RX arrayant dict; same format as `ant_tx`
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`
- **`path_gain`** — Linear-scale path gains, one column per input frequency; `(n_path, n_freq_in)`
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `(n_path,)`
- **`M`** — Polarization transfer matrix, row order as above; `(8, n_path, n_freq_in)` full pol or
  `(2, n_path, n_freq_in)` scalar pressure
- **`tx_pos`** — Transmitter position; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `(n_freq_in,)`
- **`freq_out`** — Target output frequencies; `(n_freq_out,)`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path_out, n_freq_out)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path_out, n_freq_out)`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency equivalent)
- [generate_speaker](#generate_speaker) (acoustic source construction)

---
## get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX port pairs across `n_path` propagation
  paths under the planar wave assumption.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based; the input angles are not used for LOS detection.
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `freq` selects the frequency index that is processed; a single frequency is returned per call.
- If `center_freq` is `None` or the selected entry is `0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == False`, the straight-line TX-RX delay between the phase centres (LOS delay) is subtracted
  from all paths; per-element delays of pairs that are closer to each other than the phase centres are then
  negative, down to about -(half the TX array size + half the RX array size) / speed of light; [quantize_delays](#quantize_delays) rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.
- If `add_fake_los_path == True`, one output path is always added (output size `n_path + 1`): slot 0 holds the LOS
  path, with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the
  TX-RX distance; it is moved to slot 0 and its original slot (input index + 1) holds a zero-power path. All other
  paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0)`; with `add_fake_los_path = True` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path.

### Usage:
```
coeff_re, coeff_im, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ant_tx, ant_rx, \
    aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ..., complex=True )

coeff_re, coeff_im, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ..., freq=1 )
```

### Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [generate](#generate)
- **`ant_rx`** — Receive arrayant dict; see [generate](#generate)
- **`aod`** — Departure azimuth angles in rad; `(n_path,)`
- **`eod`** — Departure elevation angles in rad; `(n_path,)`
- **`aoa`** — Arrival azimuth angles in rad; `(n_path,)`
- **`eoa`** — Arrival elevation angles in rad; `(n_path,)`
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`
- **`M`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; set an entry to `0` or pass `None` to skip
  phase computation; default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`
- **`freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `M` and `center_freq` that is
  processed; default: `0`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = toward path, negative = away; `(n_path,)`

### See also:
- [get_channels_spherical](#get_channels_spherical) (spherical wave variant)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (IEEE compliant channels using `get_channels_planar` internally)
- [generate](#generate) (antenna array generator)

---
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using the spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [qrt_file_read](#qrt_file_read) with more than two interactions) is added to the FBS-LBS segment. Delays and phases follow this
  length; the per-element offsets follow the FBS / LBS geometry.
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `freq` selects the frequency index that is processed; a single frequency is returned per call.
- If `center_freq` is `None` or the selected entry is `0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == False`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [quantize_delays](#quantize_delays) maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == True`, one output path is always added (`n_path + 1` outputs): slot 0 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 0 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0)`; with `add_fake_los_path = True` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair (also per snapshot in batch
  mode).
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path

### Batch mode:
- If `fbs_pos` is a `list` of arrays, the function switches to batch mode and processes `n_out` snapshots
  in one call, distributing the snapshots over all available cores (OpenMP).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `M` are then lists of length `n_out`,
  matching the output format of [qrt_file_read](#qrt_file_read); list entries that are Fortran-contiguous float64 arrays (as
  returned by [qrt_file_read](#qrt_file_read)) are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_out)` arrays with one column per snapshot; a `(3,)` array
  or a single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_out`, one entry per snapshot.
- Snapshots without any path produce an empty output entry with `n_path = 0`, or only the zero-power LOS path
  when `add_fake_los_path` is set (as for a single snapshot).
- Batch mode releases the GIL while processing, so it can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

### Usage:
```
coeff_re, coeff_im, delays = quadriga_lib.arrayant.get_channels_spherical( ant_tx, ant_rx, \
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff, delays = quadriga_lib.arrayant.get_channels_spherical( ..., complex=True )

coeff_re, coeff_im, delays, aod, eod, aoa, eoa = quadriga_lib.arrayant.get_channels_spherical( ..., angles=True )

coeff_re, coeff_im, delays = quadriga_lib.arrayant.get_channels_spherical( ..., freq=1 )
```

### Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [generate](#generate)
- **`ant_rx`** — Receive arrayant dict; see [generate](#generate)
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`, or a list of length `n_out` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`, or a list of length `n_out` in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list of length `n_out` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `(n_path,)`, or a list of length `n_out` in batch mode
- **`M`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list of length `n_out`
  in batch mode
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`, or `(3, n_out)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_out)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`, or `(3, n_out)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_out)` in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; set an entry to `0` or pass `None` to skip phase computation; default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`angles`** — If `True`, also return departure/arrival angles in antenna-local coordinates; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`
- **`freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `M` and `center_freq` that is
  processed; default: `0`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`aod`** — Azimuth of departure in rad; `(n_ports_rx, n_ports_tx, n_path)`; only when `angles=True`
- **`eod`** — Elevation of departure in rad; same shape; only when `angles=True`
- **`aoa`** — Azimuth of arrival in rad; same shape; only when `angles=True`
- **`eoa`** — Elevation of arrival in rad; same shape; only when `angles=True`
- In batch mode, every output is a list of length `n_out` whose entries have the shapes given above.

### See also:
- [get_channels_multifreq](#get_channels_multifreq) (multi-frequency extension)
- [get_channels_planar](#get_channels_planar) (planar wave variant)
- [get_channels_irs](#get_channels_irs) (IRS-assisted communication)
- [qrt_file_read](#qrt_file_read) (produces the list format used by batch mode)

---
## get_ieee_indoor
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
- Per-snapshot data (`coeff`, `delay`, `path_gain`) is returned as a list of per-snapshot arrays, 
  or stacked into one array along an appended snapshot axis when `stack = True`

### Usage:
```
chan = quadriga_lib.channel.get_ieee_indoor( ap_array, sta_array, ChannelType, CarrierFreq_Hz, \
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, \
   Dist_m, n_floors, uplink, offset_angles, n_subpath, Doppler_effect, seed, \
   KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m, n_walls, wall_loss, stack )
```

### Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see arrayant-[generate](#generate)
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see arrayant-[generate](#generate)
- **`ChannelType`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency (finite); default: 5.25e9
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** *(optional)* — Channel observation time in seconds; must be finite; default: 0
- **`update_rate`** *(optional)* — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** *(optional)* — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`Dist_m`** *(optional)* — TX-to-RX distance(s) (finite); `(n_users,)` or `(1,)`; default: 4.99
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `(n_users,)` or `(1,)`; default: 0
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: false
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `(4, n_users)`; default: [] (auto-generate)
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** *(optional)* — RNG seed for repeatability; `-1` uses the system random device; default: -1
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default; default: NAN
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN restores model default; must otherwise be finite and > 0 (0, negative and Inf throw); default: NAN
- **`SF_std_dB_LOS`** *(optional)* — Overrides LOS shadow fading std in dB (applied when d < dBP); NAN restores model default; default: NAN
- **`SF_std_dB_NLOS`** *(optional)* — Overrides NLOS shadow fading std in dB (applied when d >= dBP); NAN restores model default; default: NAN
- **`dBP_m`** *(optional)* — Overrides breakpoint distance; NAN or negative restores model default; default: NAN
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `(n_users,)` or `(1,)`; default: 0
- **`wall_loss`** *(optional)* — Penetration loss for a single wall; TGax defines 5 or 7; default: 5
- **`stack`** *(optional)* — If `True`, per-snapshot data is stacked into a single array along an appended 
  snapshot axis instead of being returned as a list of per-snapshot arrays; default: False

### Output:
- **`chan`** — List of length `n_users` containing dictionaries of channel data. Only the keys
  listed below are populated; all other channel keys are omitted by this generator.
  | Key                | Description                                                              | Shape `stack = False`            | Shape `stack = True`           |
  | ------------------ | ------------------------------------------------------------------------ | -------------------------------- | ------------------------------ |
  | `name`             | Channel name                                                             | str                              | str                            |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `coeff`            | Channel coefficients, complex valued                                     | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`            | `(n_path, n_snap)`             |
  | `center_frequency` | Center Frequency in Hz                                                   | `(1,)`                           | `(1,)`                         |
  | `initial_position` | 0-based index of the reference snapshot                                  | int                              | int                            |

See also:
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf">S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025</a>
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data to a HDF5 file)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data to a HDF5 file)

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
  and `sig`; if no set has finite values (possible only with given `bins`), `cdf_avg`, `mu` and `sig` are NaN
- If `bins` is empty, equally spaced bins spanning the data range are generated

### Usage:
```
cdf_per_set, bins_out, cdf_avg, mu, sig = quadriga_lib.tools.acdf( data, bins, n_bins )
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `(n_samples, n_sets)`
- **`bins`** — Bin centers; used as-is if non-empty (must be finite and strictly ascending, else a
  `ValueError` is raised); if empty, equally spaced bins spanning the data range are generated;
  `(n_bins_in,)` or empty; if `None` or empty, bins are auto-generated.
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins_in`
  are provided; default: 201

### Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data (NaN for a set without finite values); `(n_bins_out, n_sets)`
- **`bins_out`** — Auto-generated bins; copy of `bins` when non-empty `bins_in` are provided;
  `n_bins_out = n_bins` or `n_bins_out = n_bins_in`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across the data sets with finite values; `(n_bins_out,)`
- **`mu`** — Mean of the 0.1–0.9 quantiles across the data sets with finite values; `(9,)`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values; `(9,)`

---
## calc_angular_spread
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- RMS spread uses the 3GPP TR 38.901 second-moment definition of wrapped deviations from the
  circular mean
- When `wrapping = True`, the power-weighted mean direction is computed in Cartesian coordinates
  and all paths are rotated so the centroid lies on the equator before computing spreads,
  avoiding pole singularity artifacts
- When `wrapping = False`, spreads are computed directly from raw angles; `orientation` is zero
  and `phi`/`theta` equal the input `az`/`el`
- When `calc_bank_angle = True`, an optimal bank angle maximizing azimuth spread is derived
  analytically from eigenvectors of the 2x2 power-weighted covariance matrix; only used when
  `wrapping = True`
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed

### Usage:
```
azimuth_spread, elevation_spread, orientation, phi, theta = \
    quadriga_lib.tools.calc_angular_spread( az, el, powers, wrapping, calc_bank_angle, quantize )
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; list of length `n_cir`, each element a 1D array of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers; list of length `n_cir`, each element a 1D array of length `n_path`
- **`wrapping`** — If True, enables spherical rotation; default: False
- **`calc_bank_angle`** — If True, computes the optimal bank angle analytically; only used when `wrapping = True`; default: False
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0.0 (no quantization);
  otherwise it must be finite and >= 1e-3 deg, else a `ValueError` is raised

### Outputs:
- **`azimuth_spread`** — RMS azimuth angular spread; `(n_cir,)`
- **`elevation_spread`** — RMS elevation angular spread; `(n_cir,)`
- **`orientation`** — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `(3, n_cir)`
- **`phi`** — Rotated azimuth angles; list of length `n_cir`
- **`theta`** — Rotated elevation angles; list of length `n_cir`

---
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- Inputs are lists so each CIR can have a different number of paths
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis
- `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation V/H), stored column-major with
  interleaved real/imaginary parts, 8 rows per path: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx),
  rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its receive and transmit
  polarisation: `M_vv` = rows 0-1, `M_hv` = rows 2-3, `M_vh` = rows 4-5, `M_hh` = rows 6-7
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
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default
- Normalization of `M` does not affect XPR (cancels in the ratio) but does affect `pg`
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR
  is set to 0
- TX/RX positions may be fixed `(3, 1)` or mobile `(3, n_cir)`

### Usage:
```
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, \
    include_los, window_size )
```

### Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`M`** — Polarization transfer matrices with interleaved real/imag parts per CIR; list of length
  `n_cir`, each element a 2D array of shape `(8, n_path)`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`include_los`** — If True, includes LOS and near-LOS paths in the XPR calculation; default: False
- **`window_size`** — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded when
  `include_los = False`; default: 0.01

### Outputs:
- **`xpr`** — XPR on linear scale; `(n_cir, 6)`; columns (0-based):
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_vh)^2) (receive V)          |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_hv)^2) (receive H)          |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_LR)^2) (receive L)       |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_RL)^2) (receive R)       |
- **`pg`** — Total path gain summed over all paths (including LOS) as
  `0.5 · sum(powers · (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `(n_cir,)`

---
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Computes RMS delay spread from a set of delays and corresponding linear-scale powers per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- Paths with power below `p_max / 10^(0.1 · threshold)` are excluded; the default threshold of
  100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their
  powers summed before the spread is computed

### Usage:
```
ds, mean_delay = quadriga_lib.tools.calc_delay_spread( delays, powers, threshold, granularity )
```

### Inputs:
- **`delays`** — Delays per CIR; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`threshold`** — Power threshold in [dB] relative to the strongest path; paths below threshold
  are excluded; default: 100.0
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0.0 (no grouping); must be
  finite and non-negative; when > 0, all delays must be finite and the delay span / granularity must be below 2^53,
  else a `ValueError` is raised

### Outputs:
- **`ds`** — RMS delay spread per CIR; `(n_cir,)`
- **`mean_delay`** — Mean delay per CIR; `(n_cir,)`

---
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

- Computes the Rician K-Factor (KF): ratio of LOS power to NLOS power per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- LOS paths are those with length ≤ `dTR + window_size`, where `dTR` is the direct TX-RX distance;
  remaining paths are NLOS
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is set to 0
- TX/RX positions may be fixed `(3,)` / `(3, 1)` (reused for all CIRs) or mobile `(3, n_cir)`

### Usage:
```
kf, pg = quadriga_lib.tools.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size )
```

### Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path` matching `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS;
  default: 0.01

### Outputs:
- **`kf`** — Rician K-Factor on linear scale; `(n_cir,)`
- **`pg`** — Total path gain (sum of all path powers) per CIR; `(n_cir,)`

---

# Geometry-based stochastic model

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the
*scenario* (e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The
functions of this section implement the large-scale part of the model of 3GPP TR 38.901, clause 7.5 steps 1 to 4:
path loss, shadow fading, LOS probability, O2I penetration loss and the large-scale parameters (LSPs) of each link,
i.e. the spreads of delays and angles, the K-factor and the cross-polarization ratio. They are in the submodule
`quadriga_lib.gbsm`.

The statistics of a scenario are read from a conf file by [conf_file_read](#conf_file_read) into two dicts, the scenario
parameters `scenpar` and the path-loss parameters `plpar`. The folder `data/config` holds 23 files for TR 38.901
v19.5.0; [scenpar_validate](#scenpar_validate) completes a parameter set from another source, [conf_file_list](#conf_file_list) finds the files
and [conf_file_write](#conf_file_write) writes them.

Per drop (layout): [o2i_loss](#o2i_loss) gives the penetration loss and the indoor distance of each indoor UT; then
[los_probability](#los_probability) evaluates the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS
state and the indoor state with [sos.rand](#sosrand) / [sos.randn](#sosrandn) at the decorrelation distances of the keys
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

- Lists the regular files and symbolic links to regular files whose name ends in `.conf` (case-sensitive)
  and is longer than `.conf`, not recursive. The scenario name is the file name without this extension,
  whatever characters it has.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order
  of their names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first
  directory wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = True` each file is read and one entry is added right after the file for
  every line `ShortName = name` that is not a comment. Other lines are not examined; a UTF-8 byte order
  mark at the start of the file is skipped.
- Raises `ValueError` if a short name equals an earlier entry, if a file name equals an earlier short
  name, or if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`, `_`
- Raises `RuntimeError` if a listed file cannot be opened for reading the short names
- A directory that does not exist, is not a directory or cannot be listed is skipped
- An empty str `""` is the current directory

### Usage:
```
scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list( dirs, parse_shortnames )
```

### Inputs:
- **`dirs`** — Directories, searched in this order; one path or a list or tuple of paths, each a str,
  bytes or path-like object (e.g. `pathlib.Path`); an empty list or `None` lists nothing
- **`parse_shortnames`** — `True`: read the files and add their short names; `False`: list the files
  only, which is faster; bool or `numpy.bool_`; default: True; `None` selects the default

### Outputs:
- **`scenarios`** — Scenario names and short names; list of str; length `n_scen`
- **`file_names`** — File name including `.conf` for each entry; list of str; length `n_scen`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs` (a path-like object
  converted with `os.fspath`); list of str; length `n_scen`

### See also:
- [conf_file_read](#conf_file_read) (read one of the files)

---
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank
  lines are allowed; line ends can be LF or CRLF; space and tab are white space; a UTF-8 byte order mark at the
  start of the file is skipped
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
  (`e` or `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted in any case
- A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the smallest
  subnormal reads as `0` (or `-0`)
- Numbers are read independent of the locale (the decimal point is always `.`)
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 171 numeric keys are 138 keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys`; the permittivity is complex and is set in the simulation
- Raises `RuntimeError` if the file cannot be opened or is a directory
- Raises `ValueError` with file name and line number for a line that is not blank, not a comment and
  not a well-formed assignment: no `=`, more or less than one name before the `=`, a missing value, a
  word where a number is required or the reverse, a second value or other text after the value, `NaN`,
  complex numbers, a number out of range, a `PL_` key without a name, a `ShortName` value stated twice
- With `validate = True`, a value that fails the checks of [scenpar_validate](#scenpar_validate) is an error as well
- A correlation matrix that is not positive definite is not an error; see `lsp_xcorr_chk`

### Usage:
```
scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
    quadriga_lib.gbsm.conf_file_read( fn, validate )
```

### Inputs:
- **`fn`** — Path to the conf file; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`validate`** — `True`: apply the value checks of [scenpar_validate](#scenpar_validate); `False`: only the line
  grammar is checked; bool or `numpy.bool_`; default: True; `None` selects the default

### Outputs:
- **`scenpar`** — Scenario parameters; dict with 172 entries in the QuaDRiGa order: the 138 numeric
  keys of QuaDRiGa with `SubpathMethod` (str) at position 106 (0-based), between `absTOA_lambda` and
  `LOS_scatter_radius`, then the 33 new keys; all other values are float
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str, only if the file states
  `PL_model`) first, then one entry per `PL_` key in file order, named without the prefix `PL_`; each
  is a 1D array with one or more values; `{}` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; bool
- **`short_names`** — Values of all `ShortName` lines in file order; list of str
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; list of str

### See also:
- [scenpar_validate](#scenpar_validate) (value checks, defaults and the list of validation rules)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)

---
## conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files
  shipped with QuaDRiGa, so that [conf_file_read](#conf_file_read) returns the same values bit for bit
- Header: `% Config file for scenario "name"` with the file name without the extension `.conf`,
  then the reference `ref`, the validity range and one line `ShortName = name` per short name
- Up to 10 sections follow, each with a header and, if the entry of `ref_sec` is not empty, a line
  `% See: ref; ref_sec`. Index in `ref_sec` and write order:
  - 0: Large scale distributions, with an overview formula per LSP; 1: Large-Scale fading
    decorrelation distances; 2: Inter-parameter correlations, with the correlation matrix as
    comment; 3: Model parameters; 8: EoD offset; 4: Decorrelation distance for the small-scale
    fading spatial consistency; 5: Parameters for the ground reflection extension; 6: Absolute
    time of arrival offset; 9: Optional modelling components; 7: Path-loss model
  - The indices 0 to 7 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- The overview formula of an LSP shows its mean and STD with the frequency, distance, height and elevation
  terms; with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`; the ESD line also has the
  terms of `ES_D_mu_A` to `ES_D_mu_D` and, if `ES_D_mu_min` is not `-Inf`, the lower limit of the mean;
  the DS line has the term `log10( a · V/S + b )` when `DS_VS > 0` (a = `DS_VS_a`, b = `DS_VS_b`;
  `V/S` stays symbolic)
- Each line is `name = value % comment`, aligned in columns; the comment states the unit
- The path-loss section has the formula blocks of the model (`logdist`, `dual_slope`, `nlos`,
  `tripple_slope`), the line `PL_model`, then the entries of `plpar` in dict order with the prefix
  `PL_`; several values of one entry are written on one line
- Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back to
  the same double in [conf_file_read](#conf_file_read); `Inf` and `-Inf` are written as such
- With `write_defaults = False` a key is written only if its value differs from the default;
  `SubpathMethod` only if it is not `legacy`; a section without a written key is left out
- With `write_defaults = True` all 171 numeric keys and `SubpathMethod` are written
- The inputs are checked with the rules of [scenpar_validate](#scenpar_validate) before the file is opened: unknown
  keys (including `GR_epsilon`), `NaN`, values outside their range and a wrong `SubpathMethod`
  are errors. Further errors: a line break in the file name, a `model` that is not one name, an empty or
  `NaN` path-loss value, short names that are not one name or not distinct, line breaks in `ref` or
  `ref_sec`, more than 10 entries in `ref_sec`, a `valid_range` that is not empty and not `(4, 2)`.
- Raises `ValueError` for invalid input and `RuntimeError` if the file cannot be opened or written
- The text is written to a temporary file in the same directory, which must be writable, and then renamed
  to `fn`; a symbolic link `fn` is written through
- A failed call does not create a file and leaves an existing file unchanged; an existing file is replaced
  and keeps its permissions
- Every file written passes [conf_file_read](#conf_file_read) with `validate = True` and without ignored keys

### Usage:
```
quadriga_lib.gbsm.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, write_defaults )
```

### Inputs:
- **`fn`** — Path of the conf file to write; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`scenpar`** — Scenario parameters; dict with any subset of the 172 keys returned by
  [conf_file_read](#conf_file_read); numeric values are real scalars (int, float, bool or a numpy value with one
  element); `SubpathMethod` is a str; missing keys have their default; `None` or `{}` writes the
  default set
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str) and one real scalar, list or
  1D array per further entry, named without the prefix `PL_`, as returned by [conf_file_read](#conf_file_read);
  written in dict order; `None` or `{}`: no path-loss section; default: None
- **`short_names`** — Short names of the scenario; list of str or one str; each one name of the
  characters `A-Z`, `a-z`, `0-9`, `_`; distinct; default: None
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; str; default: None (no reference)
- **`ref_sec`** — Reference per section, indexed as listed above; list of up to 10 str; an empty or
  missing entry writes no reference; default: None
- **`valid_range`** — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, MT height in m; columns: lower and upper end, one value is written if both are equal; a
  row with `NaN` is not written; `(4, 2)`; default: None
- **`write_defaults`** — `True`: write all keys; `False`: write only keys that differ from their
  default; bool or `numpy.bool_`; default: False; `None` selects the default

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
- Soft LOS state (TR 38.901 eq. 7.6-18), computed when `rnd_g` and `fc` are both given (not `None`):
  `los_soft(l, k) = 1/2 + atan(sqrt(20/lambda_k)·(rnd_g(l) + F(p_los(l))))/pi`, `lambda_k = c/fc(k)` with
  `c = 299792458` m/s and `F(p) = sqrt(2)·erfinv(2p - 1)` the inverse of the standard normal CDF (accurate to
  `1e-10·max(1, abs(F))` for `p` in [1e-12, 1 - 1e-12]). `p = 1` gives 1, `p = 0` gives 0. `los_soft > 0.5`
  exactly when `rnd_g > -F(p)`, which has the probability `p` for `rnd_g` from N(0,1)
- The LOS state itself (a draw against `p_los`) and its spatial correlation are the caller's: e.g. a spatially
  consistent uniform value from [sos.rand](#sosrand), or `rnd_g` from [sos.randn](#sosrandn), at the decorrelation distance of the
  scenario key `LOS_state_lambda` (TR 38.901 Table 7.6.3.1-2). The key applies to all links of a scenario and is
  stated only in its NLOS conf file (0 in the LOS and O2I files means "not stated here"). The `LOS_state_lambda` of
  the InF conf files is `d_clutter / 2` of the TR defaults; it does not follow a `d_clutter` argument, so a caller
  who changes `d_clutter` passes its own correlation distance
- O2I UTs: Table 7.4.2-1 uses the outdoor distance `d2d_out = d2D - d2D-in` with `d2D-in` from [o2i_loss](#o2i_loss), which
  therefore runs first; the caller clamps `d2d_out` at 0 (negative distances are rejected). Indoor scenarios
  (InH, InF) pass the 2D distance
- Validation, in this order: the model name; `d2d_out` finite and >= 0; `h_bs` and `h_ut` with 1 or `n_link`
  elements and finite; `d_clutter` NaN or finite and > 0, `r_clutter` NaN or in [0, 1), `h_clutter` NaN or finite
  and >= 0 (for every model); if `los_soft` is computed: `rnd_g` with `n_link` finite elements, `fc`
  non-empty, finite and > 0. Without `los_soft`, a given `rnd_g` or `fc` is ignored. All errors are raised as
  `ValueError`

### Usage:
```
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, \
    h_clutter, rnd_g, fc )
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`;
  str
- **`d2d_out`** — 2D outdoor distance in m (indoor scenarios: 2D distance), finite and >= 0; `(n_link,)`,
  `n_link = 0` allowed
- **`h_bs`** — BS height in m; one element is used for all links; float or `(n_link,)`
- **`h_ut`** — UT height in m; one element is used for all links; float or `(n_link,)`
- **`d_clutter`** — Clutter size in m of the InF and SMa models; float; `None` or nan = TR default of the model;
  default: None
- **`r_clutter`** — Clutter density of the InF models, vegetation density `r_veg` of SMa, in [0, 1); float;
  `None` or nan = TR default of the model; default: None
- **`h_clutter`** — Clutter height in m of InF-SH and InF-DH; float; `None` or nan = TR default of the model;
  default: None
- **`rnd_g`** — Normal(0,1) value per link for the soft LOS state (G of eq. 7.6-18); `(n_link,)`;
  default: None
- **`fc`** — Carrier frequencies in Hz for the soft LOS state; `(n_freq,)`; default: None

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `(n_link,)`
- **`los_soft`** — Soft LOS state in [0, 1]; `(n_link, n_freq)`; `None` if `rnd_g` or `fc` is `None`

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos.rand](#sosrand), [sos.randn](#sosrandn) (spatially consistent LOS-state draw and `rnd_g`)
- [conf_file_read](#conf_file_read) (key `LOS_state_lambda`)

---
## lsf_draw
Spatially correlated normalized random values of the large-scale generators at a set of links

- Evaluates the 11 large-scale fading generators of [lsf_init_phases](#lsf_init_phases) at the links: column 0 DS, 1 KF, 2 SF,
  3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal(0,1)), 9 ground permittivity, 10 UMa environment
  height hE (Uniform(0,1)); each generator with the sum-of-sinusoids frequencies of `preset` at its own
  decorrelation distance (a distance of 0 is used as 0.1 m)
- `scenpar` is completed and validated as in [lsf_init_phases](#lsf_init_phases); the messages of [scenpar_validate](#scenpar_validate) pass
  through unchanged
- Single mobility (`dual_mobility = False`): column g is the field of generator g at the Rx position,
  `x_g = s_g(rx)` with `s_g(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and the phases of
  slice 0; the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 1, i.e. the field of the mirror generator
  `m(g)` at the Tx position (ASD and ASA, ESD and ESA exchanged). Self-mirrored generators (DS, KF, SF, XPR,
  TOA, columns 9 and 10) share one map: `x_g = ( s_g(rx) + s_g(tx) ) / sqrt( 2 + 2·A_g(D) )`, with the Tx-Rx
  distance `D` and the ACF `A_g` of the preset, linearly interpolated. The pairs are two maps:
  `x_g = ( s_g(rx) + s_m(tx) ) / sqrt(2)`. Columns 3 to 6 thus stay Normal(0,1), and ASD/ASA and ESD/ESA have
  the correlation `( A_g(D) + A_m(D) ) / 2` at the Tx-Rx distance, 1 at `D = 0` (then the two columns are
  equal)
- Swap reciprocity in dual mobility: swapping `tx_pos` and `rx_pos` leaves columns 0, 1, 2, 7, 8, 9, 10
  unchanged and swaps column 3 with 4 and column 5 with 6, exactly (the same evaluations in swapped roles), also
  for unequal decorrelation distances and with bias-removed phases, when both calls use the same `phase`
- Uniform columns 9 and 10: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = True`: the positions are projected on the x-y plane (z = 0) and the values do not depend on z;
  `False`: 3D positions. `D` is measured in the same space
- 2D and 3D mode give the same per-link statistics. In 3D mode the height separates the floors: the LSPs of UTs
  on different floors are correlated by the ACF of their height difference, while TR 38.901 (Table 7.5-6
  note 3) assumes no correlation between floors. `floor_id` is therefore not accepted with `horizontal = False`
- Floors: links with different floor IDs see different fields, links with equal IDs share one field. Floor `q`
  adds the offsets `2·pi · k · n / N + psi_b(n)` to the phases of all generators on both sides, with
  `N = n_coeff`, `k = q mod N`, `b = q div N`, `n = 0 ... N-1`, `psi_0 = 0` and `psi_b(n) = 2·pi · u_n` for
  `b >= 1`, where `u_n` are the uniform numbers of a `std::mt19937_64` seeded with `b` (the upper 53 bits of
  each raw output times 2^-53). Floor 0 and `floor_id = None` give the same values
- Fields of two floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3): exactly
  for two IDs in the same block of `N` IDs, with a random residual of standard deviation `1/sqrt(2N)` for IDs
  in different blocks (0.044 for 256 sinusoids). At other horizontal lags the cross-correlation of two floors
  is not zero (up to about 0.16 for 256 sinusoids), the same level as for random offsets. Mirror relations and
  swap reciprocity hold per floor
- `floor_id` is the caller's floor index of the UT, starting at 0: for the UT heights
  `hUT = 3·(n_fl - 1) + 1.5` of TR 38.901 Table 7.2-1 it is `n_fl - 1`; outdoor UTs use 0. The IDs are read as
  float64, so they must be below 2^53 (larger values would not be exact)
- Dual mobility with floors: `floor_id` is one value per link and its offsets are added on both sides, i.e. both
  devices of a link are on the link's floor. A Tx device on another floor than its Rx (e.g. a D2D link between two
  floors) is evaluated on the floor of the Rx, so the same Tx device sees different fields on links to Rx devices
  on different floors
- The value of a link does not depend on the other links of the call; `n_link = 0` returns an array of
  shape `(0, 11)`
- Precision: the sinusoids are evaluated by [sos.val](#sosval) with the AVX2 kernel where available (sine and cosine
  in single precision, accumulation in double); the values deviate from a double-precision evaluation by less
  than `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [lsf_init_phases](#lsf_init_phases). A wrong preset with the same number of
  sinusoids (e.g. `Exp256` for `Comb256`) cannot be detected
- The flags `dual_mobility` and `horizontal` take a bool or `numpy.bool_`; `None` selects the default, any other
  type raises `TypeError`
- Validation order: `floor_id` (non-negative integers below 2^53, checked by the wrapper), positions (3 columns, rows of
  `tx_pos`, finite values), `preset`, `scenpar`, size of `phase`, finite `phase`, length of `floor_id`,
  `floor_id` with `horizontal = False`

### Usage:
```
rnd = quadriga_lib.gbsm.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, preset, horizontal )
```

### Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; only the 10 decorrelation distances are used
- **`phase`** — Phases in rad from [lsf_init_phases](#lsf_init_phases); slice 0 = Rx side, slice 1 = Tx side; must be finite;
  `(n_coeff, 11, 2)`, `n_coeff` of `preset`
- **`tx_pos`** — Tx positions in m; one row is used for all links; not used in single mobility (but
  validated); `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`floor_id`** — Floor index per link, starting at 0; non-negative integers below 2^53, given as int or float
  array; `None` or empty = all links on floor 0; in dual mobility the floor of both devices; only with
  `horizontal = True`; `(n_link,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move, the Tx side is evaluated as a second device;
  default: False
- **`preset`** — Sum-of-sinusoids preset, see [sos.preset](#sospreset); must match [lsf_init_phases](#lsf_init_phases);
  default: `"Comb256"`
- **`horizontal`** — `True`: 2D correlation in the x-y plane (TR 38.901 behaviour); `False`: 3D correlation
  (legacy QuaDRiGa behaviour); must match [lsf_init_phases](#lsf_init_phases); default: True

### Output:
- **`rnd`** — Random values; columns 0 to 8 Normal(0,1), columns 9 and 10 Uniform(0,1); `(n_link, 11)`

### See also:
- [lsf_init_phases](#lsf_init_phases) (draws the phases)
- [path_loss](#path_loss) (column 10 as `rnd_he`)
- [lsp_generate](#lsp_generate) (the large-scale parameters from `rnd`)
- [sos.val](#sosval) (evaluation of a sum-of-sinusoids generator)

---
## lsf_init_phases
Draw the random phases of the large-scale generators of one BS site and scenario

- The large-scale fading (LSF) of a site is described by 11 spatially correlated sum-of-sinusoids generators,
  one column each: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal fields),
  9 ground permittivity, 10 UMa environment height hE (Uniform fields)
- Their decorrelation distances are the scenario parameters `DS_lambda`, `KF_lambda`, `SF_lambda`,
  `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`, `XPR_lambda`, `absTOA_lambda`, `SC_lambda` and
  again `SF_lambda` for hE; a decorrelation distance of 0 is used as 0.1 m. The generator of column 8 exists
  also when the absolute TOA is disabled
- This function draws the phases; [lsf_draw](#lsf_draw) evaluates them at the links. The caller stores `phase` and
  evaluates the same fields again along tracks or on maps
- `scenpar` is completed with the defaults of missing keys and validated as in [scenpar_validate](#scenpar_validate); its
  messages (e.g. for an unknown key or a negative decorrelation distance) pass through unchanged as
  `ValueError`. An invalid value of a key that is not used here raises as well
- Slice 0 (Rx side) holds the phases of the 11 generators. Without bias removal it equals
  `quadriga_lib.sos.init_phases( n_coeff, 11, seed )` of [sos.init_phases](#sosinit_phases), with `n_coeff` the number of
  sinusoids of `preset`. The same seed gives the same phases
- Slice 1 (Tx side) is always the mirror of slice 0:
  `phase[:, :, 1] = phase[:, [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10], 0]`, i.e. ASD and ASA and ESD and ESA are
  exchanged, all others mirror themselves. For dual mobility the departure spreads at the Tx are thus the
  arrival spreads of the Tx as a second device
- Bias removal (`remove_bias = True`, `rx_pos` given, `n_link > 2`): one realisation of a correlated field has
  a non-zero mean over a drop. For each Normal generator (columns 0 to 8) one common phase offset is applied
  with [sos.remove_bias](#sosremove_bias), so that the sample mean of its field over the positions is exactly 0. Single
  mobility: over the Rx positions; `tx_pos` is not used. Dual mobility: over the Rx and Tx positions together
  (`2·n_link` positions), because the field of a generator is shared by both devices (the Tx side of ASD is
  the ASA field); `tx_pos` is then required. The Uniform generators 9 and 10 are never corrected
- The correction is applied to slice 0 and the Tx side is mirrored afterwards, so the mirror relation holds
  exactly with and without bias removal. Swapping `tx_pos` and `rx_pos` in dual mobility gives the same phases
  up to rounding (the sums over the positions are formed in another order)
- Consequences for [lsf_draw](#lsf_draw) at the same positions: in single mobility the link mean of columns 0 to 8 is 0;
  in dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have the same Tx-Rx distance
  (the normalisation is then the same for all links), while ASD/ASA and ESD/ESA have a zero link mean only
  approximately (their two devices see different fields, each corrected over the union of the positions)
- The positions are used as in [lsf_draw](#lsf_draw): projected on the x-y plane (z = 0) for `horizontal = True`, in
  3D for `horizontal = False`. `preset` and `horizontal` must be the same in [lsf_draw](#lsf_draw)
- 2D mode: all positions in `rx_pos` are treated as one field (floor 0): one phase correction per generator
  cannot zero the means of several floor fields at once. For a drop with UTs on several floors pass the
  positions of one floor (e.g. the links of floor 0) or set `remove_bias = False`; the fields of the other
  floors keep their sample mean. 3D mode (`horizontal = False`): there are no floor IDs (the height separates
  the floors) and the correction covers all given positions as one 3D field
- 2D and 3D mode give the same per-link statistics. In 3D mode the LSPs of UTs on different floors are
  correlated by the ACF of their height difference, while TR 38.901 (Table 7.5-6 note 3) assumes no
  correlation between floors
- The correction removes variance: the field is conditioned on a zero mean over the positions, which costs
  little only when the positions spread over many decorrelation distances (see [sos.remove_bias](#sosremove_bias))
- With fewer than 3 positions, without `rx_pos` or with `remove_bias = False` no correction is made
- TR 38.901 clause 7.4.4 literally (exponential ACF, no bias removal): `preset = "Exp256"` and
  `remove_bias = False`; the defaults `Comb256` and the bias removal are QuaDRiGa choices
- Reason for `Comb256`: the TR gives the exponential ACF (eq. 7.4-5) as a fit "with sufficient accuracy",
  not as an exact requirement; its kink at zero distance makes the values evolve roughly along a track.
  `Comb256` (Gaussian near zero, exponential tail, same decorrelation distance) evolves smoothly
- The flags `dual_mobility`, `remove_bias` and `horizontal` take a bool or `numpy.bool_`; `None` selects the
  default, any other type raises `TypeError`
- Validation order: `seed`, `preset`, `scenpar`, then the positions (3 columns, rows of `tx_pos` 1 or
  `n_link` when both are given, finite values), then the requirement of `tx_pos` for the bias removal in
  dual mobility

### Usage:
```
phase = quadriga_lib.gbsm.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, remove_bias, \
    preset, horizontal )
```

### Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; only the 10 decorrelation distances are used
- **`seed`** — Seed of the random phases; int from -1 to 2^63-1; -1 = system random device, a new set of
  phases on each call; default: -1
- **`tx_pos`** — Tx positions in m; used only for the bias removal with dual mobility, required there; one row
  is used for all links; `(1, 3)` or `(n_link, 3)`; default: None
- **`rx_pos`** — Rx positions of the drop in m (one floor in 2D mode, all floors in 3D mode) for the bias
  removal; `None`, empty or fewer than 3 rows = no bias removal; `(n_link, 3)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move, the Tx is a second device on the same maps;
  default: False
- **`remove_bias`** — `True`: remove the sample mean of the Normal fields over the positions; default: True
- **`preset`** — Sum-of-sinusoids preset, see [sos.preset](#sospreset); sets `n_coeff`; default: `"Comb256"`
- **`horizontal`** — `True`: 2D correlation in the x-y plane (TR 38.901 behaviour); `False`: 3D correlation
  (legacy QuaDRiGa behaviour); default: True

### Output:
- **`phase`** — Phases in rad; slice 0 = Rx side, slice 1 = Tx side (mirror of slice 0); all values in
  [-pi, pi); `(n_coeff, 11, 2)`

### See also:
- [conf_file_read](#conf_file_read) (the scenario parameters)
- [lsf_draw](#lsf_draw) (evaluates the phases at the links)
- [sos.init_phases](#sosinit_phases) (the raw phases)
- [sos.remove_bias](#sosremove_bias) (the bias removal of one generator)

---
## lsp_generate
Large-scale parameters of a set of links from normalized random values

- Generates the 8 large-scale parameters (LSPs) of TR 38.901 clause 7.5 step 4 per link and frequency: delay
  spread DS, Ricean K-factor KF, shadow fading SF, azimuth spreads of departure and arrival ASD, ASA, elevation
  spreads of departure and arrival ESD, ESA and cross-polarization ratio XPR. The random values `rnd` are the
  normalized, spatially correlated values of [lsf_draw](#lsf_draw) (columns 0 to 7 Normal(0,1) for the 8 LSPs, column 8
  Normal(0,1) for the absolute TOA, column 9 Uniform(0,1) for the ground permittivity; column 10 is not used here)
- `scenpar` is completed with the defaults of missing keys and validated as in [scenpar_validate](#scenpar_validate); its messages pass
  through unchanged. An invalid value of a key that is not used here raises as well
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
- `sf_sigma` (e.g. from [path_loss](#path_loss)) replaces the SF standard deviation per link and frequency, also in
  `lsp_sigma`; it must be finite and >= 0 (a negative value would flip the sign of the shadow fading)
- Cross-correlation: `x = Q·s` with `s = rnd(l, 0..7)` and a root `Q` of the LSP cross-correlation matrix `R` of
  `scenpar` (`Q·Q^T = R`). `cholesky = true` (TR 38.901 clause 7.5 step 4): Cholesky factorization in the TR order
  [SF, KF, DS, ASD, ASA, ESD, ESA], XPR appended last; the SF value is passed through (`x_SF = s_SF`).
  `cholesky = false`: the symmetric square root of `R` (principal root, legacy QuaDRiGa behaviour); it does not
  depend on the order of the parameters. Both raise `ValueError` if `R` is not positive definite
- The same `x` is used for all frequencies of a link (TR 38.901 clause 7.6.5)
- Linear values: `lsp = 10^(mu + sigma·x)` for DS, ASD, ASA, ESD, ESA; `10^(0.1·(mu + sigma·x))` for KF, SF, XPR.
  `clip_spreads = true` (TR 38.901 clause 7.5 step 4): ASD and ASA are limited to 104 deg, ESD and ESA to 52 deg;
  `lsp_mu` and `lsp_sigma` are not affected
- `rnd = numpy.zeros((n_link, 11))` gives the median LSPs; `lsp_mu` and `lsp_sigma` are the mean and std without any
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
  reflection formulas take the complex conjugate `gr_eps.conj()`. The permittivity is always computed, so
  column 9 of `rnd` must always be in [0, 1]
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
- `n_link = 0` gives outputs with 0 rows
- Validation order: positions, `fc`, `rnd`, `sf_sigma`, `scenpar`, positive definiteness, column 9 of `rnd`, the
  coefficients and frequency terms of the 8 LSPs, the `EoD_offset_` coefficients and `absTOA_sigma`; all errors are
  raised as `ValueError`
- `complex = True` returns the permittivity as one complex array `gr_eps = gr_eps_re + 1j·gr_eps_im`;
  `complex = False` returns the two real arrays `gr_eps_re`, `gr_eps_im` in its place (one more output)
- The flags `cholesky`, `clip_spreads` and `complex` take a bool or `numpy.bool_`; `None` selects the default,
  any other type raises `TypeError`

### Usage:
```
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, fc, \
    tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads, complex=True )

# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( ..., \
    complex=False )
```

### Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; the LSP cross-correlation matrix must be positive definite
- **`fc`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; one row is used for all links; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`rnd`** — Normalized random values from [lsf_draw](#lsf_draw), finite; columns 0 to 7 for the 8 LSPs, column 8 for
  the absolute TOA, column 9 in [0, 1] for the ground permittivity, column 10 not used; `(n_link, 11)`
- **`sf_sigma`** — SF standard deviation in dB, finite and >= 0, e.g. from [path_loss](#path_loss); replaces the value
  from `scenpar`; `None` or empty = the scenario value; `(n_link, n_freq)`; default: None
- **`cholesky`** — `True`: Cholesky root in the TR order; `False`: symmetric square root (legacy);
  default: True
- **`clip_spreads`** — `True`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg; default: True
- **`complex`** — `True`: return the ground permittivity as one complex array `gr_eps`; `False`: as two real
  arrays `gr_eps_re`, `gr_eps_im`; default: True

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in s, 1 KF (linear), 2 SF (linear power ratio), 3 ASD,
  4 ASA, 5 ESD, 6 ESA in deg, 7 XPR (linear); `(n_link, n_freq, 8)`
- **`abs_toa`** — Absolute TOA offset in s, 0 if disabled; `(n_link,)`
- **`gr_eps`** — Relative permittivity of the ground (`complex=True`); imaginary part >= 0; complex128;
  `(n_link, n_freq)`
- **`gr_eps_re`** — Relative permittivity of the ground, real part (`complex=False`); `(n_link, n_freq)`
- **`gr_eps_im`** — Relative permittivity of the ground, imaginary part, >= 0 (`complex=False`);
  `(n_link, n_freq)`
- **`eod_offset_deg`** — EoD offset in deg, positive = up; `(n_link, n_freq)`
- **`lsp_mu`** — Mean in the log domain (lg for DS, ASD, ASA, ESD, ESA; dB for KF, SF, XPR);
  `(n_link, n_freq, 8)`
- **`lsp_sigma`** — Standard deviation in the log domain, >= 0; `(n_link, n_freq, 8)`

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
- The caller draws `rnd` with [sos.rand](#sosrand) and [sos.randn](#sosrandn) at the correlation distances of TR 38.901 clause
  7.6.3.3 (building type 50 m for column 0, d2D-in 25 m for columns 1 and 2, sigma_P 10 m for column 3) and adds
  `loss` to the path loss of the indoor UT. `d2d_in` gives the outdoor distance `d2D - d2D-in` for
  [los_probability](#los_probability) (call `o2i_loss` first) and is the input of the legacy outdoor distance of [path_loss](#path_loss)
- Applicability (Table 7.4.3-2): RMa uses `low` only, InF `high` only; UMa, UMi: `low` and `high`; SMa: `low`,
  `high` and `low_A`. TR maximum indoor distance: 25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and
  SMa residential buildings (conf-file key `O2I_d2D_in_max`). The TR SMa layout has 90 % residential and 10 %
  commercial buildings (Table 7.2-5): split the UTs by building type and call `o2i_loss` once per type
- Validation, in this order: `fc` non-empty, finite and > 0; `rnd` with 4 columns, columns 0 to 2 in [0, 1],
  column 3 finite; `models` non-empty, each a known name; `weights` with one finite element >= 0 per model and a
  sum of 1 within 1e-6; `d2d_in_max` finite and >= 0. All errors are raised as `ValueError`

### Usage:
```
loss, d2d_in, model_index = quadriga_lib.gbsm.o2i_loss( fc, rnd, models, weights, d2d_in_max )
```

### Inputs:
- **`fc`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`rnd`** — Random values per UT: Uniform(0,1) model choice, Uniform(0,1) d2D-in #1, Uniform(0,1) d2D-in #2,
  Normal(0,1) deviation; `(n_ut, 4)`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; list of str or one
  str; `n_model` entries
- **`weights`** — Probability of each model, >= 0, sum 1; `(n_model,)`
- **`d2d_in_max`** — Maximum indoor 2D distance in m, finite and >= 0 (conf-file key `O2I_d2D_in_max`);
  default: 25.0

### Outputs:
- **`loss`** — Penetration loss in dB; `(n_ut, n_freq)`
- **`d2d_in`** — Indoor 2D distance in m, 0 for the car models; `(n_ut,)`
- **`model_index`** — Chosen model per UT, 0-based index into `models`; uint64; `(n_ut,)`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)
- [sos.rand](#sosrand), [sos.randn](#sosrandn) (spatially consistent random values for `rnd`)

---
## path_loss
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier
  frequency. `plpar` is the dict of [conf_file_read](#conf_file_read): its entry `model` is the model (value of `PL_model`), the
  other entries are the `PL_` keys without the prefix; the TR 38.901 v19.5.0 conf files in `data/config`
  reproduce Table 7.4.1-1 with them
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
  - no model (`plpar` is `None`, `{}` or has no entry `model`): `pl = 0`
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
  [lsp_generate](#lsp_generate) is then neutral. `SF_sigma`, `SF_delta`, `SF_kappa`, `SF_tau`, `SF_beta` must be finite, and
  a frequency term with `SF_delta != 0` requires a finite `SF_omega` and `SF_omega + f' > 0`
- Validation: positions, `fc`, the lengths of `d2d_in` and `rnd_he`, their values, the model name, the
  parameters of the model in the order the model reads them (a required key that is missing raises
  `Path-loss model '<model>' requires the parameter '<key>'.`, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every key that the model reads must have exactly one value, which must be finite; keys not read are ignored),
  the breakpoint factors `E`, `E1`, `E2` and `G2n` of `nlos` (> 0) and `A1 != A2` of `winner_los`, then `scenpar`
  ([scenpar_validate](#scenpar_validate), its messages pass through), then the SF coefficients and frequency term; all errors are
  raised as `ValueError`
- The models are evaluated as written for any positive frequency, also outside their validity range

### Usage:
```
pl, sf_sigma = quadriga_lib.gbsm.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he )
```

### Inputs:
- **`plpar`** — Path-loss parameters; dict as returned by [conf_file_read](#conf_file_read): the entry `model` (str) and one
  real scalar or 1D array with one value per further parameter, named without "PL_"; `None`, `{}` or no
  entry `model` = no model, zero loss
- **`scenpar`** — Scenario parameters (SF std terms, `LSP_fc_min`); dict as returned by [conf_file_read](#conf_file_read) or
  [scenpar_validate](#scenpar_validate) (the entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have
  their default value; `None` = all defaults
- **`fc`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; one row is used for all links; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`d2d_in`** — Indoor 2D distance in m of the UT of each link (e.g. from [o2i_loss](#o2i_loss)), finite and >= 0;
  given = legacy mode with the outdoor distance; 0 = outdoor link; `None` or empty = geometric distance
  (TR 38.901); `(n_link,)`; default: None
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 10 of [lsf_draw](#lsf_draw), in
  [0, 1]; `None` or empty = fixed `hE`; `(n_link,)`; default: None

### Outputs:
- **`pl`** — Path loss in dB; `(n_link, n_freq)`
- **`sf_sigma`** — Shadow-fading standard deviation in dB, >= 0 for the default; `(n_link, n_freq)`

### See also:
- [conf_file_read](#conf_file_read) (the model and its parameters)
- [lsf_draw](#lsf_draw) (the random values, column 10 for `rnd_he`)
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss and indoor distance)

---
## scenpar_validate
Validate scenario parameters and complete them with default values

- Checks a dict of QuaDRiGa scenario parameters that did not come from a conf file and adds every
  missing key with its default value, so that `scenpar_out` always has all 172 entries
- `None` or an empty dict returns the complete default parameter set
- A dict with the keys of QuaDRiGa is completed with the 33 keys for 3GPP TR 38.901 v19.5.0, which
  have no effect at their default of 0
- All values are real scalars, except `SubpathMethod`, which is a str
- A key that is not one of the 172 names is an error, also with `validate = False`; this includes
  the QuaDRiGa key `GR_epsilon`, which is not supported
- Value checks with `validate = True`; `NaN` is an error for every key:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`,
    `PerClusterES_D`, `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`,
    `EoD_offset_b`, `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`,
    `NF_alpha`, `NF_beta`, `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`,
    `LOS_state_lambda`, `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`, `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other numeric keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
  - `SubpathMethod` must be `legacy`, `Laplacian` or `mmMAGIC`
- With `validate = False` no value is checked: `NaN` and any `SubpathMethod` are accepted
- The symmetric matrix `lsp_xcorr` is built from the 28 correlation keys and has a unit diagonal.
  It is tested by a Cholesky factorization. A matrix that is not positive definite is not an error.
- Raises `ValueError` for invalid input; `scenpar_in` is not modified

### Usage:
```
scenpar_out, lsp_xcorr, lsp_xcorr_chk = quadriga_lib.gbsm.scenpar_validate( scenpar_in, validate )
```

### Inputs:
- **`scenpar_in`** — Scenario parameters; dict with any subset of the 172 keys returned by
  [conf_file_read](#conf_file_read); numeric values are real scalars (int, float, bool or a numpy value with one
  element); `SubpathMethod` is a str; `None` or `{}` returns the default set; default: None
- **`validate`** — `False`: only complete with defaults, no value checks; bool or `numpy.bool_`;
  default: True; `None` selects the default

### Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing keys have their
  default; dict with 172 entries in the QuaDRiGa order: the 138 numeric keys of QuaDRiGa with
  `SubpathMethod` (str) at position 106 (0-based), then the 33 new keys; all other values are float
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` also if it contains
  `NaN`; bool

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---

# Math functions

---
## cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Computes: length = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / length, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit length); length returns the Euclidean norm
- z / length is clamped to [-1, 1] before asin to guard against length == 0 and rounding artifacts
- Azimuth is in [-pi, pi], elevation in [-pi/2, pi/2]; elevation pi/2 points to the zenith, 0 to the horizon
- Special values of atan2 as in numpy (e.g. atan2(0, -0) = pi); NaN input gives NaN in az, el and length
- Two mutually exclusive input forms: a combined array `cart` of shape (3, n, m), or separate `x`, `y`, `z` of shape (n, m)
- The AVX2 kernel computes the angles in single precision (length and z / length in double precision); use
  `use_kernel = 1` for full double precision

### Usage:
```
geo_coords     = quadriga_lib.tools.cart2geo( cart, y, z, combine, use_kernel )
az, el, length = quadriga_lib.tools.cart2geo( cart, y, z, combine=False )
```

### Inputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; or, when `y` and `z` are given, the x-coordinates `(n, m)`
- **`y`** — Y-coordinates `(n, m)`; provide together with `z` for separate inputs; default: None
- **`z`** — Z-coordinates `(n, m)`; provide together with `y`; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate az, el, length arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws if unavailable); other values raise; default: 1

### Outputs:
- **`geo_coords`** — Combined geographic coordinates `(3, n, m)`; row 0 = azimuth, row 1 = elevation, row 2 = vector length; returned when `combine` is True
- **`az`** — Azimuth angles `(n, m)`; returned when `combine` is False
- **`el`** — Elevation angles `(n, m)`; returned when `combine` is False
- **`length`** — Vector length `(n, m)`; returned when `combine` is False

---
## geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: x = cos(el)·cos(az)·len, y = cos(el)·sin(az)·len, z = sin(el)·len
- Two mutually exclusive input forms: a combined array `az` holding geographic coordinates
  (3, n, m), or separate `az`, `el`, `len` of shape (n, m)
- For combined input, row 0 = azimuth, row 1 = elevation, row 2 = length
- `len` is optional in separate mode; omit it for unit length
- Inverse of `cart2geo`; round-trips in both the combined and separate forms
- The AVX2 kernel computes sine and cosine in single precision (angles reduced in double precision first; max. error
  about 1.3e-7 for angles up to 1e10); use `use_kernel = 1` for full double precision

### Usage:
```
cart    = quadriga_lib.tools.geo2cart( az, el, len, combine, use_kernel )
x, y, z = quadriga_lib.tools.geo2cart( az, el, len, combine=False )
```

### Inputs:
- **`az`** — Azimuth angles `(n, m)`; or, when `el` is omitted, combined geographic coordinates
  `(3, n, m)` (row 0 = azimuth, row 1 = elevation, row 2 = length)
- **`el`** — Elevation angles `(n, m)`; omit for combined input; default: None
- **`len`** — Vector length `(n, m)`; omit for unit length; ignored for combined input; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate x, y, z  arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws
  if unavailable); other values raise; default: 1

### Outputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; row 0 = x, row 1 = y, row 2 = z; returned when `combine` is True
- **`x`** — X-coordinates `(n, m)`; returned when `combine` is False
- **`y`** — Y-coordinates `(n, m)`; returned when `combine` is False
- **`z`** — Z-coordinates `(n, m)`; returned when `combine` is False

---

# Miscellaneous / Tools

---
## components
Returns the version numbers of all quadriga-lib sub-components

### Usage:
```
components = quadriga_lib.components()
```

---
## version
Returns the quadriga-lib version number

### Usage:
```
version = quadriga_lib.version();
```

### Caveat:
- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  an suffix `_AVX2` is added after the version number

---
## write_png
Write data to a PNG file

### Description:
- Converts input data into a color-coded PNG file for visualization
- Support optional selection of a colormap, as well a minimum and maximum value limits
- Uses the <a href="https://github.com/lvandeve/lodepng">LodePNG</a> library for PNG writing
- The auto-detected range ignores `NaN` and `+-inf`; `inf` and `-inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value; also an infinite `min_val` or `max_val`) maps every value to the first colour
- Raises `ValueError` for empty `data` and for `min_val > max_val` (both given)

### Declaration:
```
import quadriga_lib

quadriga_lib.tools.write_png( fn, data, colormap, min_val, max_val, log_transform )
```

### Arguments:
- **`fn`**
  Filename of the PNG file, string, required

- **`data`**
  Data matrix, required, size `[N, M]` 

- **`colormap`** (optional)
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'

- **`min_val`** (optional)
  Minimum value. Values below this value will have be encoded with the color of the smallest value.
  If `NAN` is provided (default), the lowest values is determined from the data.

- **`max_val`** (optional)
  Maximum value. Values above this value will have be encoded with the color of the largest value.
  If `NAN` is provided (default), the largest values is determined from the data.

- `**log_transform**` (optional)
  If enabled, the `data` values are transformed to the log-domain (`10*log10(data)`) before processing.
  Default: false (disabled)

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
  never; see [ray_state_update](#ray_state_update); default: 0
- **`apply_fspl`** — If `True`, folds the geometric spreading loss into `gain`, making it an absolute
  propagation gain instead of a value relative to FSPL. The reference distance is the direct TX-RX
  distance, matching the `path_length` used by [qrt_file_read](#qrt_file_read) for a
  transmission-only path. EM mode: `32.45 + 20*log10(f_GHz) + 20*log10(d_m)` dB; scalar mode:
  `20*log10(d_m) + alpha(f)*d_m` dB, with `alpha` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm.;
  default: `False`
- **`apply_phase`** — If `True`, multiplies `xprmat` by the propagation phase `exp(-j*2*pi*d/lambda)`
  using the same direct TX-RX distance. The wavelength follows from `center_frequency` and the medium
  wave speed (speed of light in EM mode, speed of sound in scalar mode). The term has unit magnitude,
  so `xprmat` stays normalized. This is the phase [get_channels_spherical](#get_channels_spherical) would otherwise add from
  `path_length`; default: `False`

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `(n_pos,)`. Relative to FSPL by
  default (bounded by 0 and 1), absolute propagation gain when `apply_fspl = True`
- **`xprmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase; `gain` holds the power. EM mode:
  interleaved complex, column-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx), the row
  order of `M` in [get_channels_spherical](#get_channels_spherical), `(8, n_pos)`; scalar
  mode: scalar pressure coefficient `[Re Im]`, `(2, n_pos)`
- **`coord`** — Diffracted path coordinates excluding endpoints; `(3, n_seg-1, n_pos)` with `n_seg` from `lod`
  (e.g. `(3, 2, n_pos)` for lod 2, `(3, 9, n_pos)` for lod 7, `(3, 0, n_pos)` for lod 0)

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines the material format)
- [qrt_file_read](#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

---
## cube
Construct a triangulated cube mesh

- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1)
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 · n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with obj_file_write
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- A negative scale mirrors the cube; when the product of the scales is negative, the winding is swapped so that
  the normals stay outward (as Blender's "Apply Scale")

### Usage:
```
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
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Usage:
```
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
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
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
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

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
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
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
- [ray_mesh_interact](#ray_mesh_interact) (calculating interactions between rays and the triangular mesh)

---
## obj_file_write
Write a Wavefront .obj file

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
- The material model and its columns: <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>; the CSV format
  of the material table: <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
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
## plane
Construct a triangulated plane mesh

- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices at +/-1, z = 0)
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 · n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with obj_file_write
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- When the product of the x and y scales is negative (mirrored plane), the winding is swapped so that the normals
  still point in +z direction before the rotation; the z scale is ignored

### Usage:
```
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
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_ind` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Usage:
```
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
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

### Usage:
```
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
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided (0-based), returns the enclosing object as a 1-based index (so 0 stays reserved for "outside")

### Usage:
```
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
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin,
  enabling energy spread simulation
- Reports, for each point, the list of ray indices whose beam intersects that point
- Results use a compressed sparse row layout: `hit_index` holds the ray indices grouped by point and
  `hit_offset` marks where each point's block starts
- The rays hitting point `i` are `hit_index[hit_offset[i]:hit_offset[i+1]]`, a zero-copy numpy view
- `numpy.split(hit_index, hit_offset[1:-1])` produces a per-point list of arrays if that form is needed
- All internal computations use single precision

### Usage:
```
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
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Usage:
```
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
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; real entries in `mesh_index` are
  1-based, with 0 reserved to mark padding entries
- If `mtl_ind` is provided, material indices are reordered and padded in the same way; padding
  entries get index 0

### Usage:
```
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

The functions are in the submodule `quadriga_lib.sos`: [sos.val](#sosval) is called as `quadriga_lib.sos.val`. In the
examples, `np` is numpy.

**Quick start**
[sos.randn](#sosrandn) sets up the generators, draws the phases and returns the values in one call. [sos.rand](#sosrand)
and [sos.randi](#sosrandi) do the same for Uniform and integer values.

```
# 1000 positions; one row per position, columns are (x, y, z) in meters
pos = np.random.default_rng(0).random((1000, 3)) * 100

# One independent generator per decorrelation distance: 10 m and 50 m
dist_decorr = [10, 50]

# Normal distributed values with zero mean and unit variance, shape (1000, 2)
val = quadriga_lib.sos.randn(dist_decorr, pos, seed=42)

# Uniform values between 0 and 1, and integers from 1 to 6
uni = quadriga_lib.sos.rand(dist_decorr, pos, seed=42)
dice = quadriga_lib.sos.randi(dist_decorr, pos, 6, seed=42)
```

The argument `seed` is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field. The functions have their own
random stream; the random state of numpy has no effect on them.

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.
**Keeping a generator**
A generator is two arrays: the frequencies from [sos.preset](#sospreset) and the phases from [sos.init_phases](#sosinit_phases).
[sos.val](#sosval) evaluates them. Each column of the phase array is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```
# Frequencies of a predefined generator for a decorrelation distance of 10 m
freq, acf, dist = quadriga_lib.sos.preset("Comb256", 10)

# Phases of 20 generators; each column is one independent random field
phase = quadriga_lib.sos.init_phases(freq.shape[0], 20, 42)

# Values of all 20 generators at the positions, shape (1000, 20)
val_all = quadriga_lib.sos.val(freq, phase, pos)

# The same fields at other positions, at any later time
pos_new = pos + 2.5
val_new = quadriga_lib.sos.val(freq, phase, pos_new)

# Uniform distributed output
val_uni = quadriga_lib.sos.val(freq, phase, pos, uniform=True)

# Map on a regular grid, shape (n_y, n_x, 1, 20); the first generator is saved as an image
x = np.arange(0, 250.5, 0.5)
y = np.arange(0, 100.5, 0.5)
grid = quadriga_lib.sos.map(freq, phase, x, y)
quadriga_lib.tools.write_png("map.png", grid[:, :, 0, 0])
```

- **Positions:** one row per position, `(n_pos, 3)`. Results have one row per position and one column per
  generator, `(n_pos, n_gen)`. A one-dimensional array is one coordinate per position: `[x, y, z]` is three
  positions on the x-axis; one position is a two-dimensional array with one row.
- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `"Exp512"`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [sos.calc_mse](#soscalc_mse) reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `"Comb256"` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves.

```
# Positions of the two ends of 1000 links; one value per link and generator, shape (1000, 20)
pos_tx = pos
pos_rx = np.random.default_rng(1).random((1000, 3)) * 100
val_link = quadriga_lib.sos.val(freq, phase, pos_tx, pos_rx, acf=acf, dist=dist)
```

- **Mean over an area:** the values at positions that lie within a few decorrelation distances of each
  other are similar, so their average is not zero for a single realisation. This is how a correlated
  field behaves. [sos.remove_bias](#sosremove_bias) can force the average to zero if an application needs it. This
  removes variance: for 200 positions in a square with an edge of 20 decorrelation distances 2.5 %, for
  an edge of 5 decorrelation distances 14 %, for 2 already 44 %. It is meant for positions that spread
  over many decorrelation distances.
- **Coordinates:** use local coordinates. The argument `f_n·x` of a sinusoid is computed in double
  precision and is known to about 1e-16 of its size, so the values lose accuracy with the distance from
  the origin counted in wavelengths. Up to 1e9 wavelengths (more than 1e9 m for a decorrelation distance
  of 10 m) the results stay within the tolerance stated in [sos.val](#sosval).
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [sos.generate](#sosgenerate), which runs on one thread. No result depends on the number of threads.
  [sos.val](#sosval) computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**
[sos.generate](#sosgenerate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos.calc_mse](#soscalc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```
# Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
dist_own = np.arange(200) * 0.125
acf_own = np.exp(-np.square(dist_own / 5))

# 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
freq_own, ase, test_dir = quadriga_lib.sos.generate(acf_own, dist_own, 256, n_dir=300, freq_min=0.01, seed=1)

# Approximation error in dB, larger is better; about 35 dB for this example
mse, mse_core, mse_all, acf_tail, test_dir = quadriga_lib.sos.calc_mse(freq_own, acf_own, dist_own)
```

The result is used like a preset: draw phases with [sos.init_phases](#sosinit_phases) and evaluate with [sos.val](#sosval). To
change the decorrelation distance afterwards, divide the frequencies and multiply the distances by the
same factor.

**Reference**
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
<a target="_blank" rel="noopener noreferrer" href="https://ieeexplore.ieee.org/document/8644265">IEEE Xplore</a> (Original version),
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.pdf">PDF version</a>,
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.md">Markdown version</a>

---
## sos.acf_2d
Target ACF on a 2-D grid

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is at index `2·n_acf - 2`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos.acf_approx](#sosacf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on
  the regular axis

### Usage:
```
acf_2d, dist_2d = quadriga_lib.sos.acf_2d( acf, dist )
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`

### Outputs:
- **`acf_2d`** — ACF on the grid; `(n_grid, n_grid)`
- **`dist_2d`** — Grid axis; `(n_grid,)`

### See also:
- [sos.acf_approx](#sosacf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos.calc_mse](#soscalc_mse) (difference between the two)

---
## sos.acf_approx
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [sos.val](#sosval) over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [sos.acf_2d](#sosacf_2d): from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- The output is always 3-dimensional
- `n_dim = 1`: along the axis; `(n_grid, 1, 1)`
- `n_dim = 2`: on the x-y plane; `(n_grid, n_grid, 1)`; element `[iy, ix, 0]`
- `n_dim = 3`: on the x-y, x-z and y-z plane (index 0, 1, 2 of the last axis); `(n_grid, n_grid, 3)`;
  the row index belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center, exactly
- Equally spaced has the meaning given in [sos.acf_2d](#sosacf_2d)

### Usage:
```
acf_approx, dist_2d = quadriga_lib.sos.acf_approx( freq, dist )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `(n_acf,)`

### Outputs:
- **`acf_approx`** — Approximated ACF; shape depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `(n_grid,)`

### See also:
- [sos.acf_2d](#sosacf_2d) (target ACF on the same grid)
- [sos.calc_mse](#soscalc_mse) (difference between the two)

---
## sos.acf_estimate
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: an array `(2·n_pos, 3)` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [sos.init_phases](#sosinit_phases)
- Bin `n` covers the distances from the midpoint between `bins[n-1]` and `bins[n]` (excluded) to
  the midpoint between `bins[n]` and `bins[n+1]` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins[-1]`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is
  [sos.val](#sosval) with both positions. Two links form a pair if the distance between their first
  devices falls into the bin and the distance between their second devices does not exceed the
  upper limit of the bin; requires `acf` and `dist`
- `acf_est_dual` is computed if `acf` or `dist` is given, and is `None` if both are omitted; giving
  only one of the two is an error
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

### Usage:
```
acf_est, dist_est, acf_est_dual = quadriga_lib.sos.acf_estimate( freq, phase, bins, n_pos, \
    acf, dist, seed )
```

### Inputs:
- **`freq`** — Frequencies; `(n_coeff, 3)`
- **`phase`** — Phases of the generator; `(n_coeff,)` or `(n_coeff, 1)`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `(n_bin,)`
- **`n_pos`** — Number of random positions per device; `int`; must be > 1
- **`acf`** — Target ACF; required for `acf_est_dual`, must then be > -1;
  `(n_acf,)`; default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`;
  default: `None`
- **`seed`** — Seed for the positions; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Outputs:
- **`acf_est`** — Estimated ACF; `(n_bin,)`
- **`dist_est`** — Mean distance of the pairs in each bin; `(n_bin,)`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; `(n_bin,)`; `None` if `acf`
  and `dist` were not given

### See also:
- [sos.acf_approx](#sosacf_approx) (ACF of the sinusoids in closed form)
- [sos.val](#sosval) (evaluation of the generator)

---
## sos.acfi
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [sos.val](#sosval) and [sos.remove_bias](#sosremove_bias) for two devices on the same map

### Usage:
```
val = quadriga_lib.sos.acfi( acf, dist, d )
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`
- **`d`** — Distances; must be >= 0; `(n_d,)`

### Output:
- **`val`** — ACF at the distances `d`; `(n_d,)`

### See also:
- [sos.preset](#sospreset) (ACF tables of the predefined models)

---
## sos.calc_mse
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core`: error on the grid of [sos.acf_approx](#sosacf_approx) against [sos.acf_2d](#sosacf_2d), for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse` equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- All outputs are always computed, including the 2-D grids behind `mse_core` and `mse_all`
- A perfect fit has a mean squared error of 0 and is returned as `inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [sos.acf_2d](#sosacf_2d)

### Usage:
```
mse, mse_core, mse_all, acf_tail, test_dir = quadriga_lib.sos.calc_mse( freq, acf, dist, n_dir )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10

### Outputs:
- **`mse`** — Error along the test directions in [dB]; `float`
- **`mse_core`** — Error within the core radius in [dB]; `float`
- **`mse_all`** — Error on the complete grid in [dB]; `float`
- **`acf_tail`** — Approximated ACF beyond the fitted range; `(4,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

### See also:
- [sos.generate](#sosgenerate) (finds the frequencies; minimises the error returned here)
- [sos.acf_approx](#sosacf_approx), [sos.acf_2d](#sosacf_2d) (the two grids that are compared)

---
## sos.generate
Find the frequencies of a sum-of-sinusoids generator for a given ACF

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = True` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [sos.calc_mse](#soscalc_mse)), `False` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the
  residual between the target and the contribution of all other sinusoids along its own direction:
  coarse search over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The new
  frequency is kept only if the mean squared error over the test directions decreases. The
  iteration stops after a pass that changes nothing or lowers the error by less than a factor of
  1e-6, or after 1000 passes
- `random_init = True` starts from uniformly random frequencies; `False` builds the set up
  sinusoid by sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is
  not controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [sos.calc_mse](#soscalc_mse)). To prevent this, set `freq_min`
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; 0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- The search grid has `( freq_max - freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are
  stored for all sample points. It must not have more than 100000 points (the default has 151),
  which limits `freq_max - freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [sos.preset](#sospreset) were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

### Usage:
```
freq, ase, test_dir = quadriga_lib.sos.generate( acf, dist, n_coeff, n_dim, uniform_smp, n_dir, \
    random_init, freq_min, freq_max, seed )
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_dim`** — Number of dimensions; `int`; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (`True`) or random (`False`) directions of the sinusoids;
  default: `True`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (`True`) or build up sinusoid by sinusoid
  (`False`); default: `True`
- **`freq_min`** — Lower limit of the frequency magnitude; must be >= 0; 0 = none;
  default: 0.0
- **`freq_max`** — Upper limit of the frequency magnitude; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; limited by the size of the search
  grid, see above; default: 0.0
- **`seed`** — Seed; `int`; must be >= -1; -1 seeds from the system random device; default: -1

### Outputs:
- **`freq`** — Frequencies; `(n_coeff, n_dim)`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale;
  `(n_pass,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

### See also:
- [sos.calc_mse](#soscalc_mse) (error of a frequency table)
- [sos.preset](#sospreset) (precomputed tables)

---
## sos.init_phases
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [sos.val](#sosval)
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: 64-bit Mersenne Twister (`std::mt19937_64`) seeded with `seed`; each value is
  `2·pi · ( u - 0.5 )`, where `u` is the upper 53 bits of one generator output divided by 2^53.
  A seed >= 0 gives the same phases on all platforms; the random state of numpy is not used
- The matrix is filled column by column from one stream: for the same seed, column `g` does not
  depend on `n_gen`
- A seed stands for a set of phases. Generators that use phases of the same seed with different
  frequencies (another decorrelation distance or preset) produce related fields, which are equal
  at the coordinate origin and correlated around it; draw independent quantities as columns of
  one call or with different seeds

### Usage:
```
phase = quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_gen`** — Number of generators; `int`; must be > 0; default: 1
- **`seed`** — Seed of the random stream; `int`; must be >= -1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

### Output:
- **`phase`** — Phases; `(n_coeff, n_gen)`

### See also:
- [sos.val](#sosval) (evaluates the generators)
- [sos.preset](#sospreset) (frequencies that belong to the phases)

---
## sos.map
Spatially correlated random values on a regular grid

- Evaluates the generators of [sos.val](#sosval) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `[iy, ix, iz, g]` belongs to the position
  `(x[ix], y[iy], z[iz])` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; a `y` or `z`
  that is `None` or empty means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [sos.val](#sosval) at the grid points, bit for bit; kernels, accuracy and the
  Uniform mapping are described there

### Usage:
```
map = quadriga_lib.sos.map( freq, phase, x, y, z, uniform, use_kernel )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`x`** — Grid points on the x-axis; `(nx,)`
- **`y`** — Grid points on the y-axis; requires `n_dim >= 2`; `None` = one sample at 0, `ny = 1`;
  `(ny,)`; default: `None`
- **`z`** — Grid points on the z-axis; requires `n_dim = 3`; `None` = one sample at 0, `nz = 1`;
  `(nz,)`; default: `None`
- **`uniform`** — Output distribution: `False` = Normal(0,1), `True` = Uniform between 0 and 1;
  default: `False`
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (throws if
  unavailable); `int`; default: 0

### Output:
- **`map`** — Values; `(ny, nx, nz, n_gen)`

### See also:
- [sos.val](#sosval) (evaluation at arbitrary positions, two devices)

---
## sos.preset
Frequencies and ACF of a predefined sum-of-sinusoids generator

- Returns a precomputed frequency table for [sos.val](#sosval) together with the autocorrelation
  function (ACF) it was fitted to
- Presets: "Exp128", "Exp256", "Exp512", "Exp1024", "Gauss128", "Gauss256", "Gauss512",
  "Gauss1024", "Comb128", "Comb256", "Comb512", "Comb1024". The number is the number of
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
freq, acf, dist = quadriga_lib.sos.preset( name, dist_decorr )
```

### Inputs:
- **`name`** — Name of the preset, e.g. "Comb256"; `str`
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10.0

### Outputs:
- **`freq`** — Frequencies; `(n_coeff, 3)`
- **`acf`** — Target ACF; `(200,)`
- **`dist`** — Sample points of the ACF; `(200,)`

### See also:
- [sos.init_phases](#sosinit_phases) (random phases for the generator)
- [sos.val](#sosval) (evaluation)
- [sos.generate](#sosgenerate) (frequencies for other ACFs)

---
## sos.rand
Spatially correlated Uniform distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos.val](#sosval) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
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
  without AVX2 differ by the tolerance stated in [sos.val](#sosval). The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

### Usage:
```
val = quadriga_lib.sos.rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed )
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must be
  >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [sos.preset](#sospreset); `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `(n_pos, n_gen)`

### See also:
- [sos.val](#sosval) (evaluation with given frequencies and phases)
- [sos.preset](#sospreset), [sos.init_phases](#sosinit_phases) (the two steps this function combines)
- [sos.randn](#sosrandn), [sos.randi](#sosrandi) (Normal and integer values)

---
## sos.randi
Spatially correlated integer random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos.val](#sosval) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- The Uniform value `u` of [sos.rand](#sosrand) is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos.val](#sosval), and an integer can then differ by 1
  where the Uniform value lies at a step. The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

### Usage:
```
val = quadriga_lib.sos.randi( dist_decorr, pos, imax, pos_b, dist_decorr_b, preset, seed )
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`imax`** — Largest value; `int`; must be > 0
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must be
  >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [sos.preset](#sospreset); `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability;
  returned as `float64`; one row per position, one column per generator; `(n_pos, n_gen)`

### See also:
- [sos.val](#sosval) (evaluation with given frequencies and phases)
- [sos.preset](#sospreset), [sos.init_phases](#sosinit_phases) (the two steps this function combines)
- [sos.randn](#sosrandn), [sos.rand](#sosrand) (Normal and Uniform values)

---
## sos.randn
Spatially correlated Normal distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos.val](#sosval) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
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
  without AVX2 differ by the tolerance stated in [sos.val](#sosval). The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

### Usage:
```
val = quadriga_lib.sos.randn( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed )
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must be
  >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [sos.preset](#sospreset); `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row
  per position, one column per generator; `(n_pos, n_gen)`

### See also:
- [sos.val](#sosval) (evaluation with given frequencies and phases)
- [sos.preset](#sospreset), [sos.init_phases](#sosinit_phases) (the two steps this function combines)
- [sos.rand](#sosrand), [sos.randi](#sosrandi) (Uniform and integer values)

---
## sos.remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF)
  over all pairs of positions, which is much larger than `1 / n_pos` when the positions lie within
  a few decorrelation distances of each other. This is a property of any correlated field, not an
  error of the generator
- This function adds one offset to all phases of a generator such that the mean of the Normal
  distributed values of [sos.val](#sosval) over the given positions becomes zero
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
- Two devices on the same map need an ACF above -1 at all sample points, as in [sos.val](#sosval)
- The input arrays `phase_in` and `phase_b_in` are not modified; the corrected phases are new arrays
- An optional array that is `None` or empty counts as not given

### Usage:
```
phase_out, phase_b_out, offset, mean_before = quadriga_lib.sos.remove_bias( freq, phase_in, pos, \
    pos_b, phase_b_in, acf, dist, freq_scale_b )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase_in`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; not more columns than `freq`; a 1-D array is one column,
  i.e. one x-coordinate per position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos_b, 1)`, `(n_pos_b, 2)`
  or `(n_pos_b, 3)` as for `pos`; `n_pos_b = n_pos` for the same map, any number for different
  maps; default: `None`
- **`phase_b_in`** — Phases of the second device; requires `pos_b`; `None` = same map;
  `(n_coeff, n_gen)`; default: `None`
- **`acf`** — Target ACF; required for two devices on the same map, must then be > -1; `(n_acf,)`;
  default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`;
  default: `None`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `phase_b_in`;
  default: 1.0

### Outputs:
- **`phase_out`** — Corrected phases; `(n_coeff, n_gen)`
- **`phase_b_out`** — Corrected phases of the second device; `(n_coeff, n_gen)`; `None` if
  `phase_b_in` was not given
- **`offset`** — Offsets added to the phases; `(n_gen, 1)`, or `(n_gen, 2)` with
  `phase_b_in` (column 0: `phase_in`, column 1: `phase_b_in`)
- **`mean_before`** — Sample mean before the correction; `(n_gen, 1)`: mean of the values of
  [sos.val](#sosval) for the given arguments; with `phase_b_in` `(n_gen, 2)`: mean of the field of each
  device on its own positions, the mean of the combined value being their sum divided by `sqrt(2)`

### See also:
- [sos.val](#sosval) (the values whose mean is removed)

---
## sos.val
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of
  `freq`, in cycles per meter) and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function
  (ACF) `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the
  frequencies were fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The
  sines and cosines of the positions are computed once and reused for all generators
- Positions are given as one row per position with the columns `(x, y, z)`; they may have fewer
  columns than `freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and
  `pos_b` and `A` the target ACF interpolated as in [sos.acfi](#sosacfi); `acf` and `dist` are required.
  The value does not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `freq · freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same
  array as `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- Two devices on the same map need an ACF above -1 at all sample points, because the value is
  divided by `sqrt( 2 + 2·A(D) )`
- `uniform = True` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and
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
- An optional array that is `None` or empty counts as not given

### Usage:
```
val = quadriga_lib.sos.val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, \
    uniform, use_kernel )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; not more columns than `freq`; a 1-D array is one column,
  i.e. one x-coordinate per position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`phase_b`** — Phases of the second device; requires `pos_b`; `None` = same map;
  `(n_coeff, n_gen)`; default: `None`
- **`acf`** — Target ACF; required for two devices on the same map, must then be > -1; `(n_acf,)`;
  default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite;
  `(n_acf,)`; default: `None`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; default: 1.0
- **`uniform`** — Output distribution: `False` = Normal(0,1), `True` = Uniform between 0 and 1;
  default: `False`
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (throws if
  unavailable); `int`; default: 0

### Output:
- **`val`** — Values; one row per position, one column per generator; `(n_pos, n_gen)`

### See also:
- [sos.preset](#sospreset) (frequencies and ACF of the predefined models)
- [sos.init_phases](#sosinit_phases) (random phases)
- [sos.map](#sosmap) (evaluation on a grid)
- [sos.randn](#sosrandn) (generator set-up and evaluation in one call)

