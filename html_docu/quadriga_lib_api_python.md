---
title: "Python API Documentation for Quadriga-Lib v0.11.8"
author: "Stephan Jaeckel"
date: "16.06.2026"
lang: en-US
---

<!-- PLACEHOLDER: Python API preamble -->
<!-- Edit this section to add pip install instructions, import conventions, and general usage notes for the Python API. -->

# Function Index

| Function | Section | Line |
| --- | --- | --- |
| [calc_beamwidth](#calc_beamwidth) | Array antenna functions | 71 |
| [calc_directivity](#calc_directivity) | Array antenna functions | 107 |
| [combine_pattern](#combine_pattern) | Array antenna functions | 136 |
| [concat](#concat) | Array antenna functions | 177 |
| [copy_element](#copy_element) | Array antenna functions | 207 |
| [export_obj_file](#export_obj_file) | Array antenna functions | 243 |
| [generate](#generate) | Array antenna functions | 268 |
| [generate_speaker](#generate_speaker) | Array antenna functions | 364 |
| [interpolate](#interpolate) | Array antenna functions | 448 |
| [qdant_read](#qdant_read) | Array antenna functions | 525 |
| [qdant_write](#qdant_write) | Array antenna functions | 565 |
| [rotate_pattern](#rotate_pattern) | Array antenna functions | 603 |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | 653 |
| [channel_export_obj_file](#channel_export_obj_file) | Channel functions | 721 |
| [hdf5_create_file](#hdf5_create_file) | Channel functions | 766 |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | 795 |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | 854 |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | 885 |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | 912 |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | 937 |
| [hdf5_write_channel](#hdf5_write_channel) | Channel functions | 958 |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | 1000 |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | 1030 |
| [qrt_file_read](#qrt_file_read) | Channel functions | 1063 |
| [quantize_delays](#quantize_delays) | Channel functions | 1118 |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | 1180 |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | 1238 |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | 1295 |
| [get_ieee_indoor](#get_ieee_indoor) | Channel generation functions | 1355 |
| [acdf](#acdf) | Channel statistics | 1437 |
| [calc_angular_spread](#calc_angular_spread) | Channel statistics | 1468 |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | 1507 |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | 1557 |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | 1585 |
| [cart2geo](#cart2geo) | Miscellaneous / Tools | 1619 |
| [components](#components) | Miscellaneous / Tools | 1647 |
| [version](#version) | Miscellaneous / Tools | 1656 |
| [write_png](#write_png) | Miscellaneous / Tools | 1669 |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | 1712 |
| [icosphere](#icosphere) | Site-specific simulation tools | 1760 |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Site-specific simulation tools | 1792 |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | 1825 |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | 1875 |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | 1922 |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | 1949 |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | 1979 |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | 2012 |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | 2049 |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | 2092 |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | 2118 |

---

# Array antenna functions

---
## calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern
  (≈100x finer grid in each direction than the antenna sampling grid)
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

### Usage:
```
# Single frequency
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant )
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, freq, azimuth_grid, elevation_grid )

# Multiple frequencies (freq as a 1D array, or a frequency-dependent input antenna)
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, freq, azimuth_grid, elevation_grid )
```

### Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as documented in [generate](#generate)
- **`freq`** — Alternative center frequency in Hz; scalar or 1D array; if given, the pattern is
  recomputed at each value; if `None`, each input entry's `center_freq` is used; a scalar `<= 0`
  is treated as not given; default: `None`
- **`azimuth_grid`** — Alternative output azimuth grid in rad; -pi to pi; sorted; `(n_azimuth_out,)`;
  if `None`, the input grid is used; default: `None`
- **`elevation_grid`** — Alternative output elevation grid in rad; -pi/2 to pi/2; sorted;
  `(n_elevation_out,)`; if `None`, the input grid is used; default: `None`

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
  overrides default 3GPP/ULA element pattern; default: empty dict (use built-in pattern)

### Inputs (multibeam only):
- **`beam_az`** — Azimuth beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_el`** — Elevation beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_weight`** — Per-beam scaling factor; normalized so the sum equals 1; same length as `beam_az`; default: `[1.0, 1.0, ...]` (all-ones)
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
   | `center_freq`    | Center frequency in Hz                          | scalar                                 |
   | `name`           | Name of the array antenna object                | string                                 |

---
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency
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
- **`speaker`** — Multi-frequency arrayant dict with fields:
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
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
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
  (Fortran-contiguous) so they can be read without a copy; raises if not; default: `False`

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
  holds multiple entries, or a plain single-entry dict (3D fields) when it holds exactly one
- Reading all entries is the inverse of `qdant_write` with a 4D-pattern dict

### Usage:
```
# Read a single entry (default: the first entry)
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
  - `layout` — Matrix of element IDs describing how the entries are arranged in the file; uint32

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
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`id`** — Target 1-based ID of the entry inside the file; `0` appends after the highest
  existing ID (or 1 if the file does not exist); ignored for 4D input; default: 0
- **`layout`** — uint32 matrix organizing multiple antenna IDs within the file; must reference
  only IDs present in the file; ignored for 4D input; default: `None`

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
- Internal arithmetic is single-precision; double inputs are narrowed to float, results widened back

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
- **`remove_delay_phase`** — If `True`, undo the delay-induced phase baked in by the channel
  generator before SLERP, then re-apply it analytically per output carrier; multi-freq only;
  default: `True`

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
  `(3, sum(no_interact))`, or an ndarray `(3, max(sum(no_interact)), n_snap)`
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
- **`storage_space`** — Storage dimensions used by the new file, `[nx, ny, nz, nw]`; `(4,)`

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
  | `no_interact`      | Number of interaction points per path; uint32                            | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
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
- The returned type and shape are defined by the dataset's HDF5 dataspace
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
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `(4,)`
- **`has_data`** — Slot occupancy mask; 1 where a slot holds data, 0 otherwise; `(nx, ny, nz, nw)`

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
- A warning is issued if any selected slot already contains data (it is overwritten)
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape
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
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `(4,)`; uint32

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
- If `downlink = True`, origin is TX and destination is RX; if `False`, the roles are swapped
- Per-snapshot outputs are returned as lists with one entry per requested snapshot

### Usage:
```
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, M, aod, eod, \
    aoa, eoa, path_coord, no_int, coord = quadriga_lib.channel.qrt_file_read( fn, cir, orig, downlink, normalize_M )
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`cir`** — Snapshot indices to read; `(n_out,)` or `None`; if `None` or empty, all snapshots are read; default: `None`
- **`orig`** — Origin index (origin = TX for downlink); scalar; default: 0
- **`downlink`** — If `True`, origin=TX and destination=RX; if `False`, the roles are swapped; default: `True`
- **`normalize_M`** — Controls `M` and `path_gain` scaling; 0 = values as stored in the QRT file with
  `path_gain` = -PL; 1 = `M` columns scaled to max power 1 and `path_gain` = -PL minus material losses; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |

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

### See also:
- [generate](#generate) (for generating antenna arrays)
- [get_channels_planar](#get_channels_planar) (for embedding antennas using departure and arrival angles)
- [get_channels_spherical](#get_channels_spherical) (for embedding antennas using FBS/LBS positions)
- [get_channels_multifreq](#get_channels_multifreq) (for multi-frequency antenna embedding)

---
## quantize_delays
Fixes the path delays to a grid of delay bins

### Description:
- For channel emulation with finite delay resolution, path delays must be mapped to a fixed grid
  of delay bins (taps). This function approximates each path delay using two adjacent taps with
  power-weighted coefficients, producing smooth transitions in the frequency domain.
- For a path at fractional offset &delta; between tap indices, two taps are created with complex
  coefficients scaled by (1&minus;&delta;)^&alpha; and &delta;^&alpha;, where &alpha; is the power
  exponent.
- Input delays may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`. Output
  delay shape depends on `fix_taps` mode.
- The number of paths `n_path_s` may differ across snapshots.

### Usage:
```
import quadriga_lib
coeff_re_q, coeff_im_q, delay_q = quadriga_lib.channel.quantize_delays( \
    coeff_re, coeff_im, delay, tap_spacing=5e-9, max_no_taps=48, power_exponent=1.0, fix_taps=0)
```

### Arguments:
- `list ***coeff_re**` (input)
  Channel coefficients, real part. List of `n_snap` numpy arrays, each of shape
  `[n_rx, n_tx, n_path_s]`. The number of paths may differ per snapshot.

- `list ***coeff_im**` (input)
  Channel coefficients, imaginary part. Same shapes as `coeff_re`.

- `list ***delay**` (input)
  Path delays in seconds. List of `n_snap` numpy arrays, each of shape
  `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`.

- `float **tap_spacing** = 5e-9` (input)
  Spacing of the delay bins in seconds.

- `int **max_no_taps** = 48` (input)
  Maximum number of output taps. 0 means unlimited.

- `float **power_exponent** = 1.0` (input)
  Interpolation exponent. Use 1.0 for narrowband or 0.5 for wideband.

- `int **fix_taps** = 0` (input)
  Delay sharing mode: 0 = per tx-rx pair and snapshot, 1 = single grid for all,
  2 = per snapshot, 3 = per tx-rx pair.

### Returns:
- `np.ndarray **coeff_re_q**` (output)
  Output coefficients, real part. 4D array of shape `[n_rx, n_tx, n_taps, n_snap]`.

- `np.ndarray **coeff_im_q**` (output)
  Output coefficients, imaginary part. 4D array of shape `[n_rx, n_tx, n_taps, n_snap]`.

- `np.ndarray **delay_q**` (output)
  Output delays in seconds. 4D array of shape `[n_rx, n_tx, n_taps, n_snap]` or
  `[1, 1, n_taps, n_snap]`.

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
- `M` accepts 8 rows (full polarimetric: ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) or 2 rows (scalar pressure: ReVV, ImVV only)
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`
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
- **`ant_tx`** — Multi-frequency TX arrayant dict; pattern fields 3D `(n_el, n_az, n_elem)` or 4D `(n_el, n_az, n_elem, n_freq)`; see [generate](#generate)
- **`ant_rx`** — RX arrayant dict; same format as `ant_tx`
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`
- **`path_gain`** — Linear-scale path gains, one column per input frequency; `(n_path, n_freq_in)`
- **`path_length`** — Absolute TX-to-RX path lengths; `(n_path,)`
- **`M`** — Polarization transfer matrix, interleaved Re/Im; `(8, n_path, n_freq_in)` full pol (VV, VH, HV, HH)
  or `(2, n_path, n_freq_in)` scalar pressure (ReVV, ImVV only)
- **`tx_pos`** — Transmitter position; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `(n_freq_in,)`
- **`freq_out`** — Target output frequencies; `(n_freq_out,)`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, prepends a zero-power LOS path when none is present; default: `False`
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
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH).
- If `center_freq == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == False`, the straight-line TX-RX delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == True`, a zero-power LOS path is prepended when none is present, making the
  output size `n_path + 1`.
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path.

### Usage:
```
coeff_re, coeff_im, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ant_tx, ant_rx, \
    aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff, delays, rx_Doppler = quadriga_lib.arrayant.get_channels_planar( ..., complex=True )
```

### Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [generate](#generate)
- **`ant_rx`** — Receive arrayant dict; see [generate](#generate)
- **`aod`** — Departure azimuth angles in rad; `(n_path,)`
- **`eod`** — Departure elevation angles in rad; `(n_path,)`
- **`aoa`** — Arrival azimuth angles in rad; `(n_path,)`
- **`eoa`** — Arrival elevation angles in rad; `(n_path,)`
- **`path_gain`** — Path gains in linear scale; `(n_path,)`
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`
- **`M`** — Polarization transfer matrix, interleaved Re/Im; `(8, n_path)` (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`center_freq`** — Center frequency in Hz; set to `0` to skip phase computation; default: `0.0`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, prepends a zero-power LOS path when none is present; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`

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
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH components).
- If `center_freq == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == False`, the minimum delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == True`, a zero-power LOS path is prepended when no LOS path is detected.
- `complex=True` returns one combined complex coefficient array `coeff`; `complex=False` (default) returns
  separate real `coeff_re` and `coeff_im` via a zero-copy fast path

### Usage:
```
coeff_re, coeff_im, delays = quadriga_lib.arrayant.get_channels_spherical( ant_tx, ant_rx, \
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, \
    center_freq, use_absolute_delays, add_fake_los_path )

coeff, delays = quadriga_lib.arrayant.get_channels_spherical( ..., complex=True )

coeff_re, coeff_im, delays, aod, eod, aoa, eoa = quadriga_lib.arrayant.get_channels_spherical( ..., angles=True )
```

### Inputs:
- **`ant_tx`** — Transmit arrayant dict; see [generate](#generate)
- **`ant_rx`** — Receive arrayant dict; see [generate](#generate)
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`
- **`path_gain`** — Path gains in linear scale; `(n_path,)`
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`
- **`M`** — Polarization transfer matrix, interleaved Re/Im; `(8, n_path)` (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`center_freq`** — Center frequency in Hz; set to `0` to skip phase computation; default: `0.0`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, prepends a zero-power LOS path when none is present; default: `False`
- **`angles`** — If `True`, also return departure/arrival angles in antenna-local coordinates; default: `False`
- **`complex`** — If `True`, combine coefficients into a single complex array `coeff`; if `False`, return
  separate `coeff_re` and `coeff_im`; default: `False`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients (`complex=False`); `(n_ports_rx, n_ports_tx, n_path)`
- **`coeff_im`** — Imaginary part of channel coefficients (`complex=False`); same shape as `coeff_re`
- **`coeff`** — Complex channel coefficients (`complex=True`), replaces `coeff_re`/`coeff_im`; same shape
- **`delays`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`aod`** — Azimuth of departure in rad; `(n_ports_rx, n_ports_tx, n_path)`; only when `angles=True`
- **`eod`** — Elevation of departure in rad; same shape; only when `angles=True`
- **`aoa`** — Azimuth of arrival in rad; same shape; only when `angles=True`
- **`eoa`** — Elevation of arrival in rad; same shape; only when `angles=True`

### See also:
- [get_channels_multifreq](#get_channels_multifreq) (multi-frequency extension)
- [get_channels_planar](#get_channels_planar) (planar wave variant)
- [get_channels_irs](#get_channels_irs) (IRS-assisted communication)

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
- Floor floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz
- NAN or negative value for any override parameter restores the model default
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
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency; default: 5.25e9
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** *(optional)* — Channel observation time in seconds; default: 0
- **`update_rate`** *(optional)* — Channel update interval in seconds; relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** *(optional)* — Station speed in km/h; movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h; use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`Dist_m`** *(optional)* — TX-to-RX distance(s); `(n_users,)` or `(1,)`; default: 4.99
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models; `(n_users,)` or `(1,)`; default: 0
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction; default: false
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `(4, n_users)`; default: [] (auto-generate)
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** *(optional)* — RNG seed for repeatability; `-1` uses the system random device; default: -1
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default; default: NAN
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN or negative restores model default; default: NAN
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
  | `center_frequency` | Center Frequency in Hz                                                   | scalar                           | scalar                         |

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
- If `bins` is empty, equally spaced bins spanning the data range are generated

### Usage:
```
cdf_per_set, bins_out, cdf_avg, mu, sig = quadriga_lib.tools.acdf( data, bins, n_bins )
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `(n_samples, n_sets)`
- **`bins`** — Bin centers; used as-is if non-empty; if empty, equally spaced bins spanning the
  data range are generated; `(n_bins_in,)` or empty; if `None` or empty, bins are auto-generated.
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins_in`
  are provided; default: 201

### Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data; `(n_bins_out, n_sets)`
- **`bins_out`** — Auto-generated bins; copy of `bins` when non-empty `bins_in` are provided;
  `n_bins_out = n_bins` or `n_bins_out = n_bins_in`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across data sets; `(n_bins_out,)`
- **`mu`** — Mean of the 0.1–0.9 quantiles across data sets; `(9,)`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across data sets; `(9,)`

---
## calc_angular_spread
Calculate azimuth and elevation angular spreads with spherical wrapping

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
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0.0 (no quantization)

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
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis via the Jones matrix
  transform `M_circ = T · M_lin · T^-1`
- `M` is stored with interleaved real/imaginary parts, 8 rows per path:
  `[Re(M_vv); Im(M_vv); Re(M_vh); Im(M_vh); Re(M_hv); Im(M_hv); Re(M_hh); Im(M_hh)]`
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths
  with `path_length < dTR + window_size` are excluded by default
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
- **`window_size`** — LOS exclusion window; paths within `dTR + window_size` are excluded when
  `include_los = False`; default: 0.01

### Outputs:
- **`xpr`** — XPR on linear scale; `(n_cir, 6)`; columns (0-based):
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_hv)^2)                      |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_vh)^2)                      |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_RL)^2)                   |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_LR)^2)                   |
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
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0.0 (no grouping)

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

# Miscellaneous / Tools

---
## cart2geo
Transform Cartesian (x,y,z) coordinates to Geographic (az, el, length) coordinates

### Description:
This function transforms Cartesian (x,y,z) coordinates to Geographic (azimuth, elevation, length)
coordinates. A geographic coordinate system is a three-dimensional reference system that locates
points on the surface of a sphere. A point has three coordinate values: azimuth, elevation and length
where azimuth and elevation measure angles. In the geographic coordinate system, the elevation angle
θ = 90° points to the zenith and θ = 0° points to the horizon.

### Usage:
```
import quadriga_lib
geo_coords = quadriga_lib.tools.cart2geo(cart_coords)
```

### Input Argument:
- **`cart_coords`**
  Cartesian coordinates (x,y,z), Shape: `(3, n_row, n_col)`

### Output Arguments:
- **`geo_coords`**
  Geographic coordinates, Shape: `(3, n_row, n_col)`
  First row: Azimuth angles in [rad], values between -pi and pi.
  Second row: Elevation angles in [rad], values between -pi/2 and pi/2.
  Third row: Vector length, i.e. the distance from the origin to the point defined by x,y,z.

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
- For a detailed description of the material model see
  <a href="http://quadriga-lib.org/formats.html">Data Formats</a> section

### Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, \
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode )

# Unpacked outputs
gain, coord = quadriga_lib.RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, \
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id, scalar_mode )
```

### Inputs:
- **`orig`** — TX positions; `(n_pos, 3)`
- **`dest`** — RX positions; `(n_pos, 3)`
- **`mesh`** — Triangle vertices, each row `[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]`; `(n_mesh, 9)`
- **`mtl_ind`** — 0-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read)); `(n_mesh,)`
- **`mtl_prop`** — Material properties as a `dict`; each key is one column (the `csv_prop` output of [obj_file_read](#obj_file_read)) mapping to a 1D array of length `n_mtl`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail (0–6), controls `n_path` and `n_seg`; see [generate_diffraction_paths](#generate_diffraction_paths); default: 2
- **`verbose`** — Verbosity level; default: 0
- **`sub_mesh_index`** — 0-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation); `(n_mesh,)` or `None`; default: `None`
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0
- **`scalar_mode`** — If `True`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging; default: `False`

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear scale; `(n_pos,)`
- **`coord`** — Diffracted path coordinates excluding endpoints; `(3, n_seg-1, n_pos)`

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_index`)
- [obj_file_read](#obj_file_read) (defines the material format)

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
- **`radius`** — Radius of icosphere in meters; default: 1
- **`direction_xyz`** *(optional)* — Output directions in Cartesian (true) or spherical
  azimuth/elevation (false); default: false

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `(n_faces, 3)`
- **`length`** — Distance from origin to face plane; equals the magnitude of each
  `center` vector; `(n_faces,)`
- **`vert`** — Vertex offsets from face center `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_faces, 9)`
- **`direction`** — Edge directions; spherical `{az1,el1,az2,el2,az3,el3}` or Cartesian
  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per `direction_xyz` flag; `(n_faces, 6)` or `[n_faces, 9)`

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
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

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
- **`map_to_itu_materials`** *(optional)* — If `true`, maps material names to ITU presets recognised by Sionna RT

### See also:
- [obj_file_read](#obj_file_read) (source for mesh data and BSDF field layout)

---
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

- Parses a triangulated `.obj`; quads and n-gons are rejected
- Parses a triangulated `.obj`; quads and n-gons are rejected. Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- Unmatched names raise an error when `csv_strict = True`; otherwise they map to index 0 (no material)
- Geometry indices (`face_ind`, `obj_ind`) are 0-based; material indices (`mtl_ind`, `csv_ind`) are
  1-based, with 0 reserved for the outside / no-material state
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- For a detailed description of the material model see
  <a href="http://quadriga-lib.org/formats.html">Data Formats</a> section

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
- Without `mtl_ind`, no `usemtl` tags and no `.mtl` file are written
- The `.mtl` file is named after the `.obj` and lists each used material; values default to a gray material when `bsdf` is omitted
- If `csv_names` is given, the EM/acoustic material table is written to a companion `.csv` (named after the
  `.obj`); columns follow a fixed canonical order then any extra `csv_prop` fields (alphabetical);
  `csv_write_defaults` also emits canonical columns absent from `csv_prop`, filled with their defaults
  (`a`, `e`, `fRef` = 1, else 0)
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

### Usage:
```
vert_list_out, face_ind_out = quadriga_lib.RTtools.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, \
    mtl_names, vert_list, face_ind, bsdf, threshold, csv_ind, csv_names, csv_prop, csv_write_defaults )
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

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `(n_vert, 3)`
- **`face_ind_out`** — 0-based face indices derived from `mesh`, or a copy of `face_ind`; `(n_mesh, 3)`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)
- [mitsuba_xml_file_write](#mitsuba_xml_file_write) (for exporting to Mitsuba scene file format)

---
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_index` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Usage:
```
aabb = quadriga_lib.RTtools.point_cloud_aabb( points, sub_cloud_ind, vec_size )
```

### Input Arguments:
- **`points`** — 3D point coordinates; `(n_points, 3)`
- **`sub_cloud_index`** *(optional)* — 0-based row indices marking the start of each sub-cloud;
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
  this value; no padding when `1`; default: 1

### Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `(n_points_out, 3)`
- **`sub_cloud_index`** — 0-based starting index of each sub-cloud within `points_out`; uint32; `(n_sub,)`
- **`forward_index`** *(optional)* — 1-based index map from `points` to `points_out`; padding entries are `0`; uint32; `(n_points_out,)`
- **`reverse_index`** *(optional)* — 0-based index map from `points_out` back to `points`; uint32; `(n_points,)`

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
  index (with `obj_ind`); uint32; size `(n_points,)`

See also:
- [obj_file_read](#obj_file_read) (for reading `mesh` and `obj_ind` from an .obj file)

---
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin, enabling energy spread simulation
- Returns, for each point, the list of ray indices whose beam intersects that point
- All internal computations use single precision

### Usage:
```
hit_count, ray_ind = quadriga_lib.RTtools.ray_point_intersect( orig, trivec, tridir, points, sub_cloud_ind, use_kernel, gpu_id )
```

### Inputs:
- **`orig`** — Ray origin positions in global Cartesian coordinates; `(n_ray, 3)`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `(n_ray, 9)`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `(n_ray, 9)`
- **`points`** — 3D point cloud coordinates; `(n_points, 3)`
- **`sub_cloud_index`** *(optional)* — 0-based segment boundary indices for the point cloud
  (see `quadriga_lib.point_cloud_segmentation`); uint32; `(n_sub,)`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2,
  3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_points >= 10000` and CUDA is
  available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA; default: 0

### Outputs:
- **`hit_count`** — Number of beams intersecting each point; uint32; `(n_points,)`
- **`ray_ind`** — List of length `(n_points,)`; each list entry is a 1-D array of 0-based ray indices
  that hit that point. Entries may be empty if no hit was detected; uint32

### See also:
- [icosphere](#icosphere) (for generating beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (for generating point cloud segments)
- [ray_triangle_intersect](#ray_triangle_intersect) (for calculating intersection of rays and triangles)

---
## ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- Internal computations always use single precision for AVX2 and CUDA kernels; only GENERIC has `double` support

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
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `(n_mesh, 9)`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; 0-based; uint32; `(n_sub)`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row:
  `{x_min x_max y_min y_max z_min z_max}`; if empty or omitted, AABBs are computed from `mesh`; `(n_sub, 6)`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA;
  throws if unavailable; auto mode selects CUDA when `n_ray >= 10000` and CUDA is available, else AVX2,
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
- **`triangles`** — Triangle mesh vertices in global Cartesian coordinates; `(n_triangles, 9)`
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
- **`mesh_index`** — 1-based mapping from original to reorganized mesh (0 = padding); uint32; `(n_triangles_out,)`
- **`mtl_ind_out`** — Reordered and padded material indices; `(n_triangles_out,)`; empty if `mtl_ind` is not provided

