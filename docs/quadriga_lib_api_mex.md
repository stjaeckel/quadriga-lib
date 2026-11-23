---
title: "MATLAB / Octave API Documentation for Quadriga-Lib v0.12.5"
author: "Stephan Jaeckel"
lang: en-US
---

<a id="sec-general-usage-notes"></a>
# General usage notes

- Each function has a 1-line short description, optional detailed notes, a Usage block, and Inputs/Outputs sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- The conventions and shared quantities of the C++ [Glossary](quadriga_lib_api_cpp.md#glossary) apply; the points below are specific to MATLAB / Octave.
- All functions live in the `+quadriga_lib` package and are called as `quadriga_lib.function_name(...)`.
- Numeric inputs accept any numeric type and are cast internally; default is `double` unless stated otherwise (e.g. `uint64` for some index arguments).
- Arguments have the names, order and defaults of the C++ function unless the function states otherwise.
- Inputs with a stated default, or marked *(optional)*, can be omitted or passed as `[]`; all others are required. Output arguments are optional via `nargout`; a call without assignment returns the first output as `ans`, and outputs after the first are returned only when requested.
- Invalid inputs (shape/domain) and problems with a file's access or content raise a MATLAB error with the identifier `quadriga_lib:CPPerror`.
- Index conventions: 1-based unless the field is explicitly called 0-based.
- The canonical arrayant struct is defined in [generate_antenna](#generate_antenna). Mandatory fields: `e_theta_re/im`, `e_phi_re/im`, `azimuth_grid`, `elevation_grid`. Optional field defaults: `element_pos = zeros(3, n_elements)`, identity port weights, `center_freq = 299792458`, empty `name`. A struct array represents a frequency-dependent (multi-frequency) model with one entry per frequency. Each call builds a new arrayant from the struct, so its data is checked in full on every call.

# Function Index

| Function | Section | Summary |
| --- | --- | --- |
| [apply_port_weights](#apply_port_weights) | Antennas, transducers and patterns | Apply the port weights of an antenna array: one pattern per port |
| [calc_beamwidth](#calc_beamwidth) | Antennas, transducers and patterns | Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees |
| [calc_directivity](#calc_directivity) | Antennas, transducers and patterns | Calculate the directivity in dBi of the elements of an antenna array |
| [copy_element](#copy_element) | Antennas, transducers and patterns | Copy a single element of an antenna array to one or more destination slots |
| [export_pattern_obj](#export_pattern_obj) | Antennas, transducers and patterns | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [generate_antenna](#generate_antenna) | Antennas, transducers and patterns | Generate an antenna array of a given type |
| [generate_speaker](#generate_speaker) | Antennas, transducers and patterns | Generate a parametric frequency-dependent loudspeaker directivity model |
| [interpolate_pattern](#interpolate_pattern) | Antennas, transducers and patterns | Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles |
| [join_arrays](#join_arrays) | Antennas, transducers and patterns | Join the elements of two antenna arrays into a new array |
| [qdant_read](#qdant_read) | Antennas, transducers and patterns | Read arrayant objects from a QDANT file |
| [qdant_write](#qdant_write) | Antennas, transducers and patterns | Write arrayant data to a QDANT (XML) file |
| [rotate_pattern](#rotate_pattern) | Antennas, transducers and patterns | Rotate the patterns of an antenna array using Euler rotations |
| [baseband_freq_response](#baseband_freq_response) | Channel functions | Compute the baseband frequency response of a MIMO channel |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Channel functions | Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients |
| [channel_export_obj_file](#channel_export_obj_file) | Channel functions | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [hdf5_create_file](#hdf5_create_file) | Channel functions | Create a new HDF5 channel file with a custom storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Channel functions | Read one or more channel objects from an HDF5 file |
| [hdf5_read_dset](#hdf5_read_dset) | Channel functions | Read a single unstructured dataset from an HDF5 file |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel functions | Read names of unstructured datasets stored at a 4D slot in an HDF5 file |
| [hdf5_read_layout](#hdf5_read_layout) | Channel functions | Read the storage layout of channel data inside an HDF5 file |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel functions | Reshape the storage layout inside an existing HDF5 file |
| [hdf5_write_channel](#hdf5_write_channel) | Channel functions | Write one or more channel objects to an HDF5 file |
| [hdf5_write_dset](#hdf5_write_dset) | Channel functions | Write a single unstructured dataset to an HDF5 file |
| [qrt_file_parse](#qrt_file_parse) | Channel functions | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Channel functions | Read ray-tracing CIR data from a QRT file |
| [quantize_delays](#quantize_delays) | Channel functions | Map path delays to a fixed tap grid using two-tap power-weighted interpolation |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generation functions | Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models |
| [get_channels_irs](#get_channels_irs) | Channel generation functions | Calculate MIMO channel coefficients for IRS-assisted communication |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generation functions | Compute channel coefficients for spherical waves across multiple frequencies |
| [get_channels_planar](#get_channels_planar) | Channel generation functions | Calculate MIMO channel coefficients for planar wave paths |
| [get_channels_spherical](#get_channels_spherical) | Channel generation functions | Calculate MIMO channel coefficients and delays for spherical wave propagation |
| [acdf](#acdf) | Channel statistics | Calculate the empirical averaged cumulative distribution function (CDF) |
| [calc_angular_spread](#calc_angular_spread) | Channel statistics | Calculate azimuth and elevation angular spreads, optionally with spherical wrapping |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Channel statistics | Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases |
| [calc_delay_spread](#calc_delay_spread) | Channel statistics | Calculates RMS delay spread from per-CIR delays and linear-scale powers |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel statistics | Calculate the Rician K-Factor from channel impulse response data |
| [conf_file_list](#conf_file_list) | Geometry-based stochastic model | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Geometry-based stochastic model | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Geometry-based stochastic model | Write a QuaDRiGa scenario configuration file |
| [gbsm_draw](#gbsm_draw) | Geometry-based stochastic model | Spatially correlated normalized random values of one random set at a set of links |
| [gbsm_init_phases](#gbsm_init_phases) | Geometry-based stochastic model | SOS phases of one random set of a BS site and scenario |
| [gbsm_rays_from_paths](#gbsm_rays_from_paths) | Geometry-based stochastic model | Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read |
| [gbsm_single_link_channel](#gbsm_single_link_channel) | Geometry-based stochastic model | Channel coefficients of one link at one frequency over one segment |
| [los_probability](#los_probability) | Geometry-based stochastic model | LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models |
| [lsp_generate](#lsp_generate) | Geometry-based stochastic model | Large-scale parameters of a set of links from normalized random values |
| [o2i_loss](#o2i_loss) | Geometry-based stochastic model | O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3 |
| [path_loss](#path_loss) | Geometry-based stochastic model | Path loss and shadow-fading standard deviation of a set of links |
| [scenpar_validate](#scenpar_validate) | Geometry-based stochastic model | Validate scenario parameters and complete them with default values |
| [ssf_clusters](#ssf_clusters) | Geometry-based stochastic model | Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7) |
| [ssf_paths](#ssf_paths) | Geometry-based stochastic model | Antenna-free paths of one link at one position |
| [ssf_rays](#ssf_rays) | Geometry-based stochastic model | Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16) |
| [ssf_scatterers](#ssf_scatterers) | Geometry-based stochastic model | First- and last-bounce scatterers of the rays of one link (two-bounce solver) |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | Calculate rotation matrices from Euler angles |
| [fast_cart2geo](#fast_cart2geo) | Math functions | Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length |
| [fast_geo2cart](#fast_geo2cart) | Math functions | Convert elementwise azimuth/elevation angles to Cartesian coordinates |
| [fast_sincos](#fast_sincos) | Math functions | Compute elementwise approximate sine and/or cosine of a vector |
| [interp](#interp) | Math functions | Perform linear interpolation (1D or 2D) on single or multiple data sets |
| [version](#version) | Miscellaneous / Tools | Returns the quadriga-lib version number |
| [write_png](#write_png) | Miscellaneous / Tools | Write a data matrix to a color-coded PNG file |
| [calc_diffraction_gain](#calc_diffraction_gain) | Site-specific simulation tools | Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh |
| [cube](#cube) | Site-specific simulation tools | Construct a triangulated cube mesh |
| [generate_diffraction_paths](#generate_diffraction_paths) | Site-specific simulation tools | Generate elliptic propagation paths and weights for diffraction gain estimation |
| [icosphere](#icosphere) | Site-specific simulation tools | Construct a geodesic polyhedron from recursive icosahedron subdivision |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Site-specific simulation tools | Write a triangular mesh to a Mitsuba 3 XML scene file |
| [obj_file_read](#obj_file_read) | Site-specific simulation tools | Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials |
| [obj_file_write](#obj_file_write) | Site-specific simulation tools | Write a Wavefront .obj file |
| [plane](#plane) | Site-specific simulation tools | Construct a triangulated plane mesh |
| [point_cloud_aabb](#point_cloud_aabb) | Site-specific simulation tools | Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud |
| [point_cloud_segmentation](#point_cloud_segmentation) | Site-specific simulation tools | Reorganize a point cloud into spatial sub-clouds for efficient processing |
| [point_inside_mesh](#point_inside_mesh) | Site-specific simulation tools | Test whether 3D points are inside a triangle mesh using raycasting |
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces |
| [ray_point_intersect](#ray_point_intersect) | Site-specific simulation tools | Calculate intersections of ray beams with points in 3D space |
| [ray_triangle_intersect](#ray_triangle_intersect) | Site-specific simulation tools | Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm |
| [subdivide_triangles](#subdivide_triangles) | Site-specific simulation tools | Subdivide triangles into smaller triangles |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Site-specific simulation tools | Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Site-specific simulation tools | Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing |
| [sos_acf_2d](#sos_acf_2d) | Sum-of-sinusoids random number generator | Target ACF on a 2-D grid |
| [sos_acf_approx](#sos_acf_approx) | Sum-of-sinusoids random number generator | ACF produced by a set of sinusoids |
| [sos_acf_estimate](#sos_acf_estimate) | Sum-of-sinusoids random number generator | Monte-Carlo estimate of the ACF of one generator |
| [sos_acfi](#sos_acfi) | Sum-of-sinusoids random number generator | Target ACF at given distances |
| [sos_calc_mse](#sos_calc_mse) | Sum-of-sinusoids random number generator | Approximation error of the ACF of a set of sinusoids |
| [sos_generate](#sos_generate) | Sum-of-sinusoids random number generator | Find the spatial frequencies of a sum-of-sinusoids generator for a given ACF |
| [sos_init_phases](#sos_init_phases) | Sum-of-sinusoids random number generator | Draw random phases for a set of sum-of-sinusoids generators |
| [sos_map](#sos_map) | Sum-of-sinusoids random number generator | Spatially correlated random values on a regular grid |
| [sos_preset](#sos_preset) | Sum-of-sinusoids random number generator | Spatial frequencies and ACF of a predefined sum-of-sinusoids generator |
| [sos_rand](#sos_rand) | Sum-of-sinusoids random number generator | Spatially correlated Uniform distributed random numbers |
| [sos_randi](#sos_randi) | Sum-of-sinusoids random number generator | Spatially correlated integer random numbers |
| [sos_randn](#sos_randn) | Sum-of-sinusoids random number generator | Spatially correlated Normal distributed random numbers |
| [sos_remove_bias](#sos_remove_bias) | Sum-of-sinusoids random number generator | Remove the sample mean of sum-of-sinusoids generators over a set of positions |
| [sos_val](#sos_val) | Sum-of-sinusoids random number generator | Spatially correlated random values at given positions |

---

<a id="sec-antennas-transducers-and-patterns"></a>
# Antennas, transducers and patterns

The functions of this section create, inspect, transform and store antenna and acoustic arrays. They share one data
structure, the arrayant struct (fields in [generate_antenna](#generate_antenna)): complex polarimetric patterns of the elements on an
azimuth and elevation grid, the element positions and the port weights. The channel functions take it as transmit and
receive antenna.

**Terms** (full list in the [Glossary](quadriga_lib_api_cpp.md#glossary)):

- **Element**: one radiator or sensor (antenna element, loudspeaker driver, microphone capsule) with a pattern, a
  position and one row of the port weights
- **Grid**: azimuth φ in [-π, π], measured from +x toward +y, and elevation θ in [-π/2, π/2], measured from the x-y
  plane (TR 38.901 uses the zenith angle 90° - θ), both in rad and strictly increasing. Boresight is +x
- **Pattern**: the direction-dependent far field of an element: F_θ (`e_theta_re/im`) along
  `(sin θ cos φ, sin θ sin φ, -cos θ)`, the zenith-angle unit vector of TR 38.901, which points toward decreasing
  elevation, and F_φ (`e_phi_re/im`) along `(-sin φ, cos φ, 0)`. For an antenna, `|F_θ|² + |F_φ|²` is the linear gain
  over an isotropic radiator
- **Port**: one signal of the array, one column of the port weights
- **Port weights**: the fixed complex feed network `w(i,p)` from element i to port p, `[n_elements, n_ports]`. It is not
  the mutual coupling between elements. Empty means identity (one port per element)
- **Element position**: local Cartesian coordinates in m, `[3, n_elements]`; empty means all at the origin
- **Multi-frequency array**: a 1xN struct array, one entry per `center_freq`, with equal grids, element positions and
  port weight sizes. [interpolate_pattern](#interpolate_pattern) and [apply_port_weights](#apply_port_weights) interpolate over frequency between the entries

The field of port p in a direction is the weighted sum of the element fields, each with the phase of its position,

`F_p = Σ_i w(i,p) · F_i · exp(-j 2π f d_i / c)`

where `d_i = -r · x_i` is the distance of element i at position x_i from the wavefront plane through the origin,
negative for elements closer to the source, r the unit vector toward the source, f the frequency and c the propagation
speed.

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Create | [generate_antenna](#generate_antenna), [generate_speaker](#generate_speaker), [qdant_read](#qdant_read) |
| Evaluate | [interpolate_pattern](#interpolate_pattern), [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) |
| Transform | [apply_port_weights](#apply_port_weights), [rotate_pattern](#rotate_pattern) |
| Edit and combine | [join_arrays](#join_arrays), [copy_element](#copy_element) |
| Store, show | [qdant_write](#qdant_write), [export_pattern_obj](#export_pattern_obj) |

Arrays are stored in the [QDANT file format](formats/qdant.md).

---
<a id="apply_port_weights"></a>
## apply_port_weights
Apply the port weights of an antenna array: one pattern per port

Also in: [C++](quadriga_lib_api_cpp.md#apply_port_weights) · [Python](quadriga_lib_api_python.md#apply_port_weights)

- Combines the elements of an array into one pattern per port, for beamforming and to speed up MIMO channel computation
- Each port pattern is the weighted sum of the element fields from the section overview, evaluated on the output grid
  and including the phase of each element position
- The result has one element per port and identity port weights; its elements are at the origin, or at the centers
  of their ports with `center_ports`

### Usage:
```matlab
arrayant_out = quadriga_lib.apply_port_weights( arrayant_in, freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed, center_ports );
```

### Inputs:
- **`arrayant_in`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array
- **`freq_new`** — Output frequencies in Hz, positive and finite; a 1x1 struct has one pattern, so that only the phase
  of the element positions follows `freq_new`; `[n_freq_new]`; default: the `center_freq` of the entries, one output per
  entry
- **`azimuth_grid_new`** — Output azimuth grid in rad; default: the input grid (of the first entry)
- **`elevation_grid_new`** — Output elevation grid in rad; default: the input grid (of the first entry)
- **`propagation_speed`** — Wave propagation speed in m/s, positive and finite; 299792458 is the speed of light for
  radio waves, use about 343 for acoustic arrays (e.g. from [generate_speaker](#generate_speaker)); default: 299792458
- **`center_ports`** — If `true`, places each port at the power-weighted center of its elements,
  `sum(|w|² x) / sum(|w|²)`, with the phase referenced there; a port with zero weights stays at the origin. A
  plane-wave channel is unchanged, and the slower phase over angle makes interpolation of coupled sub-arrays more
  accurate; default: `false`

### Output:
- **`arrayant_out`** — Arrayant struct with `n_ports` elements, each holding the pattern of that port; `name` is kept,
  `center_freq` is the output frequency; a 1x1 struct for one output frequency, otherwise a 1xN struct array with one
  entry per output frequency

### Notes:
- **Frequency interpolation** (struct array): the fields are interpolated over frequency as in [interpolate_pattern](#interpolate_pattern),
  the port weights spherically between the two entries that bracket the output frequency. Output frequencies outside the
  entries take the nearest entry. The position phase uses the output frequency

### See also:
- [generate_antenna](#generate_antenna) (creates arrays with port weights)
- [interpolate_pattern](#interpolate_pattern) (evaluates the element patterns without the port weights)
- [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) (work per element, so they measure the port patterns of the result)
- [rotate_pattern](#rotate_pattern) (turns the combined pattern)

---
<a id="calc_beamwidth"></a>
## calc_beamwidth
Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees

Also in: [C++](quadriga_lib_api_cpp.md#calc_beamwidth) · [Python](quadriga_lib_api_python.md#calc_beamwidth)

- Measures the width of the main lobe of each element in azimuth and elevation at a threshold below its peak, and the
  direction of the peak
- Calculated per element, not per port: the port weights are ignored

### Usage:
```matlab
[ beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg ] = ...
    quadriga_lib.calc_beamwidth( arrayant, i_element, threshold_dB );
```

### Inputs:
- **`arrayant`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array with `n_freq`
  entries
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements
- **`threshold_dB`** — Threshold below the peak in dB, positive and finite; 3 dB is the half-power beamwidth;
  loudspeakers state the coverage angle between the -6 dB points, use 6; default: 3

### Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degrees; `[n_out, n_freq]`
- **`beamwidth_el_deg`** — Elevation beamwidth in degrees; `[n_out, n_freq]`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degrees; `[n_out, n_freq]`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degrees; `[n_out, n_freq]`

### Notes:
- **Main lobe**: starting at the maximum, the search goes outward until the power drops below the threshold; side lobes
  above the threshold are not included
- **Resolution**: the cuts are interpolated as in [interpolate_pattern](#interpolate_pattern), about 100 times finer than the sampling grid
- **Azimuth cut**: at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360°
  when the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- **Elevation cut**: along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°

### See also:
- [calc_directivity](#calc_directivity) (directivity in dBi of the elements)
- [apply_port_weights](#apply_port_weights) (combines the elements into ports, to measure the beams of the ports)
- [generate_antenna](#generate_antenna) (creates elements with a given beamwidth)

---
<a id="calc_directivity"></a>
## calc_directivity
Calculate the directivity in dBi of the elements of an antenna array

Also in: [C++](quadriga_lib_api_cpp.md#calc_directivity) · [Python](quadriga_lib_api_python.md#calc_directivity)

- Measures how strongly an element concentrates its power: `D = 10 log10(P_max / P_mean)` with the peak radiation
  intensity `P_max` and its mean `P_mean` over the sphere; an isotropic radiator has 0 dBi
- Calculated per element, not per port: the port weights are ignored

### Usage:
```matlab
directivity_dBi = quadriga_lib.calc_directivity( arrayant, i_element );
```

### Inputs:
- **`arrayant`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array with `n_freq`
  entries
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements

### Output:
- **`directivity_dBi`** — Directivity in dBi; `[n_out, n_freq]`

### Notes:
- **Scale**: the result does not depend on the scale of the pattern; an element whose pattern is zero everywhere gives 0
  dBi

### See also:
- [calc_beamwidth](#calc_beamwidth) (beamwidth and pointing angles of the elements)
- [apply_port_weights](#apply_port_weights) (combines the elements into ports, to measure the directivity of the ports)

---
<a id="copy_element"></a>
## copy_element
Copy a single element of an antenna array to one or more destination slots

Also in: [C++](quadriga_lib_api_cpp.md#copy_element) · [Python](quadriga_lib_api_python.md#copy_element)

- Copies the pattern and the position of an element, e.g. to build an array from a single element; the port weights of
  existing elements do not change
- A destination beyond the last element enlarges the array. Each added element gets its own new port with weight 1;
  added elements that are not a destination have a zero pattern and position

### Usage:
```matlab
arrayant_out = quadriga_lib.copy_element( arrayant_in, i_source, i_target );
```

### Inputs:
- **`arrayant_in`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array; the same
  copy is made in every entry
- **`i_source`** — Index of the element to copy, 1-based, at most `n_elements`; scalar, or `[n_copy]` for pairwise
  copies from `i_source(k)` to `i_target(k)`
- **`i_target`** — Destination indices, 1-based, not empty; a scalar `i_source` is copied to each of them, a vector
  `i_source` needs the same length; `[n_copy]`

### Output:
- **`arrayant_out`** — Arrayant with the copies, in the form of `arrayant_in`

### See also:
- [join_arrays](#join_arrays) (joins two arrays)
- [generate_antenna](#generate_antenna) (creates the element to copy)
- [rotate_pattern](#rotate_pattern) (turns single copies, through `i_element`)

---
<a id="export_pattern_obj"></a>
## export_pattern_obj
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [C++](quadriga_lib_api_cpp.md#export_pattern_obj) · [Python](quadriga_lib_api_python.md#export_pattern_obj)

- Writes the power pattern of each element as a colored 3D surface, one object per element, placed at its position in
  `element_pos`
- The distance of the surface from the element is linear in the level in dB: `object_radius` at the peak over all
  exported elements, 0 at `directivity_range_dB` below it. The color shows the same level

### Usage:
```matlab
quadriga_lib.export_pattern_obj( fn, arrayant, directivity_range_dB, colormap, object_radius, ...
    icosphere_n_div, i_element, i_freq );
```

### Inputs:
- **`fn`** — Output OBJ file name, must end in `.obj`; the colormap is written to a `.mtl` file of the same name;
  existing files are overwritten
- **`arrayant`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array; the exported
  elements must have finite field values and must not be zero in all directions
- **`directivity_range_dB`** — Range of the plot below the peak in dB, finite and not 0; the sign is ignored; default:
  30
- **`colormap`** — Colormap name: `jet`, `parula`, `winter`, `hot`, `turbo`, `copper`, `spring`, `cool`, `gray`,
  `autumn` or `summer`; default: `jet`
- **`object_radius`** — Radius of the surface at the peak in m, positive and finite; default: 1
- **`icosphere_n_div`** — Subdivisions n of the [icosphere](#icosphere) that carries the surface; higher = finer mesh with 20 n²
  faces; default: 4
- **`i_element`** — Element indices to export, 1-based; default: all elements
- **`i_freq`** — Entry of a multi-frequency array to export, 1-based, at most `numel(arrayant)`; default: 1

### See also:
- [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) (the same pattern as numbers)
- [channel_export_obj_file](#channel_export_obj_file) (exports propagation paths to an OBJ file)

---
<a id="generate_antenna"></a>
## generate_antenna
Generate an antenna array of a given type

Also in: [C++](quadriga_lib_api_cpp.md#generate_antenna) · [Python](quadriga_lib_api_python.md#generate_antenna)

- Creates every built-in antenna: single elements, uniform linear arrays, 3GPP TR 38.901 panel arrays, multibeam arrays,
  and the handheld UT and CPE devices. `type` selects the antenna; the other inputs set its parameters
- The result holds one pattern per element, the element positions and the port weights. Ports stay separate from the
  elements unless `apply_weights` is set; [apply_port_weights](#apply_port_weights) combines them later
- Panel inputs use the TR notation: `P` polarizations, `M` rows, `N` columns, `Mg` and `Ng` panels, and `M_P`, `N_P`
  ports per panel and polarization
- Tuple inputs may be partial: entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their
  default, e.g. `panel = [4, 8]` is one panel of 4 rows and 8 columns
- An input that the selected type does not take raises an error

### Usage:
```matlab
% omni, xpol, dipole, half-wave-dipole
ant = quadriga_lib.generate_antenna( type, res, center_freq );

% Custom-shape element
ant = quadriga_lib.generate_antenna( 'custom', res, center_freq, element );

% Uniform linear array (omni element) and TR 38.901 panel array
ant = quadriga_lib.generate_antenna( 'ula', res, center_freq, [], panel, spacing, pol, ports, steer, ...
    [], [], [], pattern, [], apply_weights );
ant = quadriga_lib.generate_antenna( '3GPP', res, center_freq, element, panel, spacing, pol, ports, steer, ...
    [], [], [], pattern, [], apply_weights );

% Handheld UT and CPE
ant = quadriga_lib.generate_antenna( '3GPP-UT', res, center_freq, element, [], [], pol, [], [], ...
    [], device_size, locations, pattern );

% Multi-beam panel array
ant = quadriga_lib.generate_antenna( 'multibeam', res, center_freq, element, panel, spacing, pol, [], [], ...
    beams, [], [], [], separate_beams, apply_weights );
```

### Inputs:
- **`type`** — Antenna type:
   | `type`                       | Antenna                                                                    | Inputs besides `res`, `center_freq`                                                |
   | ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
   | `omni`                       | Isotropic element, vertical polarization                                   | none                                                                               |
   | `xpol`                       | Two isotropic elements, vertical and horizontal polarization               | none                                                                               |
   | `dipole` (or `short-dipole`) | Short dipole, vertical polarization                                        | none                                                                               |
   | `half-wave-dipole`           | Half-wave dipole, vertical polarization                                    | none                                                                               |
   | `custom`                     | One element with given 3 dB beamwidths, peak gain equal to its directivity | `element`                                                                          |
   | `ula`                        | Uniform linear array of omni elements                                      | `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)           | Panel array of 3GPP TR 38.901 clause 7.3                                   | `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights` |
   | `multibeam`                  | Panel array with beams toward given directions                             | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
   | `3GPP-UT` (or `3gpp-ut`)     | Handheld UT of 3GPP TR 38.901 clause 7.3                                   | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
   | `3GPP-CPE` (or `3gpp-cpe`)   | CPE of 3GPP TR 38.901 clause 7.3                                           | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
- **`res`** — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; the grid has `floor(360 / res)` azimuth
  and `floor(180 / res)` elevation steps; NaN or <= 0: the default; ignored if `pattern` is given; default: 10 for
  `omni` and `xpol`, 1 otherwise
- **`center_freq`** — Center frequency in Hz; positive and finite; NaN: the default; default: 299792458 (wavelength 1 m)
- **`element`** — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern
    as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no
    taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in
    dBi (finite or NaN); default `[65, 65, 30, 30, 8]`; not together with `pattern`
  - `3GPP-UT`: the same tuple, default `[125, 125, 22.5, 22.5, 5.3]` (Table 7.3-2); `3GPP-CPE`: default
    `[90, 90, 25, 25, 5]`; `[Inf, Inf, NaN, NaN, 0]` is isotropic
- **`panel`** — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; at most 1e7
  elements per polarization (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row); default: 1
- **`spacing`** — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and
  horizontal panel spacing (center to center); finite and >= 0 (0 places elements or panels at the same position);
  default: `[0.5, d_V, M·d_V, N·d_H]`
- **`pol`** — `[polarization, model]`. Polarization: 1 = one element (ζ = 0°, vertical), 2 = two elements (ζ = 0° and
  90°), 3 = two elements (ζ = +45° and -45°). Model of TR 38.901 clause 7.3.2: 1 = the element pattern is turned by ζ
  about its boresight (Model-1, eq. 7.3-3), 2 = the power pattern is kept and the field is split by ζ (Model-2, eq.
  7.3-4, 7.3-5); default: `[1, 1]`; for `3GPP-UT` with polarization 2 or 3 the default model is 2, which gives the
  reference fields of the handheld model (eq. 7.3-6)
- **`ports`** — `[M_P, N_P]`: ports (transceiver units, TXRUs) per column and per row of a panel and polarization; must
  divide `M` and `N`; default: `[M, N]` (one port per element)
- **`steer`** — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt and pan of the sub-array weights; `[1, 2]` for
  all panels or `[Mg·Ng, 2]` with one row per panel (panel order `mg`, `ng`); one column: pan 0; a non-zero tilt needs
  `M_P < M`, a non-zero pan `N_P < N`; default: 0
- **`beams`** — `[2, n_beams]` or `[3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1);
  finite; default: `[0; 0; 1]`
- **`device_size`** — `[X, Y, Z]` in m (depth, width, height); NaN or left off: the default; default: `3GPP-UT`
  `[0.15, 0.07, 0]` with X, Y > 0 and Z = 0, `3GPP-CPE` `[0, 0.2, 0.2]` with X = 0 and Y, Z >= 0
- **`locations`** — TR location labels (`3GPP-UT` 1 to 8, `3GPP-CPE` 1 to 9), not array indices; any order, repeats
  allowed; default: all
- **`pattern`** — Element struct to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`,
  `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, port weights and
  `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with
  several elements is a polarization set used as given and needs `pol` `[1, 1]`; for `3GPP-UT`, a pattern on an
  irregular grid is first resampled to a regular one; default: `[]` (the element of the type)
- **`separate_beams`** — If `true`, one port per beam (weights ignored); default: `false`
- **`apply_weights`** — If `true`, combine each port into one pattern at its sub-array center; default: `false`

### Output:
- **`ant`** — Arrayant struct; panel arrays have `P·M·N·Mg·Ng` elements and `P·M_P·N_P·Mg·Ng` ports (`ula`, `3GPP`);
  with `apply_weights`, one element per port. Fields:
   | Field             | Description                                               | Size                                   |
   | ----------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`      | e-theta field component, real part                        | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`      | e-theta field component, imaginary part                   | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`        | e-phi field component, real part                          | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`        | e-phi field component, imaginary part                     | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`    | Azimuth angles in rad, -π to π, strictly increasing       | `[1, n_azimuth]`                       |
   | `elevation_grid`  | Elevation angles in rad, -π/2 to π/2, strictly increasing | `[1, n_elevation]`                     |
   | `element_pos`     | Element (x,y,z) positions in meters                       | `[3, n_elements]`                      |
   | `port_weights_re` | Port weights, real part                                   | `[n_elements, n_ports]`                |
   | `port_weights_im` | Port weights, imaginary part                              | `[n_elements, n_ports]`                |
   | `center_freq`     | Center frequency in Hz                                    | scalar                                 |
   | `name`            | Name of the antenna array object                          | string                                 |

   As an input to other functions, the fields `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`,
   `elevation_grid` are mandatory; the others are optional (defaults: identity port weights, zero positions, 299792458
   Hz).

### Notes:
- **Element tuple** (`3GPP`, `3GPP-UT`, `3GPP-CPE`): entries left off at the end or set to NaN take their default, where
  `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `3GPP` with `[90]` gives `[90, 90, 30, 30, 8]`);
  `G_max` left off takes the default of the type, NaN sets the peak gain to the directivity
- **Element and port order** (panel arrays): polarization fastest, then row `m` (upward along z), column `n` (along
  +y), panel row `mg`, panel column `ng`, centered on the origin; ports in the same order with `m_p`, `n_p`
- **Sub-array weights** (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds `K = M / M_P` by
  `L = N / N_P` elements. With tilt θ_t and pan φ_p, element (k, l), counted from k = 0 and l = 0, gets

  `w(k,l) = exp(j 2π ((k - (K-1)/2) d_V sin θ_t - (l - (L-1)/2) d_H sin φ_p)) / √(K L)`

  A positive tilt points the beam down, a positive pan toward +y
- **Polarization** (TR 38.901 clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45°, in this
  element order. Model-1 turns the element by ζ about its boresight (eq. 7.3-3). Model-2 keeps the power pattern
  `A' = |F_θ|² + |F_φ|²` and sets `F_θ = √A' cos ζ`, `F_φ = √A' sin ζ` (eq. 7.3-4, 7.3-5)
- **`multibeam`**: maximum-ratio-transmission weights for the directions in `beams`, exact for one beam and approximate
  for several; only the ratios of the beam weights matter. One port per polarization, with `separate_beams` one per
  polarization and beam
- **`3GPP-UT`** (TR 38.901 clause 7.3): `P` elements at each entry of `locations` (Fig. 7.3-2), polarization fastest.
  Each element is rolled about its boresight by 90° (45° for two field patterns: `pol` 2 or a two-element `pattern`) and
  turned to the bearing `atan2(y, x)` of its location (Tables A.6-1, A.6-2). `pol` 1 and 2 with their default model are
  the TR model (eq. 7.3-6 to 7.3-8), other values are extensions. For an element symmetric under a 180° roll (every
  `element` tuple), the other z'' direction of Fig. 7.3-7 is the same antenna with a negated port weight
- **`3GPP-CPE`**: `P` elements at each entry of `locations` (Fig. 7.3-6 seen from the front, in the y-z plane), facing
  +x. TR 38.901 defines no CPE element; the default is the UE element of TR 38.803 Table 5.2.3.3-1

### See also:
- [apply_port_weights](#apply_port_weights) (combines the ports into patterns)
- [rotate_pattern](#rotate_pattern) (turns an antenna)
- [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) (check the generated patterns)
- [qdant_write](#qdant_write) (stores the antenna in a file)

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [C++](quadriga_lib_api_cpp.md#generate_speaker) · [Python](quadriga_lib_api_python.md#generate_speaker)

- Creates one arrayant per frequency with a single element at the origin. Its real-valued pattern in `e_theta_re` is the
  product of the driver directivity, the enclosure radiation and the bandpass frequency response
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining the results with
  [join_arrays](#join_arrays) and the field `element_pos`; the crossover follows from the overlapping bandpass responses

### Usage:
```matlab
arrayant = quadriga_lib.generate_speaker( driver_type, radius, lower_cutoff, upper_cutoff, ...
    lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB, radiation_type, hor_coverage_deg, ...
    ver_coverage_deg, horn_control_freq, baffle_width, baffle_height, center_freq, ...
    angular_resolution_deg );
```

### Inputs:
- **`driver_type`** — Driver directivity model: `piston` (circular piston in a baffle, narrows with increasing
  frequency), `horn` (set coverage, toward omni at low frequencies) or `omni` (independent of frequency); default:
  `piston`
- **`radius`** — Effective radiating radius in m, > 0; cone or dome radius for `piston`, mouth radius for `horn`;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB frequency of the bandpass in Hz, > 0; default: 80
- **`upper_cutoff`** — Upper -3 dB frequency of the bandpass in Hz, larger than `lower_cutoff`; default: 12000
- **`lower_rolloff_slope_dB`** — Low-frequency rolloff in dB per octave, > 0; 12 is a second-order Butterworth filter;
  default: 12
- **`upper_rolloff_slope_dB`** — High-frequency rolloff in dB per octave, > 0; default: 12
- **`sensitivity_dB`** — On-axis sensitivity in dB SPL at 1 W and 1 m, finite; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole` (no change), `hemisphere` (sealed box with baffle-step
  transition), `dipole` (figure-8, rear hemisphere inverted) or `cardioid`; default: `hemisphere`
- **`hor_coverage_deg`** — `horn`: horizontal coverage in degrees, below 180, between the -6 dB points at full pattern
  control; NaN or <= 0: 90°; default: NaN
- **`ver_coverage_deg`** — `horn`: vertical coverage in degrees, below 180, as above; NaN or <= 0: 60°; default: NaN
- **`horn_control_freq`** — `horn`: frequency in Hz above which the horn controls its pattern; NaN or <= 0: `c / (2π a)`
  with the speed of sound c and a = `radius`; default: NaN
- **`baffle_width`** — Baffle width in m, > 0; used by `hemisphere`; default: 0.15
- **`baffle_height`** — Baffle height in m, > 0; used by `hemisphere`; default: 0.25
- **`center_freq`** — Frequency samples in Hz, positive and finite; `[n_freq]`; default: third-octave bands
  `1000 · 2^(n/3)` Hz, those enclosing the cutoffs plus one more on each side, limited to 20 to 20000 Hz
- **`angular_resolution_deg`** — Resolution of the azimuth and elevation grid in degrees; 0.1 to 90; a value r gives
  `floor(360 / r)` azimuth and `floor(180 / r)` elevation steps; default: 5

### Output:
- **`arrayant`** — Struct array with one arrayant per frequency, with `center_freq` set to that frequency; fields as in
  [generate_antenna](#generate_antenna); `[1, n_freq]`

### Notes:
- **Frequency response**: a Butterworth-style bandpass, -3 dB at the cutoffs, scaled by the sensitivity S =
  `sensitivity_dB` relative to 85 dB SPL:

  `H(f) = 10^((S - 85) / 20) / ( √(1 + (f_l / f)^(2 n_l)) · √(1 + (f / f_u)^(2 n_u)) )`

  with f_l = `lower_cutoff`, f_u = `upper_cutoff` and n_l, n_u the rolloff slopes in dB per octave divided by 6
- **Driver directivity** D, with the angle α from the forward axis +x (cos α = cos θ cos φ), the wavenumber k = 2π f /
  c, the speed of sound c = 343 m/s and a = `radius`:
  - `piston`: `D = 2 J1(x) / x` with `x = k a sin α`, rotationally symmetric about +x
  - `horn`: `D = 1 - b + b cos^m_h(φ) cos^m_v(θ)` in the front half (φ between -90° and 90°) and `D = 1 - b` behind it,
    with `b = f² / (f² + f_c²)` and f_c = `horn_control_freq`. The exponents follow from the coverage angle Θ by
    `cos^m(Θ / 2) = 1/2`
  - `omni`: D = 1
- **Enclosure radiation** R:
  - `monopole`: R = 1
  - `hemisphere`: `R = 1 - b + b (1 + cos α) / 2` with `b = f² / (f² + f_b²)` and the baffle-step frequency
    `f_b = c / (π √(W H))`, W = `baffle_width`, H = `baffle_height`; omnidirectional at low and cardioid-like at high
    frequencies
  - `dipole`: `R = abs(cos α)`; the pattern is negative in the rear hemisphere (180° phase inversion)
  - `cardioid`: `R = (1 + cos α) / 2`

### See also:
- [join_arrays](#join_arrays) (builds multi-driver systems)
- [interpolate_pattern](#interpolate_pattern) (evaluates the model at any direction and frequency)
- [calc_beamwidth](#calc_beamwidth) (measures the coverage angle, with `threshold_dB = 6`)
- [qdant_write](#qdant_write) (stores the model in a file)

---
<a id="interpolate_pattern"></a>
## interpolate_pattern
Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles

Also in: [C++](quadriga_lib_api_cpp.md#interpolate_pattern) · [Python](quadriga_lib_api_python.md#interpolate_pattern)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the array and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation)
- Multi-frequency mode (a struct array, or target frequencies given): the two entries whose center frequencies bracket
  each target frequency are interpolated and blended

### Usage:
```matlab
% Single-frequency mode
[ V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma ] = quadriga_lib.interpolate_pattern( ...
    arrayant, azimuth, elevation, i_element, orientation, element_pos, [], use_kernel );

% Multi-frequency mode
[ V_re, V_im, H_re, H_im ] = quadriga_lib.interpolate_pattern( arrayant, azimuth, elevation, ...
    i_element, orientation, element_pos, freq_new, use_kernel );
```

### Inputs:
- **`arrayant`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array; entries need
  not be sorted by frequency
- **`azimuth`** — Azimuth angles in GCS in rad; one row: the same angles for all outputs (planar wave), `n_out` rows:
  one angle per output (spherical wave); `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS in rad; same size as `azimuth`
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements in order
  (`n_out = n_elements`)
- **`orientation`** — Euler angles (bank, tilt, heading) of the array, for all outputs and angles, per output, per
  angle, or both, applied at all frequencies; `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]` or `[3, n_out, n_ang]`; default:
  no rotation
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; multi-frequency mode: checked for
  its size only, since no output depends on the positions; `[3, n_out]`; default: the positions of the selected elements
- **`freq_new`** — Target frequencies in Hz, finite; selects the multi-frequency mode; ascending or descending order is
  fastest; `[n_freq]`; default: the center frequencies of the entries in order
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (error if unavailable or if the pattern has more than
  2^31 - 1 samples); other values raise an error; default: 0

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`, multi-frequency mode
  `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; same size as `V_re`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the array; negative for elements
  closer to the source; `[n_out, n_ang]`; single-frequency mode only
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `[n_out, n_ang]`;
  single-frequency mode only
- **`gamma`** — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`; single-frequency mode
  only

### Notes:
- **Interpolation**: linear where neighboring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- **Frequency interpolation** (multi-frequency mode): spherical interpolation of the complex values over frequency,
  linear where the two values have nearly opposite phases or one of them is zero. Target frequencies outside the range
  of the center frequencies take the nearest entry (no extrapolation). Consecutive target frequencies with the same
  bracketing entries reuse the spatial results
- **Angles**: not range-checked; values outside [-π, π] and [-π/2, π/2] denote the same direction; an elevation beyond
  ±π/2 crosses the pole, so e-theta and e-phi have the opposite sign of the same direction given in range. NaN or
  infinite angles give NaN in all outputs of that angle and output, without an error. `azimuth` of size `[1, 0]` or
  `[n_out, 0]` gives empty outputs; `[]` raises an error
- **Poles**: on a pole of the local frame (a direction along the z-axis of the oriented array) the local azimuth is
  chosen so that the local and the global theta unit vectors agree: the polarization is not rotated there
- **Kernels**: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the
  peak field value and 2e-6 rad in the angles, and near a pole of the local frame up to about 2e-7 / d rad at a distance
  d (in rad). Auto currently selects GENERIC

### See also:
- [apply_port_weights](#apply_port_weights) (patterns per port instead of per element)
- [rotate_pattern](#rotate_pattern) (applies an orientation to the stored patterns)
- [join_arrays](#join_arrays), [generate_speaker](#generate_speaker) (build multi-element, multi-frequency arrays)
- [get_channels_spherical](#get_channels_spherical) (channel coefficients from the interpolated patterns)

---
<a id="join_arrays"></a>
## join_arrays
Join the elements of two antenna arrays into a new array

Also in: [C++](quadriga_lib_api_cpp.md#join_arrays) · [Python](quadriga_lib_api_python.md#join_arrays)

- Combines two arrays with identical sampling grids, e.g. the drivers of a loudspeaker; `name` and `center_freq` are
  taken from the first array
- The ports stay separate: elements of the first array connect only to its ports, and likewise for the second:
   | Element \ Port        | P1…Pp1 (first) | Pp1+1…Pp1+p2 (second) |
   | --------------------- | :------------: | :-------------------: |
   | E1…En1 (first)        |    W1 block    |           0           |
   | En1+1…En1+n2 (second) |       0        |       W2 block        |

### Usage:
```matlab
arrayant_out = quadriga_lib.join_arrays( arrayant_in1, arrayant_in2 );
```

### Inputs:
- **`arrayant_in1`** — First array; arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency
  array; its elements come first in the output
- **`arrayant_in2`** — Second array with the same sampling grids; a struct array needs as many entries as the first and
  matching `center_freq` values at each index

### Output:
- **`arrayant_out`** — New arrayant with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports, in the form of
  the inputs

### See also:
- [copy_element](#copy_element) (duplicates an element within one array)
- [generate_speaker](#generate_speaker) (creates the driver models to join)
- [qdant_write](#qdant_write) (stores the joined model)

---
<a id="qdant_read"></a>
## qdant_read
Read arrayant objects from a QDANT file

Also in: [C++](quadriga_lib_api_cpp.md#qdant_read) · [Python](quadriga_lib_api_python.md#qdant_read)

- Loads arrays from a file in the [QDANT format](formats/qdant.md). A file can hold several arrays, each with a
  numeric ID, e.g. one per frequency of a frequency-dependent model, where `center_freq` of each entry identifies the
  frequency

### Usage:
```matlab
% Multi-frequency read (struct array, entries in the layout)
[ ant, layout ] = quadriga_lib.qdant_read( fn );

% Single-frequency read (struct output)
[ ant, layout ] = quadriga_lib.qdant_read( fn, ids );

% Several entries (struct array)
[ ant, layout ] = quadriga_lib.qdant_read( fn, [3,1,3] );
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`ids`** — 1-based IDs of the entries to read, in this order, duplicates allowed; scalar or vector; default: the
  entries listed in the layout, each once, in order of first appearance (column-major scan), or all entries if the file
  has no layout; several entries must then form a multi-frequency array, otherwise an error asks to pass `ids`

### Outputs:
- **`ant`** — Arrayant struct ([generate_antenna](#generate_antenna)) for one entry, 1xN struct array for several
- **`layout`** — The file's layout matrix of entry IDs (0 = empty slot); uint32

### Notes:
- **IDs**: the number used in the layout matrix and by [qdant_write](#qdant_write). A first entry without an `id` attribute (files
  from other tools) counts as ID 1
- **Errors**: an ID that is 0 or not in the file, and a missing, unreadable or malformed file, raise an error
- **Elements without data**: elements without pattern nodes read as zeros. A file whose element count is not backed by
  its data, or that exceeds the [reader limits](formats/qdant.md), raises an error

### See also:
- [qdant_write](#qdant_write) (writes such files)
- [generate_antenna](#generate_antenna), [generate_speaker](#generate_speaker) (create arrays without a file)

---
<a id="qdant_write"></a>
## qdant_write
Write arrayant data to a QDANT (XML) file

Also in: [C++](quadriga_lib_api_cpp.md#qdant_write) · [Python](quadriga_lib_api_python.md#qdant_write)

- Stores an array in a file in the [QDANT format](formats/qdant.md). A file can hold several arrays with distinct IDs
  and a layout matrix that arranges them
- A struct array stores a frequency-dependent model in a new file, each entry holding the pattern at its `center_freq`

### Usage:
```matlab
id_in_file = quadriga_lib.qdant_write( fn, arrayant, id, layout );
```

### Inputs:
- **`fn`** — Output QDANT file name, must not be empty; an existing file is updated; struct array: the file must not
  exist (raises an error and leaves the file unchanged)
- **`arrayant`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array, written with the sequential IDs
  `1, 2, ..., N` and an `[N, 1]` layout of these IDs
- **`id`** — Target ID in the file; an entry with this ID is replaced; 0 appends with `max_existing_id + 1`, or 1 in a
  new file; struct array: must be 0 or omitted; default: 0
- **`layout`** — Matrix of entry IDs that arranges the file; every value must be an ID in the file (0 is not accepted);
  struct array: must be empty; default: `[]`, which replaces the layout in the file by a `[1, n]` row of all IDs in file
  order, so a custom layout must be passed again with every write that should keep it

### Output:
- **`id_in_file`** — ID assigned to the written antenna within the file; 0 for a struct array; uint32

### Notes:
- **Pattern accuracy**: the file has single-precision-equivalent accuracy relative to the element peak. Pattern
  magnitudes (in dB) and phases (in degrees) get the fewest decimals (0 to 6) that keep the error of each complex sample
  at or below 6e-8 times the peak amplitude of its element (over `e_theta` and `e_phi`). Samples near the peak are
  accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the peak (about 6e-4 at
  -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at a low level is
  quantized relative to its own peak
- **Other values**: element positions and port weights (magnitude, phase) are written as the shortest text that reads
  back as the same float (at most 9 significant digits); sampling grids are rounded to 0.001 degree
- **Zeros**: all non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element
  that is zero in both field components is written as an all `-inf` `e_theta_mag` node

### See also:
- [qdant_read](#qdant_read) (reads the file back)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrays)

---
<a id="rotate_pattern"></a>
## rotate_pattern
Rotate the patterns of an antenna array using Euler rotations

Also in: [C++](quadriga_lib_api_cpp.md#rotate_pattern) · [Python](quadriga_lib_api_python.md#rotate_pattern)

- Turns the patterns, the polarization and the element positions of an array, e.g. for a mechanical downtilt. The
  rotated pattern equals, at the grid points, what [interpolate_pattern](#interpolate_pattern) returns for the unrotated pattern with the
  same `orientation`
- Port weights, `center_freq` and `name` are kept

### Usage:
```matlab
arrayant_out = quadriga_lib.rotate_pattern(arrayant_in, orientation, usage, i_element);
```

### Inputs:
- **`arrayant_in`** — Arrayant struct ([generate_antenna](#generate_antenna)), or 1xN struct array for a multi-frequency array
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; a positive tilt turns the main beam down, as in
  3GPP TR 38.901; 3 elements
- **`usage`** — Rotation mode; default: 0. Modes 0 and 1 adjust the sampling grid of non-uniformly sampled patterns,
  which needs all elements: for such a pattern, a subset in `i_element` raises an error. A struct array never adjusts
  the grid, so that all entries keep the same grid: there 0 equals 3, and 1 equals 4. Element positions turn in all
  modes except 2. For scalar acoustic fields (pressure in `e_theta_re` only), use 1 to avoid spurious polarization
  effects:
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`i_element`** — Element indices to rotate, 1-based, no duplicates; `[n]`; default: all elements

### Output:
- **`arrayant_out`** — Rotated arrayant, in the form of `arrayant_in`

### See also:
- [interpolate_pattern](#interpolate_pattern) (evaluates a pattern for an orientation without changing it)
- [calc_rotation_matrix](#calc_rotation_matrix) (rotation matrix of the orientation)
- [generate_antenna](#generate_antenna) (electrical tilt through the port weights)
- [join_arrays](#join_arrays) (joins arrays that are then rotated together)

---

<a id="sec-channel-functions"></a>
# Channel functions

---
<a id="baseband_freq_response"></a>
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

Also in: [C++](quadriga_lib_api_cpp.md#baseband_freq_response) · [Python](quadriga_lib_api_python.md#baseband_freq_response)

- Computes the frequency-domain channel matrix at the carrier positions `pilot_grid` by a DFT over the time-domain
  path coefficients and delays
- One snapshot is a 3D array; a batch of snapshots is a cell array of such arrays (`coeff_re`, `coeff_im` and `delay`
  cell arrays of the same length), processed in parallel; the number of paths may differ between snapshots
- A channel struct (e.g. from [hdf5_read_channel](#hdf5_read_channel)) holds its snapshots on the 4th dimension, zero-padded;
  `num2cell(ch.coeff_re, [1 2 3])`, and the same for `coeff_im` and `delay`, gives the cell arrays (the zero-padded
  paths do not change the result)
- Multi-frequency coefficients (e.g. from [get_channels_multifreq](#get_channels_multifreq)) and absolute carrier frequencies are handled by
  [baseband_freq_response_multi](#baseband_freq_response_multi); the two functions take different arguments, so they are two functions, while
  [interpolate_pattern](#interpolate_pattern) serves both of its cases with one argument set
- `delay` supports broadcasting: shape `[1, 1, n_path]` applies the same delays to all RX/TX pairs
- The phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to one period
  before sin / cos are evaluated in single precision; coefficients and sums are single precision; double inputs are
  narrowed to float internally, results widened back

### Usage:
```matlab
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, pilot_grid, bandwidth, ...
    i_snap, use_kernel );
```

### Inputs:
- **`coeff_re`** — Real part of the channel coefficients; `[n_rx, n_tx, n_path]`, or a cell array of length `n_snap`
  with one such array per snapshot
- **`coeff_im`** — Imaginary part of the channel coefficients; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; same layout as `coeff_re`, each array `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`
- **`pilot_grid`** — Sub-carrier positions relative to the bandwidth, `0.0` = carrier, `1.0` = carrier + `bandwidth`;
  `[n_carrier, 1]`
- **`bandwidth`** — Baseband bandwidth in Hz
- **`i_snap`** — Snapshots to process, 1-based; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`

### See also:
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-frequency coefficients, absolute carrier frequencies)
- [get_channels_spherical](#get_channels_spherical) (single-frequency channel generator)

---
<a id="baseband_freq_response_multi"></a>
## baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

Also in: [C++](quadriga_lib_api_cpp.md#baseband_freq_response_multi) · [Python](quadriga_lib_api_python.md#baseband_freq_response_multi)

- Interpolates the complex channel coefficients from the input frequencies `freq` to the output carrier
  frequencies `freq_new` using SLERP (magnitude and unwrapped phase each linear), then applies the delay-induced
  phase `exp(-j·2π·freq_new·delay)` per output carrier in double precision
- One snapshot is a 4D array `[n_rx, n_tx, n_path, n_freq]`; a batch of snapshots is a cell array of such arrays
  (`coeff_re`, `coeff_im` and `delay` cell arrays of the same length), processed in parallel; the number of paths may
  differ between snapshots
- Single-frequency coefficients with absolute output carriers: `n_freq = 1` (a 3D array), `freq` the frequency the
  coefficients were generated at; with `remove_delay_phase = true` (the default) and coefficients that carry the delay
  phase at `freq`, as the channel generators return them, this gives the result of [baseband_freq_response](#baseband_freq_response) with
  `pilot_grid · bandwidth = freq_new - freq`
- `delay` is frequency-independent: `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`; of a 4D `delay` (as returned by
  [get_channels_multifreq](#get_channels_multifreq)) only the first frequency is used
- With `remove_delay_phase`, the channels from [get_channels_multifreq](#get_channels_multifreq) / [get_channels_spherical](#get_channels_spherical) must be
  generated with `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also
  when `delay` is relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq`,
  which gives wrong phases (magnitudes stay right)
- Output frequencies outside the range of `freq` use constant extrapolation from the nearest endpoint

### Usage:
```matlab
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response_multi( coeff_re, coeff_im, delay, freq, freq_new, ...
    remove_delay_phase, i_snap, use_kernel );
```

### Inputs:
- **`coeff_re`** — Real part of the channel coefficients; `[n_rx, n_tx, n_path, n_freq]`, or a cell array of length
  `n_snap` with one such array per snapshot
- **`coeff_im`** — Imaginary part of the channel coefficients; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; an array `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` (a 4th dimension of length
  `n_freq` is allowed, only the first frequency is used), or a cell array of length `n_snap` with one such array per
  snapshot
- **`freq`** — Input sample frequencies in Hz, sorted ascending; `[n_freq, 1]`
- **`freq_new`** — Output carrier frequencies in Hz; `[n_carrier, 1]`
- **`remove_delay_phase`** — Remove the delay phase `exp(-j·2π·freq·delay)` from the coefficients before the
  interpolation and apply it again at `freq_new`; `true` for the output of the channel generators, `false` for pure
  envelope coefficients; default: `true`
- **`i_snap`** — Snapshots to process, 1-based; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-frequency coefficients, relative carrier positions)
- [get_channels_multifreq](#get_channels_multifreq) (multi-frequency channel generator)

---
<a id="channel_export_obj_file"></a>
## channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](quadriga_lib_api_cpp.md#channel_export_obj_file) · [Python](quadriga_lib_api_python.md#channel.export_obj_file)

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min_dB` are excluded; `max_paths` limits the total number of exported paths
- Paths with non-finite coordinates are skipped

### Usage:
```matlab
quadriga_lib.channel_export_obj_file( fn, chan, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, ...
    radius_max, radius_min, n_edges );
```

### Inputs:
- **`fn`** — Output `.obj` file path
- **`chan`** — Channel; 1x1 struct with the fields of [hdf5_read_channel](#hdf5_read_channel); needs `rx_pos`, `tx_pos`,
  `center_freq`, `n_interact`, `interact_coord` and `coeff_re` / `coeff_im` with `delay`, or `polmat`;
  columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Max paths to export; 0 includes all paths above `gain_min_dB`; default: 0
- **`gain_max_dB`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min_dB`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`i_snap`** — Snapshot indices to include; 1-based; range [1 ... n_snap]; default: `[]` (all snapshots)
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_create_file) · [Python](quadriga_lib_api_python.md#hdf5_create_file)

- Initializes a new HDF5 file for storing wireless channel data
- Defines a 4D layout `(nx, ny, nz, nw)` where each index combination maps to one channel storage slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be altered later, except by reshaping while
  keeping the total slot count constant
- Errors if the target file already exists (also one that cannot be read); delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

### Usage:
```matlab
quadriga_lib.hdf5_create_file( fn, nx, ny, nz, nw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; string
- **`nx`** — Number of slots along dimension X; default: 65536
- **`ny`** — Number of slots along dimension Y; default: 1
- **`nz`** — Number of slots along dimension Z; default: 1
- **`nw`** — Number of slots along dimension W; default: 1

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Read one or more channel objects from an HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_channel) · [Python](quadriga_lib_api_python.md#hdf5_read_channel)

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (omitted/empty = read full extent along that dimension)
- Slots are visited in column-major order and empty slots are skipped.
- Structured fields are stored in single precision in the file and returned in double.
- Unstructured datasets keep their stored type and shape.
- If no data is found, both outputs are empty `0x0` structs.
- Errors if the file does not exist or is not a quadriga-lib channel file; files modified with other HDF5 tools
  are not supported and may be rejected.

### Usage:
```matlab
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, i_snap );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`ix`** — 1-based slot indices along dimension X; scalar or vector; default: `1:nx`
- **`iy`** — 1-based slot indices along dimension Y; scalar or vector; default: `1:ny`
- **`iz`** — 1-based slot indices along dimension Z; scalar or vector; default: `1:nz`
- **`iw`** — 1-based slot indices along dimension W; scalar or vector; default: `1:nw`
- **`i_snap`** — Snapshot indices to read; 1-based; default: all snapshots. Applies to every
  selected slot; errors if an index exceeds the number of snapshots of a selected slot.

### Outputs:
- **`chan`** — Struct array of length `N` (number of non-empty slots in the selection) with the channel data. Fields per element:
  | Field              | Description                                                              | Type / Size                         |
  | ------------------ | ------------------------------------------------------------------------ | ----------------------------------- |
  | `name`             | Channel name                                                             | String                              |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | `[3, 1]` or `[3, n_snap]`           |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `[3, 1]` or `[3, n_snap]`           |
  | `coeff_re`         | Channel coefficients, real part                                          | `[n_rx, n_tx, n_path, n_snap]`      |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | `[n_rx, n_tx, n_path, n_snap]`      |
  | `delay`            | Propagation delays in seconds                                            | `[n_rx, n_tx, n_path, n_snap]`      |
  | `path_gain`        | Path gain before antenna, linear scale                                   | `[n_path, n_snap]`                  |
  | `path_length`      | Path length in m                                                         | `[n_path, n_snap]`                  |
  | `polmat`           | Polarization transfer matrix, interleaved complex                        | `[8, n_path, n_snap]`               |
  | `path_angles`      | Departure and arrival angles [AOD, EOD, AOA, EOA] in rad                 | `[n_path, 4, n_snap]`               |
  | `fbs_pos`          | First-bounce scatterer positions                                         | `[3, n_path, n_snap]`               |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | `[3, n_path, n_snap]`               |
  | `n_interact`       | Number of interaction points per path; uint32                            | `[n_path, n_snap]`                  |
  | `interact_coord`   | Interaction coordinates                                                  | `[3, max(sum(n_interact)), n_snap]`|
  | `center_freq`      | Center Frequency in Hz                                                   | Scalar or `[n_snap]`                |
  | `initial_position` | Reference snapshot (start of the current segment); 1-based; absent = 1   | int32, scalar                       |
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
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_dset) · [Python](quadriga_lib_api_python.md#hdf5_read_dset)

- Reads a user-defined dataset stored under `prefix + name` (e.g. `"par_carrier_frequency"`)
- Type and shape of the returned data are determined by the dataset's HDF5 dataspace
- Returns an empty `[]` matrix if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element 
  types: single, double, int32, uint32, int64, uint64

### Usage:
```matlab
dset = quadriga_lib.hdf5_read_dset( fn, name, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; string
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; string; `''` looks up `name` as given; default (omitted or `[]`): `'par_'`

### Output:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; empty `[]` if the dataset is missing

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_dset_names) · [Python](quadriga_lib_api_python.md#hdf5_read_dset_names)

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`
- Returned names have the prefix stripped
- Returns an empty cell array if no matching datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file

### Usage:
```matlab
names = quadriga_lib.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — Path to the HDF5 file; string
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix used to identify unstructured datasets; string; `''` lists all datasets with their full names; default (omitted or `[]`): `'par_'`

### Output:
- **`names`** — Names of all datasets at the given slot, with the prefix stripped; cell array of strings; `[n_names, 1]`

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_layout) · [Python](quadriga_lib_api_python.md#hdf5_read_layout)

- Returns the dimensions of the 4D channel slot grid stored inside an HDF5 file
- Returns `[0, 0, 0, 0]` if the file does not exist; errors if the file exists but is not a
  quadriga-lib channel file
- Also reports which slots already contain data, so callers can locate free slots without scanning the file

### Usage:
```matlab
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

### Input:
- **`fn`** — Filename of the HDF5 file; string

### Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; uint32; `[1, 4]`
- **`has_data`** — Slot occupancy mask; `true` where data exists, `false` otherwise; `[nx, ny, nz, nw]`; logical

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshape the storage layout inside an existing HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_reshape_layout) · [Python](quadriga_lib_api_python.md#hdf5_reshape_layout)

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file
- One writer per file at a time; files are not locked

### Usage:
```matlab
quadriga_lib.hdf5_reshape_layout( fn, nx, ny, nz, nw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`nx`** — New number of slots along dimension X
- **`ny`** — New number of slots along dimension Y; default: 1
- **`nz`** — New number of slots along dimension Z; default: 1
- **`nw`** — New number of slots along dimension W; default: 1

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the existing layout)

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_write_channel) · [Python](quadriga_lib_api_python.md#hdf5_write_channel)

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file if it does not exist, with the layout that just holds all slots (largest index per dimension);
  use [hdf5_create_file](#hdf5_create_file) for another layout
- All inputs are checked before anything is written: every slot inside the layout of an existing file, no slot
  twice, all channels valid
- A slot that already contains data is replaced completely: its old structured data and all its unstructured
  datasets (also those written with [hdf5_write_dset](#hdf5_write_dset)) are removed; `code` shows which slots were replaced. The
  file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [hdf5_write_dset](#hdf5_write_dset))
- Each scalar location input is broadcast to all `numel(chan)` channels; each vector input must have exactly `numel(chan)` elements

### Usage:
```matlab
code = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`chan`** — Structured channel data; non-empty struct array; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; struct array of the same size as `chan`. Field names become HDF5 dataset
  names per slot (each prefixed with `par_`). Empty fields are skipped; default: `[]` (none).
  Field values follow the type rules of [hdf5_write_dset](#hdf5_write_dset) (at most 3 dimensions).
- **`ix`** — 1-based slot index along dimension X; scalar or vector of length `numel(chan)`; default: `1:numel(chan)`
- **`iy`** — 1-based slot index along dimension Y; scalar or vector of length `numel(chan)`; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar or vector of length `numel(chan)`; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar or vector of length `numel(chan)`; default: 1

### Output:
- **`code`** — Per channel: `0` if the slot was empty, `1` if existing data was replaced; `[numel(chan), 1]`; int32

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_write_dset) · [Python](quadriga_lib_api_python.md#hdf5_write_dset)

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must not be empty and must contain only alphanumeric characters and underscores
- Creates the file with the layout `[ix, iy, iz, iw]` if it does not exist; use [hdf5_create_file](#hdf5_create_file) for another
  layout
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [hdf5_write_channel](#hdf5_write_channel) to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element
  types: single, double, int32, uint32, int64, uint64
- Arrays with more than 3 dimensions, sparse and complex arrays are rejected
- Row vectors are stored as given (`[1, n]`)
- One writer per file at a time; files are not locked

### Usage:
```matlab
quadriga_lib.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; string
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; string; `''` stores `name` without a prefix; default (omitted or `[]`): `'par_'`

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [C++](quadriga_lib_api_cpp.md#qrt_file_parse) · [Python](quadriga_lib_api_python.md#qrt_file_parse)

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count,
  CIR offsets, names, positions, orientations, and file version
- When `n_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```matlab
[ n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, ...
     cir_pos, cir_orientation, orig_pos, orig_orientation ] = quadriga_lib.qrt_file_parse( fn );
```

### Input:
- **`fn`** — Path to the QRT file; string

### Outputs:
- **`n_cir`** — Number of channel snapshots per origin point; uint64 scalar
- **`n_orig`** — Number of origin points (TX); uint64 scalar
- **`n_dest`** — Number of destination points (RX); uint64 scalar
- **`n_freq`** — Number of frequency bands; uint64 scalar
- **`cir_offset`** — Number of CIRs before each destination (an offset, not an index; the first entry is 0); the CIRs of
  destination `d` start at `i_cir = cir_offset(d) + 1`; uint64; `[n_dest, 1]`
- **`orig_names`** — Names of origin points; cell array of strings; `[n_orig, 1]`
- **`dest_names`** — Names of destination points; cell array of strings; `[n_dest, 1]`
- **`version`** — QRT file version number; int32 scalar
- **`center_freq`** — Center frequencies in Hz, converted from GHz for files that store GHz (v4, and v5 when all stored values are below 1000); single; `[n_freq, 1]`
- **`cir_pos`** — CIR positions in Cartesian coordinates; single; `[n_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles; single; `[n_cir, 3]`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; single; `[n_orig, 3]`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; single; `[n_orig, 3]`

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read ray-tracing CIR data from a QRT file

Also in: [C++](quadriga_lib_api_cpp.md#qrt_file_read) · [Python](quadriga_lib_api_python.md#qrt_file_read)

- Reads channel impulse response data from QRT files
- All output arguments are optional; MATLAB only computes outputs that are requested
- Reads in chunks of 16 snapshots; besides the output, at most two chunks are held in memory
- If `uplink = false`, origin is TX and destination is RX; if `true`, roles are swapped and `polmat` is the transpose
  of the downlink `polmat` (reciprocal channel, no conjugation) with the same `path_gain`; `n_interact`, `coord` and
  `interact_type` keep the stored order (origin to destination) in both directions, and the type codes describe
  that traversal

### Usage:
```matlab
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, uplink, normalize_polmat );
```

### Inputs:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** — Snapshot indices; 1-based; `[n_out]` or empty; default: `[]` (read all; for a file
  without CIRs, all outputs are empty with `n_out = 0`)
- **`i_orig`** — Origin index; 1-based; scalar; default: 1
- **`uplink`** — If `false`, origin=TX, destination=RX; if `true`, roles are swapped (origin=RX, destination=TX); logical scalar; default: `false`
- **`normalize_polmat`** — Controls `polmat` and `path_gain` scaling where PL is the propagation-only path loss; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_polmat` | `polmat`                                   | `path_gain`                      |
    | :-----------: | :---------------------------------------------: | :------------------------------: |
    | 0             | As stored in QRT file                           | -PL                              |
    | 1             | Max column power = 1; uplink: max row power = 1 | -PL minus material losses        |

    Other values are rejected with an error.

### Outputs:
- **`center_freq`** — Center frequency in Hz; `[n_freq, 1]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** — First-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`path_gain`** — Path gain in linear scale; Cell of length `n_out`; elements `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; Cell of length `n_out`; elements `[n_path, 1]`
- **`polmat`** — Polarization transfer matrix; Cell of length `n_out`;
  elements `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`eod`** — Departure elevation angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`aoa`** — Arrival azimuth angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`eoa`** — Arrival elevation angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`path_coord`** — Interaction coordinates per path; Cell of length `n_out`;
  elements Cell of length `n_path`, each `[3, n_interact + 2]`
- **`n_interact`** — Number of mesh interactions per path; 0 indicates LOS; uint32;
  Cell of length `n_out`; elements `[n_path, 1]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single;
  Cell of length `n_out`; elements `[3, sum(n_interact)]`
- **`interact_type`** — Interaction type codes (flat, concatenated across paths, matching `coord`); uint8;
  Cell of length `n_out`; elements `[sum(n_interact), 1]`. Empty for v4 legacy files. GBSM paths written with the C++ API
  carry 129 (GR) and 193 (NLOS); LOS and GR paths are not marked as GBSM.

### See also:
- [generate_antenna](#generate_antenna) (for generating antenna arrays)
- [get_channels_planar](#get_channels_planar) (for embedding antennas using departure and arrival angles)
- [get_channels_spherical](#get_channels_spherical) (for embedding antennas using FBS/LBS positions)
- [get_channels_multifreq](#get_channels_multifreq) (for multi-frequency antenna embedding)

---
<a id="quantize_delays"></a>
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

Also in: [C++](quadriga_lib_api_cpp.md#quantize_delays) · [Python](quadriga_lib_api_python.md#quantize_delays)

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α and δ^α, 
  where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband (incoherent power preservation)
- If all fractional per-tap offsets are below 0.01 or above 0.99, weight computation is skipped 
  (nearest neighbor selection) but tap-selection logic still applies
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
```matlab
[ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
    tap_spacing, max_taps, power_exponent, fix_taps );
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`; non-negative
  with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate; default: 5e-9
- **`max_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
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

<a id="sec-channel-generation-functions"></a>
# Channel generation functions

---
<a id="get_channels_ieee_indoor"></a>
## get_channels_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_ieee_indoor) · [Python](quadriga_lib_api_python.md#channel.ieee_indoor)

- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions
- 2D model: azimuth angles and planar motion only, no elevation
- Supported channel types: `A, B, C, D, E, F` (TGn definitions)
- MU-MIMO supported (`n_users > 1`) with per-user distances/floors and optional angle offsets per TGac
- Time-evolving channels via `observation_time`, `update_interval`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Floor penetration loss according to TGah for `center_freq` < 1 GHz and TGax for above 1 GHz; floors add attenuation
  (and the TGah shadow fading below 1 GHz) but no delay; the station is placed 3 m per floor above the AP
- Taps with zero power in the model tables are removed; the number of paths does not depend on distance, floors or walls
- Time evolution: each sub-path is rotated by `exp(+j 2 pi f_D t)`, i.e. a station moving towards the AP or a scatterer
  sees a positive Doppler shift `f_D`, as the channel of [get_channels_spherical](#get_channels_spherical) along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- Override parameters (`kf`, `xpr_nlos`, `sf_std_los_dB`, `sf_std_nlos_dB`, `dist_breakpoint`): `NaN` or `[]` selects the model default

### Usage:
```matlab
chan = quadriga_lib.get_channels_ieee_indoor( ap_array, sta_array, channel_type, center_freq, ...
    tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, ...
    dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, ...
    kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB );
```

### Inputs:
- **`ap_array`** — Access point antenna array; `n_tx` = number of ports; see [generate_antenna](#generate_antenna)
- **`sta_array`** — Mobile station antenna array; `n_rx` = number of ports; see [generate_antenna](#generate_antenna)
- **`channel_type`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`center_freq`** — Carrier frequency in Hz (finite, above 100 MHz); default: 5.25e9
- **`tap_spacing`** — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** — Channel observation time in seconds; must be finite; default: 0
- **`update_interval`** — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist`** — TX-to-RX distance(s) in meters (finite, > 0); `[n_users]` or `[1]`; default: 4.99
- **`n_floors`** — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `[n_users]` or `[1]`; default: 0
- **`uplink`** — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: false
- **`offset_angles_deg`** — Azimuth offset angles in degrees (finite); rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; `[4, n_users]`;
  default: `[]` (TGac auto-defaults for `n_users > 1`)
- **`n_subpath`** — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** — Special Doppler: models D/E use the mains frequency in Hz, model F the vehicle speed in km/h (finite, >= 0); 0 disables; default: 50
- **`seed`** — RNG seed; any other value than `-1` gives the same channels on all platforms; `-1` uses the system random device; default: -1
- **`kf`** — Overrides the model KF (linear scale), finite and >= 0; default: A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0;
  applied to first tap only; with `kf` given, the LOS path is kept also beyond the breakpoint and on other floors
- **`xpr_nlos`** — Overrides the NLOS cross-polarization ratio (linear scale), finite and > 0;
  default: 2 (3 dB)
- **`sf_std_los_dB`** — Overrides the LOS shadow fading std in dB (applied when d < breakpoint), finite and >= 0; default: 3 dB, 2 dB below 1 GHz (TGah)
- **`sf_std_nlos_dB`** — Overrides the NLOS shadow fading std in dB (applied when d >= breakpoint), finite and >= 0;
  default: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB, each 1 dB lower below 1 GHz (TGah)
- **`dist_breakpoint`** — Overrides the breakpoint distance in meters, finite and > 0; default: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- **`n_walls`** — Number of walls per user TGax models; `[n_users]` or `[1]`; default: 0
- **`wall_loss_dB`** — Penetration loss for a single wall in dB (finite); TGax defines 5 or 7; default: 5

### Output:
- **`chan`**\
  Struct array of length `n_users` containing the channel data with the following fields:
  | Field              | Description                                                              | Type                                  |
  | ------------------ | ------------------------------------------------------------------------ | ------------------------------------- |
  | `name`             | Channel name                                                             | String                                |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `tx_orientation`   |  Transmitter orientation, Euler angles (AP for downlink, STA for uplink) | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `coeff_re`         | Channel coefficients, real part                                          | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `delay`            | Propagation delays in seconds                                            | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `path_gain`        | Path gain before antenna, linear scale                                   | Size: `[n_path, n_snap]`              |
  | `center_freq` | Center Frequency in Hz                                                   | Scalar                                |

### See also:
- [IEEE 802.11-03/940r4 - TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc)
- [IEEE 802.11-09/0308r12 - TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc)
- [IEEE 802.11-11/0968r4 - TGah Channel Model](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx)
- [IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx)
- [S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025](http://quadriga-lib.org/publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf)
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data to a HDF5 file)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data to a HDF5 file)

---
<a id="get_channels_irs"></a>
## get_channels_irs
Calculate MIMO channel coefficients for IRS-assisted communication

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_irs) · [Python](quadriga_lib_api_python.md#channel.irs)

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array of `n_irs` elements; each element reflects once with its weight w_k from column
  `i_irs` of the IRS port weights (the codebook entry): `H = B · diag(w) · A` per path pair, with the per-element
  channels A (TX → IRS) and B (IRS → RX); delays and angles of a pair are |w_k|²-weighted means over the elements
- The IRS port weights hold the IRS weights and must be given: an `irs_array` struct without `port_weights_re` /
  `port_weights_im` throws, unlike in the other functions, where it stands for identity
- Polarization coupling is applied via the 8-row transfer matrices `polmat_1`, `polmat_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx),
  rows 7-8 = (H rx, H tx)
- Each segment is computed like [get_channels_spherical](#get_channels_spherical): NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths exceeding `threshold_dB`; the pair of
  segment-1 path `i_1` and segment-2 path `i_2` (1-based) has index `(i_1 - 1) · n_path_2 + i_2` in `active_path` and
  `active_path_out`, and the output keeps this order
- A segment without paths (all its path inputs empty) gives empty outputs `[n_rx, n_tx, 0]`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- The outputs come in another order than in C++: `active_path_out` comes before the angles, so that the rarely needed angles can be left off at the end
- Optional `irs_array_2` provides the patterns of the RX-facing side (asymmetric IRS); the weights and element positions
  come from `irs_array`
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

### Usage:
```matlab
[ coeff_re, coeff_im, delay, active_path_out, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    tx_array, rx_array, irs_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, irs_array_2, active_path, use_kernel );
```

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports, see [generate_antenna](#generate_antenna)
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports, see [generate_antenna](#generate_antenna)
- **`irs_array`** — IRS antenna array (TX-facing side) with `n_irs` elements; the port weights hold the IRS weights,
  one codebook entry per column, `[n_irs, n_codebook]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`irs_pos`** — IRS position; `[3, 1]`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1, 1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center for TX → IRS paths; values shorter than the
  polyline (e.g. 0) use the polyline length; `[n_path_1, 1]`
- **`polmat_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2, 1]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center for IRS → RX paths; values shorter than the
  polyline (e.g. 0) use the polyline length; `[n_path_2, 1]`
- **`polmat_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `[8, n_path_2]`
- **`i_irs`** — IRS codebook port index, 1-based (one port per column of the IRS port weights); default: `1`
- **`threshold_dB`** — Gain threshold in dB; path combinations below this are discarded; dB; default: `-140`
- **`center_freq`** — Center frequency in Hz; 0 (or empty) gives the coefficients at f = 0 (see above); default: `0`
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`irs_array_2`** — Second IRS antenna array for the RX-facing side, with the same number of elements as `irs_array`;
  only its patterns are used, its port weights and `element_pos` are ignored; default: `[]` (`irs_array` on both sides)
- **`active_path`** — Bitmask selecting active path pairs; overrides `threshold_dB` when non-empty; logical; `[n_path_1 · n_path_2, 1]`; default: `[]`
- **`use_kernel`** — Kernel of both segments, as in [get_channels_spherical](#get_channels_spherical): 0 = auto, 1 = GENERIC, 2 = AVX2 (single
  precision; error if unavailable); auto selects GENERIC in this version; default: 0

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
- [generate_antenna](#generate_antenna) (antenna array generator)

---
<a id="get_channels_multifreq"></a>
## get_channels_multifreq
Compute channel coefficients for spherical waves across multiple frequencies

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_multifreq) · [Python](quadriga_lib_api_python.md#channel.multifreq)

- Multi-frequency extension of [get_channels_spherical](#get_channels_spherical) with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (from each `tx_array.center_freq`), RX array, input samples (`freq`), and output (`freq_new`)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `polmat` is interpolated via SLERP per complex entry pair to preserve phase
- Extrapolation clamps to the nearest frequency entry on all four grids
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `polmat` accepts 8 rows (full polarimetric: the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx),
  rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry.
- Port weights are interpolated across frequencies (SLERP for complex pairs), identical to antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 1 holds the LOS path
  (zero power if no LOS path was found; LOS = the path whose length is closest to the TX-RX distance, within 0.1 mm);
  a LOS path found in the input is moved to slot 1 and leaves a zero-power path in its original slot
  (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0, n_freq_new]`; with
  `add_fake_los_path = true` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [quantize_delays](#quantize_delays) rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [baseband_freq_response_multi](#baseband_freq_response_multi) removes the phase of `delay` before it interpolates across
  `center_freq`, so it needs channels generated with `use_absolute_delays = true`; with relative delays, the phase of
  the TX-RX distance stays in the coefficients, is interpolated across the frequency grid and gives wrong phases

### Usage:
```matlab
[ coeff_re, coeff_im, delay ] = quadriga_lib.get_channels_multifreq( tx_array, rx_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, ...
    freq, freq_new, use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel );
```

### Inputs:
- **`tx_array`** — Multi-frequency TX arrayant struct array; one entry per input frequency, see [generate_antenna](#generate_antenna)
- **`rx_array`** — Multi-frequency RX arrayant struct array; one entry per input frequency, see [generate_antenna](#generate_antenna)
- **`tx_pos`** — Transmitter position, finite; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `[3, 1]`
- **`rx_pos`** — Receiver position, finite; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `[3, 1]`
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Linear-scale path gains per input frequency; `[n_path, n_freq]`
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `[n_path, 1]`
- **`polmat`** — Polarization transfer matrix, row order as above; `[8, n_path, n_freq]` (full pol) or
  `[2, n_path, n_freq]` (scalar pressure)
- **`freq`** — Input sample frequencies for `path_gain` and `polmat`; `[n_freq, 1]`
- **`freq_new`** — Target output frequencies; `[n_freq_new, 1]`
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `false`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; error if
  unavailable); auto selects GENERIC in this version, see [interpolate_pattern](#interpolate_pattern); default: 0

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_out, n_freq_new]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_out, n_freq_new]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_out, n_freq_new]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency equivalent)
- [generate_speaker](#generate_speaker) (acoustic source construction)

---
<a id="get_channels_planar"></a>
## get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_planar) · [Python](quadriga_lib_api_python.md#channel.planar)

- Computes complex channel coefficients and delays for all TX/RX element pairs across `n_path` propagation paths.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based (angles ignored).
- Polarization transfer matrix `polmat` must be normalized; it is the 2x2 Jones matrix (rows = RX polarisation V/H,
  columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- If `add_fake_los_path` is true, one output path is always added (output size `n_path+1`): slot 1 holds the LOS path,
  with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the TX-RX
  distance; it is moved to slot 1 and its original slot (input index + 1) holds a zero-power path. All other paths
  keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance (between the phase centres) from all path
  lengths before converting to delay; per-element delays of pairs that are closer to each other than the phase
  centres are then negative, down to about -(half the TX array size + half the RX array size) / speed of light; [quantize_delays](#quantize_delays) rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.

### Batch mode:
- If `aod` is a cell array, the function processes `n_snap` snapshots in one call, the snapshots in parallel (OpenMP).
- The path inputs `aod`, `eod`, `aoa`, `eoa`, `path_gain`, `path_length` and `polmat` are then cell arrays of length
  `n_snap`.
- The position and orientation inputs are `[3, n_snap]`, one column per snapshot, or a 3-element vector for all
  snapshots.
- All outputs are cell arrays of length `n_snap`, `[n_snap, 1]`, whose entries have the shapes given below.
- Snapshots without any path produce an empty entry, or only the zero-power LOS path with `add_fake_los_path` (as for
  a single snapshot).

### Usage:
```matlab
[ coeff_re, coeff_im, delay, rx_Doppler ] = quadriga_lib.get_channels_planar( tx_array, rx_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, polmat, ...
    center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports, see [generate_antenna](#generate_antenna)
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports, see [generate_antenna](#generate_antenna)
- **`tx_pos`** — Transmitter position; `[3, 1]`, or `[3, n_snap]` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`, or `[3, n_snap]` in
  batch mode
- **`rx_pos`** — Receiver position; `[3, 1]`, or `[3, n_snap]` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`, or `[3, n_snap]` in
  batch mode
- **`aod`** — Departure azimuth angles; rad; `[n_path, 1]`, or a cell of length `n_snap` in batch mode
- **`eod`** — Departure elevation angles; rad; `[n_path, 1]`, or a cell in batch mode
- **`aoa`** — Arrival azimuth angles; rad; `[n_path, 1]`, or a cell in batch mode
- **`eoa`** — Arrival elevation angles; rad; `[n_path, 1]`, or a cell in batch mode
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`, or a cell in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; `[n_path, 1]`, or a cell in batch mode
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`, or a cell in batch mode
- **`center_freq`** — Center frequency in Hz; 0 (or empty) gives the coefficients at f = 0 (see above); default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `false`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; error if
  unavailable); auto selects GENERIC in this version, see [interpolate_pattern](#interpolate_pattern); default: 0

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = moving toward path, negative = away; `[1, n_path(+1)]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (spherical wave variant accounting for per-element angle differences)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (for generating IEEE compliant channels using `get_channels_planar` internally)
- [generate_antenna](#generate_antenna) (antenna array generator)
- [baseband_freq_response](#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed grid)

---
<a id="get_channels_spherical"></a>
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_spherical) · [Python](quadriga_lib_api_python.md#channel.spherical)

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `polmat`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [qrt_file_read](#qrt_file_read) with more than two interactions) is added to the FBS-LBS segment. Delays and
  phases follow this length; the per-element offsets follow the FBS / LBS geometry.
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
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

### Batch mode:
- If `fbs_pos` is a cell array, the function processes `n_snap` snapshots in one call, the snapshots in parallel
  (OpenMP).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `polmat` are then cell arrays of length `n_snap`, as
  [qrt_file_read](#qrt_file_read) returns them for one frequency.
- The position and orientation inputs are `[3, n_snap]`, one column per snapshot, or a 3-element vector for all
  snapshots.
- All outputs are cell arrays of length `n_snap`, `[n_snap, 1]`, whose entries have the shapes given below.
- Snapshots without any path produce an empty entry, or only the zero-power LOS path with `add_fake_los_path` (as for
  a single snapshot).

### Usage:
```matlab
[ coeff_re, coeff_im, delay, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( tx_array, ...
    rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports, see [generate_antenna](#generate_antenna)
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports, see [generate_antenna](#generate_antenna)
- **`tx_pos`** — Transmitter position, finite; `[3, 1]`, or `[3, n_snap]` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `[3, 1]`, or
  `[3, n_snap]` in batch mode
- **`rx_pos`** — Receiver position, finite; `[3, 1]`, or `[3, n_snap]` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `[3, 1]`, or
  `[3, n_snap]` in batch mode
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`, or a cell of length `n_snap` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`, or a cell of length `n_snap` in batch mode
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`, or a cell of length `n_snap` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `[n_path, 1]`, or a cell of length `n_snap` in batch mode
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`, or a cell of length `n_snap` in
  batch mode
- **`center_freq`** — Center frequency in Hz; 0 (or empty) gives the coefficients at f = 0 (see above); default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `false`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto, 1 = GENERIC,
  2 = AVX2 (single precision; error if unavailable); auto selects GENERIC in this version, see
  [interpolate_pattern](#interpolate_pattern); default: 0

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
- [generate_antenna](#generate_antenna) (antenna array generator)
- [baseband_freq_response](#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed grid)

---

<a id="sec-channel-statistics"></a>
# Channel statistics

---
<a id="acdf"></a>
## acdf
Calculate the empirical averaged cumulative distribution function (CDF)

Also in: [C++](quadriga_lib_api_cpp.md#acdf) · [Python](quadriga_lib_api_python.md#acdf)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each 
  column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- A data set without any finite value gets a NaN column in `cdf_per_set` and is left out of `cdf_avg`, `mu`
  and `sig`; if no set has finite values (possible only with `bins_in`), `cdf_avg`, `mu` and `sig` are NaN
- If `bins_in` is empty, equally spaced bins spanning the data range are generated

### Usage:
```matlab
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; magnitudes below about 1e18, 
  larger values overflow in `sig`; `[n_samples, n_sets]`
- **`bins_in`** — Bin centers; used as-is if non-empty (must be finite and strictly ascending, else an
  error is raised); if empty, equally spaced bins spanning the data range are generated; `[n_bins_in]` or
  empty; default: `[]`
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins_in` 
  are provided; default: 201

### Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data (NaN for a set without finite values); `[n_bins_out, n_sets]`
- **`bins_out`** — Auto-generated bins; copy of `bins_in` when non-empty `bins_in` are provided;
  `[n_bins_out, 1]` with `n_bins_out = n_bins` or `n_bins_out = n_bins_in`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across the data sets with finite values; `[n_bins_out, 1]`
- **`mu`** — Mean of the 0.1–0.9 quantiles across the data sets with finite values; `[9, 1]`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values; `[9, 1]`

---
<a id="calc_angular_spread"></a>
## calc_angular_spread
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

Also in: [C++](quadriga_lib_api_cpp.md#calc_angular_spread) · [Python](quadriga_lib_api_python.md#calc_angular_spread)

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
- When `quantize_deg > 0`, paths within that angular distance are grouped and their powers
  summed before computing spreads

### Usage:
```matlab
[ azimuth_spread, elevation_spread, orientation, phi, theta ] = quadriga_lib.calc_angular_spread( ...
    az, el, powers, wrapping, calc_bank_angle, quantize_deg );
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_path, n_cir]`
- **`el`** — Elevation angles; range -pi/2 to pi/2; `[n_path, n_cir]`
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`wrapping`** — If true, enables spherical rotation; default: false
- **`calc_bank_angle`** — If true, computes optimal bank angle analytically; only used when `wrapping = true`; default: false
- **`quantize_deg`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0 (no quantization);
  otherwise it must be finite and >= 1e-3 deg, else an error is raised

### Outputs:
- **`azimuth_spread`** — RMS azimuth angular spread; `[n_cir, 1]`
- **`elevation_spread`** — RMS elevation angular spread; `[n_cir, 1]`
- **`orientation`** — Euler angles [bank; tilt; heading] of an antenna pointing at the power-weighted mean
  direction (a positive tilt points down); `[3, n_cir]`
- **`phi`** — Rotated azimuth angles, the path azimuths in the local frame of that antenna; `[n_path, n_cir]`
- **`theta`** — Rotated elevation angles; `[n_path, n_cir]`

---
<a id="calc_cross_polarization_ratio"></a>
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

Also in: [C++](quadriga_lib_api_cpp.md#calc_cross_polarization_ratio) · [Python](quadriga_lib_api_python.md#calc_cross_polarization_ratio)

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method: co-pol 
  and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default
- Polarization transfer matrix `polmat` is the 2x2 Jones matrix `M` (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with interleaved real/imaginary parts, 8 rows per path: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its
  receive and transmit polarisation: `M_vv` = rows 1-2, `M_hv` = rows 3-4, `M_vh` = rows 5-6, `M_hh` = rows 7-8
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `polmat` as get_channels_spherical couples antenna patterns,
  `c(r, t) = r^T M t` (plain transpose), with the same circular vectors at both ends, each in that antenna's own
  (e_theta, e_phi) frame: L = (V, H) = (1, +j) / sqrt(2) (LHCP), R = (1, -j) / sqrt(2) (RHCP); names after the IEEE
  definition for the time convention exp(+j 2 pi f t) of the library's coefficients exp(-j 2 pi f tau).
  `M_<rx><tx>`: `M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2`, `M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2`,
  `M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2` (receive L, transmit R), `M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2`.
  Co-polar means the same circular vector at both ends: the LOS matrix `diag(1, -1)` is co-polar (handedness
  kept), `diag(1, 1)` (one mirrored component, as by a single reflection) is cross-polar
- LHCP-XPR and RHCP-XPR are ratios per receive polarisation, like V-XPR and H-XPR
- Normalization of `polmat` does not affect XPR (cancels in the ratio) but does affect `pg`; magnitudes in `polmat` above about 1e18 overflow and give NaN
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR is set to 0
- TX/RX positions may be fixed `[3, 1]` or mobile `[3, n_cir]`

### Usage:
```matlab
[ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, polmat, path_length, tx_pos, rx_pos, ...
    include_los, window_size );
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`polmat`** — Polarization transfer matrices with interleaved real/imag parts; `[8, n_path, n_cir]`
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
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir, 1]`

---
<a id="calc_delay_spread"></a>
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

Also in: [C++](quadriga_lib_api_cpp.md#calc_delay_spread) · [Python](quadriga_lib_api_python.md#calc_delay_spread)

- Paths with power below `p_max / 10^(0.1 * threshold_dB)` are excluded; the default
  of 100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have
  their powers summed before computing the spread; binning is applied before the spread calculation

### Usage:
```matlab
[ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold_dB, granularity );
```

### Inputs:
- **`delays`** — Delays in [s] per CIR; `[n_path, n_cir]`
- **`powers`** — Path powers on linear scale in [W]; `[n_path, n_cir]`
- **`threshold_dB`** — Power threshold in [dB] relative to strongest path; paths below it are excluded; default: 100
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0 (no grouping); must be finite
  and non-negative; when > 0, all delays must be finite and the delay span / granularity must be below 2^53, else an
  error is raised

### Outputs:
- **`ds`** — RMS delay spread in [s] per CIR; `[n_cir, 1]`
- **`mean_delay`** — Mean delay in [s] per CIR; `[n_cir, 1]`

### See also:
- [quantize_delays](#quantize_delays) (for mapping delays to a fixed tap grid)
- [calc_rician_k_factor](#calc_rician_k_factor) (for calculating K-factor)

---
<a id="calc_rician_k_factor"></a>
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

Also in: [C++](quadriga_lib_api_cpp.md#calc_rician_k_factor) · [Python](quadriga_lib_api_python.md#calc_rician_k_factor)

- KF = LOS power / NLOS power; LOS paths are those with length ≤ `dTR + window_size`, where
  `dTR` is the direct TX-RX distance
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is
  set to 0
- TX/RX positions may be fixed `[3, 1]` (reused for all snapshots) or mobile `[3, n_cir]`

### Usage:
```matlab
[ kf, pg ] = quadriga_lib.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size );
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, n_cir]`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS; default: 0.01

### Outputs:
- **`kf`** — Rician K-Factor on linear scale; `[n_cir, 1]`
- **`pg`** — Total path gain (sum of all path powers) in [W]; `[n_cir, 1]`

---

<a id="sec-geometry-based-stochastic-model"></a>
# Geometry-based stochastic model

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 2 to 10 and the sub-cluster split of step 11 (LOS probability, path
loss, outdoor-to-indoor (O2I) penetration loss, large-scale parameters (LSPs), clusters and rays, without antennas) with
the spatial consistency of clause 7.6.3, oxygen absorption (clause 7.6.1) and polarization power variability (clause
7.6.16); [gbsm_single_link_channel](#gbsm_single_link_channel) computes the coefficients of step 11 with the antennas.

**Scenario parameters.** A scenario is a struct of numeric fields (`scenpar`), a path-loss struct (`plpar`: the field
`model` and the model parameters) and optional short names, read from a conf file by [conf_file_read](#conf_file_read). 23 files for
TR 38.901 v19.5.0 come with the library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`,
`_NLOS`, `_LOS_O2I`, `_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the
source tree and in `share/quadriga-lib/config` of an installed package; [conf_file_list](#conf_file_list) finds them. Their comments
give the meaning and unit of each key; [conf_file_write](#conf_file_write) with `write_defaults = true` writes the complete default
set with these comments. Every function that takes `scenpar` accepts the struct of [conf_file_read](#conf_file_read) or
[scenpar_validate](#scenpar_validate) or any subset of its fields (`[]` = all defaults), completes a copy with the defaults of missing
fields and validates it with [scenpar_validate](#scenpar_validate), whose messages pass through, also for fields the function does not
use. A field `SubpathMethod` (QuaDRiGa's structs) must be `'legacy'`.

**Workflow** for one drop of a set of links (terms: see Terms below):

1. [conf_file_read](#conf_file_read): the scenarios, one per LOS and indoor state
2. Per link, by the caller with own SOS generators ([sos_init_phases](#sos_init_phases), [sos_rand](#sos_rand), [sos_randn](#sos_randn)): the indoor
   state, a Uniform(0,1) value at the decorrelation distance `IO_state_lambda` of the O2I files below an indoor ratio
   of the caller's choice (e.g. TR 38.901 Table 7.2-1), and for indoor UTs [o2i_loss](#o2i_loss) (`loss_dB`, `d2d_in`); then the
   LOS state, a Uniform(0,1) value at `LOS_state_lambda` of the NLOS file below `p_los` of [los_probability](#los_probability) at
   `d2d_out = max(d2D - d2d_in, 0)`. The caller chooses the `los_probability` model and the `o2i_loss` models and
   weights (TR 38.901 Tables 7.4.2-1, 7.4.3-2). The states assign each link to a scenario and give the `o2i` flags of
   [ssf_clusters](#ssf_clusters); steps 3 to 6 run per scenario
3. Per BS site and scenario, once, with one seed for all sets: [gbsm_init_phases](#gbsm_init_phases) of the random sets in the table
   below; `'lsf'` with the link positions for the bias removal
4. [gbsm_draw](#gbsm_draw) of each set at the links (the shared `'ray'` set with `center_freq`, the list later passed to
   [ssf_rays](#ssf_rays)); [path_loss](#path_loss) (`pl_dB`, `sf_sigma_dB`; column 11 of the `'lsf'` set as `rnd_he`); [lsp_generate](#lsp_generate)
   with `sf_sigma_dB`, all seven outputs
5. [ssf_clusters](#ssf_clusters) (`n_cluster` is the caller's choice) and [ssf_rays](#ssf_rays) (one ray struct per link), called as in the
   table below
6. Per link and segment: [ssf_scatterers](#ssf_scatterers) if the mobility is `drift`; then [gbsm_single_link_channel](#gbsm_single_link_channel) with the
   link gain (see Terms) as `link_gain_dB` (TR methods: once per ray struct, which holds one frequency; `'quadriga'`:
   once per `i_freq`), or [ssf_paths](#ssf_paths) for other channel functions

| Method | Random sets (`set_id`) | Calls in step 5 |
| --- | --- | --- |
| `'38.901'` | `'lsf'`; `'cluster'` 0 and one per frequency; `'ray'` one per frequency | per frequency `k`: [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) (`i_freq = 1`) with slice `k` of all per-frequency inputs (`center_freq`, `lsp`, `lsp_mu`, `lsp_sigma`, `rnd_freq`, `rnd_ray`, `eod_offset_deg`, `gr_eps_re`, `gr_eps_im`) |
| `'38.901_MF'` | as `'38.901'` | [lsp_generate](#lsp_generate) at the anchor frequency with the same `rnd` gives `lsp_anchor`; [ssf_clusters](#ssf_clusters) once; [ssf_rays](#ssf_rays) per frequency `k` with `i_freq = k` |
| `'quadriga'` | `'lsf'`; `'cluster'` 0; `'ray'` 0 | [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) once for all frequencies |

For moved positions, evaluate [gbsm_draw](#gbsm_draw) (`'lsf'`), [path_loss](#path_loss), [lsp_generate](#lsp_generate) and for indoor UTs [o2i_loss](#o2i_loss)
again with the same SOS phases: they give the link gain (see Terms) and `kf_ratio` = KF(position) / KF(drop) from
slice 2 of `lsp`. Ray structs can also come from path data, with [gbsm_rays_from_paths](#gbsm_rays_from_paths) (e.g. of [qrt_file_read](#qrt_file_read)),
or be filled directly (fields in [ssf_rays](#ssf_rays)); continue with step 6.

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric fields `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (column `l` of `rx_pos`) |
| drop | one realization of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | column of the [ssf_clusters](#ssf_clusters) outputs: 1 LOS, 2 ground reflection (GR, if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the `NumSubPaths` rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| kept ray | a LOS or GR ray, or an NLOS ray with power > 0 at some frequency (at `i_freq` in [gbsm_single_link_channel](#gbsm_single_link_channel)) |
| path | a propagation path as `get_channels_*` take it and QRT files store it; [ssf_paths](#ssf_paths) makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in one output tap: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`'quadriga'`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting; validity in [ssf_paths](#ssf_paths)); `auto`: `phase` for the TR methods, `drift` for `'quadriga'` |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (field `power` of the ray struct): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [ssf_clusters](#ssf_clusters) sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | `-pl_dB - loss_dB + 10·lg(SF)` in dB of a link at a position and frequency (`link_gain_dB`), with `pl_dB` of [path_loss](#path_loss), `loss_dB` of [o2i_loss](#o2i_loss) (0 outdoors) and SF the linear shadow fading, slice 3 of `lsp` |
| random set | the spatially correlated random values of one `set_type` and `set_id` (table in [gbsm_draw](#gbsm_draw)) |
| generator | one spatially correlated random field, a [sum-of-sinusoids (SOS)](#sec-sum-of-sinusoids-random-number-generator) generator; one column of a random set |
| SOS phases | the random phases of the sinusoids of the generators of a random set (`phase`); evaluated at positions, they give its random values. "Phase" alone is the phase of a coefficient or ray |
| block | the generators of one random quantity in a set |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap: rays with equal value form one tap; [ssf_rays](#ssf_rays) gives 3·(s - 1) + i,
with s the slot and i the sub-cluster (0, 1, 2) of a split cluster, else 0 (LOS 0, GR 3, the first NLOS cluster 3
without and 6 with GR). Both are labels, not indices: the wrappers do not shift them.

**Model properties.** UTs closer than `SC_lambda` get correlated clusters and rays (TR 38.901 clause 7.6.3.1). In the TR
methods, the tap fading of separately dropped UTs then stays correlated over metres instead of a few wavelengths
(tap-power correlation 0.5 at about 0.3·`SC_lambda`, 0.1 at 0.8·`SC_lambda`); `'quadriga'` avoids this. In all methods,
the cluster delays and angles of separately dropped UTs change faster than for a moved UT, so their single-frequency
channel and Doppler are wrong at small distances: for close UTs, drop once and move them with [ssf_paths](#ssf_paths) or
[gbsm_single_link_channel](#gbsm_single_link_channel). Across bands, `'quadriga'` uses one ray set, so the fading is correlated; the TR methods
draw coupling, XPR and phases per frequency, so it is independent, also with `'38.901_MF'`, which keeps the delays and
angles.

**Conventions.** `N = NumClusters - 1` is the number of NLOS clusters: `NumClusters` counts the LOS slot, as in
QuaDRiGa. Positions are in m; in the batch functions one column (x, y, z) per link: `rx_pos` is `[3, n_link]` (empty: no
links), `tx_pos` is `[3, 1]` (one Tx for all links) or `[3, n_link]`; heights are the z coordinates. The one-link
functions take a position as a `[3]` vector, and [gbsm_single_link_channel](#gbsm_single_link_channel) one column per snapshot; scatterer
positions are `[3, n]`. Angles are elevations, not TR zenith angles (EOD, EOA, ESD for ZOD, ZOA, ZSD; also in the
conf-file keys): in rad in the GCS, azimuth in [-pi, pi), elevation in [-pi/2, pi/2] with +pi/2 up. Defaults follow TR
38.901 except two deliberate library choices: the SOS preset `Comb256` with bias removal ([gbsm_init_phases](#gbsm_init_phases),
[gbsm_draw](#gbsm_draw)) and `method = 'quadriga'` of [gbsm_rays_from_paths](#gbsm_rays_from_paths); QuaDRiGa variants are options. Units: frequencies
in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, spreads and `_deg` offsets in degrees; conf-file
keys and `scenpar` fields have the units of the file comments (e.g. `PerClusterDS` in ns).

---
<a id="conf_file_list"></a>
## conf_file_list
List the scenarios defined by the conf files in a set of directories

Also in: [C++](quadriga_lib_api_cpp.md#conf_file_list) · [Python](quadriga_lib_api_python.md#conf_file_list)

- Lists the regular files and symbolic links to regular files named `<scenario>.conf`
  (case-sensitive, non-empty scenario name) in each directory, not recursive: the directories in the
  given order, the files of one directory in byte-wise sorted order of their names
- A file name found in an earlier directory is skipped, so a local folder can override the files of
  a shared one
- With `parse_shortnames = true` every line `ShortName = name` that is not a comment adds an entry
  right after its file; other lines are not examined
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if a `ShortName` value is not one name, or if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty
  string `''` is the current directory

### Usage:
```matlab
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
<a id="conf_file_read"></a>
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

Also in: [C++](quadriga_lib_api_cpp.md#conf_file_read) · [Python](quadriga_lib_api_python.md#conf_file_read)

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or
  CRLF line ends and a UTF-8 byte order mark at the start are accepted; keys are case-sensitive
  names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - the numeric scenario parameters (see [scenpar_validate](#scenpar_validate)), one number each; keys that are not
    in the file have their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the
  QuaDRiGa key `GR_epsilon` (the complex ground permittivity comes from [lsp_generate](#lsp_generate)). A key
  stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of
  `SubpathMethod` is an error, also with `validate = false`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`),
  read independent of the locale; `Inf`, `+Inf` and `-Inf` in any case
- An error is raised if the file cannot be opened or is a directory
- An error with file name and line number for a line that is not blank, not a comment and not a
  well-formed assignment (no `=`, not one name before the `=`, a missing value, a word where a
  number is required or the reverse, text after the value, `NaN`, a number beyond the range of a
  double, a `PL_` key without a name, a `ShortName` value stated twice), and with `validate = true`
  for a value that fails [scenpar_validate](#scenpar_validate)
- The outputs differ from C++: the path-loss model `pl_model` is the field `model` of `plpar`, the C++
  outputs `key_order` and `pl_order` set the order of the fields, and `lsp_xcorr_pd` is called `lsp_xcorr_chk`

### Usage:
```matlab
[ scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys ] = ...
    quadriga_lib.conf_file_read( fn, validate );
```

### Inputs:
- **`fn`** — Path to the conf file; string
- **`validate`** — `true`: apply the value checks of [scenpar_validate](#scenpar_validate); `false`: only the line
  grammar is checked; default: true

### Outputs:
- **`scenpar`** — Scenario parameters; struct with all scenario fields as double scalars in the
  canonical order of [scenpar_validate](#scenpar_validate)
- **`plpar`** — Path-loss parameters; struct with the field `model` (string, only if the file states
  `PL_model`) followed by one field per `PL_` key in file order, named without the prefix `PL_`;
  each is a row vector with one or more values; `[]` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite; `false` is no error here, but [lsp_generate](#lsp_generate)
  raises an error for the scenario; logical
- **`short_names`** — Values of all `ShortName` lines in file order; cell array of strings;
  `[n_short, 1]`
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; cell array
  of strings; `[n_ignored, 1]`

### See also:
- [scenpar_validate](#scenpar_validate) (value checks and defaults)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)

---
<a id="conf_file_write"></a>
## conf_file_write
Write a QuaDRiGa scenario configuration file

Also in: [C++](quadriga_lib_api_cpp.md#conf_file_write) · [Python](quadriga_lib_api_python.md#conf_file_write)

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files
  of QuaDRiGa; every written file passes [conf_file_read](#conf_file_read) with `validate = true`, without ignored
  keys and with the same values bit for bit
- The header has the scenario name (the file name without `.conf`), `ref`, the validity range and
  the short names. Up to 10 sections follow, each with an optional line `% See: ref; ref_sec` and,
  for the LSPs and the path-loss model, the formulas as comments; the `plpar` fields are written in
  struct order. Sections by entry of `ref_sec`: 1 large-scale distributions, 2 large-scale
  decorrelation distances, 3 inter-parameter correlations, 4 model parameters, 5 small-scale
  decorrelation distance, 6 ground reflection, 7 absolute TOA, 8 path-loss model, 9 EoD offset, 10
  optional modelling components; 1 to 8 are those of QuaDRiGa, so its 8 section references can be
  passed unchanged
- Numbers are written independent of the locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = false`: only fields whose value differs from the default (bit for bit); a
  section without a written field is left out
- All inputs are validated before the file is opened: the checks of [scenpar_validate](#scenpar_validate); `model`,
  the other fields of `plpar` and the short names are names, the short names distinct; every `plpar`
  field has at least one value and no `NaN`; `fn`, `ref` and `ref_sec` have no line break
- A failed call creates no file and leaves an existing file unchanged; the directory must be writable

### Usage:
```matlab
quadriga_lib.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, ...
    write_defaults );
```

### Inputs:
- **`fn`** — Path of the conf file to write; string
- **`scenpar`** — Scenario parameters; struct with any subset of the fields returned by
  [conf_file_read](#conf_file_read); real scalars of any numeric class or logical; missing fields have their default; `[]` writes
  the default set
- **`plpar`** — Path-loss parameters; struct with the field `model` (string) and one real vector (row or
  column) of any numeric class per further field, named without the prefix `PL_`, as returned by
  [conf_file_read](#conf_file_read); a matrix is an error; written in struct order; default: no path-loss section
- **`short_names`** — Short names of the scenario; cell array of strings or one string; each one name
  of the characters `A-Z`, `a-z`, `0-9`, `_`; distinct; default: none
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; string; default: no reference
- **`ref_sec`** — Reference per section, numbered as listed above; cell array of up to 10 strings;
  an empty or missing entry writes no reference; default: none
- **`valid_range`** — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, UT height in m; columns: lower and upper end, one value is written if both are equal; a
  row with `NaN` is not written; `[4, 2]`; default: none
- **`write_defaults`** — `true`: write all fields; `false`: write only fields that differ from their
  default; default: false

### See also:
- [conf_file_read](#conf_file_read) (read the file back)
- [scenpar_validate](#scenpar_validate) (value checks and defaults)

---
<a id="gbsm_draw"></a>
## gbsm_draw
Spatially correlated normalized random values of one random set at a set of links

Also in: [C++](quadriga_lib_api_cpp.md#gbsm_draw) · [Python](quadriga_lib_api_python.md#gbsm.draw)

- Evaluates the SOS phases of [gbsm_init_phases](#gbsm_init_phases) at the links as [sos_val](#sos_val), one column per generator. The value of a
  link depends neither on the other links of the call nor on the other columns; with `dual_mobility = false` not on
  `tx_pos`
- `dual_mobility = true`: the Tx is a second device on the same fields; its side of each generator uses the mirror
  generator (e.g. departure and arrival quantities exchange), and every column keeps its distribution
- 2D mode (`horizontal = true`, TR): the values are correlated in the x-y plane; links with equal `floor_id` share one
  field, fields of different floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3):
  exactly for IDs in the same group of `n_coeff` consecutive IDs (0 to `n_coeff - 1`, ...), otherwise up to a random
  residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- 3D mode (legacy QuaDRiGa): the height separates the floors, so UTs on different floors are correlated by the ACF of
  their height difference. The per-link statistics are the same in both modes
- An empty set (`NumClusters = 1`) gives `[n_link, 0]` and accepts any empty `phase`

Random sets, with `N = NumClusters - 1`, `M = NumSubPaths`, columns from 1 and blocks from 0; a set never changes size
(new random quantities come in new sets), and a decorrelation distance of 0 is used as 0.1 m:

| `set_type`, `set_id` | `n_gen` | Columns | Decorrelation distance | Used by |
| --- | --- | --- | --- | --- |
| `'lsf'` | 11 | 1 to 8 DS, KF, SF, ASD, ASA, ESD, ESA, XPR and 9 absolute TOA, Normal(0,1); 10 ground permittivity and 11 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (1 to 8), `absTOA_lambda` (9), `SC_lambda` (10), `SF_lambda` (11) | [lsp_generate](#lsp_generate) `rnd`, [path_loss](#path_loss) `rnd_he` (column 11) |
| `'cluster'`, 0 | `9·N` | column `b·N + n` for block `b` and NLOS cluster `n` (from 1): block 0 delay, Uniform(0,1); blocks 1 to 8 Normal(0,1): 1 to 4 sign of AOD, AOA, EOD, EOA, 5 to 8 offset of the same | `SC_lambda` | [ssf_clusters](#ssf_clusters) `rnd` |
| `'cluster'`, `k` | `N` | cluster shadowing at frequency `k`, Normal(0,1) | `SC_lambda` | [ssf_clusters](#ssf_clusters) `rnd_freq`, slice `k` |
| `'ray'`, 0 or `k` | `14·N·M + N` | blocks 0 to 14, below | `SC_lambda`; block 14 of set 0: 1.6·c·c_DS / `PerClusterAS_A` (in rad), within 0.1 m to 10 km | [ssf_rays](#ssf_rays) `rnd_ray`: 0 for `'quadriga'`, `k` as slice `k` for the TR methods |

A `'ray'` set holds block `b` up to 12 for ray `m` of NLOS cluster `n` (both from 1) in column
`b·N·M + (n - 1)·M + m`: 0 to 3 coupling ranks of AOD, AOA, EOD, EOA; 4 XPR; 5 to 8 initial phases θθ, θφ, φθ, φφ; 9 to
12 polarization power variability of the same. Block 13, the XPR per cluster, is in column `13·N·M + n`; block 14, the
ray delay of `'quadriga'`, in column `13·N·M + N + (n - 1)·M + m`. All blocks are Normal(0,1) except block 14,
Uniform(0,1).

### Usage:
```matlab
rnd = quadriga_lib.gbsm_draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, ...
    center_freq, dual_mobility, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, `PerClusterDS*` and
  `PerClusterAS_A`
- **`set_type`** — Set type of [gbsm_init_phases](#gbsm_init_phases); string
- **`phase`** — SOS phases from [gbsm_init_phases](#gbsm_init_phases), finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[3, 1]` or `[3, n_link]`
- **`rx_pos`** — Rx positions in [m]; `[3, n_link]`; empty = no links
- **`set_id`** — Set of [gbsm_init_phases](#gbsm_init_phases): `0` = shared set, `k >= 1` = set of frequency `k`; ignored for `'lsf'`;
  another set with the same `n_gen` is not detected; default: 0
- **`preset`** — Preset of [gbsm_init_phases](#gbsm_init_phases); another preset with the same `n_coeff` is not detected; default:
  `'Comb256'`
- **`floor_id`** — Floor number of the UT per link, ground floor 0 (a label, not shifted; `n_fl - 1` for the UT heights
  of TR 38.901 Table 7.2-1, 0 for outdoor UTs); 2D mode only; `[n_link]`; default: all 0
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz]; required by the shared `'ray'` set, whose block 14 uses
  their mean c_DS ([ssf_rays](#ssf_rays)): pass the same list as to [ssf_rays](#ssf_rays) (another list gives other `'quadriga'` rays);
  ignored and not checked for all other sets; `[n_freq]`
- **`dual_mobility`** — `true`: Tx and Rx both move; default: false
- **`horizontal`** — `true`: 2D mode; `false`: 3D mode; default: true

### Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

### See also:
- [gbsm_init_phases](#gbsm_init_phases) (the SOS phases)
- [lsp_generate](#lsp_generate) (the large-scale parameters from the `'lsf'` set)
- [path_loss](#path_loss) (column 11 of the `'lsf'` set as `rnd_he`)
- [ssf_clusters](#ssf_clusters) (the clusters from the `'cluster'` sets)
- [ssf_rays](#ssf_rays) (the rays from the `'ray'` sets)

---
<a id="gbsm_init_phases"></a>
## gbsm_init_phases
SOS phases of one random set of a BS site and scenario

Also in: [C++](quadriga_lib_api_cpp.md#gbsm_init_phases) · [Python](quadriga_lib_api_python.md#gbsm.init_phases)

- Draws the SOS phases of one random set (set table in [gbsm_draw](#gbsm_draw)): the random phases of the sinusoids of its
  [sum-of-sinusoids (SOS)](#sec-sum-of-sinusoids-random-number-generator) generators, as
  [sos_init_phases](#sos_init_phases); [gbsm_draw](#gbsm_draw) evaluates them at the links. Store `phase` or the seed to evaluate the same fields
  again along tracks or on maps
- The SOS phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `'lsf'`
  sets within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `'lsf'` bias removal (`remove_bias`): [sos_remove_bias](#sos_remove_bias) sets the sample mean of columns 1 to 9 over the positions
  (Rx, and Tx in dual mobility; in 2D mode all on floor 0) to 0, so these means do not vary from drop to drop
- `NumClusters = 1`: all sets but `'lsf'` are empty
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal

### Usage:
```matlab
phase = quadriga_lib.gbsm_init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, ...
    dual_mobility, remove_bias, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances, `NumClusters` and `NumSubPaths` are used
- **`set_type`** — `'lsf'`, `'cluster'` or `'ray'`; string
- **`seed`** — Seed of the site and scenario; integer from -1 to 2^63-1; -1 = system random device, new SOS phases on
  each call; default: -1
- **`set_id`** — `0` = shared set, `k >= 1` = set of frequency `k`; ignored for `'lsf'`; default: 0
- **`preset`** — Sum-of-sinusoids preset, see [sos_preset](#sos_preset); sets `n_coeff`; default: `'Comb256'`
- **`tx_pos`**, **`rx_pos`** *(optional)* — Bias-removal positions in [m] as in [gbsm_draw](#gbsm_draw); `'lsf'` only, else an
  error
- **`dual_mobility`** — `'lsf'` with bias removal: Tx and Rx both move; as in [gbsm_draw](#gbsm_draw); default: false
- **`remove_bias`** — `'lsf'` only: remove the sample mean over the positions; needs `rx_pos` with at least 3 columns,
  else no effect; default: true
- **`horizontal`** — `'lsf'` with bias removal: `true` 2D, `false` 3D mode; as in [gbsm_draw](#gbsm_draw); default: true

### Output:
- **`phase`** — SOS phases in [rad], in [-pi, pi), row = sinusoid, column = generator; slice 1 = Rx side, slice 2 = Tx
  side; `[n_coeff, n_gen, 2]`

### See also:
- [gbsm_draw](#gbsm_draw) (evaluates the SOS phases at the links)
- [sos_init_phases](#sos_init_phases), [sos_val](#sos_val) (SOS phases and evaluation of one generator), [sos_remove_bias](#sos_remove_bias) (its bias removal)

---
<a id="gbsm_rays_from_paths"></a>
## gbsm_rays_from_paths
Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read

Also in: [C++](quadriga_lib_api_cpp.md#gbsm_rays_from_paths) · [Python](quadriga_lib_api_python.md#gbsm.rays_from_paths)

- Returns the ray struct of [ssf_rays](#ssf_rays) (`fallback` empty) from path data in the names and order of [qrt_file_read](#qrt_file_read),
  read with `normalize_polmat = 1` (`uplink = true` gives the reverse link; `normalize_polmat = 0` leaves only the
  free-space gain in `power`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else NLOS. A
  GR path needs `path_type`: for QRT files whose paths were built from GBSM scatterers pass the `n_interact` of
  [qrt_file_read](#qrt_file_read) (interaction count = path type); for ray-traced files the user sets it (values > 2 are an error)
- The GR point (`lbs_pos`) must be within 1 mm of z = 0

### Usage:
```matlab
rays = quadriga_lib.gbsm_rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, method, path_type, tap, gr_eps_re, gr_eps_im );
```

### Inputs:
- **`center_freq`** — Frequencies in [Hz]; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m]; `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`polmat`** — Polarization transfer matrices, rows as in [get_channels_planar](#get_channels_planar); `[8, n_path, n_freq]`
- **`method`** — Value of `rays.method`; sets the GR power law and the default mobility; default: `'quadriga'`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `[n_path]`
- **`tap`** — Tap index; `[n_path]`; default: LOS 0, GR 3, the others 6, 9, ... in path order (3 is kept for GR also
  without a GR path)
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity, given together; needed only where a GR ray is
  recomputed, i.e. in the mobility `drift` at moved ends; `[n_freq]`

### Output:
- **`rays`** — Ray struct with `n_ray = n_path`, see [ssf_rays](#ssf_rays); use with `link_gain_dB = 0` and without `kf_ratio`

### See also:
- [qrt_file_read](#qrt_file_read) (source of the path data)
- [ssf_paths](#ssf_paths) (paths at moved positions)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel of the link)

---
<a id="gbsm_single_link_channel"></a>
## gbsm_single_link_channel
Channel coefficients of one link at one frequency over one segment

Also in: [C++](quadriga_lib_api_cpp.md#gbsm_single_link_channel) · [Python](quadriga_lib_api_python.md#gbsm.single_link_channel)

- Computes the MIMO coefficients of one link with its antennas over the `n_snap` snapshots of one segment at
  `rays.center_freq(i_freq)`: per snapshot, the paths of [ssf_paths](#ssf_paths) at the snapshot positions with the link
  gain and K-factor ratio of the snapshot, for the kept rays (see Terms)
- `'phase'`: [get_channels_planar](#get_channels_planar) with the arrays at the positions of `rays` and the orientations of the snapshot;
  delays, path lengths and angles stay those of the ray-data positions. `'drift'`: [get_channels_spherical](#get_channels_spherical) with the
  arrays at the positions and orientations of the snapshot
- `'taps'`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays (fixed
  per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap per
  element pair is the mean of the rays' delays weighted by `rays.power` at `i_freq` (equal weights where they sum to
  0)
- `'taps'` is exact at the phase center for the TR methods (the rays of a tap share one delay); for `'quadriga'` only
  at the carrier: a tap sums rays of different delays and is frequency-selective within the band, so use `'rays'` for
  wide bands. `'rays'`: one path per kept ray in ray order, exact at any bandwidth; the path data are those of
  `ssf_paths` at the snapshot for these rays
- `n_snap` is the largest column count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  columns, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries (per snapshot at `i_freq`, while `ssf_paths` takes one
  per frequency); one column or entry applies to all snapshots. A static link with a gain per snapshot repeats its
  position
- Powers are not renormalized, and `chan` has no `initial_position`. Validity limits: see `ssf_paths`
- An error is raised for an invalid `tx_array` or `rx_array`, invalid
  `rays`, `'drift'` without scatterers or with a GR ray at moved ends without permittivity or below z = 0, and a column
  or entry count other than 1 or `n_snap`

### Usage:
```matlab
[ chan, par ] = quadriga_lib.gbsm_single_link_channel( tx_array, rx_array, rays, link_gain_dB, ...
    mobility, output, i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, ...
    use_absolute_delays );
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Structs containing the arrayant data with `n_tx` and `n_rx` ports; field layout
  as documented in [generate_antenna](#generate_antenna)
- **`rays`** — Ray struct of the link, see [ssf_rays](#ssf_rays); `[1, 1]` struct
- **`link_gain_dB`** — Link gain (see Terms) at `rays.center_freq(i_freq)` in [dB]; `[n_snap]` or `[1]`
- **`mobility`** — `'phase'`, `'drift'` or `'auto'` (see Terms; `'auto'` is `'drift'` for `'quadriga'` rays, which then
  need [ssf_scatterers](#ssf_scatterers)); default: `'auto'`
- **`output`** — `'taps'` or `'rays'`; default: `'taps'`
- **`i_freq`** — Frequency index in `rays.center_freq`, 1-based; ray data of the TR methods hold one frequency: 1;
  default: 1
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in [m]; `[3, n_snap]` or `[3, 1]`; default: the positions of
  `rays`
- **`tx_orientation`**, **`rx_orientation`** — Euler angles (bank, tilt, heading) in [rad]; `[3, n_snap]` or `[3, 1]`;
  default: 0
- **`kf_ratio`** — KF(snapshot) / KF(drop) at `i_freq`, >= 0; `[n_snap]` or `[1]`; default: 1
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1, see `ssf_paths`; default: `false`
- **`use_absolute_delays`** — Include the LOS delay in all delays (`'phase'`: of the ray-data positions); default:
  `false`

### Outputs:
- **`chan`** — Channel struct, layout as documented in [hdf5_read_channel](#hdf5_read_channel): `center_freq` scalar;
  `tx_pos`, `rx_pos` (the snapshot positions), `tx_orientation`, `rx_orientation` `[3, n_snap]`;
  `coeff_re`, `coeff_im`, `delay` `[n_rx, n_tx, n_tap, n_snap]` (`'taps'`) or `[n_rx, n_tx, n_path, n_snap]`
  (`'rays'`); `'rays'` also `path_gain`, `path_length`, `polmat`, `path_angles` and in `'drift'`
  `fbs_pos`, `lbs_pos`
- **`par`** — Struct with the fields below

### Fields of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `[n_tap, 1]` or `[n_path, 1]` | Tap index per output tap or per path; uint32 |
| `n_ray_tap` | `[n_tap, 1]` | Number of kept rays per tap; uint32; `'taps'` only |
| `fbs_pos`, `lbs_pos` | `[3, n_kept]` | Scatterers of the kept rays at the ray-data positions; `'drift'` with `'taps'` only; single, the precision of path data in HDF5 files |
| `gr_pos` | `[3, n_snap]` | GR point per snapshot; `'drift'` with `'taps'` and a GR ray only; single |
| `link_gain_dB` | `[n_snap, 1]` | Link gain per snapshot |
| `b_med`, `b_eff` | `[2, 1]` | Of [ssf_scatterers](#ssf_scatterers) at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `[2, 1]` | Validity limit per end in [m] (see [ssf_paths](#ssf_paths)): `sqrt(λ·b_eff/8)` in `'phase'`, `0.2·b_med` in `'drift'`; Inf where `b_eff` or `b_med` is NaN |
| `ratio` | `[2, 1]` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | `sqrt(ratio(1)² + ratio(2)²)`; valid up to 1 |

### See also:
- [ssf_rays](#ssf_rays) (the ray struct)
- [ssf_paths](#ssf_paths) (paths of one link at one position, validity limits)
- [ssf_scatterers](#ssf_scatterers) (scatterers for `'drift'`, `b_med` and `b_eff`)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data)

---
<a id="los_probability"></a>
## los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

Also in: [C++](quadriga_lib_api_cpp.md#los_probability) · [Python](quadriga_lib_api_python.md#los_probability)

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS
  and UT heights, and, when `rnd_g` and `center_freq` are given, the soft LOS state of TR 38.901 eq.
  (7.6-18); the LOS state itself (a draw against `p_los`) and its spatial correlation are the
  caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office).
  Where the table leaves cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT >
  23` m; InF-SH and InF-DH give 1 for `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS
  <= hUT`; SMa gives 1 for `hBS <= hUT` and uses `max(h_x - hUT, 0)` in all three terms (the TR only
  in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly
  interpolated; `Suburban` and `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`:
  0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another
  `d_clutter` draw the LOS state at a matching decorrelation distance

### Usage:
```matlab
[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, ...
    d_clutter, r_clutter, h_clutter, rnd_g, center_freq );
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`,
  `38.901_InH_Open`, `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`,
  `38.901_InF_HH`, `38.901_SMa`, `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`,
  `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in [m], finite and >= 0: `d2D - d2d_in` for O2I UTs, clamped
  at 0 by the caller; the 2D distance in indoor scenarios; `[n_link]`
- **`h_bs`** — BS height in [m], finite; `[n_link]` or `[1]` for all links
- **`h_ut`** — UT height in [m], finite; `[n_link]` or `[1]` for all links
- **`d_clutter`** — Clutter size in [m] of the InF and SMa models, scalar; NaN, empty or <= 0: TR
  default (InF-SL and InF-SH 10 m, InF-DL and InF-DH 2 m, SMa 30 m); default: NaN
- **`r_clutter`** — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and
  DH), vegetation density of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1), scalar; NaN or
  empty = default; default: NaN
- **`h_clutter`** — Clutter height in [m] of InF-SH (default 2 m) and InF-DH (6 m), >= 0, scalar;
  NaN or empty = default. The other models ignore the clutter arguments, which are validated for every
  model; default: NaN
- **`rnd_g`** *(optional)* — Normal(0,1) value per link, G of eq. (7.6-18); `[n_link]`
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link, 1]`
- **`los_soft`** — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq.
  (7.6-18), which has the probability `p_los`; `[n_link, n_freq]`; `[]` if `rnd_g` or `center_freq`
  is omitted or empty; `zeros(0, n_freq)` for no link and `center_freq` given

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos_rand](#sos_rand), [sos_randn](#sos_randn) (spatially consistent LOS-state draw and `rnd_g`)

---
<a id="lsp_generate"></a>
## lsp_generate
Large-scale parameters of a set of links from normalized random values

Also in: [C++](quadriga_lib_api_cpp.md#lsp_generate) · [Python](quadriga_lib_api_python.md#lsp_generate)

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values
  of [gbsm_draw](#gbsm_draw) (`'lsf'`), and the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB
  for KF, SF and XPR), with the field prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`,
  `XPR`), `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h/d)` and `f' = max(center_freq/1e9,
  LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) +
    P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR
    38.901 Table 7.5-6 Part-3). ESD adds
    `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)` and is
    limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0
    requires `P_omega + f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and
  XPR; `x` are the columns 1 to 8 of `rnd`, cross-correlated by a root of the LSP cross-correlation
  matrix of `scenpar`, the same for all frequencies (TR 38.901 clause 7.6.5). `rnd = zeros(n_link,
  11)` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu +
  absTOA_sigma·x)` with column 9 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of
  their calibration scenario; for another hall set these fields in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over
  column 10 of `rnd` (0 dry, 0.5 medium, 1 wet) with `center_freq` not limited by `LSP_fc_min`; they
  are not the materials of TR 38.901 Table 7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e +
  EoD_offset_e_gamma·lg f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg
  f')·lg(max(EoD_offset_b, d)) + EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if
  `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi` if `hUT <
  EoD_offset_hcl`
- Outputs that are not requested are not computed

### Usage:
```matlab
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ... 
    quadriga_lib.lsp_generate( scenpar, center_freq, tx_pos, rx_pos, rnd, sf_sigma_dB, cholesky, ...
    clip_spreads );
```

### Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[3, 1]` or `[3, n_link]`
- **`rx_pos`** — Rx positions in [m]; `[3, n_link]`
- **`rnd`** — Random values from [gbsm_draw](#gbsm_draw) (`'lsf'`), finite; column 10 in [0, 1] when 3 or more
  outputs are requested; column 11 is not used; `[n_link, 11]`
- **`sf_sigma_dB`** — SF standard deviation in [dB] (e.g. from [path_loss](#path_loss)), finite and >= 0;
  replaces the value from `scenpar`, also in `lsp_sigma`; `[n_link, n_freq]`; default: the scenario
  value
- **`cholesky`** — `true`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS,
  ASD, ASA, ESD, ESA; XPR last); `false`: symmetric square root (legacy QuaDRiGa); default: true
- **`clip_spreads`** — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause
  7.5 step 4); `lsp_mu` and `lsp_sigma` are not affected; default: true

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 1 DS in [s]; 2 KF, 3 SF and 8 XPR linear; 4 ASD, 5 ASA, 6 ESD,
  7 ESA in [deg]; `[n_link, n_freq, 8]`
- **`abs_toa`** — Absolute TOA offset in [s], 0 if not active; `[n_link, 1]`
- **`gr_eps_re`**, **`gr_eps_im`** — Relative permittivity of the ground, real part and imaginary part >= 0; the
  complex value of TR 38.901 eq. (7.6-40) is `gr_eps_re - j·gr_eps_im`; `[n_link, n_freq]`
- **`eod_offset_deg`** — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** — Mean `mu` in the log domain, slices as `lsp`: lg of [s] for DS, lg of [deg] for the angular
  spreads, [dB] for KF, SF and XPR; required by [ssf_rays](#ssf_rays); `[n_link, n_freq, 8]`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain of `lsp_mu`, >= 0; required by [ssf_rays](#ssf_rays);
  `[n_link, n_freq, 8]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma_dB`)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)
- [ssf_rays](#ssf_rays) (takes `lsp`, `lsp_mu` and `lsp_sigma`)

---
<a id="o2i_loss"></a>
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

Also in: [C++](quadriga_lib_api_cpp.md#o2i_loss) · [Python](quadriga_lib_api_python.md#o2i_loss)

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier
  frequency; the caller adds `loss_dB` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table
  7.4.3-1), `legacy_6GHz` (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz), `car` and
  `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the first model `j` with `rnd(u, 1)` below the sum of the weights up to `j`, else
  the last model with a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 2 and 3 of `rnd` times `d2d_in_max` (`low`, `high`,
  `low_A`, clause 7.4.3.1), column 2 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [sos_rand](#sos_rand) and [sos_randn](#sos_randn) at the correlation distances of TR 38.901 clause
  7.6.3.3: 50 m for column 1, 25 m for columns 2 and 3, 10 m for column 4
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`,
  SMa `low`, `high` and `low_A`. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5)
  with different maximum indoor distances: call `o2i_loss` once per building type

### Usage:
```matlab
[ loss_dB, d2d_in, model_ind ] = quadriga_lib.o2i_loss( center_freq, rnd, models, weights, d2d_in_max );
```

### Inputs:
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`rnd`** — Random values per UT: columns 1 to 3 Uniform in [0, 1] (model choice, two indoor
  distances), column 4 Normal(0,1), finite (deviation `sigma_P`, car loss); `[n_ut, 4]`, `n_ut = 0`
  allowed
- **`models`** — Models to mix, names as listed above; cell array of strings or one string; `n_model`
  entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `[n_model]`
- **`d2d_in_max`** — Maximum indoor 2D distance in [m], finite and >= 0 (`scenpar.O2I_d2D_in_max` of
  the O2I conf files; TR: 25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings);
  default: 25

### Outputs:
- **`loss_dB`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** — Indoor 2D distance in [m], 0 for the car models; `[n_ut, 1]`
- **`model_ind`** — Chosen model per UT, 1-based index into `models`; uint64; `[n_ut, 1]`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)

---
<a id="path_loss"></a>
## path_loss
Path loss and shadow-fading standard deviation of a set of links

Also in: [C++](quadriga_lib_api_cpp.md#path_loss) · [Python](quadriga_lib_api_python.md#path_loss)

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and
  carrier frequency; the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with
  them. The O2I penetration loss is not included (see [o2i_loss](#o2i_loss)). The models are evaluated as
  written for any positive frequency, also outside their validity range
- Notation: `f = center_freq/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D
  distance, at least 0.1 m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) =
  A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std. Fields of `plpar` in parentheses: the
  required ones, after the semicolon the optional ones (`X = v`: default)
- Models (the field `model` of `plpar`):
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb =
    max(hBS, 1.5)`, `hm = max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp
    = 10^((G2 - G1)/(A1 - A2))`; `Aj·lg d2 + Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <=
    bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as
    `winner_los`; `(Aj + Ahj·lg hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1`
    for `f < 1.5`, 2 for `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb =
    max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 =
    sqrt(dBP² + (hb - hm)²)`; `PL1(d3)`, `sf = sig1` for `d2 <= dBP`, else `PL1(dBP3) +
    A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope`, spelled as in QuaDRiGa (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1,
    hE1 = 0, hE2 = 0, sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1` of `E1, hE1` (at most `dBP2`)
    and `dBP2` of `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax,
    Bx, Cx, sig): `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb +
    D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm +
    Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with `LOSmax != 0` (which requires the `dual_slope`
    fields) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at least `Ax·lg d3 + Bx + Cx·lg
    f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for
    `sa <= 0`, else `A·lg d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by
    atmospheric gases over `sa` (1 to 100 GHz, constant beyond) if `usePLa != 0`
  - no model (`plpar` empty or without the field `model`): 0
- Where the model gives no `sf` (an `SF` or `sig` field not set, or a model without one): the SF std of
  [lsp_generate](#lsp_generate) at the geometric distance, so that passing `sf_sigma_dB` to [lsp_generate](#lsp_generate) is neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note
  1, drawn from `rnd_he`, which makes it spatially consistent at the decorrelation distance
  `SF_lambda` (the TR draws it independently per link); the UT height is then clamped at `hE + 0.1`
  instead of `1.1·hE`. Along a track the loss beyond the breakpoint steps where hE changes; evaluate
  `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa),
  links with `d2d_in > 0` use the outdoor part of the distances instead, at least 10 m
- A required field that is missing is an error, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every field that the model reads must have one finite value; the breakpoint factors `E`, `E1`,
  `E2` and `G2n` must be > 0

### Usage:
```matlab
[ pl_dB, sf_sigma_dB ] = quadriga_lib.path_loss( plpar, scenpar, center_freq, tx_pos, rx_pos, d2d_in, rnd_he );
```

### Inputs:
- **`plpar`** — Path-loss parameters; struct as returned by [conf_file_read](#conf_file_read): the field `model`
  (string) and one numeric field per parameter, named without "PL_"; fields not read are ignored; `[]`:
  no model (loss 0)
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[3, 1]` or `[3, n_link]`
- **`rx_pos`** — Rx positions in [m]; `[3, n_link]`
- **`d2d_in`** — Indoor 2D distance in [m] of the UT of each link, legacy QuaDRiGa only (leave it empty
  for TR 38.901, which uses the geometric distance); finite and >= 0; 0 = outdoor link; `[n_link]`;
  default: the geometric distance
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 11 of
  [gbsm_draw](#gbsm_draw) (`'lsf'`); `[n_link]`; default: fixed `hE`

### Outputs:
- **`pl_dB`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma_dB`** — Shadow-fading standard deviation in [dB]; `[n_link, n_freq]`

### See also:
- [lsp_generate](#lsp_generate) (takes `sf_sigma_dB`)
- [o2i_loss](#o2i_loss) (penetration loss, added by the caller)

---
<a id="scenpar_validate"></a>
## scenpar_validate
Validate scenario parameters and complete them with default values

Also in: [C++](quadriga_lib_api_cpp.md#scenpar_validate) · [Python](quadriga_lib_api_python.md#scenpar_validate)

- Checks scenario parameters, e.g. ones built in code, and adds every missing field with its default value, so that
  `scenpar_out` has all 171 fields: the 138 numeric fields of QuaDRiGa and 33 fields for TR 38.901 v19.5.0, which have
  no effect at their default of 0. An empty input (`[]` or a struct without fields) gives the complete default set
- A name that is not one of the fields is an error, also with `validate = false`
- Value checks with `validate = true`; `NaN` is an error for every field:
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
  - All other numeric fields accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
- `LOS_scatter_radius`, `NF_*` and `SNS_*` are checked but not used by any function of this version

### Usage:
```matlab
[ scenpar_out, lsp_xcorr, lsp_xcorr_chk ] = quadriga_lib.scenpar_validate( scenpar_in, validate );
```

### Inputs:
- **`scenpar_in`** — Scenario parameters; struct with any subset of the fields; numeric fields are real
  scalars of any numeric class or logical; default: `[]` (the complete default set)
- **`validate`** — `false`: only complete with defaults, no value checks (`NaN` is accepted); default: true

### Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing fields have their
  default; struct with all fields as double scalars in the canonical order, QuaDRiGa's fields first
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite; `false` also if it contains `NaN`; `false` is no
  error here, but [lsp_generate](#lsp_generate) raises an error for such parameters; logical

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---
<a id="ssf_clusters"></a>
## ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

Also in: [C++](quadriga_lib_api_cpp.md#ssf_clusters) · [Python](quadriga_lib_api_python.md#ssf_clusters)

- Turns the LSPs of [lsp_generate](#lsp_generate) and the cluster random values of [gbsm_draw](#gbsm_draw) into the clusters of each link, by
  one of three methods:
  - `'38.901'` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalization); the K-factor in the scaling factors is
    limited to >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` (columns
    N + 1 to 9·N) at the drop or segment start and keep them along the track
  - `'38.901_MF'`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `'quadriga'`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links; delays and angles depend on the frequency list, so keep it fixed to reproduce a seed

| Property | `'38.901'` | `'38.901_MF'` | `'quadriga'` |
| --- | --- | --- | --- |
| Continuous along tracks | no | NLOS only | yes |
| Same delays and angles at all frequencies | n/a | NLOS, or K the same at all frequencies | yes |
| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one frequency without ground reflection |
| Per-cluster shadowing | yes | yes, per frequency | no |

- `n_slot = NumClusters + G` slots per link (see Terms), `G = 1` if `GR_enabled > 0`. NLOS cluster `n` (from 1) is in
  slot `1 + G + n` and belongs to the columns `b·N + n` of `rnd` (no sorting). Clusters `n > n_cluster(l)` (TR clause
  7.6.15) get delay 0, power 0 and the LOS angles. Identify the LOS path by its slot: NLOS clusters of the TR methods
  can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever `lsp`
  holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; slot 1 holds the geometric LOS direction. Powers are the coefficient
  powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `'38.901'`, `'38.901_MF'` add `abs_toa(l)` to the NLOS delays;
  `'quadriga'` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`'quadriga'`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR eqs.
  (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarization
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must be
  empty, `rnd_freq` and `lsp_anchor` are not used
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are checked

### Usage:
```matlab
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, ...
    rx_pos, lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, ...
    o2i, n_cluster );
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`,
  `GR_enabled`
- **`tx_pos`** — Tx positions in [m]; z >= 0 with ground reflection; `[3, 1]` or `[3, n_link]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[3, n_link]`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate), slices and units as there; DS, ASD, ASA, ESD, ESA (slices 1, 4 to 7) > 0,
  KF (slice 2) >= 0; `'38.901'`: one frequency, call once per frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [gbsm_draw](#gbsm_draw) (`'cluster'`, `set_id = 0`), columns 1 to N in [0, 1];
  `[n_link, 9·N]`
- **`method`** — `'38.901'`, `'38.901_MF'` or `'quadriga'`; default: `'38.901'`
- **`rnd_freq`** *(optional)* — Per-frequency cluster sets of [gbsm_draw](#gbsm_draw), slice `k` from `set_id = k` (`'38.901'`:
  one slice, the set of the frequency in `lsp`); required by the TR methods; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz) from [lsp_generate](#lsp_generate); required by
  `'38.901_MF'`; `[n_link, 1, 8]`
- **`eod_offset_deg`** — EoD offset in [deg] from [lsp_generate](#lsp_generate); used by `'38.901'` and `'quadriga'` in NLOS
  scenarios; `[n_link, n_freq]`; default: 0
- **`abs_toa`** — Absolute TOA offset in [s] from [lsp_generate](#lsp_generate), >= 0; required by `'quadriga'` in NLOS scenarios
  with `absTOA_mu > -30`; `[n_link]`; default: no offset
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im` from [lsp_generate](#lsp_generate),
  real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** — O2I flag per link (mean arrival elevation 0), logical or numeric; used by `'38.901'` in NLOS scenarios;
  `[n_link]`; default: no O2I link
- **`n_cluster`** — Active NLOS clusters per link (a count) in `[min(max(1, NumClusters_min - 1), N), N]`; `[n_link]`;
  default: N

### Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_freq]` for `'38.901_MF'`, else `[n_link, n_slot, 1]`
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`eod`** — Azimuth in [-pi, pi) and elevation in [-pi/2, pi/2] of departure in [rad]; sizes as `delay`
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in [rad], ranges as departure; sizes as `delay`
- **`n_active`** — Number of NLOS clusters with power > 0; uint32; `[n_link, 1]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [ssf_rays](#ssf_rays) (the rays of the clusters)
- [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread) (spreads of the output)

---
<a id="ssf_paths"></a>
## ssf_paths
Antenna-free paths of one link at one position

Also in: [C++](quadriga_lib_api_cpp.md#ssf_paths) · [Python](quadriga_lib_api_python.md#ssf_paths)

- Computes the antenna-free paths (gains, lengths, polarization matrices, angles and scatterers) of the kept rays of
  `rays` (see Terms) at new Tx and Rx positions (default: the ray-data positions), LOS and GR also at zero power; paths
  in ray order, `ray_ind` maps them to the rays (e.g. `rays.tap(ray_ind)`)
- `'phase'`: use the outputs with [get_channels_planar](#get_channels_planar) and the arrays at the positions of `rays`; the displacement
  phase of TR 38.901 eq. (7.5-22) is in `polmat`. With the arrays at the new positions the LOS path is no longer identified
  and the displacement phase counts twice, without an error
- `'drift'`: use the outputs at the new positions with [get_channels_spherical](#get_channels_spherical) or [get_channels_multifreq](#get_channels_multifreq); needs
  the scatterers of [ssf_scatterers](#ssf_scatterers)

| Quantity | `'phase'` | `'drift'` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `polmat` | Ray `polmat` times the displacement phase | Ray `polmat`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `[3, 0]` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `'drift'` with an end moved (any coordinate differs from `rays`): GR power by the power law of `rays.method`
  (`(d3D / d_GR)²` for the TR methods, 1 for `'quadriga'`), GR `polmat` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times `10^(link_gain_dB / 10)`;
  with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each frequency) over each path's own
  length, without the delay offset τ_Δ of eq. (7.6-1) and at the carrier only (no in-band variation of eq. (7.6-2))
- Validity per moving end, with `b_eff` and `b_med` of [ssf_scatterers](#ssf_scatterers): `'phase'` up to about `sqrt(λ · b_eff / 8)`,
  typically 0.1 to 1 m at the carrier, about 0.15 to 0.35 m at the edges of a 400 MHz band; `'drift'` up to about
  `0.2 · b_med`, typically 1.5 to 12 m, new segment from `0.5 · b_med`; in `'drift'` each path length changes by at most
  the displacement. [gbsm_single_link_channel](#gbsm_single_link_channel) reports the displacement relative to the limit
- An error is raised for invalid input; in `'drift'` also without scatterers, and for a GR ray at moved ends without
  permittivity in `rays` or with an end below z = 0

### Usage:
```matlab
[ path_gain, path_length, polmat, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind ] = ...
    quadriga_lib.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen );
```

### Inputs:
- **`rays`** — Ray struct of the link, see [ssf_rays](#ssf_rays); `[1, 1]` struct
- **`link_gain_dB`** — Link gain (see Terms) at the position in [dB], per frequency; 0 for data from paths; `[n_freq]`
- **`mobility`** — `'phase'`, `'drift'` or `'auto'` (see Terms; `'auto'` is `'drift'` for `'quadriga'` rays, which then
  need [ssf_scatterers](#ssf_scatterers)); default: `'auto'`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`; default: the positions of `rays`
- **`kf_ratio`** — KF(position) / KF(drop), >= 0, per frequency; `[n_freq]`; default: 1
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`

### Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path, 1]`
- **`polmat`** — Polarization transfer matrices, rows as in [get_channels_planar](#get_channels_planar); `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in [rad], GCS; `[n_path, 1]`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in [m]; `[3, n_path]`, `[3, 0]` in `'phase'`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS; uint32; `[n_path, 1]`
- **`ray_ind`** — Index of the ray in `rays`, 1-based; uint64; `[n_path, 1]`

### See also:
- [ssf_scatterers](#ssf_scatterers) (scatterers for `'drift'`, validity scales `b_eff` and `b_med`)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients of one link over a segment)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data, e.g. of a QRT file)

---
<a id="ssf_rays"></a>
## ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

Also in: [C++](quadriga_lib_api_cpp.md#ssf_rays) · [Python](quadriga_lib_api_python.md#ssf_rays)

- Splits the clusters of [ssf_clusters](#ssf_clusters) into rays and returns one ray struct per link (fields below): delays,
  coefficient powers, angles, polarization transfer matrices with initial phases, path types and tap indices;
  `fbs_pos`, `lbs_pos` and `fallback` stay empty (see [ssf_scatterers](#ssf_scatterers))
- A pure function of the clusters, the LSPs and the random values, without a seed: a link does not depend on the other
  links of the call
- Ray layout per link, fixed per scenario: ray 1 LOS, 2 GR (if `GR_enabled`), then ray `m` of NLOS cluster `n` (both
  from 1) at `1 + G + (n - 1)·M + m` (`G` = 1 with GR, `M` = `NumSubPaths`); clusters at power 0 keep their rays at
  power 0
- LOS and GR rays: delay, angles and power of their slot; Jones matrix diag(1, -1) (LOS), GR as in [ssf_clusters](#ssf_clusters)
- NLOS rays: power P_n / `NumSubPaths`; offsets of Table 7.5-3 scaled by the cluster spreads `PerClusterAS_D`,
  `PerClusterAS_A`, `PerClusterES_A` and the cluster ESD of eq. (7.5-20), `PerClusterES_D_scale`·10^(`lsp_mu` of
  ESD), or `PerClusterES_D` if the scale is 0; coupling (step 8): within a cluster or sub-cluster, the ray with the k-th
  smallest value of a quantity's rank block gets the k-th smallest offset, separately for AOD, AOA, EOD and EOA
- c_DS, the cluster DS of Table 7.5-6: `max(PerClusterDS_min, PerClusterDS + PerClusterDS_gamma·lg f')` ns with
  `f' = max(center_freq/1e9, PerClusterDS_fc_min)`; 3.91 ns where `PerClusterDS` = 0
- `NumSubPaths` other than 20: the offsets of QuaDRiGa-NG (the MATLAB QuaDRiGa fork that uses this library) below 20
  (mean 0, rms 1), Laplacian quantiles at the rms of Table 7.5-3 above 20; no sub-clusters

| Method | `'38.901'`, `'38.901_MF'` | `'quadriga'` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray`; `'38.901'` takes one frequency | all; one geometry, per frequency powers, XPR and Jones matrices; the shared `'ray'` set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-5) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`NumSubPaths` = 20), else cluster | cluster |
| c_DS, cluster ESD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `'additive'`: offsets added to azimuth and elevation, elevation folded at the poles (step 7) | `'rotation'`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `'38.901'`: XPR and four initial phases per ray (steps 9, 10) | `'quadriga'`: rotation polarization model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `false` | `true`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma` (field default 0: no effect); `pol_model = '38.901'` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq` (equal powers: lower index first),
  rays at the positions 1-10, 11-16 and 17-20 (TR rays 1-8, 19, 20; 9-12, 17, 18; 13-16); every tap is a contiguous
  block of rays
- `pol_model = 'quadriga'`: the matrix refers to the end with the larger z, then x, then y (coordinates within 1 mm
  count as equal; the Tx if all tie), so UTs dropped separately at equal heights may get transposed matrices

### Ray struct:
One link at one drop. All fields are present on output (empty where empty). As input, missing optional fields are empty,
other missing or unknown fields are an error; integer fields accept any real class. The struct has scatterers when
`fbs_pos` has `n_ray` columns (a struct with no rays counts as having them). Rays added from another source take tap
indices the struct does not use; at most one LOS and one GR ray and one `method` for all rays. [ssf_scatterers](#ssf_scatterers)
overwrites given scatterers, so solve rays without scatterers separately and concatenate the fields.

| Field | Size | Type | Meaning | Required |
| --- | --- | --- | --- | --- |
| `method` | string | char | `'38.901'`, `'38.901_MF'` or `'quadriga'`; GR power law and default mobility | yes |
| `tx_pos`, `rx_pos` | `[3, 1]` | double | Tx and Rx position the ray data refer to in [m] | yes |
| `center_freq` | `[n_freq, 1]` | double | Carrier frequencies in [Hz] | yes |
| `delay` | `[n_ray, 1]` | double | Delays in [s] relative to the LOS delay d3D / c, d3D the Tx-Rx distance | yes |
| `power` | `[n_ray, n_freq]` | double | Coefficient powers (linear) without link gain; data from paths: path gain | yes |
| `aod`, `eod`, `aoa`, `eoa` | `[n_ray, 1]` | double | Departure and arrival azimuth and elevation in [rad], GCS | yes |
| `polmat` | `[8, n_ray, n_freq]` | double | Polarization transfer matrices incl. initial phases, rows as in [get_channels_planar](#get_channels_planar) | yes |
| `path_type` | `[n_ray, 1]` | uint32 | 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted) | yes |
| `tap` | `[n_ray, 1]` | uint32 | Tap index (a code, not shifted); rays with equal index form one tap | yes |
| `gr_eps_re`, `gr_eps_im` | `[n_freq, 1]` or empty | double | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | double | First- and last-bounce scatterer positions in [m] | no |
| `fallback` | `[n_ray, 1]` or empty | uint32 | 1 = ray lengthened by the solver fallback | no |

### Usage:
```matlab
rays = quadriga_lib.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, ...
    power, aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps_re, ...
    gr_eps_im, pol_variability, random_phase );
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumSubPaths`, the `PerCluster*` fields, `GR_enabled`,
  `PolVar_sigma`
- **`tx_pos`** — Tx positions in [m]; `[3, 1]` or `[3, n_link]`
- **`rx_pos`** — Rx positions in [m]; `[3, n_link]`
- **`center_freq`** — Carrier frequencies in [Hz]; `[n_freq]`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [lsp_generate](#lsp_generate), slices and units as
  there; used: the ESD mean and the XPR mean and std, the XPR LSP for `pol_model = 'quadriga'`; `[n_link, n_freq, 8]`
- **`delay`**, **`power`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays, powers and angles, outputs 1 to
  6 of [ssf_clusters](#ssf_clusters) in this order (`n_slot` slots); `power` `[n_link, n_slot, n_freq]`, the others
  `[n_link, n_slot, n_freq]` for `'38.901_MF'`, else `[n_link, n_slot, 1]`
- **`rnd_ray`** — `'ray'` sets from [gbsm_draw](#gbsm_draw) (`n_gen` columns); TR methods: slice `k` = set `k`,
  `[n_link, n_gen, n_freq]`; `'quadriga'`: the shared set (`set_id = 0`), `[n_link, n_gen]`
- **`method`** — Method of [ssf_clusters](#ssf_clusters): `'38.901'`, `'38.901_MF'` or `'quadriga'`; default: `'38.901'`
- **`i_freq`** — TR methods: index into `center_freq`, 1-based; `'quadriga'`: must be 1; default: 1
- **`angle_model`**, **`pol_model`** — See the option table; default: `'additive'`, `'38.901'`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [lsp_generate](#lsp_generate); required if
  `GR_enabled`, else ignored; `[n_link, n_freq]`
- **`pol_variability`** — TR 7.6.16 polarization power variability, see the option table; default: `false`
- **`random_phase`** — `false`: all initial phases 0; default: `true`

### Output:
- **`rays`** — Ray structs, one per link, TR methods at frequency `i_freq` only; `[1, n_link]` struct array

### See also:
- [ssf_clusters](#ssf_clusters) (the clusters)
- [ssf_scatterers](#ssf_scatterers) (scatterer positions and validity scales of one link)
- [ssf_paths](#ssf_paths) (paths of one link)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients from a ray struct)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray struct from path data)

---
<a id="ssf_scatterers"></a>
## ssf_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

Also in: [C++](quadriga_lib_api_cpp.md#ssf_scatterers) · [Python](quadriga_lib_api_python.md#ssf_scatterers)

- Places the scatterers that the mobility `drift` needs and returns the validity scales of [ssf_paths](#ssf_paths)
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal distances
  from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx distance) and the
  angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 23 % of the rays (most in the TR methods without absolute TOA) and lengthens them by up to
  25 ns (median below 5 ns), the DS of a link by up to 3 ns; `b_med` can then equal `d_min`
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular point
  on z = 0, which requires both ends at z >= 0
- `rays_out` is a copy of `rays` (all fields; unchanged fields keep their class) with new `fbs_pos`, `lbs_pos` and
  `fallback`; `rays` is not changed, also when it already holds scatterers
- `b_med`, the scale of `drift`: the lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS
  rays with power > 0, fallback rays included; NaN without NLOS power. `b_eff`, the scale of `phase`:
  `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0`, with `b` the end-to-scatterer distance (NLOS), d3D
  (LOS) or the GR path length (GR)
- One link per call; loop over the elements of a struct array: `rays(l) = quadriga_lib.ssf_scatterers( rays(l) )`

### Usage:
```matlab
[ rays_out, b_med, b_eff ] = quadriga_lib.ssf_scatterers( rays, d_min );
```

### Inputs:
- **`rays`** — Ray struct of one link, see [ssf_rays](#ssf_rays); `[1, 1]` struct
- **`d_min`** — Minimum distance of the scatterers to both ends in [m], finite and > 0; default: 2

### Outputs:
- **`rays_out`** — Ray struct with `fbs_pos`, `lbs_pos` `[3, n_ray]` and `fallback` `[n_ray, 1]` (uint32)
- **`b_med`**, **`b_eff`** — Validity scales in [m], rows Tx and Rx, column `k` for frequency `k`; `[2, n_freq]`

### See also:
- [ssf_rays](#ssf_rays) (ray data of a set of links)
- [ssf_paths](#ssf_paths) (mobility `drift` needs the scatterers)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (reports `b_med`, `b_eff` and the displacement ratios)

---

<a id="sec-math-functions"></a>
# Math functions

---
<a id="calc_rotation_matrix"></a>
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

Also in: [C++](quadriga_lib_api_cpp.md#calc_rotation_matrix) · [Python](quadriga_lib_api_python.md#calc_rotation_matrix)

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, heading) in column-major order (9 elements
  per orientation)
- Single-precision input is cast to double; output is always double

### Usage:
```matlab
rotation = quadriga_lib.calc_rotation_matrix( orientation, transposed );
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, heading); `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`transposed`** — Returns the transpose of the rotation matrix; logical; default: false

### Output:
- **`rotation`** — Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9, 1]`

---
<a id="fast_cart2geo"></a>
## fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

Also in: [C++](quadriga_lib_api_cpp.md#fast_cart2geo) · [Python](quadriga_lib_api_python.md#fast_cart2geo)

- Computes: `len = sqrt(x² + y² + z²)`, `az = atan2(y, x)`, `el = asin(clamp(z / len, -1, 1))`
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- `z/len` is clamped to [-1, 1] before `asin` to guard against `len == 0` and rounding artifacts
  pushing `abs(z/len)` slightly above 1
- Azimuth in [-pi, pi], with the special values of `atan2` (e.g. `atan2(0, -0) = pi`); NaN input gives NaN
  in `az`, `el` and `len`
- The AVX2 kernel computes the angles in single precision; `len` and `z/len` in double precision
- `x`, `y` and `z` have the same size; the outputs have that size; empty input gives empty outputs

### Usage:
```matlab
[ az, el, len ] = quadriga_lib.fast_cart2geo( x, y, z, use_kernel );
```

### Inputs:
- **`x`** — X-coordinates; up to 4 dimensions
- **`y`** — Y-coordinates; size of `x`
- **`z`** — Z-coordinates; size of `x`
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 
  (throws if AVX2 unavailable); other values throw; default: 0

### Outputs:
- **`az`** — Azimuth angles in radians; size of `x`
- **`el`** — Elevation angles in radians; size of `x`
- **`len`** — Euclidean vector length `sqrt(x² + y² + z²)`; size of `x`

---
<a id="fast_geo2cart"></a>
## fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

Also in: [C++](quadriga_lib_api_cpp.md#fast_geo2cart) · [Python](quadriga_lib_api_python.md#fast_geo2cart)

- Conversion: `x = cos(el) cos(az) len`, `y = cos(el) sin(az) len`, `z = sin(el) len`
- Optional outputs `sin_az`, `cos_az`, `sin_el`, `cos_el` return intermediate sin/cos values; omit from the
  output list to skip their computation
- The AVX2 kernel (selected by `use_kernel=0` where available, forced by `use_kernel=2`) computes sine and
  cosine in single precision (angles reduced in double precision first; max. error about 1.3e-7 for angles up
  to 1e10), results widened back; `use_kernel=1` keeps full double precision
- `el` and `len` have the same size as `az`; the outputs have that size; empty input gives empty outputs

### Usage:
```matlab
[ x, y, z, sin_az, cos_az, sin_el, cos_el ] = quadriga_lib.fast_geo2cart( az, el, len, use_kernel );
```

### Inputs:
- **`az`** — Azimuth angles in radians; up to 4 dimensions
- **`el`** — Elevation angles in radians; size of `az`
- **`len`** — Euclidean vector length sqrt(x^2 + y^2 + z^2); size of `az`; default: 1
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); other values throw; default: 0

### Outputs:
- **`x`** — X-coordinates; size of `az`
- **`y`** — Y-coordinates; size of `az`
- **`z`** — Z-coordinates; size of `az`
- **`sin_az`** — sin(az); size of `az`
- **`cos_az`** — cos(az); size of `az`
- **`sin_el`** — sin(el); size of `az`
- **`cos_el`** — cos(el); size of `az`

---
<a id="fast_sincos"></a>
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

Also in: [C++](quadriga_lib_api_cpp.md#fast_sincos) · [Python](quadriga_lib_api_python.md#fast_sincos)

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for
  abs(x) <= 1e10 (the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; Inf and NaN give NaN
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision, also for double input
- Request one or two outputs to control which results are returned
- With one output, set the optional `cos_only` flag to `true` to return cosine instead of sine

### Usage:
```matlab
[sin_x, cos_x] = quadriga_lib.fast_sincos(x);
sin_x = quadriga_lib.fast_sincos(x);
cos_x = quadriga_lib.fast_sincos(x, true);
[sin_x, cos_x] = quadriga_lib.fast_sincos(x, [], use_kernel);
```

### Inputs:
- **`x`** — Input angles; radians; vector, matrix or 3-D array
- **`cos_only`** — For a single output: `true` returns `cos(x)`; `false` or `[]` returns `sin(x)`; default: false
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`sin_x`** — sin(x); single; size of `x`
- **`cos_x`** — cos(x); single; size of `x`

---
<a id="interp"></a>
## interp
Perform linear interpolation (1D or 2D) on single or multiple data sets

Also in: [C++](quadriga_lib_api_cpp.md#interp_2D) · [Python](quadriga_lib_api_python.md#interp)

- Interpolates given input data at specified output points.
- Outside the range of the data sampling points, the nearest end value is returned (constant extrapolation);
  output sampling points that are NaN give NaN.
- The data sampling points need not be sorted; repeated values are allowed; they must be finite.
- Supports single and multiple data sets.
- Data types: `single` or `double`

### Usage:

```matlab
output = quadriga_lib.interp( x, y, input, x_new, y_new );    % 2D case

output = quadriga_lib.interp( x, [], input, x_new );           % 1D case
```

### Inputs:
- **`x`** — Data x-axis sampling points; Length: `[nx]`
- **`y`** — Data y-axis sampling points; Length: `[ny]`
- **`input`** — Input data array/matrix; `[ny, nx, ne]` or `[1, nx, ne]` for 1D case; 3rd dimension
  enables interpolation for multiple datasets simultaneously.
- **`x_new`** — Output x-axis sampling points; Length: `[nx_new]`
- **`y_new`** — Output y-axis sampling points; Length: `[ny_new]`; required when `y` is not empty, empty or omitted
  when `y` is empty

### Output:
- **`output`**  —  Interpolated data `[ny_new, nx_new, ne]` or `[1, nx_new, ne]` for 1D case

---

<a id="sec-miscellaneous-tools"></a>
# Miscellaneous / Tools

---
<a id="version"></a>
## version
Returns the quadriga-lib version number

Also in: [C++](quadriga_lib_api_cpp.md#quadriga_lib_version) · [Python](quadriga_lib_api_python.md#version)

- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  a suffix `_AVX2` is added after the version number
- If Quadriga-Lib was compiled with CUDA support and a CUDA-capable GPU is available,
  a suffix `_CUDA` is added after the version number

### Usage:
```matlab
version = quadriga_lib.version;
```

### Output:
- **`version`** — Version string (e.g. "0.11.5_AVX2_CUDA")

---
<a id="write_png"></a>
## write_png
Write a data matrix to a color-coded PNG file

Also in: [C++](quadriga_lib_api_cpp.md#write_png) · [Python](quadriga_lib_api_python.md#write_png)

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NaN`
- The auto-detected range ignores `NaN` and `Inf`; `Inf` and `-Inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value) maps every value to the first colour
- Raises an error for empty `data`, an infinite `min_val` or `max_val`, and for `min_val > max_val` (both given)
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

### Usage:
```matlab
quadriga_lib.write_png( fn, data, colormap, min_val, max_val, log_transform );
```

### Inputs:
- **`fn`** — Output `.png` file path; string
- **`data`** — Input data matrix; `[n_rows, n_cols]`
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`min_val`** — Lower clipping bound; NaN: data minimum; default: `NaN`
- **`max_val`** — Upper clipping bound; NaN: data maximum; default: `NaN`
- **`log_transform`** — Apply 10·log10(data) before mapping; non-positive values map to the minimum color; logical; default: false

---

<a id="sec-site-specific-simulation-tools"></a>
# Site-specific simulation tools

---
<a id="calc_diffraction_gain"></a>
## calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

Also in: [C++](quadriga_lib_api_cpp.md#calc_diffraction_gain) · [Python](quadriga_lib_api_python.md#calc_diffraction_gain)

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided
  into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping
  triangles whose bounding box does not intersect the TX-RX path
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](formats.md)

### Usage:
```matlab
[ gain, polmat, coord ] = quadriga_lib.calc_diffraction_gain( orig, dest, mesh, mtl_ind, mtl_prop, ...
    center_freq, lod, verbose, sub_mesh_ind, use_kernel, gpu_id, scalar_mode, thin_slab_threshold, ...
    apply_fspl, apply_phase );
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `csv_ind` output of [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output of
  [obj_file_read](#obj_file_read)); each field holds a vector of length `n_mtl`
- **`center_freq`** — Center frequency in Hz; in scalar mode the acoustic frequency
- **`lod`** — Level of detail (0–7), controls `n_path` and `n_seg`; 0 = direct ray only (`n_path = 1`, `n_seg = 1`);
  1–7 as in the table of [generate_diffraction_paths](#generate_diffraction_paths); default: 2
- **`verbose`** — Verbosity level; default: 0 (no output)
- **`sub_mesh_ind`** — 1-based sub-mesh index for acceleration; see [triangle_mesh_segmentation](#triangle_mesh_segmentation); `[n_sub, 1]`;
  default: `[]` (not using sub-meshes)
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; auto as in [ray_triangle_intersect](#ray_triangle_intersect); default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0
- **`scalar_mode`** — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging; default: `false` (EM mode)
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold; 0 = resolve always,
  1 = resolve never; default: 0
- **`apply_fspl`** — If `true`, folds the geometric spreading loss into `gain`, making it an absolute
  propagation gain instead of a value relative to FSPL. The reference distance is the direct TX-RX
  distance, matching the `path_length` used by [qrt_file_read](#qrt_file_read) for a
  transmission-only path. EM mode: `32.45 + 20*log10(f_GHz) + 20*log10(d_m)` dB; scalar mode:
  `20*log10(d_m) + alpha(f)*d_m` dB, with `alpha` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm; default: `false`
- **`apply_phase`** — If `true`, multiplies `polmat` by the propagation phase `exp(-j*2*pi*d/lambda)`
  using the same direct TX-RX distance. The wavelength follows from `center_freq` and the medium wave
  speed (speed of light in EM mode, speed of sound in scalar mode). The term has unit magnitude, so
  `polmat` stays normalized. This is the phase [get_channels_spherical](#get_channels_spherical) would otherwise add from
  `path_length`; default: `false`

### Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear power scale; `[n_pos, 1]`. Relative to FSPL by
  default (bounded by 0 and 1), absolute propagation gain when `apply_fspl = true`
- **`polmat`** — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase; `gain` holds the power. EM mode:
  interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx), the row order
  of `polmat` in [get_channels_spherical](#get_channels_spherical); `[8, n_pos]`; scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]` with `n_seg` from `lod`
  (e.g. `[3, 2, n_pos]` for lod 2, `[3, 9, n_pos]` for lod 7, `[3, 0, n_pos]` for lod 0)

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths) (controls path/segment count via `lod`)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (generates `sub_mesh_ind`)
- [obj_file_read](#obj_file_read) (defines `mtl_prop` format)
- [qrt_file_read](#qrt_file_read) (same path-loss model, used by `apply_fspl`)
- [get_channels_spherical](#get_channels_spherical) (applies the same propagation phase as `apply_phase`)

---
<a id="cube"></a>
## cube
Construct a triangulated cube mesh

Also in: [C++](quadriga_lib_api_cpp.md#cube) · [Python](quadriga_lib_api_python.md#cube)

- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1)
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 · n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with obj_file_write
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- A negative scale mirrors the cube; when the product of the scales is negative, the winding is swapped so that
  the normals stay outward (as Blender's "Apply Scale")

### Usage:
```matlab
mesh = quadriga_lib.cube( scale, rotation, location, n_div );
```

### Inputs:
- **`scale`** — Length 1 scales all axes uniformly; length 3 scales {x,y,z}
  independently; default: `[]` (1, no scaling)
- **`rotation`** — Euler angles about {x,y,z}, applied as R = Rz·Ry·Rx (Blender
  XYZ); length 3; default: `[]` (no rotation)
- **`location`** — Translation {x,y,z}; length 3; default: `[]` (origin)
- **`n_div`** — Number of subdivisions per edge; yields 12 · n_div^2 triangles;
  default: 1

### Output:
- **`mesh`** — Triangle mesh; each row holds {x1,y1,z1,x2,y2,z2,x3,y3,z3}; `[12 · n_div^2, 9]`

### See also:
- [icosphere](#icosphere)
- [obj_file_write](#obj_file_write)

---
<a id="generate_diffraction_paths"></a>
## generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

Also in: [C++](quadriga_lib_api_cpp.md#generate_diffraction_paths) · [Python](quadriga_lib_api_python.md#generate_diffraction_paths)

- Generates inputs required by [calc_diffraction_gain](#calc_diffraction_gain): elliptic-arc paths sampling
  the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the
  semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so
  sum(prod(weights,3),2) = 1

### Usage:
```matlab
[ rays, weights ] = quadriga_lib.generate_diffraction_paths( orig, dest, center_freq, lod, scalar_mode );
```

### Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`center_freq`** — Center frequency in Hz; in scalar mode the acoustic frequency
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
- **`scalar_mode`** — `true` = acoustic: the Fresnel ellipsoids use the wavelength `343 / center_freq`;
  default: `false` (EM, `299792458 / center_freq`)

### Outputs:
- **`rays`** — Coordinates of path waypoints (x, y, z stacked along the 4th dimension, endpoints
  excluded); `[n_pos, n_path, n_seg-1, 3]`
- **`weights`** — Per-segment weights; `[n_pos, n_path, n_seg]`

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain) (consumes the output of this function)

---
<a id="icosphere"></a>
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

Also in: [C++](quadriga_lib_api_cpp.md#icosphere) · [Python](quadriga_lib_api_python.md#icosphere)

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

### Usage:
```matlab
[ center, length, vert, direction ] = quadriga_lib.icosphere( n_div, radius, direction_xyz );
```

### Inputs:
- **`n_div`** — Number of subdivisions; generates 20 · n_div² faces
- **`radius`** — Radius of icosphere in meters; finite and positive
- **`direction_xyz`** — Output directions in Cartesian (true) or spherical azimuth/elevation (false); default: false

### Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `[n_faces, 3]`
- **`length`** — Distance from origin to face plane; equals the magnitude of each `center` vector; `[n_faces, 1]`
- **`vert`** — Vertex offsets from face center `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_faces, 9]`
- **`direction`** — Directions of the vertex rays; spherical `{az1,el1,az2,el2,az3,el3}` or Cartesian
  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per `direction_xyz` flag; `[n_faces, 6]` or `[n_faces, 9]`

---
<a id="mitsuba_xml_file_write"></a>
## mitsuba_xml_file_write
Write a triangular mesh to a Mitsuba 3 XML scene file

Also in: [C++](quadriga_lib_api_cpp.md#mitsuba_xml_file_write) · [Python](quadriga_lib_api_python.md#mitsuba_xml_file_write)

- The XML file can be loaded by NVIDIA Sionna RT
- Creates a subdirectory `<stem>_meshes/` next to the XML file and writes one binary PLY file per object into it;
  both the XML and the mesh folder must be distributable together; the XML refers to each PLY file by its path
  relative to the XML file (`<stem>_meshes/<object>.ply`)
- An object whose faces use more than one material is split into one object per material, named
  `<obj_name>_<mtl_name>`

### Usage:
```matlab
quadriga_lib.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_ind, mtl_ind, obj_names, ...
    mtl_names, bsdf, map_to_itu_materials );
```

### Inputs:
- **`fn`** — Output file path including `.xml` extension
- **`vert_list`** — Vertex coordinates (x, y, z); `[n_vert, 3]`
- **`face_ind`** — Triangle definitions as 1-based vertex indices; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle; every face must reference a material (0 is rejected); `[n_mesh]`
- **`obj_names`** — Object names; cell array of strings; length must equal `max(obj_ind)`
- **`mtl_names`** — Material names; cell array of strings; length must equal `max(mtl_ind)`
- **`bsdf`** — BSDF material parameters per material; ignored by Sionna RT, used only by the Mitsuba renderer;
  see [obj_file_read](#obj_file_read) for the columns; `[numel(mtl_names), 17]`; default: `[]` (none)
- **`map_to_itu_materials`** — If `true`, maps material names to ITU presets recognised by Sionna RT; default: `false`

### See also:
- [obj_file_read](#obj_file_read) (source for mesh data and BSDF field layout)

---
<a id="obj_file_read"></a>
## obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

Also in: [C++](quadriga_lib_api_cpp.md#obj_file_read) · [Python](quadriga_lib_api_python.md#obj_file_read)

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
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](formats.md)

### Usage:
```matlab
[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read( fn, fn_csv, csv_strict );
```

### Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** *(optional)* — Path to an EM/acoustic material CSV; must contain a `name` column;
  unmatched faces map to index 0 (no material) unless `csv_strict` is set; empty uses the built-in default table
- **`csv_strict`** — If true, throw when a `usemtl` material is absent from the table;
  otherwise map to index 0 (no material); default: false

### Outputs:
- **`mesh`** — Triangle vertex coordinates `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}` per row; `[n_mesh, 9]`
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 1-based vertex indices into `vert_list` per triangle; uint64; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; uint64; `[n_mesh, 1]`
- **`obj_names`** — Object names; cell array of strings; length `max(obj_ind)`
- **`mtl_ind`** — 1-based visual-material index per triangle (0 = no material); uint64; `[n_mesh, 1]`
- **`mtl_names`** — Visual material names (raw `usemtl`); cell array of strings; length `n_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `[n_mtl, 17]`
- **`csv_ind`** — 1-based EM/acoustic-material index per triangle (0 = no material); uint64; `[n_mesh, 1]`
- **`csv_names`** — Material names from the table; cell array of strings; length `n_csv`
- **`csv_prop`** — Material properties as a struct; each field is one CSV column (excluding `name`)
  holding a column vector of length `n_csv`

### See also:
- [obj_file_write](#obj_file_write) (for writing OBJ files)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (used to calculate indexed mesh for faster processing)
- [ray_mesh_interact](#ray_mesh_interact) (calculating interactions between rays and the triangular mesh)

---
<a id="obj_file_write"></a>
## obj_file_write
Write a Wavefront .obj file

Also in: [C++](quadriga_lib_api_cpp.md#obj_file_write) · [Python](quadriga_lib_api_python.md#obj_file_write)

- Supply geometry as either `mesh`, or as `vert_list` and `face_ind`; giving both, or neither, is an error
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
```matlab
[ vert_list_out, face_ind_out ] = quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, ...
    obj_names, mtl_names, vert_list, face_ind, bsdf, threshold, csv_ind, csv_names, csv_prop, ...
    csv_write_defaults, split_loose_parts );
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
- **`csv_write_defaults`** — If true, also write canonical columns absent from `csv_prop` using their defaults; default: false
- **`split_loose_parts`** — If true, split each object into connected components; default: false

### Outputs:
- **`vert_list_out`** — Vertices derived from `mesh`, or a copy of `vert_list`; `[n_vert, 3]`
- **`face_ind_out`** — 1-based face indices derived from `mesh`, or a copy of `face_ind`; `[n_mesh, 3]`

### See also:
- [obj_file_read](#obj_file_read) (for reading OBJ files and the BSDF column layout)

---
<a id="plane"></a>
## plane
Construct a triangulated plane mesh

Also in: [C++](quadriga_lib_api_cpp.md#plane) · [Python](quadriga_lib_api_python.md#plane)

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
```matlab
mesh = quadriga_lib.plane( scale, rotation, location, n_div );
```

### Inputs:
- **`scale`** — Length 1 scales x and y uniformly; length 2 scales {x,y}
  independently; length 3 is accepted for compatibility with `cube`, but the third element is
  ignored; default: `[]` (1, no scaling)
- **`rotation`** — Euler angles about {x,y,z}, applied as R = Rz·Ry·Rx (Blender
  XYZ); length 3; default: `[]` (no rotation)
- **`location`** — Translation {x,y,z}; length 3; default: `[]` (origin)
- **`n_div`** — Number of subdivisions per edge; yields 2 · n_div^2 triangles;
  default: 1

### Output:
- **`mesh`** — Triangle mesh; each row holds {x1,y1,z1,x2,y2,z2,x3,y3,z3}; `[2 · n_div^2, 9]`

### See also:
- [cube](#cube)
- [icosphere](#icosphere)
- [obj_file_write](#obj_file_write)

---
<a id="point_cloud_aabb"></a>
## point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

Also in: [C++](quadriga_lib_api_cpp.md#point_cloud_aabb) · [Python](quadriga_lib_api_python.md#point_cloud_aabb)

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_ind` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

### Usage:
```matlab
aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind, vec_size );
```

### Inputs:
- **`points`** — 3D point coordinates; `[n_points, 3]`
- **`sub_cloud_ind`** — 1-based row indices marking the start of each sub-cloud; use [point_cloud_segmentation](#point_cloud_segmentation)
  to generate; `[n_sub]`; default: `[]` (not using sub-clouds)
- **`vec_size`** — SIMD alignment padding factor (e.g. 4, 8, 16); default: 1

### Output:
- **`aabb`** — Bounding box matrix; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

### See also:
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud indices)
- [ray_point_intersect](#ray_point_intersect) (use AABBs for intersection)

---
<a id="point_cloud_segmentation"></a>
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

Also in: [C++](quadriga_lib_api_cpp.md#point_cloud_segmentation) · [Python](quadriga_lib_api_python.md#point_cloud_segmentation)

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes
  at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are
  placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

### Usage:
```matlab
[ points_out, sub_cloud_ind, forward_ind, reverse_ind ] = ...
    quadriga_lib.point_cloud_segmentation( points, target_size, vec_size );
```

### Inputs:
- **`points`** — Original 3D point cloud; `[n_points, 3]`
- **`target_size`** — Maximum points per sub-cloud before padding; default: 1024
- **`vec_size`** — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`;
  [ray_point_intersect](#ray_point_intersect) accepts any value, padding (e.g. 8 for AVX2) only improves its speed; default: 1

### Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `[n_points_out, 3]`
- **`sub_cloud_ind`** — 1-based starting index of each sub-cloud within `points_out`; uint32; `[n_sub, 1]`
- **`forward_ind`** — 1-based row in `points` of each row of `points_out`; padding entries are `0`; uint32; `[n_points_out, 1]`
- **`reverse_ind`** — 1-based row in `points_out` of each row of `points`; uint32; `[n_points, 1]`

---
<a id="point_inside_mesh"></a>
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

Also in: [C++](quadriga_lib_api_cpp.md#point_inside_mesh) · [Python](quadriga_lib_api_python.md#point_inside_mesh)

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided, returns the 1-based enclosing object index instead of binary 0/1

### Usage:
```matlab
result = quadriga_lib.point_inside_mesh( points, mesh, obj_ind, distance );
```

### Inputs:
- **`points`** — 3D coordinates of test points; `[n_points, 3]`
- **`mesh`** — Triangle faces in row-major vertex format `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** — Object index per mesh element; 1-based; enables per-object output; `[n_mesh]`
- **`distance`** — Surface proximity threshold; points within this distance of the mesh surface are
  classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1); range: 0–20 m; default: 0

### Output:
- **`result`** — Indicator: `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object
  index (with `obj_ind`); uint64; `[n_points, 1]`

---
<a id="ray_mesh_interact"></a>
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

Also in: [C++](quadriga_lib_api_cpp.md#ray_mesh_interact) · Not in: Python

- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`)
- Face side determined by vertex order; CCW winding = front, CW = back (right-hand rule);
  front-side hit with FBS≠SBS → air-to-media; back-side hit with FBS≠SBS → media-to-air;
  FBS=SBS with opposing normals → media-to-media
- With `compact = true`, rays with `fbs_ind = 0` (no interaction) are dropped, so `n_rayN ≤ n_ray`; with
  `compact = false` (default) all rays are kept and no-hit rays pass through unchanged
- Output direction encoding (spherical/Cartesian) matches input `tridir` format
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves)
- Types 3–5 (scalar) use a single TE-only coefficient, suitable for acoustic simulation with
  impedance-mapped material parameters (ε derived from Z); total internal reflection is handled as
  in the EM path
- The material model and its columns: [The quadriga-lib Material Model and Ray-Surface Physics](http://quadriga-lib.org/quadriga_lib_material_model.md); the CSV format
  of the material table: [Data Formats](formats.md)

### Usage:
```matlab
[ origN, destN, fbsN, sbsN, gainN, polmatN, trivecN, tridirN, fbs_angleN, thicknessN, ...
    edge_lengthN, normal_vecN, out_typeN, path_dirN, ray_indN ] = ...
    quadriga_lib.ray_mesh_interact( interaction_type, center_freq, orig, dest, mesh, ...
    mtl_ind, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, compact );
```

### Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction,
  3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_freq`** — Center frequency in Hz; for types 3–5 the acoustic frequency
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [obj_file_read](#obj_file_read); `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `csv_ind` output of
  [obj_file_read](#obj_file_read)); `[n_mesh]`
- **`mtl_prop`** — Material properties as a struct; each field is one column (the `csv_prop` output
  of [obj_file_read](#obj_file_read)); each field holds a vector of length `n_mtl`
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); uint32; `[n_ray]`
- **`trivec`** — Beam wavefront triangle vertices relative to origin; order
  `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 9]`; default: `[]`
- **`tridir`** — Vertex-ray directions; `[n_ray, 6]` for spherical
  `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian; default: `[]`
- **`compact`** — If true, rays with no interaction (`fbs_ind = 0`) are dropped so
  `n_rayN <= n_ray`; if false, all rays are kept (`n_rayN = n_ray`) and no-hit rays are returned as a
  transparent pass-through; logical; default: false

### Outputs:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale); averaged over TE/TM polarizations for types 0–2,
  single TE coefficient for types 3–5; `[n_rayN, 1]`
- **`polmatN`** — For types 0–2: polarization transfer matrix, interleaved complex, column-major
  `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, labels `<out><in>` (= `<rx><tx>`); includes interaction gain, TE/TM coefficients,
  incidence plane orientation; excludes in-medium attenuation and FSPL; `[8, n_rayN]`. For types
  3–5 (scalar): `[Re Im]` where Re+jIm is the scalar pressure coefficient; `[2, n_rayN]`
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if
  `trivec`/`tridir` not provided
- **`fbs_angleN`** — Incidence angle at FBS; `[n_rayN, 1]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance); `[n_rayN, 1]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (Inf if partial hit);
  requires the `trivec` and `tridir` inputs, returns all zeros without them; `[n_rayN, 1]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded; uint32; `[n_rayN, 1]`
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
  order-preserving); equals `1:n_ray` when `compact = false`; uint32; `[n_rayN, 1]`

### See also:
- [obj_file_read](#obj_file_read) (for loading `mesh` and `mtl_prop` from OBJ file)
- [icosphere](#icosphere) (for generating beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for computing FBS and SBS positions)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)

---
<a id="ray_point_intersect"></a>
## ray_point_intersect
Calculate intersections of ray beams with points in 3D space

Also in: [C++](quadriga_lib_api_cpp.md#ray_point_intersect) · [Python](quadriga_lib_api_python.md#ray_point_intersect)

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the
  origin, enabling energy spread simulation
- Returns, for each point, the list of ray indices whose beam intersects that point
- All internal computations use single precision

### Usage:
```matlab
[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( points, orig, trivec, tridir, ...
    sub_cloud_ind, use_kernel, gpu_id );
```

### Inputs:
- **`points`** — 3D point cloud coordinates; `[n_points, 3]`
- **`orig`** — Ray origin positions in global Cartesian coordinates; `[n_ray, 3]`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `[n_ray, 9]`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `[n_ray, 9]`
- **`sub_cloud_ind`** — Segment boundary indices for the point cloud, 1-based; use [point_cloud_segmentation](#point_cloud_segmentation) to generate;
  sub-clouds need not be aligned to the SIMD vector size (padding with `vec_size` is optional and only improves speed);
  `[n_sub]`; default: `[]` (not using sub-clouds)
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable
  and for other values; auto mode selects CUDA when `n_points >= 500` and CUDA is available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

### Outputs:
- **`hit_count`** — Number of beams intersecting each point; uint32; `[n_points, 1]`
- **`ray_ind`** — Per-point list of 1-based ray indices that intersected that point; zero-padded to
  a regular 2D array (zero entries indicate unused slots); uint32; `[max_hits, n_points]`

### See also:
- [icosphere](#icosphere) (generate ray beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (generate sub-cloud index)
- [ray_triangle_intersect](#ray_triangle_intersect) (ray–triangle intersection)
- [ray_mesh_interact](#ray_mesh_interact) (beam–mesh interaction)

---
<a id="ray_triangle_intersect"></a>
## ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

Also in: [C++](quadriga_lib_api_cpp.md#ray_triangle_intersect) · [Python](quadriga_lib_api_python.md#ray_triangle_intersect)

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

### Usage:
```matlab
[ fbs, sbs, n_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( ...
    orig, dest, mesh, sub_mesh_ind, aabb, use_kernel, gpu_id );
```

### Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `[n_mesh, 9]`
- **`sub_mesh_ind`** — 1-based start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal;
  use [triangle_mesh_segmentation](#triangle_mesh_segmentation) to generate; `[n_sub]`; default: `[]` (not using sub-meshes)
- **`aabb`** — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; 
  if empty or omitted, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable 
  and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

### Outputs:
- **`fbs`** — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`n_interact`** — Total number of intersections per ray between `orig` and `dest`; uint32; `[n_ray, 1]`
- **`fbs_ind`** — 1-based index of first intersected mesh element; 0 = none; uint32; `[n_ray, 1]`
- **`sbs_ind`** — 1-based index of second intersected mesh element; 0 = none; uint32; `[n_ray, 1]`

### See also:
- [obj_file_read](#obj_file_read) (load mesh from OBJ file)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (compute sub-mesh indices)
- [triangle_mesh_aabb](#triangle_mesh_aabb) (compute AABBs)
- [ray_point_intersect](#ray_point_intersect) (beam interactions with sampling points)
- [icosphere](#icosphere) (generate ray beams)

---
<a id="subdivide_triangles"></a>
## subdivide_triangles
Subdivide triangles into smaller triangles

Also in: [C++](quadriga_lib_api_cpp.md#subdivide_triangles) · [Python](quadriga_lib_api_python.md#subdivide_triangles)

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in · n_div · n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles

### Usage:
```matlab
[ triangles_out, mtl_ind_out ] = quadriga_lib.subdivide_triangles( n_div, triangles_in, mtl_ind );
```

### Inputs:
- **`n_div`** — Number of subdivisions per edge
- **`triangles_in`** — Mesh vertices as `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; `[n_triangles_in, 9]`
- **`mtl_ind`** — 1-based material index per triangle (the `csv_ind` output of [obj_file_read](#obj_file_read));
  0 = no material; `[n_triangles_in]`; passed through unchanged; default: `[]`

### Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_ind_out`** — Material indices for the subdivided triangles, 1-based; uint64; `[n_triangles_out, 1]`; only
  populated if `mtl_ind` is given

---
<a id="triangle_mesh_aabb"></a>
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

Also in: [C++](quadriga_lib_api_cpp.md#triangle_mesh_aabb) · [Python](quadriga_lib_api_python.md#triangle_mesh_aabb)

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding
  non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

### Usage:
```matlab
aabb = quadriga_lib.triangle_mesh_aabb( mesh, sub_mesh_ind, vec_size );
```

### Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `[n_triangles, 9]`
- **`sub_mesh_ind`** — 1-based start indices of sub-meshes in `mesh`; use [triangle_mesh_segmentation](#triangle_mesh_segmentation) to generate;
  `[n_sub]` or empty; default: `[]` (returns AABB of the entire mesh)
- **`vec_size`** — Alignment size for SIMD/CUDA padding (e.g., 8 for AVX2); default: 1

### Output:
- **`aabb`** — Axis-aligned bounding boxes, one row per sub-mesh; `[n_sub_aligned, 6]`

### See also:
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (for calculating sub-meshes)

---
<a id="triangle_mesh_segmentation"></a>
## triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

Also in: [C++](quadriga_lib_api_cpp.md#triangle_mesh_segmentation) · [Python](quadriga_lib_api_python.md#triangle_mesh_segmentation)

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_ind` uses 0 to mark
  padding entries
- If `mtl_ind` is provided, material indices are reordered and padded in the same way

### Usage:
```matlab
[ mesh_out, sub_mesh_ind, mesh_ind, mtl_ind_out ] = ...
    quadriga_lib.triangle_mesh_segmentation( mesh, target_size, vec_size, mtl_ind );
```

### Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`target_size`** — Target triangle count per sub-mesh; for best performance set near sqrt(n_mesh); default: 1024
- **`vec_size`** — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count is rounded
  up to a multiple of this value; default: 1
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [obj_file_read](#obj_file_read));
  0 = no material; `[n_mesh]` or empty; passed through unchanged; default: `[]`

### Outputs:
- **`mesh_out`** — Reordered and padded triangle vertices; `[n_mesh_out, 9]`
- **`sub_mesh_ind`** — 1-based start indices of sub-meshes in `mesh_out`; uint32; `[n_sub, 1]`
- **`mesh_ind`** — 1-based row in `mesh` of each row of `mesh_out` (0 = padding); uint32; `[n_mesh_out, 1]`
- **`mtl_ind_out`** — Reordered and padded material indices, 1-based; padding entries get 0 (no material); uint64;
  `[n_mesh_out, 1]`; only populated if `mtl_ind` is given

---

<a id="sec-sum-of-sinusoids-random-number-generator"></a>
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

**Quick start**\
[sos_randn](#sos_randn) sets up the generators, draws the phases and returns the values in one call. [sos_rand](#sos_rand)
and [sos_randi](#sos_randi) do the same for Uniform and integer values.

```matlab
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
independent.\
**Keeping a generator**\
A generator is two matrices: the frequencies from [sos_preset](#sos_preset) and the phases from [sos_init_phases](#sos_init_phases).
[sos_val](#sos_val) evaluates them. Each column of the phase matrix is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```matlab
% Frequencies of a predefined generator for a decorrelation distance of 10 m
[spatial_freq, acf, dist] = quadriga_lib.sos_preset('Comb256', 10);

% Phases of 20 generators; each column is one independent random field
phase = quadriga_lib.sos_init_phases(size(spatial_freq, 1), 20, 42);

% Values of all 20 generators at the positions, size [1000, 20]
val_all = quadriga_lib.sos_val(spatial_freq, phase, pos);

% The same fields at other positions, at any later time
pos_new = pos + 2.5;
val_new = quadriga_lib.sos_val(spatial_freq, phase, pos_new);

% Uniform distributed output
val_uni = quadriga_lib.sos_val(spatial_freq, phase, pos, [], [], [], [], [], true);

% Map on a regular grid, size [n_y, n_x, 1, 20]
x = 0:0.5:250;  y = 0:0.5:100;
grid = quadriga_lib.sos_map(spatial_freq, phase, x, y);
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

```matlab
% Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
pos_tx = pos;
pos_rx = rand(1000, 3) * 100;
val_link = quadriga_lib.sos_val(spatial_freq, phase, pos_tx, pos_rx, [], acf, dist);
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
  except [sos_generate](#sos_generate), which runs on one thread. No result depends on the number of threads,
  except the last digits of the errors of [sos_calc_mse](#sos_calc_mse).
  [sos_val](#sos_val) computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**\
[sos_generate](#sos_generate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos_calc_mse](#sos_calc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```matlab
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

**Reference**\
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
[IEEE Xplore](https://ieeexplore.ieee.org/document/8644265) (Original version),
[PDF version](http://quadriga-lib.org/publications/2018_sum_of_sinusoids.pdf),
[Markdown version](http://quadriga-lib.org/publications/2018_sum_of_sinusoids.md)

---
<a id="sos_acf_2d"></a>
## sos_acf_2d
Target ACF on a 2-D grid

Also in: [C++](quadriga_lib_api_cpp.md#sos_acf_2d) · [Python](quadriga_lib_api_python.md#sos.acf_2d)

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 1`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos_acf_approx](#sos_acf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on
  the regular axis

### Usage:
```matlab
[ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`

### Outputs:
- **`acf_2d`** — ACF on the grid; `[n_grid, n_grid]`
- **`dist_2d`** — Grid axis; `[n_grid, 1]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
<a id="sos_acf_approx"></a>
## sos_acf_approx
ACF produced by a set of sinusoids

Also in: [C++](quadriga_lib_api_cpp.md#sos_acf_approx) · [Python](quadriga_lib_api_python.md#sos.acf_approx)

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
```matlab
[ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( spatial_freq, dist );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

### Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `[n_grid, 1]`

### See also:
- [sos_acf_2d](#sos_acf_2d) (target ACF on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
<a id="sos_acf_estimate"></a>
## sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

Also in: [C++](quadriga_lib_api_cpp.md#sos_acf_estimate) · [Python](quadriga_lib_api_python.md#sos.acf_estimate)

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
```matlab
[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( spatial_freq, phase, bins, n_pos, ...
    acf, dist, seed );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, 3]`
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
- **`acf_est`** — Estimated ACF; `[n_bin, 1]`
- **`dist_est`** — Mean distance of the pairs in each bin; `[n_bin, 1]`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; `[n_bin, 1]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF of the sinusoids in closed form)
- [sos_val](#sos_val) (evaluation of the generator)

---
<a id="sos_acfi"></a>
## sos_acfi
Target ACF at given distances

Also in: [C++](quadriga_lib_api_cpp.md#sos_acfi) · [Python](quadriga_lib_api_python.md#sos.acfi)

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [sos_val](#sos_val) and [sos_remove_bias](#sos_remove_bias) for two devices on the same map

### Usage:
```matlab
val = quadriga_lib.sos_acfi( acf, dist, d );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances; must be >= 0; `[n_d]`

### Output:
- **`val`** — ACF at the distances `d`; `[n_d, 1]`

### See also:
- [sos_preset](#sos_preset) (ACF tables of the predefined models)

---
<a id="sos_calc_mse"></a>
## sos_calc_mse
Approximation error of the ACF of a set of sinusoids

Also in: [C++](quadriga_lib_api_cpp.md#sos_calc_mse) · [Python](quadriga_lib_api_python.md#sos.calc_mse)

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse_dB`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core_dB`: error on the grid of [sos_acf_approx](#sos_acf_approx) against [sos_acf_2d](#sos_acf_2d), for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all_dB`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core_dB` is evaluated on the sample points `dist`, `mse_all_dB` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse_dB` equals `mse_core_dB`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core_dB` and `mse_all_dB` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as `Inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)

### Usage:
```matlab
[ mse_dB, mse_core_dB, mse_all_dB, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( spatial_freq, acf, dist, ...
    n_dir );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10

### Outputs:
- **`mse_dB`** — Error along the test directions in [dB]; scalar
- **`mse_core_dB`** — Error within the core radius in [dB]; scalar
- **`mse_all_dB`** — Error on the complete grid in [dB]; scalar
- **`acf_tail`** — Approximated ACF beyond the fitted range; `[4, 1]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_generate](#sos_generate) (finds the frequencies; minimises the error returned here)
- [sos_acf_approx](#sos_acf_approx), [sos_acf_2d](#sos_acf_2d) (the two grids that are compared)

---
<a id="sos_generate"></a>
## sos_generate
Find the spatial frequencies of a sum-of-sinusoids generator for a given ACF

Also in: [C++](quadriga_lib_api_cpp.md#sos_generate) · [Python](quadriga_lib_api_python.md#sos.generate)

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
  (see `acf_tail` of [sos_calc_mse](#sos_calc_mse)). To prevent this, set `spatial_freq_min`
- `spatial_freq_min` and `spatial_freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; by default there is no lower limit, and the upper limit is `15 / max(dist)`
- The search grid has `( spatial_freq_max - spatial_freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are
  stored for all sample points. It must not have more than 100000 points (the default has 151),
  which limits `spatial_freq_max - spatial_freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [sos_preset](#sos_preset) were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

### Usage:
```matlab
[ spatial_freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, n_coeff, n_dim, uniform_smp, ...
    n_dir, random_init, spatial_freq_min, spatial_freq_max, seed );
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
- **`spatial_freq_min`** — Lower limit of the frequency magnitude; must be >= 0; 0 = none;
  default: 0
- **`spatial_freq_max`** — Upper limit of the frequency magnitude; NaN or <= 0: default `15 / max(dist)`;
  must be larger than `spatial_freq_min`; limited by the size of the search grid, see above;
  default: NaN
- **`seed`** — Seed; integer from -1 to 2^63-1; -1 seeds from the system random device; default: -1

### Outputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, n_dim]`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale; `[n_pass, 1]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_calc_mse](#sos_calc_mse) (error of a frequency table)
- [sos_preset](#sos_preset) (precomputed tables)

---
<a id="sos_init_phases"></a>
## sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

Also in: [C++](quadriga_lib_api_cpp.md#sos_init_phases) · [Python](quadriga_lib_api_python.md#sos.init_phases)

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
```matlab
phase = quadriga_lib.sos_init_phases( n_coeff, n_gen, seed );
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0
- **`seed`** — Seed of the random stream; integer from -1 to 2^63-1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

### Output:
- **`phase`** — Phases; `[n_coeff, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluates the generators)
- [sos_preset](#sos_preset) (frequencies that belong to the phases)

---
<a id="sos_map"></a>
## sos_map
Spatially correlated random values on a regular grid

Also in: [C++](quadriga_lib_api_cpp.md#sos_map) · [Python](quadriga_lib_api_python.md#sos.map)

- Evaluates the generators of [sos_val](#sos_val) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz, g)` belongs to the position
  `{x(ix), y(iy), z(iz)}` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `spatial_freq`; an omitted or
  empty `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `spatial_freq` has 3 columns
- The values are those of [sos_val](#sos_val) at the grid points, bit for bit; kernels, accuracy and the
  Uniform mapping are described there

### Usage:
```matlab
map = quadriga_lib.sos_map( spatial_freq, phase, x, y, z, uniform, use_kernel );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
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
<a id="sos_preset"></a>
## sos_preset
Spatial frequencies and ACF of a predefined sum-of-sinusoids generator

Also in: [C++](quadriga_lib_api_cpp.md#sos_preset) · [Python](quadriga_lib_api_python.md#sos.preset)

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
```matlab
[ spatial_freq, acf, dist ] = quadriga_lib.sos_preset( name, dist_decorr );
```

### Inputs:
- **`name`** — Name of the preset, e.g. 'Comb256'; string
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10

### Outputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, 3]`
- **`acf`** — Target ACF; `[200, 1]`
- **`dist`** — Sample points of the ACF; `[200, 1]`

### See also:
- [sos_init_phases](#sos_init_phases) (random phases for the generator)
- [sos_val](#sos_val) (evaluation)
- [sos_generate](#sos_generate) (frequencies for other ACFs)

---
<a id="sos_rand"></a>
## sos_rand
Spatially correlated Uniform distributed random numbers

Also in: [C++](quadriga_lib_api_cpp.md#sos_rand) · [Python](quadriga_lib_api_python.md#sos.rand)

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
```matlab
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
<a id="sos_randi"></a>
## sos_randi
Spatially correlated integer random numbers

Also in: [C++](quadriga_lib_api_cpp.md#sos_randi) · [Python](quadriga_lib_api_python.md#sos.randi)

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
```matlab
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
<a id="sos_randn"></a>
## sos_randn
Spatially correlated Normal distributed random numbers

Also in: [C++](quadriga_lib_api_cpp.md#sos_randn) · [Python](quadriga_lib_api_python.md#sos.randn)

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
```matlab
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
<a id="sos_remove_bias"></a>
## sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

Also in: [C++](quadriga_lib_api_cpp.md#sos_remove_bias) · [Python](quadriga_lib_api_python.md#sos.remove_bias)

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
  `spatial_freq · spatial_freq_scale_b`; two offsets per generator. The mean of the combined value is then zero
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
- The C++ in/out arguments `phase` and `phase_b` are split into `phase_in` / `phase_out` and `phase_b_in` /
  `phase_b_out`, so that the inputs stay unchanged; C++ messages name them `phase` and `phase_b`

### Usage:
```matlab
[ phase_out, phase_b_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( spatial_freq, phase_in, ...
    pos, pos_b, phase_b_in, acf, dist, spatial_freq_scale_b );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase_in`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos_b, 1]`, `[n_pos_b, 2]` or `[n_pos_b, 3]` as for `pos`; `n_pos_b = n_pos` for the same map,
  any number for different maps
- **`phase_b_in`** *(optional)* — Phases of the second device; requires `pos_b`; omitted
  or empty = same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then
  be > -1; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device, i.e. decorrelation distance of the
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
<a id="sos_val"></a>
## sos_val
Spatially correlated random values at given positions

Also in: [C++](quadriga_lib_api_cpp.md#sos_val) · [Python](quadriga_lib_api_python.md#sos.val)

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of
  `spatial_freq`, in cycles per meter) and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function
  (ACF) `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the
  frequencies were fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The
  sines and cosines of the positions are computed once and reused for all generators
- Positions are given as one row per position with the columns `{x, y, z}`; they may have fewer
  columns than `spatial_freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and
  `pos_b` and `A` the target ACF interpolated as in [sos_acfi](#sos_acfi); `acf` and `dist` are required.
  The value does not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `spatial_freq · spatial_freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same
  matrix as `phase` and `phase_b` selects the two-map formula
- `spatial_freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
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
```matlab
val = quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, phase_b, acf, dist, spatial_freq_scale_b, ...
    uniform, use_kernel );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`phase_b`** *(optional)* — Phases of the second device; requires `pos_b`; omitted or
  empty = same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then
  be > -1; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device, i.e. decorrelation distance of the
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
