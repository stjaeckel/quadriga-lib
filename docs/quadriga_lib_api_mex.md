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
- Complex data (coefficients, frequency responses, fields, `gr_eps`) is one complex array, also in structs (`coeff` of a channel); a real input is read as imaginary part 0. Octave returns a complex result whose imaginary parts are all zero as a real array. The arrayant struct keeps its split fields `e_theta_re/im`, `e_phi_re/im` and `port_weights_re/im`.
- Octave 8.4 or later is required; older versions lack a working interleaved complex MEX interface, and the build then leaves the Octave interface out.
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
| [calc_angular_spread](#calc_angular_spread) | Channel data and processing | Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths |
| [calc_delay_spread](#calc_delay_spread) | Channel data and processing | Calculate the RMS delay spread of each CIR from its path delays and powers |
| [calc_rician_k_factor](#calc_rician_k_factor) | Channel data and processing | Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power |
| [calc_xpr](#calc_xpr) | Channel data and processing | Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis |
| [crossfade_segments](#crossfade_segments) | Channel data and processing | Join the channels of consecutive segments of one link by a cross-fade |
| [export_channel_obj](#export_channel_obj) | Channel data and processing | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [freq_response](#freq_response) | Channel data and processing | Frequency response of a MIMO channel at the given subcarriers |
| [freq_response_mf](#freq_response_mf) | Channel data and processing | Frequency response of a MIMO channel from coefficients sampled at several frequencies |
| [hdf5_create_file](#hdf5_create_file) | Channel data and processing | Create a new HDF5 channel file with a given storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Channel data and processing | Read channel objects from one or several slots of an HDF5 channel file |
| [hdf5_read_dset](#hdf5_read_dset) | Channel data and processing | Read an unstructured dataset from a slot of an HDF5 channel file |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Channel data and processing | Read the names of the unstructured datasets of a slot of an HDF5 channel file |
| [hdf5_read_layout](#hdf5_read_layout) | Channel data and processing | Read the storage layout of an HDF5 channel file |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Channel data and processing | Reshape the storage layout of an existing HDF5 channel file |
| [hdf5_write_channel](#hdf5_write_channel) | Channel data and processing | Write channel objects to one or several slots of an HDF5 channel file |
| [hdf5_write_dset](#hdf5_write_dset) | Channel data and processing | Write an unstructured dataset to a slot of an HDF5 channel file |
| [qrt_file_parse](#qrt_file_parse) | Channel data and processing | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Channel data and processing | Read the paths of CIRs from a QRT file |
| [quantize_delays](#quantize_delays) | Channel data and processing | Map the path delays to a tap grid by two-tap interpolation |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Channel generators | Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models |
| [get_channels_irs](#get_channels_irs) | Channel generators | Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface |
| [get_channels_multifreq](#get_channels_multifreq) | Channel generators | Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies |
| [get_channels_planar](#get_channels_planar) | Channel generators | Calculate MIMO channel coefficients and delays for planar wave propagation |
| [get_channels_spherical](#get_channels_spherical) | Channel generators | Calculate MIMO channel coefficients and delays for spherical wave propagation |
| [get_channels_tr38901](#get_channels_tr38901) | Channel generators | Generate the channels of a TR 38.901 scenario in one call |
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
| [acdf](#acdf) | Math functions | Empirical cumulative distribution functions (CDF) of several data sets, their average and quantile statistics |
| [calc_rotation_matrix](#calc_rotation_matrix) | Math functions | Calculate rotation matrices from Euler angles |
| [fast_cart2geo](#fast_cart2geo) | Math functions | Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length |
| [fast_geo2cart](#fast_geo2cart) | Math functions | Convert elementwise azimuth/elevation angles to Cartesian coordinates |
| [fast_sincos](#fast_sincos) | Math functions | Compute elementwise approximate sine and/or cosine of a vector |
| [interp](#interp) | Math functions | Perform linear interpolation (1D or 2D) on single or multiple data sets |
| [version](#version) | Miscellaneous / Tools | Version number of quadriga-lib |
| [write_png](#write_png) | Miscellaneous / Tools | Write a data matrix to a color-coded PNG file |
| [attach_terminals](#attach_terminals) | Network layouts and mobility | Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP |
| [drop_terminals](#drop_terminals) | Network layouts and mobility | Drop UTs into the cells of the Tx nodes of a layout |
| [eval_antennas](#eval_antennas) | Network layouts and mobility | Generate the antenna arrays of a layout from the antenna lines |
| [export_rays_qrt](#export_rays_qrt) | Network layouts and mobility | Write the rays of a layout with their link gain to a QRT file |
| [generate_layout](#generate_layout) | Network layouts and mobility | Network layout from a preset: BS sites and a drop of UTs |
| [generate_rays](#generate_rays) | Network layouts and mobility | Rays of the propagation links of a layout (stage 2 of the channel pipeline) |
| [generate_track](#generate_track) | Network layouts and mobility | Replace the rows of nodes by a generated track or a point drop, or append a node with one |
| [get_channels_from_rays](#get_channels_from_rays) | Network layouts and mobility | Channel coefficients of a layout from given rays (stage 3 of the channel pipeline) |
| [get_channels_layout](#get_channels_layout) | Network layouts and mobility | Channel coefficients of a layout in one call |
| [init_link_states](#init_link_states) | Network layouts and mobility | Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline) |
| [layout_csv_read](#layout_csv_read) | Network layouts and mobility | Read the Tx and Rx CSV files of a layout |
| [layout_csv_write](#layout_csv_write) | Network layouts and mobility | Write a layout to the Tx and Rx CSV files |
| [sample_track](#sample_track) | Network layouts and mobility | Positions and orientations of a node at snapshot times |
| [set_antenna](#set_antenna) | Network layouts and mobility | Set the antenna of layout nodes from the arguments of generate_antenna, a QDANT file, or by index |
| [set_antenna_array](#set_antenna_array) | Network layouts and mobility | Set the antenna of layout nodes from an evaluated array |
| [set_orientation](#set_orientation) | Network layouts and mobility | Set the orientation of nodes from their motion, or to a given value |
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
| [ray_mesh_interact](#ray_mesh_interact) | Site-specific simulation tools | Calculate reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces |
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
- [export_channel_obj](#export_channel_obj) (exports propagation paths to an OBJ file)

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
  product of the driver directivity, the enclosure radiation and the bandpass frequency response; a negative value is a
  180° phase inversion (piston side lobes, dipole rear)
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
  - `dipole`: `R = cos α`, negative in the rear hemisphere
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
[ V, H, dist, azimuth_loc, elevation_loc, gamma ] = quadriga_lib.interpolate_pattern( ...
    arrayant, azimuth, elevation, i_element, orientation, element_pos, [], use_kernel );

% Multi-frequency mode
[ V, H ] = quadriga_lib.interpolate_pattern( arrayant, azimuth, elevation, ...
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
- **`V`** — Complex e-theta (vertical) field; `[n_out, n_ang]`, multi-frequency mode `[n_out, n_ang, n_freq]`
- **`H`** — Complex e-phi (horizontal) field; same size as `V`
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

<a id="sec-channel-data-and-processing"></a>
# Channel data and processing

A **channel** describes the propagation between a transmit and a receive antenna array: for each snapshot, a set of
propagation paths, each with a complex coefficient per antenna pair and a delay. The functions of this section store
channels in files, turn them into frequency responses, prepare them for link-level simulation, compute their statistics
and show the paths in 3D. The channels themselves come from the functions of section "Channel generators" or from
[get_channels_layout](#get_channels_layout).

**What a channel holds** (a struct, fields in [hdf5_read_channel](#hdf5_read_channel)):

- The positions and orientations of the transmitter and the receiver, per snapshot or fixed
- The complex **coefficients** `coeff` and the **delays** `delay`, `[n_rx, n_tx, n_path, n_snap]`
- Optionally, antenna-independent path data: path gain, path length, polarization transfer matrix, angles, scatterer
  positions and the interaction points of ray-traced paths
- Unstructured metadata: named values of any type

**In MATLAB / Octave:**

- Several channels are a 1xN struct array. A field with one entry per snapshot is one array with the snapshot on the
  last dimension, zero-padded where the number of paths differs between snapshots
- [hdf5_read_channel](#hdf5_read_channel) and [hdf5_write_channel](#hdf5_write_channel) take and return the unstructured data as a struct array `par` of
  the size of the channel struct array, one field per dataset

**Typical workflow:**

1. Generate the channels, or read them: [hdf5_read_channel](#hdf5_read_channel) from a channel file, [qrt_file_read](#qrt_file_read) from the output
   of the QRT ray tracer
2. Optionally, map the delays to a tap grid with [quantize_delays](#quantize_delays), or join the segments of a track with
   [crossfade_segments](#crossfade_segments)
3. Compute the frequency response at the subcarriers with [freq_response](#freq_response), or with [freq_response_mf](#freq_response_mf) for
   coefficients sampled at several frequencies
4. Evaluate the channels: [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread), [calc_rician_k_factor](#calc_rician_k_factor), [calc_xpr](#calc_xpr)
5. Store them with [hdf5_write_channel](#hdf5_write_channel), or show the paths with [export_channel_obj](#export_channel_obj)

**Terms** (full list in the [Glossary](quadriga_lib_api_cpp.md#glossary)):

- **Snapshot**: one sample of the channel in time or position; `n_snap` snapshots, each with its own number of paths
- **Path**: one propagation path from the transmitter to the receiver, e.g. the LOS path, a reflection or a ray of a
  cluster
- **Coefficient**: the complex amplitude $a$ of one path for one antenna pair, with the antenna patterns, the
  polarization and the path gain
- **Delay**: the propagation delay $\tau$ of a path in s, per antenna pair `[n_rx, n_tx, n_path]` or shared by all
  pairs `[1, 1, n_path]`
- **CIR**: the channel impulse response of one snapshot, its coefficients and delays
- **Frequency response**: the channel matrix at subcarrier $k$, $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$, with
  $f_k$ the offset of the subcarrier from the carrier (the absolute frequency for `_mf`)
- **Multi-frequency channel** (suffix `_mf`): coefficients sampled at several frequencies, the frequency on the 4th
  dimension, as [get_channels_multifreq](#get_channels_multifreq) produces them
- **Slot**: one storage place of an HDF5 channel file, addressed by four indices `(ix, iy, iz, iw)`, e.g. Tx, Rx and
  frequency; the file holds a fixed grid of slots, its **storage layout**
- **Origin**, **destination**: a QRT file holds the paths from each origin (a Tx position) to a list of CIR positions; a
  destination (a receiver) owns a run of consecutive CIR positions, e.g. a track

**File formats.** [HDF5 channel files](formats/hdf5_channel.md) store channels with antennas, one per slot, in single
precision; [QRT files](formats/qrt.md) store propagation paths from ray tracing or the GBSM before antennas are
applied.

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Store and read channels | [hdf5_create_file](#hdf5_create_file), [hdf5_write_channel](#hdf5_write_channel), [hdf5_read_channel](#hdf5_read_channel), [hdf5_read_layout](#hdf5_read_layout), [hdf5_reshape_layout](#hdf5_reshape_layout) |
| Store and read metadata | [hdf5_write_dset](#hdf5_write_dset), [hdf5_read_dset](#hdf5_read_dset), [hdf5_read_dset_names](#hdf5_read_dset_names) |
| Path files (QRT) | [qrt_file_parse](#qrt_file_parse), [qrt_file_read](#qrt_file_read) |
| Frequency response | [freq_response](#freq_response), [freq_response_mf](#freq_response_mf) |
| Process | [quantize_delays](#quantize_delays), [crossfade_segments](#crossfade_segments) |
| Statistics | [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread), [calc_rician_k_factor](#calc_rician_k_factor), [calc_xpr](#calc_xpr) |
| Show | [export_channel_obj](#export_channel_obj) |

**Conventions.** Coefficients and delays of several snapshots are one array with the snapshot on the last dimension,
as in the channel struct, or cell arrays with one array per snapshot ([freq_response](#freq_response)). Indices are 1-based, also the
slot indices of the HDF5 files and the CIR and origin indices of the QRT files.

---
<a id="calc_angular_spread"></a>
## calc_angular_spread
Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths

Also in: [C++](quadriga_lib_api_cpp.md#calc_angular_spread) · [Python](quadriga_lib_api_python.md#calc_angular_spread)

- For one angle, with the path powers $p_l$ normalized to sum 1 over the paths of a CIR (the second-moment definition
  of 3GPP TR 38.901):

  $$\sigma = \sqrt{\sum_l p_l \, \Delta_l^2}, \quad \Delta_l = \arg e^{j (\phi_l - \bar\phi)}$$

  with the circular mean $\bar\phi = \arg \sum_l p_l \, e^{j \phi_l}$
- `wrapping = false`: the spreads of the given azimuth and elevation angles; near the poles, the azimuth spread
  overstates the angular spread
- `wrapping = true`: the paths are first rotated so that their power-weighted mean direction (the mean of the unit
  vectors) lies at azimuth 0 on the equator; `calc_bank_angle` then also rotates them about that direction so that
  the azimuth spread is largest (principal axis of the power-weighted covariance of the rotated angles); `orientation`
  is the direction of an antenna that sees the paths at the rotated angles `phi` and `theta`
- `quantize_deg` > 0: the deviations $\Delta_l$ are rounded to multiples of `quantize_deg` and the powers of equal
  deviations summed, for azimuth and elevation separately

### Usage:
```matlab
[ azimuth_spread, elevation_spread, orientation, phi, theta ] = quadriga_lib.calc_angular_spread( ...
    az, el, powers, wrapping, calc_bank_angle, quantize_deg );
```

### Inputs:
- **`az`** — Path azimuths in rad, one column per CIR; `[n_path, n_cir]`
- **`el`** — Path elevations in rad; `[n_path, n_cir]`
- **`powers`** — Path powers, linear, with a positive sum per CIR; zero-power entries pad CIRs with fewer paths;
  `[n_path, n_cir]`
- **`wrapping`** — Rotate the paths to the equator before computing the spreads; default: false
- **`calc_bank_angle`** — Rotate them also about the mean direction, for the largest azimuth
  spread; only with `wrapping`; default: false
- **`quantize_deg`** — Quantization step of the angle deviations in deg; 0 = none, else finite and >= 1e-3; default: 0

### Outputs:
- **`azimuth_spread`** — RMS azimuth spread in rad; `[n_cir, 1]`
- **`elevation_spread`** — RMS elevation spread in rad; `[n_cir, 1]`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad of an antenna pointing at the mean
  direction (a positive tilt points down), zero without `wrapping`; `[3, n_cir]`
- **`phi`** — Path azimuths in rad in the frame of that antenna (as [interpolate_pattern](#interpolate_pattern)
  returns them), the input `az` without `wrapping`; `[n_path, n_cir]`
- **`theta`** — Path elevations in rad in that frame, the input `el` without `wrapping`;
  `[n_path, n_cir]`

### See also:
- [calc_delay_spread](#calc_delay_spread) (delay spread of the same paths)
- [interpolate_pattern](#interpolate_pattern) (antenna response in the rotated frame)

---
<a id="calc_delay_spread"></a>
## calc_delay_spread
Calculate the RMS delay spread of each CIR from its path delays and powers

Also in: [C++](quadriga_lib_api_cpp.md#calc_delay_spread) · [Python](quadriga_lib_api_python.md#calc_delay_spread)

- With the path powers $p_l$ normalized to sum 1 over the paths of a CIR, the mean delay and the RMS delay spread are

  $$\bar\tau = \sum_l p_l \, \tau_l, \qquad \sigma_\tau = \sqrt{\sum_l p_l \, (\tau_l - \bar\tau)^2}$$

- Paths more than `threshold_dB` below the strongest path of their CIR are left out
- With `granularity` > 0, the delays are first rounded to the nearest multiple of `granularity`, and the powers of
  the paths with the same rounded delay are summed
- A CIR without power gives a spread and a mean delay of 0

### Usage:
```matlab
[ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold_dB, granularity );
```

### Inputs:
- **`delays`** — Path delays in s, one column per CIR; `[n_path, n_cir]`
- **`powers`** — Path powers, linear; zero-power entries pad CIRs with fewer paths; `[n_path, n_cir]`
- **`threshold_dB`** — Power range in dB below the strongest path that is kept, > 0; default: 100
- **`granularity`** — Spacing of the delay grid in s, finite and >= 0; 0 = no grid; with a grid, all
  delays must be finite and their span must be below $2^{53}$ grid steps; default: 0

### Outputs:
- **`ds`** — RMS delay spread $\sigma_\tau$ in s; `[n_cir, 1]`
- **`mean_delay`** — Mean delay $\bar\tau$ in s; `[n_cir, 1]`

### See also:
- [quantize_delays](#quantize_delays) (delays on a tap grid with interpolation)
- [calc_angular_spread](#calc_angular_spread) (angular spreads)

---
<a id="calc_rician_k_factor"></a>
## calc_rician_k_factor
Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power

Also in: [C++](quadriga_lib_api_cpp.md#calc_rician_k_factor) · [Python](quadriga_lib_api_python.md#calc_rician_k_factor)

- $K = P_\mathrm{LOS} / P_\mathrm{NLOS}$ with the sums of the path powers; LOS paths are those with a length
  `path_length` <= $d_\mathrm{TR}$ + `window_size`, $d_\mathrm{TR}$ the Tx-Rx distance, all others are NLOS
- Without NLOS power, $K$ is infinite, or 0 if the LOS power is zero too (also for a CIR without paths)

### Usage:
```matlab
[ kf, pg ] = quadriga_lib.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size );
```

### Inputs:
- **`powers`** — Path powers, linear, one column per CIR; zero-power entries pad CIRs with fewer paths;
  `[n_path, n_cir]`
- **`path_length`** — Path lengths in m; `[n_path, n_cir]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`window_size`** — LOS window in m, >= 0; default: 0.01

### Outputs:
- **`kf`** — K-factor, linear; `[n_cir, 1]`
- **`pg`** — Sum of all path powers, linear; `[n_cir, 1]`

### See also:
- [calc_xpr](#calc_xpr) (XPR with the same LOS window)
- [calc_delay_spread](#calc_delay_spread) (delay spread of the same paths)

---
<a id="calc_xpr"></a>
## calc_xpr
Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis

Also in: [C++](quadriga_lib_api_cpp.md#calc_xpr) · [Python](quadriga_lib_api_python.md#calc_xpr)

- Total-power ratio: the co- and cross-polar powers of the paths of a CIR, weighted by the path powers $p_l$, are
  summed, and the XPR is their ratio; LOS paths are left out unless `include_los` is set
- Entries of the polarization transfer matrix $M$ of a path, first index receive, second transmit polarization:
  rows 1-2 of `polmat` = $M_{vv}$, 3-4 = $M_{hv}$ (H rx, V tx), 5-6 = $M_{vh}$ (V rx, H tx), 7-8 = $M_{hh}$, each as
  real and imaginary part
- Circular basis: $M_{rt} = \mathbf{r}^T M \, \mathbf{t}$ (plain transpose), as [get_channels_spherical](#get_channels_spherical)
  couples antenna patterns, with the same vectors at both ends, each in the $(e_\theta, e_\phi)$ frame of its antenna:
  $\mathbf{L} = (1, +j) / \sqrt{2}$ (LHCP) and $\mathbf{R} = (1, -j) / \sqrt{2}$ (RHCP) in (V, H), which gives

  $$M_{LL}, M_{RR} = \frac{1}{2} \left( M_{vv} - M_{hh} \pm j (M_{vh} + M_{hv}) \right)$$

  $$M_{LR}, M_{RL} = \frac{1}{2} \left( M_{vv} + M_{hh} \pm j (M_{hv} - M_{vh}) \right)$$

  (upper sign for the first entry)

- The names L and R follow the IEEE definition (rotation seen along the propagation direction) for the time
  convention $e^{+j 2\pi f t}$, which the coefficients $e^{-j 2\pi f \tau}$ of the library imply, with
  $e_\theta \times e_\phi$ the outward direction of the pattern
- Co-polar means the same circular vector at both ends: the LOS matrix $\mathrm{diag}(1, -1)$ of
  [get_channels_spherical](#get_channels_spherical) is co-polar (handedness kept), $\mathrm{diag}(1, 1)$ (one mirrored component, as
  by a single reflection) is cross-polar

### Usage:
```matlab
[ xpr, pg ] = quadriga_lib.calc_xpr( powers, polmat, path_length, tx_pos, rx_pos, include_los, ...
    window_size );
```

### Inputs:
- **`powers`** — Path powers $p_l$, linear; `[n_path, n_cir]`
- **`polmat`** — Polarization transfer matrices, magnitudes below about 1e18; `[8, n_path, n_cir]`
- **`path_length`** — Path lengths in m; `[n_path, n_cir]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`include_los`** — Include the LOS paths in the XPR; default: false
- **`window_size`** — LOS window in m, >= 0: paths with `path_length` <= $d_\mathrm{TR}$ +
  `window_size` are LOS, $d_\mathrm{TR}$ the Tx-Rx distance; default: 0.01

### Outputs:
- **`xpr`** — XPR, linear; `[n_cir, 6]`; columns:

  | Column | XPR |
  | ------ | --- |
  | 1 | Linear: $(P_{vv} + P_{hh}) / (P_{vh} + P_{hv})$ with $P_{rt} = \sum_l p_l \lvert M_{rt} \rvert^2$ |
  | 2 | V (receive V): $P_{vv} / P_{vh}$ |
  | 3 | H (receive H): $P_{hh} / P_{hv}$ |
  | 4 | Circular: $(P_{LL} + P_{RR}) / (P_{LR} + P_{RL})$ |
  | 5 | LHCP (receive L): $P_{LL} / P_{LR}$ |
  | 6 | RHCP (receive R): $P_{RR} / P_{RL}$ |

- **`pg`** — Path gain of all paths, LOS included,
  $\frac{1}{2} \sum_l p_l \sum_{r,t \in \{v,h\}} \lvert M_{rt} \rvert^2$; `[n_cir, 1]`

### Notes:
- A ratio with zero cross-polar power is infinite, or 0 if the co-polar power is zero too (also for a CIR without
  paths, which has `pg` = 0)
- A scaling of `polmat` does not change the XPR, but `pg`

### See also:
- [calc_rician_k_factor](#calc_rician_k_factor) (LOS and NLOS power with the same LOS window)
- [get_channels_spherical](#get_channels_spherical) (polarization transfer matrices of the paths)

---
<a id="crossfade_segments"></a>
## crossfade_segments
Join the channels of consecutive segments of one link by a cross-fade

Also in: [C++](quadriga_lib_api_cpp.md#crossfade_segments) · [Python](quadriga_lib_api_python.md#crossfade_segments)

- Joins the "rays" channels of the segments of one link, e.g. of a track whose scenario or indoor state changes, into
  one channel; consecutive segments share the snapshots of a ramp, over which the clusters of the old segment are
  exchanged for those of the new one
- [get_channels_from_rays](#get_channels_from_rays) joins the segments of tracks with this function, the ramp centred on the row of the change

### Usage:
```matlab
channel = quadriga_lib.crossfade_segments( segments, n_overlap, output, par );
```

### Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients and path gains (`'rays'`), the same
  number of paths in every snapshot, equal `n_rx`, `n_tx` and `center_freq` (at most one value); struct array, fields
  in [hdf5_read_channel](#hdf5_read_channel); `[1, n_segment]`
- **`n_overlap`** — Number of snapshots at the start of each segment that are also the last snapshots of the preceding
  segment (computed at the same positions); they form the ramp. Entry 1 is 0, at most the snapshots of the preceding
  segment after its own shared ones; `[]`: the snapshots whose Tx and Rx positions match (within 1 µm), an error where
  the positions repeat (e.g. ends that do not move); `[n_segment]`; default: `[]`
- **`output`** — `'taps'` (the paths summed per tap index) or `'rays'`; default: `'taps'`
- **`par`** — Unstructured data of the segments with the fields `tap`, `path_type` and `cluster` (uint32, one value per
  path, see Notes); struct array of the size of `segments` that has one of these fields (other fields are ignored);
  `[]` = all paths NLOS, each its own cluster; default: `[]`

### Outputs:
- **`channel`** — The joined channel, 1x1 struct, fields in [hdf5_read_channel](#hdf5_read_channel); `par` is not carried

### Notes:
- **Snapshots**: segment $k$ starts with the `n_overlap(k)` snapshots of the ramp, which are also the last ones of
  segment $k - 1$; the output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots. Outside the
  ramps the snapshots are those of one segment
- **Paths**: fields of `par`: `path_type` gives the LOS and GR paths (without it, tap index 0 of `tap` and a tap index 3
  held by one path, without index 4 or 5, as [ssf_rays](#ssf_rays) numbers them), `cluster` the clusters, whose paths fade
  together (without it: the taps, without `tap`: each path)
- **Level**: at ramp snapshot $i$ of $n_r$, with $P_a$, $P_b$ the summed path gains of the old and new segment,
  $$G_i = P_{a,i}^{1 - w_i} P_{b,i}^{w_i}, \quad w_i = \frac{1}{2} \left( 1 + \sin\left( \left( \frac{i}{n_r + 1} - \frac{1}{2} \right) \pi \right) \right)$$
- **Exchange**: both cluster sets are scaled to $G$; the clusters of the old segment fade out and those of the new one
  in, one after another, each over a quarter of the exchange, in an order that keeps the delay and angular spreads close
  to those of the superposition of both sets (a segment with more than 32 clusters keeps its cluster order). The
  exchange takes the part of the ramp in which $G$ is within 10 dB of the weaker segment, so that no cluster exceeds its
  own power by more than 10 dB
- **LOS and GR**: one path each, with the data of the new segment (of the old one where the new one has no power),
  scaled to $G_i \left( (1 - q_i) g_a / P_a + q_i g_b / P_b \right)$, $g$ the path gain of the path and $q_i$ the
  exchanged share
- **A segment without power** at a ramp snapshot: all paths of the transition fade with $1 - w$ (old) and $w$ (new)
  in power
- **Taps**: the weighted paths summed per tap index, one set of tap slots for all snapshots (LOS, GR, NLOS); a new tap
  takes the slot of an old tap that has faded out before it enters. Slots without power hold zero coefficients and the
  smallest delay of the snapshot. The delay of a tap is the mean of its paths' delays weighted by their path gains
- **Other data**: delays as given (one reference for all segments); positions and orientations of the old segment;
  `name` and `center_freq` of segment 1; `'rays'` keep `path_length`, `polmat`, `path_angles`, `fbs_pos` and
  `lbs_pos` where all segments have them; `par`, `n_interact` and `interact_coord` are not carried
- **`par`**: the C++ channels carry these entries in their `par_data`; in MATLAB they are the input `par`, as in
  [hdf5_write_channel](#hdf5_write_channel), with the entries of each segment as [gbsm_single_link_channel](#gbsm_single_link_channel) returns them

### See also:
- [get_channels_from_rays](#get_channels_from_rays) (joins the segments of tracks with this function)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channels of one segment)

---
<a id="export_channel_obj"></a>
## export_channel_obj
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](quadriga_lib_api_cpp.md#export_channel_obj) · [Python](quadriga_lib_api_python.md#export_channel_obj)

- Writes the paths of ray-traced channels as tubes to an OBJ file and their colors to an MTL file of the same name
  (`.mtl` instead of `.obj`), e.g. for Blender; existing files are overwritten
- Path gain in dB: with coefficients the power summed over all antenna pairs, else from `polmat` (sum of the squared
  entries) with the free-space gain of the length of the drawn path
- The gain, clipped to `[gain_min_dB, gain_max_dB]`, sets the color (64 steps of `colormap`) and the radius (linear
  from 0 at `gain_min_dB` to `radius_max` at `gain_max_dB`, at least `radius_min`); a path with a radius up to
  0.1 mm is written as a polyline
- Paths below `gain_min_dB` and paths with non-finite coordinates are left out

### Usage:
```matlab
quadriga_lib.export_channel_obj( fn, chan, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, ...
    radius_max, radius_min, n_edges );
```

### Inputs:
- **`fn`** — Path of the OBJ file, ending in `.obj`
- **`chan`** — Channel; 1x1 struct with the fields of [hdf5_read_channel](#hdf5_read_channel); needs `rx_pos`, `tx_pos`,
  `center_freq`, `n_interact`, `interact_coord` and `coeff` with `delay`, or `polmat`; columns of
  `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Number of the strongest paths per snapshot to export; 0 = all; default: 0
- **`gain_max_dB`** — Gain in dB of the largest radius and the last color, finite; default: -60
- **`gain_min_dB`** — Gain in dB of the first color, finite and below `gain_max_dB`; weaker paths are left out;
  default: -140
- **`colormap`** — Name of the colormap: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer;
  default: `'jet'`
- **`i_snap`** — 1-based indices of the snapshots to export; default: `[]` (all snapshots)
- **`radius_max`** — Tube radius in m at `gain_max_dB`, >= 0; default: 0.05
- **`radius_min`** — Smallest tube radius in m, >= 0; at most `radius_max` (a larger value is reduced to it);
  default: 0.01
- **`n_edges`** — Vertices of the tube cross-section, >= 3; default: 5

---
<a id="freq_response"></a>
## freq_response
Frequency response of a MIMO channel at the given subcarriers

Also in: [C++](quadriga_lib_api_cpp.md#freq_response) · [Python](quadriga_lib_api_python.md#freq_response)

- The frequency response of one snapshot at the subcarriers $k$: $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$ with the
  offset $f_k$ = `bandwidth` · `pilot_grid(k)` from the carrier
- One snapshot is a 3D array; several snapshots are cell arrays with one array per snapshot (`coeff` and `delay`
  of the same length), computed in parallel; the number of paths may differ between them
- A channel struct of [hdf5_read_channel](#hdf5_read_channel) holds the snapshots on the 4th dimension, zero-padded;
  `num2cell(ch.coeff, [1 2 3])`, and the same for `delay`, gives the cell arrays (the zero-padded paths do not change
  the result)
- For a multi-frequency channel and absolute carrier frequencies, use [freq_response_mf](#freq_response_mf)
- Precision: the phase is reduced to one period in double precision; sin / cos, coefficients and sums are single
  precision

### Usage:
```matlab
hmat = quadriga_lib.freq_response( coeff, delay, pilot_grid, bandwidth, i_snap, use_kernel );
```

### Inputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_path]`, or a cell array of length `n_snap` with one such array
  per snapshot
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`, or a cell array as `coeff`
- **`pilot_grid`** — Subcarrier positions relative to the bandwidth, finite: 0 = carrier, 1 = carrier + `bandwidth`;
  `[n_carrier, 1]`
- **`bandwidth`** — Bandwidth in Hz, finite and >= 0
- **`i_snap`** — 1-based indices of the snapshots to compute; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable);
  default: 0

### Outputs:
- **`hmat`** — Complex frequency response; `[n_rx, n_tx, n_carrier, n_out]`; the selected snapshots must have equal
  `n_rx` and `n_tx`

### See also:
- [freq_response_mf](#freq_response_mf) (multi-frequency channel, absolute carrier frequencies)
- [get_channels_spherical](#get_channels_spherical) (coefficients and delays)
- [quantize_delays](#quantize_delays) (delays on a tap grid, for time-domain processing)

---
<a id="freq_response_mf"></a>
## freq_response_mf
Frequency response of a MIMO channel from coefficients sampled at several frequencies

Also in: [C++](quadriga_lib_api_cpp.md#freq_response_mf) · [Python](quadriga_lib_api_python.md#freq_response_mf)

- The frequency response of one snapshot of a multi-frequency channel at the absolute carrier frequencies
  $f_k$ = `freq_new(k)`: the coefficients of each path are interpolated from the input frequencies `freq` to $f_k$
  and the delay phase is applied there, $H_k = \sum_l \tilde a_l(f_k) \, e^{-j 2\pi f_k \tau_l}$
- Interpolation between the two neighbouring input frequencies: amplitude linear, phase along the shorter arc
  (SLERP); constant outside the range of `freq`
- `remove_delay_phase = true` interpolates the envelope $a_l(f_i) \, e^{j 2\pi f_i \tau_l}$ instead of $a_l(f_i)$, so
  that the delay phase the channel generators include is not interpolated
- One snapshot is a 4D array `[n_rx, n_tx, n_path, n_freq]`; several snapshots are cell arrays with one array per
  snapshot (`coeff` and `delay` of the same length), computed in parallel; the number of paths may differ
  between them

### Usage:
```matlab
hmat = quadriga_lib.freq_response_mf( coeff, delay, freq, freq_new, remove_delay_phase, i_snap, use_kernel );
```

### Inputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_path, n_freq]`, or a cell array of length `n_snap` with one
  such array per snapshot
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` (a 4th dimension of length `n_freq` is
  allowed, only the first frequency is used), or a cell array as `coeff`
- **`freq`** — Input frequencies in Hz, finite, strictly ascending; `[n_freq, 1]`
- **`freq_new`** — Output carrier frequencies in Hz, finite; `[n_carrier, 1]`
- **`remove_delay_phase`** — Remove the delay phase from the coefficients before the interpolation: `true` for the
  coefficients of [get_channels_multifreq](#get_channels_multifreq) and [get_channels_spherical](#get_channels_spherical), `false` for coefficients without delay
  phase; default: `true`
- **`i_snap`** — 1-based indices of the snapshots to compute; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`hmat`** — Complex frequency response; `[n_rx, n_tx, n_carrier, n_out]`; the selected snapshots must have equal
  `n_rx` and `n_tx`

### Notes:
- With `remove_delay_phase = true`, generate the channels with `use_absolute_delays = true`: their coefficients carry
  the phase of the absolute path length also when `delay` holds relative delays. With relative delays the phase of
  the Tx-Rx distance stays in the coefficients and is interpolated across `freq`, which gives wrong phases (the
  magnitudes stay right) unless the spacing of `freq` is far below $c$ / (Tx-Rx distance)
- One input frequency (3D arrays): the coefficients apply at all carriers; with `remove_delay_phase = true` the
  result is that of [freq_response](#freq_response) with `bandwidth` · `pilot_grid` = `freq_new - freq`
- Precision: the delay phase is reduced to one period in double precision (and removed in double precision with
  `remove_delay_phase`); sin / cos of the output phase and the interpolation are single
  precision, the sums double precision

### See also:
- [freq_response](#freq_response) (single-frequency coefficients, carriers relative to the bandwidth)
- [get_channels_multifreq](#get_channels_multifreq) (multi-frequency channel)

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a given storage layout

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_create_file) · [Python](quadriga_lib_api_python.md#hdf5_create_file)

- Creates a channel file with `nx · ny · nz · nw` empty slots; the slot count is fixed, [hdf5_reshape_layout](#hdf5_reshape_layout) changes
  only the dimensions
- [hdf5_write_channel](#hdf5_write_channel) and [hdf5_write_dset](#hdf5_write_dset) create a missing file themselves, with the layout that just holds their
  slots; this function sets any other one
- Standard HDF5 readers open the file, see [HDF5 channel file format](formats/hdf5_channel.md); change it only with
  the functions of this section

### Usage:
```matlab
quadriga_lib.hdf5_create_file( fn, nx, ny, nz, nw );
```

### Inputs:
- **`fn`** — File name; the file must not exist
- **`nx`** — Number of slots along x, at least 1; default: 65536
- **`ny`**, **`nz`**, **`nw`** — Number of slots along y, z and w, each at least 1; at most 4294967295 (2^32 - 1)
  slots in total; default: 1

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (writes channels into slots)
- [hdf5_read_layout](#hdf5_read_layout) (reads the layout and the occupied slots)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Read channel objects from one or several slots of an HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_channel) · [Python](quadriga_lib_api_python.md#hdf5_read_channel)

- Returns the channels of the selected slots that hold data as a struct array `chan`, and their unstructured data as
  a struct array `par` of the same size
- `ix`, `iy`, `iz` and `iw` select the slots along each dimension; the selection is every combination, in
  column-major order (`ix` fastest), and empty slots are skipped ([hdf5_read_layout](#hdf5_read_layout) tells which slots hold data)
- `i_snap` reads only the given snapshots of every selected slot (see Notes)
- No data in the selection gives `0x0` structs

### Usage:
```matlab
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, i_snap );
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot indices along x, y, z and w, 1-based, inside the layout; scalar
  or vector; default: `[]` (all slots of the dimension)
- **`i_snap`** — Snapshot indices, 1-based, any order, repeats allowed, at most `n_snap` of every selected slot that
  has snapshots; `[n_out, 1]`; default: `[]` (all snapshots)

### Outputs:
- **`chan`** — Channels, one per selected slot that holds data; struct array `[1, n_chan]` with the fields:
  | Field | Description | Class / size |
  | ----- | ----------- | ------------ |
  | `name` | Channel name | char |
  | `tx_pos`, `rx_pos` | Tx and Rx positions in m | `[3, 1]` (fixed) or `[3, n_snap]` |
  | `tx_orientation`, `rx_orientation` | Tx and Rx orientations (bank, tilt, heading) in rad | `[3, 1]` (fixed) or `[3, n_snap]` |
  | `coeff` | Complex coefficients | `[n_rx, n_tx, n_path, n_snap]` |
  | `delay` | Delays in s | `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]` |
  | `path_gain` | Path gain without antennas, linear | `[n_path, n_snap]` |
  | `path_length` | Path length in m | `[n_path, n_snap]` |
  | `polmat` | Polarization transfer matrix (Glossary) | `[8, n_path, n_snap]` |
  | `path_angles` | AOD, EOD, AOA, EOA in rad | `[n_path, 4, n_snap]` |
  | `fbs_pos`, `lbs_pos` | First- and last-bounce scatterer positions in m | `[3, n_path, n_snap]` |
  | `n_interact` | Number of interaction points per path, 0 = LOS | uint32, `[n_path, n_snap]` |
  | `interact_coord` | Interaction points of all paths, concatenated | `[3, max(sum(n_interact)), n_snap]` |
  | `center_freq` | Center frequency in Hz | scalar or `[n_snap, 1]` |
  | `initial_position` | Reference snapshot, 1-based; a missing field means 1 | int32 scalar |

  `n_path` is the largest number of paths of the snapshots; smaller snapshots are zero-padded. A field exists if a
  channel of the selection has it; the other channels get an empty value
- **`par`** — Unstructured data; struct array `[1, n_chan]` whose fields are the dataset names of all channels
  (without the prefix `par_`); a channel without a dataset gets an empty value; `0x0` struct if no channel has
  unstructured data

### Notes:
- **Snapshot selection**: fields with one entry per snapshot are sliced, also positions, orientations and
  `center_freq` stored per snapshot; values stored once and the unstructured data are kept. A slot without snapshots
  (only unstructured data) ignores `i_snap`
- **Round trip** with [hdf5_write_channel](#hdf5_write_channel): the values come back in single precision; empty orientations come back as
  `[3, 1]` zeros
- **Unstructured data**: the datasets of the slot with the prefix `par_`, in their stored class and shape
  ([hdf5_read_dset](#hdf5_read_dset)); datasets of another prefix are not read
- **Errors**: a missing file, an index outside the layout, an index in `i_snap` beyond the snapshots of a slot, or a
  file that is not a channel file of the [HDF5 channel file format](formats/hdf5_channel.md#hdf5-compatibility)

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (finds the occupied slots)
- [hdf5_write_channel](#hdf5_write_channel) (writes channels)

---
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read an unstructured dataset from a slot of an HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_dset) · [Python](quadriga_lib_api_python.md#hdf5_read_dset)

- Reads the dataset `prefix + name` of slot `(ix, iy, iz, iw)`, e.g. `par_carrier_frequency`
- The class and shape follow the stored dataset: char for a string; single, double, int32, uint32, int64 or uint64
  for a scalar, a column vector `[n, 1]` (a 1D dataset), a matrix or a 3D array; a row vector written by
  [hdf5_write_dset](#hdf5_write_dset) comes back as `[1, n]`

### Usage:
```matlab
dset = quadriga_lib.hdf5_read_dset( fn, name, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 1-based; default: 1
- **`prefix`** — Prefix of the dataset name; `''` reads `name` as given; default: `'par_'`

### Output:
- **`dset`** — The dataset; `[]` if the slot is empty or outside the layout or has no such dataset

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (lists the datasets of a slot)
- [hdf5_write_dset](#hdf5_write_dset) (writes a dataset)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read the names of the unstructured datasets of a slot of an HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_dset_names) · [Python](quadriga_lib_api_python.md#hdf5_read_dset_names)

- Lists the datasets of slot `(ix, iy, iz, iw)` whose names start with `prefix`, without the prefix, in the order
  they were written

### Usage:
```matlab
names = quadriga_lib.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 1-based; default: 1
- **`prefix`** — Prefix of the listed names; `''` lists all datasets with their full names; default: `'par_'`

### Output:
- **`names`** — Dataset names; cell array of strings; `[n_names, 1]`, empty if the slot is empty or outside the
  layout or has none

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (reads a dataset by name)
- [hdf5_write_dset](#hdf5_write_dset) (writes a dataset)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of an HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_read_layout) · [Python](quadriga_lib_api_python.md#hdf5_read_layout)

- Returns the dimensions of the slot grid and, optionally, the slots that hold data
- A missing file gives `[0, 0, 0, 0]`, so the function also tests whether a file exists; a file that is not a channel
  file raises an error

### Usage:
```matlab
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

### Input:
- **`fn`** — File name

### Outputs:
- **`storage_dims`** — Layout `[nx, ny, nz, nw]`; uint32; `[1, 4]`
- **`has_data`** — `true` for the slots that were written, also with an empty channel; logical; `[nx, ny, nz, nw]`, empty for a missing file

### See also:
- [hdf5_create_file](#hdf5_create_file) (creates a file with a given layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (changes the dimensions)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshape the storage layout of an existing HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_reshape_layout) · [Python](quadriga_lib_api_python.md#hdf5_reshape_layout)

- Sets new dimensions `[nx, ny, nz, nw]` with the slot count `nx · ny · nz · nw` of the file
- The data stays in place: in the column-major order of the slots (as in `has_data` of [hdf5_read_layout](#hdf5_read_layout)), slot
  `k` keeps its channel, which gets the slot index of position `k` in the new layout

### Usage:
```matlab
quadriga_lib.hdf5_reshape_layout( fn, nx, ny, nz, nw );
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`nx`** — New number of slots along x
- **`ny`**, **`nz`**, **`nw`** — New number of slots along y, z and w; default: 1

### See also:
- [hdf5_create_file](#hdf5_create_file) (sets the layout of a new file)
- [hdf5_read_layout](#hdf5_read_layout) (reads the layout)

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write channel objects to one or several slots of an HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_write_channel) · [Python](quadriga_lib_api_python.md#hdf5_write_channel)

- Stores `chan(k)` with the unstructured data `par(k)` in slot `(ix(k), iy(k), iz(k), iw(k))`; a scalar index
  applies to all channels
- A missing file is created with the layout that just holds the slots (the largest index per dimension);
  [hdf5_create_file](#hdf5_create_file) sets another layout
- A slot that holds data is replaced completely: its old channel and all its unstructured datasets, also those added
  with [hdf5_write_dset](#hdf5_write_dset), are removed
- All inputs are checked before anything is written, so a failing check leaves the file unchanged

### Usage:
```matlab
code = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

### Inputs:
- **`fn`** — File name
- **`chan`** — Channels to write, valid; struct array, fields in [hdf5_read_channel](#hdf5_read_channel); `n_chan` entries, at least 1
- **`par`** — Unstructured data; struct array of `n_chan` entries: a field name is a dataset name (stored with the
  prefix `par_`), the values follow the type rules of [hdf5_write_dset](#hdf5_write_dset), empty fields are skipped; default: `[]`
  (none)
- **`ix`** — Slot index along x, 1-based; scalar or `n_chan` entries; default: `1:n_chan`
- **`iy`**, **`iz`**, **`iw`** — Slot indices along y, z and w, 1-based; scalar or `n_chan` entries; default: 1

### Output:
- **`code`** — `0` if the slot was empty, `1` if its data was replaced; int32; `[n_chan, 1]`

### Notes:
- A channel without data and without `par` entries replaces the data of an occupied slot (it then reads as an empty
  channel, and [hdf5_read_layout](#hdf5_read_layout) reports it as written); it leaves an empty slot as it is and does not create a
  missing file
- The space of replaced data is not reused, so the file does not shrink; copying the channels into a new file reclaims
  it
- Files are not locked: one writer per file at a time, also for [hdf5_write_dset](#hdf5_write_dset) and [hdf5_reshape_layout](#hdf5_reshape_layout).
  Reading is safe while no one writes

### See also:
- [hdf5_read_channel](#hdf5_read_channel) (reads the channels back)
- [hdf5_create_file](#hdf5_create_file) (creates a file with a given layout)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write an unstructured dataset to a slot of an HDF5 channel file

Also in: [C++](quadriga_lib_api_cpp.md#hdf5_write_dset) · [Python](quadriga_lib_api_python.md#hdf5_write_dset)

- Adds the dataset `prefix + name` to slot `(ix, iy, iz, iw)`; everything else stored in the slot stays. An empty
  slot gets a channel without structured data
- A missing file is created with the layout `[ix, iy, iz, iw]`; [hdf5_create_file](#hdf5_create_file) sets another layout
- A dataset is written once: a dataset of the same name in the slot raises an error. A later [hdf5_write_channel](#hdf5_write_channel) to
  the slot removes it with the rest of the slot
- Invalid names and types raise an error before the file is changed
- [hdf5_read_channel](#hdf5_read_channel) returns the datasets with the prefix `par_` as the unstructured data `par`; [hdf5_read_dset](#hdf5_read_dset)
  reads each one back

### Usage:
```matlab
quadriga_lib.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — File name
- **`name`** — Dataset name without the prefix, not empty; `prefix + name` holds only letters, digits and `_`, at
  most 65000 characters
- **`data`** — Not empty: a char array (string), or a real, non-sparse single, double, int32, uint32, int64 or uint64
  array with at most 3 dimensions. A scalar is stored as a scalar, a column vector as a vector, a row vector as a
  `[1, n]` matrix
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 1-based; inside the layout of an
  existing file; default: 1
- **`prefix`** — Prefix of the dataset name; `''` stores `name` without a prefix; default: `'par_'`

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (reads a dataset by name)
- [hdf5_read_dset_names](#hdf5_read_dset_names) (lists the datasets of a slot)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [C++](quadriga_lib_api_cpp.md#qrt_file_parse) · [Python](quadriga_lib_api_python.md#qrt_file_parse)

- Reads the header of a QRT file: counts, names, frequencies, the positions and orientations of the CIRs and the
  origins, and the file version; no path data
- A file without destinations reports one destination `RX` that owns all CIRs
- Raises an error for a file that cannot be opened, is not a [QRT file](formats/qrt.md) of version 5 or 6, or is
  truncated or corrupt

### Usage:
```matlab
[ n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, ...
    cir_pos, cir_orientation, orig_pos, orig_orientation ] = quadriga_lib.qrt_file_parse( fn );
```

### Input:
- **`fn`** — File name

### Outputs:
- **`n_cir`** — Number of CIRs of each origin; uint64
- **`n_orig`** — Number of origin slots; uint64
- **`n_dest`** — Number of destinations, at least 1; uint64
- **`n_freq`** — Number of frequencies; uint64
- **`cir_offset`** — Number of CIRs before each destination (an offset, the first entry 0): the CIRs of destination
  `d` start at `i_cir = cir_offset(d) + 1`; uint64; `[n_dest, 1]`
- **`orig_names`** — Origin names, empty for a slot not written yet; cell array of strings; `[n_orig, 1]`
- **`dest_names`** — Destination names; cell array of strings; `[n_dest, 1]`
- **`version`** — File version: 5 (EM) or 6 (scalar); int32
- **`center_freq`** — Frequencies in Hz; single; `[n_freq, 1]`
- **`cir_pos`** — CIR positions in m; single; `[n_cir, 3]`
- **`cir_orientation`** — CIR orientations (bank, tilt, heading) in rad; single; `[n_cir, 3]`
- **`orig_pos`** — Origin positions in m, zeros for a slot not written yet; single; `[n_orig, 3]`
- **`orig_orientation`** — Origin orientations (bank, tilt, heading) in rad; single; `[n_orig, 3]`

### See also:
- [qrt_file_read](#qrt_file_read) (reads CIRs)

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read the paths of CIRs from a QRT file

Also in: [C++](quadriga_lib_api_cpp.md#qrt_file_read) · [Python](quadriga_lib_api_python.md#qrt_file_read)

- Reads the paths of a list of CIRs of one origin, with the positions and orientations of both ends: the inputs of
  [get_channels_spherical](#get_channels_spherical) (`fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `polmat`) or
  [get_channels_planar](#get_channels_planar) (the angles)
- Only the requested outputs are computed; with at most 5 outputs, no path data is read
- Reads in chunks of 16 CIRs: besides the outputs, at most two chunks are held in memory
- `uplink` swaps the ends: the origin becomes the receiver (see Notes)

### Usage:
```matlab
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, uplink, normalize_polmat );
```

### Inputs:
- **`fn`** — File name
- **`i_cir`** — CIR indices, 1-based, at most `n_cir`; `[n_out, 1]`; default: `[]` (all CIRs of the file)
- **`i_orig`** — Origin index, 1-based, at most `n_orig`; default: 1
- **`uplink`** — `false`: origin = Tx, destination = Rx; `true`: origin = Rx, destination = Tx; default: `false`
- **`normalize_polmat`** — Scaling of `polmat` and `path_gain`, logical (see Notes); default: `true`

### Outputs:
- **`center_freq`** — Frequencies in Hz; `[n_freq, 1]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, n_out]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientations (bank, tilt, heading) in rad; `[3, n_out]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m: the first and the last interaction
  point, the midpoint of Tx and Rx for a LOS path; cell array `[n_out, 1]` of `[3, n_path]`
- **`path_gain`** — Path gain, linear (see Notes); cell array `[n_out, 1]` of `[n_path, n_freq]`
- **`path_length`** — Length of the polyline from the Tx through the interaction points to the Rx in m; cell array
  `[n_out, 1]` of `[n_path, 1]`
- **`polmat`** — Polarization transfer matrix; cell array `[n_out, 1]` of `[8, n_path, n_freq]`; version 6: the
  complex scalar coefficient, `[2, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure (at the Tx) and arrival (at the
  Rx) in rad; cell array `[n_out, 1]` of `[n_path, 1]`
- **`path_coord`** — Points of each path from the Tx through the interaction points to the Rx; cell array
  `[n_out, 1]` of cell arrays `[n_path, 1]` of `[3, n_interact + 2]`
- **`n_interact`** — Number of interaction points of each path, 0 = LOS; cell array `[n_out, 1]` of uint32
  `[n_path, 1]`
- **`coord`** — Interaction points of all paths, concatenated in path order; cell array `[n_out, 1]` of single
  `[3, sum(n_interact)]`
- **`interact_type`** — Interaction type codes (see [QRT file format: stored values](formats/qrt.md#qrt-stored-values)),
  concatenated like `coord`; cell array `[n_out, 1]` of uint8 `[sum(n_interact), 1]`

### Notes:
- **Path gain**: the file stores per path the coefficients with the interaction (material) losses; the reader adds the
  propagation loss of the path length $d$, for EM files (version 5) the free-space loss
  $\mathrm{PL} = 32.45 + 20 \log_{10}(f / \mathrm{GHz}) + 20 \log_{10}(d / \mathrm{m})$ dB, for scalar files (version 6)
  the spherical spreading and the air absorption $\alpha(f)$ of ISO 9613-1 at 20 °C, 50 % humidity and 1 atm,
  $\mathrm{PL} = 20 \log_{10}(d / \mathrm{m}) + \alpha(f) \, d$ dB. With $p$ the larger column power of the stored
  matrix (version 6: the power of the coefficient):
  | `normalize_polmat` | `polmat` | `path_gain` |
  | ------------------ | -------- | ----------- |
  | `false` | As stored | $10^{-\mathrm{PL}/10}$ |
  | `true` | Stored matrix divided by $\sqrt{p}$: largest column power 1 (uplink: row power) | $p \cdot 10^{-\mathrm{PL}/10}$ |

- **Uplink**: `polmat` is the transpose of the downlink matrix (reciprocal channel, no conjugation) with the same
  `path_gain`; positions, orientations, angles, `fbs_pos`, `lbs_pos` and `path_coord` swap their ends.
  `n_interact`, `coord` and `interact_type` keep the stored order from the origin to the destination, and the type
  codes describe that traversal
- An origin slot that is not written yet gives zero positions and orientations and no paths
- **Errors**: an index out of range, or a file that cannot be opened, is not a [QRT file](formats/qrt.md) of version 5
  or 6, or is truncated or corrupt (counts or offsets beyond the file, path blocks of different CIRs that overlap). A
  message about one requested CIR starts with `Snapshot <k>: `, with `k` its position in `i_cir`

### See also:
- [qrt_file_parse](#qrt_file_parse) (metadata without path data)
- [get_channels_spherical](#get_channels_spherical) (coefficients from the paths)
- [get_channels_planar](#get_channels_planar) (coefficients from the angles)

---
<a id="quantize_delays"></a>
## quantize_delays
Map the path delays to a tap grid by two-tap interpolation

Also in: [C++](quadriga_lib_api_cpp.md#quantize_delays) · [Python](quadriga_lib_api_python.md#quantize_delays)

- For time-domain processing with a tapped delay line: a path with delay $\tau$ and coefficient $a$ goes to the two
  neighbouring taps $\lfloor \tau / \Delta \rfloor$ and $\lfloor \tau / \Delta \rfloor + 1$ of the grid with the
  spacing $\Delta$ = `tap_spacing`, with the coefficients $a \, (1 - \delta)^\alpha$ and $a \, \delta^\alpha$, where
  $\delta$ is the fractional offset of $\tau / \Delta$ and $\alpha$ = `power_exponent`; the coefficients change smoothly
  when a delay crosses a tap
- $\alpha = 1$ interpolates linearly (narrowband), $\alpha = 0.5$ keeps the power of each path (wideband)
- A grid holds the taps with power, at most `max_taps`: all taps of the interpolation if they fit, else the nearest
  taps of the paths, strongest first, and the remaining places for the strongest other taps. A path whose two taps
  are not both in the grid goes to its nearest tap with its full coefficient, or is dropped if that tap is not in the
  grid either
- If every delay is within 1 % of the spacing from a tap, each path goes to its nearest tap with its full coefficient
- All snapshots of the output have the same number of taps `n_taps`, the size of the largest grid; the taps beyond a
  shorter grid have zero coefficients and delay 0, taps that are zero in all snapshots at the end are removed

### Usage:
```matlab
[ coeff_q, delay_q ] = quadriga_lib.quantize_delays( coeff, delay, tap_spacing, max_taps, ...
    power_exponent, fix_taps );
```

### Inputs:
- **`coeff`** — Complex coefficients, the paths of a snapshot zero-padded to `n_path`; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`; finite, above
  `-tap_spacing / 2` and below `2^31` · `tap_spacing`
- **`tap_spacing`** — Tap spacing in s, > 0 and finite; 5 ns is a sampling rate of 200 MHz; default: 5e-9
- **`max_taps`** — Largest number of taps of a grid; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent $\alpha$, > 0; default: 1
- **`fix_taps`** — Which coefficients share a grid; default: 0

  | Value | Grid |
  | ----- | ---- |
  | 0 | One grid per antenna pair and snapshot |
  | 1 | One grid for all antenna pairs and snapshots |
  | 2 | One grid per snapshot for all its antenna pairs; the grids of the snapshots differ |
  | 3 | One grid per antenna pair for all snapshots |

### Outputs:
- **`coeff_q`** — Complex coefficients per tap; `[n_rx, n_tx, n_taps, n_snap]`
- **`delay_q`** — Tap delays in s; `[n_rx, n_tx, n_taps, n_snap]`, or `[1, 1, n_taps, n_snap]` for delays
  `[1, 1, n_path, n_snap]` with `fix_taps` 1 or 2

### Notes:
- Negative delays: delays in `(-tap_spacing / 2, 0)` count as 0, all their power goes to tap 0 (the error is below
  `tap_spacing / 2`, as for rounding to the nearest tap)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](#get_channels_spherical), [get_channels_planar](#get_channels_planar) and
  [get_channels_multifreq](#get_channels_multifreq)) are negative for element pairs that are closer to each other than the phase centres,
  by up to (half the Tx array size + half the Rx array size) / propagation speed. They stay above
  `-tap_spacing / 2` only while these half sizes add up to less than about 0.75 m for radio at 5 ns taps, or 3.6 mm
  for sound at 48 kHz taps, i.e. practically never for acoustic arrays: use absolute delays then
- The upper limit of `2^31` taps is e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz; run time and memory depend on the
  number of paths and taps, not on the delay values

### See also:
- [freq_response](#freq_response) (frequency response of the paths or taps)
- [calc_delay_spread](#calc_delay_spread) (delay spread, with an optional delay granularity)

---

<a id="sec-channel-generators"></a>
# Channel generators

The functions of this section compute the channel coefficients of a link from its propagation paths and the antenna
arrays at both ends. They apply the antenna patterns, the polarization and the motion of the antennas, and return the
coefficients and delays that the functions of section "Channel data and processing" work with. The coefficients are
given per port pair: `n_tx` and `n_rx` count the ports of the arrays, after their port weights.

**Generators:**

- [get_channels_spherical](#get_channels_spherical): paths given by their first- and last-bounce scatterers; each antenna element sees the path
  from its own position (spherical wavefronts), so it covers large arrays and short distances
- [get_channels_planar](#get_channels_planar): paths given by their departure and arrival angles; all elements of an array see the same
  direction (planar wavefronts)
- [get_channels_multifreq](#get_channels_multifreq): as [get_channels_spherical](#get_channels_spherical), at several frequencies with frequency-dependent antennas,
  path gains and polarization, for radio and acoustic waves
- [get_channels_irs](#get_channels_irs): paths over an intelligent reflecting surface (IRS), from two path sets that meet at the IRS array
- [get_channels_ieee_indoor](#get_channels_ieee_indoor): complete channels of the IEEE TGn, TGac, TGah and TGax indoor models
- [get_channels_tr38901](#get_channels_tr38901): complete channels of a 3GPP TR 38.901 scenario (UMa, UMi, RMa, SMa, InH, InF) in one call

The paths come from a ray tracer (e.g. [qrt_file_read](#qrt_file_read)), from the geometry-based stochastic model, or from the user.
For complete channels of a network layout, see [get_channels_layout](#get_channels_layout).

**Terms** (full list in the [Glossary](quadriga_lib_api_cpp.md#glossary)):

- **Path gain**: the linear power gain of a path without antennas, `[n_path, 1]`
- **Path length**: the length of a path from the Tx to the Rx phase center in m, which sets its delay
- **Polarization transfer matrix** `polmat`: the Jones matrix of each path, `[8, n_path]` with interleaved real and
  imaginary parts, rows 1-2 (rx V, tx V), 3-4 (rx H, tx V), 5-6 (rx V, tx H), 7-8 (rx H, tx H) (Glossary,
  "Polarization")
- **Absolute delays**: `use_absolute_delays = true` keeps the LOS delay in all paths; otherwise the delays are
  relative to the LOS path
- **Vector form**: [get_channels_planar](#get_channels_planar) and [get_channels_spherical](#get_channels_spherical) take the path inputs of several snapshots as
  cell arrays, one cell per snapshot, and return cell arrays

---
<a id="get_channels_ieee_indoor"></a>
## get_channels_ieee_indoor
Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_ieee_indoor) · [Python](quadriga_lib_api_python.md#channel.ieee_indoor)

- Channels of the IEEE 802.11 indoor models A to F (TGn), with the TGac extensions (several users with offset angles,
  tap interpolation for shorter tap spacings), the floor loss of TGah below 1 GHz and the floor and wall loss of TGax
  above; one channel per user
- 2D model: azimuth angles only and motion in the horizontal plane; [get_channels_planar](#get_channels_planar) computes the coefficients
  with the AP and station arrays
- Static channels (`observation_time = 0`, one snapshot) or time-evolving channels with the Doppler shifts of the
  station and the environment

### Usage:
```matlab
chan = quadriga_lib.get_channels_ieee_indoor( ap_array, sta_array, channel_type, center_freq, ...
    tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, ...
    dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, ...
    kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB );
```

### Inputs:
- **`ap_array`**, **`sta_array`** — AP and station array, struct, fields in [generate_antenna](#generate_antenna); `n_ap` and `n_sta`
  ports
- **`channel_type`** — Model: `'A'`, `'B'`, `'C'`, `'D'`, `'E'` or `'F'`
- **`center_freq`** — Carrier frequency in Hz, finite, above 100 MHz; default: 5.25e9
- **`tap_spacing`** — Tap spacing in s, 10 ns divided by a power of 2; default: 10e-9
- **`n_users`** — Number of users, >= 1; default: 1
- **`observation_time`** — Observation time in s, finite, >= 0; 0: one snapshot; default: 0
- **`update_interval`** — Time between snapshots in s, finite, > 0; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h, finite; default: 0
- **`speed_env_kmh`** — Speed of the environment in km/h, finite; TGn 1.2, TGac 0.089; default: 1.2
- **`dist`** — Horizontal AP-station distance in m per user, finite, > 0; one value for all users or `[n_users, 1]`;
  default: 4.99
- **`n_floors`** — Floors between AP and station per user, 0 to 4; one value or `[n_users, 1]`; default: 0
- **`uplink`** — Return the uplink, with the station as Tx; default: `false`
- **`offset_angles_deg`** — Offset angles in degrees per user, finite; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  `[4, n_users]`; default: `[]` (0 for one user, the pseudo-random offsets of TGac for several)
- **`n_subpath`** — Sub-paths per tap, >= 1; default: 20
- **`doppler_effect`** — Special Doppler (Notes): the mains frequency in Hz for models D and E, the vehicle speed in
  km/h for model F; finite, >= 0; 0 disables it; default: 50
- **`seed`** — Seed of the random numbers; -1: the system random device; other values give the same channels on all
  platforms; default: -1
- **`kf`** — Ricean K-factor of the first tap, linear, finite, >= 0; default: `NaN` (model default, Notes)
- **`xpr_nlos`** — Cross-polarization ratio of the NLOS paths, linear, finite, > 0; default: `NaN` (model default)
- **`sf_std_los_dB`**, **`sf_std_nlos_dB`** — Standard deviation of the shadow fading in dB below and above the
  breakpoint distance, finite, >= 0; default: `NaN` (model default)
- **`dist_breakpoint`** — Breakpoint distance in m, finite, > 0; default: `NaN` (model default)
- **`n_walls`** — Walls between AP and station per user (TGax); one value or `[n_users, 1]`; default: 0
- **`wall_loss_dB`** — Loss of one wall in dB, finite; TGax: 5 or 7; default: 5

### Output:
- **`chan`** — Channels, struct array `[1, n_users]`, fields in [hdf5_read_channel](#hdf5_read_channel): `name` (`'user_<i>'`, 0-based),
  `center_freq`, the positions and orientations (station positions `[3, n_snap]`), `coeff`, `delay`
  `[n_rx, n_tx, n_path, n_snap]` (downlink `[n_sta, n_ap, n_path, n_snap]`) and `path_gain` `[n_path, n_snap]`: the
  path powers without antennas, with path loss and shadow fading, whose mean over the snapshots is the power of the
  model

### Notes:
- **Model defaults** (`NaN` in the override inputs), with the LOS shadow fading 3 dB and the NLOS XPR 2 (3 dB) for all
  models; below 1 GHz (TGah) both shadow fading values are 1 dB lower:

  | Model | A | B | C | D | E | F |
  | --- | --- | --- | --- | --- | --- | --- |
  | KF, linear | 1 | 1 | 1 | 2 | 4 | 4 |
  | Breakpoint distance in m | 5 | 5 | 5 | 10 | 20 | 30 |
  | NLOS shadow fading in dB | 4 | 4 | 5 | 5 | 6 | 6 |

- **LOS**: a user has a LOS path below the breakpoint distance on the floor of the AP, and always when `kf` is given;
  the KF splits the power of the first tap between the LOS path and the first NLOS cluster
- **Paths**: one per tap of the model tables with non-zero power, plus the LOS path; each tap sums `n_subpath` sub-paths
  with Laplacian angular spreads. Below 10 ns `tap_spacing`, the taps are interpolated (TGac)
- **Path loss** in dB at the distance $d$ = `dist`: $20 \lg(4\pi f_c / c) + 20 \lg d$ below the breakpoint distance
  $d_\mathrm{BP}$, $20 \lg(4\pi f_c / c) + 20 \lg d_\mathrm{BP} + 35 \lg(d / d_\mathrm{BP})$ beyond, plus the floor
  loss (TGah below 1 GHz, TGax above) and `n_walls` times `wall_loss_dB`. The shadow fading uses the LOS value below
  the breakpoint, else the NLOS value; below 1 GHz on another floor, the TGah floor values replace it
- **Geometry**: the AP at the origin, the station at the distance `dist` in the direction of the AoD LOS offset and
  3 m higher per floor; floors add loss but no delay. The delays are relative to the LOS delay
- **Time evolution**: `1 + floor(observation_time / update_interval)` snapshots, `update_interval` apart; each sub-path
  turns by $e^{j 2\pi f_D t}$ with the Doppler shift $f_D$ of the station speed plus a random environment Doppler
  (bell-shaped spectrum, TGn). The station moves along its heading, toward the AP for an AoA LOS offset of 0; moving
  toward the AP or a scatterer gives a positive $f_D$, as [get_channels_spherical](#get_channels_spherical) along the returned station
  positions
- **Special Doppler** (`doppler_effect` > 0): models D and E modulate three taps with fluorescent light at the
  harmonics of the mains frequency, model F adds the reflection of a moving vehicle to one tap (TGn)
- **Uplink**: the coefficients and delays are the transposed downlink ones (reciprocal channel, no conjugation, the same
  Doppler shift); the positions and orientations are swapped, the station is the Tx

### See also:
- [get_channels_planar](#get_channels_planar) (computes the coefficients)
- [hdf5_write_channel](#hdf5_write_channel) (store the channels)
- [IEEE 802.11-03/940r4 - TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc)
- [IEEE 802.11-09/0308r12 - TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc)
- [IEEE 802.11-11/0968r4 - TGah Channel Model](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx)
- [IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx)
- [S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025](http://quadriga-lib.org/publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf)

---
<a id="get_channels_irs"></a>
## get_channels_irs
Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_irs) · [Python](quadriga_lib_api_python.md#channel.irs)

- Paths of one link over an intelligent reflecting surface (IRS) with `n_irs` elements: segment 1 from the Tx to the
  IRS and segment 2 from the IRS to the Rx, each computed by [get_channels_spherical](#get_channels_spherical) with the IRS elements as Rx and
  Tx elements; each output path combines one path of each segment
- Each IRS element $k$ reflects once with its weight $w_k$, column `i_irs` of the IRS port weights (a codebook entry):
  per path pair, $\mathbf{H} = \mathbf{B} \, \mathrm{diag}(\mathbf{w}) \, \mathbf{A}$ with the element channels
  $\mathbf{A}$ of segment 1 `[n_irs, n_tx]` and $\mathbf{B}$ of segment 2 `[n_rx, n_irs]`
- The delay of a pair is the sum of its segment delays, each the $|w_k|^2$-weighted mean over the IRS elements; the
  departure angles come from segment 1, the arrival angles from segment 2, as $|w_k|^2$-weighted mean directions
- Kept are the path pairs whose strongest coefficient has a power above `threshold_dB`, or those set in `active_path`;
  the pair of segment-1 path $i_1$ and segment-2 path $i_2$ (1-based) has the index $(i_1 - 1) n_{\mathrm{path},2} + i_2$
  in `active_path` and `active_path_out`, and the output keeps this order
- The outputs come in another order than in C++: `active_path_out` comes before the angles, so that the angles can be
  left off at the end

### Usage:
```matlab
[ coeff, delay, active_path_out, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    tx_array, rx_array, irs_array, tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, irs_array_2, active_path, use_kernel );
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, struct, fields in [generate_antenna](#generate_antenna); `n_tx` and `n_rx` ports
- **`irs_array`** — IRS array with `n_irs` elements, struct, valid, with port weights `[n_irs, n_codebook]` that hold
  the reflection weights; its patterns serve both segments unless `irs_array_2` is given
- **`tx_pos`**, **`rx_pos`**, **`irs_pos`** — Tx, Rx and IRS position in m, finite; 3 elements
- **`tx_orientation`**, **`rx_orientation`**, **`irs_orientation`** — Orientations, Euler angles (bank, tilt, heading)
  in rad, finite; 3 elements
- **`fbs_pos_1`**, **`lbs_pos_1`** — First- and last-bounce scatterer positions of the segment-1 paths in m;
  `[3, n_path_1]`
- **`path_gain_1`** — Path gains of segment 1, linear power; `[n_path_1, 1]`
- **`path_length_1`** — Path lengths of segment 1 in m between the Tx and IRS phase centres, as in
  [get_channels_spherical](#get_channels_spherical); `[n_path_1, 1]`
- **`polmat_1`** — Jones matrices of segment 1, the IRS receiving (section description); `[8, n_path_1]`
- **`fbs_pos_2`**, **`lbs_pos_2`**, **`path_gain_2`**, **`path_length_2`**, **`polmat_2`** — The same for segment 2
  from the IRS to the Rx, the IRS transmitting; `n_path_2` paths
- **`i_irs`** — Codebook entry, 1-based column of the IRS port weights, at most `n_codebook`; default: 1
- **`threshold_dB`** — Minimum power in dB of the strongest coefficient of a kept path pair; not used with
  `active_path`; default: -140
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0, as in
  [get_channels_spherical](#get_channels_spherical); default: 0
- **`use_absolute_delays`** — Keep the delay of the direct Tx-Rx distance in all delays; `false`: delays relative to it
  (Notes); default: `false`
- **`irs_array_2`** — Patterns of the IRS for segment 2 (asymmetric IRS), struct, valid, `n_irs` elements; its port
  weights and element positions are not used; default: `[]` (`irs_array`)
- **`active_path`** — Path pairs to keep, replacing `threshold_dB`; logical or numeric (nonzero = keep);
  `[n_path_1 * n_path_2, 1]`; default: `[]` (`threshold_dB`)
- **`use_kernel`** — Kernel of both segments, as in [get_channels_spherical](#get_channels_spherical); default: 0

### Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_path_irs]` with the number of kept path pairs `n_path_irs`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** — Kept path pairs, for `active_path` of a later call; logical; `[n_path_1 * n_path_2, 1]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_path_irs]`

### Notes:
- **IRS port weights**: an `irs_array` without `port_weights_re` / `port_weights_im` gives an error here, unlike in the
  other functions, where they mean identity; the IRS elements are not combined into ports
- **Segments**: path lengths, LOS paths and the per-element geometry as in [get_channels_spherical](#get_channels_spherical); a segment
  without paths (all its path inputs empty) gives the outputs `[n_rx, n_tx, 0]`
- **Relative delays**: the delay of the direct Tx-Rx distance (not via the IRS) is subtracted; delays can be slightly
  negative when the IRS is close to the Tx-Rx line

### See also:
- [get_channels_spherical](#get_channels_spherical) (the channel of each segment)

---
<a id="get_channels_multifreq"></a>
## get_channels_multifreq
Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_multifreq) · [Python](quadriga_lib_api_python.md#channel.multifreq)

- As [get_channels_spherical](#get_channels_spherical), at the output frequencies `freq_new`, with multi-frequency arrays (frequency-dependent
  patterns and port weights), path gains and Jones matrices; for radio waves and, with `propagation_speed` = 343 m/s
  and scalar `polmat`, for sound
- The geometry (element-pair path lengths, angles, LOS path) is computed once; at each output frequency, the patterns
  and port weights of the arrays and `path_gain` and `polmat` are interpolated between their two neighbouring entries
- The coefficients follow the formula of [get_channels_spherical](#get_channels_spherical) with $f_c$ = `freq_new(k)` and $c$ =
  `propagation_speed`; the frequency is the 4th dimension of the outputs, a multi-frequency channel

### Usage:
```matlab
[ coeff, delay ] = quadriga_lib.get_channels_multifreq( tx_array, rx_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, ...
    freq, freq_new, use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel );
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx multi-frequency array, struct array `[1, n]` with one entry per frequency
  in any order, fields in [generate_antenna](#generate_antenna), valid, at least one entry; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; 3 elements
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; 3 elements
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`
- **`path_gain`** — Path gains at the frequencies `freq`, linear power; `[n_path, n_freq]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres, as in [get_channels_spherical](#get_channels_spherical);
  `[n_path, 1]`
- **`polmat`** — Jones matrices at the frequencies `freq` (section description): `[8, n_path, n_freq]`, or
  `[2, n_path, n_freq]` for a scalar coefficient (real and imaginary part, sound)
- **`freq`** — Sample frequencies of `path_gain` and `polmat` in Hz, in any order; `[n_freq, 1]`, at least one
- **`freq_new`** — Output frequencies in Hz; `[n_freq_new, 1]`, at least one
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative to it;
  default: `false`
- **`add_fake_los_path`** — Put the LOS path into an added slot 1, with zero power if there is none; default: `false`
- **`propagation_speed`** — Wave speed in m/s, > 0; default: 299792458
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC, 2 = AVX2
  (single-precision arithmetic; error if unavailable); default: 0

### Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_out, n_freq_new]`, `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out, n_freq_new]`

### Notes:
- **Interpolation**: `path_gain` linear; patterns, port weights and `polmat` per complex value, the amplitude linear and
  the phase along the shorter arc (linear for nearly opposite phases); constant outside the range of each grid
- **Scalar `polmat`**: only the $F_\theta$ patterns are used,
  $a_l = \sqrt{g_l} \, F_{\theta,r} \, m_l \, F_{\theta,t} \, e^{-j 2\pi f d / c}$ with the complex scalar $m_l$
- **Paths**: NLOS path length, LOS path, `add_fake_los_path` and 0 paths as in [get_channels_spherical](#get_channels_spherical)
- **Relative delays**: negative delays as in [get_channels_spherical](#get_channels_spherical), with $c$ = `propagation_speed`; for sound, this
  affects practically every array, so use absolute delays for [quantize_delays](#quantize_delays)
- **Delay phase**: the coefficients carry the phase of the absolute path length also with relative delays;
  [freq_response_mf](#freq_response_mf) with `remove_delay_phase = true` therefore needs `use_absolute_delays = true`

### See also:
- [get_channels_spherical](#get_channels_spherical) (one frequency)
- [freq_response_mf](#freq_response_mf) (frequency response of a multi-frequency channel)
- [join_arrays](#join_arrays) (multi-frequency arrays)
- [generate_speaker](#generate_speaker) (loudspeaker arrays)

---
<a id="get_channels_planar"></a>
## get_channels_planar
Calculate MIMO channel coefficients and delays for planar wave propagation

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_planar) · [Python](quadriga_lib_api_python.md#channel.planar)

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; a path is given by its
  departure and arrival angles, which all elements of an array share (planar wavefronts)
- Path $l$ of the element pair $(t, r)$ has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \quad \tau_l = d / c, \quad d = \ell_l + d_t + d_r$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  at the path angles, the path length $\ell_l$ between the phase centres (used as given), the distances $d_t$, $d_r$ of
  the elements from the wavefront planes through the phase centres (`dist` of [interpolate_pattern](#interpolate_pattern)), the center
  frequency $f_c$ and the speed of light $c$; the port weights then combine the elements into ports
- The positions enter only through the Tx-Rx distance (relative delays, LOS path)
- Vector form: with cell arrays as path inputs, one cell per snapshot, the outputs are cell arrays; the snapshots are
  computed in parallel

### Usage:
```matlab
[ coeff, delay, rx_Doppler ] = quadriga_lib.get_channels_planar( tx_array, rx_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, polmat, ...
    center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, struct, fields in [generate_antenna](#generate_antenna); `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; 3 elements; vector form: 3 elements (all snapshots) or
  `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`aod`**, **`eod`** — Azimuth and elevation of departure in rad, GCS; `[n_path, 1]`; vector form: cell arrays with
  `n_snap` entries
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in rad, GCS (the direction from the Rx toward the incoming
  path); `[n_path, 1]`; vector form: cell arrays
- **`path_gain`** — Path gains, linear power; `[n_path, 1]`; vector form: cell array
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; `[n_path, 1]`; vector form: cell array
- **`polmat`** — Jones matrices, real and imaginary parts interleaved (section description); `[8, n_path]`; vector
  form: cell array
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes); default: 0
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative to it
  (Notes); default: `false`
- **`add_fake_los_path`** — Put the LOS path into an added slot 1, with zero power if there is none (Notes); default:
  `false`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC, 2 = AVX2
  (single-precision arithmetic; error if unavailable); default: 0

### Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_out]` with `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`; vector form: cell arrays `[n_snap, 1]` of such arrays
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`rx_Doppler`** — Doppler weight of each path for an Rx moving along its local x-axis: the cosine of the angle
  between this axis and the arrival direction; the Doppler shift at the speed $v$ is $v f_c / c$ times this weight,
  positive when the Rx moves toward the incoming path; `[1, n_out]`

### Notes:
- **LOS path** (only with `add_fake_los_path`): the strongest path whose `path_length` is within 0.1 mm of the Tx-Rx
  distance; the angles are not checked
- **`add_fake_los_path`**: slot 1 holds the LOS path, with zero power (and `rx_Doppler` 0) if there is none; a LOS path
  among the inputs moves to slot 1 and leaves a zero-power path in its own slot (input index + 1); the other paths
  keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays; element pairs can then get
  negative delays, as in [get_channels_spherical](#get_channels_spherical)
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays of a port pair are means over its element pairs, weighted by the port-weight powers
- **Vector form**: as in [get_channels_spherical](#get_channels_spherical)

### See also:
- [get_channels_spherical](#get_channels_spherical) (paths given by scatterers, spherical wavefronts)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (IEEE indoor channels, computed with this function)
- [freq_response](#freq_response) (frequency response of the coefficients)
- [quantize_delays](#quantize_delays) (delays on a tap grid)

---
<a id="get_channels_spherical"></a>
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_spherical) · [Python](quadriga_lib_api_python.md#channel.spherical)

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; every element sees a path
  from its own position (spherical wavefronts), so delays and phases follow the geometry of each element pair
- A path runs from the Tx over its first-bounce scatterer (FBS) and last-bounce scatterer (LBS) to the Rx; the patterns
  of both arrays are interpolated at the departure and arrival angles of each element pair ([interpolate_pattern](#interpolate_pattern))
- Path $l$ of an element pair at the distance $d$ along the path has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \qquad \tau_l = d / c$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  of the Rx and Tx element, the center frequency $f_c$ and the speed of light $c$; the port weights then combine the
  elements into ports
- Vector form: with cell arrays as path inputs, one cell per snapshot, the outputs are cell arrays; the snapshots are
  computed in parallel

### Usage:
```matlab
[ coeff, delay, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( tx_array, ...
    rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, struct, fields in [generate_antenna](#generate_antenna); `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; 3 elements; vector form: 3 elements (all snapshots) or
  `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`; vector form: cell
  arrays with `n_snap` entries, as [qrt_file_read](#qrt_file_read) returns them
- **`path_gain`** — Path gains, linear power; `[n_path, 1]`; vector form: cell array
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; a value below the Tx-FBS-LBS-Rx polyline
  (e.g. 0) gives the polyline length (Notes); `[n_path, 1]`; vector form: cell array
- **`polmat`** — Jones matrices, real and imaginary parts interleaved (section description); `[8, n_path]`; vector
  form: cell array
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes); default: 0
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative to it
  (Notes); default: `false`
- **`add_fake_los_path`** — Put the LOS path into an added slot 1, with zero power if there is none (Notes); default:
  `false`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto (GENERIC in this
  version), 1 = GENERIC, 2 = AVX2 (single-precision arithmetic; error if unavailable); default: 0

### Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_out]` with `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`; vector form: cell arrays `[n_snap, 1]` of such arrays
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_out]`

### Notes:
- **NLOS path length**: the length between the phase centres is $\max(\ell, d_\mathrm{poly})$ with `path_length`
  $\ell$ and the length $d_\mathrm{poly}$ of the Tx-FBS-LBS-Rx polyline; the excess, e.g. from more than two
  interactions as in [qrt_file_read](#qrt_file_read), is added to the FBS-LBS segment
- **LOS path**: the path whose length is closest to the Tx-Rx distance, within 0.1 mm; its element pairs use the
  direct line between the elements instead of the FBS and LBS
- **`add_fake_los_path`**: slot 1 holds the LOS path, with zero power if there is none; a LOS path among the inputs moves
  to slot 1 and leaves a zero-power path in its own slot (input index + 1); the other paths keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays. Element pairs closer to each other
  than the phase centres then get negative delays, down to about $-(D_t + D_r) / (2c)$ with the array sizes $D$;
  [quantize_delays](#quantize_delays) rejects delays at or below `-tap_spacing / 2`, so large arrays need absolute delays
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays and angles of a port pair are means over its element pairs, weighted by the
  port-weight powers (the angles via the mean direction)
- **Vector form**: `n_snap` is the length of the cell arrays; an error names the lowest failing snapshot
  (`Snapshot <i>: `, 1-based); invalid arrays and `use_kernel` give errors without this prefix

### See also:
- [get_channels_planar](#get_channels_planar) (paths given by angles, planar wavefronts)
- [get_channels_multifreq](#get_channels_multifreq) (several frequencies)
- [get_channels_irs](#get_channels_irs) (paths over an IRS)
- [qrt_file_read](#qrt_file_read) (path data of many snapshots)
- [freq_response](#freq_response) (frequency response of the coefficients)
- [quantize_delays](#quantize_delays) (delays on a tap grid)

---
<a id="get_channels_tr38901"></a>
## get_channels_tr38901
Generate the channels of a TR 38.901 scenario in one call

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_tr38901) · [Python](quadriga_lib_api_python.md#channel.tr38901)

- Channels between the UTs of a drop and the BSs of a 3GPP TR 38.901 scenario (UMa, UMi, RMa, SMa, InH, InF), with the
  given BS and UT antenna arrays; by default one channel per UT and frequency, to its serving BS sector
- The scenario sets the BS layout and the drop of the UTs as in TR 38.901 (table in Notes); optional inputs change
  single values, the time evolution and which links are returned
- Built on the layout functions ([generate_layout](#generate_layout), [init_link_states](#init_link_states), [attach_terminals](#attach_terminals), [generate_rays](#generate_rays),
  [get_channels_from_rays](#get_channels_from_rays)); `layout` returns the simulated layout, to continue there

### Usage:
```matlab
[ chan, gain_dB, layout, index ] = quadriga_lib.get_channels_tr38901( bs_array, ut_array, scenario, ...
    center_freq, n_ut, seed, spacing, grid, area, bs, ut, indoor, time_grid, los, output, attach, links, ...
    link_gain, uplink, o2i_low, config_dirs );
```

### Inputs:
- **`bs_array`** — BS antenna array, struct, fields in [generate_antenna](#generate_antenna); every sector uses it, turned to the sector heading
- **`ut_array`** — UT antenna array, struct, fields in [generate_antenna](#generate_antenna)
- **`scenario`** — `'UMa'`, `'UMi'`, `'RMa'`, `'SMa'`, `'InH_Open'`, `'InH_Mixed'`, `'InF_SL'`, `'InF_DL'`,
  `'InF_SH'`, `'InF_DH'` or `'InF_HH'`; also with the prefix `'38.901_'`
- **`center_freq`** — Carrier frequencies in Hz, ascending; one set of channels per frequency; default: 3.5e9;
  `[n_freq]`
- **`n_ut`** — UTs per cell (per BS sector), >= 1; default: 1
- **`seed`** — Seed of the random numbers: the same seed gives the same drop and channels; -1 = a random seed,
  which `layout.seed` returns; default: -1

Optional changes to the scenario layout (defaults in Notes). Give only the values to change; NaN keeps a default.
[generate_layout](#generate_layout) describes all values:

- **`spacing`** — Inter-site distance in m, e.g. `200`; indoor: BS spacing in x and y; default: `[]`
- **`grid`** — Rings of sites around the centre site, e.g. `0` = one site, `2` = 19 sites; the
  number of sectors as third value, e.g. `[0, NaN, 1]`; indoor: BSs in x and y; default: `[]`
- **`area`** — Hall size in m, length, width, height (InH, InF); default: `[]`
- **`bs`** — BS height in m, e.g. `30`; then the mechanical downtilt and an offset of the sector
  headings in degrees; default: `[]`
- **`ut`** — UT height and minimum distance to the BS in m, speed in m/s, e.g. `[NaN, NaN, 30 / 3.6]` =
  all UTs at 30 km/h; then the orientation (bank, tilt, heading) in degrees; default: `[]`
- **`indoor`** — Share of the UTs in buildings, e.g. `0.5` (UMa, UMi, RMa, SMa); then the range of floors
  and the floor height; default: `[]`

Simulation settings:

- **`time_grid`** — `[T, dt]` in s: snapshots at 0, dt, 2 dt, … up to T, while the UTs move; `dt`
  defaults to 1 ms, e.g. `0.1` = 101 snapshots over 100 ms; `[]` = one snapshot (static channel); default: `[]`
- **`los`** — `'auto'`: LOS or NLOS as drawn from the LOS probability of the scenario; `'los'` or `'nlos'`: that state for
  all links, also of the UTs in buildings and to the other sectors, so it can change the serving sector (`'InF_HH'` is
  always LOS); default: `'auto'`
- **`output`** — `'taps'`: the rays of a cluster combined into taps; `'rays'`: every ray as a path, with angles
  and path gains; default: `'taps'`
- **`attach`** — How the serving sector is chosen: `'large_scale'` (coupling gain) or `'rsrp'` (Notes); default: `'large_scale'`
- **`links`** — Which links are returned: `'serving'`, the serving sector of each UT; `'site'`, all sectors of its
  serving site; `'all'`, all BS sectors, e.g. for interference studies (Outputs); default: `'serving'`
- **`link_gain`** — false: without path loss, building loss and shadow fading, so that the power is that of the
  antennas and the small-scale fading; `gain_dB` and the serving sector still include them; default: true
- **`uplink`** — true: the uplink, with the UT as Tx (Notes); default: false
- **`o2i_low`** — Share of buildings with the low-loss model of the building penetration loss, in [0, 1];
  NaN = the TR 38.901 value (50 % in UMa and UMi); not for InH, InF; default: NaN
- **`config_dirs`** — Folders with own scenario files (`.conf`), searched before the shipped ones; default: `{}`

### Outputs:
- **`chan`** — Channels as struct array, fields in [hdf5_read_channel](#hdf5_read_channel); what taps and rays hold: [gbsm_single_link_channel](#gbsm_single_link_channel)
  - `'serving'`: one channel per UT and frequency, all UTs at the first frequency, then at the next: `chan(i_ut + (i_freq - 1) * n_ut_total)`, 1-based
  - `'site'`, `'all'`: one channel per BS sector, UT and frequency; the sectors of the first UT one after the other,
    then those of the next UT, then the next frequency; `index` names the BS sector, UT and frequency of each channel
  - UTs in drop order: the first `n_ut` were dropped in cell 1 (site BS1, sector 0), the next `n_ut` in cell 2, …;
    the serving sector is the one with the highest gain, often not the cell of the drop. The name says it too, e.g.
    `'BS5:0_UT11'` = site BS5, sector 0, UT 11
  - Coefficients and delays `[n_ut_port, n_bs_port, n_tap, n_snap]` (downlink; `uplink` swaps the first two)
- **`gain_dB`** — Coupling gain or RSRP in dB of each UT to each BS sector, e.g. for the SIR; `[n_ut_total, n_bs_node]`
- **`layout`** — The simulated layout: BS and UT positions, link states, pairing; fields in [layout_csv_read](#layout_csv_read)
- **`index`** — BS sector (node), UT and frequency of each channel, 1-based; uint64; `[3, n_channel]`

### Example:
```matlab
bs = quadriga_lib.generate_antenna( '3gpp', 10, 3.5e9, [], [4, 4, 1, 1] );
ut = quadriga_lib.generate_antenna( 'xpol' );
chan = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 10, 1 );                          % 19 sites
chan = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 10, 1, [], 0, [], [], [], [], 0.1 );   % one site, 100 ms
```

### Notes:
- **Scenario defaults** (TR 38.901 Tables 7.2-1 to 7.2-5 and 7.8-7); the UTs are at 1.5 m (InH 1 m, InF_HH 3 m),
  in buildings on their floor:

  | Scenario | BS layout | BS height | UTs indoors | UT speed |
  | --- | --- | --- | --- | --- |
  | UMa | 19 sites × 3 sectors, ISD 500 m, wrap-around | 25 m | 80 % | 3 km/h |
  | UMi | 19 sites × 3 sectors, ISD 200 m, wrap-around | 10 m | 80 % | 3 km/h |
  | RMa | 19 sites × 3 sectors, ISD 1732 m, wrap-around | 35 m | 50 %, the others in cars | 3 km/h, cars 40 km/h |
  | SMa | 19 sites × 3 sectors, ISD 1299 m, wrap-around | 35 m | 80 %, the others in cars | 3 km/h, cars 40 km/h |
  | InH_Open, InH_Mixed | Hall 120 × 50 m, 12 BSs 20 m apart | 3 m | all | 3 km/h |
  | InF_SL, InF_DH | Hall 120 × 60 × 10 m, 18 BSs 20 m apart | 1.5 m (SL), 8 m (DH) | all | 3 km/h |
  | InF_DL, InF_SH, InF_HH | Hall 300 × 150 × 10 m, 18 BSs 50 m apart | 1.5 m (DL), 8 m (SH, HH) | all | 3 km/h |

  `n_ut` UTs are dropped in each cell (`n_ut_total` = `n_ut` × the number of BS sectors); with the wrap-around, the
  outer cells see the interference of a large network, also with one site
- **Serving sector**: the BS sector with the highest coupling gain (`'large_scale'`, TR 38.901 Table 7.8-1) or RSRP
  (`'rsrp'`, Table 7.8-2) at the first frequency, see [attach_terminals](#attach_terminals); `'rsrp'` and `links` = `'all'` generate
  the rays of all links, the others only those to the serving sites
- **Time evolution**: the UTs move from their drop position along their direction at their speed; delays and angles of
  the clusters stay, the phases follow the motion (mobility `'phase'` of [get_channels_from_rays](#get_channels_from_rays))
- **Several frequencies**: the clusters come from the same random values at all frequencies, scaled by the spreads of
  each frequency (method `'38.901_MF'`, TR 38.901 clause 7.6.5)
- **Uplink**: the transposed downlink (reciprocal channel, the same Doppler shift), with the two ends swapped in all
  members and the name; `index` and `gain_dB` keep the downlink meaning. This is the uplink of a TDD system; for FDD,
  add the uplink frequency to `center_freq`
- **Memory**: about 90 kB per snapshot for the 57 UTs of the default UMa drop with 2 × 2 ports (16 to 25 taps);
  `links` = `'all'` multiplies it by the number of BS sectors; `'site'` by the sectors per site (3)
- The C++ outputs come after `chan` in their C++ order; the wrapper adds the folder of the shipped conf files to
  `config_dirs`

### See also:
- [hdf5_read_channel](#hdf5_read_channel) (the fields of the channels)
- [freq_response](#freq_response) (frequency response of the channels at given subcarriers)
- [generate_layout](#generate_layout) (the scenario layouts and their inputs in full)
- [get_channels_layout](#get_channels_layout) (channels of any layout)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (the IEEE indoor models)

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
`model` and the model parameters) and optional short names, read from a conf file by [conf_file_read](#conf_file_read). 23 files for TR
38.901 v19.5.0 come with the library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`,
`_NLOS`, `_LOS_O2I`, `_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the
source tree and in `share/quadriga-lib/config` of an installed package; [conf_file_list](#conf_file_list) lists them after the caller's
folders, and [conf_file_read](#conf_file_read) reads one by its name. The MEX files find this folder next to their own folder
`+quadriga_lib`, also after the installation was moved. Their comments give the meaning and unit of each key;
[conf_file_write](#conf_file_write) with `write_defaults = true` writes the complete default set with these comments. Every function
that takes `scenpar` accepts the struct of [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) or any subset of its fields (`[]`
= all defaults), completes a copy with the defaults of missing fields and validates it with [scenpar_validate](#scenpar_validate), whose
messages pass through, also for fields the function does not use. A field `SubpathMethod` (QuaDRiGa's structs) must be
`'legacy'`.

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
| `'38.901'` | `'lsf'`; `'cluster'` 0 and one per frequency; `'ray'` one per frequency | per frequency `k`: [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) (`i_freq = 1`) with slice `k` of all per-frequency inputs (`center_freq`, `lsp`, `lsp_mu`, `lsp_sigma`, `rnd_freq`, `rnd_ray`, `eod_offset_deg`, `gr_eps`) |
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
- With `include_shipped = true` the folder of the conf files that come with the library is searched
  last, by the same rules, so a file in `dirs` overrides the shipped file of the same name
- With `parse_shortnames = true` every line `ShortName = name` that is not a comment adds an entry
  right after its file; other lines are not examined
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if a `ShortName` value is not one name, or if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty
  string `''` is the current directory

### Usage:
```matlab
[ scenarios, file_names, file_dirs ] = quadriga_lib.conf_file_list( dirs, parse_shortnames, include_shipped );
```

### Inputs:
- **`dirs`** — Directories, searched in this order; one string or a cell array of strings; can be
  an empty cell array
- **`parse_shortnames`** — `true`: read the files and add their short names; `false`: list the files
  only, which is faster; default: true
- **`include_shipped`** — `true`: search the folder of the shipped conf files after `dirs`; `false`:
  only `dirs`; default: true

### Outputs:
- **`scenarios`** — Scenario names and short names; cell array of strings; `[n_scen, 1]`
- **`file_names`** — File name including `.conf` for each entry; cell array of strings;
  `[n_scen, 1]`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs`, or the absolute
  path of the shipped folder; cell array of strings; `[n_scen, 1]`

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
- An error is raised if the file cannot be opened or is a directory, or if `fn` is a name (see Inputs)
  that is not a shipped scenario
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
- **`fn`** — Path to the conf file, or the name or short name of a shipped scenario (see
  [conf_file_list](#conf_file_list)) if it does not end in `.conf` and contains no `/` or `\`; a file without the
  extension is read by its path, e.g. `./name`; string
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
- `dual_mobility = true`: Tx and Rx both move; each column is the two-device generator of [sos_val](#sos_val) with the
  decorrelation distance of the column at the Rx and that of its mirror generator at the Tx (e.g. departure and
  arrival quantities exchange). While one end stays, a column changes as in single mobility; when one end moves far,
  the link decorrelates. Swapping Tx and Rx exchanges the mirror columns; a self-mirrored column keeps its value,
  except block 14 of the shared 'ray' set when `PerClusterAS_D` and `PerClusterAS_A` differ. Every column keeps
  its distribution
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
| `'ray'`, 0 or `k` | `14·N·M + N` | blocks 0 to 14, below | `SC_lambda`; block 14 of set 0: 1.6·c·c_DS / `PerClusterAS_A` (in rad), within 0.1 m to 10 km; at the Tx in dual mobility `PerClusterAS_D` in place of `PerClusterAS_A` | [ssf_rays](#ssf_rays) `rnd_ray`: 0 for `'quadriga'`, `k` as slice `k` for the TR methods |

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
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, `PerClusterDS*`,
  `PerClusterAS_A` and `PerClusterAS_D`
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
- `'lsf'` bias removal (`remove_bias`): [sos_remove_bias](#sos_remove_bias) sets the sample mean of columns 1 to 9 over the links
  (in 2D mode all on floor 0) to 0, so these means do not vary from drop to drop; in dual mobility columns 5 (ASA)
  and 7 (ESA), the ASD and ESD fields with Tx and Rx exchanged, are not corrected
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
  read with `normalize_polmat = true` (`uplink = true` gives the reverse link; `normalize_polmat = false` leaves only
  the free-space gain in `power`)
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
    path_length, polmat, method, path_type, tap, gr_eps );
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
- **`gr_eps`** *(optional)* — Complex ground permittivity, used as `conj(gr_eps)`; needed only where a GR ray is
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
  `coeff`, `delay` `[n_rx, n_tx, n_tap, n_snap]` (`'taps'`) or `[n_rx, n_tx, n_path, n_snap]`
  (`'rays'`); `'rays'` also `path_gain`, `path_length`, `polmat`, `path_angles` and in `'drift'`
  `fbs_pos`, `lbs_pos`
- **`par`** — Struct with the fields below

### Fields of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `[n_tap, 1]` or `[n_path, 1]` | Tap index per output tap or per path; uint32 |
| `path_type` | `[n_tap, 1]` or `[n_path, 1]` | 0 = LOS, 1 = GR, 2 = NLOS per path; per tap the type of its rays, 2 if they differ; uint32 |
| `cluster` | `[n_tap, 1]` or `[n_path, 1]` | Cluster per tap or path: tap index / 3, the slot of [ssf_rays](#ssf_rays) (0 = LOS, 1 = GR if enabled, then the NLOS clusters); uint32 |
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
[ lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma ] = ... 
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
- **`gr_eps`** — Complex relative permittivity of the ground, imaginary part >= 0; the complex value of TR 38.901
  eq. (7.6-40) is `conj(gr_eps)`; `[n_link, n_freq]`
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
    rx_pos, lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps, o2i, n_cluster );
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
- **`gr_eps`** *(optional)* — Complex ground permittivity from [lsp_generate](#lsp_generate), used as `conj(gr_eps)`, real part >= 1,
  imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
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
| `gr_eps` | `[n_freq, 1]` or empty | complex double | Ground permittivity, used as `conj(gr_eps)`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | double | First- and last-bounce scatterer positions in [m] | no |
| `fallback` | `[n_ray, 1]` or empty | uint32 | 1 = ray lengthened by the solver fallback | no |

### Usage:
```matlab
rays = quadriga_lib.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, ...
    power, aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps, ...
    pol_variability, random_phase );
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
- **`gr_eps`** *(optional)* — Complex ground permittivity from [lsp_generate](#lsp_generate); required if `GR_enabled`, else
  ignored; `[n_link, n_freq]`
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
<a id="acdf"></a>
## acdf
Empirical cumulative distribution functions (CDF) of several data sets, their average and quantile statistics

Also in: [C++](quadriga_lib_api_cpp.md#acdf) · [Python](quadriga_lib_api_python.md#acdf)

- The CDF of each data set (column of `data`) at the bin centers: each finite value counts for the nearest bin (values
  outside the bins for the first or last one), and the CDF is the cumulative count divided by the number of finite
  values of the set
- Average `cdf_avg` in quantile space: for the probabilities $P = 0, 0.001, \dots, 0.999$, the bins at which the
  CDFs of the sets reach $P$ are averaged over the sets, and the averaged quantile function is mapped back to the
  bins; this averages the position of the distributions instead of their CDF values
- `mu` and `sig`: mean and standard deviation (normalized by $N - 1$) of the quantiles at
  $P = 0.1, 0.2, \dots, 0.9$ over the sets, each taken over the 11 probabilities within $\pm 0.005$; for one set its
  quantiles and 0
- Non-finite values are left out; a set without finite values gets a NaN CDF and is left out of `cdf_avg`, `mu` and
  `sig`, which are NaN if no set has finite values (possible only with given bins)
- Generated bins: `n_bins` equally spaced bins from the smallest to the largest finite value of `data` (±10 % of the
  value, or ±1 for 0, if all are equal); `data` without finite values then throws

### Usage:
```matlab
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

### Inputs:
- **`data`** — Data sets, one per column, magnitudes below about 1e18; `[n_samples, n_sets]`
- **`bins_in`** — Bin centers, finite and strictly ascending; `[n_bins_in, 1]`; default: `[]` (generated bins)
- **`n_bins`** — Number of generated bins, >= 2; not used with `bins_in`; default: 201

### Outputs:
- **`cdf_per_set`** — CDF of each set; `[n_bins_out, n_sets]`
- **`bins_out`** — Bin centers, `bins_in` or the generated ones; `[n_bins_out, 1]`
- **`cdf_avg`** — Average CDF; `[n_bins_out, 1]`
- **`mu`** — Mean of the quantiles at 0.1, ..., 0.9; `[9, 1]`
- **`sig`** — Standard deviation of these quantiles; `[9, 1]`

### See also:
- [calc_delay_spread](#calc_delay_spread) (delay spreads of many CIRs, a typical input)

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
Version number of quadriga-lib

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

<a id="sec-network-layouts-and-mobility"></a>
# Network layouts and mobility

A **layout** describes a radio network for channel simulation: the transmitters (Tx) and receivers (Rx), where they are
and how they move, which antenna each one has and which scenario applies. The functions of this section create layouts,
store them in files, move the nodes along tracks and compute the channels of all Tx-Rx pairs.

**What a layout holds** (a struct, fields in [layout_csv_read](#layout_csv_read)):

- The **Tx nodes** in `layout.tx` (base stations, access points) and the **Rx nodes** in `layout.rx` (user terminals,
  stations), each a struct of flat arrays. A node has a name, a sector number, an antenna and one or more rows:
  positions with orientation, scenario and indoor state
- The **antennas** of the nodes and the simulation frequencies `layout.center_freq`
- The **link states** `layout.links` of one realisation: LOS and indoor state, scenario and seeds of each Tx-Rx pair
  ([init_link_states](#init_link_states))

**Typical workflow** (for the channels of a TR 38.901 scenario in one call, see [get_channels_tr38901](#get_channels_tr38901)):

1. Create a layout: [generate_layout](#generate_layout) from a preset of 3GPP TR 38.901, ITU-R M.2412 or 3GPP TR 37.885, or
   [layout_csv_read](#layout_csv_read) from two CSV files; [drop_terminals](#drop_terminals) drops new UTs into the cells of the Tx nodes
2. Set the frequencies `layout.center_freq` (or give them to [generate_layout](#generate_layout)) and the antennas ([set_antenna](#set_antenna),
   [set_antenna_array](#set_antenna_array)); [eval_antennas](#eval_antennas) generates the antennas for new frequencies
3. Optionally, move the nodes: [generate_track](#generate_track) creates tracks, [set_orientation](#set_orientation) turns the antennas along the
   motion, [sample_track](#sample_track) gives the positions at given times
4. Compute the channels with [get_channels_layout](#get_channels_layout)
5. Store the layout with [layout_csv_write](#layout_csv_write)

For more control, step 4 runs in three stages: [init_link_states](#init_link_states) draws the link states, [generate_rays](#generate_rays) the rays of
the geometry-based stochastic model (GBSM), and [get_channels_from_rays](#get_channels_from_rays) applies the antennas, the motion and the link
gain. Between the stages, link states and rays can be inspected or edited, [attach_terminals](#attach_terminals) selects the serving cell
of each terminal, and [export_rays_qrt](#export_rays_qrt) stores the rays in a QRT file.

**Antennas.** Each node has one antenna. [set_antenna](#set_antenna) sets it from the arguments of [generate_antenna](#generate_antenna), from an
entry of a QDANT file or by the index of an antenna that the layout already has, [set_antenna_array](#set_antenna_array) from an array; in
the CSV files, antenna columns describe it. [eval_antennas](#eval_antennas) generates the arrays from these descriptions. A node has
one antenna for all bands or one per band. A node without an antenna uses the generic antenna: two co-located isotropic
elements with vertical and horizontal polarization.

**Terms** (full list in the [Glossary](quadriga_lib_api_cpp.md#glossary)):

- **Tx nodes**, **Rx nodes**: the two node structs of a layout, `layout.tx` and `layout.rx`. A function that works on
  one of them takes the struct itself or `rx` (`false`: Tx nodes, `true`: Rx nodes)
- **Node**: one transmitter or receiver, identified by its `name` and `sector`, with one antenna and one track
- **Site**: all Tx nodes of one name, e.g. the three sectors of a base station; they share position and scenario
- **Terminal**: all Rx nodes of one name, e.g. the antenna panels of one device; they share their positions
- **Row**: one position and orientation of a node, with the values that apply from there on (speed, scenario, indoor
  state)
- **Point drop**: a node with one row; it stays where it is or, with a speed, moves along its direction
- **Track**: a node with several rows; it moves along its positions
- **Snapshot**: one sample of the channel in time; without snapshot times, each row is one snapshot
- **Segment**: a part of a track with one scenario and indoor state; a new segment starts where one of them changes
- **Propagation link**: one site, one terminal and one segment, with its own LOS and indoor state, large-scale
  parameters and rays
- **Channel link**: one Tx node and one Rx node; it applies their antennas to the rays of its propagation link, which
  the sectors of a site share
- **Band**: one of the simulation frequencies `layout.center_freq`
- **Scenario**: a family (e.g. `UMa`), which draws the LOS and indoor state of each link, or a conf name (e.g.
  `UMa_NLOS`), which fixes them
- **Wrap-around**: copies of the sites, displaced by the lattice vectors `layout.wrap`, so that a cluster of cells tiles
  the plane; each link uses the copy (*image*) nearest to the first position of the terminal, kept for the whole track
- **Orientation**: bank, tilt and heading of the antenna; a positive tilt points down. **Direction**: azimuth and
  elevation of the motion of a point drop

**Conventions.** Angles are in radians in the structs and in degrees in the CSV files. Indices are 1-based: the index
arguments, the returned indices and the index fields `offset`, `pairing` and, in `layout.links`, `i_tx`, `i_rx`,
`i_segment`, `i_conf` and `o2i_model`. `i_antenna`, the antenna of a node, is 0 for the generic antenna and 1 for the
first antenna of the layout; `sector`, `image`, `floor_id`, `indoor`, `floor` and the seeds are values, not indices.
Random values follow from the seeds and the positions, not from the order of evaluation, the batches or the number of
threads. The file format is described in [Layout CSV files](formats/layout_csv.md).

---
<a id="attach_terminals"></a>
## attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

Also in: [C++](quadriga_lib_api_cpp.md#attach_terminals) · [Python](quadriga_lib_api_python.md#attach_terminals)

- Evaluates every Tx node for each terminal and returns the one with the highest coupling gain (`'large_scale'`) or the
  highest RSRP (`'rsrp'`), with its site and the gains of all nodes
- Use it to link each terminal to its serving site in `layout.pairing` (see Example), so that the channel functions
  compute only the serving links
- One call attaches at one frequency, `layout.center_freq(i_freq)`, the first band by default; this suits offloading
  scenarios, where the higher bands serve as data pipes. The result is one serving Tx node per terminal

### Usage:
```matlab
[ node, site, gain_dB ] = quadriga_lib.attach_terminals( layout, rule, rays, i_freq, tx_port );
```

### Inputs:
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states); fields in [layout_csv_read](#layout_csv_read); Tx nodes required if
  there are terminals
- **`rule`** — `'large_scale'` or `'rsrp'`; default: `'large_scale'`
- **`rays`** — `'rsrp'`: rays of all links from [generate_rays](#generate_rays), possibly edited, required; cell array `[n_link, 1]`
  of `1xN` struct arrays, `N = 1` or `n_freq`; default: `{}`
- **`i_freq`** — Frequency index into `layout.center_freq`, 1-based; default: 1
- **`tx_port`** — Port of the Tx antennas, 1-based; default: 1

### Outputs:
- **`node`** — Serving Tx node per terminal, 1-based; uint64; `[n_rx_terminal, 1]`
- **`site`** — Serving site per terminal, 1-based; uint64; `[n_rx_terminal, 1]`
- **`gain_dB`** — Coupling gain or RSRP in [dB]; `[n_rx_terminal, n_tx_node]`

### Example:
```matlab
[ ~, site ] = quadriga_lib.attach_terminals( layout, 'large_scale' );
layout.pairing = [ site.'; 1:numel( site ) ];
```

### Notes:
- **Evaluation**: at the first rows of both nodes and `layout.center_freq(i_freq)`, with the link states of the first
  segment, the wrap image and the antennas as in [get_channels_from_rays](#get_channels_from_rays); `layout.pairing` is ignored
- **`'large_scale'`** (TR 38.901 Table 7.8-1): the link gain of [get_channels_from_rays](#get_channels_from_rays) plus the gain of one
  unit-gain LOS path through both antennas (port weights, element positions, orientations), as [get_channels_planar](#get_channels_planar)
  gives it; no rays
- **`'rsrp'`** (TR 36.873 eq. (8.1-1)): per Rx port the sum of `abs(h)^2` over the rays of [gbsm_single_link_channel](#gbsm_single_link_channel)
  (`output = 'rays'`, mobility `'phase'`), link gain included, transmit power 1
- **Ports**: both rules take `abs(h)^2` of Tx port `tx_port` and average it over the Rx ports, so a dual-polarized UT
  (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT antenna of the TR
- **Choice**: a terminal takes the highest value of its nodes; ties go to the lowest index. A Tx node without a
  propagation link (co-located) has -Inf; a terminal without any link gets node 1
- **Pairing**: `layout.pairing` holds pairs of site and terminal, so the Example links all sectors of the serving site;
  for multi-connectivity, `layout.pairing` may list several sites per terminal, e.g. all sites within a margin of the
  best value in `gain_dB`, and the channel functions then compute the links to all of them

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays) (link states and rays)
- [get_channels_from_rays](#get_channels_from_rays) (channels of the linked pairs)

---
<a id="drop_terminals"></a>
## drop_terminals
Drop UTs into the cells of the Tx nodes of a layout

Also in: [C++](quadriga_lib_api_cpp.md#drop_terminals) · [Python](quadriga_lib_api_python.md#drop_terminals)

- Replaces the Rx nodes by new UTs `UT1`, `UT2`, ..., dropped uniformly into the cells of the sites with the rules of
  the scenario of each site (minimum distance, height, speed, indoor state), as [generate_layout](#generate_layout) drops them
- Use it to drop UTs for any set of Tx nodes, e.g. sites read from a CSV file or edited after [generate_layout](#generate_layout)
- The Tx nodes, the antenna lists and `layout.seed` stay; `pairing` and `links` are emptied. The new UTs get the antenna
  that all old Rx nodes shared (`i_antenna`, with its bands), else the generic antenna

### Usage:
```matlab
layout_out = quadriga_lib.drop_terminals( layout_in, n_ut, bounds, ut, indoor, options, per_cell, seed );
```

### Inputs:
- **`layout_in`** — Valid layout struct, fields in [layout_csv_read](#layout_csv_read); the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 1e7 in total; 0 = no UTs
- **`bounds`** — `[x_min, x_max, y_min, y_max]` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless
  `layout_in.wrap` has two vectors; default: `[]`
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** — As in [generate_layout](#generate_layout); an entry or option that no family
  of the sites uses raises an error; defaults: `[]`, `[]`, `[]`, `true`
- **`seed`** — Seed >= 0; -1 = `layout_in.seed` if it is set, else the system random device; default: -1

### Outputs:
- **`layout_out`** — The layout with the Rx nodes replaced by the dropped UTs

### Notes:
- **Cells**: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the
  wrap images when `layout_in.wrap` is set. A site with several sectors is split into wedges by the nearest sector
  heading (±60° around each boresight for three sectors at equal spacing)
- **Drop area**: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the
  images 0, ±a1, ±a2 and ±(a1 - a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- **Sites without a scenario family** (no scenario, or a conf name): height 1.5 m, no minimum distance unless `ut` entry
  1 gives one, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states
- **Sites with `cell` = 0**, e.g. the micro sites of `M.2412_DenseUrban` config 3, form no cells: with `per_cell`, each
  of their Tx nodes adds `n_ut` UTs to the cell that holds its position, and the UTs keep the minimum distance of the
  site's own family (`ut` entry 1 if given) from them. A site closer than twice the minimum distance to another one can
  leave a cell too little room; such a cell raises an error
- **Order and reproducibility**: with `per_cell` the UTs are ordered by cell. The drop is a function of the seed and
  does not depend on the thread count; each UT draws its heading, direction and floor from its own random stream, keyed
  by its name
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [generate_layout](#generate_layout) (layouts from presets, with this drop)
- [generate_track](#generate_track) (tracks for the UTs)

---
<a id="eval_antennas"></a>
## eval_antennas
Generate the antenna arrays of a layout from the antenna lines

Also in: [C++](quadriga_lib_api_cpp.md#layout) · [Python](quadriga_lib_api_python.md#eval_antennas)

- Generates the arrays `antenna` of a layout from its antenna descriptions `antenna_args`, for the entries that are not
  evaluated yet, e.g. after [layout_csv_read](#layout_csv_read) or [set_antenna](#set_antenna) with the evaluation turned off
- With `center_freq`, sets new simulation frequencies and generates all antennas for them

### Usage:
```matlab
layout_out = quadriga_lib.eval_antennas( layout_in, center_freq );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read)
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; replaces `layout_in.center_freq`
  and regenerates all descriptions; `[]` = keep `layout_in.center_freq`; default: `[]`; `[n_freq]`

### Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

### Notes:
- **Antenna descriptions**: a line of `antenna_args` holds the cells of the antenna columns `antenna_cols` as in the
  [Layout CSV files](formats/layout_csv.md#layout-csv-antennas). Nodes with equal lines share one antenna; `i_antenna` of a node
  is its index, 0 the generic antenna. No line or an empty line is an array without description (set with
  [set_antenna_array](#set_antenna_array)); a line without antenna type is a band that is not set yet
- **Evaluation**: a line for all bands is generated once per band at its frequency, so that an array spaced in
  wavelengths keeps its spacing (an array of co-located elements is stored once for all bands); a line of a band at the
  frequency of that band; a `qdant` line is read as stored. Until an antenna is evaluated (its entry of `antenna` holds
  no arrays, or an array without elements), its nodes use the generic antenna
- **New frequencies**: arrays without a description are kept and must fit the new list
- **Errors**: a band without description and an array spaced in wavelengths in a layout without frequencies raise an
  error; the message names the node
- This is the member function `eval_antennas` of the C++ layout class; its in/out argument `layout` is split into
  `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [layout_csv_read](#layout_csv_read) (reads a layout, optionally with evaluated antennas)
- [set_antenna](#set_antenna) (sets the antennas of nodes)

---
<a id="export_rays_qrt"></a>
## export_rays_qrt
Write the rays of a layout with their link gain to a QRT file

Also in: [C++](quadriga_lib_api_cpp.md#export_rays_qrt) · [Python](quadriga_lib_api_python.md#export_rays_qrt)

- Stores the rays of the linked pairs with their link gain as CIRs: one origin per site, one destination per terminal
  and one CIR per row of the terminal
- Separates the ray generation from the antennas: [qrt_file_read](#qrt_file_read), [gbsm_rays_from_paths](#gbsm_rays_from_paths) and
  [gbsm_single_link_channel](#gbsm_single_link_channel) turn a CIR into channels for any arrays, orientations and positions, without the GBSM
  chain
- Positions in the file are local (see [QRT file format: precision](formats/qrt.md#qrt-precision)); add the returned shift
  to the positions read back

### Usage:
```matlab
shift = quadriga_lib.export_rays_qrt( fn, layout, rays );
```

### Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states): one frequency for the TR methods (`'quadriga'`: 1
  to 127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal
  at a site position; fields in [layout_csv_read](#layout_csv_read)
- **`rays`** — Rays of all links from [generate_rays](#generate_rays), possibly edited, one object with all frequencies per link; `{}`
  = generated for batches of sites with up to 256 links, which bounds the memory; cell array `[n_link, 1]` of 1x1
  structs; default: `{}`

### Outputs:
- **`shift`** — Shift of the local coordinates in x and y in [m]; `[2, 1]`

### Notes:
- **Origins and destinations**: an origin at each site, with the orientation of sector 0 (else of the first node of the
  site) and the site name; a destination per terminal with its name and one CIR per row, with the orientation of the
  first node of the terminal (its nodes share their positions)
- **Content of a CIR**: the rays of the link and of the segment that holds the row (scatterers added where missing), the
  link gain and the K-factor ratio of [get_channels_from_rays](#get_channels_from_rays) at the row, [ssf_paths](#ssf_paths) with `'drift'`. A CIR read
  back gives the channel of [get_channels_from_rays](#get_channels_from_rays) (`output = 'rays'`, `mobility = 'drift'`) at the row within
  float32 accuracy, outside the cross-fade ramps between segments
- **Reading back**: [gbsm_rays_from_paths](#gbsm_rays_from_paths) with the method of the links and the `n_interact` of the file as
  `path_type`, then [gbsm_single_link_channel](#gbsm_single_link_channel) with link gain 0 and mobility `'drift'`
- **Point drops with speed**: one CIR; the motion is applied at the evaluation, at the link gain of the drop
- **Pairs**: only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- **Shift**: the centroid of the site and CIR positions in x and y is subtracted; z is unchanged
- **Limits of the [QRT file format](formats/qrt.md)**: no tap index and no ground permittivity, so a CIR gives rays,
  not the taps of the TR, and moving a ground-reflection ray from a file raises an error
- The file is written to `fn + '.part'` and renamed at the end; a failed call leaves an existing file unchanged

### See also:
- [generate_rays](#generate_rays), [get_channels_from_rays](#get_channels_from_rays) (the rays and the link gain)
- [qrt_file_read](#qrt_file_read), [gbsm_rays_from_paths](#gbsm_rays_from_paths), [gbsm_single_link_channel](#gbsm_single_link_channel) (evaluation of the file)

---
<a id="generate_layout"></a>
## generate_layout
Network layout from a preset: BS sites and a drop of UTs

Also in: [C++](quadriga_lib_api_cpp.md#generate_layout) · [Python](quadriga_lib_api_python.md#generate_layout)

- Creates a standard evaluation layout with the values of its tables: the hexagonal and indoor layouts of 3GPP TR
  38.901, the test environments of ITU-R M.2412 and the road layouts of 3GPP TR 37.885
- The result has the sites `BS1`, `BS2`, ... with one Tx node per sector and the UTs `UT1`, `UT2`, ... as point drops;
  set the antennas next ([set_antenna](#set_antenna)), until then the nodes use the generic antenna
- The other inputs change single values of the preset: an entry of a tuple that is NaN or omitted keeps the value of the
  preset; an entry or option that the preset does not use raises an error

### Usage:
```matlab
layout = quadriga_lib.generate_layout( type, n_ut, center_freq, spacing, grid, area, bs, ut, indoor, ...
    options, per_cell, seed );
```

### Inputs:
- **`type`** — Preset name, not case sensitive:
  | `type` | Layout |
  | ------ | ------ |
  | `38.901_UMi`, `38.901_UMa`, `38.901_RMa`, `38.901_SMa` | Hexagonal, 19 sites of 3 sectors, ISD 200, 500, 1732, 1299 m, BS height 10, 25, 35, 35 m |
  | `38.901_InH_Open`, `38.901_InH_Mixed` | Hall of 120 × 50 m, 6 × 2 BSs at 20 m spacing, BS height 3 m |
  | `38.901_InF_SL`, `38.901_InF_DH` | Hall of 120 × 60 × 10 m, 6 × 3 BSs at 20 m spacing, BS height 1.5 m (SL), 8 m (DH) |
  | `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_HH` | Hall of 300 × 150 × 10 m, 6 × 3 BSs at 50 m spacing, BS height 1.5 m (DL), 8 m (SH, HH) |
  | `M.2412_InH`, `M.2412_DenseUrban`, `M.2412_Rural`, `M.2412_UrbanMacro` | Test environments of Report ITU-R M.2412-0 (see Notes) |
  | `37.885_Urban`, `37.885_Highway` | Roads and vehicles of TR 37.885, no BSs (see Notes) |
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 1e7 in total; 0 = no UTs; default: 0
- **`center_freq`** — Simulation frequencies (bands) in [Hz], finite, positive and strictly ascending, stored as
  `layout.center_freq`; default: `[]` (not set)
- **`spacing`** — `[d_x, d_y]` in [m], > 0: the ISD of a hexagonal preset (`d_y` not used), the spacing of the indoor
  grid or of the urban road grid; default: `[]`
- **`grid`** — `[n_x, n_y, n_sector]`, whole numbers: rings of a hexagonal preset (0 to 50, 0 = one site; `n_y` not
  used) or the size of the indoor grid (1 to 100); sectors per site (1 to 12), with the boresights 30° (indoor 0°) + k ·
  360° / `n_sector`; default: `[]`
- **`area`** — `[length, width, height]` of the hall or road area along x, y and z in [m], > 0; indoor and road presets,
  the height InF only, the width not for the highway; the BS grid must fit into it; default: `[]`
- **`bs`** — `[height, tilt, azimuth offset]` of the BSs in [m], [deg], [deg]: the mechanical tilt (positive down) and
  an offset added to the boresights; default: `[]`
- **`ut`** — `[height, min distance, speed, bank, tilt, heading]` of the UTs in [m], [m], [m/s], [deg], [deg], [deg]; a
  value replaces the rule of the preset for all UTs (building UTs get the floor from a given height; the speed of the
  lanes also sets the gaps); min distance (not for roads) and speed >= 0; default: `[]`
- **`indoor`** — `[indoor ratio, min floors, max floors, floor height]`: share of building UTs in [0, 1] (presets with
  O2I states); range of `N_fl`, whole numbers in [1, 1000] (UMi, UMa, Dense Urban); floor height in [m], > 0 (UMi, UMa,
  SMa, Dense Urban); default: `[]`
- **`options`** — Struct with one value per field: `p_residential` (SMa), the share of residential buildings in [0, 1];
  `config` (M.2412), `service` (`M.2412_UrbanMacro`), `h_micro` (`M.2412_DenseUrban` config 3) in [m]; `lane_width` in
  [m], `n_lane` (1 to 100), `p_truck` in [0, 1] (TR 37.885); other fields raise an error; default: `[]`
- **`per_cell`** — `true`: `n_ut` UTs in each cell; `false`: `n_ut` UTs uniform over all cells; no effect for the TR
  37.885 presets; default: `true`
- **`seed`** — Seed >= 0; -1 = system random device; default: -1

### Outputs:
- **`layout`** — Layout struct, fields in [layout_csv_read](#layout_csv_read)

### Example:
```matlab
layout = quadriga_lib.generate_layout( '38.901_UMa', 10, 3.5e9, [], [], [], [], [NaN NaN NaN 0 90], [], [], true, 42 );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2] ) );
```

### Notes:
- **Hexagonal presets** (2 rings, 3 sectors, wrap-around): site (q, r) with `max(abs(q), abs(r), abs(q + r)) <= n` for n
  rings is at `x = sqrt(3) · q · d / 2`, `y = (q / 2 + r) · d` with the ISD d, ordered by ring and then by azimuth in
  (-30°, 330°]. The wrap vectors `a1 = (sqrt(3) · n · d / 2, (3 · n / 2 + 1) · d)` and a2, a1 turned by 60°, make the
  cluster tile the plane. With the wrap-around, each link uses the copy of its site nearest to the first position of the
  terminal, kept for the whole track, so that the cells at the edge get the interference of a large network (TR 38.901
  Table 7.8-1; see the Notes of [layout_csv_read](#layout_csv_read))
- **`38.901_SMa`**: mechanical tilt 5° at ISD 1299 m and 2° at 1732 m (other ISDs: that of the nearer one); option
  `p_residential` (0.9)
- **Indoor presets**: centred at the origin, one sector, no wrap-around; the BSs form a lattice centred in the hall,
  ordered by y and then by x. The InF presets set the scenario parameters `DS_VS` (V/S of the hall) and `absTOA_Lmax`
  (its largest dimension) on the Tx rows, the TR 38.901 InH presets `absTOA_Lmax` (the largest of length, width and the
  office height 3 m). `38.901_InF_HH` has the layout of SH, since the TR has no calibration layout for it
- **Scenario**: the Tx rows of the TR 38.901 presets get the scenario family (the preset name without `38.901_`); the
  other presets set no scenario
- **ITU-R M.2412 presets** (Report M.2412-0, section 8.3 and Table 5): geometry and drop only, mechanical tilt 0; option
  `config` 1, 2, 3 for the configurations A, B, C (default 1); the minimum distance and O2I fields of the TR 38.901
  model that the Report maps the environment to. `d2d_in` is the TR 38.901 minimum of two uniforms (a library choice;
  the Report's model A at or below 6 GHz uses one uniform). The building-loss shares of Table 5 (20 % high, 80 % low
  loss) are set with `o2i_low = 0.8` of [init_link_states](#init_link_states), not by the preset
  - `M.2412_InH`: as `38.901_InH_Open` with UT height 1.5 m, without scenario and scenario parameters
  - `M.2412_DenseUrban`: as `38.901_UMa` with ISD 200 m; 80 % in buildings with floors, 20 % in cars at 30 km/h. Config
    3 adds 3 micro sites per macro TRxP, uniform in its cell, one TRxP each, `h_micro` high (option, 10 m), at least 35
    m from the macro sites and 20 m from each other; they form no cells (`cell` = 0): a macro TRxP cell receives `n_ut`
    UTs for itself and for each of its micro TRxPs, at least 10 m from the micro sites, over the whole macro layer
    (Table 5 b: within a cluster)
  - `M.2412_Rural`: as `38.901_RMa` with UT height 1.5 m; configs 1 and 2: 50 % in buildings, 50 % in cars at 120 km/h;
    config 3: ISD 6000 m, 40 % in buildings, 40 % outdoors, 20 % in cars at 30 km/h. With a given indoor ratio, config 3
    keeps one third of the other UTs in cars
  - `M.2412_UrbanMacro`: BS height 25 m, UT height 1.5 m; option `service` 1 (mMTC, default): ISD 500 m (config 1) or
    1732 m (config 2), 80 % in buildings, 3 km/h; 2 (URLLC, configs 1 and 2): ISD 500 m, 20 % in buildings, outdoors 30
    km/h
  - The M.2412 layouts carry no scenario, so a new drop with [drop_terminals](#drop_terminals) reproduces the distances of the preset
    only with `ut` entry 1 (e.g. 35 m) or with the scenario `UMa` on the macro rows and `UMi` on the micro rows
- **TR 37.885 presets** (Table A-1, clause 6.1.2, option A): roads and vehicles, no BSs; `n_ut` must be 0, the vehicles
  follow from the gaps
  - `37.885_Urban`: road grid of `spacing` 433 × 250 m in an `area` of 1299 × 750 m, centred, 2 lanes per direction of
    3.5 m, sidewalks 3 m, 60 km/h
  - `37.885_Highway`: `area` length 2000 m along x, 3 lanes per direction of 4 m, 140 km/h, wrap-around along the road
  - Lanes on the right side of their direction; options `lane_width`, `n_lane` (per direction) and `p_truck` (0);
    passenger vehicles 5 m long with the antenna at 1.6 m, trucks 13 m and 3 m. Along a lane, the gap from the front of
    a vehicle to the rear of the next one is max(2 m, Exp(mean = speed · 2 s)); vehicles head along their lane
- **UT drop**: point drops, `n_ut` per cell or in total, uniform over the cells (the sector part of a site hexagon, the
  hexagon without sectors, the lattice cell around an indoor BS):
  - Minimum 2D distance to the nearest image of the nearest site: UMi 10 m, UMa 35 m (indoor UTs: to d2D-out, TR 36.873
    Table 6-1), RMa and SMa 35 m, InH 0, InF 1 m. A rejected UT is drawn again with its O2I state, so the d2D-out rule
    lowers the building share slightly below its nominal value; a building UT of UMi and UMa needs d2D >= d2D-in also at
    `ut` min distance 0
  - Height 1.5 m (InH 1 m, InF-HH 3 m); in UMi and UMa buildings `3 · (n_fl - 1) + 1.5` m with `n_fl ~ U(1, N_fl)`,
    `N_fl ~ U(4, 8)`; SMa residential buildings 1 or 2 floors, commercial 1 to 5
  - Speed 3 km/h, in cars (RMa, SMa) 40 km/h; heading uniform in [0°, 360°), bank and tilt 0; direction uniform in
    azimuth, elevation 0. Calibration orientations by `ut`: TR Table 7.8-2 is tilt 90°, Table 7.8-2A configs B and C
    tilt 45°, config D tilt 0° and heading 0°
  - O2I states (`indoor`, `floor`, `d2d_in`) for UMi, UMa (80 % in buildings, the others outdoors), RMa (50 % in
    buildings, 50 % in cars) and SMa (80 % in buildings, 90 % of them residential, 20 % in cars), from spatially
    correlated fields at the UT positions (TR 38.901 clause 7.6.3.3); `d2d_in` up to 25 m (UMi, UMa, SMa commercial) or
    10 m (RMa, SMa residential)
- **Reproducibility**: the layout is a function of the seed and does not depend on the thread count; seed -1 draws a
  seed, which is stored in `layout.seed`

### See also:
- [drop_terminals](#drop_terminals) (the same drop for any set of Tx nodes)
- [generate_track](#generate_track) (tracks for the UTs)
- [set_antenna](#set_antenna) (antennas of the nodes)

---
<a id="generate_rays"></a>
## generate_rays
Rays of the propagation links of a layout (stage 2 of the channel pipeline)

Also in: [C++](quadriga_lib_api_cpp.md#generate_rays) · [Python](quadriga_lib_api_python.md#generate_rays)

- Runs the chain of the section "Geometry-based stochastic model" for the chosen links, from the link states of
  [init_link_states](#init_link_states): large-scale parameters, clusters, rays and, on request, the scatterers
- The rays have no antennas; [get_channels_from_rays](#get_channels_from_rays) applies them. Edit the rays in between to study a change of the
  propagation, or pass them to [attach_terminals](#attach_terminals) and [export_rays_qrt](#export_rays_qrt)
- The rays of a link do not depend on the other requested links

### Usage:
```matlab
rays = quadriga_lib.generate_rays( layout, i_link, scatterers );
```

### Inputs:
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states); fields in [layout_csv_read](#layout_csv_read); raises an error if a
  member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** — Propagation links, 1-based, without repetition; `[]` = all (none for links without entries); default:
  `[]`; `[n_out]`
- **`scatterers`** — Solve the first- and last-bounce scatterers, needed for the mobility `'drift'`; default: `false`

### Outputs:
- **`rays`** — Rays per link and frequency, a cell array `[n_out, 1]` of `1xN` struct arrays with the fields of
  [ssf_rays](#ssf_rays), `N = n_freq` with one frequency each for the TR methods, `N = 1` with all frequencies for `'quadriga'`

### Notes:
- **Chain**: [gbsm_init_phases](#gbsm_init_phases) and [gbsm_draw](#gbsm_draw) of the random sets, [path_loss](#path_loss) (for `rnd_he` and the SF standard
  deviation), [lsp_generate](#lsp_generate), [ssf_clusters](#ssf_clusters), [ssf_rays](#ssf_rays) and, with `scatterers`, [ssf_scatterers](#ssf_scatterers), once for the
  links of one site, image, conf and seed
- **Position**: the first row of the segment of the terminal and the site (moved by its wrap image) at the same time; a
  moving site is sampled at the time of the row, without times on the Rx track at the row of the same index. A site with
  more than one row or a speed uses dual mobility; the other sites are not affected
- **TR methods**: the SOS preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance,
  `angle_model = 'additive'`, `pol_model = '38.901'`, one set of clusters and rays per frequency (`'38.901_MF'`:
  clusters at the anchor frequency, rays per frequency)
- **`'quadriga'`**: bias removal over the positions of all links of the group, the legacy outdoor distance,
  `'rotation'`, `'quadriga'`, one ray set for all frequencies, always with scatterers
- **Edited link states**: the conf and the seed of a link follow from its scenario and the `los` and `indoor` of
  `layout.links`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged.
  Building UTs are on the floor of the Rx nodes (else from z) at the distance of the stored O2I inputs, the others on
  floor 0 at 0 m

### See also:
- [init_link_states](#init_link_states) (stage 1), [get_channels_from_rays](#get_channels_from_rays) (stage 3)
- [ssf_rays](#ssf_rays) (the ray data)

---
<a id="generate_track"></a>
## generate_track
Replace the rows of nodes by a generated track or a point drop, or append a node with one

Also in: [C++](quadriga_lib_api_cpp.md#generate_track) · [Python](quadriga_lib_api_python.md#generate_track)

- Gives nodes a straight, circular or street-like track, or makes them point drops with a speed and a direction, the
  short-term mobility of 3GPP TR 38.901
- All nodes with the name of a given node (the sectors of a site, the panels of a terminal) get the same track. It
  starts at the first position of the node and is horizontal at the height of the first row
- The antennas look along the motion, as [set_orientation](#set_orientation) turns them; [sample_track](#sample_track) gives the positions at
  snapshot times
- An empty `i_node` appends a new node at `start` and gives it the track (see Notes)

### Usage:
```matlab
nodes_out = quadriga_lib.generate_track( nodes_in, i_node, type, length, speed, direction, spacing, ...
    options, seed, name, start );
```

### Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [layout_csv_read](#layout_csv_read)
- **`i_node`** — Node indices, 1-based; `[]` = append a new node
- **`type`** — Type of the track, not case sensitive:
  - `point`: a point drop with `speed` and a horizontal `direction`; the rows after the first are removed; `length` is
    not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction
  - `circular`: a closed circle of circumference `length` (default 20π m, radius 10 m); the first and the last point are
    equal
  - `street`: the route of a car through a city grid of straight streets and crossings, of `length` (default 1000 m),
    see Notes
- **`length`** — Length in [m], finite and >= 0; 0 = default of the type; default: 0
- **`speed`** — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of the node if it
  is valid (> 0 for a track); without one, `point` gets 0 and the other types raise an error; default: NaN
- **`direction`** — Start direction (azimuth) in [rad], finite; one value for all nodes, or one per entry of `i_node`
  (nodes of one name need equal values); `[]` = uniform random in [0, 2π); default: `[]`
- **`spacing`** — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m (`street`);
  default: NaN
- **`options`** — Struct of options of `street`, one value per field: `street_length_min` = 50, `street_length_mu` =
  187, `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m],
  `turn_probability` = 0.5 (in [0, 1]); other fields, and these fields for the other types, raise an error; default:
  `[]`
- **`seed`** — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 raises an error; default: -1
- **`name`** — Name of the new node (`i_node = []` only); `''` = the next free name; default: `''`
- **`start`** — First position `[x, y, z]` of the new node in [m] (`i_node = []` only); `[]` = the origin; default: `[]`

### Outputs:
- **`nodes_out`** — The nodes with the generated rows

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( '', 'rx.csv' );
layout.rx = quadriga_lib.generate_track( layout.rx, 1, 'street', 500, 10, [], NaN, struct( 'turn_probability', 0.3 ), 42 );
layout.rx = quadriga_lib.generate_track( layout.rx, [], 'linear', 100, 1.5, 0, NaN, [], 42, 'walker', [10, 20, 1.5] );
```

### Notes:
- **Points**: `linear` has two points, or `ceil(length / spacing) + 1` equidistant points; `circular` 129 points, or
  `max(3, ceil(length / spacing)) + 1`; `street` is sampled with the point spacing `spacing`, and the chords between the
  points are slightly shorter than the route at the corners
- **Direction of a circle**: the angle of the start point seen from the centre of the circle. A positive value runs
  counter-clockwise, a negative one clockwise: 0 starts in the east of the circle and runs north, -π/2 starts in the
  south and runs west. A random direction is drawn in [0, 2π), so random circles run counter-clockwise
- **Street route** (as `qd_track.generate` of the NG library): straight streets of random length (normal distribution,
  drawn again below `street_length_min`) are separated by crossings. At a crossing the car turns with the probability
  `turn_probability` by about 90° (standard deviation 15°) to the left or right, cutting the corner along a path of
  length `2 · r · cos(φ / 2)` for the turn angle φ and the `curve_radius` r. The direction stays within 105° of the
  start direction, so that the car does not drive in circles
- **Values of the new rows**: `linear`, `circular` and `street` give tracks of equal `speed` without `direction`,
  oriented along the motion as [set_orientation](#set_orientation) with `force = false`; nodes with `relative = 1` keep the orientation
  of their first row on all rows. `point` keeps the orientation of the node. `scenario`, `indoor`, `floor`, `d2d_in` and
  the extra columns take the values of the first row; `time` is cleared
- **Reproducibility**: a seed >= 0 gives the same track on all platforms. Each name (all its nodes) has its own random
  stream, so the track does not depend on the other nodes or the order in `i_node`
- **Append mode**: with an empty `i_node`, a new node is appended with one row at `start` and the default values (sector
  0, the generic antenna, orientation 0, speed 0, no scenario), and gets the track; a track type therefore needs
  `speed`. It is the last node of `nodes_out`. The default name is the common prefix of the names before their number
  with the next number (`UT3` after `UT1`, `UT2`), else `N<n>`. `name` or `start` with a non-empty `i_node` raises an
  error
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

### See also:
- [sample_track](#sample_track) (snapshots of a track)
- [set_orientation](#set_orientation) (orientation from the motion)

---
<a id="get_channels_from_rays"></a>
## get_channels_from_rays
Channel coefficients of a layout from given rays (stage 3 of the channel pipeline)

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_from_rays) · [Python](quadriga_lib_api_python.md#get_channels_from_rays)

- Stage 3 alone, from given rays, e.g. for edited link states or rays; [get_channels_layout](#get_channels_layout) computes the channels
  from the layout in one call. Applies the antennas of the nodes, the motion along their tracks and the link gain to the
  rays of [generate_rays](#generate_rays), and returns one channel per Tx node, Rx node and frequency
- Only the channel links of the given propagation links and of the pairs in `layout.pairing` (empty: all) are computed;
  the sectors of a site share the rays of their propagation link
- Tracks with several segments are joined by the cross-fade of [crossfade_segments](#crossfade_segments), over a ramp centred on the
  first row of each new segment

### Usage:
```matlab
[ channels, index ] = quadriga_lib.get_channels_from_rays( layout, rays, i_link, times, output, mobility, ...
    oxygen, use_absolute_delays, overlap, link_gain );
```

### Inputs:
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states); fields in [layout_csv_read](#layout_csv_read); raises an error if a
  member that determines the links changed since then
- **`rays`** — Rays of the links from [generate_rays](#generate_rays), possibly edited, with the frequencies of the layout; cell array
  `[n_out, 1]` of `1xN` struct arrays, `N = 1` or `n_freq`
- **`i_link`** — The propagation links of `rays`, 1-based; `[]` = all links; default: `[]`; `[n_out]`
- **`times`** — Snapshot times in [s], finite, the same for all nodes; `[]` = one snapshot per row; default: `[]`;
  `[n_snap]`
- **`output`** — `'taps'` or `'rays'`, see [gbsm_single_link_channel](#gbsm_single_link_channel); default: `'taps'`
- **`mobility`** — `'phase'`, `'drift'` (needs rays with scatterers) or `'auto'` (phase for the TR methods, drift for
  `'quadriga'`); default: `'auto'`
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`
- **`use_absolute_delays`** — Include the LOS delay in all delays; default: `false`
- **`overlap`** — Length of the cross-fade ramp between consecutive segments as a part of the shorter one, in [0, 1],
  centred on the first row of the new segment; 0 = a switch at that row, see [crossfade_segments](#crossfade_segments); default: 0.5
- **`link_gain`** — false: the link gain is left out (0 dB), so the coefficients have no path loss, O2I loss and
  shadow fading; the K-factor ratio stays; default: true

### Outputs:
- **`channels`** — Channels, struct array, fields in [hdf5_read_channel](#hdf5_read_channel); `[1, n_channel]`
- **`index`** — Tx node, Rx node and frequency index of each channel, 1-based; uint64; `[3, n_channel]`

### Notes:
- **Order and names**: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the
  sector where all Tx nodes (or all Rx nodes) have sector 0
- **Snapshots** of both ends from [sample_track](#sample_track) over `times`; a node with one snapshot is repeated to the count of
  the other; two tracks without `times` need equal row counts
- **Link gain** per snapshot, `G = -PL - L_O2I + 10 · log10(SF)` in dB: the path loss PL of [path_loss](#path_loss) with the
  random environment height `rnd_he` held at the start of the segment, the shadow fading SF and the K-factor ratio of
  [lsp_generate](#lsp_generate) from [gbsm_draw](#gbsm_draw) of the `'lsf'` set at the snapshot, and the O2I loss of [o2i_loss](#o2i_loss) for building
  and car UTs with the model of the link state (the deviation of a building UT at the snapshot, that of a car UT at the
  first row of the terminal for its whole track, TR 38.901 eq. 7.4-4). A link whose ends do not move is evaluated once
- **Antennas**: [gbsm_single_link_channel](#gbsm_single_link_channel) with the node antennas at the frequency (entry `i_freq` of the `1xN` struct
  array of the antenna in `layout.antenna`, or its one entry for all bands) and the Tx positions moved by the wrap image
  of the link. Nodes without an antenna and antennas that are not evaluated use the generic antenna; an antenna spaced
  in wavelengths that is not evaluated raises an error
- **Frequencies**: rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency
  `i_freq` (`'quadriga'`)
- **Segments**: the snapshots of a segment are those whose row (see [sample_track](#sample_track)) is in it; each segment also covers
  the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row. The
  selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing
  `times`
- **Relative delays**: in `'phase'` the paths of each segment relative to its own LOS, in `'drift'` relative to the LOS
  at the snapshot
- The C++ output `index` is the second output

### See also:
- [generate_rays](#generate_rays) (stage 2)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="get_channels_layout"></a>
## get_channels_layout
Channel coefficients of a layout in one call

Also in: [C++](quadriga_lib_api_cpp.md#get_channels_layout) · [Python](quadriga_lib_api_python.md#get_channels_layout)

- Computes the channels in one call from the layout: one channel per Tx node, Rx node and frequency, for the pairs in
  `layout.pairing` (empty: all pairs); [get_channels_from_rays](#get_channels_from_rays) is the third stage alone, from given rays
- Supports tracks (with times or speeds), point drops that move with their speed, tracks of several segments, joined by
  the cross-fade of [crossfade_segments](#crossfade_segments), and both output forms, `'taps'` and `'rays'`
- Runs the stages [init_link_states](#init_link_states), [generate_rays](#generate_rays) and [get_channels_from_rays](#get_channels_from_rays) with their default settings;
  the results are those of the stages called one by one
- Without link states in `layout.links`, draws them for this call only and leaves the layout unchanged; with link
  states, uses them, also when they were edited

### Usage:
```matlab
[ channels, index ] = quadriga_lib.get_channels_layout( layout, times, output, method, mobility, ...
    overlap, seed, config_dirs, link_gain );
```

### Inputs:
- **`layout`** — Layout struct with `center_freq`; fields in [layout_csv_read](#layout_csv_read); antennas as evaluated in
  `layout.antenna`, nodes without one use the generic antenna (see [get_channels_from_rays](#get_channels_from_rays))
- **`times`** — Snapshot times in [s]; `[]` = one snapshot per row; default: `[]`; `[n_snap]`
- **`output`** — `'taps'` or `'rays'`; default: `'taps'`
- **`method`** — `'38.901'`, `'38.901_MF'` or `'quadriga'`; `''` = the method of filled links, else `'38.901'`; another
  method than that of filled links raises an error; default: `''`
- **`mobility`** — `'phase'`, `'drift'` or `'auto'`; default: `'auto'`
- **`overlap`** — Length of the cross-fade ramp between consecutive segments as a part of the shorter one, in [0, 1],
  centred on the first row of the new segment; 0 = a switch at that row, see [crossfade_segments](#crossfade_segments); default: 0.5
- **`seed`** — Master seed for empty links, as in [init_link_states](#init_link_states); with filled links only -1; default: -1
- **`config_dirs`** — Folders of user conf files for empty links, searched before the shipped folder; string or cell
  array of strings; default: `{}`
- **`link_gain`** — false: without the link gain, as in [get_channels_from_rays](#get_channels_from_rays); default: true

### Outputs:
- **`channels`** — Channels, Tx node fastest, then Rx node, then frequency; struct array, fields in
  [hdf5_read_channel](#hdf5_read_channel); `[1, n_channel]`
- **`index`** — Tx node, Rx node and frequency index of each channel, 1-based; uint64; `[3, n_channel]`

### Example:
```matlab
layout = quadriga_lib.generate_layout( '38.901_UMa', 1, 3.5e9 );
channels = quadriga_lib.get_channels_layout( layout );
```

### Notes:
- **Memory**: the links are processed in batches of at most 256 propagation links (the segments of a track in one
  batch), so that the rays of one batch bound the memory; the results do not depend on the batches
- **Link states**: `layout.links` holds none when `layout.links.method` is empty, e.g. after `layout.links = []`. With
  link states, `seed` and `config_dirs` are not used. A layout without linked pairs gives an empty result
- **Scatterers**: the mobility `'drift'` (also `'auto'` with `'quadriga'`) solves the scatterers
- The C++ output `index` is the second output; the wrapper adds the folder of the shipped conf files to `config_dirs`

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays), [get_channels_from_rays](#get_channels_from_rays) (the three stages)
- [generate_layout](#generate_layout) (layouts of the TR scenarios)

---
<a id="init_link_states"></a>
## init_link_states
Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

Also in: [C++](quadriga_lib_api_cpp.md#init_link_states) · [Python](quadriga_lib_api_python.md#init_link_states)

- Draws for each propagation link (one site, one terminal, one segment) the state that the later stages need: the wrap
  image, the scenario and conf file, the LOS and indoor state and the seeds, stored in `layout_out.links` (fields in
  [layout_csv_read](#layout_csv_read))
- The states follow from the seeds and the positions, so adding a terminal does not change the states of the others
- `layout.pairing` does not limit the links; it selects at the output of the channel functions

### Usage:
```matlab
layout_out = quadriga_lib.init_link_states( layout_in, method, o2i_low, pol_variability, anchor_freq, ...
    sos_preset, seed, config_dirs );
```

### Inputs:
- **`layout_in`** — Layout struct, valid, with `center_freq`; fields in [layout_csv_read](#layout_csv_read)
- **`method`** — `'38.901'`, `'38.901_MF'` (TR 38.901 clause 7.6.5.1) or `'quadriga'`, see [generate_rays](#generate_rays); default:
  `'38.901'`
- **`o2i_low`** — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; the other families have
  one building model; `[]` = the share of TR 38.901 Table 7.8-1 (0.5); default: `[]`
- **`pol_variability`** — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only; default: `false`
- **`anchor_freq`** — Anchor frequency of `'38.901_MF'` in [Hz]; default: 2e9
- **`sos_preset`** — SOS preset of the GBSM random sets, see [sos_preset](#sos_preset); `''` = `Exp256` (TR methods), `Comb256`
  (`'quadriga'`); default: `''`
- **`seed`** — Master seed, >= -1; -1 = `layout_in.seed`, if that is -1 one draw of the system random device; default:
  -1
- **`config_dirs`** — Folders of user conf files, searched before the shipped folder; string or cell array of strings;
  default: `{}`

### Outputs:
- **`layout_out`** — The layout with `links` and `seed` (the seed used) set; the rest is unchanged

### Notes:
- **Links**: one per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major
  (all sites of terminal 1 first). Co-located pairs have no link; a later segment that starts at the site raises an
  error. A layout without any pair farther apart gives links without entries, and the later stages return empty results
- **Scenario** of a link: the value of the terminal, else that of the site. A scenario family (e.g. `UMa`) draws the LOS
  and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes
  them, and any other name is a user conf file, used in either state without O2I loss. A conf that several families list
  (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found
  with [conf_file_list](#conf_file_list), the folders of `config_dirs` first
- **Wrap image**: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the
  terminal, ties to the lowest index in the order 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2); all distances of the
  link use this virtual site position
- **O2I state** per terminal, from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3):
  building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A
  (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the
  O2I model and `d2d_in` as in [o2i_loss](#o2i_loss). The values `indoor`, `floor` and `d2d_in` of the Rx nodes fix them; an
  exact conf name that contradicts a fixed `indoor` raises an error
- **One O2I state per terminal** (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal
  linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a
  site fixes it for the links to all sites; InH and InF have none
- **LOS state** per link: a uniform field per site, image and family at the first position of the segment, at the
  `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [los_probability](#los_probability) at
  `d2d_out = max(d2d - d2d_in, 0)`; InF-HH is always LOS
- **Segments**: on the Rx nodes, a row whose `scenario` or `indoor` differs from the row before it starts a segment (an
  empty `scenario` differs from an explicit one). A segment has its own propagation links (order: terminal, site,
  segment), LOS state, indoor state, conf and seed, drawn at its first row, with the distances to a moving site at the
  time of the segment start. The wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are
  those of the first row of the track; a fixed `d2d_in` must be equal at all segment starts. A segment keeps its LOS
  state along its length, so long tracks need explicit segments
- **Random values**: [sos_rand](#sos_rand) with the preset `Exp256`; each field has its own seed, hashed from the site seed (else
  the layout seed) and a key of its purpose, site, image and family or conf. `gbsm_seed` is the seed of
  [gbsm_init_phases](#gbsm_init_phases)
- **Frequencies**: the TR methods draw the cluster and ray sets of frequency `k` with the index `k` into
  `layout.center_freq`, so a frequency inserted below another changes the rays of the higher bands
- **Layout hash**: `links.layout_hash` covers the positions, names, sectors, frequencies, site seeds, scenarios, the O2I
  columns, clutter, scenario overrides, wrap vectors and whether a site moves. The later stages raise an error when it
  no longer matches the layout; `layout.links = []` discards the links. Orientations, antennas and `pairing` may change
  without new links
- **Forced states**: editing `los` or `indoor` of `layout.links` before [generate_rays](#generate_rays) forces that state; the conf
  and its seed follow from it. `floor_id` and `d2d_in` stay those of the drawn state; the later stages derive the values
  of the forced state from `indoor`
- **Restoring**: the links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after
  another realisation
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged; the
  wrapper adds the folder of the shipped conf files to `config_dirs`

### See also:
- [generate_rays](#generate_rays), [get_channels_from_rays](#get_channels_from_rays) (stages 2 and 3)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="layout_csv_read"></a>
## layout_csv_read
Read the Tx and Rx CSV files of a layout

Also in: [C++](quadriga_lib_api_cpp.md#layout_csv_read) · [Python](quadriga_lib_api_python.md#layout.csv_read)

- Reads the Tx file and the Rx file of a layout in the [Layout CSV format](formats/layout_csv.md) and returns the
  layout struct, with angles in rad
- Generates the antennas of the nodes from their antenna columns, unless `eval_antennas` is `false`
- Keeps unknown columns and returns their names, so that a typo in a column name is visible

### Usage:
```matlab
[ layout, extra_tx, extra_rx ] = quadriga_lib.layout_csv_read( fn_tx, fn_rx, center_freq, eval_antennas );
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = no Tx nodes or no Rx nodes
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `[]` = from the files; default:
  `[]`; `[n_freq]`
- **`eval_antennas`** — `true`: generate the antennas; `false`: `antenna` stays empty, `antenna_args` and `i_antenna`
  are kept, and the nodes use the generic antenna until [eval_antennas](#eval_antennas) is called; default: `true`

### Outputs:
- **`layout`** — The layout, a 1x1 struct, which all functions of this section take and return; `name` and `links` are
  empty. Angles in rad; `offset`, `pairing`, `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model` are 1-based. A field
  that is missing or empty in an input struct takes the value of an empty member; unknown fields are rejected.

  Fields of `layout`:
  - `name` — Name of the layout; not stored in the CSV files; String
  - `tx`, `rx` — Tx nodes (BS, AP) and Rx nodes (UT, STA), fields below; Struct
  - `links` — Link states of the current realisation, fields below; `[]` = none; Struct
  - `seed` — Seed of the layout, -1 = not set; int64, scalar
  - `center_freq` — Simulation frequencies (bands) in [Hz], strictly ascending; `[n_freq, 1]`
  - `antenna_cols` — Names of the antenna columns of the lines in `antenna_args`; Cell of strings
  - `antenna_args` — Antenna descriptions, one line for all bands or one per band (see [eval_antennas](#eval_antennas)); Cell of cells
    of strings
  - `antenna` — Evaluated arrays, entry `k` a 1xN struct array as in [generate_antenna](#generate_antenna), N = 0 (not evaluated), 1 (all
    bands) or `n_freq`; Cell `[n_antenna, 1]`
  - `wrap` — Lattice vectors of the wrap-around in [m], one per column (see Notes); two vectors form a reduced basis,
    `0 <= a1·a2 <= min(abs(a1)^2, abs(a2)^2) / 2`, so that ±a1, ±a2 and ±(a1 - a2) are the nearest images; empty = none;
    `[2, 1]` or `[2, 2]`
  - `pairing` — Site index and terminal index of the linked pairs; empty = all pairs; uint64, `[2, n_pair]`

  Fields of the node structs `layout.tx` and `layout.rx`:
  - `name` — Site or terminal name, 1 to 255 bytes, without comma, semicolon, quote and line break, and without leading
    or trailing blank; Cell `[n_node, 1]`
  - `sector` — Sector number; (`name`, `sector`) is unique; uint32, `[n_node, 1]`
  - `i_antenna` — Antenna: index into the antenna lists of the layout, 0 = generic antenna; uint32, `[n_node, 1]`
  - `offset` — First row of each node; the rows of node `i` are `offset(i)` to `offset(i+1) - 1` (the last node to the
    end); uint64, `[n_node, 1]`
  - `pos` — Positions in [m]; `[3, n_row]`
  - `orientation` — Bank, tilt, heading in [rad]; `[3, n_row]`
  - `direction` — Azimuth, elevation of travel in [rad]; NaN azimuth = along the orientation; NaN on all rows of a
    track; empty = all NaN; `[2, n_row]`
  - `speed` — Speed in [m/s] from this row on; empty = 0; `[n_row, 1]`
  - `time` — Time in [s]; NaN on all rows of a node = not given; empty = none; `[n_row, 1]`
  - `scenario` — Scenario family or conf name, `''` = none; empty = none; Cell `[n_row, 1]`
  - `indoor` — -1 = draw, 0 outdoor, 1 building, 2 car, 3 residential building; empty = all -1; int32, `[n_row, 1]`
  - `floor` — Floor number from 1, -1 = from z; empty = all -1; int32, `[n_row, 1]`
  - `d2d_in` — Indoor 2D distance in [m], NaN = draw; empty = all NaN; `[n_row, 1]`
  - `clutter` — `d_clutter`, `r_clutter` and `h_clutter`, NaN = model default; empty = all NaN; `[3, n_node]`
  - `scenpar_override` — Overrides of scenario parameters, one field per key, NaN = none; Struct of `[n_node, 1]`
  - `seed` — Seed of the site, -1 = derived from the seed of the layout; empty = all -1; int64, `[n_node, 1]`
  - `relative` — 1 = orientation relative to the direction of motion; empty = all 0; uint32, `[n_node, 1]`
  - `cell` — Tx nodes only: 0 = the site has no drop cells of its own ([drop_terminals](#drop_terminals)); empty = all 1; uint32,
    `[n_node, 1]`
  - `extra_names`, `extra` — Names of unknown CSV columns and their cells as text; Cell `[n_extra, 1]` of cells
    `[n_row, 1]`

  Fields of `layout.links`:
  - `i_tx`, `i_rx`, `i_segment` — Site, terminal and segment of each link; uint32, `[n_link, 1]`
  - `image` — Wrap image of the site, 0 = the real site; uint32, `[n_link, 1]`
  - `i_conf`, `conf_names` — Index into `conf_names`, from scenario, LOS and indoor state; resolved conf names; uint32,
    `[n_link, 1]`; cell
  - `los`, `indoor`, `floor_id`, `d2d_in` — 1 = line of sight; indoor state at the start of the segment, as in the
    nodes; floor number - 1 (0 outdoors) and indoor 2D distance in [m] of the drawn state; `[n_link, 1]`
  - `o2i_rnd`, `o2i_model` — Random inputs of [o2i_loss](#o2i_loss); building O2I model, index into the list of the family, for
    every terminal of a family with O2I, also outdoors; `[n_terminal, 4]`; uint32, `[n_terminal, 1]`
  - `gbsm_seed`, `seed` — Seed of the random phases, equal for one site, image and conf; master seed, the base of the
    random values of all stages, -1 = not set; int64, `[n_link, 1]`; scalar
  - `method`, `sos_preset`, `anchor_freq`, `pol_variability` — Method and SOS preset as used (an empty `method` means
    not filled); anchor frequency of the multi-frequency model in [Hz]; polarization power variability; Strings, scalar,
    logical
  - `conf_files`, `layout_hash` — Resolved paths of the conf files, empty = not resolved; hash of the layout fields that
    determine the links (see [init_link_states](#init_link_states)); Cell; uint64
- **`extra_tx`**, **`extra_rx`** — Names of the unknown columns of the Tx and Rx file, in file order; cell array of
  strings; `[n_extra, 1]`

### Notes:
- **Errors**: a violation of the format and a layout that is not valid raise an error with the file name, the line and
  the column (see [Reading](formats/layout_csv.md#layout-csv-reading)); with `eval_antennas`, so do a line that
  [generate_antenna](#generate_antenna) rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without
  frequencies
- **Validity**: a valid layout, which the functions of this section check, has node structs with fitting sizes, a
  fitting `offset`, finite positions and orientations, valid names and values in range, and follows the rules of the
  [Layout CSV files](formats/layout_csv.md#layout-csv-columns) for `time`, `speed`, `direction`, `floor` and `d2d_in` and for the
  values that the nodes of one site or terminal share ([Nodes and rows](formats/layout_csv.md#layout-csv-rows)); the antenna
  lists (a node with a band that is not set is reported), the frequencies, `wrap`, `pairing` and `links` are checked too
- **Wrap-around**: a cluster of a few rings of sites is finite, so the cells at its edge lack the interferers that
  surround them in a large network. With `wrap`, the cluster is repeated periodically by the lattice vectors, and each
  link uses the copy (image) of its site nearest to the first position of the terminal, kept for the whole track, which
  the channel generation treats as a site of its own (TR 38.901 Table 7.8-1: geographical distance based wrapping;
  clause 7.5, step 1). Every terminal then has neighbours on all sides, as in the middle of a network. One vector
  repeats the cluster along one direction, e.g. along a road
- **Antennas**: equal descriptions share one entry of the antenna lists, Tx nodes first; a relative `qdant_file` is
  stored as a normalised absolute path
- **Memory**: large arrays on the default 1° grid need much memory ([Antennas](formats/layout_csv.md#layout-csv-antennas)); give
  a `res` column or read with `eval_antennas = false`

### See also:
- [layout_csv_write](#layout_csv_write) (stores a layout)
- [generate_antenna](#generate_antenna) (the antenna types and arguments of the antenna columns)

---
<a id="layout_csv_write"></a>
## layout_csv_write
Write a layout to the Tx and Rx CSV files

Also in: [C++](quadriga_lib_api_cpp.md#layout_csv_write) · [Python](quadriga_lib_api_python.md#layout.csv_write)

- Writes the Tx nodes to the Tx file and the Rx nodes to the Rx file in the
  [Layout CSV format](formats/layout_csv.md); reading the files back gives the same layout within the written digits
  (9 significant digits, positions 1 µm, times 1 ns), except for `name` and `links`, which are not stored
- Antennas without a description, e.g. a measured array set with [set_antenna_array](#set_antenna_array), are written to QDANT files next
  to the CSV file
- A failed call leaves the existing files unchanged

### Usage:
```matlab
quadriga_lib.layout_csv_write( fn_tx, fn_rx, layout, compact );
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = not written; different files (also if the paths are spelled
  differently)
- **`layout`** — Layout struct, must be valid; fields in [layout_csv_read](#layout_csv_read)
- **`compact`** — `true`: cells that the reader fills in are left empty; `false`: every cell is written; default: `true`

### Notes:
- **Rules of the writer**: the columns written, the number format, the compact form, the names of the QDANT files and
  what cannot be written are described in [Writing](formats/layout_csv.md#layout-csv-writing)
- **QDANT files**: one file `<stem>_<k>.qdant` per antenna without description, with one entry per band. Files of this
  name pattern next to a written CSV file that the new files do not refer to are deleted. The arrays must be evaluated:
  a layout whose `antenna` does not hold them raises an error. An antenna set from a QDANT file ([set_antenna](#set_antenna)) is
  written as a reference to its file, not exported

### See also:
- [layout_csv_read](#layout_csv_read) (reads a layout)

---
<a id="sample_track"></a>
## sample_track
Positions and orientations of a node at snapshot times

Also in: [C++](quadriga_lib_api_cpp.md#sample_track) · [Python](quadriga_lib_api_python.md#sample_track)

- Returns the position and the orientation of one node at each snapshot time, for point drops and tracks;
  [get_channels_from_rays](#get_channels_from_rays) samples both ends of each link this way
- Without `times`, each row of the node is one snapshot
- The orientation includes the rotation of mounted antennas (`relative = 1`), so it is the orientation in the GCS

### Usage:
```matlab
[ pos, orientation, i_row ] = quadriga_lib.sample_track( nodes, i_node, times );
```

### Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [layout_csv_read](#layout_csv_read)
- **`i_node`** — Node index, 1-based
- **`times`** — Snapshot times in [s], finite; `[]` = one snapshot per row; `[n_snap]`

### Outputs:
- **`pos`** — Positions in [m]; `[3, n_snap]`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation;
  `[3, n_snap]`
- **`i_row`** — Row at or before each snapshot, 1-based, index into the arrays of `nodes` (not into the rows of the
  node); uint64; `[n_snap, 1]`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
[ pos, ori ] = quadriga_lib.sample_track( layout.rx, 1, 0:0.1:1 );
```

### Notes:
- **Point drop** (one row): `p(t) = p0 + v · t · d` with the speed v of the row and the unit vector d of its direction
  (azimuth, elevation; an elevation of NaN is 0). Without a direction (NaN azimuth),
  `d = (cos(β) · cos(α), cos(β) · sin(α), -sin(β))` with the tilt β and the heading α of the orientation; the bank has
  no effect. The orientation is constant
- **Track** (several rows): the time of row i is `time`, else it follows from the speed v(i) of the row, `t(1) = 0`,
  `t(i+1) = t(i) + abs(p(i+1) - p(i)) / v(i)`; a step of zero length with speed 0 takes no time. The position is
  interpolated linearly in time, bank and heading along the shorter way and the tilt linearly; before the first and
  after the last time, the first and the last row hold. If several rows have the same time, the later row holds at that
  time. The times in `times` need not increase. A track with neither `time` nor a speed raises an error if `times` is
  given; a moving step with speed 0 raises an error
- **Mounted antennas** (`relative = 1`): the orientation is relative to the direction of motion, and the output is
  `R_motion · R_orientation`, with bank 0 in `R_motion`. For a point drop, the motion is its direction, which must be
  given. For a track, it is the direction of the step from the row at or before the snapshot to the next row (the last
  step repeated at the end); a step of zero length takes the previous direction, at the start the next one that exists.
  The stored orientation is interpolated first, then rotated by the motion
- **Checks**: fields of `nodes` with sizes that do not fit, and a `direction` on a track, raise an error; the other
  rules of a valid layout ([layout_csv_read](#layout_csv_read)) are not checked

### See also:
- [generate_track](#generate_track) (creates tracks)
- [set_orientation](#set_orientation) (orientation from the motion)

---
<a id="set_antenna"></a>
## set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna, a QDANT file, or by index

Also in: [C++](quadriga_lib_api_cpp.md#set_antenna) · [Python](quadriga_lib_api_python.md#set_antenna)

- Sets the antenna of Tx nodes or Rx nodes from the arguments of [generate_antenna](#generate_antenna) or from an entry of a QDANT file,
  or, with a numeric second input, to antennas that the layout already has, by their index; [set_antenna_array](#set_antenna_array) sets
  an array, e.g. a measured pattern read with [qdant_read](#qdant_read)
- Without `center_freq`, the antenna applies to all bands; with `center_freq`, it is the antenna of that band
- Stores an antenna description, which [layout_csv_write](#layout_csv_write) writes as antenna columns (a QDANT file as a reference,
  without a copy)

### Usage:
```matlab
layout_out = quadriga_lib.set_antenna( layout_in, type, args, rx, i_node, center_freq, evaluate );
layout_out = quadriga_lib.set_antenna( layout_in, i_antenna, rx, i_node, center_freq );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read); its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [generate_antenna](#generate_antenna), e.g. `'3GPP'`, `'dipole'`; a QDANT file, i.e. a name ending in
  `.qdant` (any case), absolute or relative to the current folder; or `'none'` (no antenna, for all bands, without
  `args`)
- **`args`** — Struct with the arguments of [generate_antenna](#generate_antenna) as fields, with the meaning given there: `res`,
  `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers,
  vectors or matrices in the shapes given there (a `beams` vector is one beam), `separate_beams` and `apply_weights` as
  0 or 1; a field that the type does not use raises an error; `pattern` is not available (an array built with it is set
  with [set_antenna_array](#set_antenna_array)); the field `center_freq` is the same as the argument; e.g.
  `struct( 'panel', [4 4 1 2], 'pol', 3 )`. For a QDANT file only `id`, the 1-based entry of the file (default 1).
  Default: `[]` (none)
- **`i_antenna`** — Antenna entries of the layout, 1-based as `i_antenna` of the nodes, 0 = generic antenna; one value
  for all chosen nodes or one per node in the order of `i_node`; `[]` = 0
- **`rx`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; with `i_antenna`, a band of the layout,
  which the nodes take from the given entries; default: NaN
- **`evaluate`** — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none, the nodes of
  new antennas use the generic antenna until [eval_antennas](#eval_antennas) is called; default: `true`

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2], 'pol', 3 ), false, [], 3.5e9 );
layout = quadriga_lib.set_antenna( layout, 'measured/handset.qdant', struct( 'id', 2 ), true );
layout = quadriga_lib.set_antenna( layout, 2, true, [ 1, 2 ] );
```

### Notes:
- **Bands**: with `center_freq`, only that band of the antenna of the nodes is set. A frequency that is not in
  `layout_in.center_freq` is added; the other bands of the nodes keep their antenna, and nodes that shared the old entry
  keep it. Until all bands of a node are set, the layout is not valid, and the functions that check it name the node
- **Bands set one by one**: such a node keeps one line per band, also when the lines are equal, so a band added later is
  reported as not set for it. In a layout with one band, the line of a node set for that band also covers bands added
  later
- **Antenna lists**: after the call, equal descriptions share one entry, entries are numbered in order of first use (Tx
  nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- **Deferred evaluation**: an array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in
  a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- **QDANT file**: stored as a normalised absolute path; [layout_csv_write](#layout_csv_write) writes it relative to the CSV file. The
  array is used as stored, not rescaled to the frequency of a band. A missing or unreadable file raises an error;
  without evaluation, an entry that the file does not have raises an error when the antennas are evaluated
- **Index form**: with `center_freq`, that band of the antenna of each node becomes the band of its entry, copied as in
  the other forms; the generic antenna (0) applies to all bands only. A node listed twice in `i_node` with different
  entries raises an error
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [generate_antenna](#generate_antenna) (the meaning of the arguments)
- [qdant_read](#qdant_read) (reads the entries of a QDANT file)
- [eval_antennas](#eval_antennas) (generates the antennas)
- [layout_csv_write](#layout_csv_write) (writes the descriptions and exports the arrays)
- [set_antenna_array](#set_antenna_array) (sets an array)

---
<a id="set_antenna_array"></a>
## set_antenna_array
Set the antenna of layout nodes from an evaluated array

Also in: [Python](quadriga_lib_api_python.md#set_antenna_array) · Not in: C++

- Sets the antenna of Tx nodes or Rx nodes from an array, e.g. a measured pattern read with [qdant_read](#qdant_read)
- Without `center_freq`, the array applies to all bands; with `center_freq`, it is the antenna of that band
- Stores the array, which [layout_csv_write](#layout_csv_write) exports to a QDANT file

### Usage:
```matlab
layout_out = quadriga_lib.set_antenna_array( layout_in, array, rx, i_node, center_freq );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read); its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, 1x1 struct, fields as in [generate_antenna](#generate_antenna)
- **`rx`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna_array( layout, quadriga_lib.qdant_read( 'handset.qdant' ), true );
```

### Notes:
- **Bands and antenna lists**: as in [set_antenna](#set_antenna)
- This is the second form of the C++ function `set_antenna`, which takes an array; it has its own name in MATLAB,
  because MATLAB does not overload by type. The C++ in/out argument `layout` is split into `layout_in` / `layout_out`,
  so that the input stays unchanged

### See also:
- [set_antenna](#set_antenna) (sets the antenna from the arguments of generate_antenna)
- [qdant_read](#qdant_read) (reads an array from a QDANT file)
- [layout_csv_write](#layout_csv_write) (exports the array)

---
<a id="set_orientation"></a>
## set_orientation
Set the orientation of nodes from their motion, or to a given value

Also in: [C++](quadriga_lib_api_cpp.md#set_orientation) · [Python](quadriga_lib_api_python.md#set_orientation)

- Turns the antennas of tracks along their motion, e.g. for a UT held in the direction of travel, or sets all rows of
  the nodes to a fixed orientation
- `force = false`: tracks get bank 0 and the tilt and heading of the steps between their rows, plus `orientation` as an
  offset; point drops keep their orientation plus the offset
- `force = true`: every row of the nodes gets exactly `orientation`

### Usage:
```matlab
nodes_out = quadriga_lib.set_orientation( nodes_in, orientation, i_node, force );
```

### Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [layout_csv_read](#layout_csv_read)
- **`orientation`** — Bank, tilt, heading in [rad], 3 finite values; `[]` = 0; default: `[]`
- **`i_node`** — Node indices, 1-based; `[]` = all nodes; default: `[]`
- **`force`** — `false`: orientation from the motion of tracks, plus `orientation` as offset; `true`: `orientation` on
  all rows; default: `false`

### Outputs:
- **`nodes_out`** — The nodes with the new orientations

### Example:
```matlab
layout.rx = quadriga_lib.set_orientation( layout.rx );                    % The antennas look along the motion
layout.rx = quadriga_lib.set_orientation( layout.rx, [0 0 0.5] );         % The same, turned by 0.5 rad
```

### Notes:
- **Tracks** (`force = false`): forward differences between the rows, the last step repeated, or for a closed track the
  step to the first row, as `qd_track.calc_orientation` of the NG library does; the angles are wrapped to [-π, π]. A
  step of zero length keeps the previous orientation, at the start the next one that exists. A vertical step has the
  tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising
  track has a negative tilt; a track that does not move keeps its orientation (plus the offset)
- **Point drops** (`force = false`): without an offset, the orientation is not touched; random orientations are not
  generated here
- **Mounted antennas**: `force = false` leaves nodes with `relative = 1` unchanged, offset included, since their
  orientation is relative to the motion, which [sample_track](#sample_track) applies at each snapshot; `force = true` sets them too
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

### See also:
- [generate_track](#generate_track) (creates tracks with this orientation)
- [sample_track](#sample_track) (applies the motion to mounted antennas)

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
Calculate reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

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
val_uni = quadriga_lib.sos_val(spatial_freq, phase, pos, [], [], [], true);

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
  The value then changes smoothly when either device moves, keeps the correlation of one device while the
  other stays, and does not change when the two ends swap.

```matlab
% Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
pos_tx = pos;
pos_rx = rand(1000, 3) * 100;
val_link = quadriga_lib.sos_val(spatial_freq, phase, pos_tx, pos_rx);
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
- `acf_est_dual`: the same for two devices, reciprocal as in [sos_val](#sos_val). The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is [sos_val](#sos_val) with both
  positions. Two links form a pair if the distances between their first devices and between their second devices
  fall into the same bin; for the two-device generator of [sos_val](#sos_val) the estimate approximates the square of
  the ACF
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

### Usage:
```matlab
[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( spatial_freq, phase, bins, n_pos, seed );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, 3]`
- **`phase`** — Phases of the generator; `[n_coeff]`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`seed`** — Seed for the positions; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Outputs:
- **`acf_est`** — Estimated ACF; `[n_bin, 1]`
- **`dist_est`** — Mean distance of the pairs in each bin; `[n_bin, 1]`
- **`acf_est_dual`** — Estimated ACF for two devices; `[n_bin, 1]`

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
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`, with or without a
  second device; the phases of a generator therefore do not depend on the number of generators
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices (`pos_b` given): the two-device generator of [sos_val](#sos_val); with the same decorrelation distance
  (`dist_decorr_b` omitted, or equal to `dist_decorr` for that generator) `reciprocal = true`, the value does
  not change when the devices swap positions; with different ones `reciprocal = false` and
  `spatial_freq_scale_b = dist_decorr / dist_decorr_b`
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
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`, with or without a
  second device; the phases of a generator therefore do not depend on the number of generators
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices (`pos_b` given): the two-device generator of [sos_val](#sos_val); with the same decorrelation distance
  (`dist_decorr_b` omitted, or equal to `dist_decorr` for that generator) `reciprocal = true`, the value does
  not change when the devices swap positions; with different ones `reciprocal = false` and
  `spatial_freq_scale_b = dist_decorr / dist_decorr_b`
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
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`, with or without a
  second device; the phases of a generator therefore do not depend on the number of generators
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices (`pos_b` given): the two-device generator of [sos_val](#sos_val); with the same decorrelation distance
  (`dist_decorr_b` omitted, or equal to `dist_decorr` for that generator) `reciprocal = true`, the value does
  not change when the devices swap positions; with different ones `reciprocal = false` and
  `spatial_freq_scale_b = dist_decorr / dist_decorr_b`
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
- One offset per generator; with two devices (`pos_b` given) it zeroes the mean of the two-device value
  of [sos_val](#sos_val) with the same `spatial_freq_scale_b` and `reciprocal`
- Two reciprocal devices at a position where the normalization of [sos_val](#sos_val) is singular (`c = -1`) throw
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
- The C++ in/out argument `phase` is split into `phase_in` and `phase_out`, so that the input stays
  unchanged; C++ messages name it `phase`

### Usage:
```matlab
[ phase_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( spatial_freq, phase_in, pos, pos_b, ...
    spatial_freq_scale_b, reciprocal );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase_in`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device as in [sos_val](#sos_val); must be > 0;
  other than 1 requires `pos_b` and `reciprocal = false`; default: 1
- **`reciprocal`** — Two devices: the mode of [sos_val](#sos_val); default: true

### Outputs:
- **`phase_out`** — Corrected phases; `[n_coeff, n_gen]`
- **`offset`** — Offsets added to the phases; `[n_gen, 1]`
- **`mean_before`** — Sample mean of the values of [sos_val](#sos_val) for the given arguments before the
  correction; `[n_gen, 1]`

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
- Two devices (`pos_b` given): one generator over the positions of both,
  `s(x, y) = sqrt(2/n_coeff) · sum_n cos( 2·pi · (f_n·x + b·f_p(n)·y) + phi_n )`, with
  `b = spatial_freq_scale_b` and `p` a fixed pseudo-random permutation of the sinusoids that depends only on
  `n_coeff` and pairs no sinusoid with itself. The correlation of two links approximates the product of the
  ACFs of the two moves; while one device stays at its position, it is the ACF of one device (with
  `reciprocal = true` for devices more than about one decorrelation distance apart)
- `reciprocal = true` (requires `b = 1`): `val = ( s(pos, pos_b) + s(pos_b, pos) ) / sqrt( 2 + 2·c )` with
  `c = (1/n_coeff) · sum_n cos( 2·pi · (f_n - f_p(n))·(pos - pos_b) )`. The value does not change when the
  two devices swap their positions; for devices closer than about one decorrelation distance the
  correlation of two links contains the term this symmetry requires
- `reciprocal = false`: `val = s(pos, pos_b)`. With `pos` and `pos_b` exchanged, the same phases give a
  second field with the same statistics, for example the arrival value of a link whose departure value
  is `val`
- For large moves of both devices the correlation keeps a residual with rms `1/sqrt(2·n_coeff)`
- With `reciprocal = true`, a position where `c = -1` (possible only for tables with few sinusoids) gives NaN
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and
  1 as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the
  tail is rounded to them
- Kernels: GENERIC computes in double precision. AVX2 evaluates sine and cosine in single
  precision; its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by
  less than 1e-6; with two devices and `reciprocal = true` the difference is scaled by
  `2 / sqrt( 2 + 2·c )`, at most about 1.7 for the presets
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The
  bound between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both
  kernels lose accuracy at the same rate
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor
  on the number of threads

### Usage:
```matlab
val = quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, spatial_freq_scale_b, reciprocal, ...
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
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `pos_b` and
  `reciprocal = false`; default: 1
- **`reciprocal`** — Two devices: true = the value does not change when the devices swap positions;
  default: true
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
