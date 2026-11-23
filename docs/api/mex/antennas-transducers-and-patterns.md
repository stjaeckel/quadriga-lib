# Antennas, transducers and patterns

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-antennas-transducers-and-patterns

| Function | Summary |
| --- | --- |
| [apply_port_weights](#apply_port_weights) | Apply the port weights of an antenna array: one pattern per port |
| [calc_beamwidth](#calc_beamwidth) | Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees |
| [calc_directivity](#calc_directivity) | Calculate the directivity in dBi of the elements of an antenna array |
| [copy_element](#copy_element) | Copy a single element of an antenna array to one or more destination slots |
| [export_pattern_obj](#export_pattern_obj) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [generate_antenna](#generate_antenna) | Generate an antenna array of a given type |
| [generate_speaker](#generate_speaker) | Generate a parametric frequency-dependent loudspeaker directivity model |
| [interpolate_pattern](#interpolate_pattern) | Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles |
| [join_arrays](#join_arrays) | Join the elements of two antenna arrays into a new array |
| [qdant_read](#qdant_read) | Read arrayant objects from a QDANT file |
| [qdant_write](#qdant_write) | Write arrayant data to a QDANT (XML) file |
| [rotate_pattern](#rotate_pattern) | Rotate the patterns of an antenna array using Euler rotations |

The functions of this section create, inspect, transform and store antenna and acoustic arrays. They share one data
structure, the arrayant struct (fields in [generate_antenna](#generate_antenna)): complex polarimetric patterns of the elements on an
azimuth and elevation grid, the element positions and the port weights. The channel functions take it as transmit and
receive antenna.

**Terms** (full list in the [Glossary](../cpp/general-usage-notes.md#glossary)):

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

Arrays are stored in the [QDANT file format](../../formats/qdant.md).

---
<a id="apply_port_weights"></a>
## apply_port_weights
Apply the port weights of an antenna array: one pattern per port

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#apply_port_weights) · [Python](../python/antennas-transducers-and-patterns.md#apply_port_weights)

- Combines the elements of an array into one pattern per port, for beamforming and to speed up MIMO channel computation
- Each port pattern is the weighted sum of the element fields from the section overview, evaluated on the output grid
  and including the phase of each element position
- The result has one element per port, identity port weights and all element positions at the origin

### Usage:
```matlab
arrayant_out = quadriga_lib.apply_port_weights( arrayant_in, freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );
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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#calc_beamwidth) · [Python](../python/antennas-transducers-and-patterns.md#calc_beamwidth)

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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#calc_directivity) · [Python](../python/antennas-transducers-and-patterns.md#calc_directivity)

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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#copy_element) · [Python](../python/antennas-transducers-and-patterns.md#copy_element)

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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#export_pattern_obj) · [Python](../python/antennas-transducers-and-patterns.md#export_pattern_obj)

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
- **`icosphere_n_div`** — Subdivisions n of the [icosphere](site-specific-simulation-tools.md#icosphere) that carries the surface; higher = finer mesh with 20 n²
  faces; default: 4
- **`i_element`** — Element indices to export, 1-based; default: all elements
- **`i_freq`** — Entry of a multi-frequency array to export, 1-based, at most `numel(arrayant)`; default: 1

### See also:
- [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) (the same pattern as numbers)
- [channel_export_obj_file](channel-functions.md#channel_export_obj_file) (exports propagation paths to an OBJ file)

---
<a id="generate_antenna"></a>
## generate_antenna
Generate an antenna array of a given type

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#generate_antenna) · [Python](../python/antennas-transducers-and-patterns.md#generate_antenna)

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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#generate_speaker) · [Python](../python/antennas-transducers-and-patterns.md#generate_speaker)

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
- **`hor_coverage_deg`** — `horn`: horizontal coverage in degrees, between the -6 dB points at full pattern
  control; NaN or <= 0: 90°; default: NaN
- **`ver_coverage_deg`** — `horn`: vertical coverage in degrees, as above; NaN or <= 0: 60°; default: NaN
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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#interpolate_pattern) · [Python](../python/antennas-transducers-and-patterns.md#interpolate_pattern)

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
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (channel coefficients from the interpolated patterns)

---
<a id="join_arrays"></a>
## join_arrays
Join the elements of two antenna arrays into a new array

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#join_arrays) · [Python](../python/antennas-transducers-and-patterns.md#join_arrays)

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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#qdant_read) · [Python](../python/antennas-transducers-and-patterns.md#qdant_read)

- Loads arrays from a file in the [QDANT format](../../formats/qdant.md). A file can hold several arrays, each with a
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
  its data, or that exceeds the [reader limits](../../formats/qdant.md), raises an error

### See also:
- [qdant_write](#qdant_write) (writes such files)
- [generate_antenna](#generate_antenna), [generate_speaker](#generate_speaker) (create arrays without a file)

---
<a id="qdant_write"></a>
## qdant_write
Write arrayant data to a QDANT (XML) file

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#qdant_write) · [Python](../python/antennas-transducers-and-patterns.md#qdant_write)

- Stores an array in a file in the [QDANT format](../../formats/qdant.md). A file can hold several arrays with distinct IDs
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

Also in: [C++](../cpp/antennas-transducers-and-patterns.md#rotate_pattern) · [Python](../python/antennas-transducers-and-patterns.md#rotate_pattern)

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
- [calc_rotation_matrix](math-functions.md#calc_rotation_matrix) (rotation matrix of the orientation)
- [generate_antenna](#generate_antenna) (electrical tilt through the port weights)
- [join_arrays](#join_arrays) (joins arrays that are then rotated together)
