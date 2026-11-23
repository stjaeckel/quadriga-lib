# Antennas, transducers and patterns

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-antennas-transducers-and-patterns

| Function | Summary |
| --- | --- |
| [apply_port_weights](#apply_port_weights) | Apply the port weights of an antenna array: one pattern per port |
| [arrayant](#arrayant) | Class for storing and manipulating an array of antenna or acoustic elements |
| [calc_beamwidth](#calc_beamwidth) | Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees |
| [calc_directivity](#calc_directivity) | Calculate the directivity in dBi of the elements of an antenna array |
| [copy_element](#copy_element) | Copy a single element of an antenna array to one or more destination slots |
| [export_pattern_obj](#export_pattern_obj) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [generate_antenna](#generate_antenna) | Generate an antenna array of a given type |
| [generate_speaker](#generate_speaker) | Generate a parametric frequency-dependent loudspeaker directivity model |
| [interpolate_pattern](#interpolate_pattern) | Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles |
| [is_valid](#is_valid) | Validate a vector of arrayant objects for multi-frequency consistency |
| [join_arrays](#join_arrays) | Join the elements of two antenna arrays into a new array |
| [qdant_read](#qdant_read) | Read arrayant objects from a QDANT file |
| [qdant_write](#qdant_write) | Write arrayant data to a QDANT (XML) file |
| [rotate_pattern](#rotate_pattern) | Rotate the patterns of an antenna array using Euler rotations, in place |
| [set_element_pos](#set_element_pos) | Set element positions for all entries in a multi-frequency arrayant vector |
| [trim_pattern](#trim_pattern) | Remove zero-valued entries from antenna pattern data, reducing its size |

---
<a id="apply_port_weights"></a>
## apply_port_weights
Apply the port weights of an antenna array: one pattern per port

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#apply_port_weights) · [Python](../python/antennas-transducers-and-patterns.md#apply_port_weights)

- Evaluates the element patterns on the output grid, applies the phase shift of each element position (which moves
  every pattern to the array origin) and sums the elements with the port weights
- The position phase is `-2 pi f d / propagation_speed` with `f = center_freq` and `d` the distance of the element
  from the wavefront plane through the origin; the default speed of light applies to radio waves, use about 343 m/s
  for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))
- The result has one element per port, identity port weights and all element positions at the origin
- Empty `element_pos` counts as zeros, empty port weights as identity
- Useful for beamforming and MIMO channel computation speedup
- Vector form: one output entry per output frequency (`freq_new`, or one per input entry); the fields are interpolated
  over frequency as in [interpolate_pattern](#interpolate_pattern), the port weights spherically between the bracketing input entries;
  output frequencies outside the entries take the nearest entry; the position phase uses each output frequency

### Declarations:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::apply_port_weights(
    const arrayant<dtype> &ant,
    const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(),
    double propagation_speed = 299792458.0);

std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::apply_port_weights(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &freq_new = arma::Col<dtype>(),
    double propagation_speed = 299792458.0);
```

### Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of arrayant objects that passes [is_valid](#is_valid)
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], strictly increasing; defaults
  to the input grid (of the first entry)
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], strictly increasing;
  defaults to the input grid (of the first entry)
- **`freq_new`** *(optional)* — Vector form: output frequencies in Hz, positive and finite; defaults to the
  `center_freq` of the entries; `[n_freq_new]`
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

### Returns:
- New `arrayant` with `n_ports` elements (= number of columns in `port_weights_re/im`), each holding the pattern of
  that port; vector form: `n_freq_new` such entries

### See also:
- [interpolate_pattern](#interpolate_pattern) (used internally to evaluate the element patterns)
- [rotate_pattern](#rotate_pattern) (useful for orienting antenna array patterns)

---
<a id="arrayant"></a>
## arrayant
Class for storing and manipulating an array of antenna or acoustic elements

Not in: MATLAB, Python

- Each element (antenna element, loudspeaker driver, microphone capsule) has a pattern and a position relative to the array phase-center
- The port weights map the elements to the ports: complex, `[n_elements, n_ports]`
- Field pattern cubes `e_theta_re/im`, `e_phi_re/im` must all be `[n_elevation, n_azimuth, n_elements]`
- `element_pos` is optional (empty = all elements at origin); `port_weights_re/im` are optional (empty = identity); all
  functions give the same results for empty members as for zeros / identity, except the IRS array of
  [get_channels_irs](channel-generation-functions.md#get_channels_irs), whose port weights hold the IRS weights and must be given. In a multi-frequency vector all
  entries use the same form (see [is_valid](#is_valid))
- Field values and port weights must be finite with magnitudes below about 1e18; larger values overflow in the
  interpolation and in the channel functions and give NaN results

### Attributes:
| Attribute                         | Size                                   | Description                                                    |
| --------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `arma::Cube<dtype> e_theta_re`    | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, real part                            |
| `arma::Cube<dtype> e_theta_im`    | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, imaginary part                       |
| `arma::Cube<dtype> e_phi_re`      | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, real part                            |
| `arma::Cube<dtype> e_phi_im`      | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, imaginary part                       |
| `arma::Col<dtype> azimuth_grid`   | `[n_azimuth]`                          | Azimuth angles in rad, in [-pi, pi], strictly increasing       |
| `arma::Col<dtype> elevation_grid` | `[n_elevation]`                        | Elevation angles in rad, in [-pi/2, pi/2], strictly increasing |
| `arma::Mat<dtype> element_pos`    | `[3, n_elements]` or empty             | Element positions in local Cartesian coords                    |
| `arma::Mat<dtype> port_weights_re`    | `[n_elements, n_ports]` or empty       | Port weights, real part                                     |
| `arma::Mat<dtype> port_weights_im`    | `[n_elements, n_ports]` or empty       | Port weights, imaginary part                                |
| `dtype center_freq`          | scalar                                 | Center frequency                                               |
| `std::string name`                | string                                 | Name of the antenna array object                               |

### Member functions:
| Function                                                  | Description                                                                                                                                                         |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.n_elevation()`                                          | Number of elevation angles                                                                                                                                          |
| `.n_azimuth()`                                            | Number of azimuth angles                                                                                                                                            |
| `.n_elements()`                                           | Number of antenna elements                                                                                                                                          |
| `.n_ports()`                                              | Number of ports (columns of port weights)                                                                                                                        |
| `.reset()`                                                | Clears all data, resetting size to zero                                                                                                                             |
| `.set_size(n_elevation, n_azimuth, n_elements, n_ports)` | Resizes all members (`arma::uword` sizes); no-op if they match; after a resize `element_pos` is zero, `port_weights_re` identity, `port_weights_im` zero, field data undefined |
| `.is_valid()`                                             | `std::string`: empty if the object is valid, else an error message                                                                                                  |

- `is_valid()` always checks the data: sizes of all members, ranges and order of the angle grids, sizes of element positions and port weights; empty `element_pos` and port weights are allowed
- A passing check marks the object as validated: the library functions then skip the full check as long as no data member was reallocated or resized; values changed in place after the check are not seen by them
- The mark is updated atomically, so `is_valid()` may be called while other threads read the object; a copy is marked if its source passes the full check, a moved-to object takes over the mark of its source
- The free functions of this section compute on an arrayant and read and write it (e.g. [calc_beamwidth](#calc_beamwidth), [calc_directivity](#calc_directivity), [join_arrays](#join_arrays), [apply_port_weights](#apply_port_weights), [copy_element](#copy_element), [rotate_pattern](#rotate_pattern), [qdant_write](#qdant_write)); [generate_antenna](#generate_antenna) creates one

### See also:
- [is_valid](#is_valid) (multi-frequency counterpart of `is_valid()`)

---
<a id="calc_beamwidth"></a>
## calc_beamwidth
Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#calc_beamwidth) · [Python](../python/antennas-transducers-and-patterns.md#calc_beamwidth)

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Loudspeakers state the coverage angle between the -6 dB points: use `threshold_dB = 6`
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in each direction than the antenna sampling grid)
- The width is that of the main lobe: starting at the maximum, the search goes outwards until the power drops below the
  threshold; side lobes above the threshold are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360° when
  the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°
- Calculated per element, not per port; ignores the port weights
- Vector form: the same computation on every entry of a multi-frequency vector; column `i_freq` of each output holds
  the results of `arrayant_vec[i_freq]`

### Declarations:
```cpp
void quadriga_lib::calc_beamwidth(
    const arrayant<dtype> &ant,
    const arma::uvec &i_element = arma::uvec(),
    double threshold_dB = 3.0,
    arma::Col<dtype> *beamwidth_az_deg = nullptr,
    arma::Col<dtype> *beamwidth_el_deg = nullptr,
    arma::Col<dtype> *az_point_ang_deg = nullptr,
    arma::Col<dtype> *el_point_ang_deg = nullptr);

void quadriga_lib::calc_beamwidth(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::uvec &i_element = arma::uvec(),
    double threshold_dB = 3.0,
    arma::Mat<dtype> *beamwidth_az_deg = nullptr,
    arma::Mat<dtype> *beamwidth_el_deg = nullptr,
    arma::Mat<dtype> *az_point_ang_deg = nullptr,
    arma::Mat<dtype> *el_point_ang_deg = nullptr);
```

### Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of `n_freq` arrayant objects that passes [is_valid](#is_valid)
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`
- **`threshold_dB`** *(optional)* — Threshold in dB, positive and finite; 3 dB = FWHM

### Outputs:
- **`beamwidth_az_deg`** *(optional)* — Azimuth beamwidth in degree; `[n_out]`, vector form `[n_out, n_freq]`
- **`beamwidth_el_deg`** *(optional)* — Elevation beamwidth in degree; `[n_out]`, vector form `[n_out, n_freq]`
- **`az_point_ang_deg`** *(optional)* — Azimuth pointing angle of the main beam in degree; `[n_out]`, vector form
  `[n_out, n_freq]`
- **`el_point_ang_deg`** *(optional)* — Elevation pointing angle of the main beam in degree; `[n_out]`, vector form
  `[n_out, n_freq]`

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("custom", 1.0, NAN, {30.0, 20.0});
arma::vec bw_az, bw_el;
quadriga_lib::calc_beamwidth(ant, {}, 3.0, &bw_az, &bw_el); // bw_az(0) = 30, bw_el(0) = 20
```

### See also:
- [calc_directivity](#calc_directivity) (directivity in dBi of the elements)
- [apply_port_weights](#apply_port_weights) (apply the port weights before calculating the beamwidth)

---
<a id="calc_directivity"></a>
## calc_directivity
Calculate the directivity in dBi of the elements of an antenna array

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#calc_directivity) · [Python](../python/antennas-transducers-and-patterns.md#calc_directivity)

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Calculated per element, not per port; ignores the port weights
- Vector form: the same computation on every entry of a multi-frequency vector; column `i_freq` of the output holds
  the results of `arrayant_vec[i_freq]`

### Declarations:
```cpp
arma::Col<dtype> quadriga_lib::calc_directivity(
    const arrayant<dtype> &ant,
    const arma::uvec &i_element = arma::uvec());

arma::Mat<dtype> quadriga_lib::calc_directivity(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of `n_freq` arrayant objects that passes [is_valid](#is_valid)
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`

### Returns:
- Directivity in dBi; `[n_out]`, vector form `[n_out, n_freq]`

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("dipole");
arma::vec directivity_dBi = quadriga_lib::calc_directivity(ant); // about 1.76 dBi
```

### See also:
- [calc_beamwidth](#calc_beamwidth) (beamwidth and pointing angles of the elements)
- [apply_port_weights](#apply_port_weights) (apply the port weights before calculating the directivity)

---
<a id="copy_element"></a>
## copy_element
Copy a single element of an antenna array to one or more destination slots

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#copy_element) · [Python](../python/antennas-transducers-and-patterns.md#copy_element)

- Works in place; the array is enlarged if any destination index exceeds the current number of elements
- Added elements get their own port with identity port weights; positions of added elements that are not copied are zero
- When the array grows, an empty `element_pos` and empty port weights are written out first (zeros, identity), so
  that all elements can be edited afterwards; otherwise empty members stay empty
- Throws `std::invalid_argument` for an empty `i_target`, a source index out of bound, or a destination index
  equal to the largest `arma::uword` (e.g. `-1` cast to unsigned)
- Vector form: the same copy on every entry of a multi-frequency vector

### Declarations:
```cpp
void quadriga_lib::copy_element(
    arrayant<dtype> &ant,
    arma::uword i_source,
    const arma::uvec &i_target);

void quadriga_lib::copy_element(
    std::vector<arrayant<dtype>> &arrayant_vec,
    arma::uword i_source,
    const arma::uvec &i_target);
```

### Inputs:
- **`ant`** — Valid arrayant; updated in-place
- **`arrayant_vec`** — Vector form: non-empty vector of valid arrayant objects; updated in-place
- **`i_source`** — Index of the element to copy, 0-based
- **`i_target`** — Target indices, 0-based; a single index is passed as `{i}`

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("dipole");
quadriga_lib::copy_element(ant, 0, {1, 2, 3}); // 4 identical elements
ant.element_pos.row(1) = arma::rowvec({-0.75, -0.25, 0.25, 0.75});
```

### See also:
- [set_element_pos](#set_element_pos) (set element positions of all entries of a vector after copying)
- [join_arrays](#join_arrays) (join two arrays)

---
<a id="export_pattern_obj"></a>
## export_pattern_obj
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#export_pattern_obj) · [Python](../python/antennas-transducers-and-patterns.md#export_pattern_obj)

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives finer mesh
- Each element is placed at its position in `element_pos` (empty: at the origin)

### Declaration:
```cpp
void quadriga_lib::export_pattern_obj(
    const std::string &fn,
    const arrayant<dtype> &ant,
    double directivity_range_dB = 30.0,
    const std::string &colormap = "jet",
    double object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`fn`** — Output OBJ filename; must end in .obj; the colormap is written to a .mtl file of the same name
- **`ant`** — Valid arrayant
- **`directivity_range_dB`** *(optional)* — Dynamic range of the visualized directivity pattern in dB
- **`colormap`** *(optional)* — Colormap name; see [colormap](site-specific-simulation-tools.md#colormap) for supported options
- **`object_radius`** *(optional)* — Radius of the exported object
- **`icosphere_n_div`** *(optional)* — Icosphere subdivision count; higher = finer mesh, see [icosphere](site-specific-simulation-tools.md#icosphere)
- **`i_element`** *(optional)* — 0-based element indices to export; empty exports all elements

### See also:
- [colormap](site-specific-simulation-tools.md#colormap) (Used for setting the colormap)
- [icosphere](site-specific-simulation-tools.md#icosphere) (Used internally to generate icosphere primitive)
- [channel_export_obj_file](channel-functions.md#channel_export_obj_file) (exports propagation paths to OBJ 3D visualization)

---
<a id="generate_antenna"></a>
## generate_antenna
Generate an antenna array of a given type

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#generate_antenna) · [Python](../python/antennas-transducers-and-patterns.md#generate_antenna)

- Tuples (`element`, `panel`, `spacing`, `pol`, `ports`, `steer`): entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their default, e.g. `panel = {4, 8}` is one panel of 4 rows and 8 columns.
- Panel arrays: elements are ordered polarization fastest, then row `m` (upwards along z), column `n` (along +y), panel row `mg`, panel column `ng`, centred on the origin; ports in the same order with `m_p`, `n_p`. The port weights `[n_elements, n_ports]` holds the weights; `apply_weights` combines each port into one element at its sub-array centre.
- Sub-array weights (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds `K = M / M_P` × `L = N / N_P` elements; element `(k, l)` gets `exp(j 2π ((k - (K-1)/2) d_V sin(tilt) - (l - (L-1)/2) d_H sin(pan))) / sqrt(K L)`; a positive tilt points the beam down, a positive pan towards +y.
- Polarization (TR 38.901 clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45° (element 0 first). Model-1 turns the element by ζ about its boresight (eq. 7.3-3); Model-2 sets F_θ = sqrt(A') cos ζ and F_φ = sqrt(A') sin ζ with A' = |F_θ|² + |F_φ|² of the element (eq. 7.3-4, 7.3-5).
- `multibeam`: maximum-ratio-transmission weights for the directions in `beams` (exact for one beam, approximate for several; only the ratios of the beam weights matter); one port per polarization, with `separate_beams` one per polarization and beam.
- An input that the type does not take throws.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_antenna(
    const std::string &type,
    double res = NAN,
    double center_freq = 299792458.0,
    const arma::vec &element = {},
    const arma::uvec &panel = {},
    const arma::vec &spacing = {},
    const arma::uvec &pol = {},
    const arma::uvec &ports = {},
    const arma::mat &steer = {},
    const arma::mat &beams = {},
    const quadriga_lib::arrayant<dtype> &pattern = quadriga_lib::arrayant<dtype>(),
    bool separate_beams = false,
    bool apply_weights = false);
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
   | `multibeam`                  | Panel array with beams towards given directions                            | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 10° for `omni` and `xpol`, 1° otherwise; ignored if `pattern` is given
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite; NaN: 299792458 (wavelength 1 m)
- **`element`** *(optional)* — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi (finite or NaN); default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity; not together with `pattern`
- **`panel`** *(optional)* — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; default 1; at most 1e7 elements per polarization (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** *(optional)* — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing; finite and >= 0 (0 places elements or panels at the same position); default `[0.5, d_V, M d_V, N d_H]`
- **`pol`** *(optional)* — `[polarization, model]`: polarization 1 (vertical), 2 (H/V) or 3 (±45°); model 1 (Model-1) or 2 (Model-2); default `[1, 1]`
- **`ports`** *(optional)* — `[M_P, N_P]`: ports (TXRUs) per column and per row of a panel and polarization; must divide `M` and `N`; default `[M, N]` (one port per element)
- **`steer`** *(optional)* — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt and pan of the sub-array weights; `[1, 2]` for all panels or `[Mg Ng, 2]` with one row per panel (panel order `mg`, `ng`); one column: pan 0; default 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`
- **`beams`** *(optional)* — `[2, n_beams]` or `[3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); finite; default `[0; 0; 1]`
- **`pattern`** *(optional)* — Element to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; empty (no elements): the element of the type
- **`separate_beams`** *(optional)* — If `true`, one port per beam (weights ignored)
- **`apply_weights`** *(optional)* — If `true`, combine each port into one pattern at its sub-array centre

### Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna array object; panel arrays have `P M N Mg Ng` elements and `P M_P N_P Mg Ng` ports (`ula`, `3GPP`); with `apply_weights`, one element per port

### Example:
```cpp
// Vertically polarized omni antenna, 1 degree grid; custom element with 30 x 20 degree beamwidth at 299792458 Hz
auto omni = quadriga_lib::generate_antenna("omni", 1.0);
auto beam = quadriga_lib::generate_antenna("custom", 1.0, NAN, {30.0, 20.0});

// Table 7.8-2, BS config 1: two 4 x 4 panels 2.5 wavelengths apart, +/-45 deg Model-2, one port per polarization
// and panel, 12 degree electrical downtilt
auto bs = quadriga_lib::generate_antenna("3GPP", 1.0, 6.0e9, {}, {4, 4, 1, 2}, {0.5, 0.5, 2.5, 2.5}, {3, 2}, {1, 1}, {12.0, 0.0});
```

### See also:
- [apply_port_weights](#apply_port_weights) (combines the ports into patterns)
- [rotate_pattern](#rotate_pattern) (turns an antenna)

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#generate_speaker) · [Python](../python/antennas-transducers-and-patterns.md#generate_speaker)

- Returns one [arrayant](#arrayant) object per frequency sample; each has a single element with the real-valued 
  directivity pattern in `e_theta_re` and `center_freq` set to the corresponding frequency.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining results 
  via `append` and `element_pos`; crossover behavior emerges from overlapping bandpass responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`, 
  where `n = slope_dB_per_octave / 6`; −3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity_dB − 85) / 20)`.
- If `center_freq` is empty, third-octave band center frequencies are auto-generated from one band below 
  `lower_cutoff` to one band above `upper_cutoff`, clipped to 20–20000 Hz.
- Speed of sound assumed to be 343 m/s.
- **Driver models** (`driver_type`):
  - `piston` — circular piston in baffle, `D(θ) = 2·J1(ka·sinθ)/(ka·sinθ)`, rotationally symmetric, narrows with increasing `ka`
  - `horn` — separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`
  - `omni` — frequency-independent omnidirectional pattern.
- **Enclosure models** (`radiation_type`):
  - `monopole` — no modification
  - `hemisphere` — sealed box with baffle-step transition, `f_baffle = c/(π·sqrt(W·H))`
  - `dipole` — figure-8, `R = abs(cos(θ_off))` with sign inversion in rear hemisphere
  - `cardioid` — `R = 0.5·(1+cos(θ_off))`
- For `"horn"`, `hor_coverage_deg`, `ver_coverage_deg` and `horn_control_freq` set to NaN (the default) or <= 0 select 90°,
  60° and `f_ctrl = c/(2π·radius)`.

### Declaration:
```cpp
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::generate_speaker(
        const std::string &driver_type = "piston",
        double radius = 0.05,
        double lower_cutoff = 80.0,
        double upper_cutoff = 12000.0,
        double lower_rolloff_slope_dB = 12.0,
        double upper_rolloff_slope_dB = 12.0,
        double sensitivity_dB = 85.0,
        const std::string &radiation_type = "hemisphere",
        double hor_coverage_deg = NAN,
        double ver_coverage_deg = NAN,
        double horn_control_freq = NAN,
        double baffle_width = 0.15,
        double baffle_height = 0.25,
        const arma::Col<dtype> &center_freq = arma::Col<dtype>(),
        double angular_resolution_deg = 5.0);
```

### Inputs:
- **`driver_type`** *(optional)* — Driver directivity model: `"piston"`, `"horn"`, or `"omni"`
- **`radius`** *(optional)* — Effective radiating radius in m, > 0; cone/dome radius for piston, mouth radius for horn
- **`lower_cutoff`** *(optional)* — Lower −3 dB bandpass frequency in Hz, > 0
- **`upper_cutoff`** *(optional)* — Upper −3 dB bandpass frequency in Hz, larger than `lower_cutoff`
- **`lower_rolloff_slope_dB`** *(optional)* — Low-frequency rolloff in dB/octave, > 0 (12 dB/oct = 2nd-order Butterworth)
- **`upper_rolloff_slope_dB`** *(optional)* — High-frequency rolloff in dB/octave, > 0
- **`sensitivity_dB`** *(optional)* — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude
- **`radiation_type`** *(optional)* — Enclosure radiation model: `"monopole"`, `"hemisphere"`, `"dipole"`, or `"cardioid"`
- **`hor_coverage_deg`** *(optional)* — Horn horizontal coverage angle in degrees; NaN or <= 0: default 90°
- **`ver_coverage_deg`** *(optional)* — Horn vertical coverage angle in degrees; NaN or <= 0: default 60°
- **`horn_control_freq`** *(optional)* — Horn pattern control frequency; NaN or <= 0: default `c/(2π·radius)`
- **`baffle_width`** *(optional)* — Baffle width in m, > 0; used by `"hemisphere"` model
- **`baffle_height`** *(optional)* — Baffle height in m, > 0; used by `"hemisphere"` model
- **`center_freq`** *(optional)* — Frequency sample points in Hz, `[n_freq]`; empty: third-octave bands
- **`angular_resolution_deg`** *(optional)* — Azimuth and elevation sampling grid resolution in degrees; 0.1 to 90

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One arrayant per frequency sample with directivity in `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180° phase inversion

### Example:
```cpp
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
               12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
auto horn = quadriga_lib::generate_speaker<double>("horn");
arma::vec sub_freqs = {30.0, 50.0, 80.0, 120.0, 200.0};
auto sub = quadriga_lib::generate_speaker<double>("omni", 0.13, 30.0, 200.0,
               12.0, 24.0, 92.0, "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, sub_freqs, 10.0);
```

---
<a id="interpolate_pattern"></a>
## interpolate_pattern
Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#interpolate_pattern) · [Python](../python/antennas-transducers-and-patterns.md#interpolate_pattern)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the array and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation),
  with linear interpolation where neighbouring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- `azimuth` with one row gives the same angles for all outputs (planar wave), `n_out` rows one angle per output
  (spherical wave); `n_ang = 0` gives outputs of size `[n_out, 0]` (vector form `[n_out, 0, n_freq]`)
- Angles are not range-checked; values outside [-pi, pi] and [-pi/2, pi/2] denote the corresponding direction. NaN or
  infinite angles give NaN in all outputs of that angle and output, without an exception
- Outputs are resized only when their size differs, so pre-allocated outputs are written in place
- On a pole of the local frame (a direction along the z-axis of the oriented array) the local azimuth is chosen so
  that the local and the global theta unit vectors agree: the polarization is not rotated there
- Kernels: GENERIC computes in the precision of `dtype`; AVX2 computes in single precision also for `double`, with
  errors of up to about 2e-6 of the peak field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole
  of the local frame up to about 2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most
  4e-4 rad) in the local elevation; auto selects GENERIC in this version, which may change in a later version
- Throws `std::invalid_argument` for an invalid `ant`, sizes that do not match, element indices `>= n_elements`, an
  unknown or unavailable kernel, and outputs that are an input, a data member of `ant` or another output, or share
  memory with one
- Vector form (multi-frequency array, one entry per center frequency): for each target frequency, interpolates the two
  entries whose center frequencies bracket it and blends their fields: spherical interpolation of the complex values
  over frequency, linear where the two values have nearly opposite phases or one of them is zero. Target frequencies
  outside the range of the center frequencies take the nearest entry (no extrapolation). Consecutive target
  frequencies with the same bracketing entries reuse the spatial results; ascending or descending order is fastest.
  The entries must pass [is_valid](#is_valid); it also throws for an empty `arrayant_vec` and target frequencies that are not
  finite

### Declarations:
```cpp
void quadriga_lib::interpolate_pattern(
    const arrayant<dtype> &ant,
    const arma::Mat<dtype> &azimuth,
    const arma::Mat<dtype> &elevation,
    arma::Mat<dtype> &V_re,
    arma::Mat<dtype> &V_im,
    arma::Mat<dtype> &H_re,
    arma::Mat<dtype> &H_im,
    const arma::uvec &i_element = arma::uvec(),
    const arma::Cube<dtype> &orientation = arma::Cube<dtype>(),
    const arma::Mat<dtype> &element_pos = arma::Mat<dtype>(),
    arma::Mat<dtype> *dist = nullptr,
    arma::Mat<dtype> *azimuth_loc = nullptr,
    arma::Mat<dtype> *elevation_loc = nullptr,
    arma::Mat<dtype> *gamma = nullptr,
    arma::Mat<std::complex<dtype>> *V = nullptr,
    arma::Mat<std::complex<dtype>> *H = nullptr,
    int use_kernel = 0);

void quadriga_lib::interpolate_pattern(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Mat<dtype> &azimuth,
    const arma::Mat<dtype> &elevation,
    arma::Cube<dtype> &V_re,
    arma::Cube<dtype> &V_im,
    arma::Cube<dtype> &H_re,
    arma::Cube<dtype> &H_im,
    const arma::Col<dtype> &freq_new = arma::Col<dtype>(),
    const arma::uvec &i_element = arma::uvec(),
    const arma::Cube<dtype> &orientation = arma::Cube<dtype>(),
    const arma::Mat<dtype> &element_pos = arma::Mat<dtype>(),
    arma::Cube<std::complex<dtype>> *V = nullptr,
    arma::Cube<std::complex<dtype>> *H = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`ant`** — Antenna array with `n_elements` elements; see [arrayant](#arrayant)
- **`arrayant_vec`** — Vector form: multi-frequency array, one entry per center frequency; entries need not be sorted
  by frequency
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`freq_new`** *(optional)* — Vector form: target frequencies in Hz; empty: the center frequencies of the entries
  in vector order; `[n_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements in
  order (`n_out = n_elements`); `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) of the array, for all outputs and angles,
  per output, per angle, or both, applied at all frequencies; empty: no rotation; `[3, 1, 1]`, `[3, n_out, 1]`,
  `[3, 1, n_ang]` or `[3, n_out, n_ang]`
- **`element_pos`** *(optional)* — Element positions in local antenna coordinates, used for `dist`; empty: the
  columns `i_element` of `ant.element_pos`, zeros if that is empty; vector form: checked for its size only, since no
  output depends on the positions; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (throws if unavailable or if the pattern
  has more than `INT_MAX` samples); other values throw

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`, vector form
  `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang]`, vector form
  `[n_out, n_ang, n_freq]`
- **`dist`** *(optional)* — Distance of each element from the wavefront plane through the origin of the array;
  negative for elements closer to the source; `[n_out, n_ang]` (single form only)
- **`azimuth_loc`** / **`elevation_loc`** *(optional)* — Azimuth and elevation in the local antenna frame;
  `[n_out, n_ang]` (single form only)
- **`gamma`** *(optional)* — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`
  (single form only)
- **`V`** / **`H`** *(optional)* — e-theta / e-phi field as complex values, replacing `V_re`, `V_im` / `H_re`, `H_im`
  (then left empty); `[n_out, n_ang]`, vector form `[n_out, n_ang, n_freq]`

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("custom");
arma::mat azimuth = {0.0, 0.5 * arma::datum::pi, -0.5 * arma::datum::pi, arma::datum::pi};
arma::mat elevation(1, azimuth.n_elem, arma::fill::zeros);
arma::mat V_re, V_im, H_re, H_im;
quadriga_lib::interpolate_pattern(ant, azimuth, elevation, V_re, V_im, H_re, H_im);

auto speaker = quadriga_lib::join_arrays(woofer, tweeter); // Vectors from generate_speaker
arma::vec freq = {250.0, 1500.0, 8000.0};
arma::cube Vf_re, Vf_im, Hf_re, Hf_im;
quadriga_lib::interpolate_pattern(speaker, azimuth, elevation, Vf_re, Vf_im, Hf_re, Hf_im, freq);
```

### See also:
- [arrayant](#arrayant) (antenna array class)
- [join_arrays](#join_arrays) (build multi-element, multi-frequency models)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrays)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (channel coefficients from the interpolated patterns)

---
<a id="is_valid"></a>
## is_valid
Validate a vector of arrayant objects for multi-frequency consistency

Not in: MATLAB, Python

- Each entry is validated individually via its `is_valid` member (full check; a passing entry is marked as validated).
- Cross-entry checks (all vs. the first entry): azimuth/elevation grid sizes and values, number of elements, element positions, port_weights_re shape, and port_weights_im presence and size.
- Pattern data, `center_freq`, and port weights values are not compared (expected to vary).
- All entries must use the same form for `element_pos`, `port_weights_re` and `port_weights_im`: all empty or all given; a
  mix of empty members and explicit zeros / identity is rejected.
- Stops at first error and returns a message identifying the failing entry and property.

### Declaration:
```cpp
std::string quadriga_lib::is_valid(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Input:
- **`arrayant_vec`** — Non-empty vector of arrayant objects to validate

### Returns:
- Empty string if valid; otherwise a message such as `"Entry 3: Azimuth grid values do not match the first entry."`

### See also:
- [arrayant](#arrayant) (`is_valid()`, the per-entry validation called internally)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="join_arrays"></a>
## join_arrays
Join the elements of two antenna arrays into a new array

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#join_arrays) · [Python](../python/antennas-transducers-and-patterns.md#join_arrays)

- Both arrays must share identical sampling grids; throws otherwise
- The port weights are block-diagonal: elements of the first array connect only to its ports, and likewise for the
  second; name and `center_freq` are taken from the first array:
   | Element \ Port        | P1…Pp1 (first) | Pp1+1…Pp1+p2 (second) |
   | --------------------- | :------------: | :-------------------: |
   | E1…En1 (first)        |    W1 block    |           0           |
   | En1+1…En1+n2 (second) |       0        |       W2 block        |
- An empty `element_pos` counts as zeros, an empty `port_weights_re` as identity (one port per element), an empty
  `port_weights_im` as zeros
- Vector form: joins the entries of two multi-frequency vectors pairwise; both must pass [is_valid](#is_valid), have equal entry
  counts, identical angular grids and matching `center_freq` values at each index; each output entry is validated

### Declarations:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::join_arrays(
    const arrayant<dtype> &ant_1,
    const arrayant<dtype> &ant_2);

std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::join_arrays(
    const std::vector<arrayant<dtype>> &arrayant_vec1,
    const std::vector<arrayant<dtype>> &arrayant_vec2);
```

### Inputs:
- **`ant_1`** / **`arrayant_vec1`** — First arrayant (vector form: vector); its elements come first in the output
- **`ant_2`** / **`arrayant_vec2`** — Second arrayant (vector form: vector); sampling grids (and center frequencies)
  must match the first

### Returns:
- New `arrayant` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports; vector form: one such entry per
  frequency

### See also:
- [set_element_pos](#set_element_pos) (position drivers before joining)
- [rotate_pattern](#rotate_pattern) (rotate elements after joining)
- [qdant_write](#qdant_write) (persist the joined model)

---
<a id="qdant_read"></a>
## qdant_read
Read arrayant objects from a QDANT file

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#qdant_read) · [Python](../python/antennas-transducers-and-patterns.md#qdant_read)

- Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the entries with the given IDs. The ID is the number used in the layout matrix and by [qdant_write](#qdant_write).
- Without `ids`, the entries listed in the layout are returned, each once, in order of first appearance (column-major scan); all entries if the file has no layout. This is the way to load frequency-dependent models, where `center_freq` of each entry identifies the frequency.
- Throws `std::invalid_argument` if an ID is 0 or the file has no entry with it. The first entry of a file is ID 1 only
  if it has no `id` attribute (files written by other tools); otherwise ID 1 must exist.
- A missing, unreadable or malformed file throws `std::runtime_error`. The number of elements (`NoElements`) must be
  backed by data in the file (element positions, port weights, or pattern nodes of the last element). Elements without
  pattern nodes read as zeros; their zero patterns, and the identity port weights of a file without `CouplingAbs`, are
  limited to 2^24 values each (e.g. 256 elements without pattern data on a 1-degree grid, 4096 elements without
  port weights).

### Declaration:
```cpp
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read(
        const std::string &fn,
        const arma::uvec &ids = {},
        arma::u32_mat *layout = nullptr);
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`ids`** *(optional)* — 1-based IDs of the entries to read, duplicates allowed; empty reads the entries in the layout
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix of entry IDs (0 = empty slot)

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per requested ID, in the order of `ids`

### See also:
- [qdant_write](#qdant_write) (write a single arrayant, or a vector with sequential IDs)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---
<a id="qdant_write"></a>
## qdant_write
Write arrayant data to a QDANT (XML) file

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#qdant_write) · [Python](../python/antennas-transducers-and-patterns.md#qdant_write)

- Multiple antennas can be stored in the same file using distinct `id` values
- If `id = 0` and the file exists, the antenna is appended with `id = max_existing_id + 1`; if file does not exist, `id = 1` gets assigned
- Without `layout` (empty), the layout in the file is replaced by a `[1, n]` row of all IDs in the file
  in file order; a custom layout written before is lost. Pass the layout again with every write that should keep it.
- The file has single-precision-equivalent accuracy relative to the element peak, also for double-precision data:
  pattern magnitudes (in dB) and phases (in degree) get the fewest decimals (0 to 6) that keep the error of each
  complex sample at or below 6e-8 times the peak amplitude of its element (over e_theta and e_phi). Samples near
  the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the
  peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at
  a low level is quantized relative to its own peak. Element positions and port weights (magnitude, phase) are written
  as the shortest text that reads back as the same float (at most 9 significant digits); sampling grids are rounded
  to 0.001 degree
- The port weights are stored in the tags `CouplingAbs` and `CouplingPhase` of the file format
- An empty `element_pos` or empty port weights are not written; [qdant_read](#qdant_read) returns zeros and identity for them.
  Above 4096 elements the identity port weights are written, because the reader requires them in the file
- All non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element that is
  zero in both field components is written as an all `-inf` `EthetaMag` node.
- Vector form: writes each entry with sequential 1-based IDs and a `[n_entries, 1]` layout `1, 2, ..., n_entries`; the
  file must not exist yet (an existing file throws `std::runtime_error` and is left unchanged); for frequency-dependent
  models where each entry holds the pattern at its `center_freq`; read back with [qdant_read](#qdant_read)

### Declarations:
```cpp
unsigned quadriga_lib::qdant_write(
    const std::string &fn,
    const arrayant<dtype> &ant,
    unsigned id = 0,
    const arma::u32_mat &layout = arma::u32_mat());

void quadriga_lib::qdant_write(
    const std::string &fn,
    const std::vector<arrayant<dtype>> &arrayant_vec);
```

### Inputs:
- **`fn`** — Output QDANT filename; must not be empty; vector form: must not exist
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of valid arrayant objects to store
- **`id`** *(optional)* — Target ID in file; `0` appends with auto-assigned ID
- **`layout`** *(optional)* — Matrix organizing multiple antenna IDs within the file; must reference only IDs present in the file

### Returns:
- ID assigned to the written antenna within the file (single form)

### See also:
- [qdant_read](#qdant_read) (read back QDANT files)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---
<a id="rotate_pattern"></a>
## rotate_pattern
Rotate the patterns of an antenna array using Euler rotations, in place

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#rotate_pattern) · [Python](../python/antennas-transducers-and-patterns.md#rotate_pattern)

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [interpolate_pattern](#interpolate_pattern) returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); port weights, `center_freq` and `name` are kept
- Usage 0 and 1 adjust the sampling grid for non-uniformly sampled patterns; this needs all elements, so a subset
  in `i_element` throws when the grid must be adjusted
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects
- Vector form: rotates every entry of a multi-frequency vector with grid adjustment always disabled (required for
  uniform-grid consistency across frequencies); usage 0 to 2

### Declarations:
```cpp
void quadriga_lib::rotate_pattern(
    arrayant<dtype> &ant,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const arma::uvec &i_element = arma::uvec());

void quadriga_lib::rotate_pattern(
    std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`ant`** — Valid arrayant; updated in place
- **`arrayant_vec`** — Vector form: non-empty vector of arrayant objects; updated in place
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; `[3, 1]` or `[1, 3]`
- **`usage`** *(optional)* — Rotation mode:
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`i_element`** *(optional)* — Element indices to rotate, 0-based, no duplicates; empty: all elements

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("3GPP", 1.0, NAN, {}, {4});
arma::vec downtilt = {0.0, 10.0 * arma::datum::pi / 180.0, 0.0}; // 10 degree mechanical downtilt
quadriga_lib::rotate_pattern(ant, downtilt);
```

### See also:
- [interpolate_pattern](#interpolate_pattern) (same orientation convention)
- [calc_rotation_matrix](math-functions.md#calc_rotation_matrix) (rotation matrix of the orientation)
- [join_arrays](#join_arrays) (join arrays before rotating)

---
<a id="set_element_pos"></a>
## set_element_pos
Set element positions for all entries in a multi-frequency arrayant vector

Not in: MATLAB, Python

- Updates `element_pos` in-place on every entry in the vector identically.
- If `i_element` is empty, all positions are replaced and `element_pos` must have `n_elements` columns.
- If `i_element` is non-empty, only those indexed columns are updated; `element_pos` column count must match `i_element` length.
- All entries must have the same element count; uninitialized `element_pos` fields are zero-initialized before update.

### Declaration:
```cpp
void quadriga_lib::set_element_pos(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`element_pos`** — New (x, y, z) positions; `[3, n_update]`
- **`i_element`** *(optional)* — Indices of elements to update; empty: all elements are replaced

### See also:
- [copy_element](#copy_element) (replicate elements before setting positions)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="trim_pattern"></a>
## trim_pattern
Remove zero-valued entries from antenna pattern data, reducing its size

Not in: MATLAB, Python

- Works in place on the sampling grid and the pattern cubes; positions, port weights, name and frequency are unchanged
- A column or row is removed if the total power summed over all field components and array elements does not exceed
  1e-12 times the peak power of a single sample (summed over field components and elements); the result does not
  depend on the scale of the pattern; an all-zero pattern is left unchanged
- Immediate neighbors of retained entries are also kept unless the local grid spacing is strongly asymmetric (factor > 1.5)
- To keep the original, call it on a copy (`auto b = ant;`)

### Declaration:
```cpp
void quadriga_lib::trim_pattern(arrayant<dtype> &ant);
```

### Input:
- **`ant`** — Valid arrayant; updated in-place
