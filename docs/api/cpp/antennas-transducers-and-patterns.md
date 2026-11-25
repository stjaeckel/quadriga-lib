# Antennas, transducers and patterns

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-antennas-transducers-and-patterns

| Function | Summary |
| --- | --- |
| [arrayant](#arrayant) | Class for storing and manipulating an array of antenna or acoustic elements |
| [apply_port_weights](#apply_port_weights) | Apply the port weights of an antenna array: one pattern per port |
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

The functions of this section create, inspect, transform and store antenna and acoustic arrays. They share one data
structure, [arrayant](#arrayant): complex polarimetric patterns of the elements on an azimuth and elevation grid, the element
positions and the port weights. The channel functions take it as transmit and receive antenna.

**Terms** (full list in the [Glossary](general-usage-notes.md#glossary)):

- **Element**: one radiator or sensor (antenna element, loudspeaker driver, microphone capsule) with a pattern, a
  position and one row of the port weights
- **Grid**: azimuth $\phi$ in $[-\pi, \pi]$, measured from +x toward +y, and elevation $\theta$ in $[-\pi/2, \pi/2]$,
  measured from the x-y plane (TR 38.901 uses the zenith angle $90° - \theta$), both in rad and strictly increasing.
  Boresight is +x
- **Pattern**: the direction-dependent far field of an element: $F_\theta$ (`e_theta_re/im`) along
  $\hat\theta = (\sin\theta\cos\phi, \sin\theta\sin\phi, -\cos\theta)$, the zenith-angle unit vector of TR 38.901, which
  points toward decreasing elevation, and $F_\phi$ (`e_phi_re/im`) along $\hat\phi = (-\sin\phi, \cos\phi, 0)$. For an
  antenna, $\lvert F_\theta \rvert^2 + \lvert F_\phi \rvert^2$ is the linear gain over an isotropic radiator
- **Port**: one signal of the array, one column of the port weights
- **Port weights**: the fixed complex feed network $w_{i,p}$ from element $i$ to port $p$, `[n_elements, n_ports]`. It
  is not the mutual coupling between elements. Empty means identity (one port per element)
- **Element position**: local Cartesian coordinates in m, `[3, n_elements]`; empty means all at the origin
- **Multi-frequency array**: a `std::vector` of arrayants, one entry per `center_freq`, with equal grids, element
  positions and port weight sizes ([is_valid](#is_valid)). [interpolate_pattern](#interpolate_pattern) and [apply_port_weights](#apply_port_weights) interpolate over
  frequency between the entries

The field of port $p$ in a direction is the weighted sum of the element fields, each with the phase of its position,

$$F_p = \sum_i w_{i,p} \, F_i \, e^{-j 2\pi f d_i / c}$$

where $d_i = -\hat{r} \cdot x_i$ is the distance of element $i$ at position $x_i$ from the wavefront plane through the
origin, negative for elements closer to the source, $\hat{r}$ the unit vector toward the source, $f$ the frequency and
$c$ the propagation speed.

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Create | [generate_antenna](#generate_antenna), [generate_speaker](#generate_speaker), [qdant_read](#qdant_read) |
| Evaluate | [interpolate_pattern](#interpolate_pattern), [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) |
| Transform | [apply_port_weights](#apply_port_weights), [rotate_pattern](#rotate_pattern), [trim_pattern](#trim_pattern) |
| Edit and combine | [join_arrays](#join_arrays), [copy_element](#copy_element), [set_element_pos](#set_element_pos) |
| Check, store, show | [is_valid](#is_valid), [qdant_write](#qdant_write), [export_pattern_obj](#export_pattern_obj) |

Arrays are stored in the [QDANT file format](../../formats/qdant.md).

---
<a id="arrayant"></a>
## arrayant
Class for storing and manipulating an array of antenna or acoustic elements

Not in: MATLAB, Python

- All functions give the same results for an empty `element_pos` and empty port weights as for zeros and identity,
  except the IRS array of [get_channels_irs](channel-generation-functions.md#get_channels_irs), whose port weights hold the IRS weights and must be given
- In a multi-frequency vector all entries use the same form, either empty or given (see [is_valid](#is_valid))

### Attributes:
| Attribute                         | Size                                   | Description                                                    |
| --------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `arma::Cube<dtype> e_theta_re`    | `[n_elevation, n_azimuth, n_elements]` | e-theta (vertical) field, real part                            |
| `arma::Cube<dtype> e_theta_im`    | `[n_elevation, n_azimuth, n_elements]` | e-theta (vertical) field, imaginary part                       |
| `arma::Cube<dtype> e_phi_re`      | `[n_elevation, n_azimuth, n_elements]` | e-phi (horizontal) field, real part                            |
| `arma::Cube<dtype> e_phi_im`      | `[n_elevation, n_azimuth, n_elements]` | e-phi (horizontal) field, imaginary part                       |
| `arma::Col<dtype> azimuth_grid`   | `[n_azimuth]`                          | Azimuth angles in rad, in [-pi, pi], strictly increasing       |
| `arma::Col<dtype> elevation_grid` | `[n_elevation]`                        | Elevation angles in rad, in [-pi/2, pi/2], strictly increasing |
| `arma::Mat<dtype> element_pos`    | `[3, n_elements]` or empty             | Element positions relative to the phase center, in m           |
| `arma::Mat<dtype> port_weights_re`    | `[n_elements, n_ports]` or empty       | Port weights, real part                                     |
| `arma::Mat<dtype> port_weights_im`    | `[n_elements, n_ports]` or empty       | Port weights, imaginary part                                |
| `dtype center_freq`          | scalar                                 | Center frequency in Hz; default 299792458                      |
| `std::string name`                | string                                 | Name of the antenna array object                               |

### Member functions:
| Function                                                  | Description                                                                                                                                                         |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.n_elevation()`                                          | Number of elevation angles                                                                                                                                          |
| `.n_azimuth()`                                            | Number of azimuth angles                                                                                                                                            |
| `.n_elements()`                                           | Number of antenna elements                                                                                                                                          |
| `.n_ports()`                                              | Number of ports (columns of port weights)                                                                                                                        |
| `.reset()`                                                | Empties all data; `center_freq` returns to its default, `name` is kept                                                                                              |
| `.set_size(n_elevation, n_azimuth, n_elements, n_ports)` | Resizes all members (`arma::uword` sizes); no-op if they match; after a resize `element_pos` is zero, `port_weights_re` identity, `port_weights_im` zero, field data undefined |
| `.is_valid()`                                             | `std::string`: empty if the object is valid, else an error message                                                                                                  |

### Notes:
- **Value range**: field values and port weights must be finite with magnitudes below about 1e18; larger values overflow
  in the interpolation and in the channel functions and give NaN results
- **Validation**: `is_valid()` always checks the data: sizes of all members, ranges and order of the angle grids, sizes
  of element positions and port weights. A passing check marks the object as validated: the library functions then skip
  the full check as long as no data member was reallocated or resized; values changed in place after the check are not
  seen by them
- **Threads and copies**: the mark is updated atomically, so `is_valid()` may be called while other threads read the
  object; a copy is marked if its source passes the full check, a moved-to object takes over the mark of its source

### See also:
- [is_valid](#is_valid) (checks a multi-frequency vector)
- [generate_antenna](#generate_antenna), [generate_speaker](#generate_speaker) (create arrays)
- [qdant_read](#qdant_read), [qdant_write](#qdant_write) (load and store arrays)

---
<a id="apply_port_weights"></a>
## apply_port_weights
Apply the port weights of an antenna array: one pattern per port

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#apply_port_weights) · [Python](../python/antennas-transducers-and-patterns.md#apply_port_weights)

- Combines the elements of an array into one pattern per port, for beamforming and to speed up MIMO channel computation
- Each port pattern is the weighted sum of the element fields from the section overview, evaluated on the output grid
  and including the phase of each element position
- The result has one element per port and identity port weights; its elements are at the origin, or at the centers
  of their ports with `center_ports`

### Declarations:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::apply_port_weights(
    const arrayant<dtype> &ant,
    const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(),
    double propagation_speed = 299792458.0,
    bool center_ports = false);

std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::apply_port_weights(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &freq_new = arma::Col<dtype>(),
    double propagation_speed = 299792458.0,
    bool center_ports = false);
```

### Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of arrayant objects that passes [is_valid](#is_valid)
- **`azimuth_grid_new`** *(optional)* — Output azimuth grid in rad; empty: the input grid (of the first entry)
- **`elevation_grid_new`** *(optional)* — Output elevation grid in rad; empty: the input grid (of the first entry)
- **`freq_new`** *(optional)* — Vector form: output frequencies in Hz, positive and finite; empty: the `center_freq` of
  the entries, one output per input entry; `[n_freq_new]`
- **`propagation_speed`** *(optional)* — Wave propagation speed $c$ in m/s, positive and finite; the default is the
  speed of light for radio waves, use about 343 for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))
- **`center_ports`** *(optional)* — If `true`, places each port $p$ at the power-weighted center of its elements,
  $\sum_i \lvert w_{i,p} \rvert^2 x_i / \sum_i \lvert w_{i,p} \rvert^2$, with the phase referenced there; a port with
  zero weights stays at the origin. A plane-wave channel is unchanged, and the slower phase over angle makes
  interpolation of coupled sub-arrays more accurate

### Returns:
- New `arrayant` with `n_ports` elements, each holding the pattern of that port; `name` is kept, and so is `center_freq`
  in the single form; vector form: `n_freq_new` such entries, each with its output frequency as `center_freq`

### Notes:
- **Frequency interpolation** (vector form): the fields are interpolated over frequency as in [interpolate_pattern](#interpolate_pattern),
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

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#calc_beamwidth) · [Python](../python/antennas-transducers-and-patterns.md#calc_beamwidth)

- Measures the width of the main lobe of each element in azimuth and elevation at a threshold below its peak, and the
  direction of the peak
- Calculated per element, not per port: the port weights are ignored

### Declarations:
```cpp
void quadriga_lib::calc_beamwidth(
    const arrayant<dtype> &ant,
    const uint_list &i_element = {},
    double threshold_dB = 3.0,
    arma::Col<dtype> *beamwidth_az_deg = nullptr,
    arma::Col<dtype> *beamwidth_el_deg = nullptr,
    arma::Col<dtype> *az_point_ang_deg = nullptr,
    arma::Col<dtype> *el_point_ang_deg = nullptr);

void quadriga_lib::calc_beamwidth(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const uint_list &i_element = {},
    double threshold_dB = 3.0,
    arma::Mat<dtype> *beamwidth_az_deg = nullptr,
    arma::Mat<dtype> *beamwidth_el_deg = nullptr,
    arma::Mat<dtype> *az_point_ang_deg = nullptr,
    arma::Mat<dtype> *el_point_ang_deg = nullptr);
```

### Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of `n_freq` arrayant objects that passes [is_valid](#is_valid); column
  `i_freq` of each output holds the results of `arrayant_vec[i_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`
- **`threshold_dB`** *(optional)* — Threshold below the peak in dB, positive and finite; 3 dB is the half-power
  beamwidth; loudspeakers state the coverage angle between the -6 dB points, use 6

### Outputs:
- **`beamwidth_az_deg`** *(optional)* — Azimuth beamwidth in degrees; `[n_out]`, vector form `[n_out, n_freq]`
- **`beamwidth_el_deg`** *(optional)* — Elevation beamwidth in degrees; `[n_out]`, vector form `[n_out, n_freq]`
- **`az_point_ang_deg`** *(optional)* — Azimuth pointing angle of the main beam in degrees; `[n_out]`, vector form
  `[n_out, n_freq]`
- **`el_point_ang_deg`** *(optional)* — Elevation pointing angle of the main beam in degrees; `[n_out]`, vector form
  `[n_out, n_freq]`

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("custom", 1.0, NAN, {30.0, 20.0});
arma::vec bw_az, bw_el;
quadriga_lib::calc_beamwidth(ant, {}, 3.0, &bw_az, &bw_el); // about 30 and 20 degrees
```

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

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#calc_directivity) · [Python](../python/antennas-transducers-and-patterns.md#calc_directivity)

- Measures how strongly an element concentrates its power: $D = 10 \log_{10}(P_\mathrm{max} / \bar{P})$ with the peak
  radiation intensity $P_\mathrm{max}$ and its mean $\bar{P}$ over the sphere; an isotropic radiator has 0 dBi
- Calculated per element, not per port: the port weights are ignored

### Declarations:
```cpp
arma::Col<dtype> quadriga_lib::calc_directivity(
    const arrayant<dtype> &ant,
    const uint_list &i_element = {});

arma::Mat<dtype> quadriga_lib::calc_directivity(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const uint_list &i_element = {});
```

### Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of `n_freq` arrayant objects that passes [is_valid](#is_valid); column
  `i_freq` of the output holds the results of `arrayant_vec[i_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`

### Returns:
- Directivity in dBi; `[n_out]`, vector form `[n_out, n_freq]`

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("dipole");
arma::vec directivity_dBi = quadriga_lib::calc_directivity(ant); // about 1.76 dBi
```

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

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#copy_element) · [Python](../python/antennas-transducers-and-patterns.md#copy_element)

- Copies the pattern and the position of one element in place, e.g. to build an array from a single element; the port
  weights of existing elements do not change
- A destination beyond the last element enlarges the array. Each added element gets its own new port with weight 1;
  added elements that are not a destination have a zero pattern and position

### Declarations:
```cpp
void quadriga_lib::copy_element(
    arrayant<dtype> &ant,
    arma::uword i_source,
    const uint_list &i_target);

void quadriga_lib::copy_element(
    std::vector<arrayant<dtype>> &arrayant_vec,
    arma::uword i_source,
    const uint_list &i_target);
```

### Inputs:
- **`ant`** — Valid arrayant; updated in place
- **`arrayant_vec`** — Vector form: non-empty vector of valid arrayant objects; the same copy is made in every entry;
  updated in place
- **`i_source`** — Index of the element to copy, 0-based; must be below `n_elements`
- **`i_target`** — Destination indices, 0-based, not empty; a single index is passed as `{i}`; the largest `arma::uword`
  (e.g. `-1` cast to unsigned) throws

### Example:
```cpp
auto ant = quadriga_lib::generate_antenna<double>("dipole");
quadriga_lib::copy_element(ant, 0, {1, 2, 3}); // 4 identical elements
ant.element_pos.row(1) = arma::rowvec({-0.75, -0.25, 0.25, 0.75});
```

### Notes:
- **Empty members**: when the array grows, an empty `element_pos` and empty port weights are first written out as zeros
  and identity, so that all elements can be edited afterward; otherwise they stay empty

### See also:
- [set_element_pos](#set_element_pos) (sets the positions of the copies in all entries of a vector)
- [join_arrays](#join_arrays) (joins two arrays)

---
<a id="export_pattern_obj"></a>
## export_pattern_obj
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#export_pattern_obj) · [Python](../python/antennas-transducers-and-patterns.md#export_pattern_obj)

- Writes the power pattern of each element as a colored 3D surface, one object per element, placed at its position in
  `element_pos`
- The distance of the surface from the element is linear in the level in dB: `object_radius` at the peak over all
  exported elements, 0 at `directivity_range_dB` below it. The color shows the same level

### Declaration:
```cpp
void quadriga_lib::export_pattern_obj(
    const std::string &fn,
    const arrayant<dtype> &ant,
    double directivity_range_dB = 30.0,
    const std::string &colormap = "jet",
    double object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    const uint_list &i_element = {});
```

### Inputs:
- **`fn`** — Output OBJ file name, must end in `.obj`; the colormap is written to a `.mtl` file of the same name;
  existing files are overwritten
- **`ant`** — Valid arrayant; the exported elements must have finite field values and must not be zero in all directions
- **`directivity_range_dB`** *(optional)* — Range of the plot below the peak in dB, finite and not 0; the sign is
  ignored
- **`colormap`** *(optional)* — Colormap name; see [colormap](site-specific-simulation-tools.md#colormap) for the supported maps
- **`object_radius`** *(optional)* — Radius of the surface at the peak in m, positive and finite
- **`icosphere_n_div`** *(optional)* — Subdivisions of the [icosphere](site-specific-simulation-tools.md#icosphere) that carries the surface; higher = finer mesh
  with $20 n^2$ faces
- **`i_element`** *(optional)* — Element indices to export, 0-based; empty: all elements

### See also:
- [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) (the same pattern as numbers)
- [colormap](site-specific-simulation-tools.md#colormap) (the available color scales)
- [channel_export_obj_file](channel-functions.md#channel_export_obj_file) (exports propagation paths to an OBJ file)

---
<a id="generate_antenna"></a>
## generate_antenna
Generate an antenna array of a given type

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#generate_antenna) · [Python](../python/antennas-transducers-and-patterns.md#generate_antenna)

- Creates every built-in antenna: single elements, uniform linear arrays, 3GPP TR 38.901 panel arrays, multibeam
  arrays, and the handheld UT and CPE devices. `type` selects the antenna; the other inputs set its parameters
- The result is an [arrayant](#arrayant) with one pattern per element, the element positions and the port weights. Ports stay
  separate from the elements unless `apply_weights` is set; [apply_port_weights](#apply_port_weights) combines them later
- Panel inputs use the TR notation: `P` polarizations, `M` rows, `N` columns, `Mg` and `Ng` panels, and `M_P`, `N_P`
  ports per panel and polarization
- Tuple inputs may be partial: entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their
  default, e.g. `panel = {4, 8}` is one panel of 4 rows and 8 columns
- An input that the selected type does not take throws

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_antenna(
    const std::string &type,
    double res = NAN,
    double center_freq = 299792458.0,
    const arma::vec &element = {},
    const uint_list &panel = {},
    const arma::vec &spacing = {},
    const uint_list &pol = {},
    const uint_list &ports = {},
    const arma::mat &steer = {},
    const arma::mat &beams = {},
    const arma::vec &device_size = {},
    const uint_list &locations = {},
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
   | `multibeam`                  | Panel array with beams toward given directions                             | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
   | `3GPP-UT` (or `3gpp-ut`)     | Handheld UT of 3GPP TR 38.901 clause 7.3                                   | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
   | `3GPP-CPE` (or `3gpp-cpe`)   | CPE of 3GPP TR 38.901 clause 7.3                                           | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; the grid has
  `floor(360 / res)` azimuth and `floor(180 / res)` elevation steps; NaN or <= 0: 10° for `omni` and `xpol`, 1°
  otherwise; ignored if `pattern` is given
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite; NaN: 299792458 (wavelength 1 m)
- **`element`** *(optional)* — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern
    as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no
    taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in
    dBi (finite or NaN); default `[65, 65, 30, 30, 8]`; not together with `pattern`
  - `3GPP-UT`: the same tuple, default `[125, 125, 22.5, 22.5, 5.3]` (Table 7.3-2); `3GPP-CPE`: default
    `[90, 90, 25, 25, 5]`; `[Inf, Inf, NaN, NaN, 0]` is isotropic
- **`panel`** *(optional)* — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per
  row; default 1; at most 1e7 elements per polarization (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal
  row)
- **`spacing`** *(optional)* — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing,
  vertical and horizontal panel spacing; finite and >= 0 (0 places elements or panels at the same position); default
  `[0.5, d_V, M·d_V, N·d_H]`
- **`pol`** *(optional)* — `[polarization, model]`. Polarization: 1 = one element (ζ = 0°, vertical), 2 = two elements
  (ζ = 0° and 90°), 3 = two elements (ζ = +45° and -45°). Model of TR 38.901 clause 7.3.2: 1 = the element pattern is
  turned by ζ about its boresight (Model-1, eq. 7.3-3), 2 = the power pattern is kept and the field is split by ζ
  (Model-2, eq. 7.3-4, 7.3-5). Default `[1, 1]`; for `3GPP-UT` with polarization 2 or 3 the default model is 2, which
  gives the reference fields of the handheld model (eq. 7.3-6).
- **`ports`** *(optional)* — `[M_P, N_P]`: ports (transceiver units, TXRUs) per column and per row of a panel and
  polarization; must divide `M` and `N`; default `[M, N]` (one port per element)
- **`steer`** *(optional)* — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt and pan of the sub-array weights;
  `[1, 2]` for all panels or `[Mg·Ng, 2]` with one row per panel (panel order `mg`, `ng`); one column: pan 0; default 0;
  a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`
- **`beams`** *(optional)* — `[2, n_beams]` or `[3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear;
  2 rows: 1); finite; default `[0; 0; 1]`
- **`device_size`** *(optional)* — `[X, Y, Z]` in m (depth, width, height); NaN or left off: the default; `3GPP-UT`
  `[0.15, 0.07, 0]` with X, Y > 0 and Z = 0, `3GPP-CPE` `[0, 0.2, 0.2]` with X = 0 and Y, Z >= 0
- **`locations`** *(optional)* — TR location labels (`3GPP-UT` 1 to 8, `3GPP-CPE` 1 to 9), not 0-based indices; any
  order, repeats allowed; default all
- **`pattern`** *(optional)* — Element to use instead of the default element: its field patterns (`e_theta_re`,
  `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, port
  weights and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of
  `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; empty (no elements): the
  element of the type; for `3GPP-UT`, a `pattern` on an irregular grid is first resampled to a regular one
- **`separate_beams`** *(optional)* — If `true`, one port per beam (weights ignored)
- **`apply_weights`** *(optional)* — If `true`, combine each port into one pattern at its sub-array center

### Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna array object; panel arrays have `P·M·N·Mg·Ng` elements and `P·M_P·N_P·Mg·Ng`
  ports (`ula`, `3GPP`); with `apply_weights`, one element per port

### Example:
```cpp
// Vertically polarized omni antenna, 1 degree grid; custom element with 30 x 20 degree beamwidth at 299792458 Hz
auto omni = quadriga_lib::generate_antenna("omni", 1.0);
auto beam = quadriga_lib::generate_antenna("custom", 1.0, NAN, {30.0, 20.0});

// Table 7.8-2, BS config 1: two 4 x 4 panels 2.5 wavelengths apart, +/-45 deg Model-2, one port per polarization
// and panel, 12 degree electrical downtilt
auto bs = quadriga_lib::generate_antenna("3GPP", 1.0, 6.0e9, {}, {4, 4, 1, 2}, {0.5, 0.5, 2.5, 2.5}, {3, 2}, {1, 1}, {12.0, 0.0});

// Handheld UT at the locations (1, 7, 3, 5) of Table 7.8-2A, Config B
auto ut = quadriga_lib::generate_antenna("3GPP-UT", 1.0, 7.0e9, {}, {}, {}, {}, {}, {}, {}, {}, {1, 7, 3, 5});
```

### Notes:
- **Element tuple** (`3GPP`, `3GPP-UT`, `3GPP-CPE`): entries left off at the end or set to NaN take their default, where
  `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `3GPP` with `[90]` gives `[90, 90, 30, 30, 8]`);
  `G_max` left off takes the default of the type, NaN sets the peak gain to the directivity
- **Element and port order** (panel arrays): polarization fastest, then row `m` (upward along z), column `n` (along
  +y), panel row `mg`, panel column `ng`, centered on the origin; ports in the same order with `m_p`, `n_p`
- **Sub-array weights** (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds $K = M / M_P$ by
  $L = N / N_P$ elements. With tilt $\theta_t$ and pan $\varphi_p$, element $(k, l)$ gets

  $$w_{k,l} = \frac{1}{\sqrt{K L}} \exp\left( j 2\pi \left[ \left(k - \tfrac{K-1}{2}\right) d_V \sin\theta_t - \left(l - \tfrac{L-1}{2}\right) d_H \sin\varphi_p \right] \right)$$

  A positive tilt points the beam down, a positive pan toward +y
- **Polarization** (TR 38.901 clause 7.3.2): `pol` 2 has the slants $\zeta = 0°$ and $90°$, `pol` 3 $\zeta = +45°$ and
  $-45°$, element 0 first. Model-1 turns the element by $\zeta$ about its boresight (eq. 7.3-3). Model-2 keeps the power
  pattern $A' = \lvert F_\theta \rvert^2 + \lvert F_\phi \rvert^2$ and sets $F_\theta = \sqrt{A'} \cos\zeta$,
  $F_\phi = \sqrt{A'} \sin\zeta$ (eq. 7.3-4, 7.3-5)
- **`multibeam`**: maximum-ratio-transmission weights for the directions in `beams`, exact for one beam and approximate
  for several; only the ratios of the beam weights matter. One port per polarization, with `separate_beams` one per
  polarization and beam
- **`3GPP-UT`** (TR 38.901 clause 7.3): `P` elements at each entry of `locations` (Fig. 7.3-2), polarization fastest.
  Each element is rolled about its boresight by 90° (45° for two field patterns: `pol` 2 or a two-element `pattern`)
  and turned to the bearing $\operatorname{atan2}(y, x)$ of its location (Tables A.6-1, A.6-2). `pol` 1 and 2 with
  their default model are the TR model (eq. 7.3-6 to 7.3-8), other values are extensions. For an element symmetric
  under a 180° roll (every `element` tuple), the other z'' direction of Fig. 7.3-7 is the same antenna with a negated
  port weight
- **`3GPP-CPE`**: `P` elements at each entry of `locations` (Fig. 7.3-6 seen from the front, in the y-z plane), facing
  +x. TR 38.901 defines no CPE element; the default is the UE element of TR 38.803 Table 5.2.3.3-1

### See also:
- [arrayant](#arrayant) (the returned data structure)
- [apply_port_weights](#apply_port_weights) (combines the ports into patterns)
- [rotate_pattern](#rotate_pattern) (turns an antenna)
- [calc_directivity](#calc_directivity), [calc_beamwidth](#calc_beamwidth) (check the generated patterns)
- [qdant_write](#qdant_write) (stores the antenna in a file)

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#generate_speaker) · [Python](../python/antennas-transducers-and-patterns.md#generate_speaker)

- Creates one [arrayant](#arrayant) per frequency with a single element at the origin. Its real-valued pattern in `e_theta_re` is
  the product of the driver directivity, the enclosure radiation and the bandpass frequency response; a negative value
  is a 180° phase inversion (piston side lobes, dipole rear)
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining the results with
  [join_arrays](#join_arrays) and [set_element_pos](#set_element_pos); the crossover follows from the overlapping bandpass responses

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
- **`driver_type`** *(optional)* — Driver directivity model: `"piston"` (circular piston in a baffle, narrows with
  increasing frequency), `"horn"` (set coverage, toward omni at low frequencies) or `"omni"` (independent of frequency)
- **`radius`** *(optional)* — Effective radiating radius in m, > 0; cone or dome radius for `"piston"`, mouth radius for
  `"horn"`
- **`lower_cutoff`** *(optional)* — Lower -3 dB frequency of the bandpass in Hz, > 0
- **`upper_cutoff`** *(optional)* — Upper -3 dB frequency of the bandpass in Hz, larger than `lower_cutoff`
- **`lower_rolloff_slope_dB`** *(optional)* — Low-frequency rolloff in dB per octave, > 0; 12 is a second-order
  Butterworth filter
- **`upper_rolloff_slope_dB`** *(optional)* — High-frequency rolloff in dB per octave, > 0
- **`sensitivity_dB`** *(optional)* — On-axis sensitivity in dB SPL at 1 W and 1 m, finite; 85 dB gives unity amplitude
- **`radiation_type`** *(optional)* — Enclosure radiation model: `"monopole"` (no change), `"hemisphere"` (sealed box
  with baffle-step transition), `"dipole"` (figure-8, rear hemisphere inverted) or `"cardioid"`
- **`hor_coverage_deg`** *(optional)* — `"horn"`: horizontal coverage in degrees, below 180, between the -6 dB points at
  full pattern control; NaN or <= 0: 90°
- **`ver_coverage_deg`** *(optional)* — `"horn"`: vertical coverage in degrees, below 180, as above; NaN or <= 0: 60°
- **`horn_control_freq`** *(optional)* — `"horn"`: frequency in Hz above which the horn controls its pattern; NaN or <=
  0: $c / (2\pi a)$ with the speed of sound $c$ and $a$ = `radius`
- **`baffle_width`, `baffle_height`** *(optional)* — Baffle size in m, > 0; used by `"hemisphere"`
- **`center_freq`** *(optional)* — Frequency samples in Hz, positive and finite; `[n_freq]`; empty: third-octave bands
  `1000 · 2^(n/3)` Hz, those enclosing the cutoffs plus one more on each side, limited to 20 to 20000 Hz
- **`angular_resolution_deg`** *(optional)* — Resolution of the azimuth and elevation grid in degrees; 0.1 to 90; a
  value r gives `floor(360 / r)` azimuth and `floor(180 / r)` elevation steps

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One arrayant per frequency, with `center_freq` set to that frequency

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

### Notes:
- **Frequency response**: a Butterworth-style bandpass, -3 dB at the cutoffs, scaled by the sensitivity $S$ =
  `sensitivity_dB` relative to 85 dB SPL:

  $$H(f) = \frac{10^{(S - 85) / 20}}{\sqrt{1 + (f_l / f)^{2 n_l}} \, \sqrt{1 + (f / f_u)^{2 n_u}}}$$

  with $f_l$ = `lower_cutoff`, $f_u$ = `upper_cutoff` and $n_l$, $n_u$ the rolloff slopes in dB per octave divided by 6
- **Driver directivity** $D$, with the angle $\alpha$ from the forward axis +x ($\cos\alpha = \cos\theta \cos\phi$), the
  wavenumber $k = 2\pi f / c$, the speed of sound $c$ = 343 m/s and $a$ = `radius`:
  - `piston`: $D = 2 J_1(x) / x$ with $x = k a \sin\alpha$, rotationally symmetric about +x
  - `horn`: $D = 1 - b + b \cos^{m_h}\phi \, \cos^{m_v}\theta$ in the front half ($\lvert\phi\rvert < 90°$) and
    $D = 1 - b$ behind it, with $b = f^2 / (f^2 + f_c^2)$ and $f_c$ = `horn_control_freq`. The exponents follow from the
    coverage angle $\Theta$ by $\cos^m(\Theta / 2) = 1/2$
  - `omni`: $D = 1$
- **Enclosure radiation** $R$:
  - `monopole`: $R = 1$
  - `hemisphere`: $R = 1 - b + b (1 + \cos\alpha) / 2$ with $b = f^2 / (f^2 + f_b^2)$ and the baffle-step frequency
    $f_b = c / (\pi \sqrt{W H})$, $W$ = `baffle_width`, $H$ = `baffle_height`; omnidirectional at low and cardioid-like
    at high frequencies
  - `dipole`: $R = \cos\alpha$, negative in the rear hemisphere
  - `cardioid`: $R = (1 + \cos\alpha) / 2$

### See also:
- [join_arrays](#join_arrays), [set_element_pos](#set_element_pos) (build multi-driver systems)
- [interpolate_pattern](#interpolate_pattern) (evaluates the model at any direction and frequency)
- [calc_beamwidth](#calc_beamwidth) (measures the coverage angle, with `threshold_dB = 6`)
- [qdant_write](#qdant_write) (stores the model in a file)

---
<a id="interpolate_pattern"></a>
## interpolate_pattern
Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#interpolate_pattern) · [Python](../python/antennas-transducers-and-patterns.md#interpolate_pattern)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the array and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation)
- Vector form: for a multi-frequency array, the two entries whose center frequencies bracket each target frequency are
  interpolated and blended

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
    const uint_list &i_element = {},
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
    const uint_list &i_element = {},
    const arma::Cube<dtype> &orientation = arma::Cube<dtype>(),
    const arma::Mat<dtype> &element_pos = arma::Mat<dtype>(),
    arma::Cube<std::complex<dtype>> *V = nullptr,
    arma::Cube<std::complex<dtype>> *H = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`ant`** — Antenna array with `n_elements` elements
- **`arrayant_vec`** — Vector form: multi-frequency array that passes [is_valid](#is_valid), not empty; entries need not be
  sorted by frequency
- **`azimuth`** — Azimuth angles in GCS in rad; one row: the same angles for all outputs (planar wave), `n_out` rows:
  one angle per output (spherical wave); `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS in rad; same size as `azimuth`
- **`freq_new`** *(optional)* — Vector form: target frequencies in Hz, finite; empty: the center frequencies of the
  entries in vector order; ascending or descending order is fastest; `[n_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements in order
  (`n_out = n_elements`); `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) of the array, for all outputs and angles, per
  output, per angle, or both, applied at all frequencies; empty: no rotation; `[3, 1, 1]`, `[3, n_out, 1]`,
  `[3, 1, n_ang]` or `[3, n_out, n_ang]`
- **`element_pos`** *(optional)* — Element positions in local antenna coordinates, used for `dist`; empty: the columns
  `i_element` of `ant.element_pos`, zeros if that is empty; vector form: checked for its size only, since no output
  depends on the positions; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (throws if unavailable or if the pattern has
  more than `INT_MAX` samples); other values throw

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`, vector form
  `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang]`, vector form
  `[n_out, n_ang, n_freq]`
- **`dist`** *(optional)* — Distance of each element from the wavefront plane through the origin of the array; negative
  for elements closer to the source; `[n_out, n_ang]` (single form only)
- **`azimuth_loc`** / **`elevation_loc`** *(optional)* — Azimuth and elevation in the local antenna frame;
  `[n_out, n_ang]` (single form only)
- **`gamma`** *(optional)* — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]` (single
  form only)
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

### Notes:
- **Interpolation**: linear where neighboring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- **Frequency interpolation** (vector form): spherical interpolation of the complex values over frequency, linear where
  the two values have nearly opposite phases or one of them is zero. Target frequencies outside the range of the center
  frequencies take the nearest entry (no extrapolation). Consecutive target frequencies with the same bracketing entries
  reuse the spatial results
- **Angles**: not range-checked; values outside $[-\pi, \pi]$ and $[-\pi/2, \pi/2]$ denote the same direction; an
  elevation beyond $\pm\pi/2$ crosses the pole, so e-theta and e-phi have the opposite sign of the same direction given
  in range. NaN or infinite angles give NaN in all outputs of that angle and output, without an exception. `n_ang = 0`
  gives outputs of size `[n_out, 0]` (vector form `[n_out, 0, n_freq]`)
- **Poles**: on a pole of the local frame (a direction along the z-axis of the oriented array) the local azimuth is
  chosen so that the local and the global theta unit vectors agree: the polarization is not rotated there
- **Outputs** are resized only when their size differs, so pre-allocated outputs are written in place
- **Kernels**: GENERIC computes in the precision of `dtype`; AVX2 in single precision also for `double`, with errors of
  up to about 2e-6 of the peak field value and 2e-6 rad in the angles, and near a pole of the local frame up to about
  2e-7 / d rad at a distance d (in rad). Auto currently selects GENERIC
- **Exceptions**: `std::invalid_argument` for an invalid `ant`, sizes that do not match, element indices
  `>= n_elements`, an unknown or unavailable kernel, and outputs that are an input, a data member of `ant` or another
  output, or share memory with one; vector form also for an empty `arrayant_vec` and target frequencies that are not
  finite

### See also:
- [apply_port_weights](#apply_port_weights) (patterns per port instead of per element)
- [rotate_pattern](#rotate_pattern) (applies an orientation to the stored patterns)
- [join_arrays](#join_arrays), [generate_speaker](#generate_speaker) (build multi-element, multi-frequency arrays)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (channel coefficients from the interpolated patterns)

---
<a id="is_valid"></a>
## is_valid
Validate a vector of arrayant objects for multi-frequency consistency

Not in: MATLAB, Python

- Checks that the entries can be used together as one multi-frequency array and stops at the first error
- Each entry gets the full check of its `is_valid` member, which marks a passing entry as validated. It is then compared
  with the first entry: size and values of the angle grids, number of elements, element positions, size of the port
  weights
- All entries must use the same form for `element_pos`, `port_weights_re` and `port_weights_im`: all empty or all given;
  a mix of empty members and explicit zeros or identity is rejected
- Pattern data, `center_freq` and port weight values are not compared, since they vary with frequency

### Declaration:
```cpp
std::string quadriga_lib::is_valid(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Input:
- **`arrayant_vec`** — Vector of arrayant objects to validate; an empty vector is reported as invalid

### Returns:
- Empty string if valid; otherwise a message that names the failing entry and property, such as
  `"Entry 3: Azimuth grid values do not match the first entry."`

### See also:
- [arrayant](#arrayant) (its `is_valid()` member checks a single object)
- [set_element_pos](#set_element_pos) (keeps the positions of all entries equal)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="join_arrays"></a>
## join_arrays
Join the elements of two antenna arrays into a new array

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#join_arrays) · [Python](../python/antennas-transducers-and-patterns.md#join_arrays)

- Combines two arrays with identical sampling grids, e.g. the drivers of a loudspeaker; `name` and `center_freq` are
  taken from the first array
- The ports stay separate: elements of the first array connect only to its ports, and likewise for the second:
   | Element \ Port        | P1…Pp1 (first) | Pp1+1…Pp1+p2 (second) |
   | --------------------- | :------------: | :-------------------: |
   | E1…En1 (first)        |    W1 block    |           0           |
   | En1+1…En1+n2 (second) |       0        |       W2 block        |

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
- **`ant_1`** / **`arrayant_vec1`** — First array; its elements come first in the output; vector form: must pass
  [is_valid](#is_valid)
- **`ant_2`** / **`arrayant_vec2`** — Second array with the same sampling grids; vector form: must pass [is_valid](#is_valid),
  with as many entries as the first and matching `center_freq` values at each index

### Returns:
- New `arrayant` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports; vector form: one such entry per
  frequency

### See also:
- [set_element_pos](#set_element_pos) (positions the drivers of a multi-frequency array)
- [copy_element](#copy_element) (duplicates an element within one array)
- [generate_speaker](#generate_speaker) (creates the driver models to join)
- [qdant_write](#qdant_write) (stores the joined model)

---
<a id="qdant_read"></a>
## qdant_read
Read arrayant objects from a QDANT file

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#qdant_read) · [Python](../python/antennas-transducers-and-patterns.md#qdant_read)

- Loads arrays from a file in the [QDANT format](../../formats/qdant.md). A file can hold several arrays, each with a
  numeric ID, e.g. one per frequency of a frequency-dependent model, where `center_freq` of each entry identifies the
  frequency

### Declaration:
```cpp
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read(
        const std::string &fn,
        const uint_list &ids = {},
        arma::u32_mat *layout = nullptr);
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`ids`** *(optional)* — 1-based IDs of the entries to read, duplicates allowed; empty: the entries listed in the
  layout, each once, in order of first appearance (column-major scan), or all entries if the file has no layout

### Output:
- **`layout`** *(optional)* — The file's layout matrix of entry IDs (0 = empty slot)

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per requested ID, in the order of `ids`

### Notes:
- **IDs**: the number used in the layout matrix and by [qdant_write](#qdant_write). A first entry without an `id` attribute (files
  from other tools) counts as ID 1
- **Exceptions**: `std::invalid_argument` if an ID is 0 or the file has no entry with it; `std::runtime_error` naming
  the file for a missing, unreadable or malformed file
- **Elements without data**: elements without pattern nodes read as zeros. A file whose element count is not backed by
  its data, or that exceeds the [reader limits](../../formats/qdant.md), throws

### See also:
- [qdant_write](#qdant_write) (writes such files)
- [generate_antenna](#generate_antenna), [generate_speaker](#generate_speaker) (create arrays without a file)

---
<a id="qdant_write"></a>
## qdant_write
Write arrayant data to a QDANT (XML) file

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#qdant_write) · [Python](../python/antennas-transducers-and-patterns.md#qdant_write)

- Stores an array in a file in the [QDANT format](../../formats/qdant.md). A file can hold several arrays with distinct IDs
  and a layout matrix that arranges them
- Vector form: stores a frequency-dependent model in a new file, each entry holding the pattern at its `center_freq`

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
- **`fn`** — Output QDANT file name, must not be empty; an existing file is updated; vector form: must not exist (throws
  `std::runtime_error` and leaves the file unchanged)
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of valid arrayant objects, written with the sequential IDs
  `1, 2, ..., n_entries` and a `[n_entries, 1]` layout of these IDs
- **`id`** *(optional)* — Target ID in the file; an entry with this ID is replaced; `0` appends with
  `max_existing_id + 1`, or 1 in a new file
- **`layout`** *(optional)* — Matrix of entry IDs that arranges the file; every value must be an ID in the file (0 is
  not accepted); empty: the layout in the file is replaced by a `[1, n]` row of all IDs in file order, so a custom
  layout must be passed again with every write that should keep it

### Returns:
- ID assigned to the written antenna within the file (single form)

### Notes:
- **Pattern accuracy**: the file has single-precision-equivalent accuracy relative to the element peak, also for
  double-precision data. Pattern magnitudes (in dB) and phases (in degrees) get the fewest decimals (0 to 6) that keep
  the error of each complex sample at or below 6e-8 times the peak amplitude of its element (over `e_theta` and
  `e_phi`). Samples near the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per
  20 dB below the peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak.
  An element at a low level is quantized relative to its own peak
- **Other values**: element positions and port weights (magnitude, phase) are written as the shortest text that reads
  back as the same float (at most 9 significant digits); sampling grids are rounded to 0.001 degree
- **Empty members**: an empty `element_pos` or empty port weights are not written; [qdant_read](#qdant_read) returns zeros and
  identity for them. Above 4096 elements the identity port weights are written, because the reader requires them in the
  file
- **Zeros**: all non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element
  that is zero in both field components is written as an all `-inf` `e_theta_mag` node

### See also:
- [qdant_read](#qdant_read) (reads the file back)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---
<a id="rotate_pattern"></a>
## rotate_pattern
Rotate the patterns of an antenna array using Euler rotations, in place

Also in: [MATLAB](../mex/antennas-transducers-and-patterns.md#rotate_pattern) · [Python](../python/antennas-transducers-and-patterns.md#rotate_pattern)

- Turns the patterns, the polarization and the element positions of an array, e.g. for a mechanical downtilt. The
  rotated pattern equals, at the grid points, what [interpolate_pattern](#interpolate_pattern) returns for the unrotated pattern with the
  same `orientation`
- Port weights, `center_freq` and `name` are kept

### Declarations:
```cpp
void quadriga_lib::rotate_pattern(
    arrayant<dtype> &ant,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const uint_list &i_element = {});

void quadriga_lib::rotate_pattern(
    std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const uint_list &i_element = {});
```

### Inputs:
- **`ant`** — Valid arrayant; updated in place
- **`arrayant_vec`** — Vector form: non-empty vector of arrayant objects; updated in place; the grid is never adjusted,
  so that all entries keep the same grid
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; a positive tilt turns the main beam down, as in
  3GPP TR 38.901; `[3, 1]` or `[1, 3]`
- **`usage`** *(optional)* — Rotation mode; vector form: 0 to 2, where 0 and 1 act as 3 and 4. In the single form, modes
  0 and 1 adjust the sampling grid of non-uniformly sampled patterns, which needs all elements: a subset in `i_element`
  then throws. Element positions turn in all modes except 2. For scalar acoustic fields (pressure in `e_theta_re` only),
  use 1 to avoid spurious polarization effects:
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
- [interpolate_pattern](#interpolate_pattern) (evaluates a pattern for an orientation without changing it)
- [calc_rotation_matrix](math-functions.md#calc_rotation_matrix) (rotation matrix of the orientation)
- [generate_antenna](#generate_antenna) (electrical tilt through the port weights)
- [join_arrays](#join_arrays) (joins arrays that are then rotated together)

---
<a id="set_element_pos"></a>
## set_element_pos
Set element positions for all entries in a multi-frequency arrayant vector

Not in: MATLAB, Python

- Writes the same positions to every entry in place, since [is_valid](#is_valid) requires equal positions in all entries

### Declaration:
```cpp
void quadriga_lib::set_element_pos(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        const uint_list &i_element = {});
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects with equal element counts; updated in place; an empty
  `element_pos` of an entry is filled with zeros first
- **`element_pos`** — New (x, y, z) positions in m; `[3, n_update]` with `n_update` the length of `i_element`, or
  `n_elements` if `i_element` is empty
- **`i_element`** *(optional)* — Indices of the elements to update, 0-based; empty: all elements

### See also:
- [copy_element](#copy_element) (adds elements before their positions are set)
- [join_arrays](#join_arrays) (joins arrays, e.g. the drivers of a loudspeaker)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="trim_pattern"></a>
## trim_pattern
Remove zero-valued entries from antenna pattern data, reducing its size

Not in: MATLAB, Python

- Removes the azimuth columns and elevation rows without power from the sampling grid and the patterns, in place
- Positions, port weights, name and frequency are unchanged

### Declaration:
```cpp
void quadriga_lib::trim_pattern(arrayant<dtype> &ant);
```

### Input:
- **`ant`** — Valid arrayant; updated in place; to keep the original, call it on a copy (`auto b = ant;`)

### Notes:
- **Threshold**: a column or row is removed if its total power, summed over all field components and elements, does not
  exceed $10^{-12}$ times the peak power of a single sample (summed over field components and elements). The result does
  not depend on the scale of the pattern; an all-zero pattern is left unchanged
- **Neighbors**: the immediate neighbors of retained entries are also kept, unless the local grid spacing is strongly
  asymmetric (by a factor above 1.5)

### See also:
- [rotate_pattern](#rotate_pattern) (trims the grid it adjusts)
- [qdant_write](#qdant_write) (stores the reduced pattern)
