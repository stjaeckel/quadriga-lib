# Array antenna functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-array-antenna-functions

| Function | Summary |
| --- | --- |
| [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) | Calculate the beamwidth and pointing angles of array antenna elements in degrees |
| [arrayant_calc_beamwidth_multi](#arrayant_calc_beamwidth_multi) | Calculate the beamwidth and pointing angles in degrees for each entry of a multi-frequency arrayant vector |
| [arrayant_calc_directivity](#arrayant_calc_directivity) | Calculate the directivity in dBi of array antenna elements |
| [arrayant_calc_directivity_multi](#arrayant_calc_directivity_multi) | Calculate the directivity in dBi for each entry of a multi-frequency arrayant vector |
| [arrayant_combine_pattern](#arrayant_combine_pattern) | Combine element patterns, positions, and coupling weights into effective radiation patterns |
| [arrayant_combine_pattern_multi](#arrayant_combine_pattern_multi) | Combine element patterns, positions, and coupling weights into effective radiation patterns (multi-frequency) |
| [arrayant_concat](#arrayant_concat) | Concatenate the elements of two arrayants into a new arrayant |
| [arrayant_concat_multi](#arrayant_concat_multi) | Concatenate two multi-frequency arrayant vectors into a single multi-element model |
| [arrayant_copy_element](#arrayant_copy_element) | Copy a single antenna element to one or more destination slots |
| [arrayant_copy_element_multi](#arrayant_copy_element_multi) | Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector |
| [arrayant_export_obj_file](#arrayant_export_obj_file) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [arrayant_generate](#arrayant_generate) | Generate an array antenna of a given type |
| [arrayant_interpolate](#arrayant_interpolate) | Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles |
| [arrayant_interpolate_multi](#arrayant_interpolate_multi) | Interpolate a multi-frequency array antenna at given angles and frequencies |
| [arrayant_is_valid_multi](#arrayant_is_valid_multi) | Validate a vector of arrayant objects for multi-frequency consistency |
| [arrayant_remove_zeros](#arrayant_remove_zeros) | Remove zero-valued entries from antenna pattern data, reducing its size |
| [arrayant_rotate_pattern](#arrayant_rotate_pattern) | Rotate antenna radiation patterns using Euler rotations, in place |
| [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) | Apply Euler rotations to all entries in a multi-frequency arrayant vector, in place |
| [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) | Set element positions for all entries in a multi-frequency arrayant vector |
| [generate_arrayant_3GPP](#generate_arrayant_3GPP) | Generate a TR 38.901 panel array antenna |
| [generate_arrayant_custom](#generate_arrayant_custom) | Generate an antenna with custom 3dB beamwidth |
| [generate_arrayant_dipole](#generate_arrayant_dipole) | Generate a short dipole antenna with vertical polarization |
| [generate_arrayant_half_wave_dipole](#generate_arrayant_half_wave_dipole) | Generate a half-wave dipole antenna with vertical polarization |
| [generate_arrayant_multibeam](#generate_arrayant_multibeam) | Generate a planar multi-element antenna array with multiple beam directions |
| [generate_arrayant_omni](#generate_arrayant_omni) | Generate an isotropic radiator with vertical polarization |
| [generate_arrayant_ula](#generate_arrayant_ula) | Generate a uniform linear array (ULA) |
| [generate_arrayant_xpol](#generate_arrayant_xpol) | Generate a cross-polarized isotropic radiator |
| [generate_speaker](#generate_speaker) | Generate a parametric frequency-dependent loudspeaker directivity model |
| [qdant_read](#qdant_read) | Read an arrayant object from a QDANT file |
| [qdant_read_multi](#qdant_read_multi) | Read all arrayant objects from a QDANT file into a vector |
| [qdant_write](#qdant_write) | Write arrayant data to a QDANT (XML) file |
| [qdant_write_multi](#qdant_write_multi) | Write a vector of arrayant objects to a single QDANT file |

---
<a id="arrayant_calc_beamwidth"></a>
## arrayant_calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_beamwidth) · [Python](../python/array-antenna-functions.md#calc_beamwidth)

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in each direction than the antenna sampling grid)
- The width is that of the main lobe: starting at the maximum, the search goes outwards until the power drops below the
  threshold; side lobes above the threshold are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360° when
  the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°
- Calculated per element, not per port; ignores element coupling

### Declaration:
```cpp
void quadriga_lib::arrayant_calc_beamwidth(
    const arrayant<dtype> &ant,
    const arma::uvec &i_element = arma::uvec(),
    double threshold_dB = 3.0,
    arma::Col<dtype> *beamwidth_az_deg = nullptr,
    arma::Col<dtype> *beamwidth_el_deg = nullptr,
    arma::Col<dtype> *az_point_ang_deg = nullptr,
    arma::Col<dtype> *el_point_ang_deg = nullptr);
```

### Inputs:
- **`ant`** — Valid arrayant
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`
- **`threshold_dB`** *(optional)* — Threshold in dB, positive and finite; 3 dB = FWHM

### Outputs:
- **`beamwidth_az_deg`** *(optional)* — Azimuth beamwidth in degree; `[n_out]`
- **`beamwidth_el_deg`** *(optional)* — Elevation beamwidth in degree; `[n_out]`
- **`az_point_ang_deg`** *(optional)* — Azimuth pointing angle of the main beam in degree; `[n_out]`
- **`el_point_ang_deg`** *(optional)* — Elevation pointing angle of the main beam in degree; `[n_out]`

### Example:
```cpp
auto ant = quadriga_lib::generate_arrayant_custom<double>(30.0, 20.0);
arma::vec bw_az, bw_el;
quadriga_lib::arrayant_calc_beamwidth(ant, {}, 3.0, &bw_az, &bw_el); // bw_az(0) = 30, bw_el(0) = 20
```

### See also:
- [arrayant_calc_beamwidth_multi](#arrayant_calc_beamwidth_multi) (multi-freq counterpart)
- [arrayant_calc_directivity](#arrayant_calc_directivity) (directivity in dBi of array antenna elements)
- [arrayant_combine_pattern](#arrayant_combine_pattern) (apply element coupling before calculating the beamwidth)

---
<a id="arrayant_calc_beamwidth_multi"></a>
## arrayant_calc_beamwidth_multi
Calculate the beamwidth and pointing angles in degrees for each entry of a multi-frequency arrayant vector

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_beamwidth) · [Python](../python/array-antenna-functions.md#calc_beamwidth)

- Same computation as [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) on every entry; column `i_freq` of each output holds the results
  of `arrayant_vec[i_freq]`
- Calculated per element, not per port; ignores element coupling

### Declaration:
```cpp
void quadriga_lib::arrayant_calc_beamwidth_multi(
        const std::vector<arrayant<dtype>> &arrayant_vec,
        const arma::uvec &i_element = arma::uvec(),
        double threshold_dB = 3.0,
        arma::Mat<dtype> *beamwidth_az_deg = nullptr,
        arma::Mat<dtype> *beamwidth_el_deg = nullptr,
        arma::Mat<dtype> *az_point_ang_deg = nullptr,
        arma::Mat<dtype> *el_point_ang_deg = nullptr);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of `n_freq` arrayant objects that passes [arrayant_is_valid_multi](#arrayant_is_valid_multi)
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`
- **`threshold_dB`** *(optional)* — Threshold in dB, positive and finite; 3 dB = FWHM

### Outputs:
- **`beamwidth_az_deg`** *(optional)* — Azimuth beamwidth in degree; `[n_out, n_freq]`
- **`beamwidth_el_deg`** *(optional)* — Elevation beamwidth in degree; `[n_out, n_freq]`
- **`az_point_ang_deg`** *(optional)* — Azimuth pointing angle of the main beam in degree; `[n_out, n_freq]`
- **`el_point_ang_deg`** *(optional)* — Elevation pointing angle of the main beam in degree; `[n_out, n_freq]`

### See also:
- [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) (single-frequency counterpart)
- [arrayant_calc_directivity_multi](#arrayant_calc_directivity_multi) (directivity of multi-frequency arrayant vectors)

---
<a id="arrayant_calc_directivity"></a>
## arrayant_calc_directivity
Calculate the directivity in dBi of array antenna elements

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_directivity) · [Python](../python/array-antenna-functions.md#calc_directivity)

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Calculated per element, not per port; ignores element coupling

### Declaration:
```cpp
arma::Col<dtype> quadriga_lib::arrayant_calc_directivity(
    const arrayant<dtype> &ant,
    const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`ant`** — Valid arrayant
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`

### Returns:
- Directivity in dBi; `[n_out]`

### Example:
```cpp
auto ant = quadriga_lib::generate_arrayant_dipole<double>();
arma::vec directivity_dBi = quadriga_lib::arrayant_calc_directivity(ant); // about 1.76 dBi
```

### See also:
- [arrayant_calc_directivity_multi](#arrayant_calc_directivity_multi) (multi-freq counterpart)
- [arrayant_calc_beamwidth](#arrayant_calc_beamwidth) (beamwidth and pointing angles of array antenna elements)
- [arrayant_combine_pattern](#arrayant_combine_pattern) (apply element coupling before calculating the directivity)

---
<a id="arrayant_calc_directivity_multi"></a>
## arrayant_calc_directivity_multi
Calculate the directivity in dBi for each entry of a multi-frequency arrayant vector

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_directivity) · [Python](../python/array-antenna-functions.md#calc_directivity)

- Same computation as [arrayant_calc_directivity](#arrayant_calc_directivity) on every entry; column `i_freq` of the output holds the results
  of `arrayant_vec[i_freq]`
- Calculated per element, not per port; ignores element coupling

### Declaration:
```cpp
arma::Mat<dtype> quadriga_lib::arrayant_calc_directivity_multi(
        const std::vector<arrayant<dtype>> &arrayant_vec,
        const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of `n_freq` arrayant objects that passes [arrayant_is_valid_multi](#arrayant_is_valid_multi)
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements; `[n_out]`

### Returns:
- Directivity in dBi; `[n_out, n_freq]`

### See also:
- [arrayant_calc_directivity](#arrayant_calc_directivity) (single-frequency counterpart)
- [arrayant_calc_beamwidth_multi](#arrayant_calc_beamwidth_multi) (beamwidth of multi-frequency arrayant vectors)

---
<a id="arrayant_combine_pattern"></a>
## arrayant_combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_combine_pattern) · [Python](../python/array-antenna-functions.md#combine_pattern)

- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` to produce one output element per port (column) of the coupling matrix
- The element positions add the phase `-2 pi f d / propagation_speed` with `f = center_freq` and `d` the distance
  of the element from the wavefront plane through the origin; the default speed of light applies to radio waves, use
  about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))
- Empty `element_pos` counts as zeros, empty coupling as identity
- Useful for beamforming and MIMO channel computation speedup

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_combine_pattern(
    const arrayant<dtype> &ant,
    const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(),
    double propagation_speed = 299792458.0);
```

### Inputs:
- **`ant`** — Valid arrayant
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

### Returns:
- New `arrayant` with `n_ports` elements (= number of columns in `coupling_re/im`), each holding the combined effective pattern for that port

### See also:
- [arrayant_combine_pattern_multi](#arrayant_combine_pattern_multi) (multi-freq counterpart)
- [arrayant_interpolate](#arrayant_interpolate) (used internally to compute effective radiation patterns)
- [arrayant_rotate_pattern](#arrayant_rotate_pattern) (useful for orienting array antenna patterns)

---
<a id="arrayant_combine_pattern_multi"></a>
## arrayant_combine_pattern_multi
Combine element patterns, positions, and coupling weights into effective radiation patterns (multi-frequency)

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_combine_pattern) · [Python](../python/array-antenna-functions.md#combine_pattern)

- Multi-frequency counterpart to [arrayant_combine_pattern](#arrayant_combine_pattern)
- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` across all entries to produce one output element per port (column of the coupling matrix) at each requested output frequency
- Output length = `freq_new.n_elem` if provided; otherwise one entry per input arrayant
- Field interpolation across frequency is delegated to [arrayant_interpolate_multi](#arrayant_interpolate_multi) (SLERP with linear-interpolation fallback)
- Coupling matrices are SLERP-interpolated between bracketing input entries; out-of-range output frequencies are clamped to the nearest input entry
- Each output arrayant has identity coupling and zero element positions (patterns are pre-combined)
- The element positions add the phase `-2 pi f d / propagation_speed` at each output frequency `f`; the default speed
  of light applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Declaration:
```cpp
std::vector<arrayant<dtype>> arrayant_combine_pattern_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> &azimuth_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &elevation_grid_new = arma::Col<dtype>(),
    const arma::Col<dtype> &freq_new = arma::Col<dtype>(),
    double propagation_speed = 299792458.0);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects (must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi))
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to `arrayant_vec[0].azimuth_grid`
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to `arrayant_vec[0].elevation_grid`
- **`freq_new`** *(optional)* — Alternative frequency grid in Hz, positive and finite; defaults to per-entry
  `center_freq`
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

### Returns:
- Vector of arrayant objects (length = `n_freq_new`), each with `n_ports` elements equal to the number of columns in the coupling matrix

### See also:
- [arrayant_combine_pattern](#arrayant_combine_pattern) (single-frequency counterpart)
- [arrayant_interpolate_multi](#arrayant_interpolate_multi) (used internally for spatial+frequency field interpolation)
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (input validation)

---
<a id="arrayant_concat"></a>
## arrayant_concat
Concatenate the elements of two arrayants into a new arrayant

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_concat) · [Python](../python/array-antenna-functions.md#concat)

- Both arrays must share identical sampling grids; throws otherwise
- Coupling is block-diagonal (see [arrayant_concat_multi](#arrayant_concat_multi) diagram); name and center_freq are taken from `ant_1`
- An empty `element_pos` counts as zeros, an empty `coupling_re` as identity (one port per element), an empty
  `coupling_im` as zeros

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_concat(
    const arrayant<dtype> &ant_1,
    const arrayant<dtype> &ant_2);
```

### Inputs:
- **`ant_1`** — First arrayant; its elements come first in the output
- **`ant_2`** — Second arrayant; sampling grids must match `ant_1`

### Returns:
- New `arrayant` containing all elements of both arrays

### See also:
- [arrayant_concat_multi](#arrayant_concat_multi) (multi-freq counterpart)

---
<a id="arrayant_concat_multi"></a>
## arrayant_concat_multi
Concatenate two multi-frequency arrayant vectors into a single multi-element model

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_concat) · [Python](../python/array-antenna-functions.md#concat)

- Both inputs must have equal entry counts, identical angular grids, and matching `center_freq` values at each index.
- Per frequency entry: pattern cubes are joined along the element (slice) dimension; `element_pos` matrices are horizontally concatenated (empty positions treated as zeros).
- Both inputs are validated with [arrayant_is_valid_multi](#arrayant_is_valid_multi) before processing; each output entry is validated before returning.
- Output inherits name, azimuth/elevation grids, and `center_freq` from `arrayant_vec1`.

### Declaration:
```cpp
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::arrayant_concat_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec1,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec2);
```

### Inputs:
- **`arrayant_vec1`** — First validated, mutually consistent arrayant vector
- **`arrayant_vec2`** — Second arrayant vector; must match entry count, grids, and center frequencies of `arrayant_vec1`

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports per entry
- Coupling matrices are assembled block-diagonally — elements from `vec1` connect only to ports from `vec1` and vice versa:
   | Element \ Port      | P1…Pp1 (vec1) | Pp1+1…Pp1+p2 (vec2) |
   | ------------------- | :-----------: | :-----------------: |
   | E1…En1 (vec1)       |   C1 block    |          0          |
   | En1+1…En1+n2 (vec2) |       0       |      C2 block       |

### See also:
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (validation called on both inputs)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (position drivers before concatenating)
- [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) (rotate elements after concatenating)
- [qdant_write_multi](#qdant_write_multi) (persist the combined model)

---
<a id="arrayant_copy_element"></a>
## arrayant_copy_element
Copy a single antenna element to one or more destination slots

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_copy_element) · [Python](../python/array-antenna-functions.md#copy_element)

- Works in place; the array is enlarged if any destination index exceeds the current number of elements
- Added elements get their own port with identity coupling; positions of added elements that are not copied are zero
- When the array grows, an empty `element_pos` and an empty coupling are written out first (zeros, identity), so
  that all elements can be edited afterwards; otherwise empty members stay empty
- Throws `std::invalid_argument` for an empty `i_target`, a source index out of bound, or a destination index
  equal to the largest `arma::uword` (e.g. `-1` cast to unsigned)

### Declaration:
```cpp
void quadriga_lib::arrayant_copy_element(
    arrayant<dtype> &ant,
    arma::uword i_source,
    const arma::uvec &i_target);
```

### Inputs:
- **`ant`** — Valid arrayant; updated in-place
- **`i_source`** — Index of the element to copy, 0-based
- **`i_target`** — Target indices, 0-based; a single index is passed as `{i}`

### Example:
```cpp
auto ant = quadriga_lib::generate_arrayant_dipole<double>();
quadriga_lib::arrayant_copy_element(ant, 0, {1, 2, 3}); // 4 identical elements
ant.element_pos.row(1) = arma::rowvec({-0.75, -0.25, 0.25, 0.75});
```

### See also:
- [arrayant_copy_element_multi](#arrayant_copy_element_multi) (multi-freq counterpart)

---
<a id="arrayant_copy_element_multi"></a>
## arrayant_copy_element_multi
Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_copy_element) · [Python](../python/array-antenna-functions.md#copy_element)

- Calls [arrayant_copy_element](#arrayant_copy_element) on every entry in the vector with the same source and destination indices.
- If any destination index exceeds the current element count, all entries are enlarged; new elements receive an identity coupling entry.
- Source and destination indices are 0-based.

### Declaration:
```cpp
void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword i_source,
        const arma::uvec &i_target);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects; modified in-place
- **`i_source`** — Index of the element to copy from; must be within current element count
- **`i_target`** — Indices of target elements; enlarges all entries if any index exceeds current count; a single index is passed as `{i}`

### Example:
```cpp
arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
auto driver = quadriga_lib::generate_speaker<double>(
    "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
    0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
quadriga_lib::arrayant_copy_element_multi(driver, 0, {1, 2, 3});
```

### See also:
- [arrayant_copy_element](#arrayant_copy_element) (per-entry operation called internally)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions after copying)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multiple arrayant vectors)

---
<a id="arrayant_export_obj_file"></a>
## arrayant_export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_export_obj_file) · [Python](../python/array-antenna-functions.md#export_obj_file)

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives finer mesh
- Each element is placed at its position in `element_pos` (empty: at the origin)

### Declaration:
```cpp
void quadriga_lib::arrayant_export_obj_file(
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
<a id="arrayant_generate"></a>
## arrayant_generate
Generate an array antenna of a given type

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- One function for all generators, with the TR 38.901 parameter tuples of [generate_arrayant_3GPP](#generate_arrayant_3GPP) (fill rules, units, ranges, element and port order there); the MATLAB / Octave and Python `arrayant_generate` wrap it.
- An input that a type does not use throws, also `separate_beams` or `apply_weights` set to `true`. The result has the given `center_freq`.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant_generate(
    const std::string &type,
    double res = NAN,
    double center_freq = 299792458.0,
    const arma::uvec &panel = {},
    const arma::vec &spacing = {},
    const arma::uvec &ports = {},
    const arma::uvec &pol = {},
    const arma::mat &steer = {},
    const arma::vec &element = {},
    const quadriga_lib::arrayant<dtype> &pattern = quadriga_lib::arrayant<dtype>(),
    const arma::mat &beams = {},
    bool separate_beams = false,
    bool apply_weights = false);
```

### Inputs:
- **`type`** — Antenna type and the inputs it takes besides `res` and `center_freq`:
   | `type`                                         | Generator                                  | Inputs                                                                                 |
   | ---------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
   | `omni`, `xpol`                                 | [generate_arrayant_omni](#generate_arrayant_omni), [generate_arrayant_xpol](#generate_arrayant_xpol) | none                                                                       |
   | `dipole` (or `short-dipole`), `half-wave-dipole` | [generate_arrayant_dipole](#generate_arrayant_dipole), [generate_arrayant_half_wave_dipole](#generate_arrayant_half_wave_dipole) | none                                                 |
   | `custom`                                       | [generate_arrayant_custom](#generate_arrayant_custom)               | `element`                                                                              |
   | `ula`                                          | [generate_arrayant_3GPP](#generate_arrayant_3GPP) with the omni element | `panel`, `spacing`, `ports`, `pol`, `steer`, `pattern`, `apply_weights`        |
   | `3GPP` (or `3gpp`)                             | [generate_arrayant_3GPP](#generate_arrayant_3GPP)                 | `panel`, `spacing`, `ports`, `pol`, `steer`, `element`, `pattern`, `apply_weights` |
   | `multibeam`                                    | [generate_arrayant_multibeam](#generate_arrayant_multibeam)            | `panel`, `spacing`, `pol`, `element`, `beams`, `separate_beams`, `apply_weights`    |
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 10° for `omni` and `xpol`, 1° otherwise
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite
- **`panel`** *(optional)* — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; default 1 (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** *(optional)* — `[d_V, d_H, d_gV, d_gH]` in wavelengths; default `[0.5, d_V, M d_V, N d_H]`
- **`ports`** *(optional)* — `[M_P, N_P]`: ports per column and per row of a panel and polarization; default `[M, N]`
- **`pol`** *(optional)* — `[polarization, model]`: 1 vertical, 2 H/V, 3 ±45°; model 1 or 2; default `[1, 1]`
- **`steer`** *(optional)* — `[tilt, pan]` in degrees, one row for all panels or one row per panel; default 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`
- **`element`** *(optional)* — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi; default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity
- **`pattern`** *(optional)* — Element to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; empty (no elements): the element of the type
- **`beams`** *(optional)* — `[2 or 3, n_beams]`: azimuth in degrees; elevation in degrees; weight (2 rows: 1); default `[0; 0; 1]`
- **`separate_beams`** *(optional)* — If `true`, one port per beam
- **`apply_weights`** *(optional)* — If `true`, combine each port into one pattern

### Returns:
- `quadriga_lib::arrayant<dtype>` — Array antenna object

### Example:
```cpp
// Table 7.8-6A, config 1: 24 x 32 elements, d_V = 0.7, d_H = 0.5 wavelengths, 8 x 32 ports per polarization, +/-45 deg Model-2
auto ant = quadriga_lib::arrayant_generate("3GPP", 5.0, 7.0e9, {24, 32}, {0.7, 0.5}, {8, 32}, {3, 2}, {12.0});
```

### See also:
- [generate_arrayant_3GPP](#generate_arrayant_3GPP) (tuples, element and port order)
- [generate_arrayant_multibeam](#generate_arrayant_multibeam) (beam weights)

---
<a id="arrayant_interpolate"></a>
## arrayant_interpolate
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_interpolate) · [Python](../python/array-antenna-functions.md#interpolate)

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation),
  with linear interpolation where neighbouring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- `azimuth` with one row gives the same angles for all outputs (planar wave), `n_out` rows one angle per output
  (spherical wave); `n_ang = 0` gives outputs of size `[n_out, 0]`
- Angles are not range-checked; values outside [-pi, pi] and [-pi/2, pi/2] denote the corresponding direction. NaN or
  infinite angles give NaN in all outputs of that angle and output, without an exception
- Outputs are resized only when their size differs, so pre-allocated outputs are written in place
- On a pole of the local frame (a direction along the z-axis of the oriented antenna) the local azimuth is chosen so
  that the local and the global theta unit vectors agree: the polarization is not rotated there
- Kernels: GENERIC computes in the precision of `dtype`; AVX2 computes in single precision also for `double`, with
  errors of up to about 2e-6 of the peak field value and 2e-6 rad in the angles; at a distance d (in rad) from a pole
  of the local frame up to about 2e-7 / d rad in the local azimuth and the polarization angle and 1e-7 / d rad (at most
  4e-4 rad) in the local elevation; auto selects GENERIC in this version, which may change in a later version
- Throws `std::invalid_argument` for an invalid `ant`, sizes that do not match, element indices `>= n_elements`, an
  unknown or unavailable kernel, and outputs that are an input, a data member of `ant` or another output, or share
  memory with one

### Declaration:
```cpp
void quadriga_lib::arrayant_interpolate(
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
```

### Inputs:
- **`ant`** — Array antenna with `n_elements` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements in
  order (`n_out = n_elements`); `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) of the antenna, for all outputs and angles,
  per output, per angle, or both; empty: no rotation; `[3, 1, 1]`, `[3, n_out, 1]`, `[3, 1, n_ang]` or
  `[3, n_out, n_ang]`
- **`element_pos`** *(optional)* — Element positions in local antenna coordinates, used for `dist`; empty: the
  columns `i_element` of `ant.element_pos`, zeros if that is empty; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (throws if unavailable or if the pattern
  has more than `INT_MAX` samples); other values throw

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang]`
- **`dist`** *(optional)* — Distance of each element from the wavefront plane through the origin of the antenna;
  negative for elements closer to the source; `[n_out, n_ang]`
- **`azimuth_loc`** / **`elevation_loc`** *(optional)* — Azimuth and elevation in the local antenna frame;
  `[n_out, n_ang]`
- **`gamma`** *(optional)* — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`
- **`V`** / **`H`** *(optional)* — e-theta / e-phi field as complex values, replacing `V_re`, `V_im` / `H_re`, `H_im`
  (then left empty); `[n_out, n_ang]`

### Example:
```cpp
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
arma::mat azimuth = {0.0, 0.5 * arma::datum::pi, -0.5 * arma::datum::pi, arma::datum::pi};
arma::mat elevation(1, azimuth.n_elem, arma::fill::zeros);
arma::mat V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate(ant, azimuth, elevation, V_re, V_im, H_re, H_im);
```

### See also:
- [arrayant_interpolate_multi](#arrayant_interpolate_multi) (frequency-dependent array antennas)
- [arrayant](array-antenna-class.md#arrayant) (array antenna class)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (channel coefficients from the interpolated patterns)

---
<a id="arrayant_interpolate_multi"></a>
## arrayant_interpolate_multi
Interpolate a multi-frequency array antenna at given angles and frequencies

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_interpolate) · [Python](../python/array-antenna-functions.md#interpolate)

- For each target frequency, interpolates the two entries whose center frequencies bracket it, each as
  [arrayant_interpolate](#arrayant_interpolate), and blends their fields: spherical interpolation of the complex values over frequency,
  linear where the two values have nearly opposite phases or one of them is zero
- Target frequencies outside the range of the center frequencies take the nearest entry (no extrapolation)
- Consecutive target frequencies with the same bracketing entries reuse the spatial results; target frequencies in
  ascending or descending order are fastest
- The entries must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi): same grids, number of elements and element positions
- Angles, element indices, orientation, kernels and the complex outputs as in [arrayant_interpolate](#arrayant_interpolate); `n_ang = 0`
  gives outputs of size `[n_out, 0, n_freq]`
- Throws `std::invalid_argument` for an empty or invalid `arrayant_vec`, target frequencies that are not finite, and
  for the errors of [arrayant_interpolate](#arrayant_interpolate)

### Declaration:
```cpp
void quadriga_lib::arrayant_interpolate_multi(
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
- **`arrayant_vec`** — Multi-frequency array antenna, one entry per center frequency; entries need not be sorted by
  frequency
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`freq_new`** *(optional)* — Target frequencies in Hz; empty: the center frequencies of the entries
  in vector order; `[n_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; empty: all elements;
  `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading), applied at all frequencies; shapes as in
  [arrayant_interpolate](#arrayant_interpolate)
- **`element_pos`** *(optional)* — Element positions in m; checked for its size only, since no output depends on the
  positions; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2; as in [arrayant_interpolate](#arrayant_interpolate)

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang, n_freq]`
- **`V`** / **`H`** *(optional)* — e-theta / e-phi field as complex values, replacing `V_re`, `V_im` / `H_re`, `H_im`
  (then left empty); `[n_out, n_ang, n_freq]`

### Example:
```cpp
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
arma::mat el(1, 4, arma::fill::zeros);
arma::vec freq = {250.0, 1500.0, 8000.0};
arma::cube V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate_multi(speaker, az, el, V_re, V_im, H_re, H_im, freq);
```

### See also:
- [arrayant_interpolate](#arrayant_interpolate) (single-frequency interpolation)
- [arrayant_concat_multi](#arrayant_concat_multi) (build multi-element, multi-frequency models)
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (validation of the entries)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency array antennas)

---
<a id="arrayant_is_valid_multi"></a>
## arrayant_is_valid_multi
Validate a vector of arrayant objects for multi-frequency consistency

Not in: MATLAB, Python

- Each entry is validated individually via its `is_valid` member (full check; a passing entry is marked as validated).
- Cross-entry checks (all vs. the first entry): azimuth/elevation grid sizes and values, number of elements, element positions, coupling_re shape, and coupling_im presence and size.
- Pattern data, `center_freq`, and coupling matrix values are not compared (expected to vary).
- All entries must use the same form for `element_pos`, `coupling_re` and `coupling_im`: all empty or all given; a
  mix of empty members and explicit zeros / identity is rejected.
- Stops at first error and returns a message identifying the failing entry and property.

### Declaration:
```cpp
std::string quadriga_lib::arrayant_is_valid_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Input:
- **`arrayant_vec`** — Non-empty vector of arrayant objects to validate

### Returns:
- Empty string if valid; otherwise a message such as `"Entry 3: Azimuth grid values do not match the first entry."`

### See also:
- [.is_valid](array-antenna-class.md#.is_valid) (per-entry validation called internally)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="arrayant_remove_zeros"></a>
## arrayant_remove_zeros
Remove zero-valued entries from antenna pattern data, reducing its size

Not in: MATLAB, Python

- Works in place on the sampling grid and the pattern cubes; positions, coupling, name and frequency are unchanged
- A column or row is removed if the total power summed over all field components and array elements does not exceed
  1e-12 times the peak power of a single sample (summed over field components and elements); the result does not
  depend on the scale of the pattern; an all-zero pattern is left unchanged
- Immediate neighbors of retained entries are also kept unless the local grid spacing is strongly asymmetric (factor > 1.5)
- To keep the original, call it on a copy (`auto b = ant;`)

### Declaration:
```cpp
void quadriga_lib::arrayant_remove_zeros(arrayant<dtype> &ant);
```

### Input:
- **`ant`** — Valid arrayant; updated in-place

---
<a id="arrayant_rotate_pattern"></a>
## arrayant_rotate_pattern
Rotate antenna radiation patterns using Euler rotations, in place

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_rotate_pattern) · [Python](../python/array-antenna-functions.md#rotate_pattern)

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [arrayant_interpolate](#arrayant_interpolate) returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); coupling, `center_freq` and `name` are kept
- Usage 0 and 1 adjust the sampling grid for non-uniformly sampled antennas; this needs all elements, so a subset
  in `i_element` throws when the grid must be adjusted
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects

### Declaration:
```cpp
void quadriga_lib::arrayant_rotate_pattern(
    arrayant<dtype> &ant,
    const arma::Mat<dtype> &orientation,
    unsigned usage = 0,
    const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`ant`** — Valid arrayant; updated in place
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
auto ant = quadriga_lib::generate_arrayant_3GPP<double>({4});
arma::vec downtilt = {0.0, 10.0 * arma::datum::pi / 180.0, 0.0}; // 10 degree mechanical downtilt
quadriga_lib::arrayant_rotate_pattern(ant, downtilt);
```

### See also:
- [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) (multi-freq counterpart)
- [arrayant_interpolate](#arrayant_interpolate) (same orientation convention)
- [calc_rotation_matrix](math-functions.md#calc_rotation_matrix) (rotation matrix of the orientation)

---
<a id="arrayant_rotate_pattern_multi"></a>
## arrayant_rotate_pattern_multi
Apply Euler rotations to all entries in a multi-frequency arrayant vector, in place

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_rotate_pattern) · [Python](../python/array-antenna-functions.md#rotate_pattern)

- Calls [arrayant_rotate_pattern](#arrayant_rotate_pattern) on every entry with grid adjustment always disabled (required for uniform-grid
  consistency across frequencies); orientation convention as there (a positive tilt is a downtilt)
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects

### Declaration:
```cpp
void quadriga_lib::arrayant_rotate_pattern_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &orientation,
        unsigned usage = 0,
        const arma::uvec &i_element = arma::uvec())
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; `[3, 1]`
- **`usage`** *(optional)* — Rotation mode: `0` = pattern + polarization, `1` = pattern only, `2` = polarization only
- **`i_element`** *(optional)* — Element indices to rotate, 0-based, no duplicates; empty: all elements

### See also:
- [arrayant_rotate_pattern](#arrayant_rotate_pattern) (per-entry operation called internally)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multi-frequency vectors before rotating)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions in multi-frequency vectors)

---
<a id="arrayant_set_element_pos_multi"></a>
## arrayant_set_element_pos_multi
Set element positions for all entries in a multi-frequency arrayant vector

Not in: MATLAB, Python

- Updates `element_pos` in-place on every entry in the vector identically.
- If `i_element` is empty, all positions are replaced and `element_pos` must have `n_elements` columns.
- If `i_element` is non-empty, only those indexed columns are updated; `element_pos` column count must match `i_element` length.
- All entries must have the same element count; uninitialized `element_pos` fields are zero-initialized before update.

### Declaration:
```cpp
void quadriga_lib::arrayant_set_element_pos_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        const arma::uvec &i_element = arma::uvec());
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`element_pos`** — New (x, y, z) positions; `[3, n_update]`
- **`i_element`** *(optional)* — Indices of elements to update; empty: all elements are replaced

### See also:
- [arrayant_copy_element_multi](#arrayant_copy_element_multi) (replicate elements before setting positions)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="generate_arrayant_3GPP"></a>
## generate_arrayant_3GPP
Generate a TR 38.901 panel array antenna

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- The uniform rectangular panel array of 3GPP TR 38.901 clause 7.3: `Mg × Ng` panels of `M × N` elements with one or two polarizations (P = 1 or 2); the element is the Table 7.3-1 pattern or the field patterns of `pattern`.
- Elements: polarization fastest, then `m` (upwards along z), `n` (along +y, Fig. 7.3-1), `mg`, `ng`; positions centred on the origin, panel spacings centre to centre. Ports: polarization fastest, then `m_p`, `n_p`, `mg`, `ng`.
- Each port feeds a sub-array (TXRU) of `K = M / M_P` × `L = N / N_P` elements. Eq. 7.3-1 applies per sub-array, vertically with `tilt` (θ_etilt = 90° + `tilt`, a positive tilt points the beam down) and by analogy horizontally with `pan` (a positive pan turns the beam towards +y): element `(k, l)` of a sub-array has the weight `exp(j 2π ((k - (K-1)/2) d_V sin(tilt) - (l - (L-1)/2) d_H sin(pan))) / sqrt(K L)`, phase reference at the sub-array centre.
- By default all elements are kept and the weights form the coupling matrix `[n_elements, n_ports]`; `apply_weights` combines each port into one pattern ([arrayant_combine_pattern](#arrayant_combine_pattern)) placed at its sub-array centre.
- Polarization (clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45° (element 0 first). Model-1 turns the element (pattern and polarization) by ζ about its boresight (eq. 7.3-3); Model-2 sets F_θ = sqrt(A') cos ζ and F_φ = sqrt(A') sin ζ with A' = |F_θ|² + |F_φ|² of the element (eq. 7.3-4, 7.3-5), also for `pol` 1. A `pattern` with more than one element is used as given and needs `pol` 1.
- Tuples: trailing entries may be omitted; 0 (in `panel`, `ports`, `pol`) or NaN (in `spacing`, `steer`, `element`) selects the default of that entry; more entries than listed throw.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(
    const arma::uvec &panel = {},
    double center_freq = 299792458.0,
    const arma::vec &spacing = {},
    const arma::uvec &ports = {},
    const arma::uvec &pol = {},
    const arma::mat &steer = {},
    const arma::vec &element = {},
    const quadriga_lib::arrayant<dtype> &pattern = quadriga_lib::arrayant<dtype>(),
    double res = 1.0,
    bool apply_weights = false);
```

### Inputs:
- **`panel`** *(optional)* — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; default 1; at most 1e7 elements per polarization
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite
- **`spacing`** *(optional)* — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing; finite and >= 0 (0 places elements or panels at the same position); defaults `d_V` 0.5, `d_H = d_V`, `d_gV = M d_V`, `d_gH = N d_H`
- **`ports`** *(optional)* — `[M_P, N_P]`: ports (TXRUs) per column and per row of a panel and polarization; must divide `M` and `N`; default `[M, N]` (one port per element)
- **`pol`** *(optional)* — `[polarization, model]`: polarization 1 (vertical), 2 (H/V) or 3 (±45°), 4 to 6 reserved; model 1 (Model-1) or 2 (Model-2); default `[1, 1]`
- **`steer`** *(optional)* — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt and pan of the sub-array weights; `[1, 2]` for all panels or `[Mg Ng, 2]` with one row per panel (panel order `mg`, `ng`); one column: pan 0; default 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`
- **`element`** *(optional)* — Table 7.3-1 constants `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]`: beamwidths in degrees (> 0, Inf: no taper), SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit), G_max in dBi (finite or NaN); default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity; not together with `pattern`
- **`pattern`** *(optional)* — Element to use instead of the TR element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; empty (no elements): TR element
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°; ignored if `pattern` is provided
- **`apply_weights`** *(optional)* — If `true`, combine each port into one pattern at its sub-array centre

### Returns:
- `quadriga_lib::arrayant<dtype>` — `P M N Mg Ng` elements and `P M_P N_P Mg Ng` ports; with `apply_weights`, one element per port

### Example:
```cpp
// Table 7.8-2, BS config 1: two 4 x 4 panels 2.5 wavelengths apart, +/-45 deg Model-2, one port per polarization
// and panel, 12 degree electrical downtilt
auto ant = quadriga_lib::generate_arrayant_3GPP<double>({4, 4, 1, 2}, 6.0e9, {0.5, 0.5, 2.5, 2.5}, {1, 1}, {3, 2}, {12.0, 0.0});
```

### See also:
- [arrayant_generate](#arrayant_generate) (all generators in one function)
- [generate_arrayant_ula](#generate_arrayant_ula) (linear array with the omni element)
- [arrayant_combine_pattern](#arrayant_combine_pattern) (combines the ports into patterns)

---
<a id="generate_arrayant_custom"></a>
## generate_arrayant_custom
Generate an antenna with custom 3dB beamwidth

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a single-element, vertically polarized antenna with independently configurable azimuth and elevation 3 dB (FWHM) beamwidths, normalized so that its peak gain equals its directivity.
- The power pattern is `rear_gain_lin + (1 - rear_gain_lin) C(az) D(el)` with the azimuth and elevation cuts `C` and `D` (peak 1); `rear_gain_lin` is the floor towards the rear, `0.0` means no rear radiation.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(
    double az_3dB = 90.0,
    double el_3dB = 90.0,
    double rear_gain_lin = 0.0,
    double res = 1.0);
```

### Inputs:
- **`az_3dB`** *(optional)* — Azimuth 3 dB beamwidth in degrees; > 0, Inf: no taper (flat in azimuth); NaN: 90
- **`el_3dB`** *(optional)* — Elevation 3 dB beamwidth in degrees; > 0, Inf: no taper (flat in elevation); NaN: 90
- **`rear_gain_lin`** *(optional)* — Floor of the power pattern relative to its peak, linear; in [0, 1) (1 would be isotropic); NaN: 0
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°

### Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna object with specified beamwidth and rear gain

---
<a id="generate_arrayant_dipole"></a>
## generate_arrayant_dipole
Generate a short dipole antenna with vertical polarization

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a single-element short dipole antenna pattern with vertical polarization.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole(double res = 1.0);
```

### Input:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized short dipole antenna object

---
<a id="generate_arrayant_half_wave_dipole"></a>
## generate_arrayant_half_wave_dipole
Generate a half-wave dipole antenna with vertical polarization

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a single-element half-wave dipole antenna pattern with vertical polarization.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole(double res = 1.0);
```

### Input:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized half-wave dipole antenna object

---
<a id="generate_arrayant_multibeam"></a>
## generate_arrayant_multibeam
Generate a planar multi-element antenna array with multiple beam directions

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns the panel array of [generate_arrayant_3GPP](#generate_arrayant_3GPP) (`panel`, `spacing`, `pol`; element and port order there) with a custom-shape element ([generate_arrayant_custom](#generate_arrayant_custom)) and beamforming weights computed via maximum-ratio transmission (MRT); weights are phasors.
- MRT is optimal for a single beam; approximate when multiple beams are specified. The beam directions are refined so that the gain peaks at the requested angles.
- Weights control relative beam contribution; only their ratios matter, not absolute values.
- Ports: one per polarization; with `separate_beams = true` one per polarization and beam (polarization fastest; weights ignored).
- Tuples: trailing entries may be omitted; 0 (in `panel`, `pol`) or NaN (in `spacing`, `element`) selects the default of that entry.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_multibeam(
    const arma::uvec &panel = {},
    double center_freq = 299792458.0,
    const arma::vec &spacing = {},
    const arma::uvec &pol = {},
    const arma::vec &element = {},
    const arma::mat &beams = {},
    double res = 1.0,
    bool separate_beams = false,
    bool apply_weights = false);
```

### Inputs:
- **`panel`** *(optional)* — `[M, N, Mg, Ng]`; see [generate_arrayant_3GPP](#generate_arrayant_3GPP)
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite
- **`spacing`** *(optional)* — `[d_V, d_H, d_gV, d_gH]` in wavelengths; see [generate_arrayant_3GPP](#generate_arrayant_3GPP)
- **`pol`** *(optional)* — `[polarization, model]`; see [generate_arrayant_3GPP](#generate_arrayant_3GPP)
- **`element`** *(optional)* — Custom-shape element `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper) and the floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[120, 120, Inf]`
- **`beams`** *(optional)* — `[2, n_beams]` or `[3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); finite; default `[0; 0; 1]`
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°
- **`separate_beams`** *(optional)* — If `true`, generate one independent beam per column of `beams`
- **`apply_weights`** *(optional)* — If `true`, combine each port into one pattern ([arrayant_combine_pattern](#arrayant_combine_pattern))

### Returns:
- `quadriga_lib::arrayant<dtype>` — Multibeam planar array antenna object

---
<a id="generate_arrayant_omni"></a>
## generate_arrayant_omni
Generate an isotropic radiator with vertical polarization

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a single-element antenna array with omnidirectional pattern and vertical polarization.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni(double res = 1.0);
```

### Input:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°

### Returns:
- `quadriga_lib::arrayant<dtype>` — Isotropic radiator antenna object

---
<a id="generate_arrayant_ula"></a>
## generate_arrayant_ula
Generate a uniform linear array (ULA)

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns `N` elements along the y-axis (horizontal), centred on the origin: the panel array of [generate_arrayant_3GPP](#generate_arrayant_3GPP) with `panel = {1, N}` and the vertically polarized omni element; [arrayant_generate](#arrayant_generate) with type `ula` takes all panel tuples.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_ula(
    arma::uword N = 1,
    double center_freq = 299792458.0,
    double spacing = 0.5,
    const quadriga_lib::arrayant<dtype> &pattern = quadriga_lib::arrayant<dtype>(),
    double res = 1.0);
```

### Inputs:
- **`N`** *(optional)* — Number of elements; 0: 1
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite
- **`spacing`** *(optional)* — Element spacing in wavelengths; finite and >= 0; NaN: 0.5
- **`pattern`** *(optional)* — Element to use instead of the omni element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position (a pattern with several elements gives a group of elements per position); its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; empty (no elements): omni
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°; ignored if `pattern` is provided

### Returns:
- `quadriga_lib::arrayant<dtype>` — ULA antenna array object

---
<a id="generate_arrayant_xpol"></a>
## generate_arrayant_xpol
Generate a cross-polarized isotropic radiator

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a two-element antenna array with omnidirectional patterns in vertical and horizontal polarization.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol(double res = 1.0);
```

### Input:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 1°

### Returns:
- `quadriga_lib::arrayant<dtype>` — Cross-polarized isotropic radiator antenna object

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [MATLAB](../mex/array-antenna-functions.md#generate_speaker) · [Python](../python/array-antenna-functions.md#generate_speaker)

- Returns one [arrayant](array-antenna-class.md#arrayant) object per frequency sample; each has a single element with the real-valued 
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
<a id="qdant_read"></a>
## qdant_read
Read an arrayant object from a QDANT file

Also in: [MATLAB](../mex/array-antenna-functions.md#qdant_read) · [Python](../python/array-antenna-functions.md#qdant_read)

- Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the arrayant for the given ID.
- Throws `std::invalid_argument` if the file has no entry with the given ID. The first entry of a file is ID 1 only
  if it has no `id` attribute (files written by other tools); otherwise ID 1 must exist.
- A missing, unreadable or malformed file throws `std::runtime_error`. The number of elements (`NoElements`) must be
  backed by data in the file (element positions, coupling, or pattern nodes of the last element). Elements without
  pattern nodes read as zeros; their zero patterns, and the identity coupling of a file without `CouplingAbs`, are
  limited to 2^24 values each (e.g. 256 elements without pattern data on a 1-degree grid, 4096 elements without
  coupling).

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(
        const std::string &fn,
        unsigned id = 1,
        arma::u32_mat *layout = nullptr);
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`id`** *(optional)* — 1-based ID of the antenna entry to read
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix of element IDs

### Returns:
- `quadriga_lib::arrayant<dtype>` constructed from the specified entry in the file

### See also:
- [qdant_write](#qdant_write) (write a single arrayant)
- [qdant_write_multi](#qdant_write_multi) (write multiple arrayants with sequential IDs)

---
<a id="qdant_read_multi"></a>
## qdant_read_multi
Read all arrayant objects from a QDANT file into a vector

Also in: [MATLAB](../mex/array-antenna-functions.md#qdant_read) · [Python](../python/array-antenna-functions.md#qdant_read)

- Reads all entries from a QDANT file by probing ID 1 to obtain the layout, then reading each unique non-zero ID in order of first appearance (column-major scan).
- Each unique ID is read exactly once regardless of how many times it appears in the layout.
- Counterpart to [qdant_write_multi](#qdant_write_multi); primary mechanism for loading frequency-dependent models where `center_freq` on each entry identifies the corresponding frequency.

### Declaration:
```cpp
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read_multi(
        const std::string &fn,
        arma::u32_mat *layout = nullptr);
```

### Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix; non-zero values are entry IDs

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per unique ID, ordered by first appearance in the layout

### See also:
- [qdant_read](#qdant_read) (read a single entry by ID)
- [qdant_write_multi](#qdant_write_multi) (write a vector of arrayants)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---
<a id="qdant_write"></a>
## qdant_write
Write arrayant data to a QDANT (XML) file

Also in: [MATLAB](../mex/array-antenna-functions.md#qdant_write) · [Python](../python/array-antenna-functions.md#qdant_write)

- Multiple antennas can be stored in the same file using distinct `id` values
- If `id = 0` and the file exists, the antenna is appended with `id = max_existing_id + 1`; if file does not exist, `id = 1` gets assigned
- Without `layout` (empty), the layout in the file is replaced by a `[1, n]` row of all IDs in the file
  in file order; a custom layout written before is lost. Pass the layout again with every write that should keep it.
- The file has single-precision-equivalent accuracy relative to the element peak, also for double-precision data:
  pattern magnitudes (in dB) and phases (in degree) get the fewest decimals (0 to 6) that keep the error of each
  complex sample at or below 6e-8 times the peak amplitude of its element (over e_theta and e_phi). Samples near
  the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the
  peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at
  a low level is quantized relative to its own peak. Element positions and coupling (magnitude, phase) are written
  as the shortest text that reads back as the same float (at most 9 significant digits); sampling grids are rounded
  to 0.001 degree
- An empty `element_pos` or coupling is not written; [qdant_read](#qdant_read) returns zeros and identity for it. Above 4096
  elements the identity coupling is written, because the reader requires it in the file
- All non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element that is
  zero in both field components is written as an all `-inf` `EthetaMag` node.

### Declaration:
```cpp
unsigned quadriga_lib::qdant_write(
    const std::string &fn,
    const arrayant<dtype> &ant,
    unsigned id = 0,
    const arma::u32_mat &layout = arma::u32_mat());
```

### Inputs:
- **`fn`** — Output QDANT filename; must not be empty
- **`ant`** — Valid arrayant
- **`id`** *(optional)* — Target ID in file; `0` appends with auto-assigned ID
- **`layout`** *(optional)* — Matrix organizing multiple antenna IDs within the file; must reference only IDs present in the file

### Returns:
- ID assigned to the written antenna within the file

### See also:
- [qdant_read](#qdant_read) (read back QDANT files)
- [qdant_write_multi](#qdant_write_multi) (multi-freq counterpart)

---
<a id="qdant_write_multi"></a>
## qdant_write_multi
Write a vector of arrayant objects to a single QDANT file

Also in: [MATLAB](../mex/array-antenna-functions.md#qdant_write) · [Python](../python/array-antenna-functions.md#qdant_write)

- Writes each entry in `arrayant_vec` to a QDANT file with sequential 1-based IDs using [qdant_write](#qdant_write).
- Auto-generates a `[n_entries, 1]` layout matrix with entries `1, 2, ..., n_entries`.
- The file must not exist yet; an existing file throws `std::runtime_error` and is left unchanged.
- Primary use case: frequency-dependent models where each arrayant holds a pattern at one frequency via `center_freq`.

### Declaration:
```cpp
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Inputs:
- **`fn`** — Path of the QDANT file to write; must not be empty and must not exist
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects to store

### See also:
- [qdant_write](#qdant_write) (per-object write used internally)
- [qdant_read](#qdant_read) (read back individual entries by ID)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)
