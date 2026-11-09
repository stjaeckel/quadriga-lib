# Array antenna functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-array-antenna-functions

| Function | Summary |
| --- | --- |
| [arrayant_combine_pattern_multi](#arrayant_combine_pattern_multi) | Combine element patterns, positions, and coupling weights into effective radiation patterns (multi-frequency) |
| [arrayant_concat_multi](#arrayant_concat_multi) | Concatenate two multi-frequency arrayant vectors into a single multi-element model |
| [arrayant_copy_element_multi](#arrayant_copy_element_multi) | Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector |
| [arrayant_interpolate](#arrayant_interpolate) | Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles |
| [arrayant_interpolate_multi](#arrayant_interpolate_multi) | Interpolate a multi-frequency array antenna at given angles and frequencies |
| [arrayant_is_valid_multi](#arrayant_is_valid_multi) | Validate a vector of arrayant objects for multi-frequency consistency |
| [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) | Apply Euler rotations to all entries in a multi-frequency arrayant vector |
| [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) | Set element positions for all entries in a multi-frequency arrayant vector |
| [generate_arrayant_3GPP](#generate_arrayant_3GPP) | Generate a 3GPP-NR compliant antenna array model |
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
| [qdant_write_multi](#qdant_write_multi) | Write a vector of arrayant objects to a single QDANT file |

---
<a id="arrayant_combine_pattern_multi"></a>
## arrayant_combine_pattern_multi
Combine element patterns, positions, and coupling weights into effective radiation patterns (multi-frequency)

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_combine_pattern) · [Python](../python/array-antenna-functions.md#combine_pattern)

- Multi-frequency counterpart to [.combine_pattern](array-antenna-class.md#.combine_pattern)
- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` across all entries to produce one output element per port (column of the coupling matrix) at each requested output frequency
- Output length = `freq_grid_new->n_elem` if provided; otherwise one entry per input arrayant
- Field interpolation across frequency is delegated to [arrayant_interpolate_multi](#arrayant_interpolate_multi) (SLERP with linear-interpolation fallback)
- Coupling matrices are SLERP-interpolated between bracketing input entries; out-of-range output frequencies are clamped to the nearest input entry
- Each output arrayant has identity coupling and zero element positions (patterns are pre-combined)
- The element positions add the phase `-2 pi f d / propagation_speed` at each output frequency `f`; the default speed
  of light applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [generate_speaker](#generate_speaker))

### Declaration:
```cpp
std::vector<arrayant<dtype>> arrayant_combine_pattern_multi(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    const arma::Col<dtype> *freq_grid_new = nullptr,
    dtype propagation_speed = 299792458.0);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects (must pass [arrayant_is_valid_multi](#arrayant_is_valid_multi))
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to `arrayant_vec[0].azimuth_grid`
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to `arrayant_vec[0].elevation_grid`
- **`freq_grid_new`** *(optional)* — Alternative frequency grid in Hz, positive and finite; defaults to per-entry
  `center_frequency`
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

### Returns:
- Vector of arrayant objects (length = `n_freq_out`), each with `n_ports` elements equal to the number of columns in the coupling matrix

### See also:
- [.combine_pattern](array-antenna-class.md#.combine_pattern) (single-frequency counterpart)
- [arrayant_interpolate_multi](#arrayant_interpolate_multi) (used internally for spatial+frequency field interpolation)
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (input validation)

---
<a id="arrayant_concat_multi"></a>
## arrayant_concat_multi
Concatenate two multi-frequency arrayant vectors into a single multi-element model

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_concat) · [Python](../python/array-antenna-functions.md#concat)

- Both inputs must have equal entry counts, identical angular grids, and matching `center_frequency` values at each index.
- Per frequency entry: pattern cubes are joined along the element (slice) dimension; `element_pos` matrices are horizontally concatenated (empty positions treated as zeros).
- Both inputs are validated with [arrayant_is_valid_multi](#arrayant_is_valid_multi) before processing; each output entry is validated before returning.
- Output inherits name, azimuth/elevation grids, and `center_frequency` from `arrayant_vec1`.

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
<a id="arrayant_copy_element_multi"></a>
## arrayant_copy_element_multi
Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_copy_element) · [Python](../python/array-antenna-functions.md#copy_element)

- Calls [.copy_element](array-antenna-class.md#.copy_element) on every entry in the vector with the same source and destination indices.
- If any destination index exceeds the current element count, all entries are enlarged; new elements receive an identity coupling entry.
- Source and destination indices are 0-based.

### Declaration:
```cpp
void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uvec destination);

void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uword destination);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects; modified in-place
- **`source`** — Index of the element to copy from; must be within current element count
- **`destination`** — Index or indices of target elements; enlarges all entries if any index exceeds current count

### Example:
```cpp
arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
auto driver = quadriga_lib::generate_speaker<double>(
    "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
    0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
quadriga_lib::arrayant_copy_element_multi(driver, 0, arma::uvec{1, 2, 3});
```

### See also:
- [.copy_element](array-antenna-class.md#.copy_element) (per-entry operation called internally)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions after copying)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multiple arrayant vectors)

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
- Kernels: GENERIC computes in the precision of `dtype`; AVX2 computes in single precision also for `double`, with
  errors of up to a few 1e-5 of the peak field value and about 1e-4 rad in the angles (more near the poles of the
  local frame); auto selects GENERIC in this version, which may change in a later version
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
    const arma::uvec *i_element = nullptr,
    const arma::Cube<dtype> *orientation = nullptr,
    const arma::Mat<dtype> *element_pos = nullptr,
    arma::Mat<dtype> *dist = nullptr,
    arma::Mat<dtype> *azimuth_loc = nullptr,
    arma::Mat<dtype> *elevation_loc = nullptr,
    arma::Mat<dtype> *gamma = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`ant`** — Array antenna with `n_elements` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; `nullptr` or empty: all elements in
  order (`n_out = n_elements`); `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) of the antenna, for all outputs and angles,
  per output, per angle, or both; `nullptr` or empty: no rotation; `[3, 1, 1]`, `[3, n_out, 1]`, `[3, 1, n_ang]` or
  `[3, n_out, n_ang]`
- **`element_pos`** *(optional)* — Element positions in local antenna coordinates, used for `dist`; `nullptr` or
  empty: the columns `i_element` of `ant.element_pos`, zeros if that is empty; `[3, n_out]`
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
- Angles, element indices, orientation and kernels as in [arrayant_interpolate](#arrayant_interpolate); `n_ang = 0` gives outputs of size
  `[n_out, 0, n_freq]`
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
    const arma::Col<dtype> *frequency = nullptr,
    const arma::uvec *i_element = nullptr,
    const arma::Cube<dtype> *orientation = nullptr,
    const arma::Mat<dtype> *element_pos = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`arrayant_vec`** — Multi-frequency array antenna, one entry per center frequency; entries need not be sorted by
  frequency
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`frequency`** *(optional)* — Target frequencies in Hz; `nullptr` or empty: the center frequencies of the entries
  in vector order; `[n_freq]`
- **`i_element`** *(optional)* — Element indices, 0-based, duplicates allowed; `nullptr` or empty: all elements;
  `[n_out]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading), applied at all frequencies; shapes as in
  [arrayant_interpolate](#arrayant_interpolate)
- **`element_pos`** *(optional)* — Element positions in m; checked for its size only, since no output depends on the
  positions; `[3, n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2; as in [arrayant_interpolate](#arrayant_interpolate)

### Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; `[n_out, n_ang, n_freq]`

### Example:
```cpp
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
arma::mat el(1, 4, arma::fill::zeros);
arma::vec freq = {250.0, 1500.0, 8000.0};
arma::cube V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate_multi(speaker, az, el, V_re, V_im, H_re, H_im, &freq);
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

- Each entry is validated individually via its `is_valid` member; `quick_check` is forwarded to that call.
- Cross-entry checks (all vs. entry 0): azimuth/elevation grid sizes and values, number of elements, element positions, coupling_re shape, and coupling_im presence and size.
- Pattern data, `center_frequency`, and coupling matrix values are not compared (expected to vary).
- Stops at first error and returns a message identifying the failing entry and property.

### Declaration:
```cpp
std::string quadriga_lib::arrayant_is_valid_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        bool quick_check = true);
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects to validate
- **`quick_check`** *(optional)* — If `true`, uses fast pointer-based per-entry validation; if `false`, performs full deep validation

### Returns:
- Empty string if valid; otherwise a message such as `"Entry 3: Azimuth grid values do not match entry 0."`

### See also:
- [.is_valid](array-antenna-class.md#.is_valid) (per-entry validation called internally)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="arrayant_rotate_pattern_multi"></a>
## arrayant_rotate_pattern_multi
Apply Euler rotations to all entries in a multi-frequency arrayant vector

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_rotate_pattern) · [Python](../python/array-antenna-functions.md#rotate_pattern)

- Calls [.rotate_pattern](array-antenna-class.md#.rotate_pattern) on every entry with grid adjustment always disabled (required for uniform-grid consistency across frequencies).
- If `i_element` is empty, all elements are rotated; otherwise only the specified indices are affected.
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects.

### Declaration:
```cpp
void quadriga_lib::arrayant_rotate_pattern_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        dtype x_deg = 0.0,
        dtype y_deg = 0.0,
        dtype z_deg = 0.0,
        unsigned usage = 0,
        arma::uvec i_element = arma::uvec())
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`x_deg`** *(optional)* — Bank angle in degrees
- **`y_deg`** *(optional)* — Tilt angle in degrees
- **`z_deg`** *(optional)* — Heading angle in degrees
- **`usage`** *(optional)* — Rotation mode: `0` = pattern + polarization, `1` = pattern only, `2` = polarization only
- **`i_element`** *(optional)* — Indices of elements to rotate; if empty, all elements are rotated

### See also:
- [.rotate_pattern](array-antenna-class.md#.rotate_pattern) (per-entry operation called internally)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multi-frequency vectors before rotating)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions in multi-frequency vectors)

---
<a id="arrayant_set_element_pos_multi"></a>
## arrayant_set_element_pos_multi
Set element positions for all entries in a multi-frequency arrayant vector

Not in: MATLAB, Python

- Updates `element_pos` in-place on every entry in the vector identically.
- If `i_element` is empty, all positions are replaced and `element_pos` must have `n_elements` columns.
- If `i_element` is provided, only those indexed columns are updated; `element_pos` column count must match `i_element` length.
- All entries must have the same element count; uninitialized `element_pos` fields are zero-initialized before update.

### Declaration:
```cpp
void quadriga_lib::arrayant_set_element_pos_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        arma::uvec i_element = arma::uvec());
```

### Inputs:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`element_pos`** — New (x, y, z) positions; `[3, n_update]`
- **`i_element`** *(optional)* — Indices of elements to update; if empty, all elements are replaced

### See also:
- [arrayant_copy_element_multi](#arrayant_copy_element_multi) (replicate elements before setting positions)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
<a id="generate_arrayant_3GPP"></a>
## generate_arrayant_3GPP
Generate a 3GPP-NR compliant antenna array model

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Supports vertical (M) and horizontal (N) element stacking within panels, and multi-panel arrays (Mg × Ng).
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.
- Electrical downtilt (`tilt`) applies only to `pol` modes 4, 5, and 6.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(
    arma::uword M = 1, 
    arma::uword N = 1, 
    dtype center_freq = 299792458.0,
    unsigned pol = 1, 
    dtype tilt = 0.0, 
    dtype spacing = 0.5, 
    arma::uword Mg = 1,
    arma::uword Ng = 1, 
    dtype dgv = 0.5, 
    dtype dgh = 0.5,
    const quadriga_lib::arrayant<dtype> *pattern = nullptr, 
    dtype res = 1.0);
```

### Inputs:
- **`M`** *(optional)* — Number of vertical elements per panel
- **`N`** *(optional)* — Number of horizontal elements per panel
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:
   | `pol` | Description                          | Elements |
   | ----- | ------------------------------------ | -------- |
   | 1     | Vertical polarization                | NM       |
   | 2     | H/V polarization                     | 2NM      |
   | 3     | ±45° polarization                    | 2NM      |
   | 4     | Vertical, vertical elements combined | N        |
   | 5     | H/V, vertical elements combined      | 2N       |
   | 6     | ±45°, vertical elements combined     | 2N       |
- **`tilt`** *(optional)* — Electrical downtilt in degrees; applies to `pol` 4–6
- **`spacing`** *(optional)* — Inter-element spacing within a panel in wavelengths
- **`Mg`** *(optional)* — Number of vertically stacked panels
- **`Ng`** *(optional)* — Number of horizontally stacked panels
- **`dgv`** *(optional)* — Panel spacing in vertical direction in wavelengths
- **`dgh`** *(optional)* — Panel spacing in horizontal direction in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default 3GPP element pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

### Returns:
- `quadriga_lib::arrayant<dtype>` — 3GPP-NR antenna array object

---
<a id="generate_arrayant_custom"></a>
## generate_arrayant_custom
Generate an antenna with custom 3dB beamwidth

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a single-element antenna with independently configurable azimuth and elevation 3dB (FWHM) beamwidths.
- Rear-side gain is controlled by a linear front-to-back ratio; `0.0` means no rear radiation.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(
    dtype az_3dB = 90.0,
    dtype el_3dB = 90.0, 
    dtype rear_gain_lin = 0.0, 
    dtype res = 1.0);
```

### Inputs:
- **`az_3dB`** *(optional)* — Azimuth 3dB beamwidth in degrees
- **`el_3dB`** *(optional)* — Elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

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
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

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
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized half-wave dipole antenna object

---
<a id="generate_arrayant_multibeam"></a>
## generate_arrayant_multibeam
Generate a planar multi-element antenna array with multiple beam directions

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns an M×N planar array with beamforming weights computed via maximum-ratio transmission (MRT).
- MRT is optimal for a single beam; approximate when multiple beams are specified.
- Weights control relative beam contribution; only their ratios matter, not absolute values.
- If `separate_beams = true`, each angle pair produces an independent beam (weights ignored).
- If `apply_weights = true`, beamforming weights are baked into the element coupling matrix.
- Per-element pattern shape is controlled by `az_3dB`, `el_3dB`, and `rear_gain_lin`; see [generate_arrayant_custom](#generate_arrayant_custom).

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_multibeam(
    arma::uword M = 1,
    arma::uword N = 1,
    arma::Col<dtype> az = {0.0},
    arma::Col<dtype> el = {0.0},
    arma::Col<dtype> weight = {1.0},
    dtype center_freq = 299792458.0,
    unsigned pol = 1,
    dtype spacing = 0.5,
    dtype az_3dB = 120.0,
    dtype el_3dB = 120.0,
    dtype rear_gain_lin = 0.0,
    dtype res = 1.0,
    bool separate_beams = false,
    bool apply_weights = false);
```

### Inputs:
- **`M`** *(optional)* — Number of vertical (row) elements
- **`N`** *(optional)* — Number of horizontal (column) elements
- **`az`** *(optional)* — Azimuth beam angles in degrees; `[n_beams]`
- **`el`** *(optional)* — Elevation beam angles in degrees; `[n_beams]`
- **`weight`** *(optional)* — Per-beam scaling factors (not normalized); `[n_beams]`
- **`center_freq`** *(optional)* — Center frequency
- **`pol`** *(optional)* — Polarization mode:
   | `pol` | Description           | Elements |
   | ----- | --------------------- | -------- |
   | 1     | Vertical polarization | NM       |
   | 2     | H/V polarization      | 2NM      |
   | 3     | ±45° polarization     | 2NM      |
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`az_3dB`** *(optional)* — Per-element azimuth 3dB beamwidth in degrees; values <= 0 select the default 120
- **`el_3dB`** *(optional)* — Per-element elevation 3dB beamwidth in degrees; values <= 0 select the default 120
- **`rear_gain_lin`** *(optional)* — Per-element front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees
- **`separate_beams`** *(optional)* — If `true`, generate one independent beam per angle pair
- **`apply_weights`** *(optional)* — If `true`, bake beamforming weights into the coupling matrix

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
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Isotropic radiator antenna object

---
<a id="generate_arrayant_ula"></a>
## generate_arrayant_ula
Generate a uniform linear array (ULA)

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_generate) · [Python](../python/array-antenna-functions.md#generate)

- Returns a horizontally stacked linear array of N elements with half-wavelength spacing by default.
- Default per-element pattern is a vertically polarized isotropic radiator.
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_ula(
    arma::uword N = 1, 
    dtype center_freq = 299792458.0, 
    dtype spacing = 0.5,
    const quadriga_lib::arrayant<dtype> *pattern = nullptr, 
    dtype res = 1.0);
```

### Inputs:
- **`N`** *(optional)* — Number of elements
- **`center_freq`** *(optional)* — Center frequency
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default isotropic pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

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
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol(dtype res = 1.0);
```

### Inputs:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Cross-polarized isotropic radiator antenna object

---
<a id="generate_speaker"></a>
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

Also in: [MATLAB](../mex/array-antenna-functions.md#generate_speaker) · [Python](../python/array-antenna-functions.md#generate_speaker)

- Returns one [arrayant](array-antenna-class.md#arrayant) object per frequency sample; each has a single element with the real-valued 
  directivity pattern in `e_theta_re` and `center_frequency` set to the corresponding frequency.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining results 
  via `append` and `element_pos`; crossover behavior emerges from overlapping bandpass responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`, 
  where `n = slope_dB_per_octave / 6`; −3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity − 85) / 20)`.
- If `frequencies` is empty, third-octave band center frequencies are auto-generated from one band below 
  `lower_cutoff` to one band above `upper_cutoff`, clipped to 20–20000 Hz.
- Speed of sound assumed to be 344 m/s.
- **Driver models** (`driver_type`):
  - `piston` — circular piston in baffle, `D(θ) = 2·J1(ka·sinθ)/(ka·sinθ)`, rotationally symmetric, narrows with increasing `ka`
  - `horn` — separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`
  - `omni` — frequency-independent omnidirectional pattern.
- **Enclosure models** (`radiation_type`):
  - `monopole` — no modification
  - `hemisphere` — sealed box with baffle-step transition, `f_baffle = c/(π·sqrt(W·H))`
  - `dipole` — figure-8, `R = abs(cos(θ_off))` with sign inversion in rear hemisphere
  - `cardioid` — `R = 0.5·(1+cos(θ_off))`
- For `"horn"`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2π·radius)`.

### Declaration:
```cpp
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::generate_speaker(
        std::string driver_type = "piston",
        dtype radius = 0.05,
        dtype lower_cutoff = 80.0,
        dtype upper_cutoff = 12000.0,
        dtype lower_rolloff_slope = 12.0,
        dtype upper_rolloff_slope = 12.0,
        dtype sensitivity = 85.0,
        std::string radiation_type = "hemisphere",
        dtype hor_coverage = 0.0,
        dtype ver_coverage = 0.0,
        dtype horn_control_freq = 0.0,
        dtype baffle_width = 0.15,
        dtype baffle_height = 0.25,
        arma::Col<dtype> frequencies = arma::Col<dtype>(),
        dtype angular_resolution = 5.0);
```

### Inputs:
- **`driver_type`** *(optional)* — Driver directivity model: `"piston"`, `"horn"`, or `"omni"`
- **`radius`** *(optional)* — Effective radiating radius; cone/dome radius for piston, mouth radius for horn
- **`lower_cutoff`** *(optional)* — Lower −3 dB bandpass frequency
- **`upper_cutoff`** *(optional)* — Upper −3 dB bandpass frequency
- **`lower_rolloff_slope`** *(optional)* — Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth)
- **`upper_rolloff_slope`** *(optional)* — High-frequency rolloff in dB/octave
- **`sensitivity`** *(optional)* — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude
- **`radiation_type`** *(optional)* — Enclosure radiation model: `"monopole"`, `"hemisphere"`, `"dipole"`, or `"cardioid"`
- **`hor_coverage`** *(optional)* — Horn horizontal coverage angle in degrees; `0` defaults to 90°
- **`ver_coverage`** *(optional)* — Horn vertical coverage angle in degrees; `0` defaults to 60°
- **`horn_control_freq`** *(optional)* — Horn pattern control frequency; `0` auto-derives from `radius`
- **`baffle_width`** *(optional)* — Baffle width; used by `"hemisphere"` model
- **`baffle_height`** *(optional)* — Baffle height; used by `"hemisphere"` model
- **`frequencies`** *(optional)* — Frequency sample points; auto-generated third-octave bands if empty; `[n_freq]`
- **`angular_resolution`** *(optional)* — Azimuth and elevation sampling grid resolution in degrees

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One arrayant per frequency sample with directivity in `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180° phase inversion

### Example:
```cpp
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
               12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
auto horn = quadriga_lib::generate_speaker<double>("horn");
auto sub = quadriga_lib::generate_speaker<double>("omni", 0.13, 30.0, 200.0,
               12.0, 24.0, 92.0, "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, {30.,50.,80.,120.,200.}, 10.0);
```

---
<a id="qdant_read"></a>
## qdant_read
Read an arrayant object from a QDANT file

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_qdant_read) · [Python](../python/array-antenna-functions.md#qdant_read)

- Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the arrayant for the given ID.
- Throws `std::invalid_argument` if the file has no entry with the given ID. The first entry of a file is ID 1 only
  if it has no `id` attribute (files written by other tools); otherwise ID 1 must exist.
- Malformed files throw `std::invalid_argument`. The number of elements (`NoElements`) must be backed by data in the
  file (element positions, coupling, or pattern nodes of the last element). Elements without pattern nodes read as
  zeros; their zero patterns, and the identity coupling of a file without `CouplingAbs`, are limited to 2^24 values
  each (e.g. 256 elements without pattern data on a 1-degree grid, 4096 elements without coupling).

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(
        std::string fn,
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
- [.qdant_write](array-antenna-class.md#.qdant_write) (write a single arrayant)
- [qdant_write_multi](#qdant_write_multi) (write multiple arrayants with sequential IDs)

---
<a id="qdant_read_multi"></a>
## qdant_read_multi
Read all arrayant objects from a QDANT file into a vector

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_qdant_read) · [Python](../python/array-antenna-functions.md#qdant_read)

- Reads all entries from a QDANT file by probing ID 1 to obtain the layout, then reading each unique non-zero ID in order of first appearance (column-major scan).
- Each unique ID is read exactly once regardless of how many times it appears in the layout.
- Counterpart to [qdant_write_multi](#qdant_write_multi); primary mechanism for loading frequency-dependent models where `center_frequency` on each entry identifies the corresponding frequency.

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
<a id="qdant_write_multi"></a>
## qdant_write_multi
Write a vector of arrayant objects to a single QDANT file

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_qdant_write) · [Python](../python/array-antenna-functions.md#qdant_write)

- Writes each entry in `arrayant_vec` to a QDANT file with sequential 1-based IDs using [.qdant_write](array-antenna-class.md#.qdant_write).
- Auto-generates a `[n_entries, 1]` layout matrix with entries `1, 2, ..., n_entries`.
- All entries are validated first; an existing file is deleted only after the validation passed, then rewritten.
  Invalid input throws and leaves an existing file unchanged.
- Primary use case: frequency-dependent models where each arrayant holds a pattern at one frequency via `center_frequency`.

### Declaration:
```cpp
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Inputs:
- **`fn`** — Path of the QDANT file to write; must not be empty
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects to store

### See also:
- [.qdant_write](array-antenna-class.md#.qdant_write) (per-object write used internally)
- [qdant_read](#qdant_read) (read back individual entries by ID)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)
