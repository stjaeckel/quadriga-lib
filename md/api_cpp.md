---
title: "C++ API Documentation for Quadriga-Lib v0.11.1"
author: "Stephan Jaeckel"
date: "15.04.2026"
lang: en-EN
---

<!-- PLACEHOLDER: C++ API preamble -->
<!-- Edit this section to add installation instructions, build requirements, and general usage notes for the C++ API. -->

# Overview

- [Array antenna class](#array-antenna-class)
- [Array antenna functions](#array-antenna-functions)
- [Channel class](#channel-class)
- [Channel functions](#channel-functions)
- [Channel generation functions](#channel-generation-functions)
- [Math Functions](#math-functions)
- [Miscellaneous / Tools](#miscellaneous-tools)
- [Site-Specific Simulation Tools](#site-specific-simulation-tools)

---

# Array antenna class

| Function | Description |
| --- | --- |
| [arrayant](#arrayant) | Class for storing and manipulating array antenna models |
| [.append](#append) | Append elements of another arrayant to the current one |
| [.calc_directivity_dBi](#calc_directivity_dbi) | Calculate the directivity in dBi of a single array element |
| [.combine_pattern](#combine_pattern) | Combine element patterns, positions, and coupling weights into effective radiation patterns |
| [.copy_element](#copy_element) | Copy a single antenna element to one or more destination slots |
| [.export_obj_file](#export_obj_file) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [.interpolate](#interpolate) | Interpolate polarimetric antenna field patterns for given azimuth/elevation angles |
| [.is_valid](#is_valid) | Validate the integrity of an arrayant object |
| [.qdant_write](#qdant_write) | Write arrayant data to a QDANT (XML) file |
| [.remove_zeros](#remove_zeros) | Remove zero-valued entries from antenna pattern data, reducing its size |
| [.rotate_pattern](#rotate_pattern) | Rotate antenna radiation patterns around the principal axes using Euler rotations |
| [.set_size](#set_size) | Resize an arrayant object to new dimensions |

---
## arrayant
Class for storing and manipulating array antenna models

### Description:
- Represents a multi-element antenna array; each element has a position relative to the array phase-center
- Elements may be inter-coupled via a complex coupling matrix
- Field pattern cubes `e_theta_re/im`, `e_phi_re/im` must all be `[n_elevation, n_azimuth, n_elements]`
- `element_pos` is optional (empty = all elements at origin); `coupling_re/im` are optional (empty = identity)
- Allowed datatypes (`dtype`): `float` or `double`

### Attributes:
| Attribute | Size | Description |
|-----------|------|-------------|
| `arma::Cube<dtype> e_theta_re` | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, real part |
| `arma::Cube<dtype> e_theta_im` | `[n_elevation, n_azimuth, n_elements]` | E-theta (vertical) field, imaginary part |
| `arma::Cube<dtype> e_phi_re`   | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, real part |
| `arma::Cube<dtype> e_phi_im`   | `[n_elevation, n_azimuth, n_elements]` | E-phi (horizontal) field, imaginary part |
| `arma::Col<dtype> azimuth_grid` | `[n_azimuth]` | Azimuth angles in rad, in [-pi, pi], sorted |
| `arma::Col<dtype> elevation_grid` | `[n_elevation]` | Elevation angles in rad, in [-pi/2, pi/2], sorted |
| `arma::Mat<dtype> element_pos` | `[3, n_elements]` or empty | Element positions in local Cartesian coords |
| `arma::Mat<dtype> coupling_re` | `[n_elements, n_ports]` | Coupling matrix, real part |
| `arma::Mat<dtype> coupling_im` | `[n_elements, n_ports]` | Coupling matrix, imaginary part |
| `dtype center_frequency` | scalar | Center frequency in Hz |

### Simple member functions:
| Function | Description |
|----------|-------------|
| `.n_elevation()` | Number of elevation angles |
| `.n_azimuth()` | Number of azimuth angles |
| `.n_elements()` | Number of antenna elements |
| `.n_ports()` | Number of ports (columns of coupling matrix) |
| `.copy()` | Returns a deep copy of the arrayant object |
| `.reset()` | Clears all data, resetting size to zero |
| `.is_valid()` | Returns `""` if valid, or an error message string |

### Complex member functions:
| Function | Description |
|----------|-------------|
| [.append](#.append) | Append elements of another arrayant |
| [.calc_directivity_dBi](#.calc_directivity_dBi) | Calculate per-element directivity in dBi |
| [.combine_pattern](#.combine_pattern) | Compute effective patterns from elements, positions, and coupling |
| [.copy_element](#.copy_element) | Copy a single element to one or more destination slots |
| [.export_obj_file](#.export_obj_file) | Export pattern geometry to Wavefront OBJ |
| [.interpolate](#.interpolate) | Interpolate field patterns at given azimuth/elevation angles |
| [.qdant_write](#.qdant_write) | Write arrayant to QDANT file |
| [.remove_zeros](#.remove_zeros) | Remove zero-valued entries from pattern data |
| [.rotate_pattern](#.rotate_pattern) | Rotate pattern and/or polarization via Euler angles |
| [.set_size](#.set_size) | Resize the arrayant to new dimensions |
| [.is_valid](#.is_valid) | Validate arrayant integrity |

---
## .append
Append elements of another arrayant to the current one

### Description:
- Member function of [arrayant](#arrayant)
- Both arrays must share identical sampling grids; throws otherwise
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::append(
    const arrayant<dtype> *new_arrayant) const;
```

### Input Arguments:
- **`new_arrayant`** — Array whose elements are appended; sampling grid must match

### Returns:
- New `arrayant` containing all elements from both arrays

---
## .calc_directivity_dBi
Calculate the directivity in dBi of a single array element

### Description:
- Member function of [arrayant](#arrayant)
- Directivity = 10 * log10(radiation intensity in given direction / mean radiation intensity over all directions); isotropic radiator = 0 dBi
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const;
```

### Input Arguments:
- **`i_element`** — Element index, 0-based

### Returns:
- Directivity of the specified element in dBi

---
## .combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

### Description:
- Member function of [arrayant](#arrayant)
- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` to produce one output element per port (column) of the coupling matrix
- Useful for beamforming and MIMO channel computation speedup
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::combine_pattern(
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr) const;
```

### Input Arguments:
- **`azimuth_grid_new`** *(optional)* — Output azimuth grid in rad, in [-pi, pi], sorted; defaults to input grid
- **`elevation_grid_new`** *(optional)* — Output elevation grid in rad, in [-pi/2, pi/2], sorted; defaults to input grid

### Returns:
- New `arrayant` with `n_ports` elements (= number of columns in `coupling_re/im`), each holding the combined effective pattern for that port

---
## .copy_element
Copy a single antenna element to one or more destination slots

### Description:
- Member function of [arrayant](#arrayant)
- Array is resized if any destination index exceeds the current number of elements
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uword destination);
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uvec destination);
```

### Input Arguments:
- **`source`** — Index of the element to copy, 0-based
- **`destination`** — Target index or indices, 0-based; array resizes to fit the maximum index

---
## .export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

### Description:
- Member function of [arrayant](#arrayant)
- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives finer mesh
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::export_obj_file(
    std::string fn,
    dtype directivity_range = 30.0,
    std::string colormap = "jet",
    dtype object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    arma::uvec i_element = {}) const;
```

### Input Arguments:
- **`fn`** — Output OBJ filename; must not be empty; filename must end in .obj
- **`directivity_range`** *(optional)* — Dynamic range of the visualized directivity pattern in dB
- **`colormap`** *(optional)* — Colormap name; see [colormap](#colormap) for supported options
- **`object_radius`** *(optional)* — Radius of the exported geometry object in meters
- **`icosphere_n_div`** *(optional)* — Icosphere subdivision count; higher = finer mesh, see [icosphere](#icosphere)
- **`i_element`** *(optional)* — 0-based element indices to export; `{}` exports all elements

---
## .interpolate
Interpolate polarimetric antenna field patterns for given azimuth/elevation angles

### Description:
- Member function of [arrayant](#arrayant)
- Outputs complex e-theta (V) and e-phi (H) field components at requested angles
- `n_out` equals `n_elements` when `i_element` is omitted; equals `len(i_element)` otherwise
- Azimuth input supports planar wave mode (`[1, n_ang]`) or per-element spherical wave mode (`[n_out, n_ang]`)
- Output matrices are resized automatically if dimensions do not match; this invalidates existing data pointers
- `dist` is the projection of element positions onto the plane normal to the incident path — needed for phase computation
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::interpolate(
    const arma::Mat<dtype> *azimuth,
    const arma::Mat<dtype> *elevation,
    arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
    arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
    arma::uvec i_element = {},
    const arma::Cube<dtype> *orientation = nullptr,
    const arma::Mat<dtype> *element_pos_i = nullptr,
    arma::Mat<dtype> *dist = nullptr,
    arma::Mat<dtype> *azimuth_loc = nullptr,
    arma::Mat<dtype> *elevation_loc = nullptr,
    arma::Mat<dtype> *gamma = nullptr) const;
```

### Input Arguments:
- **`azimuth`** — Azimuth angles in rad, in [-pi, pi]; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in rad, in [-pi/2, pi/2]; `[1, n_ang]` or `[n_out, n_ang]`
- **`i_element`** *(optional)* — Element indices (0-based) to interpolate; duplicates allowed; defaults to all elements, `[n_out]` or `{}`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) in rad; `nullptr`, `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]`, or `[3, n_out, n_ang]`
- **`element_pos_i`** *(optional)* — Override element positions in m; `nullptr` uses `arrayant.element_pos`; `[3, n_out]`

### Output Arguments:
- **`V_re`** / **`V_im`** — Real/imaginary e-theta (vertical) field component; `[n_out, n_ang]`
- **`H_re`** / **`H_im`** — Real/imaginary e-phi (horizontal) field component; `[n_out, n_ang]`
- **`dist`** *(optional)* — Element distances projected onto incident-path normal plane; `nullptr` or `[n_out, n_ang]`
- **`azimuth_loc`** *(optional)* — Azimuth angles in local (rotated) element frame in rad; `nullptr` or `[n_out, n_ang]`
- **`elevation_loc`** *(optional)* — Elevation angles in local element frame in rad; `nullptr` or `[n_out, n_ang]`
- **`gamma`** *(optional)* — Polarization rotation angles in rad; `nullptr` or `[n_out, n_ang]`

### Example:
```
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
arma::mat azimuth = {0.0, 0.5*pi, -0.5*pi, pi};
arma::mat elevation(1, azimuth.n_elem);  // zeros
arma::mat V_re, V_im, H_re, H_im;
ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);
```

---
## .is_valid
Validate the integrity of an arrayant object

### Description:
- Member function of [arrayant](#arrayant)
- Returns empty string if valid; otherwise returns a descriptive error message
- Quick check (default) validates dimensions and structure; full check additionally verifies data values
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
std::string quadriga_lib::arrayant<dtype>::is_valid(bool quick_check = true) const;
```

### Input Arguments:
- **`quick_check`** *(optional)* — `true` for fast structural check, `false` for full data validation

### Returns:
- Empty string if valid; error message string if invalid

---
## .qdant_write
Write arrayant data to a QDANT (XML) file

### Description:
- Member function of [arrayant](#arrayant)
- Multiple antennas can be stored in the same file using distinct `id` values
- If `id = 0` and the file exists, the antenna is appended with `id = max_existing_id + 1`
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
unsigned quadriga_lib::arrayant<dtype>::qdant_write(
    std::string fn,
    unsigned id = 0,
    arma::u32_mat layout = {}) const;
```

### Input Arguments:
- **`fn`** — Output QDANT filename; must not be empty
- **`id`** *(optional)* — Target ID in file; `0` appends with auto-assigned ID
- **`layout`** *(optional)* — Matrix organizing multiple antenna IDs within the file; must reference only IDs present in the file

### Returns:
ID assigned to the written antenna within the file

### See also:
- [qdant_read](#qdant_read) (read back QDANT files)

---
## .remove_zeros
Remove zero-valued entries from antenna pattern data, reducing its size

### Description:
- Member function of [arrayant](#arrayant)
- Modifies in-place when `output = nullptr`; otherwise writes to `*output`
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::remove_zeros(arrayant<dtype> *output = nullptr);
```

### Input Arguments:
- **`output`** *(optional)* — Target arrayant to write result to; `nullptr` modifies in-place

---
## .rotate_pattern
Rotate antenna radiation patterns around the principal axes using Euler rotations

### Description:
- Member function of [arrayant](#arrayant)
- Rotates pattern and/or polarization around x (bank), y (tilt), z (heading) axes in degrees
- Modes 0/1: after rotation, the angular sampling grid is adjusted to follow the rotated pattern — needed for non-uniform grids (e.g. parabolic antennas with small apertures)
- Modes 3/4: grid stays fixed after rotation — correct for uniformly sampled patterns where the original grid structure should be preserved
- Modifies in-place when `output = nullptr`; otherwise writes to `*output`
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::rotate_pattern(
    dtype x_deg = 0.0,
    dtype y_deg = 0.0,
    dtype z_deg = 0.0,
    unsigned usage = 0,
    unsigned element = -1,
    arrayant<dtype> *output = nullptr);
```

### Input Arguments:
- **`x_deg`** *(optional)* — Rotation around x-axis (bank) in degrees
- **`y_deg`** *(optional)* — Rotation around y-axis (tilt) in degrees
- **`z_deg`** *(optional)* — Rotation around z-axis (heading) in degrees
- **`usage`** *(optional)* — Rotation mode:

  | Mode | Pattern | Polarization | Grid adjustment |
  |------|---------|--------------|-----------------|
  | 0    | Yes     | Yes          | Yes             |
  | 1    | Yes     | No           | Yes             |
  | 2    | No      | Yes          | No              |
  | 3    | Yes     | Yes          | No              |
  | 4    | Yes     | No           | No              |

- **`element`** *(optional)* — 0-based element index to rotate; `-1` rotates all elements
- **`output`** *(optional)* — Target arrayant; `nullptr` modifies in-place

---
## .set_size
Resize an arrayant object to new dimensions

### Description:
- Member function of [arrayant](#arrayant)
- No-op if current dimensions already match; errors if object is read-only
- After resize: element_pos is zeroed, coupling_re set to identity, coupling_im zeroed; all other field data is undefined
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant<dtype>::set_size(
    arma::uword n_elevation,
    arma::uword n_azimuth,
    arma::uword n_elements,
    arma::uword n_ports);
```

### Input Arguments:
- **`n_elevation`** — Number of elevation samples
- **`n_azimuth`** — Number of azimuth samples
- **`n_elements`** — Number of antenna elements
- **`n_ports`** — Number of ports (columns of coupling matrix)

---

# Array antenna functions

| Function | Description |
| --- | --- |
| [arrayant_concat_multi](#arrayant_concat_multi) | Concatenate two multi-frequency arrayant vectors into a single multi-element model |
| [arrayant_copy_element_multi](#arrayant_copy_element_multi) | Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector |
| [arrayant_interpolate_multi](#arrayant_interpolate_multi) | Interpolate multi-frequency arrayant patterns at arbitrary angles and frequencies |
| [arrayant_is_valid_multi](#arrayant_is_valid_multi) | Validate a vector of arrayant objects for multi-frequency consistency |
| [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) | Apply Euler rotations to all entries in a multi-frequency arrayant vector |
| [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) | Set element positions for all entries in a multi-frequency arrayant vector |
| [generate_arrayant_3GPP](#generate_arrayant_3gpp) | Generate a 3GPP-NR compliant antenna array model |
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
## arrayant_concat_multi
Concatenate two multi-frequency arrayant vectors into a single multi-element model

### Description:
- Both inputs must have equal entry counts, identical angular grids, and matching `center_frequency` values at each index.
- Per frequency entry: pattern cubes are joined along the element (slice) dimension; `element_pos` matrices are horizontally concatenated (empty positions treated as zeros).
- Both inputs are validated with [arrayant_is_valid_multi](#arrayant_is_valid_multi) before processing; each output entry is validated before returning.
- Output inherits name, azimuth/elevation grids, and `center_frequency` from `arrayant_vec1`.
- Allowed datatypes: `float` or `double`

### Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::arrayant_concat_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec1,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec2);
```

### Input Arguments:
- **`arrayant_vec1`** — First validated, mutually consistent arrayant vector
- **`arrayant_vec2`** — Second arrayant vector; must match entry count, grids, and center frequencies of `arrayant_vec1`

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports per entry
- Coupling matrices are assembled block-diagonally — elements from `vec1` connect only to ports from `vec1` and vice versa:
  | Element \ Port | P1…Pp1 (vec1) | Pp1+1…Pp1+p2 (vec2) |
  |----------------|:-------------:|:--------------------:|
  | E1…En1 (vec1)  | C1 block      | 0                    |
  | En1+1…En1+n2 (vec2) | 0        | C2 block             |

### See also:
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (validation called on both inputs)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (position drivers before concatenating)
- [arrayant_rotate_pattern_multi](#arrayant_rotate_pattern_multi) (rotate elements after concatenating)
- [qdant_write_multi](#qdant_write_multi) (persist the combined model)

---
## arrayant_copy_element_multi
Copy an antenna element to one or more destinations across all entries in a multi-frequency arrayant vector

### Description:
- Calls `arrayant::copy_element` on every entry in the vector with the same source and destination indices.
- If any destination index exceeds the current element count, all entries are enlarged; new elements receive an identity coupling entry.
- Source and destination indices are 0-based.
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uvec destination);

void quadriga_lib::arrayant_copy_element_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        arma::uword source,
        arma::uword destination);
```

### Input Arguments:
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects; modified in-place
- **`source`** — 0-based index of the element to copy from; must be within current element count
- **`destination`** — 0-based index or indices of target elements; enlarges all entries if any index exceeds current count

### Example:
```
arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
auto driver = quadriga_lib::generate_speaker<double>(
    "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
    0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
quadriga_lib::arrayant_copy_element_multi(driver, 0, arma::uvec{1, 2, 3});
```

### See also:
- [.copy_element](#.copy_element) (per-entry operation called internally)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions after copying)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multiple arrayant vectors)

---
## arrayant_interpolate_multi
Interpolate multi-frequency arrayant patterns at arbitrary angles and frequencies

### Description:
- For each requested frequency, finds the two bracketing `center_frequency` entries, runs spatial interpolation on both via `qd_arrayant_interpolate`, then blends results in the frequency dimension.
- Frequency blending uses SLERP of complex field values with automatic fallback to linear interpolation when phase difference exceeds a threshold.
- Out-of-range frequencies are clamped to the nearest entry (no extrapolation).
- Consecutive frequency requests sharing the same bracketing entries reuse cached spatial interpolation results; sort `frequency` ascending or descending for best cache utilization.
- If `validate_input` is true, calls [arrayant_is_valid_multi](#arrayant_is_valid_multi) once before processing; set to `false` in performance-critical loops after initial validation.
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant_interpolate_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> *azimuth,
        const arma::Mat<dtype> *elevation,
        const arma::Col<dtype> *frequency,
        arma::Cube<dtype> *V_re,
        arma::Cube<dtype> *V_im,
        arma::Cube<dtype> *H_re,
        arma::Cube<dtype> *H_im,
        arma::uvec i_element = {},
        const arma::Cube<dtype> *orientation = nullptr,
        const arma::Mat<dtype> *element_pos_i = nullptr,
        bool validate_input = true);
```

### Input Arguments:
- **`arrayant_vec`** — Multi-frequency arrayant vector; entries need not be sorted by frequency
- **`azimuth`** — Azimuth angles in rad; must not be NULL, `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in rad; must not be NULL; size must match `azimuth`
- **`frequency`** — Target frequencies in Hz; must not be NULL or empty, `[n_freq]`
- **`i_element`** *(optional)* — 0-based element indices to interpolate; if empty, all elements are used (`n_out = n_elements`)
- **`orientation`** *(optional)* — Antenna orientation (bank, tilt, heading) in rad, applied at all frequencies; `[3,1,1]`, `[3,n_out,1]`, `[3,1,n_ang]`, or `[3,n_out,n_ang]`
- **`element_pos_i`** *(optional)* — Override element positions in m; if `nullptr`, positions from entry 0 are used, `[3, n_out]`
- **`validate_input`** *(optional)* — If `true`, validates `arrayant_vec` with [arrayant_is_valid_multi](#arrayant_is_valid_multi) before processing

### Output Arguments:
- **`V_re`** — Real part of interpolated e-theta field; must not be NULL, `[n_out, n_ang, n_freq]`
- **`V_im`** — Imaginary part of interpolated e-theta field; must not be NULL, `[n_out, n_ang, n_freq]`
- **`H_re`** — Real part of interpolated e-phi field; must not be NULL, `[n_out, n_ang, n_freq]`
- **`H_im`** — Imaginary part of interpolated e-phi field; must not be NULL, `[n_out, n_ang, n_freq]`

### Example:
```
auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
arma::mat el(1, 4, arma::fill::zeros);
arma::vec qf = {250.0, 1500.0, 8000.0};
arma::cube V_re, V_im, H_re, H_im;
quadriga_lib::arrayant_interpolate_multi(speaker, &az, &el, &qf, &V_re, &V_im, &H_re, &H_im);
```

### See also:
- [.interpolate](#.interpolate) (single-frequency spatial interpolation)
- [arrayant_concat_multi](#arrayant_concat_multi) (build multi-element/multi-frequency models)
- [arrayant_is_valid_multi](#arrayant_is_valid_multi) (validation called when validate_input is true)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
## arrayant_is_valid_multi
Validate a vector of arrayant objects for multi-frequency consistency

### Description:
- Each entry is validated individually via its `is_valid` member; `quick_check` is forwarded to that call.
- Cross-entry checks (all vs. entry 0): azimuth/elevation grid sizes and values, number of elements, element positions, coupling_re shape, and coupling_im presence and size.
- Pattern data, `center_frequency`, and coupling matrix values are not compared (expected to vary).
- Stops at first error and returns a message identifying the failing entry and property.
- Allowed datatypes: `float` or `double`

### Declaration:
```
std::string quadriga_lib::arrayant_is_valid_multi(
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        bool quick_check = true);
```

### Input Arguments:
- **`arrayant_vec`** — Non-empty vector of arrayant objects to validate
- **`quick_check`** *(optional)* — If `true`, uses fast pointer-based per-entry validation; if `false`, performs full deep validation

### Returns:
- Empty string if valid; otherwise a message such as `"Entry 3: Azimuth grid values do not match entry 0."`

### See also:
- [.is_valid](#.is_valid) (per-entry validation called internally)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
## arrayant_rotate_pattern_multi
Apply Euler rotations to all entries in a multi-frequency arrayant vector

### Description:
- Calls `arrayant::rotate_pattern` on every entry with grid adjustment always disabled (required for uniform-grid consistency across frequencies).
- If `i_element` is empty, all elements are rotated; otherwise only the specified 0-based indices are affected.
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid spurious polarization effects.
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant_rotate_pattern_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        dtype x_deg = 0.0,
        dtype y_deg = 0.0,
        dtype z_deg = 0.0,
        unsigned usage = 0,
        arma::uvec i_element = arma::uvec())
```

### Input Arguments:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`x_deg`** *(optional)* — Bank angle in degrees
- **`y_deg`** *(optional)* — Tilt angle in degrees
- **`z_deg`** *(optional)* — Heading angle in degrees
- **`usage`** *(optional)* — Rotation mode: `0` = pattern + polarization, `1` = pattern only, `2` = polarization only
- **`i_element`** *(optional)* — 0-based indices of elements to rotate; if empty, all elements are rotated

### See also:
- [.rotate_pattern](#.rotate_pattern) (per-entry operation called internally)
- [arrayant_concat_multi](#arrayant_concat_multi) (combine multi-frequency vectors before rotating)
- [arrayant_set_element_pos_multi](#arrayant_set_element_pos_multi) (set element positions in multi-frequency vectors)

---
## arrayant_set_element_pos_multi
Set element positions for all entries in a multi-frequency arrayant vector

### Description:
- Updates `element_pos` in-place on every entry in the vector identically.
- If `i_element` is empty, all positions are replaced and `element_pos` must have `n_elements` columns.
- If `i_element` is provided, only those 0-based indexed columns are updated; `element_pos` column count must match `i_element` length.
- All entries must have the same element count; uninitialized `element_pos` fields are zero-initialized before update.
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::arrayant_set_element_pos_multi(
        std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec,
        const arma::Mat<dtype> &element_pos,
        arma::uvec i_element = arma::uvec());
```

### Input Arguments:
- **`arrayant_vec`** — Non-empty vector of arrayant objects; modified in-place
- **`element_pos`** — New (x, y, z) positions in meters, `[3, n_update]`
- **`i_element`** *(optional)* — 0-based indices of elements to update; if empty, all elements are replaced

### See also:
- [arrayant_copy_element_multi](#arrayant_copy_element_multi) (replicate elements before setting positions)
- [generate_speaker](#generate_speaker) (typical source of multi-frequency arrayant vectors)

---
## generate_arrayant_3GPP
Generate a 3GPP-NR compliant antenna array model

### Description:
- Supports vertical (M) and horizontal (N) element stacking within panels, and multi-panel arrays (Mg × Ng).
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.
- Electrical downtilt (`tilt`) applies only to `pol` modes 4, 5, and 6.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(
                arma::uword M = 1, arma::uword N = 1, dtype center_freq = 299792458.0,
                unsigned pol = 1, dtype tilt = 0.0, dtype spacing = 0.5, arma::uword Mg = 1,
                arma::uword Ng = 1, dtype dgv = 0.5, dtype dgh = 0.5,
                const quadriga_lib::arrayant<dtype> *pattern = nullptr, dtype res = 1.0);
```

### Input Arguments:
- **`M`** *(optional)* — Number of vertical elements per panel
- **`N`** *(optional)* — Number of horizontal elements per panel
- **`center_freq`** *(optional)* — Center frequency in Hz
- **`pol`** *(optional)* — Polarization mode:

  | `pol` | Description | Elements |
  |-------|-------------|----------|
  | 1 | Vertical polarization | NM |
  | 2 | H/V polarization | 2NM |
  | 3 | ±45° polarization | 2NM |
  | 4 | Vertical, vertical elements combined | N |
  | 5 | H/V, vertical elements combined | 2N |
  | 6 | ±45°, vertical elements combined | 2N |

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

### Example:
```
auto ant = quadriga_lib::generate_arrayant_3GPP<double>(4, 4, 3e9, 2);
```

---
## generate_arrayant_custom
Generate an antenna with custom 3dB beamwidth

### Description:
- Returns a single-element antenna with independently configurable azimuth and elevation 3dB (FWHM) beamwidths.
- Rear-side gain is controlled by a linear front-to-back ratio; `0.0` means no rear radiation.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(dtype az_3dB = 90.0,
                dtype el_3db = 90.0, dtype rear_gain_lin = 0.0, dtype res = 1.0);
```

### Input Arguments:
- **`az_3dB`** *(optional)* — Azimuth 3dB beamwidth in degrees
- **`el_3db`** *(optional)* — Elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna object with specified beamwidth and rear gain

---
## generate_arrayant_dipole
Generate a short dipole antenna with vertical polarization

### Description:
- Returns a single-element short dipole antenna pattern with vertical polarization.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole(dtype res = 1.0);
```

### Input Arguments:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized short dipole antenna object

---
## generate_arrayant_half_wave_dipole
Generate a half-wave dipole antenna with vertical polarization

### Description:
- Returns a single-element half-wave dipole antenna pattern with vertical polarization.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole(dtype res = 1.0);
```

### Input Arguments:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Vertically polarized half-wave dipole antenna object

---
## generate_arrayant_multibeam
Generate a planar multi-element antenna array with multiple beam directions

### Description:
- Returns an M×N planar array with beamforming weights computed via maximum-ratio transmission (MRT).
- MRT is optimal for a single beam; approximate when multiple beams are specified.
- Weights control relative beam contribution; only their ratios matter, not absolute values.
- If `separate_beams = true`, each angle pair produces an independent beam (weights ignored).
- If `apply_weights = true`, beamforming weights are baked into the element coupling matrix.
- Per-element pattern shape is controlled by `az_3dB`, `el_3dB`, and `rear_gain_lin`; see [generate_arrayant_custom](#generate_arrayant_custom).
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
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

### Input Arguments:
- **`M`** *(optional)* — Number of vertical (row) elements
- **`N`** *(optional)* — Number of horizontal (column) elements
- **`az`** *(optional)* — Azimuth beam angles in degrees, `[n_beams]`
- **`el`** *(optional)* — Elevation beam angles in degrees, `[n_beams]`
- **`weight`** *(optional)* — Per-beam scaling factors (normalized to sum = 1), `[n_beams]`
- **`center_freq`** *(optional)* — Center frequency in Hz
- **`pol`** *(optional)* — Polarization mode:

  | `pol` | Description | Elements |
  |-------|-------------|----------|
  | 1 | Vertical polarization | NM |
  | 2 | H/V polarization | 2NM |
  | 3 | ±45° polarization | 2NM |

- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`az_3dB`** *(optional)* — Per-element azimuth 3dB beamwidth in degrees
- **`el_3dB`** *(optional)* — Per-element elevation 3dB beamwidth in degrees
- **`rear_gain_lin`** *(optional)* — Per-element front-to-back gain ratio (linear)
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees
- **`separate_beams`** *(optional)* — If `true`, generate one independent beam per angle pair
- **`apply_weights`** *(optional)* — If `true`, bake beamforming weights into the coupling matrix

### Returns:
- `quadriga_lib::arrayant<dtype>` — Multibeam planar array antenna object

### Example:
```
arma::vec az = {20.0, 0.0}, el = {-7.0, 30.0}, weight = {2.0, 1.0};
auto ant = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, weight, 3.75e9);
```

---
## generate_arrayant_omni
Generate an isotropic radiator with vertical polarization

### Description:
- Returns a single-element antenna array with omnidirectional pattern and vertical polarization.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni(dtype res = 1.0);
```

### Input Arguments:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Isotropic radiator antenna object

---
## generate_arrayant_ula
Generate a uniform linear array (ULA)

### Description:
- Returns a horizontally stacked linear array of N elements with half-wavelength spacing by default.
- Default per-element pattern is a vertically polarized isotropic radiator.
- If `pattern` is provided, its radiation pattern is used for each element; element positions, coupling, and center frequency from `pattern` are ignored.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_ula(
                arma::uword N = 1, dtype center_freq = 299792458.0, dtype spacing = 0.5,
                const quadriga_lib::arrayant<dtype> *pattern = nullptr, dtype res = 1.0);
```

### Input Arguments:
- **`N`** *(optional)* — Number of elements
- **`center_freq`** *(optional)* — Center frequency in Hz
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths
- **`pattern`** *(optional)* — Custom per-element antenna pattern; overrides default isotropic pattern
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; ignored if `pattern` is provided

### Returns:
- `quadriga_lib::arrayant<dtype>` — ULA antenna array object

---
## generate_arrayant_xpol
Generate a cross-polarized isotropic radiator

### Description:
- Returns a two-element antenna array with omnidirectional patterns in vertical and horizontal polarization.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol(dtype res = 1.0);
```

### Input Arguments:
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees

### Returns:
- `quadriga_lib::arrayant<dtype>` — Cross-polarized isotropic radiator antenna object

---
## generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

### Description:
- Returns one `quadriga_lib::arrayant` per frequency sample; each has a single element with the real-valued directivity balloon in `e_theta_re` and `center_frequency` set to the corresponding frequency in Hz.
- Multi-driver systems (e.g. two-way) are built by calling this function per driver and combining results via `append` and `element_pos`; crossover behavior emerges from overlapping bandpass responses.
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) * 1/sqrt(1+(f/f_high)^(2n))`, where `n = slope_dB_per_octave / 6`; −3 dB at the cutoff frequencies.
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity − 85) / 20)`.
- If `frequencies` is empty, third-octave band center frequencies are auto-generated from one band below `lower_cutoff` to one band above `upper_cutoff`, clipped to 20–20000 Hz.
- Speed of sound assumed to be 344 m/s.
- **Driver models** (`driver_type`): `"piston"` — circular piston in baffle, `D(θ) = 2*J1(ka*sinθ)/(ka*sinθ)`, rotationally symmetric, narrows with increasing `ka`; `"horn"` — separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`; `"omni"` — frequency-independent omnidirectional pattern.
- **Enclosure models** (`radiation_type`): `"monopole"` — no modification; `"hemisphere"` — sealed box with baffle-step transition, `f_baffle = c/(π*sqrt(W*H))`; `"dipole"` — figure-8, `R = abs(cos(θ_off))` with sign inversion in rear hemisphere; `"cardioid"` — `R = 0.5*(1+cos(θ_off))`.
- For `"horn"`, if `horn_control_freq = 0`, it is auto-derived as `f_ctrl = c/(2π*radius)`.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
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

### Input Arguments:
- **`driver_type`** *(optional)* — Driver directivity model: `"piston"`, `"horn"`, or `"omni"`
- **`radius`** *(optional)* — Effective radiating radius in meters; cone/dome radius for piston, mouth radius for horn
- **`lower_cutoff`** *(optional)* — Lower −3 dB bandpass frequency in Hz
- **`upper_cutoff`** *(optional)* — Upper −3 dB bandpass frequency in Hz
- **`lower_rolloff_slope`** *(optional)* — Low-frequency rolloff in dB/octave (12 dB/oct = 2nd-order Butterworth)
- **`upper_rolloff_slope`** *(optional)* — High-frequency rolloff in dB/octave
- **`sensitivity`** *(optional)* — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude
- **`radiation_type`** *(optional)* — Enclosure radiation model: `"monopole"`, `"hemisphere"`, `"dipole"`, or `"cardioid"`
- **`hor_coverage`** *(optional)* — Horn horizontal coverage angle in degrees; `0` defaults to 90°
- **`ver_coverage`** *(optional)* — Horn vertical coverage angle in degrees; `0` defaults to 60°
- **`horn_control_freq`** *(optional)* — Horn pattern control frequency in Hz; `0` auto-derives from `radius`
- **`baffle_width`** *(optional)* — Baffle width in meters; used by `"hemisphere"` model
- **`baffle_height`** *(optional)* — Baffle height in meters; used by `"hemisphere"` model
- **`frequencies`** *(optional)* — Frequency sample points in Hz; auto-generated third-octave bands if empty, `[n_freq]`
- **`angular_resolution`** *(optional)* — Azimuth and elevation sampling grid resolution in degrees

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One arrayant per frequency sample with directivity in `e_theta_re`; dipole rear hemisphere encoded with negative sign for 180° phase inversion

### Example:
```
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
               12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
auto horn = quadriga_lib::generate_speaker<double>("horn");
auto sub = quadriga_lib::generate_speaker<double>("omni", 0.13, 30.0, 200.0,
               12.0, 24.0, 92.0, "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, {30.,50.,80.,120.,200.}, 10.0);
```

---
## qdant_read
Read an arrayant object from a QDANT file

### Description:
- Parses a QuaDRiGa Array Antenna Exchange Format (QDANT) XML file and returns the arrayant for the given ID.
- Allowed datatypes: `float` or `double`

### Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(
        std::string fn,
        unsigned id = 1,
        arma::u32_mat *layout = nullptr);
```

### Input Arguments:
- **`fn`** — Path to the QDANT file; must not be empty
- **`id`** *(optional)* — 1-based ID of the antenna entry to read
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix of element IDs

### Returns:
- `quadriga_lib::arrayant<dtype>` constructed from the specified entry in the file

### See also:
- [.qdant_write](#.qdant_write) (write a single arrayant)
- [qdant_write_multi](#qdant_write_multi) (write multiple arrayants with sequential IDs)

---
## qdant_read_multi
Read all arrayant objects from a QDANT file into a vector

### Description:
- Reads all entries from a QDANT file by probing ID 1 to obtain the layout, then reading each unique non-zero ID in order of first appearance (column-major scan).
- Each unique ID is read exactly once regardless of how many times it appears in the layout.
- Counterpart to [qdant_write_multi](#qdant_write_multi); primary mechanism for loading frequency-dependent models where `center_frequency` on each entry identifies the corresponding frequency.
- Allowed datatypes: `float` or `double`

### Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read_multi(
        const std::string &fn,
        arma::u32_mat *layout = nullptr);
```

### Input Arguments:
- **`fn`** — Path to the QDANT file; must not be empty
- **`layout`** *(optional)* — Output pointer filled with the file's layout matrix; non-zero values are entry IDs

### Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per unique ID, ordered by first appearance in the layout

### See also:
- [qdant_read](#qdant_read) (read a single entry by ID)
- [qdant_write_multi](#qdant_write_multi) (write a vector of arrayants)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---
## qdant_write_multi
Write a vector of arrayant objects to a single QDANT file

### Description:
- Writes each entry in `arrayant_vec` to a QDANT file with sequential 1-based IDs using [.qdant_write](#.qdant_write).
- Auto-generates a `[n_entries, 1]` layout matrix with entries `1, 2, ..., n_entries`.
- Deletes any existing file before writing; all entries are validated first.
- Primary use case: frequency-dependent models where each arrayant holds a pattern at one frequency via `center_frequency`.
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);
```

### Input Arguments:
- **`fn`** — Path of the QDANT file to write; must not be empty
- **`arrayant_vec`** — Non-empty vector of valid arrayant objects to store

### Example:
```
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
quadriga_lib::qdant_write_multi("speaker.qdant", spk);
auto ant = quadriga_lib::qdant_read<double>("speaker.qdant", 3); // read 3rd entry
```

### See also:
- [.qdant_write](#.qdant_write) (per-object write used internally)
- [qdant_read](#qdant_read) (read back individual entries by ID)
- [generate_speaker](#generate_speaker) (typical source of frequency-dependent arrayant vectors)

---

# Channel class

| Function | Description |
| --- | --- |
| [channel](#channel) | Class for storing and managing MIMO channel data and metadata across multiple snapshots |
| [.add_paths](#add_paths) | Append new propagation paths to an existing channel snapshot |
| [.calc_effective_path_gain](#calc_effective_path_gain) | Calculate the effective path gain per snapshot in linear scale |
| [.write_paths_to_obj_file](#write_paths_to_obj_file) | Export propagation paths to a Wavefront OBJ file for 3D visualization |

---
## channel
Class for storing and managing MIMO channel data and metadata across multiple snapshots

### Description:
- Represents path-level MIMO channel data between antenna arrays over multiple time snapshots
- Each snapshot may have a different number of propagation paths `n_path`
- Unstructured metadata supported via `par_names` / `par_data`
- Allowed datatypes: `float` or `double`

### Attributes:
| Attribute | Size | Description |
|-----------|------|-------------|
| `std::string name` | — | Name of the channel object |
| `arma::Col<dtype> center_frequency` | `[1]`, `[n_snap]`, or `[]` | Center frequency in Hz |
| `arma::Mat<dtype> tx_pos` | `[3, n_snap]` or `[3, 1]` | Transmitter positions |
| `arma::Mat<dtype> rx_pos` | `[3, n_snap]` or `[3, 1]` | Receiver positions |
| `arma::Mat<dtype> tx_orientation` | `[3, n_snap]`, `[3, 1]`, or `[]` | Transmitter orientation (Euler angles) |
| `arma::Mat<dtype> rx_orientation` | `[3, n_snap]`, `[3, 1]`, or `[]` | Receiver orientation (Euler angles) |
| `std::vector<arma::Cube<dtype>> coeff_re` | `[n_rx, n_tx, n_path]` per snap | Channel coefficients, real part |
| `std::vector<arma::Cube<dtype>> coeff_im` | `[n_rx, n_tx, n_path]` per snap | Channel coefficients, imaginary part |
| `std::vector<arma::Cube<dtype>> delay` | `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` per snap | Path delays in seconds |
| `std::vector<arma::Col<dtype>> path_gain` | `[n_path]` per snap | Path gains before antenna pattern |
| `std::vector<arma::Col<dtype>> path_length` | `[n_path]` per snap | Path lengths TX to RX in meters |
| `std::vector<arma::Mat<dtype>> path_polarization` | `[8, n_path]` per snap | Interleaved polarization transfer matrices |
| `std::vector<arma::Mat<dtype>> path_angles` | `[n_path, 4]` per snap | Angles {AOD, EOD, AOA, EOA} in rad |
| `std::vector<arma::Mat<dtype>> path_fbs_pos` | `[3, n_path]` per snap | First-bounce scatterer positions |
| `std::vector<arma::Mat<dtype>> path_lbs_pos` | `[3, n_path]` per snap | Last-bounce scatterer positions |
| `std::vector<arma::Col<unsigned>> no_interact` | `[n_path]` per snap | Number of interactions per path |
| `std::vector<arma::Mat<dtype>> interact_coord` | `[3, sum(no_interact)]` per snap | Interaction point coordinates |
| `std::vector<std::string> par_names` | — | Names of unstructured metadata fields |
| `std::vector<std::any> par_data` | — | Unstructured metadata values (string, scalar, matrix, etc.) |
| `int initial_position` | scalar | 0-based index of the reference snapshot |

### Simple member functions:
| Method | Description |
|---|---|
| `.n_snap()` | Returns the number of snapshots |
| `.n_rx()` | Returns number of receive antennas; 0 if coefficients absent |
| `.n_tx()` | Returns number of transmit antennas; 0 if coefficients absent |
| `.n_path()` | Returns number of paths per snapshot as a vector |
| `.empty()` | Returns true if the object contains no channel data |
| `.is_valid()` | Returns empty string if valid, otherwise an error message |

### Complex member functions:
- [.add_paths](#.add_paths)
- [.calc_effective_path_gain](#.calc_effective_path_gain)
- [.write_paths_to_obj_file](#.write_paths_to_obj_file)

---
## .add_paths
Append new propagation paths to an existing channel snapshot

### Description:
- Adds path-level data to snapshot `i_snap` in a `channel` object; does not modify `tx_pos`, `rx_pos`, or orientation fields
- All provided fields must have consistent length `n_path_add` and match existing snapshot structure
- Member function of [channel](#channel)
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::channel<dtype>::add_paths(
    arma::uword i_snap,
    const arma::Cube<dtype> *coeff_re_add = nullptr,
    const arma::Cube<dtype> *coeff_im_add = nullptr,
    const arma::Cube<dtype> *delay_add = nullptr,
    const arma::u32_vec *no_interact_add = nullptr,
    const arma::Mat<dtype> *interact_coord_add = nullptr,
    const arma::Col<dtype> *path_gain_add = nullptr,
    const arma::Col<dtype> *path_length_add = nullptr,
    const arma::Mat<dtype> *path_polarization_add = nullptr,
    const arma::Mat<dtype> *path_angles_add = nullptr,
    const arma::Mat<dtype> *path_fbs_pos_add = nullptr,
    const arma::Mat<dtype> *path_lbs_pos_add = nullptr);
```

### Input Arguments:
- **`i_snap`** — 0-based snapshot index to append paths to
- **`coeff_re_add`** *(optional)* — Real part of channel coefficients, `[n_rx, n_tx, n_path_add]`
- **`coeff_im_add`** *(optional)* — Imaginary part of channel coefficients, `[n_rx, n_tx, n_path_add]`
- **`delay_add`** *(optional)* — Propagation delays in seconds, `[n_rx, n_tx, n_path_add]` or `[1, 1, n_path_add]`
- **`no_interact_add`** *(optional)* — Number of interaction points per path, `[n_path_add]`
- **`interact_coord_add`** *(optional)* — Interaction point coordinates, `[3, sum(no_interact)]`
- **`path_gain_add`** *(optional)* — Path gains before antenna effects, `[n_path_add]`
- **`path_length_add`** *(optional)* — Path lengths from TX to RX phase center in meters, `[n_path_add]`
- **`path_polarization_add`** *(optional)* — Interleaved polarization transfer matrices, `[8, n_path_add]`
- **`path_angles_add`** *(optional)* — Departure/arrival angles {AOD, EOD, AOA, EOA} in rad, `[n_path_add, 4]`
- **`path_fbs_pos_add`** *(optional)* — First-bounce scatterer positions, `[3, n_path_add]`
- **`path_lbs_pos_add`** *(optional)* — Last-bounce scatterer positions, `[3, n_path_add]`

---
## .calc_effective_path_gain
Calculate the effective path gain per snapshot in linear scale

### Description:
- Sums power over all paths and TX/RX antenna pairs to produce one gain value per snapshot
- Uses `coeff_re`/`coeff_im` if available; falls back to `path_polarization` assuming ideal XPOL antennas
- Throws if neither coefficients nor polarization data are present
- Member function of [channel](#channel)
- Allowed datatypes: `float` or `double`

### Declaration:
```
arma::Col<dtype> quadriga_lib::channel<dtype>::calc_effective_path_gain(bool assume_valid = false) const;
```

### Input Arguments:
- **`assume_valid`** *(optional)* — Skip internal consistency checks for performance in trusted contexts

### Returns:
- Effective path gains in linear scale, one entry per snapshot, `[n_snap]`

---
## .write_paths_to_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

### Description:
- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for Blender)
- Tubes are color-coded by path gain using a selected colormap; radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits total count
- Member function of [channel](#channel)
- Allowed datatypes: `float` or `double`

### Declaration:
```
void quadriga_lib::channel<dtype>::write_paths_to_obj_file(
    std::string fn,
    arma::uword max_no_paths = 0,
    dtype gain_max = -60.0,
    dtype gain_min = -140.0,
    std::string colormap = "jet",
    arma::uvec i_snap = {},
    dtype radius_max = 0.05,
    dtype radius_min = 0.01,
    arma::uword n_edges = 5) const;
```

### Input Arguments:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** *(optional)* — Max paths to export; `0` includes all paths above `gain_min`
- **`gain_max`** *(optional)* — Upper gain threshold in dB for color/radius mapping; higher values are clipped
- **`gain_min`** *(optional)* — Lower gain threshold in dB; paths below this are excluded
- **`colormap`** *(optional)* — Colormap name; see [colormap](#colormap) for supported options
- **`i_snap`** *(optional)* — 0-based snapshot indices to include; empty exports all snapshots
- **`radius_max`** *(optional)* — Tube radius in meters at maximum gain
- **`radius_min`** *(optional)* — Tube radius in meters at minimum gain
- **`n_edges`** *(optional)* — Vertices per tube cross-section; must be ≥ 3

### See also:
- [path_to_tube](#path_to_tube) (generates tube geometry from path data)
- [colormap](#colormap) (colormap lookup used for coloring)

---

# Channel functions

| Function | Description |
| --- | --- |
| [any_type_id](#any_type_id) | Get type ID and raw access from a 'std::any' object |
| [baseband_freq_response](#baseband_freq_response) | Compute the baseband frequency response of a MIMO channel |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients |
| [baseband_freq_response_vec](#baseband_freq_response_vec) | Compute the baseband frequency response of multiple MIMO channels |
| [get_HDF5_version](#get_hdf5_version) | Get the version of the linked HDF5 library |
| [hdf5_create](#hdf5_create) | Create a new HDF5 channel file with a defined storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Read a channel object from an HDF5 file |
| [hdf5_read_dset](#hdf5_read_dset) | Read an unstructured dataset from an HDF5 file |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Read names of unstructured datasets from an HDF5 file |
| [hdf5_read_layout](#hdf5_read_layout) | Read the HDF5 channel storage layout |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Reshape the storage layout of an HDF5 channel file |
| [hdf5_write](#hdf5_write) | Write channel data to HDF5 file |
| [hdf5_write_dset](#hdf5_write_dset) | Write a single unstructured dataset to an HDF5 file |
| [qrt_file_parse](#qrt_file_parse) | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Read ray-tracing data from a QRT file |
| [qrt_read_cache_init](#qrt_read_cache_init) | Initialize a QRT read cache for fast repeated access |
| [quantize_delays](#quantize_delays) | Fixes the path delays to a grid of delay bins |

---
## any_type_id
Get type ID and raw access from a 'std::any' object

### Description:
- Inspects a `std::any` object and returns a type identifier for its contents.
- Optionally retrieves the dimensions of the object (if it is a matrix, vector, or cube).
- Optionally retrieves a raw pointer to the internal data.
- **Warning:** Accessing data through `dataptr` is not type-safe and bypasses `const` protection. Use with extreme caution.

### Declaration:
```
int quadriga_lib::any_type_id(
                const std::any *data,
                unsigned long long *dims = nullptr,
                void **dataptr = nullptr);
```

### Arguments:
- `const std::any ***data**` (input)
  Pointer to the `std::any` object to inspect.

- `unsigned long long ***dims** = nullptr` (optional output)
  Pointer to an array of 3 integers that will hold the dimensions of the object:
  `dims[0]`, `dims[1]`, `dims[2]`: Number of rows, columns, slices (for Armadillo types).
  For `std::string`, `dims[0]` contains the string length, `dims[1]` and `dims[2]` are zero.

- `void **\*\*dataptr** = nullptr` (optional output)
  If not `nullptr`, returns a raw pointer to the object's internal data.  This allows direct access, **without** type safety or `const` protection.

### Returns:
- `int`
  Type ID corresponding to the content of the `std::any` object. Values are:
    ID | Type                       | ID | Type                       | ID | Type
    ---|----------------------------|----|----------------------------|----|------------------------
    -2 | `no value`                 | -1 | `unsupported type`         |  9 | `std::string`
    10 | `float`                    | 11 | `double`                   | 12 | `unsigned long long int`
    13 | `long long int`            | 14 | `unsigned int`             | 15 | `int`
    20 | `arma::Mat<float>`         | 21 | `arma::Mat<double>`        | 22 | `arma::Mat<arma::uword>`
    23 | `arma::Mat<arma::sword>`   | 24 | `arma::Mat<unsigned>`      | 25 | `arma::Mat<int>`
    30 | `arma::Cube<float>`        | 31 | `arma::Cube<double>`       | 32 | `arma::Cube<arma::uword>`
    33 | `arma::Cube<arma::sword>`  | 34 | `arma::Cube<unsigned>`     | 35 | `arma::Cube<int>`
    40 | `arma::Col<float>`         | 41 | `arma::Col<double>`        | 42 | `arma::Col<arma::uword>`
    43 | `arma::Col<arma::sword>`   | 44 | `arma::Col<unsigned>`      | 45 | `arma::Col<int>`
    50 | `arma::Row<float>`         | 51 | `arma::Row<double>`        | 52 | `arma::Row<arma::uword>`
    53 | `arma::Row<arma::sword>`   | 54 | `arma::Row<unsigned>`      | 55 | `arma::Row<int>`

---
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

### Description:
- Computes the frequency-domain response of a time-domain MIMO channel using a discrete Fourier transform (DFT).
- Input consists of real and imaginary channel coefficients and corresponding delays for each MIMO sub-link.
- Outputs the complex channel response matrix `H` at given sub-carrier frequency positions.
- Internally uses AVX2 instructions for fast parallel computation of 8 carriers at once.
- Can be efficiently called in a loop (e.g., over snapshots) and parallelized with OpenMP.
- Internal arithmetic is performed in single precision for speed.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::baseband_freq_response(
                const arma::Cube<dtype> *coeff_re,
                const arma::Cube<dtype> *coeff_im,
                const arma::Cube<dtype> *delay,
                const arma::Col<dtype> *pilot_grid,
                const double bandwidth,
                arma::Cube<dtype> *hmat_re,
                arma::Cube<dtype> *hmat_im);
```

### Arguments:
- `const arma::Cube<dtype> ***coeff_re**` (input)
  Real part of channel coefficients in time domain, Size `[n_rx, n_tx, n_path]`.

- `const arma::Cube<dtype> ***coeff_im**` (input)
  Imaginary part of channel coefficients in time domain, Size `[n_rx, n_tx, n_path]`.

- `const arma::Cube<dtype> ***delay**` (input)
  Path delays in seconds. Size `[n_rx, n_tx, n_path]` or broadcastable shape `[1, 1, n_path]`.

- `const arma::Col<dtype> ***pilot_grid**` (input)
  Normalized sub-carrier positions relative to bandwidth. Range: `0.0` (center freq) to `1.0` (center + bandwidth). Length: `n_carriers`.

- `const double **bandwidth**` (input)
  Total baseband bandwidth in Hz (defines absolute frequency spacing of the pilot grid).

- `arma::Cube<dtype> ***hmat_re**` (output)
  Output: Real part of the frequency-domain channel matrix, Size `[n_rx, n_tx, n_carriers]`.

- `arma::Cube<dtype> ***hmat_im**` (output)
  Output: Imaginary part of the frequency-domain channel matrix, Size `[n_rx, n_tx, n_carriers]`.

---
## baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

### Description:
- Computes the frequency-domain channel transfer function H(f) from multi-frequency channel coefficients
  obtained by sampling the channel at a coarse set of carrier frequencies (e.g., every 500 MHz).
- Designed for ultra-wideband (UWB) channels where antenna patterns, path gains, and polarization transfer
  matrices vary across a large bandwidth (e.g., 500 MHz to 20 GHz).
- For each multipath component (MPC), the complex coefficient envelope is interpolated from the coarse
  input frequency grid `freq_in` to the dense output frequency grid `freq_out` using SLERP (Spherical
  Linear Interpolation): magnitude is interpolated linearly, phase is unwrapped and interpolated linearly
  along the shortest arc.
- The delay-induced phase rotation `exp(-j * 2 * pi * freq_out * delay)` is applied per output carrier.
  Since the output frequencies are absolute (not baseband offsets), the phase computation uses double
  precision internally to avoid loss of accuracy at high carrier frequencies.
- **Delay phase removal** (`remove_delay_phase = true`, default): Channel generation functions such as
  `get_channels_multifreq` bake the delay-induced phase `exp(-j * 2 * pi * freq_in[f] * delay)` into
  the output coefficients at each input frequency. This means the coefficient phase rotates rapidly across
  the input frequency grid (e.g., 50 full cycles per 500 MHz step for a 100 ns excess delay), which
  prevents meaningful SLERP interpolation. When `remove_delay_phase` is enabled, this function undoes the
  baked-in delay phase by multiplying each coefficient by `exp(+j * 2 * pi * freq_in[f] * delay)` before
  extracting the slowly-varying envelope for interpolation. The full delay phase at the output frequency
  is then re-applied analytically. This must be enabled when consuming output from `get_channels_multifreq`
  or `get_channels_spherical`. Set to `false` only if the input coefficients are already pure envelopes
  without baked-in delay phase (e.g., from manual construction or external sources).
- Delays are taken from `delay[0]` only, since path geometry (and thus propagation delay) is
  frequency-independent. All entries in the `delay` vector should contain identical values.
- The delay cube supports broadcastable shape `[1, 1, n_path]` (planar wave, same delay for all antenna
  pairs) or full shape `[n_rx, n_tx, n_path]` (spherical wave, per-antenna-pair delays).
- For output frequencies outside the range of `freq_in`, the nearest endpoint coefficient is used
  (constant extrapolation in magnitude/phase domain).
- The Nyquist condition for the coarse sampling grid applies to the coefficient envelope only (magnitude
  and phase of antenna patterns, path gains, and Jones matrix elements). The fast-rotating delay phase is
  handled analytically and does not require dense sampling. Adjacent envelope phase differences should stay
  well below pi to avoid interpolation artifacts.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::baseband_freq_response_multi(
                const std::vector<arma::Cube<dtype>> &coeff_re,
                const std::vector<arma::Cube<dtype>> &coeff_im,
                const std::vector<arma::Cube<dtype>> &delay,
                const arma::Col<dtype> &freq_in,
                const arma::Col<dtype> &freq_out,
                arma::Cube<dtype> *hmat_re = nullptr,
                arma::Cube<dtype> *hmat_im = nullptr,
                arma::Cube<std::complex<dtype>> *hmat = nullptr,
                bool remove_delay_phase = true);
```

### Arguments:
- `const std::vector<arma::Cube<dtype>> &**coeff_re**` (input)
  Real part of channel coefficients at each input frequency. Vector of length `n_freq_in`, each cube
  has shape `[n_rx, n_tx, n_path]`.

- `const std::vector<arma::Cube<dtype>> &**coeff_im**` (input)
  Imaginary part of channel coefficients at each input frequency. Same structure as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> &**delay**` (input)
  Path delays in seconds. Vector of length `n_freq_in`. Only `delay[0]` is used (delays are
  frequency-independent). Shape `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`.

- `const arma::Col<dtype> &**freq_in**` (input)
  Input sample frequencies in Hz, sorted in ascending order. Length `[n_freq_in]`.

- `const arma::Col<dtype> &**freq_out**` (input)
  Output carrier frequencies in Hz (absolute, not baseband offsets). Length `[n_carrier]`.

- `arma::Cube<dtype> ***hmat_re** = nullptr` (optional output)
  Real part of the frequency-domain channel matrix. Size `[n_rx, n_tx, n_carrier]`.

- `arma::Cube<dtype> ***hmat_im** = nullptr` (optional output)
  Imaginary part of the frequency-domain channel matrix. Size `[n_rx, n_tx, n_carrier]`.

- `arma::Cube<std::complex<dtype>> ***hmat** = nullptr` (optional output)
  Complex-valued frequency-domain channel matrix. Size `[n_rx, n_tx, n_carrier]`.

- `bool **remove_delay_phase** = true` (optional input)
  If `true` (default), the delay-induced phase `exp(-j * 2 * pi * freq_in[f] * delay)` that is baked
  into the input coefficients by channel generation functions (e.g., `get_channels_multifreq`) is removed
  before SLERP interpolation and re-applied at each output frequency. Must be enabled when the input
  comes from `get_channels_multifreq` or `get_channels_spherical`. Set to `false` only when the input
  coefficients contain pure slowly-varying envelopes without delay phase.

### Example:
```
// Using output from get_channels_multifreq (delay phase is baked in)
std::vector<arma::Cube<double>> coeff_re, coeff_im, delays;
arma::Col<double> freq_in = {0.5e9, 1.0e9, 1.5e9, 2.0e9};
arma::Col<double> freq_out = arma::linspace<arma::Col<double>>(0.5e9, 2.0e9, 2048);

// ... call get_channels_multifreq to populate coeff_re, coeff_im, delays ...

arma::Cube<double> Hr, Hi;
quadriga_lib::baseband_freq_response_multi(coeff_re, coeff_im, delays,
    freq_in, freq_out, &Hr, &Hi, nullptr, true);  // remove_delay_phase = true (default)
```

### See also:
- [baseband_freq_response](#baseband_freq_response) (for single-frequency narrowband channels)
- [baseband_freq_response_vec](#baseband_freq_response_vec) (for batched narrowband channels)
- [get_channels_multifreq](#get_channels_multifreq) (produces the multi-frequency coefficients consumed by this function)

---
## baseband_freq_response_vec
Compute the baseband frequency response of multiple MIMO channels

### Description:
- Computes the frequency-domain response of a batch of time-domain MIMO channels using a discrete Fourier transform (DFT).
- This function wraps `quadriga_lib::baseband_freq_response` and applies it across multiple snapshots in parallel using OpenMP.
- Input consists of vectors of real/imaginary coefficients and delay Cubes for each snapshot.
- Output is a vector of frequency-domain channel matrices `H` (one per snapshot).
- Can optionally compute a selected subset of snapshots using `i_snap`.
- Internal arithmetic is performed in single precision for performance.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::baseband_freq_response_vec(
                const std::vector<arma::Cube<dtype>> *coeff_re,
                const std::vector<arma::Cube<dtype>> *coeff_im,
                const std::vector<arma::Cube<dtype>> *delay,
                const arma::Col<dtype> *pilot_grid,
                const double bandwidth,
                std::vector<arma::Cube<dtype>> *hmat_re,
                std::vector<arma::Cube<dtype>> *hmat_im,
                const arma::u32_vec *i_snap = nullptr);
```

### Arguments:
- `const std::vector<arma::Cube<dtype>> ***coeff_re**` (input)
  Real part of channel coefficients, vector of length `n_snap`. Each cube has shape `[n_rx, n_tx, n_path]`.

- `const std::vector<arma::Cube<dtype>> ***coeff_im**` (input)
  Imaginary part of channel coefficients, same structure as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> ***delay**` (input)
  Path delays in seconds, same structure as `coeff_re`, shape can be broadcasted `[1, 1, n_path]`.

- `const arma::Col<dtype> ***pilot_grid**` (input)
  Normalized sub-carrier positions relative to bandwidth. Range: `0.0` (center freq) to `1.0` (center + bandwidth). Length: `n_carriers`.

- `const double **bandwidth**` (input)
  Total baseband bandwidth in Hz, used to compute sub-carrier frequencies.

- `std::vector<arma::Cube<dtype>> ***hmat_re**` (output)
  Output: Real part of the frequency-domain channel matrices. Vector of length `n_out`. Each cube is `[n_rx, n_tx, n_carriers]`.

- `std::vector<arma::Cube<dtype>> ***hmat_im**` (output)
  Output: Imaginary part of the frequency-domain channel matrices. Same structure as `hmat_re`.

- `const arma::u32_vec ***i_snap** = nullptr` (optional input)
  Optional subset of snapshot indices to process. If omitted, all `n_snap` snapshots are processed. Length: `n_out`.

### See also:
- [baseband_freq_response](#baseband_freq_response) (for processing a single snapshot)

---
## get_HDF5_version
Get the version of the linked HDF5 library

### Description:
- Returns the version string of the HDF5 library used during linking.
- This function is useful for diagnostics, compatibility checks, or logging.
- The function reflects the version of the compiled HDF5 library, not necessarily the version of the header used at compile time.

### Declaration:
```
std::string quadriga_lib::get_HDF5_version();
```

### Returns:
- `std::string`
  HDF5 version string in the format `"x.y.z"`, e.g., `"1.12.2"`

### Example:
```
std::string hdf5_ver = quadriga_lib::get_HDF5_version();
std::cout << "Using HDF5 version: " << hdf5_ver << std::endl;
```

---
## hdf5_create
Create a new HDF5 channel file with a defined storage layout

### Description:
- Quadriga-Lib offers an HDF5-based method for storing and managing channel data. A key feature of this
  library is its ability to organize multiple channels within a single HDF5 file while enabling access
  to individual data sets without the need to read the entire file.
- This function initializes a new HDF5 file for storing wireless channel data.
- Defines a multi-dimensional array layout for organizing channels (up to 4 dimensions).
- Typical usage: map base stations (BS), user equipment (UE), and frequencies to dimensions.
- The layout can later be reshaped using `hdf5_reshape_layout`, provided the total number of entries remains constant.
- Each combination of indices corresponds to a storage slot that can hold a channel object.

### Declaration:
```
void quadriga_lib::hdf5_create(
                std::string fn,
                unsigned nx = 65536,
                unsigned ny = 1,
                unsigned nz = 1,
                unsigned nw = 1);
```

### Arguments:
- `std::string **fn**` (input)
  Filename (including path) of the HDF5 file to be created. If the file exists, it will be overwritten.

- `unsigned **nx** = 65536` (input)
  Number of entries in the **x-dimension**, e.g., for base stations (BSs). Default: `65536`.

- `unsigned **ny** = 1` (input)
  Number of entries in the **y-dimension**, e.g., for user equipment (UEs). Default: `1`.

- `unsigned **nz** = 1` (input)
  Number of entries in the **z-dimension**, e.g., for frequency points. Default: `1`.

- `unsigned **nw** = 1` (input)
  Number of entries in the **w-dimension**, e.g., for repetitions, scenarios, or configurations. Default: `1`.

### Example:
```
// Create a file with layout: [10 BSs, 4 UEs, 2 frequencies]
quadriga_lib::hdf5_create("channels.hdf5", 10, 4, 2);
```

---
## hdf5_read_channel
Read a channel object from an HDF5 file

### Description:
- Loads a `quadriga_lib::channel<dtype>` object from the specified index in a previously created HDF5 file.
- If the selected index does not contain a valid channel, an empty channel object is returned (with `no_snapshots == 0`).
- Allowed datatypes (`dtype`): `float` or `double`
- All structured data (e.g., channel coefficients, delays, positions) is stored in **single precision** in the
  file but will be **converted** to the appropriate precision (`float` or `double`) depending on the provided template parameter.
- Unstructured or user-defined fields stored in `std::any` containers are not converted and retain their original type

### Declaration:
```
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(
                std::string fn,
                unsigned ix = 0,
                unsigned iy = 0,
                unsigned iz = 0,
                unsigned iw = 0);
```

### Arguments:
- `std::string **fn**` (input)
  Filename of the HDF5 file containing the channel data.

- `unsigned **ix** = 0` (input)
  x-Index in the file's 4D storage layout. Default: `0`.

- `unsigned **iy** = 0` (input)
  y-Index in the file's 4D storage layout. Default: `0`.

- `unsigned **iz** = 0` (input)
  z-Index in the file's 4D storage layout. Default: `0`.

- `unsigned **iw** = 0` (input)
  w-Index in the file's 4D storage layout. Default: `0`.

### Returns:
- `quadriga_lib::channel<dtype>`
  A channel object containing the channel data at the specified index. If no data is found, an empty channel object is returned.

---
## hdf5_read_dset
Read an unstructured dataset from an HDF5 file

### Description:
- Reads a user-defined, unstructured dataset stored in an HDF5 file at the specified index.
- Unstructured datasets are typically used to store additional parameters or metadata and are stored under a **name prefix** (e.g. `"par_"`) followed by the dataset name.
- Returns the dataset as a `std::any` object, which can hold any supported type (e.g., scalar values, vectors, matrices, cubes).
- Use `quadriga_lib::any_type_id` to determine the contained type and obtain a raw pointer for direct access.
- If the dataset does not exist at the specified index or name, an empty `std::any` object is returned.

### Declaration:
```
std::any quadriga_lib::hdf5_read_dset(
                std::string fn,
                std::string par_name,
                unsigned ix = 0,
                unsigned iy = 0,
                unsigned iz = 0,
                unsigned iw = 0,
                std::string prefix = "par_");
```

### Arguments:
- `std::string **fn**` (input)
  Filename of the HDF5 file containing the dataset.

- `std::string **par_name**` (input)
  Name of the dataset, **without** the prefix (e.g., `"carrier_frequency"`).

- `unsigned **ix** = 0` (input)
  x-Index in the HDF5 file’s 4D storage layout. Default: `0`.

- `unsigned **iy** = 0` (input)
  y-Index in the HDF5 file’s 4D storage layout. Default: `0`.

- `unsigned **iz** = 0` (input)
  z-Index in the HDF5 file’s 4D storage layout. Default: `0`.

- `unsigned **iw** = 0` (input)
  w-Index in the HDF5 file’s 4D storage layout. Default: `0`.

- `std::string **prefix** = "par_"` (input)
  Optional dataset name prefix. Default is `"par_"`.

### Returns:
- A `std::any` object containing the dataset. If the dataset is not present, the return value is an empty `std::any`.

---
## hdf5_read_dset_names
Read names of unstructured datasets from an HDF5 file

### Description:
- Retrieves the names of all unstructured datasets stored at a specified 4D index in an HDF5 file.
- Dataset names are identified by a common prefix (default: `"par_"`) and the actual parameter name follows the prefix.
- The returned names in `par_names` exclude the prefix for convenience.
- Returns the number of datasets found at the given index.

### Declaration:
```
arma::uword quadriga_lib::hdf5_read_dset_names(
                std::string fn,
                std::vector<std::string> *par_names,
                unsigned ix = 0,
                unsigned iy = 0,
                unsigned iz = 0,
                unsigned iw = 0,
                std::string prefix = "par_");
```

### Arguments:
- `std::string **fn**` (input)
  Filename of the HDF5 file containing the datasets.

- `std::vector<std::string> ***par_names**` (output)
  Pointer to a vector that will be filled with all dataset names found at the specified index (excluding the prefix).

- `unsigned **ix** = 0` (input)
  x-Index in the HDF5 file’s 4D layout. Default: `0`.

- `unsigned **iy** = 0` (input)
  y-Index in the HDF5 file’s 4D layout. Default: `0`.

- `unsigned **iz** = 0` (input)
  z-Index in the HDF5 file’s 4D layout. Default: `0`.

- `unsigned **iw** = 0` (input)
  w-Index in the HDF5 file’s 4D layout. Default: `0`.

- `std::string **prefix** = "par_"` (input)
  Optional prefix string used to identify unstructured datasets. Default: `"par_"`.

### Returns:
- The number of unstructured datasets found at the specified location with the given prefix.

---
## hdf5_read_layout
Read the HDF5 channel storage layout

### Description:
- Reads the structure of the storage layout in an HDF5 file created for channel data.
- Returns the size of each of the four dimensions: `{nx, ny, nz, nw}`.
- Can optionally return a serialized list of channel IDs to indicate which entries contain data.
- If the file does not exist or is not a valid channel HDF5 file, the returned layout is `{0, 0, 0, 0}`.
- Entries in `channelID` are set to `0` if no data is present at that index.

### Declaration:
```
arma::u32_vec quadriga_lib::hdf5_read_layout(
                std::string fn,
                arma::u32_vec *channelID = nullptr);
```

### Arguments:
- `std::string **fn**` (input)
  Path to the HDF5 file.

- `arma::u32_vec ***channelID** = nullptr` (optional output)
  Pointer to a vector receiving the serialized channel index contents.
  Length: `nx × ny × nz × nw`, where `channelID[i] == 0` indicates an empty slot.

### Returns:
- `arma::u32_vec`
  Containing four elements: `{nx, ny, nz, nw}`, the layout of the storage grid.

---
## hdf5_reshape_layout
Reshape the storage layout of an HDF5 channel file

### Description:
- Updates the multi-dimensional storage layout of an existing HDF5 file that contains channel data.
- The layout consists of up to four dimensions: `{nx, ny, nz, nw}`.
- This is useful for reorganizing data after initial creation (e.g., grouping entries by BS/UE/frequency).
- The total number of entries must remain unchanged, i.e., `nx × ny × nz × nw` must equal the original layout.
- Throws an error if the reshaped layout violates this constraint.

### Declaration:
```
void quadriga_lib::hdf5_reshape_layout(
                std::string fn,
                unsigned nx,
                unsigned ny = 1,
                unsigned nz = 1,
                unsigned nw = 1);
```

### Arguments:
- `std::string **fn**` (input)
  Filename of the HDF5 file to reshape.

- `unsigned **nx**` (input)
  Number of entries in the first dimension (e.g., base stations).

- `unsigned **ny** = 1` (input)
  Number of entries in the second dimension (e.g., user equipment). Default: `1`.

- `unsigned **nz** = 1` (input)
  Number of entries in the third dimension (e.g., carrier frequency). Default: `1`.

- `unsigned **nw** = 1` (input)
  Number of entries in the fourth dimension. Default: `1`.

---
## hdf5_write
Write channel data to HDF5 file

### Description:
- Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data.
- This function rites a `channel` object to a specified HDF5 file at the given 4D index location.
- If the file does not exist, a new file is created with default layout `(65535 × 1 × 1 × 1)`.

### Declaration:
```
int quadriga_lib::hdf5_write(
                const quadriga_lib::channel<dtype> *ch,
                std::string fn,
                unsigned ix = 0,
                unsigned iy = 0,
                unsigned iz = 0,
                unsigned iw = 0,
                bool assume_valid = false);
```

### Arguments:
- `const channel<dtype> ***ch**` (input)
  Pointer to the channel object to be stored.

- `std::string **fn**` (input)
  Path to the HDF5 file.

- `unsigned **ix** = 0` (input)
  Index in the x-dimension (e.g., Base Station ID). Default: `0`.

- `unsigned **iy** = 0` (input)
  Index in the y-dimension (e.g., User Equipment ID). Default: `0`.

- `unsigned **iz** = 0` (input)
  Index in the z-dimension (e.g., Frequency point). Default: `0`.

- `unsigned **iw** = 0` (input)
  Index in the w-dimension (e.g., Repetition/Scenario). Default: `0`.

- `bool **assume_valid** = false` (input)
  If `true`, skips channel integrity validation before writing. Default: `false`.

### Returns:
- `0` if a new dataset was created.
- `1` if an existing dataset was overwritten or extended.

### Caveat:
- If the file exists already, the new data is added to the exisiting file
- If the index already contains data, it will be overwritten
- Use `assume_valid = true` to skip internal validation (faster, but unsafe if data may be corrupted).
- Throws an error if the index was not reserved during `hdf5_create`.
- All structured data is written in single precision (but can can be provided as float or double)
- Unstructured datatypes are maintained in the HDF file
- Supported unstructured types: string, double, float, (u)int32, (u)int64
- Supported unstructured size: up to 3 dimensions
- Storage order of the unstructured data is maintained

---
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file

### Description:
- Writes a single unstructured data field to an HDF5 file at the specified 4D index.
- Supported scalar types: `std::string`, `unsigned`, `int`, `long long`, `unsigned long long`, `float`, `double`
- Supported Armadillo types: `arma::Col`, `arma::Row`, `arma::Mat`, and `arma::Cube` with `float`, `double`, `int`, `unsigned`, `sword`, `uword`, `unsigned long long`
- `arma::Row` vectors are converted to column vectors (`arma::Col`) before writing.
- The dataset name is prefixed with `"par_"` (default) unless another prefix is specified.
- Throws an error for unsupported types.
- Dataset names may only include alphanumeric characters and underscores.

### Declaration:
```
void quadriga_lib::hdf5_write_dset(
                std::string fn,
                std::string par_name,
                const std::any \*par_data,
                unsigned ix = 0,
                unsigned iy = 0,
                unsigned iz = 0,
                unsigned iw = 0,
                std::string prefix = "par_");
```

### Arguments:
- `std::string **fn**` (input)
  Filename of the HDF5 file to which the dataset will be written.

- `std::string **par_name**` (input)
  Name of the parameter to store (without prefix). Must contain only letters, digits, and underscores.

- `const std::any ***par_data**` (input)
  Pointer to the data to be written. Type must be supported (see above).

- `unsigned **ix** = 0` (input)
  x-Index in the HDF5 file’s 4D layout. Default: `0`.

- `unsigned **iy** = 0` (input)
  y-Index in the HDF5 file’s 4D layout. Default: `0`.

- `unsigned **iz** = 0` (input)
  z-Index in the HDF5 file’s 4D layout. Default: `0`.

- `unsigned **iw** = 0` (input)
  w-Index in the HDF5 file’s 4D layout. Default: `0`.

- `std::string **prefix** = "par_"` (input)
  Optional prefix for the dataset name. Default: `"par_"`.

---
## qrt_file_parse
Read metadata from a QRT file

### Description:
- Parses a QRT file and extracts metadata such as the number of snapshots, origins, destinations, and frequencies.
- All output arguments are optional; pass `nullptr` to skip any value you don't need.
- Can also retrieve CIR offsets per destination, human-readable names for origins and destinations, and the file version.

### Declaration:
```
void quadriga_lib::qrt_file_parse(
                const std::string &fn,
                arma::uword *no_cir = nullptr,
                arma::uword *no_orig = nullptr,
                arma::uword *no_dest = nullptr,
                arma::uword *no_freq = nullptr,
                arma::uvec *cir_offset = nullptr,
                std::vector<std::string> *orig_names = nullptr,
                std::vector<std::string> *dest_names = nullptr,
                int *version = nullptr)
```

### Arguments:
- `const std::string &**fn**` (input)
  Path to the QRT file.
- `arma::uword ***no_cir** = nullptr` (optional output)
  Number of channel snapshots per origin point.
- `arma::uword ***no_orig** = nullptr` (optional output)
  Number of origin points (TX).
- `arma::uword ***no_dest** = nullptr` (optional output)
  Number of destinations (RX).
- `arma::uword ***no_freq** = nullptr` (optional output)
  Number of frequency bands.
- `arma::uvec ***cir_offset** = nullptr` (optional output)
  CIR offset for each destination. Size `[no_dest]`.
- `std::vector<std::string> ***orig_names** = nullptr` (optional output)
  Names of the origin points (TXs). Size `[no_orig]`.
- `std::vector<std::string> ***dest_names** = nullptr` (optional output)
  Names of the destination points (RXs). Size `[no_dest]`.
- `int ***version** = nullptr` (optional output)
  QRT file version number.

### Example:
```
arma::uword no_cir, no_orig, no_dest, no_freq;
arma::uvec cir_offset;
std::vector<std::string> orig_names, dest_names;
int version;

quadriga_lib::qrt_file_parse("scene.qrt", &no_cir, &no_orig, &no_dest, &no_freq,
                              &cir_offset, &orig_names, &dest_names, &version);
```

---
## qrt_file_read
Read ray-tracing data from a QRT file

### Description:
- Reads channel impulse response (CIR) data from a QRT file for a specific snapshot and origin point.
- Supports both uplink and downlink directions by swapping TX/RX roles accordingly.
- All output arguments are optional; pass `nullptr` to skip any value you don't need.
- The `normalize_M` parameter controls how the polarization transfer matrix `M` and path gains are returned.
- For maximum performance in tight loops, pass a pre-opened `std::ifstream` and a pre-populated
  `qrt_read_cache`. With both, the per-call I/O is reduced to 2 seeks and 4 reads.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
template <typename dtype>
void quadriga_lib::qrt_file_read(
                const std::string &fn,
                arma::uword i_cir = 0,
                arma::uword i_orig = 0,
                bool downlink = true,
                arma::Col<dtype> *center_frequency = nullptr,
                arma::Col<dtype> *tx_pos = nullptr,
                arma::Col<dtype> *tx_orientation = nullptr,
                arma::Col<dtype> *rx_pos = nullptr,
                arma::Col<dtype> *rx_orientation = nullptr,
                arma::Mat<dtype> *fbs_pos = nullptr,
                arma::Mat<dtype> *lbs_pos = nullptr,
                arma::Mat<dtype> *path_gain = nullptr,
                arma::Col<dtype> *path_length = nullptr,
                arma::Cube<dtype> *M = nullptr,
                arma::Col<dtype> *aod = nullptr,
                arma::Col<dtype> *eod = nullptr,
                arma::Col<dtype> *aoa = nullptr,
                arma::Col<dtype> *eoa = nullptr,
                std::vector<arma::Mat<dtype>> *path_coord = nullptr,
                int normalize_M = 1,
                arma::u32_vec *no_int = nullptr,
                arma::fmat *coord = nullptr,
                std::ifstream *file = nullptr,
                const qrt_read_cache *cache = nullptr)
```

### Arguments:
- `const std::string &**fn**` (input)
  Path to the QRT file. Ignored when both `file` and `cache` are provided.

- `arma::uword **i_cir** = 0` (input)
  Snapshot index (0-based).

- `arma::uword **i_orig** = 0` (input)
  Origin index, 0-based. For downlink, origin corresponds to the transmitter.

- `bool **downlink** = true` (input)
  If `true`, origin is TX and destination is RX (downlink). If `false`, roles are swapped (uplink).

- `arma::Col<dtype> ***center_frequency** = nullptr` (optional output)
  Center frequency in Hz. Size `[n_freq]`.

- `arma::Col<dtype> ***tx_pos** = nullptr` (optional output)
  Transmitter position in Cartesian coordinates. Size `[3]`.

- `arma::Col<dtype> ***tx_orientation** = nullptr` (optional output)
  Transmitter orientation (bank, tilt, heading) in radians. Size `[3]`.

- `arma::Col<dtype> ***rx_pos** = nullptr` (optional output)
  Receiver position in Cartesian coordinates. Size `[3]`.

- `arma::Col<dtype> ***rx_orientation** = nullptr` (optional output)
  Receiver orientation (bank, tilt, heading) in radians. Size `[3]`.

- `arma::Mat<dtype> ***fbs_pos** = nullptr` (optional output)
  First-bounce scatterer positions. Size `[3, n_path]`.

- `arma::Mat<dtype> ***lbs_pos** = nullptr` (optional output)
  Last-bounce scatterer positions. Size `[3, n_path]`.

- `arma::Mat<dtype> ***path_gain** = nullptr` (optional output)
  Path gain on linear scale. Size `[n_path, n_freq]`.

- `arma::Col<dtype> ***path_length** = nullptr` (optional output)
  Absolute path length from TX to RX phase center. Size `[n_path]`.

- `arma::Cube<dtype> ***M** = nullptr` (optional output)
  Polarization transfer matrix. Size `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files.

- `arma::Col<dtype> ***aod** = nullptr` (optional output)
  Departure azimuth angles in radians. Size `[n_path]`.

- `arma::Col<dtype> ***eod** = nullptr` (optional output)
  Departure elevation angles in radians. Size `[n_path]`.

- `arma::Col<dtype> ***aoa** = nullptr` (optional output)
  Arrival azimuth angles in radians. Size `[n_path]`.

- `arma::Col<dtype> ***eoa** = nullptr` (optional output)
  Arrival elevation angles in radians. Size `[n_path]`.

- `std::vector<arma::Mat<dtype>> ***path_coord** = nullptr` (optional output)
  Interaction coordinates per path. Vector of length `n_path`, each matrix of size `[3, n_interact + 2]`.

- `int **normalize_M** = 1` (input)
  Normalization option for the polarization transfer matrix.
   0 | `M` as stored in QRT file, `path_gain` is -FSPL
   1 | `M` has sum-column power of 2, `path_gain` is -FSPL minus material losses

- `arma::u32_vec ***no_int** = nullptr` (optional output)
  Number of mesh interactions per path. Size `[n_path]`. A value of 0 indicates a LOS path.

- `arma::fmat ***coord** = nullptr` (optional output)
  Interaction coordinates. Size `[3, sum(no_int)]`.

- `std::ifstream ***file** = nullptr` (optional input)
  Optional pre-opened binary ifstream. If `nullptr`, the file is opened from `fn` and closed
  on return. If provided, the stream is left open after return.

- `const qrt_read_cache ***cache** = nullptr` (optional input)
  Optional pre-parsed metadata cache (from `qrt_read_cache_init`). When provided together with
  `file`, per-call I/O is reduced to 2 seeks and 4 reads.

### Example:
```
arma::vec center_freq, tx_pos, rx_pos, path_length, aod, eod, aoa, eoa;
arma::mat fbs_pos, lbs_pos, path_gain;
arma::cube M;

// Single call (opens and closes file internally):
quadriga_lib::qrt_file_read<double>("scene.qrt", 0, 0, true,
    &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
    &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
    &aod, &eod, &aoa, &eoa, nullptr, 1);

// Maximum performance with cache and shared stream:
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
quadriga_lib::qrt_read_cache cache;
quadriga_lib::qrt_read_cache_init("scene.qrt", &cache, &stream);

for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
            &aod, &eod, &aoa, &eoa, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```

---
## qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

### Description:
- Reads all fixed metadata from a QRT file into a `qrt_read_cache` struct.
- Pre-computes byte offsets so that subsequent `qrt_file_read` calls only need to
  perform 2 seeks and 4 reads (the per-CIR path data) instead of re-parsing the header.
- Populate the cache once, then pass it to `qrt_file_read` together with a shared
  `std::ifstream` for maximum performance in tight loops.

### Declaration:
```
quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(
                const std::string &fn,
                std::ifstream *file = nullptr)
```

### Arguments:
- `const std::string &**fn**` (input)
  Path to the QRT file.

- `std::ifstream ***file** = nullptr` (optional input)
  Optional pre-opened binary ifstream. If `nullptr`, the file is opened from `fn`
  and closed on return. If provided, the stream is left open.

### Returns:
- `qrt_read_cache **cache**` (output)
  Populated cache structure.

### Example:
```
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);

arma::vec center_freq, tx_pos, rx_pos;
arma::mat path_gain;
arma::cube M;

for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            nullptr, nullptr, &path_gain, nullptr, &M,
            nullptr, nullptr, nullptr, nullptr, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```

---
## quantize_delays
Fixes the path delays to a grid of delay bins

### Description:
- For channel emulation with finite delay resolution, path delays must be mapped to a fixed grid
  of delay bins (taps). Rounding delays to the nearest tap causes discontinuities in the frequency
  domain when a delay crosses a tap boundary (e.g. as a mobile terminal moves). This function
  instead approximates each path delay using two adjacent taps with power-weighted coefficients,
  producing smooth transitions.
- For a path at fractional offset &delta; between tap indices, two taps are created with complex
  coefficients scaled by (1&minus;&delta;)^&alpha; and &delta;^&alpha;, where &alpha; is the power
  exponent. The default &alpha;=1.0 (linear interpolation) is optimal for narrowband systems. Use
  &alpha;=0.5 to preserve wideband (incoherent) power.
- If input delays are already quantized (all fractional offsets below 0.01), the interpolation
  weight computation is skipped but the same delay-selection logic is used.
- The `fix_taps` parameter controls whether delay grids are shared across antenna pairs and/or
  snapshots, trading accuracy for a more compact representation.
- Input delays may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`. When
  shared and fix_taps is 0 or 3, delays are expanded internally and output delays are per-antenna.
  When shared and fix_taps is 1 or 2, output delays remain shared `[1, 1, n_taps]`.
- The number of antennas `n_rx` and `n_tx` must be the same across all snapshots, but the number
  of paths `n_path_s` may differ per snapshot.

### Declaration:
```
template <typename dtype>
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    std::vector<arma::Cube<dtype>> *coeff_re_quant,
    std::vector<arma::Cube<dtype>> *coeff_im_quant,
    std::vector<arma::Cube<dtype>> *delay_quant,
    dtype tap_spacing = (dtype)5.0e-9,
    arma::uword max_no_taps = 48,
    dtype power_exponent = (dtype)1.0,
    int fix_taps = 0);
```

### Arguments:
- `const std::vector<arma::Cube<dtype>> ***coeff_re**` (input)
  Channel coefficients, real part. Vector of length `n_snap`, each cube of size
  `[n_rx, n_tx, n_path_s]` where `n_path_s` may differ across snapshots.

- `const std::vector<arma::Cube<dtype>> ***coeff_im**` (input)
  Channel coefficients, imaginary part. Same sizes as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> ***delay**` (input)
  Path delays in seconds. Vector of length `n_snap`, each cube of size
  `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`. The number of paths must match `coeff_re`.

- `std::vector<arma::Cube<dtype>> ***coeff_re_quant**` (output)
  Output coefficients, real part. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]`.

- `std::vector<arma::Cube<dtype>> ***coeff_im_quant**` (output)
  Output coefficients, imaginary part. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]`.

- `std::vector<arma::Cube<dtype>> ***delay_quant**` (output)
  Output delays in seconds. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]` or
  `[1, 1, n_taps]`.

- `dtype **tap_spacing** = 5.0e-9` (input)
  Spacing of the delay bins in seconds. Default: 5 ns (200 MHz sampling rate).

- `arma::uword **max_no_taps** = 48` (input)
  Maximum number of output taps. 0 means unlimited.

- `dtype **power_exponent** = 1.0` (input)
  Interpolation exponent &alpha;. Use 1.0 for narrowband (linear) or 0.5 for wideband (power-preserving).

- `int **fix_taps** = 0` (input)
  Delay sharing mode: 0 = per tx-rx pair and snapshot, 1 = single grid for all,
  2 = per snapshot, 3 = per tx-rx pair.

### Example:
```
// Create synthetic test data: 2 snapshots with different numbers of paths
std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
cre[0].set_size(1, 1, 3); cim[0].set_size(1, 1, 3); dl[0].set_size(1, 1, 3);
cre[1].set_size(1, 1, 2); cim[1].set_size(1, 1, 2); dl[1].set_size(1, 1, 2);
cre[0](0,0,0) = 1.0; cre[0](0,0,1) = 0.5; cre[0](0,0,2) = 0.3;
cre[1](0,0,0) = 0.8; cre[1](0,0,1) = 0.4;
cim[0].zeros(); cim[1].zeros();
dl[0](0,0,0) = 0.0; dl[0](0,0,1) = 12.5e-9; dl[0](0,0,2) = 33.4e-9;
dl[1](0,0,0) = 0.0; dl[1](0,0,1) = 10.0e-9;

std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);
```

---

# Channel generation functions

| Function | Description |
| --- | --- |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models |
| [get_channels_irs](#get_channels_irs) | Calculate channel coefficients for intelligent reflective surfaces (IRS) |
| [get_channels_multifreq](#get_channels_multifreq) | Calculate channel coefficients for spherical waves across multiple frequencies |
| [get_channels_planar](#get_channels_planar) | Calculate channel coefficients for planar waves |
| [get_channels_spherical](#get_channels_spherical) | Calculate channel coefficients for spherical waves |

---
## get_channels_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

### Description:
- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions.
- 2D model: no elevation angles are used; azimuth angles and planar motion are considered.
- Supports channel model types `A, B, C, D, E, F` (as defined by TGn) via `ChannelType`.
- Can generate MU-MIMO channels (`n_users > 1`) with per-user distances/floors and optional angle offsets according to TGac
- Optional time evolution via `observation_time`, `update_rate`, and mobility parameters.

### Declaration:
```
std::vector<quadriga_lib::channel<double>> quadriga_lib::get_channels_ieee_indoor(
                const arrayant<double> &ap_array,
                const arrayant<double> &sta_array,
                std::string ChannelType,
                double CarrierFreq_Hz = 5.25e9,
                double tap_spacing_s = 10.0e-9,
                arma::uword n_users = 1,
                double observation_time = 0.0,
                double update_rate = 1.0e-3,
                double speed_station_kmh = 0.0,
                double speed_env_kmh = 1.2,
                arma::vec Dist_m = {4.99},
                arma::uvec n_floors = {0},
                bool uplink = false,
                arma::mat offset_angles = {},
                arma::uword n_subpath = 20,
                double Doppler_effect = 50.0,
                arma::sword seed = -1,
                double KF_linear = NAN,
                double XPR_NLOS_linear = NAN,
                double SF_std_dB_LOS = NAN,
                double SF_std_dB_NLOS = NAN,
                double dBP_m = NAN );
```

### Arguments:
- `const arrayant<double> **ap_array**` (input)
  Access point array antenna with `n_tx` elements (= ports after element coupling).

- `const arrayant<double> **sta_array**` (input)
  Mobile station array antenna with `n_rx` elements (= ports after element coupling).

- `std::string **ChannelType**` (input)
  Channel model type as defined by TGn. Supported: `A, B, C, D, E, F`.

- `double **CarrierFreq_Hz** = 5.25e9` (optional input)
  Carrier frequency in Hz.

- `double **tap_spacing_s** = 10.0e-9` (optional input)
  Tap spacing in seconds. Must be equal to `10 ns / 2^k` (TGn default = `10e-9`).

- `arma::uword **n_users** = 1` (optional input)
  Number of users (only for TGac, TGah). Output vector length equals `n_users`.

- `double **observation_time** = 0.0` (optional input)
  Channel observation time in seconds. `0.0` creates a static channel.

- `double **update_rate** = 1.0e-3` (optional input)
  Channel update interval in seconds (only relevant when `observation_time > 0`).

- `double **speed_station_kmh** = 0.0` (optional input)
  Station movement speed in km/h. Movement direction is `AoA_offset`. Only relevant when `observation_time > 0`.

- `double **speed_env_kmh** = 1.2` (optional input)
  Environment movement speed in km/h. Default `1.2` for TGn, use `0.089` for TGac. Only relevant when
  `observation_time > 0`.

- `arma::vec **Dist_m** = {4.99}` (optional input)
  TX-to-RX distance(s) in meters. Length `n_users` or length `1` (same distance for all users). Size
  `[n_users]` or `[1]`.

- `arma::uvec **n_floors** = {0}` (optional input)
  Number of floors for TGah model (per user), up to 4 floors. Length `n_users` or length `1`. Size
  `[n_users]` or `[1]`.

- `bool **uplink** = false` (optional input)
  Channel direction flag. Default is downlink; set to `true` to generate reverse (uplink) direction.

- `arma::mat **offset_angles** = {}` (optional input)
  Offset angles in degree for MU-MIMO channels. Empty uses model defaults (TGac auto for `n_users > 1`).
  Size `[4, n_users]` with rows: `AoD LOS, AoD NLOS, AoA LOS, AoA NLOS`.

- `arma::uword **n_subpath** = 20` (optional input)
  Number of sub-paths per path/cluster used for Laplacian angular spread mapping.

- `double **Doppler_effect** = 50.0` (optional input)
  Special Doppler effects: models `D, E` (fluorescent lights, value = mains freq.) and `F` (moving
  vehicle speed in km/h). Use `0.0` to disable.

- `arma::sword **seed** = -1` (optional input)
  Numeric seed for repeatability. `-1` disables the fixed seed and uses the system random device.

- `double **KF_linear** = NAN` (optional input)
  Overwrites the model-specific KF-value. If this parameter is NAN (default) or negative, model defaults are used:
  A/B/C (KF = 1 for d < dBP, 0 otherwise); D (KF = 2 for d < dBP, 0 otherwise); E/F (KF = 4 for d < dBP, 0 otherwise). 
  KF is applied to the first tap only. Breakpoint distance is ignored for `KF_linear >= 0`.

- `double **XPR_NLOS_linear** = NAN` (optional input)
  Overwrites the model-specific Cross-polarization ratio. If this parameter is NAN (default) or negative, 
  the model default of 2 (3 dB) is used. XPR is applied to all NLOS taps.

- `double **SF_std_dB_LOS** = NAN` (optional input)
  Overwrites the model-specific shadow fading for LOS channels. If this parameter is NAN (default), 
  the model default of 3 dB is used. `SF_std_dB_LOS` is applied to all LOS channels, where the 
  AP-STA distance d < dBP.

- `double **SF_std_dB_NLOS** = NAN` (optional input)
  Overwrites the model-specific shadow fading for LOS channels. If this parameter is NAN (default), 
  the model defaults are A/B: 4 dB, C/D: 5 dB, E/F: 6 dB. `SF_std_dB_NLOS` is applied to all NLOS channels, 
  where the AP-STA distance d >= dBP.

- `double **dBP_m** = NAN` (optional input)
  Overwrites the model-specific breakpoint distance. If this parameter is NAN (default) or negative, 
  the model defaults are A/B/C: 5 m, D: 10 m, E: 20 m, F: 30 m.

### Returns:
- `std::vector<quadriga_lib::channel<double>>` (output)
  Vector of channel objects with length `n_users`. Each entry contains the generated indoor channel
  realization for one user (including direction determined by `uplink`).

---
## get_channels_irs
Calculate channel coefficients for intelligent reflective surfaces (IRS)

### Description:
- Calculates MIMO channel coefficients and delays for IRS-assisted communication using two channel segments:
  1. TX → IRS; 2. IRS → RX
- The IRS is modeled as a passive antenna array with phase shifts defined via its coupling matrix.
- IRS codebook entries can be selected via a port index (`i_irs`).
- Supports combining paths from both segments to form `n_path_irs` valid output paths, subject to a gain threshold.
- Optional second IRS array allows different antenna behavior for TX-IRS and IRS-RX directions.
- Returns a boolean vector indicating which path combinations are included in the output.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
std::vector<bool> quadriga_lib::get_channels_irs(
                const arrayant<dtype> *tx_array,
                const arrayant<dtype> *rx_array,
                const arrayant<dtype> *irs_array,
                dtype Tx, dtype Ty, dtype Tz,
                dtype Tb, dtype Tt, dtype Th,
                dtype Rx, dtype Ry, dtype Rz,
                dtype Rb, dtype Rt, dtype Rh,
                dtype Ix, dtype Iy, dtype Iz,
                dtype Ib, dtype It, dtype Ih,
                const arma::Mat<dtype> *fbs_pos_1,
                const arma::Mat<dtype> *lbs_pos_1,
                const arma::Col<dtype> *path_gain_1,
                const arma::Col<dtype> *path_length_1,
                const arma::Mat<dtype> *M_1,
                const arma::Mat<dtype> *fbs_pos_2,
                const arma::Mat<dtype> *lbs_pos_2,
                const arma::Col<dtype> *path_gain_2,
                const arma::Col<dtype> *path_length_2,
                const arma::Mat<dtype> *M_2,
                arma::Cube<dtype> *coeff_re,
                arma::Cube<dtype> *coeff_im,
                arma::Cube<dtype> *delay,
                arma::uword i_irs = 0,
                dtype threshold_dB = -140.0,
                dtype center_frequency = 0.0,
                bool use_absolute_delays = false,
                arma::Cube<dtype> *aod = nullptr,
                arma::Cube<dtype> *eod = nullptr,
                arma::Cube<dtype> *aoa = nullptr,
                arma::Cube<dtype> *eoa = nullptr,
                const arrayant<dtype> *irs_array_2 = nullptr,
                const std::vector<bool> *active_path = nullptr);
```

### Arguments:
- `const arrayant<dtype> ***tx_array**` (input)
  Pointer to the transmit antenna array object (with `n_tx` elements).

- `const arrayant<dtype> ***rx_array**` (input)
  Pointer to the receive antenna array object (with `n_rx` elements).

- `const arrayant<dtype> ***irs_array**` (input)
  Pointer to the IRS array antenna (with `n_irs` elements).

- `dtype **Tx**, **Ty**, **Tz**` (input)
  Transmitter position in Cartesian coordinates [m].

- `dtype **Tb**, **Tt**, **Th**` (input)
  Transmitter orientation (Euler angles) in radians.

- `dtype **Rx**, **Ry**, **Rz**` (input)
  Receiver position in Cartesian coordinates [m].

- `dtype **Rb**, **Rt**, **Rh**` (input)
  Receiver orientation (Euler angles) in radians.

- `dtype **Ix**, **Iy**, **Iz**` (input)
  IRS position in Cartesian coordinates [m].

- `dtype **Ib**, **It**, **Ih**` (input)
  IRS orientation (Euler angles) in radians.

- `const arma::Mat<dtype> ***fbs_pos_1**` (input)
  First-bounce scatterer positions of TX → IRS paths, Size `[3, n_path_1]`.

- `const arma::Mat<dtype> ***lbs_pos_1**` (input)
  Last-bounce scatterer positions of TX → IRS paths, Size `[3, n_path_1]`.

- `const arma::Col<dtype> ***path_gain_1**` (input)
  Path gains (linear) for TX → IRS paths, Length `n_path_1`.

- `const arma::Col<dtype> ***path_length_1**` (input)
  Path lengths for TX → IRS paths, Length `n_path_1`.

- `const arma::Mat<dtype> ***M_1**` (input)
  Polarization transfer matrix for TX → IRS paths, Size `[8, n_path_1]`.

- `const arma::Mat<dtype> ***fbs_pos_2**` (input)
  First-bounce scatterer positions of IRS → RX paths, Size `[3, n_path_2]`.

- `const arma::Mat<dtype> ***lbs_pos_2**` (input)
  Last-bounce scatterer positions of IRS → RX paths, Size `[3, n_path_2]`.

- `const arma::Col<dtype> ***path_gain_2**` (input)
  Path gains (linear) for IRS → RX paths, Length `n_path_2`.

- `const arma::Col<dtype> ***path_length_2**` (input)
  Path lengths for IRS → RX paths, Length `n_path_2`.

- `const arma::Mat<dtype> ***M_2**` (input)
  Polarization transfer matrix for IRS → RX paths, Size `[8, n_path_2]`.

- `arma::Cube<dtype> ***coeff_re**` (output)
  Real part of resulting IRS-assisted channel coefficients, Size `[n_rx, n_tx, n_path_irs]`.

- `arma::Cube<dtype> ***coeff_im**` (output)
  Imaginary part of channel coefficients, Size `[n_rx, n_tx, n_path_irs]`.

- `arma::Cube<dtype> ***delay**` (output)
  Propagation delays in seconds, Size `[n_rx, n_tx, n_path_irs]`.

- `arma::uword **i_irs** = 0` (optional input)
  Index of IRS codebook entry (port number), Default: `0`.

- `dtype **threshold_dB** = -140.0` (optional input)
  Threshold (in dB) below which paths are discarded. Default: `-140.0`.

- `dtype **center_frequency** = 0.0` (optional input)
  Center frequency in Hz; `0.0` disables phase computation. Default: `0.0`.

- `bool **use_absolute_delays** = false` (optional input)
  If true, includes LOS delay in all paths. Default: `false`.

- `arma::Cube<dtype> ***aod** = nullptr` (optional output)
  Azimuth of Departure angles [rad], Size `[n_rx, n_tx, n_path_irs]`.

- `arma::Cube<dtype> ***eod** = nullptr` (optional output)
  Elevation of Departure angles [rad], Size `[n_rx, n_tx, n_path_irs]`.

- `arma::Cube<dtype> ***aoa** = nullptr` (optional output)
  Azimuth of Arrival angles [rad], Size `[n_rx, n_tx, n_path_irs]`.

- `arma::Cube<dtype> ***eoa** = nullptr` (optional output)
  Elevation of Arrival angles [rad], Size `[n_rx, n_tx, n_path_irs]`.

- `const arrayant<dtype> ***irs_array_2** = nullptr` (optional input)
  Optional second IRS array (TX side) for asymmetric IRS behavior.

- `const std::vector<bool> ***active_path** = nullptr` (optional input)
  Optional bitmask for selecting active TX-IRS and IRS-RX path pairs. Ignores `threshold_dB` when provided.

### Returns:
- `std::vector<bool>` 
  Boolean mask of length `n_path_1 * n_path_2`, indicating which path combinations were used.

### See also:
- [combine_irs_coord](#combine_irs_coord)
- [get_channels_spherical](#get_channels_spherical)
- [get_channels_planar](#get_channels_planar)

---
## get_channels_multifreq
Calculate channel coefficients for spherical waves across multiple frequencies

### Description:
- Extends `get_channels_spherical` to support frequency-dependent antenna patterns, path gains,
  and polarization transfer (Jones) matrices across multiple output frequencies.
- **Geometry is computed once**: departure angles, arrival angles, element-resolved path delays, and LOS
  path detection are frequency-independent and reused for all output frequencies. This avoids redundant
  trigonometry and distance calculations.
- **Four frequency grids** are aligned by interpolation:
  1. | TX array frequencies (defined by `tx_array[i].center_frequency`)
  2. | RX array frequencies (defined by `rx_array[i].center_frequency`)
  3. | Input sample frequencies (`freq_in`) at which `path_gain` and `M` are provided
  4. | Target output frequencies (`freq_out`) at which coefficients and delays are returned
- For each output frequency, TX and RX antenna patterns are interpolated from their respective
  multi-frequency vectors using spherical interpolation (SLERP) with linear fallback, the same
  algorithm used in `arrayant_interpolate_multi`. The private `qd_arrayant_interpolate` function is
  called directly for maximum performance.
- Path gain is interpolated linearly across frequency. The Jones matrix `M` is interpolated using
  SLERP for each complex entry pair to preserve phase coherence.
- **Extrapolation** is handled by clamping to the nearest available frequency entry in all four grids.
- **Propagation speed** can be set to support both radio (speed of light, default) and acoustic
  (speed of sound, ~343 m/s) simulations. This affects wavelength, wave number, and delay calculations.
- The Jones matrix `M` supports two formats: 8 rows for full polarimetric
  (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH), or 2 rows for scalar pressure waves
  (ReVV, ImVV only), where VH, HV, and HH entries are implicitly zero.
- Antenna element coupling is applied using the coupling matrices from the first entry of each
  multi-frequency vector (consistent across all entries by `arrayant_is_valid_multi` constraints).
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::get_channels_multifreq(
        const std::vector<arrayant<dtype>> &tx_array,
        const std::vector<arrayant<dtype>> &rx_array,
        dtype Tx, dtype Ty, dtype Tz,
        dtype Tb, dtype Tt, dtype Th,
        dtype Rx, dtype Ry, dtype Rz,
        dtype Rb, dtype Rt, dtype Rh,
        const arma::Mat<dtype> &fbs_pos,
        const arma::Mat<dtype> &lbs_pos,
        const arma::Mat<dtype> &path_gain,
        const arma::Col<dtype> &path_length,
        const arma::Cube<dtype> &M,
        const arma::Col<dtype> &freq_in,
        const arma::Col<dtype> &freq_out,
        std::vector<arma::Cube<dtype>> &coeff_re,
        std::vector<arma::Cube<dtype>> &coeff_im,
        std::vector<arma::Cube<dtype>> &delay,
        bool use_absolute_delays = false,
        bool add_fake_los_path = false,
        dtype propagation_speed = dtype(299792458.0))
```

### Arguments:
- `const std::vector<arrayant<dtype>> &**tx_array**` (input)
  Multi-frequency transmit array antenna vector. All entries must pass `arrayant_is_valid_multi`.

- `const std::vector<arrayant<dtype>> &**rx_array**` (input)
  Multi-frequency receive array antenna vector. All entries must pass `arrayant_is_valid_multi`.

- `dtype **Tx**, **Ty**, **Tz**` (input)
  Transmitter position in Cartesian coordinates [m].

- `dtype **Tb**, **Tt**, **Th**` (input)
  Transmitter orientation (bank, tilt, heading) in [rad].

- `dtype **Rx**, **Ry**, **Rz**` (input)
  Receiver position in Cartesian coordinates [m].

- `dtype **Rb**, **Rt**, **Rh**` (input)
  Receiver orientation (bank, tilt, heading) in [rad].

- `const arma::Mat<dtype> &**fbs_pos**` (input)
  First-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Mat<dtype> &**lbs_pos**` (input)
  Last-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Mat<dtype> &**path_gain**` (input)
  Path gain in linear scale, Size: `[n_path, n_freq_in]`. Each column corresponds to one input frequency.

- `const arma::Col<dtype> &**path_length**` (input)
  Absolute path lengths from TX to RX phase center, Length: `n_path`.

- `const arma::Cube<dtype> &**M**` (input)
  Polarization transfer matrix, Size: `[8, n_path, n_freq_in]` for full polarimetric or
  `[2, n_path, n_freq_in]` for scalar pressure. Each slice corresponds to one input frequency.
  Interleaved complex format: (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) for 8 rows,
  or (ReVV, ImVV) for 2 rows.

- `const arma::Col<dtype> &**freq_in**` (input)
  Input sample frequencies in [Hz] at which `path_gain` and `M` are defined, Length: `n_freq_in`.

- `const arma::Col<dtype> &**freq_out**` (input)
  Target frequencies in [Hz] at which to compute output coefficients and delays, Length: `n_freq_out`.

- `std::vector<arma::Cube<dtype>> &**coeff_re**` (output)
  Real part of channel coefficients. Vector of length `n_freq_out`, each cube of size `[n_rx, n_tx, n_path]`.

- `std::vector<arma::Cube<dtype>> &**coeff_im**` (output)
  Imaginary part of channel coefficients. Same structure as `coeff_re`.

- `std::vector<arma::Cube<dtype>> &**delay**` (output)
  Propagation delays in seconds. Same structure as `coeff_re`.

- `bool **use_absolute_delays** = false` (optional input)
  If true, LOS delay is included in all paths. Default: `false`.

- `bool **add_fake_los_path** = false` (optional input)
  Adds a zero-power LOS path if no LOS path was detected. Default: `false`.

- `dtype **propagation_speed** = 299792458.0` (optional input)
  Wave propagation speed in [m/s]. Default is the speed of light for radio simulations.
  Set to ~343.0 for acoustic simulations in air.

### Example:
```
// Build a 2-way speaker as TX (source)
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto tx_woofer = quadriga_lib::generate_speaker<double>(
    "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
auto tx_tweeter = quadriga_lib::generate_speaker<double>(
    "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
    0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
auto tx = quadriga_lib::arrayant_concat_multi(tx_woofer, tx_tweeter);

// Omnidirectional microphone as RX (single entry, clamped for all frequencies)
std::vector<quadriga_lib::arrayant<double>> rx = { quadriga_lib::generate_arrayant_omni<double>() };

// Simple LOS path setup
arma::mat fbs = arma::mat({0.5, 0.0, 0.0}).t();
arma::mat lbs = arma::mat({0.5, 0.0, 0.0}).t();
arma::vec path_length = {1.0};               // 1 meter distance

// Frequency-flat path gain and scalar Jones matrix
arma::vec freq_in = {100.0, 10000.0};
arma::mat path_gain_mat(1, 2, arma::fill::ones);
arma::cube M_cube(2, 1, 2, arma::fill::zeros);
M_cube(0, 0, 0) = 1.0; M_cube(0, 0, 1) = 1.0;  // ReVV = 1 at both freqs

arma::vec freq_out = {200.0, 1000.0, 5000.0};
std::vector<arma::cube> coeff_re, coeff_im, delays;

quadriga_lib::get_channels_multifreq(tx, rx,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   // TX at origin
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,   // RX at (1,0,0)
    fbs_pos, lbs_pos, path_gain_mat, path_length, M_cube,
    freq_in, freq_out, coeff_re, coeff_im, delays,
    false, false, 343.0);             // Speed of sound for acoustics
```

### See also:
- [get_channels_spherical](#get_channels_spherical)
- [arrayant_interpolate_multi](#arrayant_interpolate_multi)
- [arrayant_concat_multi](#arrayant_concat_multi)
- [generate_speaker](#generate_speaker)

---
## get_channels_planar
Calculate channel coefficients for planar waves

### Description:
- Calculates MIMO channel coefficients and delays for a set of planar wave paths between two antenna arrays.
- Interpolates antenna patterns (including orientation and polarization) for both transmitter and receiver arrays.
- Supports LOS path identification based on distance (angles are ignored).
- Polarization transfer matrix models polarization coupling and must be normalized.
- Doppler weights can optionally be calculated from receiver motion relative to path direction.
- Element positions and antenna orientation are fully considered for delay and phase.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::get_channels_planar(
                const arrayant<dtype> *tx_array,
                const arrayant<dtype> *rx_array,
                dtype Tx, dtype Ty, dtype Tz,
                dtype Tb, dtype Tt, dtype Th,
                dtype Rx, dtype Ry, dtype Rz,
                dtype Rb, dtype Rt, dtype Rh,
                const arma::Col<dtype> *aod,
                const arma::Col<dtype> *eod,
                const arma::Col<dtype> *aoa,
                const arma::Col<dtype> *eoa,
                const arma::Col<dtype> *path_gain,
                const arma::Col<dtype> *path_length,
                const arma::Mat<dtype> *M,
                arma::Cube<dtype> *coeff_re,
                arma::Cube<dtype> *coeff_im,
                arma::Cube<dtype> *delay,
                dtype center_frequency = dtype(0.0),
                bool use_absolute_delays = false,
                bool add_fake_los_path = false,
                arma::Col<dtype> *rx_Doppler = nullptr);
```

### Arguments:
- `const arrayant<dtype> ***tx_array**` (input)
  Pointer to the transmit antenna array object (with `n_tx` elements).

- `const arrayant<dtype> ***rx_array**` (input)
  Pointer to the receive antenna array object (with `n_rx` elements).

- `dtype **Tx**, **Ty**, **Tz**` (input)
  Transmitter position in Cartesian coordinates [m].

- `dtype **Tb**, **Tt**, **Th**` (input)
  Transmitter orientation (Euler) angles (bank, tilt, head) in [rad].

- `dtype **Rx**, **Ry**, **Rz**` (input)
  Receiver position in Cartesian coordinates [m].

- `dtype **Rb**, **Rt**, **Rh**` (input)
  Receiver orientation (Euler) angles (bank, tilt, head) in [rad].

- `const arma::Col<dtype> ***aod**` (input)
  Departure azimuth angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***eod**` (input)
  Departure elevation angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***aoa**` (input)
  Arrival azimuth angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***eoa**` (input)
  Arrival elevation angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***path_gain**` (input)
  Path gains in linear scale, Length `n_path`.

- `const arma::Col<dtype> ***path_length**` (input)
  Path lengths from TX to RX phase center, Length `n_path`.

- `const arma::Mat<dtype> ***M**` (input)
  Polarization transfer matrix of size `[8, n_path]`, interleaved: (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH).

- `arma::Cube<dtype> ***coeff_re**` (output)
  Real part of channel coefficients, Size `[n_rx, n_tx, n_path(+1)]`.

- `arma::Cube<dtype> ***coeff_im**` (output)
  Imaginary part of channel coefficients, Size `[n_rx, n_tx, n_path(+1)]`.

- `arma::Cube<dtype> ***delay**` (output)
  Propagation delays in seconds, Size `[n_rx, n_tx, n_path(+1)]`.

- `dtype **center_frequency** = 0.0` (optional input)
  Center frequency in Hz; set to 0 to disable phase calculation. Default: `0.0`

- `bool **use_absolute_delays** = false` (optional input)
  If true, includes LOS delay in all paths. Default: `false`

- `bool **add_fake_los_path** = false` (optional input)
  Adds a zero-power LOS path if no LOS is present. Default: `false`

- `arma::Col<dtype> ***rx_Doppler** = nullptr` (optional output)
  Doppler weights for moving RX, Length `n_path(+1)`. Positive = towards path, Negative = away.

---
## get_channels_spherical
Calculate channel coefficients for spherical waves

### Description:
- Calculates MIMO channel coefficients and delays for a set of spherical wave paths between two antenna arrays.
- Interpolates antenna patterns (including orientation and polarization) for both transmitter and receiver arrays.
- Accurately models path-based propagation using provided scatterer positions.
- Supports LOS path identification and handles complex polarization coupling.
- Element positions and antenna orientation are fully considered for delay and phase.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::get_channels_spherical(
                const arrayant<dtype> *tx_array,
                const arrayant<dtype> *rx_array,
                dtype Tx, dtype Ty, dtype Tz,
                dtype Tb, dtype Tt, dtype Th,
                dtype Rx, dtype Ry, dtype Rz,
                dtype Rb, dtype Rt, dtype Rh,
                const arma::Mat<dtype> *fbs_pos,
                const arma::Mat<dtype> *lbs_pos,
                const arma::Col<dtype> *path_gain,
                const arma::Col<dtype> *path_length,
                const arma::Mat<dtype> *M,
                arma::Cube<dtype> *coeff_re,
                arma::Cube<dtype> *coeff_im,
                arma::Cube<dtype> *delay,
                dtype center_frequency = dtype(0.0),
                bool use_absolute_delays = false,
                bool add_fake_los_path = false,
                arma::Cube<dtype> *aod = nullptr,
                arma::Cube<dtype> *eod = nullptr,
                arma::Cube<dtype> *aoa = nullptr,
                arma::Cube<dtype> *eoa = nullptr)
```

### Arguments:
- `const arrayant<dtype> ***tx_array**` (input)
  Pointer to the transmit antenna array object (with `n_tx` elements).

- `const arrayant<dtype> ***rx_array**` (input)
  Pointer to the receive antenna array object (with `n_rx` elements).

- `dtype **Tx**, **Ty**, **Tz**` (input)
  Transmitter position in Cartesian coordinates [m].

- `dtype **Tb**, **Tt**, **Th**` (input)
  Transmitter orientation (Euler) angles (bank, tilt, head) in [rad].

- `dtype **Rx**, **Ry**, **Rz**` (input)
  Receiver position in Cartesian coordinates [m].

- `dtype **Rb**, **Rt**, **Rh**` (input)
  Receiver orientation (Euler) angles (bank, tilt, head) in [rad].

- `const arma::Mat<dtype> ***fbs_pos**` (input)
  First-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Mat<dtype> ***lbs_pos**` (input)
  Last-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Col<dtype> ***path_gain**` (input)
  Path gains in linear scale, Length `n_path`.

- `const arma::Col<dtype> ***path_length**` (input)
  Path lengths from TX to RX phase center Length `n_path`.

- `const arma::Mat<dtype> ***M**` (input)
  Polarization transfer matrix of size `[8, n_path]`, interleaved: (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH).

- `arma::Cube<dtype> ***coeff_re**` (output)
  Real part of channel coefficients, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***coeff_im**` (output)
  Imaginary part of channel coefficients, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***delay**` (output)
  Propagation delays in seconds, Size `[n_rx, n_tx, n_path]`.

- `dtype **center_frequency** = 0.0` (optional input)
  Center frequency in Hz; set to 0 to disable phase calculation. Default: `0.0`

- `bool **use_absolute_delays** = false` (optional input)
  If true, includes LOS delay in all paths. Default: `false`

- `bool **add_fake_los_path** = false` (optional input)
  Adds a zero-power LOS path if no LOS is present. Default: `false`

- `arma::Cube<dtype> ***aod** = nullptr` (optional output)
  Azimuth of Departure angles in radians, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***eod** = nullptr` (optional output)
  Elevation of Departure angles in radians, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***aoa** = nullptr` (optional output)
  Azimuth of Arrival angles in radians, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***eoa = nullptr` (optional output)
  Elevation of Arrival angles in radians, Size `[n_rx, n_tx, n_path]`.

---

# Math Functions

| Function | Description |
| --- | --- |
| [fast_acos](#fast_acos) | Fast, approximate arc-cosine |
| [fast_asin](#fast_asin) | Fast, approximate arc-sine |
| [fast_atan2](#fast_atan2) | Fast, approximate two-argument arc-tangent |
| [fast_cart2geo](#fast_cart2geo) | Fast, approximate Cartesian-to-geographic conversion |
| [fast_geo2cart](#fast_geo2cart) | Fast, approximate geographic-to-Cartesian conversion |
| [fast_sincos](#fast_sincos) | Fast, approximate sine/cosine |
| [fast_slerp](#fast_slerp) | Fast, approximate spherical interpolation (SLERP) for complex value pairs |

---
## fast_acos
Fast, approximate arc-cosine

### Description:
Computes elementwise arc-cosine for an Armadillo vector. Designed for high throughput on modern CPUs.
- Operates on input values in [-1, 1]
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::acosf`
- For x in [-1, 1], the maximum error is approximately 2 ULP (~2.4e-7)
- Input values outside [-1, 1] produce NaN (IEEE compliant)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

### Declaration:
```
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &c);
void quadriga_lib::fast_acos(const arma::vec &x, arma::fvec &c);
```

### Arguments:
- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)
  Input values in [-1, 1]. Length `[n]`.

- `arma::fvec &**c**` (output)
  Set to `acos(x)`. Resized to length `n` if needed. Length `[n]`.

### Example:
```
arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, 1000);
arma::fvec c;
quadriga_lib::fast_acos(x, c);
```

---
## fast_asin
Fast, approximate arc-sine

### Description:
Computes elementwise arc-sine for an Armadillo vector. Designed for high throughput on modern CPUs.
- Operates on input values in [-1, 1]
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::asinf`
- For x in [-1, 1], the maximum error is approximately 2 ULP (~2.4e-7)
- Input values outside [-1, 1] produce NaN (IEEE compliant)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

### Declaration:
```
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &s);
void quadriga_lib::fast_asin(const arma::vec &x, arma::fvec &s);
```

### Arguments:
- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)
  Input values in [-1, 1]. Length `[n]`.

- `arma::fvec &**s**` (output)
  Set to `asin(x)`. Resized to length `n` if needed. Length `[n]`.

### Example:
```
arma::fvec x = arma::linspace<arma::fvec>(-1.0f, 1.0f, 1000);
arma::fvec s;
quadriga_lib::fast_asin(x, s);
```

---
## fast_atan2
Fast, approximate two-argument arc-tangent

### Description:
Computes elementwise `atan2(y, x)` for two Armadillo vectors. Designed for high throughput on modern CPUs.
- Returns angles in radians in the range (-pi, pi]
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::atan2f`
- Maximum error is approximately 3 ULP (~3.6e-7) across the full domain
- `atan2(0, 0)` returns 0; `atan2(±0, -0)` returns `±0` (not `±pi`)
- Both input vectors must have the same length
- Input and output cannot alias (in-place operation not allowed)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

### Declaration:
```
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &a);
void quadriga_lib::fast_atan2(const arma::vec &y, const arma::vec &x, arma::fvec &a);
```

### Arguments:
- `const arma::fvec &**y**` or `const arma::vec &**y**` (input)
  Y-coordinates (numerator of atan2). Length `[n]`.

- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)
  X-coordinates (denominator of atan2). Length `[n]`.

- `arma::fvec &**a**` (output)
  Set to `atan2(y, x)` in radians. Resized to length `n` if needed. Length `[n]`.

### Example:
```
arma::fvec y = {1.0f, -1.0f, 0.0f, 1.0f};
arma::fvec x = {1.0f,  1.0f, -1.0f, 0.0f};
arma::fvec a;
quadriga_lib::fast_atan2(y, x, a);
// a ≈ {0.7854, -0.7854, 3.1416, 1.5708}
```

---
## fast_cart2geo
Fast, approximate Cartesian-to-geographic conversion

### Description:
Converts elementwise unit-sphere Cartesian coordinates to azimuth/elevation angles (in radians).
- az = atan2(y, x), el = asin(clamp(z, -1, 1))
- The z-coordinate is clamped to [-1, 1] before computing asin, guarding against FMA rounding artefacts
  from upstream matrix multiplications that can push abs(z) slightly above 1
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::atan2f` / `std::asinf`
- All input vectors must have the same length
- Input and output cannot alias (in-place operation not allowed)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

### Declaration:
```
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el);
void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::fvec &az, arma::fvec &el);
```

### Arguments:
- `const arma::fvec &**x**` or `const arma::vec &**x**` (input)
  X-coordinates. Length `[n]`.

- `const arma::fvec &**y**` or `const arma::vec &**y**` (input)
  Y-coordinates. Length `[n]`.

- `const arma::fvec &**z**` or `const arma::vec &**z**` (input)
  Z-coordinates. Length `[n]`.

- `arma::fvec &**az**` (output)
  Azimuth angles in radians. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**el**` (output)
  Elevation angles in radians. Resized to length `n` if needed. Length `[n]`.

### Example:
```
arma::fvec x = {1.0f, 0.0f, -1.0f};
arma::fvec y = {0.0f, 1.0f, 0.0f};
arma::fvec z = {0.0f, 0.0f, 0.0f};
arma::fvec az, el;
quadriga_lib::fast_cart2geo(x, y, z, az, el);
// az ≈ {0.0, 1.5708, 3.1416},  el ≈ {0.0, 0.0, 0.0}
```

---
## fast_geo2cart
Fast, approximate geographic-to-Cartesian conversion

### Description:
Converts elementwise azimuth/elevation angles (in radians) to unit-sphere Cartesian coordinates.
- x = cos(el) * cos(az), y = cos(el) * sin(az), z = sin(el)
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::sinf` / `std::cosf`
- Optionally returns intermediate sin/cos values via pointer arguments; pass `nullptr` to skip
- Both input vectors must have the same length
- Input and output cannot alias (in-place operation not allowed)
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

### Declaration:
```
void quadriga_lib::fast_geo2cart(const arma::fvec &az, const arma::fvec &el,
                                 arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                 arma::fvec *sAZ = nullptr, arma::fvec *cAZ = nullptr,
                                 arma::fvec *sEL = nullptr, arma::fvec *cEL = nullptr);
void quadriga_lib::fast_geo2cart(const arma::vec &az, const arma::vec &el,
                                 arma::fvec &x, arma::fvec &y, arma::fvec &z,
                                 arma::fvec *sAZ = nullptr, arma::fvec *cAZ = nullptr,
                                 arma::fvec *sEL = nullptr, arma::fvec *cEL = nullptr);
```

### Arguments:
- `const arma::fvec &**az**` or `const arma::vec &**az**` (input)
  Azimuth angles in radians. Length `[n]`.

- `const arma::fvec &**el**` or `const arma::vec &**el**` (input)
  Elevation angles in radians. Length `[n]`.

- `arma::fvec &**x**` (output)
  X-coordinates on the unit sphere. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**y**` (output)
  Y-coordinates on the unit sphere. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**z**` (output)
  Z-coordinates on the unit sphere. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec ***sAZ** = nullptr` (optional output)
  If non-null, set to `sin(az)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

- `arma::fvec ***cAZ** = nullptr` (optional output)
  If non-null, set to `cos(az)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

- `arma::fvec ***sEL** = nullptr` (optional output)
  If non-null, set to `sin(el)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

- `arma::fvec ***cEL** = nullptr` (optional output)
  If non-null, set to `cos(el)`. Resized to length `n` if needed. Length `[n]` or `nullptr`.

### Example:
```
arma::fvec az = {0.0f, 1.5708f, 3.1416f};
arma::fvec el = {0.0f, 0.5f, -0.5f};
arma::fvec x, y, z, sAZ, cEL;
quadriga_lib::fast_geo2cart(az, el, x, y, z, &sAZ, nullptr, nullptr, &cEL);
```

---
## fast_sincos
Fast, approximate sine/cosine

### Description:
Computes elementwise sine and/or cosine for an Armadillo vector. Designed for high throughput on modern CPUs.
- Operates on input angles in radians
- AVX2-optimized (8 floats per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Results are approximate and may differ from `std::sinf` / `std::cosf`
- For x in [-pi, pi], the maximum absolute error is 2^(-22.1), and larger otherwise
- For x in [-500, 500], the maximum absolute error is 2^(-16.0)
- Either output (`s` or `c`) may be `nullptr` to skip its computation
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)

### Declaration:
```
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
void quadriga_lib::fast_sincos(const arma::vec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
```

### Arguments:
- `const arma::fvec **x**` or `const arma::vec **x**` (input)
  Input angles in radians. Size `[n]`.

- `arma::fvec ***s** = nullptr` (optional output)
  If non-null, set to `sin(x)`. Resized to length `n` if needed. Size `[n]` or `nullptr`.

- `arma::fvec ***c** = nullptr` (optional output)
  If non-null, set to `cos(x)`. Resized to length `n` if needed. Size `[n]` or `nullptr`.

### Returns:
- `void` (output)
  No return value. Results written via output pointers.

### Example:
```
arma::fvec x = arma::linspace[arma::fvec](arma::fvec)(0.0f, 6.2831853f, 1000);
arma::fvec s, c;
quadriga_lib::fast_sincos(x, &s, &c);      // compute both
quadriga_lib::fast_sincos(x, &s, nullptr); // compute sine only
quadriga_lib::fast_sincos(x, nullptr, &c); // compute cosine only
```

---
## fast_slerp
Fast, approximate spherical interpolation (SLERP) for complex value pairs

### Description:
Interpolates elementwise between two complex-valued vectors using spherical linear interpolation
(SLERP) on the normalised directions and linear interpolation of amplitudes.
- Processes per-element interpolation weights (0 = A, 1 = B)
- AVX2-optimized (8 complex pairs per lane); scalar fallback without AVX2 or on non-AVX2 CPUs
- Parallelizes across cores with OpenMP when enabled
- Near-antipodal inputs (phase angle close to pi) smoothly transition to a linear fallback
- If both input amplitudes are negligible, the output is zero
- Maximum error versus double-precision reference is approximately 5 ULP
- Allowed input datatype: `float` (Armadillo `fvec`) or `double` (Armadillo `vec`)
- All input vectors must have the same length

### Declaration:
```
void quadriga_lib::fast_slerp(const arma::fvec &Ar, const arma::fvec &Ai,
                              const arma::fvec &Br, const arma::fvec &Bi,
                              const arma::fvec &w,
                              arma::fvec &Xr, arma::fvec &Xi);
void quadriga_lib::fast_slerp(const arma::vec &Ar, const arma::vec &Ai,
                              const arma::vec &Br, const arma::vec &Bi,
                              const arma::vec &w,
                              arma::fvec &Xr, arma::fvec &Xi);
```

### Arguments:
- `const arma::fvec &**Ar**` or `const arma::vec &**Ar**` (input)
  Real part of source A. Length `[n]`.

- `const arma::fvec &**Ai**` or `const arma::vec &**Ai**` (input)
  Imaginary part of source A. Length `[n]`.

- `const arma::fvec &**Br**` or `const arma::vec &**Br**` (input)
  Real part of source B. Length `[n]`.

- `const arma::fvec &**Bi**` or `const arma::vec &**Bi**` (input)
  Imaginary part of source B. Length `[n]`.

- `const arma::fvec &**w**` or `const arma::vec &**w**` (input)
  Per-element interpolation weight in [0, 1]. 0 returns A, 1 returns B. Length `[n]`.

- `arma::fvec &**Xr**` (output)
  Real part of interpolated result. Resized to length `n` if needed. Length `[n]`.

- `arma::fvec &**Xi**` (output)
  Imaginary part of interpolated result. Resized to length `n` if needed. Length `[n]`.

### Returns:
- `void` (output)
  No return value. Results written to Xr and Xi.

### Example:
```
arma::fvec Ar = {1.0f, 0.0f}, Ai = {0.0f, 1.0f};
arma::fvec Br = {0.0f, 1.0f}, Bi = {1.0f, 0.0f};
arma::fvec w = {0.5f, 0.5f};
arma::fvec Xr, Xi;
quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);
```

---

# Miscellaneous / Tools

| Function | Description |
| --- | --- |
| [acdf](#acdf) | Calculate the empirical averaged cumulative distribution function (CDF) |
| [calc_angular_spreads_sphere](#calc_angular_spreads_sphere) | Calculate azimuth and elevation angular spreads with spherical wrapping |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases |
| [calc_delay_spread](#calc_delay_spread) | Calculate the RMS delay spread in [s] |
| [calc_rician_k_factor](#calc_rician_k_factor) | Calculate the Rician K-Factor from channel impulse response data |
| [calc_rotation_matrix](#calc_rotation_matrix) | Calculate rotation matrices from Euler angles |
| [cart2geo](#cart2geo) | Convert Cartesian coordinates to geographic coordinates (azimuth, elevation, distance) |
| [colormap](#colormap) | Generate colormap |
| [geo2cart](#geo2cart) | Transform geographic (azimuth, elevation, length) to Cartesian coordinates |
| [interp_1D / interp_2D](#interp_1d-interp_2d) | Perform linear interpolation (1D or 2D) on single or multiple data sets. |
| [write_png](#write_png) | Write data to a PNG file |

---
## acdf
Calculate the empirical averaged cumulative distribution function (CDF)

### Description:
- Calculates the empirical CDF from the given data matrix, where each column represents an
  independent data set (e.g., repeated experiment runs).
- Individual CDFs are computed per column by histogramming into the given (or auto-generated) bins
  and taking the cumulative sum normalized by the number of valid samples.
- An averaged CDF is obtained by interpolating in quantile space: for a fine grid of probability
  levels, the corresponding x-values from each individual CDF are averaged, then the result is
  mapped back to the bin grid.
- Quantile statistics (mean and standard deviation) are reported at the 0.1, 0.2, ..., 0.9
  probability levels.
- `Inf` and `NaN` values in the data are excluded from the computation.
- If `bins` points to an empty vector, 201 equally spaced bins spanning the data range are
  generated and stored back. If `bins` points to a non-empty vector, those bin centers are used.
  If `bins` is `nullptr`, bins are auto-generated internally.

### Declaration:
```
template <typename dtype>
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
                        arma::Col<dtype> *bins = nullptr,
                        arma::Mat<dtype> *Sh = nullptr,
                        arma::Col<dtype> *Sc = nullptr,
                        arma::Col<dtype> *mu = nullptr,
                        arma::Col<dtype> *sig = nullptr,
                        arma::uword n_bins = 201);
```

### Arguments:
- `const arma::Mat<dtype> &**data**` (input)
  Input data matrix. Size `[n_samples, n_sets]`. Each column is one data set.

- `arma::Col<dtype> ***bins** = nullptr` (optional input/output)
  Bin centers for the histogram. Length `[n_bins]`. If pointing to an empty vector, auto-generated
  bins are stored here. If pointing to a non-empty vector, those bin centers are used. If `nullptr`,
  bins are auto-generated internally.

- `arma::Mat<dtype> ***Sh** = nullptr` (optional output)
  Individual CDFs, one per column of data. Size `[n_bins, n_sets]`.

- `arma::Col<dtype> ***Sc** = nullptr` (optional output)
  Averaged CDF obtained by quantile-space averaging across data sets. Length `[n_bins]`.

- `arma::Col<dtype> ***mu** = nullptr` (optional output)
  Mean of the 0.1, 0.2, ..., 0.9 quantiles across data sets. Length `[9]`.

- `arma::Col<dtype> ***sig** = nullptr` (optional output)
  Standard deviation of the 0.1, 0.2, ..., 0.9 quantiles across data sets. Length `[9]`.

- `arma::uword **n_bins** = 201` (input)
  Number of bins to generate when bins are auto-generated. Must be at least 2. Ignored when
  non-empty bins are provided.

### Example:
```
#include "quadriga_tools.hpp"

// Generate random data: 10000 samples x 5 experiment runs
arma::mat data = arma::randn<arma::mat>(10000, 5);

arma::vec bins;
arma::mat Sh;
arma::vec Sc, mu, sig;
quadriga_lib::acdf(data, &bins, &Sh, &Sc, &mu, &sig);

// bins has 201 elements, Sh is [201, 5], Sc is [201], mu and sig are [9]
```

---
## calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

### Description:
- Calculates the RMS azimuth and elevation angular spreads from a set of power-weighted angles.
- Inputs and outputs use `std::vector<arma::Col<dtype>>` so that each channel impulse response
  (CIR) can have a different number of paths.
- The RMS angular spread is computed as `sqrt(sum(pw .* d.^2))` where `d` are the wrapped
  deviations from the circular mean. This is the second-moment definition used in 3GPP TR 38.901,
  as opposed to the standard-deviation form `sqrt(E[d^2] - E[d]^2)`.
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- The bank angle is derived analytically from the eigenvectors of the 2x2 power-weighted
  covariance matrix of the centered azimuth and elevation angles.
- An optional quantization step can group nearby paths before computing the spread.
- Setting `disable_wrapping` to true skips the rotation and computes spreads directly from the
  raw azimuth and elevation angles (equivalent to treating them as independent 1D variables).
  In this mode, the orientation output will be zero and phi/theta will equal the input az/el.

### Declaration:
```
template <typename dtype>
void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<dtype>> &az,
    const std::vector<arma::Col<dtype>> &el,
    const std::vector<arma::Col<dtype>> &powers,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    std::vector<arma::Col<dtype>> *phi = nullptr,
    std::vector<arma::Col<dtype>> *theta = nullptr,
    bool disable_wrapping = false,
    bool calc_bank_angle = true,
    dtype quantize = (dtype)0);
```

### Arguments:
- `const std::vector<arma::Col<dtype>> &**az**` (input)
  Azimuth angles in [rad], ranging from -pi to pi. Vector of length `n_cir`, each element has
  length `n_path` (may differ per CIR).

- `const std::vector<arma::Col<dtype>> &**el**` (input)
  Elevation angles in [rad], ranging from -pi/2 to pi/2. Vector of length `n_cir`, each element
  has length `n_path` matching the corresponding element in `az`.

- `const std::vector<arma::Col<dtype>> &**powers**` (input)
  Path powers in [W]. Vector of length `n_cir`, each element has length `n_path` matching the
  corresponding element in `az`.

- `arma::Col<dtype> ***azimuth_spread** = nullptr` (optional output)
  RMS azimuth angular spread in [rad]. Length `[n_cir]`.

- `arma::Col<dtype> ***elevation_spread** = nullptr` (optional output)
  RMS elevation angular spread in [rad]. Length `[n_cir]`.

- `arma::Mat<dtype> ***orientation** = nullptr` (optional output)
  Power-weighted mean-angle orientation using aircraft principal axes: row 0 = bank angle,
  row 1 = tilt angle, row 2 = heading angle, all in [rad]. Size `[3, n_cir]`.

- `std::vector<arma::Col<dtype>> ***phi** = nullptr` (optional output)
  Rotated azimuth angles in [rad]. Vector of length `n_cir`, each element has length `n_path`.

- `std::vector<arma::Col<dtype>> ***theta** = nullptr` (optional output)
  Rotated elevation angles in [rad]. Vector of length `n_cir`, each element has length `n_path`.

- `bool **disable_wrapping** = false` (input)
  If true, skip the spherical rotation and compute spreads directly from raw angles. The
  orientation output will be zero and phi/theta will equal the input az/el.

- `bool **calc_bank_angle** = true` (input)
  If true, the optimal bank angle is computed analytically. Only used when `disable_wrapping`
  is false.

- `dtype **quantize** = 0` (input)
  Angular quantization step in [deg]. Paths within this angular distance are grouped and their
  powers summed before computing the spread. Set to 0 to treat all paths independently.

### Example:
```
std::vector<arma::vec> az(2), el(2), powers(2);
az[0] = {0.1, -0.1, 0.05};              // CIR 0: 3 paths
az[1] = {0.2, -0.2, 0.1, -0.1};         // CIR 1: 4 paths
el[0] = {0.0, 0.0, 0.0};
el[1] = {0.05, -0.05, 0.0, 0.0};
powers[0] = {1.0, 1.0, 0.5};
powers[1] = {2.0, 1.0, 1.5, 0.5};

arma::vec as, es;
arma::mat orient;
quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es, &orient);
// as(0), as(1) contain the azimuth spreads for each CIR
```

---
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

### Description:
- Computes the aggregate cross-polarization ratio (XPR) from the polarization transfer matrices
  of all channel impulse responses (CIRs) using the total-power-ratio method (Option B).
- For each CIR, the total co-polarized and cross-polarized received powers are accumulated
  across all qualifying paths, and the XPR is obtained as a single ratio of the totals.
- This method is physically meaningful: it corresponds to what a receiver antenna measures as
  the ratio of co-polarized to cross-polarized energy across the entire channel impulse response.
- In addition to the linear V/H basis, the XPR is also computed in the circular LHCP/RHCP basis
  by applying the unitary Jones matrix transformation M_circ = T * M_lin * T^-1.
- The LOS path is identified by comparing each path's absolute length against the direct
  TX-RX distance `dTR`. All paths with `path_length < dTR + window_size` are excluded from
  the XPR calculation by default (controlled by `include_los`).
- The polarization transfer matrix `M` is stored in column-major order with interleaved
  real/imaginary parts: rows = [Re(M_vv), Im(M_vv), Re(M_hv), Im(M_hv), Re(M_vh), Im(M_vh),
  Re(M_hh), Im(M_hh)], i.e., 8 rows per path.
- `M` may or may not be normalized. Normalization does not affect the XPR since it cancels
  in the ratio. However, it does affect the path gain output `pg`.
- If the total cross-polarized power is zero and co-polarized power is positive (perfect
  polarization isolation), the XPR is set to infinity. If both are zero (no qualifying paths),
  the XPR is set to 0.

### Declaration:
```
template <typename dtype>
void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &M,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr = nullptr,
    arma::Col<dtype> *pg = nullptr,
    bool include_los = false,
    dtype window_size = 0.01);
```

### Arguments:
- `const std::vector<arma::Col<dtype>> &**powers**` (input)
  Path powers in Watts. Vector of length `[n_cir]`, each element is a column vector of length `[n_path]`.

- `const std::vector<arma::Mat<dtype>> &**M**` (input)
  Polarization transfer matrices. Vector of length `[n_cir]`, each element is a matrix of size `[8, n_path]`
  with interleaved real/imaginary parts in column-major order.

- `const std::vector<arma::Col<dtype>> &**path_length**` (input)
  Absolute path length from TX to RX phase center in meters. Vector of length `[n_cir]`,
  each element is a column vector of length `[n_path]`.

- `const arma::Mat<dtype> &**tx_pos**` (input)
  Transmitter position in Cartesian coordinates. Size `[3, 1]` (fixed TX) or `[3, n_cir]` (mobile TX).

- `const arma::Mat<dtype> &**rx_pos**` (input)
  Receiver position in Cartesian coordinates. Size `[3, 1]` (fixed RX) or `[3, n_cir]` (mobile RX).

- `arma::Mat<dtype> ***xpr** = nullptr` (optional output)
  Cross-polarization ratio in linear scale. Size `[n_cir, 6]` with columns:
  0 = Aggregate linear XPR (total V+H co-pol / total V+H cross-pol),
  1 = V-XPR (|M_vv|^2 / |M_hv|^2, power-summed over paths),
  2 = H-XPR (|M_hh|^2 / |M_vh|^2, power-summed over paths),
  3 = Aggregate circular XPR (total L+R co-pol / total L+R cross-pol),
  4 = LHCP XPR (|M_LL|^2 / |M_RL|^2, power-summed over paths),
  5 = RHCP XPR (|M_RR|^2 / |M_LR|^2, power-summed over paths).

- `arma::Col<dtype> ***pg** = nullptr` (optional output)
  Total path gain computed over all paths (including LOS). Length `[n_cir]`.
  Calculated as the sum of `powers[p] * (|M_vv|^2 + |M_hv|^2 + |M_vh|^2 + |M_hh|^2)` over all paths.

- `bool **include_los** = false` (input)
  If `true`, include LOS and near-LOS paths in the XPR calculation.
  If `false` (default), exclude paths with `path_length < dTR + window_size`.

- `dtype **window_size** = 0.01` (input)
  LOS window size in meters. Paths within `dTR + window_size` of the direct path are excluded
  from the XPR calculation when `include_los` is `false`. Default is 0.01 m (1 cm).

### Example:
```
#include "quadriga_channel.hpp"

// Single CIR with 3 paths
arma::vec pw = {1.0, 0.5, 0.3};
arma::mat M(8, 3);
M.col(0) = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}; // LOS: diagonal
M.col(1) = {0.9, 0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0};   // NLOS
M.col(2) = {0.7, 0.0, 0.2, 0.0, 0.15, 0.0, 0.6, 0.0};   // NLOS

arma::vec pl = {10.0, 12.0, 15.0};
arma::mat tx(3, 1); tx.col(0) = {0.0, 0.0, 0.0};
arma::mat rx(3, 1); rx.col(0) = {10.0, 0.0, 0.0};

std::vector<arma::vec> powers_vec = {pw};
std::vector<arma::mat> M_vec = {M};
std::vector<arma::vec> pl_vec = {pl};

arma::mat xpr;
arma::vec pg;

quadriga_lib::calc_cross_polarization_ratio(powers_vec, M_vec, pl_vec, tx, rx, &xpr, &pg);
// xpr has size [1, 6], pg has size [1]
```

---
## calc_delay_spread
Calculate the RMS delay spread in [s]

### Description:
- Computes the root-mean-square (RMS) delay spread from a given set of delays and corresponding
  linear-scale powers for each channel impulse response (CIR).
- An optional power threshold in [dB] relative to the strongest path can be applied. Paths with
  power below `p_max(dB) - threshold` are excluded from the calculation.
- An optional granularity parameter in [s] groups paths in the delay domain. Powers of paths
  falling into the same delay bin are summed before computing the delay spread. This is useful
  when the system bandwidth limits the time resolution (e.g. 50 ns at 20 MHz bandwidth).
- When granularity is applied the function recursively calls itself on the binned power delay
  profile.
- Optionally returns the mean delay for each CIR.

### Declaration:
```
template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    dtype threshold = 100.0,
    dtype granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

### Arguments:
- `const std::vector<arma::Col<dtype>> &**delays**` (input)
  Delays in [s]. A vector of length `n_cir`, where each element is an Armadillo column vector
  of length `n_path` (the number of paths may differ per CIR).

- `const std::vector<arma::Col<dtype>> &**powers**` (input)
  Path powers on a linear scale [W]. Same structure as `delays`.

- `dtype **threshold** = 100.0` (input)
  Power threshold in [dB] relative to the strongest path. Paths with power below
  `max_power / 10^(0.1 * threshold)` are excluded. Default: 100 dB (effectively all paths).

- `dtype **granularity** = 0.0` (input)
  Window size in [s] for grouping paths in the delay domain. Paths whose delays fall into the
  same bin of width `granularity` have their powers summed. Default: 0 (no grouping).

- `arma::Col<dtype> ***mean_delay** = nullptr` (optional output)
  Mean delay in [s] for each CIR. Length `[n_cir]`.

### Returns:
- `arma::Col<dtype> **ds**` (output)
  RMS delay spread in [s] for each CIR. Length `[n_cir]`.

### Example:
```
std::vector<arma::vec> delays = { {0.0, 1e-6, 2e-6} };
std::vector<arma::vec> powers = { {1.0, 0.5, 0.25} };
arma::vec mean_delay;
arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);
// ds(0) ≈ 0.6901e-6, mean_delay(0) ≈ 0.5714e-6
```

---
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

### Description:
- The Rician K-Factor (KF) is defined as the ratio of signal power in the dominant line-of-sight
  (LOS) path to the power in the scattered (non-line-of-sight, NLOS) paths.
- The LOS path is identified by matching the absolute path length with the direct distance between
  TX and RX positions (`dTR`).
- All paths arriving within `dTR + window_size` are considered LOS and their power is summed.
- Paths arriving after `dTR + window_size` are considered NLOS and their power is summed.
- If the total NLOS power is zero (i.e. no scattered paths), the K-Factor is set to infinity (`HUGE_VAL`).
- If the total LOS power is zero (i.e. no LOS paths), the K-Factor is set to zero.
- The transmitter and receiver positions can be fixed (size `[3, 1]`) or mobile (size `[3, n_cir]`).
  Fixed positions are reused for all channel snapshots.
- Optional output `pg` returns the total path gain (sum of all path powers) for each snapshot.

### Declaration:
```
template <typename dtype>
void quadriga_lib::calc_rician_k_factor(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Col<dtype> *kf = nullptr,
    arma::Col<dtype> *pg = nullptr,
    dtype window_size = 0.01);
```

### Arguments:
- `const std::vector<arma::Col<dtype>> &**powers**` (input)
  Path powers in Watts [W]. Vector of length `n_cir`, where each element is a column vector of
  length `n_path` (number of paths may vary per snapshot).

- `const std::vector<arma::Col<dtype>> &**path_length**` (input)
  Absolute path lengths from TX to RX phase center in meters. Vector of length `n_cir`, where
  each element is a column vector of length `n_path` matching the corresponding entry in `powers`.

- `const arma::Mat<dtype> &**tx_pos**` (input)
  Transmitter position in Cartesian coordinates [x; y; z]. Size `[3, 1]` for a fixed TX or
  `[3, n_cir]` for a mobile TX.

- `const arma::Mat<dtype> &**rx_pos**` (input)
  Receiver position in Cartesian coordinates [x; y; z]. Size `[3, 1]` for a fixed RX or
  `[3, n_cir]` for a mobile RX.

- `arma::Col<dtype> ***kf** = nullptr` (optional output)
  Rician K-Factor on linear scale. Length `[n_cir]`.

- `arma::Col<dtype> ***pg** = nullptr` (optional output)
  Total path gain (sum of path powers). Length `[n_cir]`.

- `dtype **window_size** = 0.01` (input)
  LOS window size in meters. Paths with length ≤ `dTR + window_size` are considered LOS.

### Example:
```
#include "quadriga_tools.hpp"

// Single snapshot with 3 paths
std::vector<arma::vec> powers(1), path_length(1);
powers[0] = {1.0, 0.5, 0.25};       // Path powers in W
path_length[0] = {10.0, 11.0, 12.0}; // Path lengths in m

arma::mat tx_pos(3, 1), rx_pos(3, 1);
tx_pos.col(0) = {0.0, 0.0, 0.0};
rx_pos.col(0) = {10.0, 0.0, 0.0};   // dTR = 10.0 m

arma::vec kf, pg;
quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);
// kf[0] = 1.0 / (0.5 + 0.25) = 1.333...
// pg[0] = 1.0 + 0.5 + 0.25 = 1.75
```

---
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

### Description:
- Computes 3D rotation matrices from input Euler angles (bank, tilt, head).
- The result is returned in column-major order as a 3×3 matrix per input orientation vector.
- Calculations are internally performed in double precision for improved numerical accuracy, even if `dtype` is `float`.
- Supports optional inversion of the y-axis and optional transposition of the output matrix.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
arma::Cube<dtype> quadriga_lib::calc_rotation_matrix(const arma::Cube<dtype> &orientation,
                bool invert_y_axis = false, bool transposeR = false);

arma::Mat<dtype> quadriga_lib::calc_rotation_matrix(const arma::Mat<dtype> &orientation,
                bool invert_y_axis = false, bool transposeR = false);

arma::Col<dtype> quadriga_lib::calc_rotation_matrix(const arma::Col<dtype> &orientation,
                bool invert_y_axis = false, bool transposeR = false);
```

### Arguments:
- `const arma::Cube<dtype> **&orientation**` or `const arma::Mat<dtype> **&orientation**` or `const arma::Col<dtype> **&orientation**` (input)
  Input Euler angles (bank, tilt, head) in radians, Size `[3, n_row, n_col]` or `[3, n_mat]` or Size `[3]`.

- `bool **invert_y_axis** = false` (optional input)
  If true, the y-axis of the rotation is inverted. Default: `false`.

- `bool **transposeR** = false` (optional input)
  If true, the transpose of the rotation matrix is returned. Default: `false`.

### Returns:
- `arma::Cube<dtype>` or `arma::Mat<dtype>` or `arma::Col<dtype>`
  Rotation matrices in column-major ordering. Size `[9, n_row, n_col]` or `[9, n_mat]` or `[9]`.

### Example:
```
arma::cube ori(3, 1, 1);
ori(0, 0, 0) = 0.0;         // bank
ori(1, 0, 0) = 0.0;         // tilt
ori(2, 0, 0) = 1.5708;      // head
auto R = quadriga_lib::calc_rotation_matrix(ori);
```

---
## cart2geo
Convert Cartesian coordinates to geographic coordinates (azimuth, elevation, distance)

### Description:
- Transforms 3D Cartesian coordinates `(x, y, z)` into geographic coordinates:
  - Azimuth angle [rad]
  - Elevation angle [rad]
  - Distance (vector norm)
- Azimuth is measured in the x-y plane from the x-axis; elevation is from the x-y plane upward.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
arma::Cube<dtype> quadriga_lib::cart2geo(const arma::Cube<dtype> &cart);

arma::Mat<dtype> quadriga_lib::cart2geo(const arma::Mat<dtype> &cart);

arma::Col<dtype> quadriga_lib::cart2geo(const arma::Col<dtype> &cart);
```

### Arguments:
- `const arma::Cube<dtype> ***cart**` or `const arma::Mat<dtype> ***cart**` or `const arma::Col<dtype> **cart**` (input)
  Cartesian coordinate vectors (x, y, z), Size `[3, n_row, n_col]` or `[3, n_row]` or `[3]`.

### Returns:
- `arma::Cube<dtype>` or `arma::Mat<dtype>` or `arma::Col<dtype>`
  Geographic coordinate vectors `(azimuth, elevation, distance)`, Size `[n_row, n_col, 3]` or `[n_row, 3]` or `[3]`.

### Example:
```
arma::vec cart = {1.0, 1.0, 1.0};
auto geo = quadriga_lib::cart2geo(cart);
```

---
## colormap
Generate colormap

### Description:
- Returns a 64x3 or 256x3 colormap matrix with RGB values in unsigned char format.
- Each row corresponds to an RGB color entry of the selected colormap.
- Useful for visualization purposes (e.g., heatmaps or 3D rendering).
- Available color maps include: `jet`, `parula`, `winter`, `hot`, `turbo`, `copper`, `spring`, `cool`, `gray`, `autumn`, `summer`.

### Declaration:
```
arma::uchar_mat quadriga_lib::colormap(std::string map, bool high_res = false)
```

### Arguments:
- `std::string **map**` (input)
  Name of the desired colormap. Must be one of:
  `"jet"`, `"parula"`, `"winter"`, `"hot"`, `"turbo"`, `"copper"`, `"spring"`, `"cool"`, `"gray"`, `"autumn"`, `"summer"`.

- `bool **high_res**` (input)
  Enables 256 color steps

### Returns:
- `arma::uchar_mat`
  A matrix of size `[64 x 3]` or `[256 x 3]` containing RGB color values as unsigned chars in the range `[0, 255]`.

### Example:
```
arma::uchar_mat cm = quadriga_lib::colormap("turbo");
```

---
## geo2cart
Transform geographic (azimuth, elevation, length) to Cartesian coordinates

### Description:
- Converts azimuth and elevation angles (in radians) into 3D Cartesian coordinates.
- Optional radial distance (`length`) can be provided; otherwise, unit vectors are returned.
- Useful for converting spherical direction data into vector representations.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
arma::Cube<dtype> quadriga_lib::geo2cart(
                const arma::Mat<dtype> &azimuth,
                const arma::Mat<dtype> &elevation,
                const arma::Mat<dtype> &length = {})
```

### Arguments:
- `const arma::Mat<dtype> **azimuth**` (input)
  Azimuth angles in radians. Size `[n_row, n_col]`.

- `const arma::Mat<dtype> **elevation**` (input)
  Elevation angles in radians. Size `[n_row, n_col]`.

- `const arma::Mat<dtype> **length** = {}` (optional input)
  Radial distance (length). Same size as azimuth/elevation or empty for unit vectors. Size `[n_row, n_col]` or `[0, 0]`.

### Returns:
- `arma::Cube<dtype>` (output)
  Cartesian coordinates with dimensions `[3, n_row, n_col]`, representing (x, y, z) for each input direction.

### Example:
```
arma::mat az(2, 2), el(2, 2), len(2, 2, arma::fill::ones);
auto cart = quadriga_lib::geo2cart(az, el, len);
```

---
## interp_1D / interp_2D
Perform linear interpolation (1D or 2D) on single or multiple data sets.

### Description:
- Interpolates given input data at specified output points.
- Supports single and multiple data sets.
- Returns interpolated results either directly or through reference argument.
- Data types (`dtype`): `float` or `double`

### Declarations:
```
void interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
               const arma::Col<dtype> &xo, const arma::Col<dtype> &yo, arma::Cube<dtype> &output);

arma::Cube<dtype> interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                            const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_2D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                           const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_1D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);

arma::Col<dtype> interp_1D(const arma::Col<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);
```

### Arguments:
- `input`: Input data array/matrix (size details below)
- `xi`: Input x-axis sampling points, vector of length `nx`
- `yi`: Input y-axis sampling points (for 2D only), vector of length `ny`
- `xo`: Output x-axis sampling points, vector of length `mx`
- `yo`: Output y-axis sampling points (for 2D only), vector of length `my`
- `output`: Interpolated data cube (modified in-place for one variant)

### Input / Output size details:
- 2D interpolation of multiple datasets (`arma::Cube`):
  Input size: `[ny, nx, ne]`, Output size: `[my, mx, ne]`

- 2D interpolation of single dataset (`arma::Mat`):
  Input size: `[ny, nx]`, Output size: `[my, mx]`

- 1D interpolation of multiple datasets (`arma::Mat`):
  Input size: `[nx, ne]`, Output size: `[mx, ne]`

- 1D interpolation of single dataset (`arma::Col`):
  Input length: `[nx]`, Output length: `[mx]`

### Examples:
- 2D interpolation example:
```
arma::cube input(5, 5, 2, arma::fill::randu); // example input data
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec yi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);
arma::vec yo = arma::linspace(0, 4, 10);

arma::cube output;
quadriga_lib::interp_2D(input, xi, yi, xo, yo, output);
```
- 1D interpolation example:
```
arma::vec input = arma::linspace(0, 1, 5);
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);

auto output = quadriga_lib::interp_1D(input, xi, xo);
```

---
## write_png
Write data to a PNG file

### Description:
- Converts input data into a color-coded PNG file for visualization
- Support optional selection of a colormap, as well a minimum and maximum value limits
- Allowed datatypes (`dtype`): `float` or `double`
- Uses the <a href="https://github.com/lvandeve/lodepng">LodePNG</a> library for PNG writing

### Declaration:
```
void write_png( const arma::Mat<dtype> &data, 
                std::string fn,              
                std::string colormap = "jet", 
                dtype min_val = NAN, 
                dtype max_val = NAN, 
                bool log_transform = false);
```

### Arguments:
- `const arma::Mat<dtype> **&data**`
  Data matrix

- `std::string **fn**`
  Path to the `.png` file to be written

- `std::string **colormap**`
  Name of the desired colormap. Must be one of:
  `"jet"`, `"parula"`, `"winter"`, `"hot"`, `"turbo"`, `"copper"`, `"spring"`, `"cool"`, `"gray"`, `"autumn"`, `"summer"`.

- `dtype **min_val**`
  Minimum value. Values below this value will have be encoded with the color of the smallest value.
  If `NAN` is provided (default), the lowest values is determined from the data.

- `dtype **max_val**`
  Maximum value. Values above this value will have be encoded with the color of the largest value.
  If `NAN` is provided (default), the largest values is determined from the data.

- `bool **log_transform**`
  If enabled, the `data` values are transformed to the log-domain (`10*log10(data)`) before processing.
  Default: false (disabled)

### See also:
- [colormap](#colormap)

---

# Site-Specific Simulation Tools

| Function | Description |
| --- | --- |
| [calc_diffraction_gain](#calc_diffraction_gain) | Calculate diffraction gain for multiple transmit and receive positions using a 3D triangular mesh |
| [combine_irs_coord](#combine_irs_coord) | Combine path interaction coordinates for channels with intelligent reflective surfaces (IRS) |
| [coord2path](#coord2path) | Convert path interaction coordinates into FBS/LBS positions, path length and angles |
| [generate_diffraction_paths](#generate_diffraction_paths) | Generate propagation paths for estimating the diffraction gain |
| [icosphere](#icosphere) | Construct a geodesic polyhedron from recursive icosahedron subdivision |
| [mitsuba_xml_file_write](#mitsuba_xml_file_write) | Write geometry and material data to a Mitsuba 3 XML scene file. |
| [obj_file_read](#obj_file_read) | Read Wavefront `.obj` file and extract geometry and material information |
| [obj_overlap_test](#obj_overlap_test) | Detect overlapping 3D objects in a triangular mesh |
| [path_to_tube](#path_to_tube) | Convert a 3D path into a tube surface for visualization |
| [point_cloud_aabb](#point_cloud_aabb) | Compute the Axis-Aligned Bounding Boxes (AABB) of a 3D point cloud |
| [point_cloud_segmentation](#point_cloud_segmentation) | Reorganize a point cloud into spatial sub-clouds for efficient processing |
| [point_cloud_split](#point_cloud_split) | Split a point cloud into two sub-clouds along a spatial axis |
| [point_inside_mesh](#point_inside_mesh) | Test whether 3D points are inside a triangle mesh using raycasting |
| [ray_mesh_interact](#ray_mesh_interact) | Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces |
| [ray_point_intersect](#ray_point_intersect) | Calculates the intersection of ray beams with points in three dimensions |
| [ray_triangle_intersect](#ray_triangle_intersect) | Calculates the intersection of rays and triangles in three dimensions |
| [subdivide_rays](#subdivide_rays) | Subdivide ray beams into four smaller sub-beams |
| [subdivide_triangles](#subdivide_triangles) | Subdivide triangles into smaller triangles |
| [triangle_mesh_aabb](#triangle_mesh_aabb) | Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes |
| [triangle_mesh_segmentation](#triangle_mesh_segmentation) | Reorganize a 3D mesh into smaller sub-meshes for faster processing |
| [triangle_mesh_split](#triangle_mesh_split) | Split a 3D mesh into two sub-meshes along a given axis |

---
## calc_diffraction_gain
Calculate diffraction gain for multiple transmit and receive positions using a 3D triangular mesh

### Description:
- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction from mesh geometry. The wave
  propagation between each TX-RX pair is divided into `n_path` elliptic-arc paths (controlled by `lod`),
  each approximated by `n_seg` line segments. 
- Individual segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients, 
  generalized to arbitrary 3D shapes.
- Optional sub-mesh indexing (see [triangle_mesh_segmentation](#triangle_mesh_segmentation)) accelerates computation by skipping
  geometry whose bounding box does not intersect the TX-RX path.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
void quadriga_lib::calc_diffraction_gain(
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *dest,
                const arma::Mat<dtype> *mesh,
                const arma::Mat<dtype> *mtl_prop,
                dtype center_frequency,
                int lod = 2,
                arma::Col<dtype> *gain = nullptr,
                arma::Cube<dtype> *coord = nullptr,
                int verbose = 0,
                const arma::u32_vec *sub_mesh_index = nullptr,
                int use_kernel = 0,
                int gpu_id = 0);
```

### Arguments:
- `const arma::Mat<dtype> ***orig**` (input)
  TX positions, size `[n_pos, 3]`.
- `const arma::Mat<dtype> ***dest**` (input)
  RX positions, size `[n_pos, 3]`.
- `const arma::Mat<dtype> ***mesh**` (input)
  Triangle vertices, each row `[X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3]`, size `[n_mesh, 9]`.
- `const arma::Mat<dtype> ***mtl_prop**` (input)
  Material properties per triangle, size `[n_mesh, 5]`. See [obj_file_read](#obj_file_read).
- `dtype **center_frequency**` (input)
  Center frequency in Hz.
- `int **lod** = 2` (optional input)
  Level of detail (0–6). Controls `n_path` and `n_seg`. See [generate_diffraction_paths](#generate_diffraction_paths).
- `arma::Col<dtype> ***gain**` (output, optional)
  Diffraction gain per TX-RX pair, linear scale, size `[n_pos]`.
- `arma::Cube<dtype> ***coord**` (output, optional)
  Diffracted path coordinates (excluding endpoints), size `[3, n_seg-1, n_pos]`.
- `int **verbose** = 0` (optional input)
  Verbosity level. Default: `0`.
- `const arma::u32_vec ***sub_mesh_index**` (input, optional)
  Sub-mesh index for acceleration, 0-based, length `[n_mesh]`. See [triangle_mesh_segmentation](#triangle_mesh_segmentation).
- `int **use_kernel** = 0` (optional input)
  Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA. Error if unavailable.
- `int **gpu_id** = 0` (optional input)
  CUDA device ID. Ignored for non-CUDA kernels.

### See also:
- [generate_diffraction_paths](#generate_diffraction_paths)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation)
- [obj_file_read](#obj_file_read)

---
## combine_irs_coord
Combine path interaction coordinates for channels with intelligent reflective surfaces (IRS)

### Description:
- Merges two propagation segments — (1) TX → IRS and (2) IRS → RX — into complete TX → RX paths via an IRS.
- Interaction coordinates of both segments are stored in a compressed format where `no_interact` is a vector of
  length `n_path` storing the number of interactions per path and `interact_coord` stores all
  interaction coordinates in path order.
- Each combined path includes interaction points from both segments, optionally reversed for each segment.
- The number of output paths `n_path_irs` is at most `n_path_1 × n_path_2`, but can be reduced by specifying `active_path`.
- Output includes the number and coordinates of interaction points per IRS-reflected path.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
void quadriga_lib::combine_irs_coord(
                dtype Ix, dtype Iy, dtype Iz,
                const arma::u32_vec *no_interact_1,
                const arma::Mat<dtype> *interact_coord_1,
                const arma::u32_vec *no_interact_2,
                const arma::Mat<dtype> *interact_coord_2,
                arma::u32_vec *no_interact,
                arma::Mat<dtype> *interact_coord,
                bool reverse_segment_1 = false,
                bool reverse_segment_2 = false,
                const std::vector<bool> *active_path = nullptr);
```

### Arguments:
- `dtype **Ix**, **Iy**, **Iz**` (input)
  IRS position in Cartesian coordinates `[m]`.

- `const arma::u32_vec ***no_interact_1**` (input)
  Number of interaction points in segment 1 (TX → IRS), vector of length `[n_path_1]`

- `const arma::Mat<dtype> ***interact_coord_1**` (input)
  Interaction coordinates for segment 1, matrix of size `[3, sum(no_interact_1)]`.

- `const arma::u32_vec ***no_interact_2**` (input)
  Number of interaction points in segment 2 (IRS → RX), vector of length `[n_path_2]`

- `const arma::Mat<dtype> ***interact_coord_2**` (input)
  Interaction coordinates for segment 2, matrix of size `[3, sum(no_interact_2)]`.

- `arma::u32_vec ***no_interact**` (output)
  Combined number of interaction points per IRS-based path, vector of length `[n_path_irs]`.

- `arma::Mat<dtype> ***interact_coord**` (output)
  Combined interaction coordinates, matrix of size `[3, sum(no_interact)]`.

- `bool **reverse_segment_1** = false` (optional input)
  If `true`, reverses the interaction coordinates of segment 1. TX and IRS positions are not swapped. Default: `false`.

- `bool **reverse_segment_2** = false` (optional input)
  If `true`, reverses the interaction coordinates of segment 2. IRS and RX positions are not swapped. Default: `false`.

- `const std::vector<bool> ***active_path** = nullptr` (optional input)
  Optional mask vector of length `[n_path_1 × n_path_2]`. If provided, only paths with `true` are combined. 
  This is generated as the output of [get_channels_irs](#get_channels_irs)

### Technical Notes:
- Paths are created by appending interaction coordinates from both segments. Reversing only affects the order of these coordinates.
- Output matrix `interact_coord` is built sequentially and should be preallocated only if the total number of interaction points is known in advance.
- The function supports sparsely activated paths via `active_path`, this is generated as a return value of [get_channels_irs](#get_channels_irs)
  and filters out paths that have very little power after being reflected by the IRS. 
- This function is typically used as a complementary step to delay or coefficient calculations involving IRS-based channels. 
  It calculated the required data for visualizing paths, e.g. in Blender or other visualization tools.

### See also:
- [get_channels_irs](#get_channels_irs) (for computing IRS channels)
- [coord2path](#coord2path) (for processing coordinates, calculating departure and arrival angels, etc.)

---
## coord2path
Convert path interaction coordinates into FBS/LBS positions, path length and angles

### Description:
- Interaction coordinates can be stored in a compressed format where `no_interact` is a vector of
  length `n_path` storing the number of interactions per path and `interact_coord` stores all
  interaction coordinates in path order.
- This function processes the interaction coordinates of to extract relevant propagation metrics.
- Calculates absolute path lengths and first/last bounce scatterer positions and angles.
- LOS paths are assigned a virtual FBS/LBS position at the midpoint between TX and RX.
- Automatically resizes output arguments if necessary.
- Optionally reverses TX and RX to simulate the reverse link.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::coord2path(
                dtype Tx, dtype Ty, dtype Tz, 
                dtype Rx, dtype Ry, dtype Rz,
                const arma::u32_vec *no_interact, 
                const arma::Mat<dtype> *interact_coord,
                arma::Col<dtype> *path_length, 
                arma::Mat<dtype> *fbs_pos, 
                arma::Mat<dtype> *lbs_pos,
                arma::Mat<dtype> *path_angles, 
                std::vector<arma::Mat<dtype>> *path_coord, 
                bool reverse_path);
```

### Arguments:
- `dtype **Tx**, **Ty**, **Tz**` (input)
  Transmitter position in Cartesian coordinates in [m]

- `dtype **Rx**, **Ry**, **Rz**` (input)
  Receiver position in Cartesian coordinates in [m]

- `const arma::u32_vec ***no_interact**` (input)
  Vector of length `n_path` indicating the number of interactions per path (0 = LOS)

- `const arma::Mat<dtype> ***interact_coord**` (input)
  Matrix of size `[3, sum(no_interact)]` containing interaction coordinates in path order

- `arma::Col<dtype> ***path_length**` (output, optional)
  Absolute path lengths from TX to RX, Length `[n_path]`

- `arma::Mat<dtype> ***fbs_pos**` (output, optional)
  First-bounce scatterer positions, Size `[3, n_path]`; For LOS, set to midpoint between TX and RX

- `arma::Mat<dtype> ***lbs_pos**` (output, optional)
  Last-bounce scatterer positions, Size `[3, n_path]`. For LOS, set to midpoint between TX and RX

- `arma::Mat<dtype> ***path_angles**` (output, optional)
  Departure and arrival angles: columns {AOD, EOD, AOA, EOA}, size `[n_path, 4]`

- `std::vector<arma::Mat<dtype>> ***path_coord**` (output, optional)
  Full path coordinates (including TX and RX), vector of size `n_path` with each entry of size `[3, n_interact+2]`

- `bool **reverse_path** = false` (optional input)
  If `true`, swaps TX and RX and reverses all interaction sequences.

### Example:
```
// Suppose we have two paths: one LOS, one single-bounce.
// Path 0: LOS (no_interact[0] = 0)
// Path 1: single bounce at {5,0,2}.

arma::u32_vec no_int = {0, 1};
arma::mat interact(3, 1);
interact.col(0) = {5.0, 0.0, 2.0};

arma::vec lengths;
arma::mat fbs, lbs;
arma::mat angles;
std::vector<arma::mat> coords;

quadriga_lib::coord2path<double>(
    0.0, 0.0, 0.0,  // TX at origin
    10.0, 0.0, 0.0, // RX at x = 10 m
    &no_int, &interact, &lengths, &fbs, &lbs, &angles, &coords);

// After the call:
// lengths: [10.0, 10.77]
// fbs:     Path 1 [5,0,0], Path 2 [5,0,2]
// lbs:     Path 1 [5,0,0], Path 2 [5,0,2]
// angles:  Path 1 [0, 0, pi, p], Path 2 [0, 22°, pi, 22°]
// coords[0]: [ [0,0,0], [10,0,0] ]
// coords[1]: [ [0,0,0], [5,0,2], [10,0,0] ]
```

---
## generate_diffraction_paths
Generate propagation paths for estimating the diffraction gain

### Description:
This function generates the elliptic propagation paths and corresponding weights necessary for the
calculation of the diffraction gain in [calc_diffraction_gain](#calc_diffraction_gain).

### Caveat:
- Each ellipsoid consists of `n_path` diffraction paths. The number of paths is determined by the
  level of detail (`lod`).
- All diffraction paths of an ellipsoid originate at `orig` and arrive at `dest`
- Each diffraction path has `n_seg` segments
- Points `orig` and `dest` lay on the semi-major axis of the ellipsoid
- The generated rays sample the volume of the ellipsoid
- Weights are calculated from the Knife-edge diffraction model when parts of the ellipsoid are shadowed
- Initial weights are normalized such that `sum(prod(weights,3),2) = 1`
- Inputs `orig` and `dest` may be provided as double or single precision
- Supported datatypes `dtype` are `float` or `double`

### Declaration:
```
void generate_diffraction_paths(
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *dest,
                dtype center_frequency,
                int lod,
                arma::Cube<dtype> *ray_x,
                arma::Cube<dtype> *ray_y,
                arma::Cube<dtype> *ray_z,
                arma::Cube<dtype> *weight);
```

### Arguments:
- `const arma::Mat<dtype> ***orig**` (input)
  Pointer to Armadillo matrix containing the origin points of the propagation ellipsoid (e.g.
  transmitter positions). Size: `[ n_pos, 3 ]`

- `const arma::Mat<dtype> ***dest**` (input)
  Pointer to Armadillo matrix containing the destination point of the propagation ellipsoid (e.g.
  receiver positions). Size: `[ n_pos, 3 ]`

- `dtype **center_frequency**` (input)
  The center frequency in [Hz], scalar, default = 299792458 Hz

- `int **lod**` (input)
  Level of detail, scalar value
  `lod = 1` | results in `n_path = 7` and `n_seg = 3`
  `lod = 2` | results in `n_path = 19` and `n_seg = 3`
  `lod = 3` | results in `n_path = 37` and `n_seg = 4`
  `lod = 4` | results in `n_path = 61` and `n_seg = 5`
  `lod = 5` | results in `n_path = 1` and `n_seg = 2` (for debugging)
  `lod = 6` | results in `n_path = 2` and `n_seg = 2` (for debugging)

- `arma::Cube<dtype> ***ray_x**` (output)
  Pointer to an Armadillo cube for the x-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- `arma::Cube<dtype> ***ray_y**` (output)
  Pointer to an Armadillo cube for the y-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- `arma::Cube<dtype> ***ray_z**` (output)
  Pointer to an Armadillo cube for the z-coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1 ]`
  Size will be adjusted if not set correctly.

- `arma::Cube<dtype> ***weight**` (output)
  Pointer to an Armadillo cube for the  weights; Size: `[ n_pos, n_path, n_seg ]`
  Size will be adjusted if not set correctly.

### See also:
- [calc_diffraction_gain](#calc_diffraction_gain)

---
## icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

### Description:
- Produces 20 × n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
arma::uword quadriga_lib::icosphere(
    arma::uword n_div,
    dtype radius,
    arma::Mat<dtype> *center,
    arma::Col<dtype> *length = nullptr,
    arma::Mat<dtype> *vert = nullptr,
    arma::Mat<dtype> *direction = nullptr,
    bool direction_xyz = false);
```

### Input Arguments:
- **`n_div`** — Number of subdivisions; generates 20 × n_div² faces
- **`radius`** — Radius of icosphere in meters
- **`direction_xyz`** (optional) — Output directions in Cartesian (true) or spherical azimuth/elevation (false)

### Output Arguments:
- **`center`** — Unit vectors to triangle face centers, `[n_faces, 3]`
- **`length`** (optional) — Magnitude of each `center` vector, `[n_faces]`
- **`vert`** (optional) — Vertex offsets from face center [x1,y1,z1,x2,y2,z2,x3,y3,z3], `[n_faces, 9]`
- **`direction`** (optional) — Edge directions; spherical [az1,el1,az2,el2,az3,el3] or Cartesian [x1,y1,z1,x2,y2,z2,x3,y3,z3] per `direction_xyz` flag, `[n_faces, 6]` or `[n_faces, 9]`

### Returns:
Number of generated triangular faces (20 × n_div²)

---
## mitsuba_xml_file_write
Write geometry and material data to a Mitsuba 3 XML scene file.

### Description:
This routine converts a triangular surface mesh stored in *quadriga-lib* data structures into the
XML format understood by **Mitsuba 3** <a href="https://www.mitsuba-renderer.org">www.mitsuba-renderer.org</a>.
The generated file can be loaded directly by **NVIDIA Sionna RT** for differentiable radio-propagation
simulations.

- Converts a 3D geometry mesh into Mitsuba 3 XML format for use with rendering tools.
- Enables exporting models from `quadriga-lib` to be used with **Mitsuba 3** or **Sionna RT**:
- <a href="https://www.mitsuba-renderer.org">Mitsuba 3</a>: Research-oriented retargetable rendering system.
- <a href="https://developer.nvidia.com/sionna">NVIDIA Sionna</a>: Hardware-accelerated differentiable ray tracer for wireless propagation, built on Mitsuba 3.
- Supports grouping faces into named objects and assigning materials by name.
- Optionally maps materials to ITU default presets used by Sionna RT.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
void quadriga_lib::mitsuba_xml_file_write(
                const std::string &fn,
                const arma::Mat<dtype> &vert_list,
                const arma::umat &face_ind,
                const arma::uvec &obj_ind,
                const arma::uvec &mtl_ind,
                const std::vector<std::string> &obj_names,
                const std::vector<std::string> &mtl_names,
                const arma::Mat<dtype> &bsdf = {},
                bool map_to_itu_materials = false);

```

### Arguments:
- `const std::string **fn**` (input)
  Output file name (including path and `.xml` extension).

- `const arma::Mat<dtype> **vert_list**` (input)
  Vertex list, size `[n_vert, 3]`, each row is a vertex (x, y, z) in Cartesian coordinates [m].

- `const arma::umat **face_ind**` (input)
  Face indices (0-based), size `[n_mesh, 3]`, each row defines a triangle via vertex indices.

- `const arma::uvec **obj_ind**` (input)
  Object indices (1-based), size `[n_mesh]`. Assigns each triangle to an object.

- `const arma::uvec **mtl_ind**` (input)
  Material indices (1-based), size `[n_mesh]`. Assigns each triangle to a material.

- `const std::vector<std::string> **obj_names**` (input)
  Names of objects. Length must be equal to `max(obj_ind)`.

- `const std::vector<std::string> **mtl_names**` (input)
  Names of materials. Length must be equal to `max(mtl_ind)`.

- `const arma::Mat<dtype> **bsdf** = {}` (optional input)
  Material reflectivity data (BSDF parameters), size `[mtl_names.size(), 17]`. If omitted, the `null` BSDF is used.
  Note that Sionna RT ignores all BSDF parameters. They are only used by the Mitsuma rendering system.
  See [obj_file_read](#obj_file_read) for a definition of the data fields.

- `bool **map_to_itu_materials** = false` (optional input)
  If true, maps material names to ITU-defined presets used by Sionna RT. Default: `false`

### See also:
- [obj_file_read](#obj_file_read)

---
## obj_file_read
Read Wavefront `.obj` file and extract geometry and material information

### Description:
- Parses a Wavefront `.obj` file containing triangularized 3D geometry.
- Extracts triangle face data, material properties, vertex indices, and optional metadata such as object/material names.
- Multiple triangles belonging to the same object are grouped together by `obj_ind`.
- Supports default and custom ITU-compliant materials encoded via the `usemtl` tag.
- Automatically resizes output matrices/vectors as needed to match the file content.
- Returns the number of triangular mesh elements found in the file.

- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::uword quadriga_lib::obj_file_read(
                std::string fn,
                arma::Mat<dtype> *mesh = nullptr,
                arma::Mat<dtype> *mtl_prop = nullptr,
                arma::Mat<dtype> *vert_list = nullptr,
                arma::umat *face_ind = nullptr,
                arma::uvec *obj_ind = nullptr,
                arma::uvec *mtl_ind = nullptr,
                std::vector<std::string> *obj_names = nullptr,
                std::vector<std::string> *mtl_names = nullptr,
                arma::Mat<dtype> *bsdf = nullptr,
                const std::string &materials_csv = "");
```

### Arguments:
- `std::string **fn**` (input)
  Path to the `.obj` file to be read.

- `arma::Mat<dtype> ***mesh** = nullptr` (optional output)
  Flattened triangle mesh data. Each row holds `[X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3]`. Size: `[n_mesh, 9]`.

- `arma::Mat<dtype> ***mtl_prop** = nullptr` (optional output)
  Material properties for each triangle. Size: `[n_mesh, 5]`.

- `arma::Mat<dtype> ***vert_list** = nullptr` (optional output)
  List of all vertex positions found in the `.obj` file. Size: `[n_vert, 3]`.

- `arma::umat ***face_ind** = nullptr` (optional output)
  Indices into `vert_list` for each triangle (0-based). Size: `[n_mesh, 3]`.

- `arma::uvec ***obj_ind** = nullptr` (optional output)
  Object index (1-based) for each triangle. Size: `[n_mesh]`.

- `arma::uvec ***mtl_ind** = nullptr` (optional output)
  Material index (1-based) for each triangle. Size: `[n_mesh]`.

- `std::vector<std::string> ***obj_names** = nullptr` (optional output)
  Names of objects found in the file. Length: `max(obj_ind)`.

- `std::vector<std::string> ***mtl_names** = nullptr` (optional output)
  Names of materials found in the file. Length: `max(mtl_ind)`.

- `arma::Mat<dtype> ***bsdf** = nullptr` (optional output)
  Principled BSDF (Bidirectional Scattering Distribution Function) values extracted from the
  .MTL file. Size `[mtl_names.size(), 17]`. Values are:
  0  | Base Color Red       | Range 0-1     | Default = 0.8
  1  | Base Color Green     | Range 0-1     | Default = 0.8
  2  | Base Color Blue      | Range 0-1     | Default = 0.8
  3  | Transparency (alpha) | Range 0-1     | Default = 1.0 (fully opaque)
  4  | Roughness            | Range 0-1     | Default = 0.5
  5  | Metallic             | Range 0-1     | Default = 0.0
  6  | Index of refraction (IOR)  | Range 0-4     | Default = 1.45
  7  | Specular Adjustment to the IOR | Range 0-1 | Default = 0.5 (no adjustment)
  8  | Emission Color Red    | Range 0-1     | Default = 0.0
  9  | Emission Color Green  | Range 0-1     | Default = 0.0
  10 | Emission Color Blue   | Range 0-1     | Default = 0.0
  11 | Sheen                 | Range 0-1     | Default = 0.0
  12 | Clearcoat             | Range 0-1     | Default = 0.0
  13 | Clearcoat roughness   | Range 0-1     | Default = 0.0
  14 | Anisotropic           | Range 0-1     | Default = 0.0
  15 | Anisotropic rotation  | Range 0-1     | Default = 0.0
  16 | Transmission          | Range 0-1     | Default = 0.0

- `std::string **materials_csv**` (optional input)
   Path to optional CSV file containing custom material properties. If empty, default ITU-R P.2040-3
   materials are used. CSV format: Header row with columns 'name', 'a', 'b', 'c', 'd', 'att' (order can vary).
   Each row defines a material with: name (string), electromagnetic parameters a,b,c,d (doubles),
   and additional attenuation att (dB). Relative permittivity: eta = a * f_GHz^b; Conductivity:
   sigma = c * f_GHz^d

### Returns:
- `arma::uword`
  Number of mesh triangles found in the file (`n_mesh`).

### Technical Notes:
- Unknown or missing materials default to `"vacuum"` (ε_r = 1, σ = 0).
- Materials are applied per triangle via the `usemtl` tag in the `.obj` file.
- Input geometry must be fully triangulated—quads and n-gons are not supported.
- File parsing is case-sensitive for material names.

### Material Tag Format:
- Default materials (ITU-R P.2040-3 Table 3): `"usemtl itu_concrete"`, `"itu_brick"`, `"itu_wood"`, `"itu_water"`, etc.
- Frequency range: 1–40 GHz (limited to 1–10 GHz for ground materials)
- Custom materials syntax: `"usemtl Name::A:B:C:D:att"` with `A, B`: Real permittivity ε_r = `A * fGHz^B`,
  `C, D`: Conductivity σ = `C * fGHz^D`, `att`: Penetration loss in dB (fixed, per interaction)

### Material properties:
Each material is defined by its electrical properties. Radio waves that interact with a building will
produce losses that depend on the electrical properties of the building materials, the material
structure and the frequency of the radio wave. The fundamental quantities of interest are the electrical
permittivity (ϵ) and the conductivity (σ). A simple regression model for the frequency dependence is
obtained by fitting measured values of the permittivity and the conductivity at a number of frequencies.
The five parameters returned in `mtl_prop` then are:

- Real part of relative permittivity at f = 1 GHz (a)
- Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
- Conductivity at f = 1 GHz (c)
- Frequency dependence of conductivity (d) such that σ = c· f^d
- Fixed attenuation in dB applied to each transition

A more detailed explanation together with a derivation can be found in ITU-R P.2040. The following
list of material is currently supported and the material can be selected by using the `usemtl` tag
in the OBJ file. When using Blender, the simply assign a material with that name to an object or face.
The following materials are defined by default:

Name                  |         a |        b  |         c |         d |       Att |  max fGHz |
----------------------|-----------|-----------|-----------|-----------|-----------|-----------|
vacuum / air          |       1.0 |       0.0 |       0.0 |       0.0 |       0.0 |       100 |
textiles              |       1.5 |       0.0 |      5e-5 |      0.62 |       0.0 |       100 |
plastic               |      2.44 |       0.0 |   2.33e-5 |       1.0 |       0.0 |       100 |
ceramic               |       6.5 |       0.0 |    0.0023 |      1.32 |       0.0 |       100 |
sea_water             |      80.0 |     -0.25 |       4.0 |      0.58 |       0.0 |       100 |
sea_ice               |       3.2 |    -0.022 |       1.1 |       1.5 |       0.0 |       100 |
water                 |      80.0 |     -0.18 |       0.6 |      1.52 |       0.0 |        20 |
water_ice             |      3.17 |    -0.005 |    5.6e-5 |       1.7 |       0.0 |        20 |
itu_concrete          |      5.24 |       0.0 |    0.0462 |    0.7822 |       0.0 |       100 |
itu_brick             |      3.91 |       0.0 |    0.0238 |      0.16 |       0.0 |        40 |
itu_plasterboard      |      2.73 |       0.0 |    0.0085 |    0.9395 |       0.0 |       100 |
itu_wood              |      1.99 |       0.0 |    0.0047 |    1.0718 |       0.0 |       100 |
itu_glass             |      6.31 |       0.0 |    0.0036 |    1.3394 |       0.0 |       100 |
itu_ceiling_board     |      1.48 |       0.0 |    0.0011 |     1.075 |       0.0 |       100 |
itu_chipboard         |      2.58 |       0.0 |    0.0217 |      0.78 |       0.0 |       100 |
itu_plywood           |      2.71 |       0.0 |      0.33 |       0.0 |       0.0 |        40 |
itu_marble            |     7.074 |       0.0 |    0.0055 |    0.9262 |       0.0 |        60 |
itu_floorboard        |      3.66 |       0.0 |    0.0044 |    1.3515 |       0.0 |       100 |
itu_metal             |       1.0 |       0.0 |     1.0e7 |       0.0 |       0.0 |       100 |
itu_very_dry_ground   |       3.0 |       0.0 |   0.00015 |      2.52 |       0.0 |        10 |
itu_medium_dry_ground |      15.0 |      -0.1 |     0.035 |      1.63 |       0.0 |        10 |
itu_wet_ground        |      30.0 |      -0.4 |      0.15 |       1.3 |       0.0 |        10 |
itu_vegetation        |       1.0 |       0.0 |    1.0e-4 |       1.1 |       0.0 |       100 |
irr_glass             |      6.27 |       0.0 |    0.0043 |    1.1925 |      23.0 |       100 |

### Example:
```
arma::mat mesh, mtl_prop, vert_list;
arma::umat face_ind;
arma::uvec obj_ind, mtl_ind;
std::vector<std::string> obj_names, mtl_names;

quadriga_lib::obj_file_read<double>("cube.obj", &mesh, &mtl_prop, &vert_list, &face_ind, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
```

---
## obj_overlap_test
Detect overlapping 3D objects in a triangular mesh

### Description:
- Tests whether any objects in a triangular mesh overlap by checking for shared volume or intersection.
- Touching faces or edges are not considered overlapping
- Returns the indices (1-based) of all objects that intersect with at least one other object.
- Can optionally output a list of overlap reasons for diagnostic purposes.
- Uses a configurable geometric tolerance to account for numerical precision.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::uvec quadriga_lib::obj_overlap_test(
                const arma::Mat<dtype> *mesh,
                const arma::uvec *obj_ind,
                std::vector[std::string](std::string) *reason = nullptr,
                dtype tolerance = 0.0005);
```

### Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)
  Triangular mesh geometry. Each row contains 3 vertices flattened as `[X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3]`. Size: `[n_mesh, 9]`.

- `const arma::uvec ***obj_ind**` (input)
  Object indices (1-based) that map multiple triangles in `mesh` to objects; Size: `[n_mesh]`;
  This is an output generated by [obj_file_read](#obj_file_read).

- `std::vector<std::string> ***reason** = nullptr` (optional output)
  Human-readable list of overlap reasons corresponding to each overlapping object. Length: `[n_overlap]`.

- `dtype **tolerance** = 0.0005` (optional input)
  Geometric tolerance (in meters) used to determine intersections. Default: `0.0005` (0.5 mm).

### Returns:
- `arma::uvec`: Vector of unique object indices (1-based) that were found to overlap, size `[n_overlap]`.

### Technical Notes:
- Overlap detection includes checks for: Intersecting triangle faces (shared volume), Vertices or edges penetrating another object’s bounding volume.
- The `tolerance` accounts for modeling inaccuracies and numerical instability—small overlaps below this threshold are ignored.
- This function does **not** modify the mesh or attempt to repair overlapping geometry — it only reports it.

### See also:
- [obj_file_read](#obj_file_read)

---
## path_to_tube
Convert a 3D path into a tube surface for visualization

### Description:
- Converts a sequence of 3D points (path) into a tubular surface using a ring of vertices around each path segment.
- Produces a quad-based mesh suitable for rendering in 3D tools such as Blender or MeshLab.
- Uses circular cross-sections with configurable radius and edge count.
- Internal calculations are performed in double precision to ensure numerical stability along complex paths.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
void quadriga_lib::path_to_tube(
                const arma::Mat<dtype> *path_coord,
                arma::Mat<dtype> *vert,
                arma::umat *faces,
                dtype radius = 1.0,
                arma::uword n_edges = 5);
```

### Arguments:
- `const arma::Mat<dtype> ***path_coord**` (input)
  Ordered list of 3D coordinates along the path, matrix of size `[3, n_coord]`.

- `arma::Mat<dtype> ***vert**` (output)
  Generated tube vertex positions. Size: `[3, n_coord × n_edges]`.

- `arma::umat ***faces**` (output)
  Quad face indices into `vert`. Each row contains 4 indices forming a quad. Size: `[4, (n_coord - 1) × n_edges]`.

- `dtype **radius** = 1.0` (optional input)
  Radius of the tube cross-section (in meters). Default: `1.0`.

- `arma::uword **n_edges** = 5` (optional input)
  Number of vertices used to approximate each circular cross-section. Must be `≥ 3`. Default: `5`.

### Technical Notes:
- The generated tube is centered around the path with perpendicular circular cross-sections.
- Orientation between path segments is handled with continuous frame alignment to reduce twisting.
- Quad faces are generated by connecting adjacent rings along the path.
- Output `faces` can be directly exported to formats like `.obj` or `.ply`.

---
## point_cloud_aabb
Compute the Axis-Aligned Bounding Boxes (AABB) of a 3D point cloud

### Description:
- Calculates the axis-aligned bounding box (AABB) for either a single point cloud or a set of sub-clouds.
- Each sub-cloud is defined by its starting row index in the input matrix.
- The result is a matrix where each row contains the minimum and maximum extents of a sub-cloud in the x, y, and z dimensions.
- For SIMD-friendly memory alignment, the result is zero-padded to the nearest multiple of `vec_size`.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(
                const arma::Mat<dtype> *points,
                const arma::u32_vec *sub_cloud_index = nullptr,
                arma::uword vec_size = 1);
```

### Arguments:
- `const arma::Mat<dtype> ***points**` (input)
  Matrix of 3D point coordinates. Size: `[n_points, 3]`.

- `const arma::u32_vec ***sub_cloud_index** = nullptr` (optional input)
  Vector of row indices indicating the start of each sub-cloud. Length: `[n_sub]`.
  If `nullptr`, the entire input is treated as a single cloud.

- `arma::uword **vec_size** = 1` (optional input)
  Vector size for SIMD alignment (e.g., 4, 8, or 16). The number of output rows is padded to a multiple of `vec_size`.
  Default: `1`.

### Returns:
- `arma::Mat<dtype>`
  Matrix of bounding boxes for each sub-cloud. Size: `[n_out, 6]`, where `n_out` is the padded number of sub-clouds.
  Each row has the format: `[x_min, x_max, y_min, y_max, z_min, z_max]`.

### Technical Notes:
- If `sub_cloud_index` is provided, the last index is assumed to span to the end of the `points` matrix.
- Padding rows (if any) are filled with zeros and should be ignored if `n_sub` is known externally.
- Suitable for preprocessing in geometry analysis, rendering pipelines, and spatial acceleration structures (e.g., BVH or octrees).
- Sub-clouds can be computed using [point_cloud_segmentation](#point_cloud_segmentation)

### See also:
- [point_cloud_segmentation](#point_cloud_segmentation)
- [point_cloud_split](#point_cloud_split)
- [ray_point_intersect](#ray_point_intersect)

---
## point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

### Description:
- Recursively partitions a 3D point cloud into smaller sub-clouds, each below a given size threshold.
- Sub-clouds are aligned to a specified SIMD vector size (e.g., for AVX or CUDA), with padding if necessary.
- Outputs (`pointsR`) a reorganized version of the input points that groups points by sub-cloud.
- Also produces forward and reverse index maps to track the reordering of points.
- Useful for optimizing spatial processing tasks such as bounding volume hierarchies or GPU batch execution.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
arma::uword quadriga_lib::point_cloud_segmentation(
                const arma::Mat<dtype> *points,
                arma::Mat<dtype> *pointsR,
                arma::u32_vec *sub_cloud_index,
                arma::uword target_size = 1024,
                arma::uword vec_size = 1,
                arma::u32_vec *forward_index = nullptr,
                arma::u32_vec *reverse_index = nullptr);
```

### Arguments:
- `const arma::Mat<dtype> ***points**` (input)
  Original 3D point cloud to be segmented. Size: `[n_points, 3]`.

- `arma::Mat<dtype> ***pointsR**` (output)
  Reorganized point cloud with points grouped by sub-cloud. Size: `[n_pointsR, 3]`.

- `arma::u32_vec ***sub_cloud_index**` (output)
  Vector of starting indices (0-based) for each sub-cloud within `pointsR`. Length: `[n_sub]`.

- `arma::uword **target_size** = 1024` (optional input)
  Maximum number of elements allowed per sub-cloud (before padding). Default: `1024`.

- `arma::uword **vec_size** = 1` (optional input)
  Vector alignment size for SIMD or CUDA. The number of points in each sub-cloud is padded to a multiple of this value. Default: `1`.

- `arma::u32_vec ***forward_index** = nullptr` (optional output)
  Index map from original `points` to reorganized `pointsR` (1-based). Size: `[n_pointsR]`. Padding indices are `0`.

- `arma::u32_vec ***reverse_index** = nullptr` (optional output)
  Index map from `pointsR` back to original `points` (0-based). Size: `[n_points]`.

### Returns:
- `arma::uword`
  Number of generated sub-clouds, `n_sub`.

### Technical Notes:
- Sub-clouds are formed using recursive spatial splitting (e.g., median-split along bounding box axes).
- Padding points are placed at the AABB center of the corresponding sub-cloud and can be ignored in processing.
- This function is typically used as a preprocessing step for GPU acceleration or bounding volume hierarchy (BVH) generation.
- If `vec_size = 1`, no padding is applied and all output maps contain valid indices only.

### See also:
- [point_cloud_aabb](#point_cloud_aabb)
- [point_cloud_split](#point_cloud_split)
- [ray_point_intersect](#ray_point_intersect)

---
## point_cloud_split
Split a point cloud into two sub-clouds along a spatial axis

### Description:
- Divides a 3D point cloud into two sub-clouds along the specified axis.
- Attempts to split the data at the median value to balance the number of points in each half.
- Returns the axis used for the split, or a negative value if a valid split was not possible (e.g., all points fall on one side).
- Output point clouds are written into `pointsA` and `pointsB`, and their size is adjusted accordingly.
- An optional indicator vector identifies the target sub-cloud (A or B) for each input point.
- Used in recursive spatial partitioning such as building BVH structures.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
int quadriga_lib::point_cloud_split(
                const arma::Mat<dtype> *points,
                arma::Mat<dtype> *pointsA,
                arma::Mat<dtype> *pointsB,
                int axis = 0,
                arma::Col<int> *split_ind = nullptr);
```

### Arguments:
- `const arma::Mat<dtype> ***points**` (input)
  Input point cloud. Size: `[n_points, 3]`.

- `arma::Mat<dtype> ***pointsA**` (output)
  First sub-cloud after split. Size: `[n_pointsA, 3]`.

- `arma::Mat<dtype> ***pointsB**` (output)
  Second sub-cloud after split. Size: `[n_pointsB, 3]`.

- `int **axis** = 0` (optional input)
  Axis to split along: `0` = longest extent (default), `1` = x-axis, `2` = y-axis, `3` = z-axis.

- `arma::Col<int> ***split_ind** = nullptr` (optional output)
  Vector of length `[n_points]`, where each element is: `1` if the point goes to `pointsA`, `2` if it goes to `pointsB`, `0` if error.

### Returns:
- `int` 
   Axis used for splitting: `1` = x, `2` = y, `3` = z,  or `-1`, `-2`, `-3` if the split failed (no points assigned to one of the outputs).

### Notes:
- The function does not modify `pointsA` or `pointsB` if the split fails.
- The selected axis is based on the bounding box if `axis == 0`
- This function is a building block for spatial acceleration structures (e.g., BVH, KD-trees), see [point_cloud_segmentation](#point_cloud_segmentation)

### See also:
- [point_cloud_aabb](#point_cloud_segmentation)
- [point_cloud_segmentation](#point_cloud_segmentation)
- [ray_point_intersect](#ray_point_intersect)

---
## point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

### Description:
- Uses raycasting to determine whether each 3D point lies inside a triangle mesh.
- Requires that the mesh is watertight and all normals are pointing outwards.
- For each point, multiple rays are cast in various directions.
- If any ray intersects a mesh element with a negative incidence angle, the point is classified as **inside**.
- Output can be binary (0 = outside, 1 = inside) or labeled with object indices.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::u32_vec quadriga_lib::point_inside_mesh(
                const arma::Mat<dtype> *points,
                const arma::Mat<dtype> *mesh,
                const arma::u32_vec *obj_ind = nullptr,
                dtype distance = 0.0);
```

### Arguments:
- `const arma::Mat<dtype> ***points**` (input)
  3D point coordinates to test, size `[n_points, 3]`.

- `const arma::Mat<dtype> ***mesh**` (input)
  Triangular mesh faces. Each row represents a triangle using 3 vertices in row-major format (x1,y1,z1,x2,y2,z2,x3,y3,z3), size `[n_mesh, 9]`.

- `const arma::u32_vec ***obj_ind** = nullptr` (optional input)
  Optional object index for each mesh element (1-based), size `[n_mesh]`. If provided, the return vector will contain the index of the enclosing object instead of binary values.

- `dtype **distance** = 0.0` (optional input)
  Optional distance in meters from objects that should be considered as *inside* the object.
  Possible range: 0 - 20 m. Using this parameter significantly increases computation time.

### Returns:
- `arma::u32_vec`, size `[n_points]`
  For each point: Returns `0` if the point is outside the mesh (or all objects), `1` if inside (or close to) any mesh object
  (if `obj_ind` not given), or returns the **1-based object index** if `obj_ind` is provided.

---
## ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

### Description:
- Computes interaction of plane waves with planar interfaces between homogeneous isotropic media.
- Supports beam-based modeling via triangular ray tubes (`trivec`, `tridir`).
- Face side determined by vertex order; front-side hit with FBS≠SBS → air-to-media; back-side hit
  with FBS≠SBS → media-to-air; FBS=SBS with opposing normals → media-to-media.
- Rays with `fbs_ind = 0` (no interaction) are omitted from output, so `n_rayN ≤ n_ray`.
- Output direction encoding (spherical/Cartesian) matches input `tridir` format.
- Overlapping mesh geometry must be avoided (materials are transparent to radio waves).
- Types 3–4 (scalar) use TE-only reflection with no total internal reflection, suitable for acoustic
  simulation with impedance-mapped material parameters (ε derived from Z).
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
void quadriga_lib::ray_mesh_interact(
    int interaction_type, dtype center_frequency,
    const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbs, const arma::Mat<dtype> *sbs,
    const arma::Mat<dtype> *mesh, const arma::Mat<dtype> *mtl_prop,
    const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
    const arma::Mat<dtype> *trivec = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    const arma::Col<dtype> *orig_length = nullptr,
    arma::Mat<dtype> *origN = nullptr, arma::Mat<dtype> *destN = nullptr,
    arma::Col<dtype> *gainN = nullptr, arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr, arma::Mat<dtype> *tridirN = nullptr,
    arma::Col<dtype> *orig_lengthN = nullptr,
    arma::Col<dtype> *fbs_angleN = nullptr,
    arma::Col<dtype> *thicknessN = nullptr,
    arma::Col<dtype> *edge_lengthN = nullptr,
    arma::Mat<dtype> *normal_vecN = nullptr,
    arma::s32_vec *out_typeN = nullptr);
```

### Input Arguments:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission
- **`center_frequency`** — Center frequency in Hz
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`fbs`**, **`sbs`** — First/second interaction points in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; `[n_mesh, 9]` (see `obj_file_read`)
- **`mtl_prop`** — Material properties per face; `[n_mesh, 5]` (see `obj_file_read`)
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`orig_length`** *(optional)* — Accumulated path length at origin; `[n_ray]`, default 0

### Output Arguments:
- **`origN`** — New origins after interaction (offset 0.001 m along travel direction); `[n_rayN, 3]`
- **`destN`** — New destinations accounting for direction change; `[n_rayN, 3]`
- **`gainN`** — Interaction gain (linear scale, includes in-medium attenuation, excludes FSPL); averaged over TE/TM polarizations for types 0–2, TE-only for types 3–4; `[n_rayN]`
- **`xprmatN`** — For types 0–2: polarization transfer matrix, interleaved complex `[ReVV ImVV ReVH ImVH ReHV ImHV ReHH ImHH]`; includes interaction gain, TE/TM coefficients, incidence plane orientation, in-medium attenuation (excludes FSPL); `[n_rayN, 8]`. For types 3–4 (scalar): `[Re Im 0 0 0 0 0 0]` where Re+jIm is the scalar pressure coefficient including in-medium attenuation; `[n_rayN, 8]`.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`orig_lengthN`** — Path length from `orig` to `origN`, added to input `orig_length` if given; `[n_rayN]`
- **`fbs_angleN`** — Incidence angle at FBS in rad; `[n_rayN]`
- **`thicknessN`** — Material thickness (FBS-to-SBS distance) in meters; `[n_rayN]`
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN, 3]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code; `[n_rayN]`

  | Code | Description                                          |
  |------|------------------------------------------------------|
  |    1 | Single hit, outside→inside                           |
  |    2 | Single hit, inside→outside                           |
  |    3 | Single hit, inside→outside, total reflection         |
  |    4 | Media-to-media, M2 hit first                         |
  |    5 | Media-to-media, M1 hit first                         |
  |    6 | Media-to-media, M1 hit first, total reflection       |
  |    7 | Overlapping faces, outside→inside                    |
  |    8 | Overlapping faces, inside→outside                    |
  |    9 | Overlapping faces, inside→outside, total reflection  |
  |   10 | Edge hit, outside→inside→outside                     |
  |   11 | Edge hit, inside→outside→inside                      |
  |   12 | Edge hit, inside→outside→inside, total reflection    |
  |   13 | Edge hit, outside→inside                             |
  |   14 | Edge hit, inside→outside                             |
  |   15 | Edge hit, inside→outside, total reflection           |

### See also:
- [obj_file_read](#obj_file_read) (for loading `mesh` and `mtl_prop` from OBJ file)
- [icosphere](#icosphere) (for generating beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for computing FBS and SBS positions)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)

---
## ray_point_intersect
Calculates the intersection of ray beams with points in three dimensions

### Description:
Unlike traditional ray tracing, where rays do not have a physical size, beam tracing models rays as
beams with volume. Beams are defined by triangles whose vertices diverge as the beam extends. This
approach is used to simulate a kind of divergence or spread in the beam, reminiscent of how radio
waves spreads as they travel from a point source. The volumetric nature of the beams allows for more
realistic modeling of energy distribution. As beams widen, the energy they carry can be distributed
across their cross-sectional area, affecting the intensity of the interaction with surfaces.
Unlike traditional ray tracing where intersections are line-to-geometry tests, beam tracing requires
volumetric intersection tests. 

- This function computes whether points in 3D Cartesian space are intersected by ray beams.
- A ray beam is defined by a ray origin and a triangular wavefront formed by three directional vectors.
- Returns a list of ray indices (0-based) that intersect with each point in the input point cloud.
- Supports three compute kernels: **GENERIC** (scalar), **AVX2** (SIMD, 8 points in parallel), and **CUDA** (GPU).
- The `use_kernel` parameter selects the kernel: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA.
- In auto mode (0), CUDA is selected only when `n_points >= 10000` and a CUDA-capable GPU is available;
  otherwise AVX2 is preferred if available, falling back to GENERIC.
- Optional support for pre-segmented point clouds (e.g., using [point_cloud_segmentation](#point_cloud_segmentation))
  to reduce computational cost.
- All internal computations use single precision for speed.
- Recommended to use with small tube radius and well-distributed points for optimal accuracy.

### Declaration:
```
template <typename dtype>
std::vector<arma::u32_vec> quadriga_lib::ray_point_intersect(
                const arma::Mat<dtype> *points,
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *trivec,
                const arma::Mat<dtype> *tridir,
                const arma::u32_vec *sub_cloud_index = nullptr,
                arma::u32_vec *hit_count = nullptr,
                int use_kernel = 0,
                int gpu_id = 0);
```

### Arguments:
- `const arma::Mat<dtype> ***points**` (input)
  3D coordinates of the point cloud. Size: `[n_points, 3]`.

- `const arma::Mat<dtype> ***orig**` (input)
  Ray origin positions in global coordinate system. Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***trivec**` (input)
  The 3 vectors pointing from the center point of the ray at the ray origin to the vertices of
  a triangular propagation tube (the beam), the values are in the order
  `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[ no_ray, 9 ]`

- `const arma::Mat<dtype> ***tridir**` (input)
  The directions of the vertex-rays. Size: `[ n_ray, 9 ]`, Values must be given in Cartesian
  coordinates in the order  `[ d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z  ]`; The vector does
  not need to be normalized.

- `const arma::u32_vec ***sub_cloud_index** = nullptr` (optional input)
  Index vector to mark boundaries between point cloud segments (e.g. for SIMD optimization). Length: `[n_sub]`.
  When using AVX2, sub-cloud start indices must be aligned to multiples of 8.

- `arma::u32_vec ***hit_count** = nullptr` (optional output)
  Output array with number of rays that intersected each point. Length: `[n_points]`.

- `int **use_kernel** = 0` (optional input)
  Selects the compute kernel: 0 = auto (default), 1 = GENERIC (scalar CPU), 2 = AVX2 (SIMD),
  3 = CUDA (GPU). An error is thrown if the requested kernel is not available at runtime.

- `int **gpu_id** = 0` (optional input)
  GPU device ID for CUDA kernel. Ignored when not using CUDA.

### Returns:
- `std::vector<arma::u32_vec>`
  List of ray indices that intersected each point (0-based). Each entry in the returned vector corresponds to one point.

### See also:
- [icosphere](#icosphere) (for generating beams)
- [point_cloud_segmentation](#point_cloud_segmentation) (for generating point cloud segments)
- [subdivide_rays](#subdivide_rays) (for subdivides ray beams into sub beams)
- [ray_triangle_intersect](#ray_triangle_intersect) (for calculating intersection of rays and triangles)
- [ray_mesh_interact](#ray_mesh_interact) (for calculating interactions of beams and a 3D model)

---
## ray_triangle_intersect
Calculates the intersection of rays and triangles in three dimensions

### Description:
- Implements the Möller–Trumbore algorithm to compute intersections between rays and triangles in 3D.
- Supports three compute kernels: **GENERIC** (scalar), **AVX2** (SIMD, 8 triangles in parallel), and **CUDA** (GPU).
- The `use_kernel` parameter selects the kernel: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA.
- In auto mode (0), CUDA is selected only when `n_ray >= 10000` and a CUDA-capable GPU is available;
  otherwise AVX2 is preferred if available, falling back to GENERIC.
- Can detect first and second intersections (FBS/SBS), number of intersections, and intersection indices.
- Allowed datatypes (`dtype`): `float` or `double`
- Internal computations are carried out in **single precision** regardless of `dtype`.

### Declaration:
```
void quadriga_lib::ray_triangle_intersect(
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *dest,
                const arma::Mat<dtype> *mesh,
                arma::Mat<dtype> *fbs = nullptr,
                arma::Mat<dtype> *sbs = nullptr,
                arma::u32_vec *no_interact = nullptr,
                arma::u32_vec *fbs_ind = nullptr,
                arma::u32_vec *sbs_ind = nullptr,
                const arma::u32_vec *sub_mesh_index = nullptr,
                const arma::Mat<dtype> *aabb = nullptr,
                int use_kernel = 0,
                int gpu_id = 0);
```

### Arguments:
- `const arma::Mat<dtype> ***orig**` (input)
  Ray origins in global coordinate system (GCS). Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***dest**` (input)
  Ray destinations in GCS. Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***mesh**` (input)
  Triangular surface mesh. Size: `[n_mesh, 9]`, where each row contains the 3 vertices
  `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`.

- `arma::Mat<dtype> ***fbs**` (optional output)
  First-bounce surface intersection points (FBS). Size: `[n_ray, 3]`.

- `arma::Mat<dtype> ***sbs**` (optional output)
  Second-bounce surface intersection points (SBS). Size: `[n_ray, 3]`.

- `arma::u32_vec ***no_interact**` (optional output)
  Number of intersections per ray (0, 1, or 2). Size: `[n_ray]`.

- `arma::u32_vec ***fbs_ind**` (optional output)
  1-based index of the first intersected mesh element, 0 = no intersection. Size: `[n_ray]`.

- `arma::u32_vec ***sbs_ind**` (optional output)
  1-based index of the second intersected mesh element, 0 = no second intersection. Size: `[n_ray]`.

- `const arma::u32_vec ***sub_mesh_index**` (optional input)
  Indexes indicating start of sub-meshes in `mesh`. Size: `[n_sub]`. Enables faster processing via segmentation.

- `const arma::Mat<dtype> ***aabb**` (optional input)
  Pre-computed axis-aligned bounding boxes per sub-mesh. Size: `[n_sub, 6]`, where each row contains
  `{x_min, x_max, y_min, y_max, z_min, z_max}`. If `nullptr`, AABBs are computed internally from `mesh`.

- `bool **transpose_inputs** = false` (optional input)
  If `true`, treats `orig`/`dest` as `[3, n_ray]` and `mesh` as `[9, n_mesh]`.

- `int **use_kernel** = 0` (optional input)
  Selects the compute kernel: 0 = auto (default), 1 = GENERIC (scalar CPU), 2 = AVX2 (SIMD),
  3 = CUDA (GPU). An error is thrown if the requested kernel is not available at runtime.

- `int **gpu_id** = 0` (optional input)
  GPU device ID for CUDA kernel. Ignored when not using CUDA.

### See also:
- [obj_file_read](#obj_file_read) (for loading `mesh` from an OBJ file)
- [icosphere](#icosphere) (for generating beams)
- [triangle_mesh_segmentation](#triangle_mesh_segmentation) (for calculating sub-meshes)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)
- [subdivide_rays](#subdivide_rays) (for subdivides ray beams into sub beams)

---
## subdivide_rays
Subdivide ray beams into four smaller sub-beams

### Description:
- Subdivides each ray beam (defined by a triangular wavefront) into four new beams with adjusted origin, shape, and direction.
- Supports input in Spherical or Cartesian direction format.
- When `dest` is not provided, the corresponding output `destN` is omitted.
- Useful for hierarchical ray tracing or angular resolution refinement.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::uword quadriga_lib::subdivide_rays(
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *trivec,
                const arma::Mat<dtype> *tridir,
                const arma::Mat<dtype> *dest = nullptr,
                arma::Mat<dtype> *origN = nullptr,
                arma::Mat<dtype> *trivecN = nullptr,
                arma::Mat<dtype> *tridirN = nullptr,
                arma::Mat<dtype> *destN = nullptr,
                const arma::u32_vec *index = nullptr,
                const double ray_offset = 0.0);
```

### Arguments:
- `const arma::Mat<dtype> ***orig**` (input)
  Ray origin points in global coordinate system (GCS).
  Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***trivec**` (input)
  Vectors pointing from the ray origin to the three triangle vertices.
  Size: `[n_ray, 9]`, order: `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`.

- `const arma::Mat<dtype> ***tridir**` (input)
  Directions of the three vertex-rays.
  Format can be Spherical `[n_ray, 6]` as `[v1az v1el v2az v2el v3az v3el]`,
  or Cartesian `[n_ray, 9]` as `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`.

- `const arma::Mat<dtype> ***dest** = nullptr` (input)
  Ray destination points. If `nullptr`, the output `destN` will remain empty.
  Size: `[n_ray, 3]`.

- `arma::Mat<dtype> ***origN**` (output)
  New ray origins after subdivision.
  Size: `[n_rayN, 3]`.

- `arma::Mat<dtype> ***trivecN**` (output)
  Updated vectors for each subdivided triangle beam.
  Size: `[n_rayN, 9]`.

- `arma::Mat<dtype> ***tridirN**` (output)
  New directions of the subdivided vertex-rays, in the same format as input.
  Size: `[n_rayN, 6]` (spherical) or `[n_rayN, 9]` (Cartesian).

- `arma::Mat<dtype> ***destN**` (output)
  Updated destination points.
  Size: `[n_rayN, 3]`, empty if input `dest` was `nullptr`.

- `const arma::u32_vec ***index**` (optional input)
  List of ray indices to be subdivided (0-based). Only the specified rays are subdivided.
  Size: `[n_ind]`.

- `const double **ray_offset** = 0.0` (optional input)
  Offset (in meters) applied to the origin of each subdivided ray along its propagation direction.
  Default: `0.0`.

### Returns:
- `arma::uword  **n_rayN**`
  Number of output rays, typically `4 × n_ray` or `4 × n_ind` if `index` is provided.

### See also:
- [icosphere](#icosphere) (for generating beams)
- [ray_point_intersect](#ray_point_intersect) (for calculating beam interactions with sampling points)
- [ray_triangle_intersect](#ray_triangle_intersect) (for calculating beam interactions with triangles)

---
## subdivide_triangles
Subdivide triangles into smaller triangles

### Description:
- Uniformly subdivides each input triangle into `n_div × n_div` smaller triangles.
- Increases spatial resolution for mesh-based processing (e.g., ray tracing or visualization).
- Optional input/output material properties are duplicated across subdivided triangles.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::uword quadriga_lib::subdivide_triangles(
                arma::uword n_div,
                const arma::Mat<dtype> *triangles_in,
                arma::Mat<dtype> *triangles_out,
                const arma::Mat<dtype> *mtl_prop = nullptr,
                arma::Mat<dtype> *mtl_prop_out = nullptr);
```

### Arguments:
- `arma::uword **n_div**` (input)
  Number of subdivisions per triangle edge;
  total output triangles: `n_triangles_out = n_triangles_in × n_div × n_div`.

- `const arma::Mat<dtype> ***triangles_in**` (input)
  Vertices of the triangular mesh in global Cartesian coordinates; each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[n_triangles_in, 9]`

- `arma::Mat<dtype> ***triangles_out**` (output)
  Vertices of the sub-divided mesh in global Cartesian coordinates; Size: `[n_triangles_out, 9]`

- `const arma::Mat<dtype> ***mtl_prop** = nullptr` (optional input)
  Material properties associated for the input triangles; Size: `[n_triangles_in, 5]`.

- `arma::Mat<dtype> ***mtl_prop_out** = nullptr` (optional output)
  Material properties for the subdivided triangles, copied from the parent triangle,
  Size: `[n_triangles_out, 5]`.

### Returns:
- `arma::uword **n_triangles_out**`
  Number of generated triangles (equals `n_triangles_in × n_div × n_div`).

---
## triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

### Description:
The axis-aligned minimum bounding box (or AABB) for a given set of triangles is its minimum
bounding box subject to the constraint that the edges of the box are parallel to the (Cartesian)
coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of
triangles. In order to find intersections with the triangles (e.g. using ray tracing), the
initial check is the intersections between the rays and the AABBs. Since it is usually a much
less expensive operation than the check of the actual intersection (because it only requires
comparisons of coordinates), it allows quickly excluding checks of the pairs that are far apart.

- This function computes the axis-aligned bounding box for each sub-mesh in a 3D triangle mesh.
- Each triangle is defined by three vertices in a flat row: `[x1, y1, z1, x2, y2, z2, x3, y3, z3]`.
- Sub-meshes are defined by the `sub_mesh_index` list, indicating the starting row of each sub-mesh.
- The resulting bounding boxes are returned as a matrix of shape `[n_sub, 6]` with columns: `[x_min, x_max, y_min, y_max, z_min, z_max]`.
- If `vec_size > 1`, the result is padded such that the number of rows in the output is a multiple of `vec_size`.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(
                const arma::Mat<dtype> *mesh,
                const arma::u32_vec *sub_mesh_index = nullptr,
                arma::uword vec_size = 1);
```

### Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)
  Vertices of the triangle mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[ n_triangles, 9 ]`

- `const arma::u32_vec ***sub_mesh_index** = nullptr` (optional input)
  Start indices of the sub-meshes in 0-based notation. If this parameter is not given, the AABB of
  the entire triangle mesh is returned; Length `[n_sub]`

- `arma::uword **vec_size** = 1` (optional input)
  Alignment size for SIMD processing (e.g., `8` for AVX2, `32` for CUDA). 
  Output is padded to a multiple of this value.

### Returns:
- `arma::Mat<dtype>` 
  A matrix of shape `[n_sub_aligned, 6]`, where each row is `[x_min, x_max, y_min, y_max, z_min, z_max]`.

---
## triangle_mesh_segmentation
Reorganize a 3D mesh into smaller sub-meshes for faster processing

### Description:
This function processes the elements of a large triangle mesh by clustering those that are
closely spaced. The resulting mesh retains the same elements but rearranges their order.
The function aims to minimize the size of the axis-aligned bounding box around each cluster,
referred to as a sub-mesh, while striving to maintain a specific number of elements within
each cluster.

- Subdivision is recursive and based on bounding box partitioning until each sub-mesh contains no more than `target_size` triangles.
- Sub-meshes are aligned to `vec_size` for SIMD or GPU optimization; padded with dummy triangles at the center of each sub-mesh if needed.
- If material properties are provided, these are also reorganized and padded accordingly.
- The function returns the number of created sub-meshes, and reorders the triangles and materials.
- Allowed datatypes (`dtype`): `float` or `double`.

### Declaration:
```
arma::uword quadriga_lib::triangle_mesh_segmentation(
                const arma::Mat<dtype> *mesh,
                arma::Mat<dtype> *meshR,
                arma::u32_vec *sub_mesh_index,
                arma::uword target_size = 1024,
                arma::uword vec_size = 1,
                const arma::Mat<dtype> *mtl_prop = nullptr,
                arma::Mat<dtype> *mtl_propR = nullptr,
                arma::u32_vec *mesh_index = nullptr);
```

### Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[n_mesh, 9]`

- `arma::Mat<dtype> ***meshR**` (output)
  Vertices of the clustered mesh in global Cartesian coordinates; Size: `[n_meshR, 9]`

- `arma::u32_vec ***sub_mesh_index**` (output)
  Start indices of the sub-meshes in 0-based notation; Vector of length `[n_sub]`

- `arma::uword **target_size** = 1024` (input)
  The target number of elements of each sub-mesh. Default value = 1024. For best performance, the
  value should be around `sgrt(n_mesh)`

- `arma::uword **vec_size** = 1` (input)
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1.
  For values > 1,the number of rows for each sub-mesh in the output is increased to a multiple
  of `vec_size`. For padding, zero-sized triangles are placed at the center of the AABB of
  the corresponding sub-mesh.

- `const arma::Mat<dtype> ***mtl_prop** = nullptr` (optional input)
  Material properties corresponding to the original mesh; Size: `[n_mesh, 5]`; See [obj_file_read](#obj_file_read)

- `arma::Mat<dtype> ***mtl_propR** = nullptr` (optional output)
  Reorganized material properties, aligned and padded if necessary, Size: `[n_meshR, 5]`

- `arma::u32_vec ***mesh_index** = nullptr` (optional output)
  1-based mapping from the original mesh to the reorganized mesh; Size: `[n_meshR]` (0 = padding)

### Returns:
- `arma::uword **n_sub**`
  The number of created sub-meshes. Output matrices are resized accordingly.

---
## triangle_mesh_split
Split a 3D mesh into two sub-meshes along a given axis

### Description:
- Divides a triangular mesh into two sub-meshes along a selected axis (or automatically the longest).
- The function chooses a split point based on the bounding box center of the selected axis.
- Returns the axis used for the split: `1 = x`, `2 = y`, `3 = z`; or negative values if the split failed.
- An optional indicator vector identifies the target sub-mesh (A or B) for each input point.
- On failure (i.e., all triangles fall into one side), outputs `meshA` and `meshB` remain unchanged.
- Allowed datatypes (`dtype`): `float` or `double`

### Declaration:
```
int quadriga_lib::triangle_mesh_split(
                const arma::Mat<dtype> *mesh,
                arma::Mat<dtype> *meshA,
                arma::Mat<dtype> *meshB,
                int axis = 0,
                arma::Col<int> *split_ind = nullptr);
```

### Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)
  Triangle mesh input; each row contains one triangle as `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`
  Size: `[n_mesh, 9]`

- `arma::Mat<dtype> ***meshA**` (output)
  First resulting sub-mesh; triangles with centroid below split threshold. Size: `[n_meshA, 9]`

- `arma::Mat<dtype> ***meshB**` (output)
  Second resulting sub-mesh; triangles with centroid above split threshold. Size: `[n_meshB, 9]`

- `int **axis** = 0` (optional input)
  Axis to split along: `0` = longest extent (default), `1` = x-axis, `2` = y-axis, `3` = z-axis.

- `arma::Col<int> ***split_ind** = nullptr` (optional output)
  Output vector indicating assignment of each triangle: `1` = meshA, `2` = meshB, `0` = not assigned (on failure)
  Length: `[n_mesh]`

### Returns:
- `int` 
  The axis used for the split (`1`, `2`, or `3`), or negative value on failure (`-1`, `-2`, or `-3`).

