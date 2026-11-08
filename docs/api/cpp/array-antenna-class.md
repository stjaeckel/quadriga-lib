# Array antenna class

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-array-antenna-class

| Function | Summary |
| --- | --- |
| [arrayant](#arrayant) | Class for storing and manipulating array antenna models |
| [.append](#.append) | Append elements of another arrayant to the current one |
| [.calc_beamwidth_deg](#.calc_beamwidth_deg) | Calculate the beamwidth and pointing angles of array antenna elements in degrees |
| [.calc_directivity_dBi](#.calc_directivity_dBi) | Calculate the directivity in dBi of a single array element |
| [.combine_pattern](#.combine_pattern) | Combine element patterns, positions, and coupling weights into effective radiation patterns |
| [.copy_element](#.copy_element) | Copy a single antenna element to one or more destination slots |
| [.export_obj_file](#.export_obj_file) | Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization |
| [.interpolate](#.interpolate) | Interpolate polarimetric antenna field patterns for given azimuth/elevation angles |
| [.is_valid](#.is_valid) | Validate the integrity of an arrayant object |
| [.qdant_write](#.qdant_write) | Write arrayant data to a QDANT (XML) file |
| [.remove_zeros](#.remove_zeros) | Remove zero-valued entries from antenna pattern data, reducing its size |
| [.rotate_pattern](#.rotate_pattern) | Rotate antenna radiation patterns around the principal axes using Euler rotations |
| [.set_size](#.set_size) | Resize an arrayant object to new dimensions |

---
<a id="arrayant"></a>
## arrayant
Class for storing and manipulating array antenna models

Not in: MATLAB, Python

- Represents a multi-element antenna array; each element has a position relative to the array phase-center
- Elements may be inter-coupled via a complex coupling matrix
- Field pattern cubes `e_theta_re/im`, `e_phi_re/im` must all be `[n_elevation, n_azimuth, n_elements]`
- `element_pos` is optional (empty = all elements at origin); `coupling_re/im` are optional (empty = identity)

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
| `arma::Mat<dtype> coupling_re`    | `[n_elements, n_ports]`                | Coupling matrix, real part                                     |
| `arma::Mat<dtype> coupling_im`    | `[n_elements, n_ports]`                | Coupling matrix, imaginary part                                |
| `dtype center_frequency`          | scalar                                 | Center frequency                                               |
| `std::string name`                | string                                 | Name of the array antenna object                               |

### Simple member functions:
| Function         | Description                                       |
| ---------------- | ------------------------------------------------- |
| `.n_elevation()` | Number of elevation angles                        |
| `.n_azimuth()`   | Number of azimuth angles                          |
| `.n_elements()`  | Number of antenna elements                        |
| `.n_ports()`     | Number of ports (columns of coupling matrix)      |
| `.copy()`        | Returns a deep copy of the arrayant object        |
| `.reset()`       | Clears all data, resetting size to zero           |
| `.is_valid()`    | Returns `""` if valid, or an error message string |

### Complex member functions:
| Function                  | Description                                                       |
| ------------------------- | ----------------------------------------------------------------- |
| [.append](#.append)               | Append elements of another arrayant                               |
| [.calc_beamwidth_deg](#.calc_beamwidth_deg)   | Calculate the beam width of an antenna element in degree          |
| [.calc_directivity_dBi](#.calc_directivity_dBi) | Calculate the directivity in dBi of a single array element        |
| [.combine_pattern](#.combine_pattern)      | Compute effective patterns from elements, positions, and coupling |
| [.copy_element](#.copy_element)         | Copy a single element to one or more destination slots            |
| [.export_obj_file](#.export_obj_file)      | Export pattern geometry to Wavefront OBJ                          |
| [.interpolate](#.interpolate)          | Interpolate field patterns at given azimuth/elevation angles      |
| [.qdant_write](#.qdant_write)          | Write arrayant to QDANT file                                      |
| [.remove_zeros](#.remove_zeros)         | Remove zero-valued entries from pattern data                      |
| [.rotate_pattern](#.rotate_pattern)       | Rotate pattern and/or polarization via Euler angles               |
| [.set_size](#.set_size)             | Resize the arrayant to new dimensions                             |
| [.is_valid](#.is_valid)             | Validate arrayant integrity                                       |

---
<a id=".append"></a>
## .append
Append elements of another arrayant to the current one

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_concat) · [Python](../python/array-antenna-functions.md#concat)

- Both arrays must share identical sampling grids; throws otherwise
- Coupling is block-diagonal (see [arrayant_concat_multi](array-antenna-functions.md#arrayant_concat_multi) diagram); center_frequency is taken from this
- An empty `coupling_re` counts as identity (one port per element), an empty `coupling_im` as zeros

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::append(
    const arrayant<dtype> *new_arrayant) const;
```

### Inputs:
- **`new_arrayant`** — Array whose elements are appended; sampling grid must match

### Returns:
- New `arrayant` containing all elements from both arrays

### See also:
- [arrayant_concat_multi](array-antenna-functions.md#arrayant_concat_multi) (multi-freq counterpart)

---
<a id=".calc_beamwidth_deg"></a>
## .calc_beamwidth_deg
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
- Ignores element coupling

### Declaration:
```cpp
void calc_beamwidth_deg(arma::uword i_element,
    dtype threshold_dB = 3.0,
    dtype *beamwidth_az = nullptr,
    dtype *beamwidth_el = nullptr,
    dtype *az_point_ang = nullptr,
    dtype *el_point_ang = nullptr) const;
```

### Inputs:
- **`i_element`** — Element index; 0-based
- **`threshold_dB`** — Threshold in dB; 3 dB = FWHM

### Outputs:
- **`beamwidth_az`** — Azimuth beamwidth in degree
- **`beamwidth_el`** — Elevation beamwidth in degree
- **`az_point_ang`** — Azimuth pointing angle for the main beam in degree
- **`el_point_ang`** — Elevation pointing angle for the main beam in degree

### See also:
- [.calc_directivity_dBi](#.calc_directivity_dBi) (directivity in dBi of a single array element)

---
<a id=".calc_directivity_dBi"></a>
## .calc_directivity_dBi
Calculate the directivity in dBi of a single array element

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_calc_directivity) · [Python](../python/array-antenna-functions.md#calc_directivity)

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Independent of the scale of the pattern; an element whose pattern is zero everywhere returns 0 dBi
- Ignores element coupling

### Declaration:
```cpp
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const;
```

### Inputs:
- **`i_element`** — Element index, 0-based

### Returns:
- Directivity of the specified element in dBi

### See also:
- [.combine_pattern](#.combine_pattern) (the per-port directivity is a typical follow-up)

---
<a id=".combine_pattern"></a>
## .combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_combine_pattern) · [Python](../python/array-antenna-functions.md#combine_pattern)

- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` to produce one output element per port (column) of the coupling matrix
- The element positions add the phase `-2 pi f d / propagation_speed` with `f = center_frequency` and `d` the distance
  of the element from the wavefront plane through the origin; the default speed of light applies to radio waves, use
  about 343 m/s for acoustic arrays (e.g. from [generate_speaker](array-antenna-functions.md#generate_speaker))
- Useful for beamforming and MIMO channel computation speedup

### Declaration:
```cpp
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::combine_pattern(
    const arma::Col<dtype> *azimuth_grid_new = nullptr,
    const arma::Col<dtype> *elevation_grid_new = nullptr,
    dtype propagation_speed = 299792458.0) const;
```

### Inputs:
- **`azimuth_grid_new`** *(optional)* — Alternative azimuth grid in rad, in [-pi, pi], strictly increasing; defaults to input grid
- **`elevation_grid_new`** *(optional)* — Alternative elevation grid in rad, in [-pi/2, pi/2], strictly increasing; defaults to input grid
- **`propagation_speed`** *(optional)* — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics

### Returns:
- New `arrayant` with `n_ports` elements (= number of columns in `coupling_re/im`), each holding the combined effective pattern for that port

### See also:
- [.interpolate](#.interpolate) (used internally to compute effective radiation patterns)
- [.rotate_pattern](#.rotate_pattern) (useful for orienting array antenna patterns)

---
<a id=".copy_element"></a>
## .copy_element
Copy a single antenna element to one or more destination slots

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_copy_element) · [Python](../python/array-antenna-functions.md#copy_element)

- Array is resized if any destination index exceeds the current number of elements
- Coupling matrix for added elements is set to identity; if not existing, it gets initialized to identity
- Throws `std::invalid_argument` for a read-only object, an empty `destination`, or a destination index equal to the largest `arma::uword` (e.g. `-1` cast to unsigned)

### Declaration:
```cpp
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uword destination);
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uvec destination);
```

### Inputs:
- **`source`** — Index of the element to copy, 0-based
- **`destination`** — Target index or indices, 0-based; array resizes to fit the maximum index

### See also:
- [arrayant_copy_element_multi](array-antenna-functions.md#arrayant_copy_element_multi) (multi-freq counterpart)

---
<a id=".export_obj_file"></a>
## .export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_export_obj_file) · [Python](../python/array-antenna-functions.md#export_obj_file)

- Pattern is mapped onto an icosphere; higher `icosphere_n_div` gives finer mesh

### Declaration:
```cpp
void quadriga_lib::arrayant<dtype>::export_obj_file(
    std::string fn,
    dtype directivity_range = 30.0,
    std::string colormap = "jet",
    dtype object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    arma::uvec i_element = {}) const;
```

### Inputs:
- **`fn`** — Output OBJ filename; must not be empty; filename must end in .obj
- **`directivity_range`** *(optional)* — Dynamic range of the visualized directivity pattern in dB
- **`colormap`** *(optional)* — Colormap name; see [colormap](site-specific-simulation-tools.md#colormap) for supported options
- **`object_radius`** *(optional)* — Radius of the exported object
- **`icosphere_n_div`** *(optional)* — Icosphere subdivision count; higher = finer mesh, see [icosphere](site-specific-simulation-tools.md#icosphere)
- **`i_element`** *(optional)* — 0-based element indices to export; `{}` exports all elements

### See also:
- [colormap](site-specific-simulation-tools.md#colormap) (Used for setting the colormap)
- [icosphere](site-specific-simulation-tools.md#icosphere) (Used internally to generate icosphere primitive)
- [.write_paths_to_obj_file](channel-class.md#.write_paths_to_obj_file) (function of Channel class to export propagation paths to OBJ 3D visualization)

---
<a id=".interpolate"></a>
## .interpolate
Interpolate polarimetric antenna field patterns for given azimuth/elevation angles

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_interpolate) · [Python](../python/array-antenna-functions.md#interpolate)

- Outputs complex e-theta (V) and e-phi (H) field components at requested angles
- `n_out` equals `n_elements` when `i_element` is omitted; equals `len(i_element)` otherwise
- Azimuth input supports planar wave mode (`[1, n_ang]`) or per-element spherical wave mode (`[n_out, n_ang]`)
- NaN or infinite angles give NaN in all outputs for that angle (and element); no exception is thrown
- Output matrices are resized automatically if dimensions do not match; this invalidates existing data pointers

### Declaration:
```cpp
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

### Inputs:
- **`azimuth`** — Azimuth angles in rad, in [-pi, pi]; `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in rad, in [-pi/2, pi/2]; `[1, n_ang]` or `[n_out, n_ang]`
- **`i_element`** *(optional)* — Element indices (0-based) to interpolate; duplicates allowed; defaults to all elements; `[n_out]` or `{}`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) in rad; `nullptr`; `[3, 1]`; `[3, n_out]`; `[3, 1, n_ang]`, or `[3, n_out, n_ang]`
- **`element_pos_i`** *(optional)* — Override element positions in m; `nullptr` uses `arrayant.element_pos`; `[3, n_out]`

### Outputs:
- **`V_re`** / **`V_im`** — Real/imaginary e-theta (vertical) field component; `[n_out, n_ang]`
- **`H_re`** / **`H_im`** — Real/imaginary e-phi (horizontal) field component; `[n_out, n_ang]`
- **`dist`** *(optional)* — Distance from the wavefront plane (normal to the incident ray direction) to each element; `nullptr` or `[n_out, n_ang]`
- **`azimuth_loc`** *(optional)* — Azimuth angles in local (rotated) element frame in rad; `nullptr` or `[n_out, n_ang]`
- **`elevation_loc`** *(optional)* — Elevation angles in local element frame in rad; `nullptr` or `[n_out, n_ang]`
- **`gamma`** *(optional)* — Polarization rotation angles in rad; `nullptr` or `[n_out, n_ang]`

### Example:
```cpp
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
arma::mat azimuth = {0.0, 0.5*pi, -0.5*pi, pi};
arma::mat elevation(1, azimuth.n_elem);  // zeros
arma::mat V_re, V_im, H_re, H_im;
ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);
```

### See also:
- [arrayant_interpolate_multi](array-antenna-functions.md#arrayant_interpolate_multi) (multi-freq counterpart)

---
<a id=".is_valid"></a>
## .is_valid
Validate the integrity of an arrayant object

Not in: MATLAB, Python

### Declaration:
```cpp
std::string quadriga_lib::arrayant<dtype>::is_valid(bool quick_check = true) const;
```

### Inputs:
- **`quick_check`** *(optional)* — `true` for fast structural check; `false` for full data validation; full check additionally verifies data values

### Returns:
- Empty string if valid; error message string if invalid

### See also:
- [arrayant_is_valid_multi](array-antenna-functions.md#arrayant_is_valid_multi) (multi-freq counterpart)

---
<a id=".qdant_write"></a>
## .qdant_write
Write arrayant data to a QDANT (XML) file

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_qdant_write) · [Python](../python/array-antenna-functions.md#qdant_write)

- Multiple antennas can be stored in the same file using distinct `id` values
- If `id = 0` and the file exists, the antenna is appended with `id = max_existing_id + 1`; if file does not exist, `id = 1` gets assigned
- Without `layout` (empty), the layout in the file is replaced by a `[1, n]` row of all IDs in the file in file
  order; a custom layout written before is lost. Pass the layout again with every write that should keep it.
- The file has single-precision-equivalent accuracy relative to the element peak, also for double-precision data:
  pattern magnitudes (in dB) and phases (in degree) get the fewest decimals (0 to 6) that keep the error of each
  complex sample at or below 6e-8 times the peak amplitude of its element (over e_theta and e_phi). Samples near
  the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the
  peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at
  a low level is quantized relative to its own peak. Element positions and coupling (magnitude, phase) are written
  as the shortest text that reads back as the same float (at most 9 significant digits); sampling grids are rounded
  to 0.001 degree
- All non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element that is
  zero in both field components is written as an all `-inf` `EthetaMag` node.

### Declaration:
```cpp
unsigned quadriga_lib::arrayant<dtype>::qdant_write(
    std::string fn,
    unsigned id = 0,
    arma::u32_mat layout = {}) const;
```

### Inputs:
- **`fn`** — Output QDANT filename; must not be empty
- **`id`** *(optional)* — Target ID in file; `0` appends with auto-assigned ID
- **`layout`** *(optional)* — Matrix organizing multiple antenna IDs within the file; must reference only IDs present in the file

### Returns:
- ID assigned to the written antenna within the file

### See also:
- [qdant_read](array-antenna-functions.md#qdant_read) (read back QDANT files)
- [qdant_write_multi](array-antenna-functions.md#qdant_write_multi) (multi-freq counterpart)

---
<a id=".remove_zeros"></a>
## .remove_zeros
Remove zero-valued entries from antenna pattern data, reducing its size

Not in: MATLAB, Python

- Modifies in-place when `output = nullptr`; otherwise writes to `*output`
- A column or row is removed if the total power summed over all field components and array elements does not exceed
  1e-12 times the peak power of a single sample (summed over field components and elements); the result does not
  depend on the scale of the pattern; an all-zero pattern is left unchanged
- Immediate neighbors of retained entries are also kept unless the local grid spacing is strongly asymmetric (factor > 1.5)

### Declaration:
```cpp
void quadriga_lib::arrayant<dtype>::remove_zeros(arrayant<dtype> *output = nullptr);
```

### Inputs:
- **`output`** *(optional)* — Target arrayant to write result to; `nullptr` modifies in-place

---
<a id=".rotate_pattern"></a>
## .rotate_pattern
Rotate antenna radiation patterns around the principal axes using Euler rotations

Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_rotate_pattern) · [Python](../python/array-antenna-functions.md#rotate_pattern)

- Rotates pattern and/or polarization around x (bank), y (tilt), z (heading) axes in degrees
- Modifies in-place when `output = nullptr`; otherwise writes to `*output`
- Coupling, `center_frequency` and `name` are kept (also in `*output`); when a single `element` is written to
  `*output`, the output holds only that element and its coupling is identity

### Declaration:
```cpp
void quadriga_lib::arrayant<dtype>::rotate_pattern(
    dtype x_deg = 0.0,
    dtype y_deg = 0.0,
    dtype z_deg = 0.0,
    unsigned usage = 0,
    unsigned element = -1,
    arrayant<dtype> *output = nullptr);
```

### Inputs:
- **`x_deg`** *(optional)* — Rotation around x-axis (bank) in degrees
- **`y_deg`** *(optional)* — Rotation around y-axis (tilt) in degrees
- **`z_deg`** *(optional)* — Rotation around z-axis (heading) in degrees
- **`usage`** *(optional)* — Rotation mode:
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`element`** *(optional)* — 0-based element index to rotate; `-1` rotates all elements (implemented as wrap-around to UINT_MAX)
- **`output`** *(optional)* — Target arrayant; `nullptr` modifies in-place

### See also:
- [arrayant_rotate_pattern_multi](array-antenna-functions.md#arrayant_rotate_pattern_multi) (multi-freq counterpart)

---
<a id=".set_size"></a>
## .set_size
Resize an arrayant object to new dimensions

Not in: MATLAB, Python

- No-op if current dimensions already match
- After resize: element_pos is zeroed, coupling_re set to identity, coupling_im zeroed; all other field data is undefined

### Declaration:
```cpp
void quadriga_lib::arrayant<dtype>::set_size(
    arma::uword n_elevation,
    arma::uword n_azimuth,
    arma::uword n_elements,
    arma::uword n_ports);
```

### Inputs:
- **`n_elevation`** — Number of elevation samples
- **`n_azimuth`** — Number of azimuth samples
- **`n_elements`** — Number of antenna elements
- **`n_ports`** — Number of ports (columns of coupling matrix)
