# Array antenna class

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-array-antenna-class

| Function | Summary |
| --- | --- |
| [arrayant](#arrayant) | Class for storing and manipulating array antenna models |
| [.calc_beamwidth_deg](#.calc_beamwidth_deg) | Calculate the beamwidth and pointing angles of array antenna elements in degrees |
| [.calc_directivity_dBi](#.calc_directivity_dBi) | Calculate the directivity in dBi of a single array element |
| [.is_valid](#.is_valid) | Validate the integrity of an arrayant object |
| [.set_size](#.set_size) | Resize an arrayant object to new dimensions |

---
<a id="arrayant"></a>
## arrayant
Class for storing and manipulating array antenna models

Not in: MATLAB, Python

- Represents a multi-element antenna array; each element has a position relative to the array phase-center
- Elements may be inter-coupled via a complex coupling matrix
- Field pattern cubes `e_theta_re/im`, `e_phi_re/im` must all be `[n_elevation, n_azimuth, n_elements]`
- `element_pos` is optional (empty = all elements at origin); `coupling_re/im` are optional (empty = identity); all
  functions give the same results for empty members as for zeros / identity, except the IRS array of
  [get_channels_irs](channel-generation-functions.md#get_channels_irs), whose coupling holds the IRS weights and must be given. In a multi-frequency vector all
  entries use the same form (see [arrayant_is_valid_multi](array-antenna-functions.md#arrayant_is_valid_multi))
- Field values and coupling weights must be finite with magnitudes below about 1e18; larger values overflow in the
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
| `arma::Mat<dtype> coupling_re`    | `[n_elements, n_ports]` or empty       | Coupling matrix, real part                                     |
| `arma::Mat<dtype> coupling_im`    | `[n_elements, n_ports]` or empty       | Coupling matrix, imaginary part                                |
| `dtype center_frequency`          | scalar                                 | Center frequency                                               |
| `std::string name`                | string                                 | Name of the array antenna object                               |

### Simple member functions:
| Function         | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `.n_elevation()` | Number of elevation angles                                             |
| `.n_azimuth()`   | Number of azimuth angles                                               |
| `.n_elements()`  | Number of antenna elements                                             |
| `.n_ports()`     | Number of ports (columns of coupling matrix)                           |
| `.copy()`        | Returns a deep copy; a copy of a valid arrayant is marked as validated |
| `.reset()`       | Clears all data, resetting size to zero                                |
| `.is_valid()`    | Returns `""` if valid, or an error message string                      |

### Complex member functions:
| Function                  | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| [.calc_beamwidth_deg](#.calc_beamwidth_deg)   | Calculate the beam width of an antenna element in degree   |
| [.calc_directivity_dBi](#.calc_directivity_dBi) | Calculate the directivity in dBi of a single array element |
| [.set_size](#.set_size)             | Resize the arrayant to new dimensions                      |
| [.is_valid](#.is_valid)             | Validate arrayant integrity                                |

Computations and file I/O on an arrayant are free functions, see section "Array antenna functions"
(e.g. [arrayant_concat](array-antenna-functions.md#arrayant_concat), [arrayant_combine_pattern](array-antenna-functions.md#arrayant_combine_pattern), [arrayant_copy_element](array-antenna-functions.md#arrayant_copy_element), [arrayant_rotate_pattern](array-antenna-functions.md#arrayant_rotate_pattern),
[qdant_write](array-antenna-functions.md#qdant_write)).

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
- [arrayant_combine_pattern](array-antenna-functions.md#arrayant_combine_pattern) (the per-port directivity is a typical follow-up)

---
<a id=".is_valid"></a>
## .is_valid
Validate the integrity of an arrayant object

Not in: MATLAB, Python

- Always checks the data: sizes of all members, ranges and order of the angle grids, sizes of element positions and
  coupling matrix; empty `element_pos` and coupling are allowed
- A passing check marks the object as validated: the library functions then skip the full check as long as no data
  member was reallocated or resized; values changed in place after the check are not seen by them
- The mark is updated atomically, so `is_valid` may be called while other threads read the object
- A copy is marked if its source passes the full check; a moved-to object takes over the mark of its source

### Declaration:
```cpp
std::string quadriga_lib::arrayant<dtype>::is_valid() const;
```

### Returns:
- Empty string if valid; error message string if invalid

### See also:
- [arrayant_is_valid_multi](array-antenna-functions.md#arrayant_is_valid_multi) (multi-freq counterpart)

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
