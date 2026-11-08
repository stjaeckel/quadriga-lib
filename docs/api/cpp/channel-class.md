# Channel class

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-channel-class

| Function | Summary |
| --- | --- |
| [channel](#channel) | Class for storing and managing MIMO channel data and metadata across multiple snapshots |
| [.add_paths](#.add_paths) | Append new propagation paths to an existing channel snapshot |
| [.calc_effective_path_gain](#.calc_effective_path_gain) | Calculate the effective path gain per snapshot in linear scale |
| [.write_paths_to_obj_file](#.write_paths_to_obj_file) | Export propagation paths to a Wavefront OBJ file for 3D visualization |

---
<a id="channel"></a>
## channel
Class for storing and managing MIMO channel data and metadata across multiple snapshots

Not in: MATLAB, Python

- Represents path-level MIMO channel data between antenna arrays over multiple time snapshots
- Each snapshot may have a different number of propagation paths `n_path`
- Unstructured metadata supported via `par_names` / `par_data`

### Attributes:
| Attribute                                         | Size                                                            | Description                                                 |
| ------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| `std::string name`                                | —                                                               | Name of the channel object                                  |
| `arma::Col<dtype> center_frequency`               | `[1]`; `[n_snap]`, or `[]`                                      | Center frequency                                            |
| `arma::Mat<dtype> tx_pos`                         | `[3, n_snap]` or `[3, 1]` = static                              | Transmitter positions                                       |
| `arma::Mat<dtype> rx_pos`                         | `[3, n_snap]` or `[3, 1]` = static                              | Receiver positions                                          |
| `arma::Mat<dtype> tx_orientation`                 | `[3, n_snap]`; `[3, 1]` = static, or `[]` = no rotation         | Transmitter orientation (Euler angles)                      |
| `arma::Mat<dtype> rx_orientation`                 | `[3, n_snap]`; `[3, 1]` = static, or `[]` = no rotation         | Receiver orientation (Euler angles)                         |
| `std::vector<arma::Cube<dtype>> coeff_re`         | per snap `[n_rx, n_tx, n_path]`                                 | Channel coefficients, real part                             |
| `std::vector<arma::Cube<dtype>> coeff_im`         | per snap `[n_rx, n_tx, n_path]`                                 | Channel coefficients, imaginary part                        |
| `std::vector<arma::Cube<dtype>> delay`            | per snap `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` = broadcast | Path delays in seconds                                      |
| `std::vector<arma::Col<dtype>> path_gain`         | per snap `[n_path]`                                             | Path gains before antenna pattern                           |
| `std::vector<arma::Col<dtype>> path_length`       | per snap `[n_path]`                                             | Path lengths TX to RX                                       |
| `std::vector<arma::Mat<dtype>> path_polarization` | per snap `[8, n_path]`                                          | Interleaved polarization transfer matrices                  |
| `std::vector<arma::Mat<dtype>> path_angles`       | per snap `[n_path, 4]`                                          | Angles {AOD, EOD, AOA, EOA} in rad                          |
| `std::vector<arma::Mat<dtype>> path_fbs_pos`      | per snap `[3, n_path]`                                          | First-bounce scatterer positions                            |
| `std::vector<arma::Mat<dtype>> path_lbs_pos`      | per snap `[3, n_path]`                                          | Last-bounce scatterer positions                             |
| `std::vector<arma::Col<unsigned>> no_interact`    | per snap `[n_path]`                                             | Number of interactions per path                             |
| `std::vector<arma::Mat<dtype>> interact_coord`    | per snap `[3, sum(no_interact)]`                                | Interaction point coordinates                               |
| `std::vector<std::string> par_names`              | —                                                               | Names of unstructured metadata fields                       |
| `std::vector<std::any> par_data`                  | —                                                               | Unstructured metadata values (string, scalar, matrix, etc.) |
| `int initial_position`                            | scalar                                                          | 0-based index of the reference snapshot                     |

### Simple member functions:
| Method        | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `.n_snap()`   | Returns the number of snapshots                               |
| `.n_rx()`     | Returns number of receive antennas; 0 if coefficients absent  |
| `.n_tx()`     | Returns number of transmit antennas; 0 if coefficients absent |
| `.n_path()`   | Returns number of paths per snapshot as a vector              |
| `.empty()`    | Returns true if the object contains no channel data           |
| `.is_valid()` | Returns empty string if valid, otherwise an error message     |

### Complex member functions:
- [.add_paths](#.add_paths)
- [.calc_effective_path_gain](#.calc_effective_path_gain)
- [.write_paths_to_obj_file](#.write_paths_to_obj_file)

---
<a id=".add_paths"></a>
## .add_paths
Append new propagation paths to an existing channel snapshot

Not in: MATLAB, Python

- Adds path-level data to snapshot `i_snap` in a `channel` object; does not modify `tx_pos`, `rx_pos`, or orientation fields
- All provided fields must have consistent length `n_path_add` and match existing snapshot structure
- The first provided field sets `n_path_add`, also when it holds 0 paths; a field with a different path count throws

### Declaration:
```cpp
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

### Inputs:
- **`i_snap`** — 0-based snapshot index to append paths to
- **`coeff_re_add`** *(optional)* — Real part of channel coefficients; `[n_rx, n_tx, n_path_add]`
- **`coeff_im_add`** *(optional)* — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_add]`
- **`delay_add`** *(optional)* — Propagation delays in seconds; `[n_rx, n_tx, n_path_add]` or `[1, 1, n_path_add]`
- **`no_interact_add`** *(optional)* — Number of interaction points per path; `[n_path_add]`
- **`interact_coord_add`** *(optional)* — Interaction point coordinates; `[3, sum(no_interact)]`
- **`path_gain_add`** *(optional)* — Path gains before antenna effects; `[n_path_add]`
- **`path_length_add`** *(optional)* — Path lengths from TX to RX phase center; `[n_path_add]`
- **`path_polarization_add`** *(optional)* — Interleaved polarization transfer matrices; `[8, n_path_add]`
- **`path_angles_add`** *(optional)* — Departure/arrival angles {AOD, EOD, AOA, EOA} in rad; `[n_path_add, 4]`
- **`path_fbs_pos_add`** *(optional)* — First-bounce scatterer positions; `[3, n_path_add]`
- **`path_lbs_pos_add`** *(optional)* — Last-bounce scatterer positions; `[3, n_path_add]`

---
<a id=".calc_effective_path_gain"></a>
## .calc_effective_path_gain
Calculate the effective path gain per snapshot in linear scale

Not in: MATLAB, Python

- Sums power over all paths and TX/RX antenna pairs to produce one gain value per snapshot
- Uses `coeff_re`/`coeff_im` if available; falls back to `path_polarization` assuming ideal XPOL antennas
- Throws if neither coefficients nor polarization data are present

### Declaration:
```cpp
arma::Col<dtype> quadriga_lib::channel<dtype>::calc_effective_path_gain(bool assume_valid = false) const;
```

### Inputs:
- **`assume_valid`** *(optional)* — Skip internal consistency checks for performance in trusted contexts

### Returns:
- Effective path gains in linear scale, one entry per snapshot; `[n_snap]`

---
<a id=".write_paths_to_obj_file"></a>
## .write_paths_to_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [MATLAB](../mex/channel-functions.md#channel_export_obj_file) · [Python](../python/channel-functions.md#channel_export_obj_file)

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for Blender)
- Tubes are color-coded by path gain using a selected colormap; radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits total count
- Paths with non-finite coordinates are skipped

### Declaration:
```cpp
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

### Inputs:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** *(optional)* — Max paths to export; `0` includes all paths above `gain_min`
- **`gain_max`** *(optional)* — Upper gain threshold in dB for color/radius mapping; higher values are clipped
- **`gain_min`** *(optional)* — Lower gain threshold in dB; paths below this are excluded
- **`colormap`** *(optional)* — Colormap name; see [colormap](site-specific-simulation-tools.md#colormap) for supported options
- **`i_snap`** *(optional)* — 0-based snapshot indices to include; empty exports all snapshots
- **`radius_max`** *(optional)* — Tube radius at maximum gain
- **`radius_min`** *(optional)* — Tube radius at minimum gain
- **`n_edges`** *(optional)* — Vertices per tube cross-section; must be ≥ 3

### See also:
- [path_to_tube](site-specific-simulation-tools.md#path_to_tube) (generates tube geometry from path data)
- [colormap](site-specific-simulation-tools.md#colormap) (colormap lookup used for coloring)
