# Channel data and processing

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-channel-data-and-processing

| Function | Summary |
| --- | --- |
| [channel](#channel) | MIMO channel between a transmit and a receive array: coefficients, delays and path data per snapshot, and metadata |
| [calc_angular_spread](#calc_angular_spread) | Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths |
| [calc_channel_gain](#calc_channel_gain) | Calculate the channel gain of each snapshot, the power summed over all paths and antenna pairs |
| [calc_delay_spread](#calc_delay_spread) | Calculate the RMS delay spread of each CIR from its path delays and powers |
| [calc_rician_k_factor](#calc_rician_k_factor) | Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power |
| [calc_xpr](#calc_xpr) | Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis |
| [crossfade_segments](#crossfade_segments) | Join the channels of consecutive segments of one link by a cross-fade |
| [export_channel_obj](#export_channel_obj) | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [freq_response](#freq_response) | Frequency response of a MIMO channel at the given subcarriers |
| [freq_response_mf](#freq_response_mf) | Frequency response of a MIMO channel from coefficients sampled at several frequencies |
| [hdf5_create_file](#hdf5_create_file) | Create a new HDF5 channel file with a given storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Read channel objects from one or several slots of an HDF5 channel file |
| [hdf5_read_dset](#hdf5_read_dset) | Read an unstructured dataset from a slot of an HDF5 channel file |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Read the names of the unstructured datasets of a slot of an HDF5 channel file |
| [hdf5_read_layout](#hdf5_read_layout) | Read the storage layout of an HDF5 channel file |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Reshape the storage layout of an existing HDF5 channel file |
| [hdf5_write_channel](#hdf5_write_channel) | Write channel objects to one or several slots of an HDF5 channel file |
| [hdf5_write_dset](#hdf5_write_dset) | Write an unstructured dataset to a slot of an HDF5 channel file |
| [qrt_file_append](#qrt_file_append) | Append the paths of one origin to an existing QRT file |
| [qrt_file_init](#qrt_file_init) | Create a new QRT file and write its metadata header |
| [qrt_file_parse](#qrt_file_parse) | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Read the paths of CIRs from a QRT file |
| [qrt_file_read_raw](#qrt_file_read_raw) | Read the stored paths of one origin of a QRT file as path objects |
| [qrt_paths_from_scatterers](#qrt_paths_from_scatterers) | Path objects of one CIR from spherical path data, for qrt_file_append |
| [qrt_read_cache_init](#qrt_read_cache_init) | Read the metadata of a QRT file into a cache for repeated reads |
| [quantize_delays](#quantize_delays) | Map the path delays to a tap grid by two-tap interpolation |

A **channel** describes the propagation between a transmit and a receive antenna array: for each snapshot, a set of
propagation paths, each with a complex coefficient per antenna pair and a delay. The functions of this section store
channels in files, turn them into frequency responses, prepare them for link-level simulation, compute their statistics
and show the paths in 3D. The channels themselves come from the functions of section "Channel generators" or from
[get_channels_layout](network-layouts-and-mobility.md#get_channels_layout).

**What a channel holds** ([channel](#channel)):

- The positions and orientations of the transmitter and the receiver, per snapshot or fixed
- The **coefficients** `coeff_re` / `coeff_im` and the **delays** `delay` of each snapshot, `[n_rx, n_tx, n_path]`
- Optionally, antenna-independent path data: path gain, path length, polarization transfer matrix, angles, scatterer
  positions and the interaction points of ray-traced paths
- Unstructured metadata: named values of any type in `par_names` / `par_data`

**Typical workflow:**

1. Generate the channels, or read them: [hdf5_read_channel](#hdf5_read_channel) from a channel file, [qrt_file_read](#qrt_file_read) from the output
   of the QRT ray tracer
2. Optionally, map the delays to a tap grid with [quantize_delays](#quantize_delays), or join the segments of a track with
   [crossfade_segments](#crossfade_segments)
3. Compute the frequency response at the subcarriers with [freq_response](#freq_response), or with [freq_response_mf](#freq_response_mf) for
   coefficients sampled at several frequencies
4. Evaluate the channels: [calc_channel_gain](#calc_channel_gain), [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread),
   [calc_rician_k_factor](#calc_rician_k_factor), [calc_xpr](#calc_xpr)
5. Store them with [hdf5_write_channel](#hdf5_write_channel), or show the paths with [export_channel_obj](#export_channel_obj)

**Terms** (full list in the [Glossary](general-usage-notes.md#glossary)):

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
- **Multi-frequency channel** (suffix `_mf`): coefficients sampled at several frequencies, one entry per frequency, as
  [get_channels_multifreq](channel-generators.md#get_channels_multifreq) produces them
- **Slot**: one storage place of an HDF5 channel file, addressed by four indices `(ix, iy, iz, iw)`, e.g. Tx, Rx and
  frequency; the file holds a fixed grid of slots, its **storage layout**
- **Origin**, **destination**: a QRT file holds the paths from each origin (a Tx position) to a list of CIR positions; a
  destination (a receiver) owns a run of consecutive CIR positions, e.g. a track

**File formats.** [HDF5 channel files](../../formats/hdf5_channel.md) store channels with antennas, one per slot, in single
precision; [QRT files](../../formats/qrt.md) store propagation paths from ray tracing or the GBSM before antennas are
applied.

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Store and read channels | [hdf5_create_file](#hdf5_create_file), [hdf5_write_channel](#hdf5_write_channel), [hdf5_read_channel](#hdf5_read_channel), [hdf5_read_layout](#hdf5_read_layout), [hdf5_reshape_layout](#hdf5_reshape_layout) |
| Store and read metadata | [hdf5_write_dset](#hdf5_write_dset), [hdf5_read_dset](#hdf5_read_dset), [hdf5_read_dset_names](#hdf5_read_dset_names) |
| Path files (QRT) | [qrt_file_init](#qrt_file_init), [qrt_file_append](#qrt_file_append), [qrt_paths_from_scatterers](#qrt_paths_from_scatterers), [qrt_file_parse](#qrt_file_parse), [qrt_file_read](#qrt_file_read), [qrt_file_read_raw](#qrt_file_read_raw), [qrt_read_cache_init](#qrt_read_cache_init) |
| Frequency response | [freq_response](#freq_response), [freq_response_mf](#freq_response_mf) |
| Process | [quantize_delays](#quantize_delays), [crossfade_segments](#crossfade_segments) |
| Statistics | [calc_channel_gain](#calc_channel_gain), [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread), [calc_rician_k_factor](#calc_rician_k_factor), [calc_xpr](#calc_xpr) |
| Show | [export_channel_obj](#export_channel_obj) |

**Conventions.** Coefficients and delays are given per snapshot as a `std::vector` of cubes. [freq_response](#freq_response),
[freq_response_mf](#freq_response_mf), [hdf5_read_channel](#hdf5_read_channel) and [hdf5_write_channel](#hdf5_write_channel) have vector forms, same-name overloads with one
entry per snapshot or slot.

---
<a id="channel"></a>
## channel
MIMO channel between a transmit and a receive array: coefficients, delays and path data per snapshot, and metadata

Not in: MATLAB, Python

- The data of one link (section description, "What a channel holds"); the fields are public and set directly, e.g.
  from [get_channels_spherical](channel-generators.md#get_channels_spherical), or read with [hdf5_read_channel](#hdf5_read_channel) and [qrt_file_read](#qrt_file_read)
- Snapshot fields are `std::vector` with one entry per snapshot; an empty field is not present. `n_snap` is the largest
  number of entries of any field; positions and orientations with one column apply to all snapshots
- `.is_valid()` checks the sizes; the functions that take a channel check it first

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::string name` | at most 255 characters | Name of the channel, default `"empty"` |
| `arma::Col<dtype> center_freq` | `[1]`, `[n_snap]` or `[]` | Center frequency in Hz |
| `arma::Mat<dtype> tx_pos` | `[3, 1]` or `[3, n_snap]` | Transmitter positions in m |
| `arma::Mat<dtype> rx_pos` | `[3, 1]` or `[3, n_snap]` | Receiver positions in m |
| `arma::Mat<dtype> tx_orientation` | `[3, 1]`, `[3, n_snap]` or `[]` | Transmitter orientation, Euler angles (bank, tilt, heading) in rad |
| `arma::Mat<dtype> rx_orientation` | `[3, 1]`, `[3, n_snap]` or `[]` | Receiver orientation, Euler angles (bank, tilt, heading) in rad |
| `std::vector<arma::Cube<dtype>> coeff_re` | per snapshot `[n_rx, n_tx, n_path]` | Coefficients, real part; `n_rx` and `n_tx` equal in all snapshots |
| `std::vector<arma::Cube<dtype>> coeff_im` | per snapshot `[n_rx, n_tx, n_path]` | Coefficients, imaginary part |
| `std::vector<arma::Cube<dtype>> delay` | per snapshot `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` | Delays in s; `[1, 1, n_path]` without coefficients |
| `std::vector<arma::Col<dtype>> path_gain` | per snapshot `[n_path]` | Path gain, linear, without antennas |
| `std::vector<arma::Col<dtype>> path_length` | per snapshot `[n_path]` | Path length in m from the Tx to the Rx phase center |
| `std::vector<arma::Mat<dtype>> polmat` | per snapshot `[8, n_path]` | Polarization transfer matrices, interleaved real and imaginary parts |
| `std::vector<arma::Mat<dtype>> path_angles` | per snapshot `[n_path, 4]` | Departure and arrival angles (AOD, EOD, AOA, EOA) in rad |
| `std::vector<arma::Mat<dtype>> fbs_pos` | per snapshot `[3, n_path]` | First-bounce scatterer positions in m |
| `std::vector<arma::Mat<dtype>> lbs_pos` | per snapshot `[3, n_path]` | Last-bounce scatterer positions in m |
| `std::vector<arma::Col<unsigned>> n_interact` | per snapshot `[n_path]` | Number of interactions of each path, 0 = LOS; with `interact_coord` |
| `std::vector<arma::Mat<dtype>> interact_coord` | per snapshot `[3, sum(n_interact)]` | Interaction points of the paths in m, in path order |
| `std::vector<std::string> par_names` | `[n_par]` | Names of the metadata fields |
| `std::vector<std::any> par_data` | `[n_par]` | Metadata values, of a type [any_type_id](miscellaneous-tools.md#any_type_id) knows |
| `int initial_position` | scalar | 0-based index of the reference snapshot, mainly for internal use |

### Simple member functions:
| Method | Description |
| ------ | ----------- |
| `.n_snap()` | Number of snapshots |
| `.n_rx()` | Number of receive antennas; 0 without coefficients |
| `.n_tx()` | Number of transmit antennas; 0 without coefficients |
| `.n_path()` | Number of paths of each snapshot, `[n_snap]` |
| `.empty()` | True if the channel holds no structured data |
| `.is_valid()` | Empty string if the channel is valid, else an error message |

Computations and file I/O on a channel are free functions in this section, e.g. [calc_channel_gain](#calc_channel_gain),
[export_channel_obj](#export_channel_obj), [hdf5_write_channel](#hdf5_write_channel) and [quantize_delays](#quantize_delays).

---
<a id="calc_angular_spread"></a>
## calc_angular_spread
Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths

Also in: [MATLAB](../mex/channel-data-and-processing.md#calc_angular_spread) · [Python](../python/channel-data-and-processing.md#calc_angular_spread)

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

### Declaration:
```cpp
void quadriga_lib::calc_angular_spread(
    const std::vector<arma::Col<dtype>> &az,
    const std::vector<arma::Col<dtype>> &el,
    const std::vector<arma::Col<dtype>> &powers,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    std::vector<arma::Col<dtype>> *phi = nullptr,
    std::vector<arma::Col<dtype>> *theta = nullptr,
    bool wrapping = false,
    bool calc_bank_angle = false,
    double quantize_deg = 0.0);
```

### Inputs:
- **`az`** — Path azimuths in rad; `[n_cir]` vectors `[n_path]`, at least one CIR and one path per CIR; `n_path` may
  differ between the CIRs
- **`el`** — Path elevations in rad; `[n_cir]` vectors `[n_path]`
- **`powers`** — Path powers, linear, with a positive sum per CIR; `[n_cir]` vectors `[n_path]`
- **`wrapping`** *(optional)* — Rotate the paths to the equator before computing the spreads
- **`calc_bank_angle`** *(optional)* — Rotate them also about the mean direction, for the largest azimuth
  spread; only with `wrapping`
- **`quantize_deg`** *(optional)* — Quantization step of the angle deviations in deg; 0 = none, else finite and >= 1e-3

### Outputs:
- **`azimuth_spread`** *(optional)* — RMS azimuth spread in rad; `[n_cir]`
- **`elevation_spread`** *(optional)* — RMS elevation spread in rad; `[n_cir]`
- **`orientation`** *(optional)* — Euler angles (bank, tilt, heading) in rad of an antenna pointing at the mean
  direction (a positive tilt points down), zero without `wrapping`; `[3, n_cir]`
- **`phi`** *(optional)* — Path azimuths in rad in the frame of that antenna (as [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern)
  returns them), the input `az` without `wrapping`; `[n_cir]` vectors `[n_path]`
- **`theta`** *(optional)* — Path elevations in rad in that frame, the input `el` without `wrapping`;
  `[n_cir]` vectors `[n_path]`

### See also:
- [calc_delay_spread](#calc_delay_spread) (delay spread of the same paths)
- [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern) (antenna response in the rotated frame)

---
<a id="calc_channel_gain"></a>
## calc_channel_gain
Calculate the channel gain of each snapshot, the power summed over all paths and antenna pairs

Not in: MATLAB, Python

- With coefficients: $G = \sum_{r,t,l} \lvert a_{r,t,l} \rvert^2$ over all receive and transmit antennas and paths
- Without coefficients, from `polmat` for ideal dual-polarized antennas (V and H, unit gain) at both ends:
  $G = \sum_l g_l \sum_{i} \lvert M_{l,i} \rvert^2$ over the four entries of the polarization transfer matrix of each
  path; $g_l$ is `path_gain`, else the free-space gain
  $10^{-(32.45 + 20 \log_{10}(f / \mathrm{GHz}) + 20 \log_{10}(d_l / \mathrm{m})) / 10}$ from `path_length` $d_l$ at
  `center_freq` $f$ (per snapshot if it has `n_snap` entries, 1 GHz if empty), else 1

### Declaration:
```cpp
arma::Col<dtype> quadriga_lib::calc_channel_gain(const channel<dtype> &chan);
```

### Input:
- **`chan`** — Channel, valid (see [channel](#channel)), with `coeff_re` / `coeff_im` or `polmat`; a channel without snapshots
  gives an empty result

### Returns:
- Channel gain, linear, 0 for a snapshot without paths; `[n_snap]`

### See also:
- [calc_xpr](#calc_xpr) (path gain and XPR from `polmat`)
- [calc_rician_k_factor](#calc_rician_k_factor) (gain split into LOS and NLOS power)

---
<a id="calc_delay_spread"></a>
## calc_delay_spread
Calculate the RMS delay spread of each CIR from its path delays and powers

Also in: [MATLAB](../mex/channel-data-and-processing.md#calc_delay_spread) · [Python](../python/channel-data-and-processing.md#calc_delay_spread)

- With the path powers $p_l$ normalized to sum 1 over the paths of a CIR, the mean delay and the RMS delay spread are

  $$\bar\tau = \sum_l p_l \, \tau_l, \qquad \sigma_\tau = \sqrt{\sum_l p_l \, (\tau_l - \bar\tau)^2}$$

- Paths more than `threshold_dB` below the strongest path of their CIR are left out
- With `granularity` > 0, the delays are first rounded to the nearest multiple of `granularity`, and the powers of
  the paths with the same rounded delay are summed
- A CIR without power gives a spread and a mean delay of 0

### Declaration:
```cpp
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    double threshold_dB = 100.0,
    double granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

### Inputs:
- **`delays`** — Path delays in s; `[n_cir]` vectors `[n_path]`, at least one CIR and one path per CIR; `n_path` may
  differ between the CIRs
- **`powers`** — Path powers, linear; `[n_cir]` vectors `[n_path]`
- **`threshold_dB`** *(optional)* — Power range in dB below the strongest path that is kept, > 0
- **`granularity`** *(optional)* — Spacing of the delay grid in s, finite and >= 0; 0 = no grid; with a grid, all
  delays must be finite and their span must be below $2^{53}$ grid steps

### Output:
- **`mean_delay`** *(optional)* — Mean delay $\bar\tau$ in s; `[n_cir]`

### Returns:
- RMS delay spread $\sigma_\tau$ in s; `[n_cir]`

### See also:
- [quantize_delays](#quantize_delays) (delays on a tap grid with interpolation)
- [calc_angular_spread](#calc_angular_spread) (angular spreads)

---
<a id="calc_rician_k_factor"></a>
## calc_rician_k_factor
Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power

Also in: [MATLAB](../mex/channel-data-and-processing.md#calc_rician_k_factor) · [Python](../python/channel-data-and-processing.md#calc_rician_k_factor)

- $K = P_\mathrm{LOS} / P_\mathrm{NLOS}$ with the sums of the path powers; LOS paths are those with a length
  `path_length` <= $d_\mathrm{TR}$ + `window_size`, $d_\mathrm{TR}$ the Tx-Rx distance, all others are NLOS
- Without NLOS power, $K$ is infinite, or 0 if the LOS power is zero too (also for a CIR without paths)

### Declaration:
```cpp
void quadriga_lib::calc_rician_k_factor(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Col<dtype> *kf = nullptr,
    arma::Col<dtype> *pg = nullptr,
    double window_size = 0.01);
```

### Inputs:
- **`powers`** — Path powers, linear; `[n_cir]` vectors `[n_path]`, at least one CIR; `n_path` may differ between the
  CIRs
- **`path_length`** — Path lengths in m; `[n_cir]` vectors `[n_path]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`window_size`** *(optional)* — LOS window in m, >= 0

### Outputs:
- **`kf`** *(optional)* — K-factor, linear; `[n_cir]`
- **`pg`** *(optional)* — Sum of all path powers, linear; `[n_cir]`

### See also:
- [calc_xpr](#calc_xpr) (XPR with the same LOS window)
- [calc_delay_spread](#calc_delay_spread) (delay spread of the same paths)

---
<a id="calc_xpr"></a>
## calc_xpr
Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis

Also in: [MATLAB](../mex/channel-data-and-processing.md#calc_xpr) · [Python](../python/channel-data-and-processing.md#calc_xpr)

- Total-power ratio: the co- and cross-polar powers of the paths of a CIR, weighted by the path powers $p_l$, are
  summed, and the XPR is their ratio; LOS paths are left out unless `include_los` is set
- Entries of the polarization transfer matrix $M$ of a path, first index receive, second transmit polarization:
  rows 0-1 of `polmat` = $M_{vv}$, 2-3 = $M_{hv}$ (H rx, V tx), 4-5 = $M_{vh}$ (V rx, H tx), 6-7 = $M_{hh}$, each as
  real and imaginary part
- Circular basis: $M_{rt} = \mathbf{r}^T M \, \mathbf{t}$ (plain transpose), as [get_channels_spherical](channel-generators.md#get_channels_spherical)
  couples antenna patterns, with the same vectors at both ends, each in the $(e_\theta, e_\phi)$ frame of its antenna:
  $\mathbf{L} = (1, +j) / \sqrt{2}$ (LHCP) and $\mathbf{R} = (1, -j) / \sqrt{2}$ (RHCP) in (V, H), which gives

  $$M_{LL}, M_{RR} = \frac{1}{2} \left( M_{vv} - M_{hh} \pm j (M_{vh} + M_{hv}) \right)$$

  $$M_{LR}, M_{RL} = \frac{1}{2} \left( M_{vv} + M_{hh} \pm j (M_{hv} - M_{vh}) \right)$$

  (upper sign for the first entry)

- The names L and R follow the IEEE definition (rotation seen along the propagation direction) for the time
  convention $e^{+j 2\pi f t}$, which the coefficients $e^{-j 2\pi f \tau}$ of the library imply, with
  $e_\theta \times e_\phi$ the outward direction of the pattern
- Co-polar means the same circular vector at both ends: the LOS matrix $\mathrm{diag}(1, -1)$ of
  [get_channels_spherical](channel-generators.md#get_channels_spherical) is co-polar (handedness kept), $\mathrm{diag}(1, 1)$ (one mirrored component, as
  by a single reflection) is cross-polar

### Declaration:
```cpp
void quadriga_lib::calc_xpr(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &polmat,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr = nullptr,
    arma::Col<dtype> *pg = nullptr,
    bool include_los = false,
    double window_size = 0.01);
```

### Inputs:
- **`powers`** — Path powers $p_l$, linear; `[n_cir]` vectors `[n_path]`, at least one; `n_path` may differ between
  the CIRs
- **`polmat`** — Polarization transfer matrices, magnitudes below about 1e18; `[n_cir]` matrices `[8, n_path]`
- **`path_length`** — Path lengths in m; `[n_cir]` vectors `[n_path]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`include_los`** *(optional)* — Include the LOS paths in the XPR
- **`window_size`** *(optional)* — LOS window in m, >= 0: paths with `path_length` <= $d_\mathrm{TR}$ +
  `window_size` are LOS, $d_\mathrm{TR}$ the Tx-Rx distance

### Outputs:
- **`xpr`** *(optional)* — XPR, linear; `[n_cir, 6]`; columns:

  | Column | XPR |
  | ------ | --- |
  | 0 | Linear: $(P_{vv} + P_{hh}) / (P_{vh} + P_{hv})$ with $P_{rt} = \sum_l p_l \lvert M_{rt} \rvert^2$ |
  | 1 | V (receive V): $P_{vv} / P_{vh}$ |
  | 2 | H (receive H): $P_{hh} / P_{hv}$ |
  | 3 | Circular: $(P_{LL} + P_{RR}) / (P_{LR} + P_{RL})$ |
  | 4 | LHCP (receive L): $P_{LL} / P_{LR}$ |
  | 5 | RHCP (receive R): $P_{RR} / P_{RL}$ |

- **`pg`** *(optional)* — Path gain of all paths, LOS included,
  $\frac{1}{2} \sum_l p_l \sum_{r,t \in \{v,h\}} \lvert M_{rt} \rvert^2$; `[n_cir]`

### Notes:
- A ratio with zero cross-polar power is infinite, or 0 if the co-polar power is zero too (also for a CIR without
  paths, which has `pg` = 0)
- A scaling of `polmat` does not change the XPR, but `pg`

### See also:
- [calc_rician_k_factor](#calc_rician_k_factor) (LOS and NLOS power with the same LOS window)
- [get_channels_spherical](channel-generators.md#get_channels_spherical) (polarization transfer matrices of the paths)

---
<a id="crossfade_segments"></a>
## crossfade_segments
Join the channels of consecutive segments of one link by a cross-fade

Also in: [MATLAB](../mex/channel-data-and-processing.md#crossfade_segments) · [Python](../python/channel-data-and-processing.md#crossfade_segments)

- Joins the "rays" channels of the segments of one link, e.g. of a track whose scenario or indoor state changes, into
  one channel; consecutive segments share the snapshots of a ramp, over which the clusters of the old segment are
  exchanged for those of the new one
- [get_channels_from_rays](network-layouts-and-mobility.md#get_channels_from_rays) joins the segments of tracks with this function, the ramp centred on the row of the change

### Declaration:
```cpp
channel<double> quadriga_lib::crossfade_segments(
    const std::vector<channel<double>> &segments,
    const arma::uvec &n_overlap = {},
    const std::string &output = "taps");
```

### Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients and path gains (`"rays"`), the same
  number of paths in every snapshot, equal `n_rx`, `n_tx` and `center_freq` (at most one value); `[n_segment]`
- **`n_overlap`** *(optional)* — Number of snapshots at the start of each segment that are also the last snapshots of
  the preceding segment (computed at the same positions); they form the ramp. The first entry is 0, at most the
  snapshots of the preceding segment after its own shared ones; omitted: the snapshots whose Tx and Rx positions match
  (within 1 µm), an error where the positions repeat (e.g. ends that do not move); `[n_segment]`
- **`output`** *(optional)* — `"taps"` (the paths summed per tap index) or `"rays"`

### Returns:
- The joined channel, see [channel](#channel)

### Notes:
- **Snapshots**: segment $k$ starts with the `n_overlap[k]` snapshots of the ramp, which are also the last ones of
  segment $k - 1$; the output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots. Outside the
  ramps the snapshots are those of one segment
- **Paths**: `par_data` entries of [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel): `path_type` gives the LOS and GR paths (without it, tap
  index 0 of `tap` and a tap index 3 held by one path, without index 4 or 5, as [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) numbers them), `cluster`
  the clusters, whose paths fade together (without it: the taps, without `tap`: each path)
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
  `name` and `center_freq` of segment 0; `"rays"` keep `path_length`, `polmat`, `path_angles`, `fbs_pos` and
  `lbs_pos` where all segments have them; `par_data`, `n_interact` and `interact_coord` are not carried

### See also:
- [get_channels_from_rays](network-layouts-and-mobility.md#get_channels_from_rays) (joins the segments of tracks with this function)
- [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (channels of one segment)

---
<a id="export_channel_obj"></a>
## export_channel_obj
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [MATLAB](../mex/channel-data-and-processing.md#export_channel_obj) · [Python](../python/channel-data-and-processing.md#export_channel_obj)

- Writes the paths of ray-traced channels as tubes to an OBJ file and their colors to an MTL file of the same name
  (`.mtl` instead of `.obj`), e.g. for Blender; existing files are overwritten
- Path gain in dB: with coefficients $10 \log_{10} \sum_{r,t} \lvert a_{r,t} \rvert^2$ over all antenna pairs, else
  from `polmat` as in [calc_channel_gain](#calc_channel_gain) with the free-space gain of the length of the drawn path
- The gain, clipped to `[gain_min_dB, gain_max_dB]`, sets the color (64 steps of `colormap`) and the radius (linear
  from 0 at `gain_min_dB` to `radius_max` at `gain_max_dB`, at least `radius_min`); a path with a radius up to
  0.1 mm is written as a polyline
- Paths below `gain_min_dB` and paths with non-finite coordinates are left out

### Declaration:
```cpp
void quadriga_lib::export_channel_obj(
    const std::string &fn,
    const channel<dtype> &chan,
    arma::uword max_paths = 0,
    double gain_max_dB = -60.0,
    double gain_min_dB = -140.0,
    const std::string &colormap = "jet",
    const uint_list &i_snap = {},
    double radius_max = 0.05,
    double radius_min = 0.01,
    arma::uword n_edges = 5);
```

### Inputs:
- **`fn`** — Path of the OBJ file, ending in `.obj`
- **`chan`** — Channel, valid and not empty (see [channel](#channel)), with `center_freq`, `n_interact`, `interact_coord` and
  `coeff_re` / `coeff_im` or `polmat`
- **`max_paths`** *(optional)* — Number of the strongest paths per snapshot to export; 0 = all
- **`gain_max_dB`** *(optional)* — Gain in dB of the largest radius and the last color, finite
- **`gain_min_dB`** *(optional)* — Gain in dB of the first color, finite and below `gain_max_dB`; weaker paths are
  left out
- **`colormap`** *(optional)* — Name of the colormap, see [colormap](site-specific-simulation-tools.md#colormap)
- **`i_snap`** *(optional)* — 0-based indices of the snapshots to export, each below `n_snap`; omitted = all
- **`radius_max`** *(optional)* — Tube radius in m at `gain_max_dB`, >= 0
- **`radius_min`** *(optional)* — Smallest tube radius in m, >= 0; at most `radius_max` (a larger value is reduced to
  it)
- **`n_edges`** *(optional)* — Vertices of the tube cross-section, >= 3

### See also:
- [path_to_tube](site-specific-simulation-tools.md#path_to_tube) (tube geometry of one path)
- [colormap](site-specific-simulation-tools.md#colormap) (the colors)
- [qrt_file_read](#qrt_file_read) (ray-traced channels with interaction points)

---
<a id="freq_response"></a>
## freq_response
Frequency response of a MIMO channel at the given subcarriers

Also in: [MATLAB](../mex/channel-data-and-processing.md#freq_response) · [Python](../python/channel-data-and-processing.md#freq_response)

- The frequency response of one snapshot at the subcarriers $k$ (section description, "Frequency response"):
  $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$ with the offset $f_k$ = `bandwidth` · `pilot_grid[k]` from the carrier
- Vector form: one cube per snapshot; the snapshots are computed in parallel, `i_snap` selects them
- For a multi-frequency channel and absolute carrier frequencies, use [freq_response_mf](#freq_response_mf)

### Declarations:
```cpp
void quadriga_lib::freq_response(
    const arma::Cube<dtype> &coeff_re,
    const arma::Cube<dtype> &coeff_im,
    const arma::Cube<dtype> &delay,
    const double_list &pilot_grid,
    const double bandwidth,
    arma::Cube<dtype> *hmat_re = nullptr,
    arma::Cube<dtype> *hmat_im = nullptr,
    arma::Cube<std::complex<dtype>> *hmat = nullptr,
    int use_kernel = 0);

void quadriga_lib::freq_response(
    const std::vector<arma::Cube<dtype>> &coeff_re,
    const std::vector<arma::Cube<dtype>> &coeff_im,
    const std::vector<arma::Cube<dtype>> &delay,
    const double_list &pilot_grid,
    const double bandwidth,
    std::vector<arma::Cube<dtype>> *hmat_re = nullptr,
    std::vector<arma::Cube<dtype>> *hmat_im = nullptr,
    std::vector<arma::Cube<std::complex<dtype>>> *hmat = nullptr,
    const uint_list &i_snap = {},
    int use_kernel = 0);
```

### Inputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_path]`; vector form:
  one cube per snapshot, `[n_snap]`, at least one
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`; vector form: one cube per snapshot
- **`pilot_grid`** — Subcarrier positions relative to the bandwidth, finite: 0 = carrier, 1 = carrier + `bandwidth`;
  `[n_carrier]`, at least one
- **`bandwidth`** — Bandwidth in Hz, finite and >= 0
- **`i_snap`** *(optional)* — Vector form: 0-based indices of the snapshots to compute, each below `n_snap`; omitted =
  all; `[n_out]`
- **`use_kernel`** *(optional)* — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2; throws if
  unavailable

### Outputs:
- **`hmat_re`** / **`hmat_im`** *(optional)* — Real and imaginary part of the frequency response;
  `[n_rx, n_tx, n_carrier]`; vector form: one cube per selected snapshot, `[n_out]`
- **`hmat`** *(optional)* — Complex frequency response; sizes as `hmat_re`

### Notes:
- Precision: the phase $f_k \tau_l$ in cycles is computed and reduced to one period in double precision, so long
  (absolute) delays keep their accuracy; sin / cos, coefficients and sums are single precision in both kernels, also
  for `dtype = double`
- A snapshot without paths gives a zero frequency response
- Vector form: the snapshots may differ in `n_rx`, `n_tx` and `n_path`; an error names the first failing snapshot, and
  the outputs may then be partly written

### See also:
- [freq_response_mf](#freq_response_mf) (multi-frequency channel, absolute carrier frequencies)
- [get_channels_spherical](channel-generators.md#get_channels_spherical) (coefficients and delays)
- [quantize_delays](#quantize_delays) (delays on a tap grid, for time-domain processing)

---
<a id="freq_response_mf"></a>
## freq_response_mf
Frequency response of a MIMO channel from coefficients sampled at several frequencies

Also in: [MATLAB](../mex/channel-data-and-processing.md#freq_response_mf) · [Python](../python/channel-data-and-processing.md#freq_response_mf)

- The frequency response of one snapshot of a multi-frequency channel at the absolute carrier frequencies
  $f_k$ = `freq_new[k]`: the coefficients of each path are interpolated from the input frequencies `freq` to $f_k$
  and the delay phase is applied there, $H_k = \sum_l \tilde a_l(f_k) \, e^{-j 2\pi f_k \tau_l}$
- Interpolation between the two neighbouring input frequencies: amplitude linear, phase along the shorter arc
  ([fast_slerp](math-functions.md#fast_slerp)); constant outside the range of `freq`
- `remove_delay_phase = true` interpolates the envelope $a_l(f_i) \, e^{j 2\pi f_i \tau_l}$ instead of $a_l(f_i)$, so
  that the delay phase the channel generators include is not interpolated
- Vector form: one vector per snapshot, indexed `[snapshot][input frequency]`; the snapshots are computed in parallel,
  `i_snap` selects them

### Declarations:
```cpp
void quadriga_lib::freq_response_mf(
    const std::vector<arma::Cube<dtype>> &coeff_re,
    const std::vector<arma::Cube<dtype>> &coeff_im,
    const std::vector<arma::Cube<dtype>> &delay,
    const double_list &freq,
    const double_list &freq_new,
    arma::Cube<dtype> *hmat_re = nullptr,
    arma::Cube<dtype> *hmat_im = nullptr,
    arma::Cube<std::complex<dtype>> *hmat = nullptr,
    bool remove_delay_phase = true,
    int use_kernel = 0);

void quadriga_lib::freq_response_mf(
    const std::vector<std::vector<arma::Cube<dtype>>> &coeff_re,
    const std::vector<std::vector<arma::Cube<dtype>>> &coeff_im,
    const std::vector<std::vector<arma::Cube<dtype>>> &delay,
    const double_list &freq,
    const double_list &freq_new,
    std::vector<arma::Cube<dtype>> *hmat_re = nullptr,
    std::vector<arma::Cube<dtype>> *hmat_im = nullptr,
    std::vector<arma::Cube<std::complex<dtype>>> *hmat = nullptr,
    bool remove_delay_phase = true,
    const uint_list &i_snap = {},
    int use_kernel = 0);
```

### Inputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients, one cube per input frequency, all of
  one size; `[n_freq]` cubes `[n_rx, n_tx, n_path]`, `n_rx` and `n_tx` >= 1; vector form: one such vector per
  snapshot, `[n_snap]`, at least one
- **`delay`** — Delays in s, one cube per input frequency, `[n_freq]`; only `delay[0]` is used, `[n_rx, n_tx, n_path]`
  or `[1, 1, n_path]`, the others may be empty; vector form: one such vector per snapshot
- **`freq`** — Input frequencies in Hz, finite, strictly ascending; `[n_freq]`, at least one
- **`freq_new`** — Output carrier frequencies in Hz, finite; `[n_carrier]`, at least one
- **`remove_delay_phase`** *(optional)* — Remove the delay phase from the coefficients before the interpolation:
  `true` for the coefficients of [get_channels_multifreq](channel-generators.md#get_channels_multifreq) and [get_channels_spherical](channel-generators.md#get_channels_spherical), `false` for coefficients
  without delay phase
- **`i_snap`** *(optional)* — Vector form: 0-based indices of the snapshots to compute, each below `n_snap`; omitted =
  all; `[n_out]`
- **`use_kernel`** *(optional)* — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else
  GENERIC), 1 = GENERIC, 2 = AVX2; throws if unavailable

### Outputs:
- **`hmat_re`** / **`hmat_im`** *(optional)* — Real and imaginary part of the frequency response;
  `[n_rx, n_tx, n_carrier]`; vector form: one cube per selected snapshot, `[n_out]`
- **`hmat`** *(optional)* — Complex frequency response; sizes as `hmat_re`

### Notes:
- With `remove_delay_phase = true`, generate the channels with `use_absolute_delays = true`: their coefficients carry
  the phase of the absolute path length also when `delay` holds relative delays. With relative delays the phase of
  the Tx-Rx distance stays in the coefficients and is interpolated across `freq`, which gives wrong phases (the
  magnitudes stay right) unless the spacing of `freq` is far below $c$ / (Tx-Rx distance)
- One input frequency: the coefficients apply at all carriers; with `remove_delay_phase = true` the result is that of
  [freq_response](#freq_response) with `bandwidth` · `pilot_grid` = `freq_new - freq`
- Precision: the delay phase is reduced to one period in double precision (and removed in double precision with
  `remove_delay_phase`); sin / cos of the output phase and the interpolation are single
  precision, the sums double precision
- A snapshot without paths gives a zero frequency response
- Vector form: an error names the first failing snapshot, and the outputs may then be partly written

### See also:
- [freq_response](#freq_response) (single-frequency coefficients, carriers relative to the bandwidth)
- [get_channels_multifreq](channel-generators.md#get_channels_multifreq) (multi-frequency channel)

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a given storage layout

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_create_file) · [Python](../python/channel-data-and-processing.md#hdf5_create_file)

- Creates a channel file with `nx · ny · nz · nw` empty slots; the slot count is fixed, [hdf5_reshape_layout](#hdf5_reshape_layout) changes
  only the dimensions
- [hdf5_write_channel](#hdf5_write_channel) and [hdf5_write_dset](#hdf5_write_dset) create a missing file themselves, with the layout that just holds their
  slots; this function sets any other one
- Standard HDF5 readers open the file, see [HDF5 channel file format](../../formats/hdf5_channel.md); change it only with
  the functions of this section

### Declaration:
```cpp
void quadriga_lib::hdf5_create_file(
    const std::string &fn,
    unsigned nx = 65536,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

### Inputs:
- **`fn`** — File name; the file must not exist
- **`nx`**, **`ny`**, **`nz`**, **`nw`** *(optional)* — Number of slots along x, y, z and w, each at least 1; at most
  4294967295 (2^32 - 1) slots in total

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (writes a channel into a slot)
- [hdf5_read_layout](#hdf5_read_layout) (reads the layout and the occupied slots)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Read channel objects from one or several slots of an HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_channel) · [Python](../python/channel-data-and-processing.md#hdf5_read_channel)

- Returns the channel of slot `(ix, iy, iz, iw)` in `dtype`; an empty slot or a slot outside the layout gives an
  empty channel (`n_snap() == 0`)
- The vector form reads the slots in the columns of `slots` with one opening of the file and returns one channel per
  column
- `i_snap` reads only the given snapshots (see Notes)

### Declarations:
```cpp
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(
    const std::string &fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    const uint_list &i_snap = {});

std::vector<quadriga_lib::channel<dtype>> quadriga_lib::hdf5_read_channel(
    const std::string &fn,
    const arma::u32_mat &slots,
    const uint_list &i_snap = {});
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`ix`**, **`iy`**, **`iz`**, **`iw`** *(optional)* — Slot index, 0-based
- **`slots`** — Slot indices `(ix, iy, iz, iw)` (vector form), 0-based, one column per slot; `[4, n_slot]`
- **`i_snap`** *(optional)* — Snapshot indices, 0-based, any order, repeats allowed, below `n_snap` of every slot
  that has snapshots; empty = all; `[n_out]`

### Returns:
- The channel; vector form: one channel per column of `slots`, `[n_slot]`

### Notes:
- **Snapshot selection**: only the selected snapshots are read. Members with one entry per snapshot are sliced, also
  positions, orientations and `center_freq` stored per snapshot; values stored once and the unstructured data are
  kept. A slot without snapshots (only unstructured data) ignores `i_snap`
- **Round trip** with [hdf5_write_channel](#hdf5_write_channel): every member with one entry per snapshot comes back with one entry per
  snapshot, also for snapshots without paths; a channel written without `center_freq` comes back with an empty one;
  empty orientations come back as `[3, 1]` zeros
- **Unstructured data**: the datasets of the slot with the prefix `par_`, in the order they were written and in their
  stored type ([hdf5_read_dset](#hdf5_read_dset) lists the types); datasets of another prefix are not read
- **Errors**: a missing file, a file that is not a channel file of the
  [supported format](../../formats/hdf5_channel.md#hdf5-compatibility), or a malformed file (e.g. a dataset whose size does not match
  the attributes of the channel, or a missing dataset) throws `std::runtime_error`

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (finds the occupied slots)
- [hdf5_write_channel](#hdf5_write_channel) (writes channels)

---
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read an unstructured dataset from a slot of an HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_dset) · [Python](../python/channel-data-and-processing.md#hdf5_read_dset)

- Reads the dataset `prefix + name` of slot `(ix, iy, iz, iw)`, e.g. `par_carrier_frequency`
- The type follows the stored dataset: a scalar dataset gives a scalar, a 1D dataset an `arma::Col` (also with one
  element, so an `arma::Row` comes back as `arma::Col`), a 2D dataset an `arma::Mat`, a 3D dataset an `arma::Cube`,
  with the element type float, double, unsigned, int, unsigned long long or long long; a string comes back without
  trailing NUL characters
- [any_type_id](miscellaneous-tools.md#any_type_id) gives the type and the shape, `std::any_cast` to that type the value

### Declaration:
```cpp
std::any quadriga_lib::hdf5_read_dset(
    const std::string &fn,
    const std::string &name,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    const std::string &prefix = "par_");
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`name`** — Dataset name without the prefix, e.g. `carrier_frequency`
- **`ix`**, **`iy`**, **`iz`**, **`iw`** *(optional)* — Slot index, 0-based
- **`prefix`** *(optional)* — Prefix of the dataset name; `""` reads `name` as given

### Returns:
- The dataset; an empty `std::any` if the slot is empty or outside the layout or has no such dataset

### Example:
```cpp
std::any data = quadriga_lib::hdf5_read_dset("channels.hdf5", "noise_figure", 2, 5);
if (quadriga_lib::any_type_id(&data) == 21) // arma::Mat<double>
{
    arma::mat nf = std::any_cast<arma::mat>(data);
    // ...
}
```

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (lists the datasets of a slot)
- [hdf5_write_dset](#hdf5_write_dset) (writes a dataset)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read the names of the unstructured datasets of a slot of an HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_dset_names) · [Python](../python/channel-data-and-processing.md#hdf5_read_dset_names)

- Lists the datasets of slot `(ix, iy, iz, iw)` whose names start with `prefix`, without the prefix, in the order
  they were written

### Declaration:
```cpp
std::vector<std::string> quadriga_lib::hdf5_read_dset_names(
    const std::string &fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    const std::string &prefix = "par_");
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`ix`**, **`iy`**, **`iz`**, **`iw`** *(optional)* — Slot index, 0-based
- **`prefix`** *(optional)* — Prefix of the listed names; `""` lists all datasets with their full names

### Returns:
- Dataset names; empty if the slot is empty or outside the layout or has none

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (reads a dataset by name)
- [hdf5_write_dset](#hdf5_write_dset) (writes a dataset)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of an HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_layout) · [Python](../python/channel-data-and-processing.md#hdf5_read_layout)

- Returns the dimensions of the slot grid and, optionally, the slots that hold data
- A missing file gives `{0, 0, 0, 0}`, so the function also tests whether a file exists; a file that is not a channel
  file throws

### Declaration:
```cpp
arma::u32_vec quadriga_lib::hdf5_read_layout(
    const std::string &fn,
    arma::u32_vec *channel_id = nullptr);
```

### Input:
- **`fn`** — File name

### Output:
- **`channel_id`** *(optional)* — Channel ID of each slot, 0 = never written; slot `(ix, iy, iz, iw)` is entry
  `ix + nx * (iy + ny * (iz + nz * iw))`; `[nx · ny · nz · nw]`; not changed for a missing file

### Returns:
- Layout `{nx, ny, nz, nw}`; `[4]`

### See also:
- [hdf5_create_file](#hdf5_create_file) (creates a file with a given layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (changes the dimensions)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshape the storage layout of an existing HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_reshape_layout) · [Python](../python/channel-data-and-processing.md#hdf5_reshape_layout)

- Sets new dimensions `{nx, ny, nz, nw}` with the slot count `nx · ny · nz · nw` of the file
- The data stays in place: the slot with the linear index `ix + nx * (iy + ny * (iz + nz * iw))` of [hdf5_read_layout](#hdf5_read_layout)
  keeps its channel, which gets the slot index of that position in the new layout

### Declaration:
```cpp
void quadriga_lib::hdf5_reshape_layout(
    const std::string &fn,
    unsigned nx,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

### Inputs:
- **`fn`** — File name of an existing channel file
- **`nx`** — New number of slots along x
- **`ny`**, **`nz`**, **`nw`** *(optional)* — New number of slots along y, z and w

### See also:
- [hdf5_create_file](#hdf5_create_file) (sets the layout of a new file)
- [hdf5_read_layout](#hdf5_read_layout) (reads the layout)

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write channel objects to one or several slots of an HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_write_channel) · [Python](../python/channel-data-and-processing.md#hdf5_write_channel)

- Stores `chan` in slot `(ix, iy, iz, iw)`; the vector form stores `chan_vec[i]` in the slot of column `i` of `slots`
- A missing file is created with the layout that just holds the slots, `(ix + 1) × (iy + 1) × (iz + 1) × (iw + 1)`
  (vector form: the largest index + 1 per dimension); [hdf5_create_file](#hdf5_create_file) sets another layout
- A slot that holds data is replaced completely: its old channel and all its unstructured datasets, also those added
  with [hdf5_write_dset](#hdf5_write_dset), are removed

### Declarations:
```cpp
int quadriga_lib::hdf5_write_channel(
    const std::string &fn,
    const quadriga_lib::channel<dtype> &chan,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0);

arma::s32_vec quadriga_lib::hdf5_write_channel(
    const std::string &fn,
    const std::vector<quadriga_lib::channel<dtype>> &chan_vec,
    const arma::u32_mat &slots);
```

### Inputs:
- **`fn`** — File name
- **`chan`** — Channel to write; valid (`.is_valid()` of [channel](#channel)), at most 4294967295 snapshots; the names in
  `par_names` unique, names and types as in [hdf5_write_dset](#hdf5_write_dset)
- **`ix`**, **`iy`**, **`iz`**, **`iw`** *(optional)* — Slot index, 0-based; inside the layout of an existing file
- **`chan_vec`** — Channels to write (vector form), each as `chan`; `[n_slot]`
- **`slots`** — Slot indices `(ix, iy, iz, iw)` (vector form), 0-based, one column per channel, each slot once,
  inside the layout of an existing file; `[4, n_slot]`

### Returns:
- `0` if the slot was empty, `1` if its data was replaced; vector form: one value per slot, `[n_slot]`

### Notes:
- Vector form: all inputs are checked before anything is written, so a failing check leaves the file unchanged
- A channel without data (`chan.empty()` and no `par_names`) replaces the data of an occupied slot (it then reads as an
  empty channel, and [hdf5_read_layout](#hdf5_read_layout) reports it as written); it leaves an empty slot as it is and does not create a
  missing file
- The space of replaced data is not reused, so the file does not shrink; copying the channels into a new file reclaims
  it
- Files are not locked: one writer per file at a time, also for [hdf5_write_dset](#hdf5_write_dset) and [hdf5_reshape_layout](#hdf5_reshape_layout).
  Reading is safe while no one writes; different files can be read and written from parallel threads

### See also:
- [hdf5_read_channel](#hdf5_read_channel) (reads the channels back)
- [hdf5_create_file](#hdf5_create_file) (creates a file with a given layout)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write an unstructured dataset to a slot of an HDF5 channel file

Also in: [MATLAB](../mex/channel-data-and-processing.md#hdf5_write_dset) · [Python](../python/channel-data-and-processing.md#hdf5_write_dset)

- Adds the dataset `prefix + name` to slot `(ix, iy, iz, iw)`; everything else stored in the slot stays. An empty
  slot gets a channel without structured data
- A missing file is created with the layout `(ix + 1) × (iy + 1) × (iz + 1) × (iw + 1)`; [hdf5_create_file](#hdf5_create_file) sets
  another layout
- A dataset is written once: a dataset of the same name in the slot throws. A later [hdf5_write_channel](#hdf5_write_channel) to the slot
  removes it with the rest of the slot
- Invalid names and types throw before the file is changed
- [hdf5_read_channel](#hdf5_read_channel) returns the datasets with the prefix `par_` as the unstructured data of the channel;
  [hdf5_read_dset](#hdf5_read_dset) reads each one back, an `arma::Row` as `arma::Col`

### Declaration:
```cpp
void quadriga_lib::hdf5_write_dset(
    const std::string &fn,
    const std::string &name,
    const std::any &data,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    const std::string &prefix = "par_");
```

### Inputs:
- **`fn`** — File name
- **`name`** — Dataset name without the prefix, not empty; `prefix + name` holds only letters, digits and `_`, at
  most 65000 characters
- **`data`** — A value of a type of [any_type_id](miscellaneous-tools.md#any_type_id): a string, a scalar, an `arma::Col`, `arma::Row`, `arma::Mat`
  or `arma::Cube`
- **`ix`**, **`iy`**, **`iz`**, **`iw`** *(optional)* — Slot index, 0-based; inside the layout of an existing file
- **`prefix`** *(optional)* — Prefix of the dataset name

### Example:
```cpp
arma::mat nf = {{6.0, 7.0}, {6.5, 7.5}};
quadriga_lib::hdf5_write_dset("channels.hdf5", "noise_figure", nf, 2, 5); // dataset par_noise_figure of slot (2, 5, 0, 0)
```

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (reads a dataset by name)
- [hdf5_read_dset_names](#hdf5_read_dset_names) (lists the datasets of a slot)

---
<a id="qrt_file_append"></a>
## qrt_file_append
Append the paths of one origin to an existing QRT file

Not in: MATLAB, Python

- Writes the paths of one origin (e.g. a Tx) into the next free origin slot of a file of [qrt_file_init](#qrt_file_init), with the
  name, position and orientation of the origin
- A path belongs to the CIR of its index `i_cir`; the paths of one CIR keep their order in `path_data`
- `length` and the counters `n_ref`, `n_tra`, `n_sub`, `n_sct` of [path](site-specific-simulation-tools.md#path) are not stored
  (see [QRT file format: stored values](../../formats/qrt.md#qrt-stored-values))
- All inputs are checked before anything is written; a file whose origin slots are all written, or whose origin table
  does not store all three angles (see [QRT file format: byte layout](../../formats/qrt.md#qrt-byte-layout)), throws
  `std::runtime_error`

### Declaration:
```cpp
size_t quadriga_lib::qrt_file_append(
    const std::string &fn,
    const std::vector<quadriga_lib::path> &path_data,
    const arma::Mat<dtype> &orig_pos,
    const arma::Mat<dtype> &orig_orientation = arma::Mat<dtype>(),
    const std::string &orig_name = "TX");
```

### Inputs:
- **`fn`** — File name of a QRT file of version 5 or 6
- **`path_data`** — Paths of the origin; `i_cir` below `n_cir` of the file, `n_freq()` and the mode
  (`is_scalar()`) as in the file
- **`orig_pos`** — Origin position in m, finite; `[3, 1]` or `[1, 3]`
- **`orig_orientation`** *(optional)* — Origin orientation (bank, tilt, heading) in rad, finite; `[3, 1]` or
  `[1, 3]`; empty = zeros
- **`orig_name`** *(optional)* — Origin name, at most 255 characters

### Returns:
- Number of paths written

### See also:
- [qrt_paths_from_scatterers](#qrt_paths_from_scatterers) (paths from spherical path data)
- [qrt_file_read_raw](#qrt_file_read_raw) (reads the paths back)

---
<a id="qrt_file_init"></a>
## qrt_file_init
Create a new QRT file and write its metadata header

Not in: MATLAB, Python

- Writes the header of a QRT file: the frequencies, the CIR positions and orientations, the destinations and `n_orig`
  empty origin slots; [qrt_file_append](#qrt_file_append) then writes the paths of one origin per call
- An EM file (version 5) stores a polarization transfer matrix per path and frequency, a scalar file (version 6, e.g.
  acoustic) one complex coefficient (see [QRT file format](../../formats/qrt.md))
- All inputs are checked before the file is opened, so an invalid input leaves an existing file unchanged

### Declaration:
```cpp
void quadriga_lib::qrt_file_init(
    const std::string &fn,
    const double_list &center_freq,
    const arma::Mat<dtype> &cir_pos,
    const arma::Mat<dtype> &cir_orientation,
    const std::vector<std::string> &dest_names = {"RX"},
    const uint_list &cir_offset = {0},
    unsigned n_orig = 1,
    bool scalar_mode = false);
```

### Inputs:
- **`fn`** — File name; an existing file is overwritten
- **`center_freq`** — Frequencies in Hz, finite and > 0, at least 1 kHz for an EM file; `[n_freq]`, 1 to 127 entries
- **`cir_pos`** — CIR positions in m, one row per CIR; `[n_cir, 3]`, `n_cir = 0` allowed
- **`cir_orientation`** — CIR orientations (bank, tilt, heading) in rad; `[n_cir, 3]`; empty = zeros
- **`dest_names`** *(optional)* — Destination names, at most 255 characters each; `[n_dest]`; omitted: one destination
  `RX` with `cir_offset` `{0}`
- **`cir_offset`** *(optional)* — Number of CIRs before each destination, i.e. the 0-based index of its first CIR; the
  first entry 0, non-decreasing, at most `n_cir` (a destination may have no CIRs); `[n_dest]`
- **`n_orig`** *(optional)* — Number of origin slots, at least 1
- **`scalar_mode`** *(optional)* — `true`: scalar file (version 6); `false`: EM file (version 5)

### See also:
- [qrt_file_append](#qrt_file_append) (writes the paths of an origin)
- [qrt_file_read](#qrt_file_read) (reads CIRs)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [MATLAB](../mex/channel-data-and-processing.md#qrt_file_parse) · [Python](../python/channel-data-and-processing.md#qrt_file_parse)

- Reads the header of a QRT file: counts, names, frequencies, the positions and orientations of the CIRs and the
  origins, and the file version; no path data
- All outputs are optional (`nullptr` skips one) and must be distinct objects; they are set after the whole header is
  read
- A file without destinations reports one destination `RX` that owns all CIRs
- Throws `std::runtime_error` for a file that cannot be opened, is not a [QRT file](../../formats/qrt.md) of version 5 or 6,
  or is truncated or corrupt

### Declaration:
```cpp
void quadriga_lib::qrt_file_parse(
    const std::string &fn,
    arma::uword *n_cir = nullptr,
    arma::uword *n_orig = nullptr,
    arma::uword *n_dest = nullptr,
    arma::uword *n_freq = nullptr,
    arma::uvec *cir_offset = nullptr,
    std::vector<std::string> *orig_names = nullptr,
    std::vector<std::string> *dest_names = nullptr,
    int *version = nullptr,
    arma::fvec *center_freq = nullptr,
    arma::fmat *cir_pos = nullptr,
    arma::fmat *cir_orientation = nullptr,
    arma::fmat *orig_pos = nullptr,
    arma::fmat *orig_orientation = nullptr,
    std::ifstream *file = nullptr);
```

### Inputs:
- **`fn`** — File name; not used when `file` is given
- **`file`** *(optional)* — Open binary stream of the file in a good state, left open; `nullptr` opens and closes the
  file

### Outputs:
- **`n_cir`** *(optional)* — Number of CIRs of each origin
- **`n_orig`** *(optional)* — Number of origin slots
- **`n_dest`** *(optional)* — Number of destinations, at least 1
- **`n_freq`** *(optional)* — Number of frequencies
- **`cir_offset`** *(optional)* — Number of CIRs before each destination, i.e. the 0-based index of its first CIR;
  `[n_dest]`
- **`orig_names`** *(optional)* — Origin names, empty for a slot not written yet; `[n_orig]`
- **`dest_names`** *(optional)* — Destination names; `[n_dest]`
- **`version`** *(optional)* — File version: 5 (EM) or 6 (scalar)
- **`center_freq`** *(optional)* — Frequencies in Hz; `[n_freq]`
- **`cir_pos`** *(optional)* — CIR positions in m; `[n_cir, 3]`
- **`cir_orientation`** *(optional)* — CIR orientations (bank, tilt, heading) in rad; `[n_cir, 3]`
- **`orig_pos`** *(optional)* — Origin positions in m, zeros for a slot not written yet; `[n_orig, 3]`
- **`orig_orientation`** *(optional)* — Origin orientations (bank, tilt, heading) in rad; `[n_orig, 3]`

### See also:
- [qrt_file_read](#qrt_file_read) (reads CIRs)
- [qrt_read_cache_init](#qrt_read_cache_init) (the same metadata as a cache for repeated reads)

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read the paths of CIRs from a QRT file

Also in: [MATLAB](../mex/channel-data-and-processing.md#qrt_file_read) · [Python](../python/channel-data-and-processing.md#qrt_file_read)

- Reads the paths of a list of CIRs of one origin, with the positions and orientations of both ends: the inputs of
  [get_channels_spherical](channel-generators.md#get_channels_spherical) (`fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `polmat`) or
  [get_channels_planar](channel-generators.md#get_channels_planar) (the angles)
- All outputs are optional (`nullptr` skips one) and must be distinct objects. Per-CIR outputs have one entry per
  requested CIR, positions and orientations one column; the path data is read only when a path output (`fbs_pos` to
  `interact_type`) is requested
- `uplink` swaps the ends: the origin becomes the receiver (see Notes)
- Without a `cache`, each call reads the whole file header first. To read a file in parts (e.g. in chunks of CIRs, to
  bound the memory), pass an open stream and a cache of [qrt_read_cache_init](#qrt_read_cache_init) (Example)

### Declaration:
```cpp
void quadriga_lib::qrt_file_read(
    const std::string &fn,
    const uint_list &i_cir = {},
    arma::uword i_orig = 0,
    bool uplink = false,
    bool normalize_polmat = true,
    arma::Col<dtype> *center_freq = nullptr,
    arma::Mat<dtype> *tx_pos = nullptr,
    arma::Mat<dtype> *tx_orientation = nullptr,
    arma::Mat<dtype> *rx_pos = nullptr,
    arma::Mat<dtype> *rx_orientation = nullptr,
    std::vector<arma::Mat<dtype>> *fbs_pos = nullptr,
    std::vector<arma::Mat<dtype>> *lbs_pos = nullptr,
    std::vector<arma::Mat<dtype>> *path_gain = nullptr,
    std::vector<arma::Col<dtype>> *path_length = nullptr,
    std::vector<arma::Cube<dtype>> *polmat = nullptr,
    std::vector<arma::Col<dtype>> *aod = nullptr,
    std::vector<arma::Col<dtype>> *eod = nullptr,
    std::vector<arma::Col<dtype>> *aoa = nullptr,
    std::vector<arma::Col<dtype>> *eoa = nullptr,
    std::vector<std::vector<arma::Mat<dtype>>> *path_coord = nullptr,
    std::vector<arma::u32_vec> *n_interact = nullptr,
    std::vector<arma::fmat> *coord = nullptr,
    std::vector<std::vector<uint8_t>> *interact_type = nullptr,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

### Inputs:
- **`fn`** — File name; not used when `file` is given
- **`i_cir`** *(optional)* — CIR indices, 0-based, below `n_cir`; empty = all CIRs of the file; `[n_out]`
- **`i_orig`** *(optional)* — Origin index, 0-based, below `n_orig`
- **`uplink`** *(optional)* — `false`: origin = Tx, destination = Rx; `true`: origin = Rx, destination = Tx
- **`normalize_polmat`** *(optional)* — Scaling of `polmat` and `path_gain` (see Notes)
- **`file`** *(optional)* — Open binary stream of the file, left open
- **`cache`** *(optional)* — Metadata of the file from [qrt_read_cache_init](#qrt_read_cache_init)

### Outputs:
- **`center_freq`** *(optional)* — Frequencies in Hz; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in m; `[3, n_out]`
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Tx and Rx orientations (bank, tilt, heading) in rad;
  `[3, n_out]`
- **`fbs_pos`**, **`lbs_pos`** *(optional)* — First- and last-bounce scatterer positions in m: the first and the
  last interaction point, the midpoint of Tx and Rx for a LOS path; `[n_out]` of `[3, n_path]`
- **`path_gain`** *(optional)* — Path gain, linear (see Notes); `[n_out]` of `[n_path, n_freq]`
- **`path_length`** *(optional)* — Length of the polyline from the Tx through the interaction points to the Rx in m;
  `[n_out]` of `[n_path]`
- **`polmat`** *(optional)* — Polarization transfer matrix, `[n_out]` of `[8, n_path, n_freq]`; version 6: the
  complex scalar coefficient, `[n_out]` of `[2, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Azimuth and elevation of departure (at the Tx) and
  arrival (at the Rx) in rad; `[n_out]` of `[n_path]`
- **`path_coord`** *(optional)* — Points of each path from the Tx through the interaction points to the Rx;
  `[n_out]` of `[n_path]` of `[3, n_interact + 2]`
- **`n_interact`** *(optional)* — Number of interaction points of each path, 0 = LOS; `[n_out]` of `[n_path]`
- **`coord`** *(optional)* — Interaction points of all paths, concatenated in path order; `[n_out]` of
  `[3, sum(n_interact)]`
- **`interact_type`** *(optional)* — Interaction type codes ([ray_state_update](site-specific-simulation-tools.md#ray_state_update); GBSM paths: see
  [QRT file format: stored values](../../formats/qrt.md#qrt-stored-values)), concatenated like `coord`; `[n_out]` of
  `[sum(n_interact)]`

### Example:
```cpp
// Read all CIRs of origin 0 in chunks of 256, so that only one chunk is in memory at a time
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);
std::vector<arma::mat> path_gain;
std::vector<arma::vec> path_length;
for (arma::uword i0 = 0; i0 < cache.n_cir; i0 += 256)
{
    arma::uvec i_cir = arma::regspace<arma::uvec>(i0, std::min(i0 + 256, (arma::uword)cache.n_cir) - 1);
    quadriga_lib::qrt_file_read<double>("", i_cir, 0, false, true, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, &path_gain, &path_length, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, &stream, &cache);
    // ... use path_gain[k], path_length[k] of CIR i_cir[k]
}
```

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
- **Errors**: an index out of range throws `std::invalid_argument` before any output is changed. A file that cannot be
  opened, is not a [QRT file](../../formats/qrt.md) of version 5 or 6, or is truncated or corrupt (counts or offsets beyond
  the file, path blocks of different CIRs that overlap; path data only when it is read) throws `std::runtime_error`, and
  outputs may then be partly written. A message about one requested CIR starts with `Snapshot <k>: `, with `k` its
  position in `i_cir`

### See also:
- [qrt_read_cache_init](#qrt_read_cache_init) (cache for reading a file in parts)
- [qrt_file_parse](#qrt_file_parse) (metadata without path data)
- [qrt_file_read_raw](#qrt_file_read_raw) (the stored paths as path objects)

---
<a id="qrt_file_read_raw"></a>
## qrt_file_read_raw
Read the stored paths of one origin of a QRT file as path objects

Not in: MATLAB, Python

- Returns the paths that [qrt_file_append](#qrt_file_append) wrote for origin `i_orig` as [path](site-specific-simulation-tools.md#path) objects, CIR by CIR, each with its
  CIR index `i_cir`, interaction points, coefficients and interaction type codes; [qrt_file_read](#qrt_file_read) returns the same
  data as CIR data
- `length` and the counters `n_ref`, `n_tra`, `n_sub`, `n_sct` of [path](site-specific-simulation-tools.md#path) are not stored and stay 0
- An origin slot that is not written yet gives an empty vector
- To read many origins, pass an open stream and a cache of [qrt_read_cache_init](#qrt_read_cache_init), so that the header is read once
- Throws `std::runtime_error` for file errors as [qrt_file_read](#qrt_file_read)

### Declaration:
```cpp
std::vector<quadriga_lib::path> quadriga_lib::qrt_file_read_raw(
    const std::string &fn,
    arma::uword i_orig = 0,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

### Inputs:
- **`fn`** — File name; not used when `file` is given
- **`i_orig`** *(optional)* — Origin index, 0-based, below `n_orig`
- **`file`** *(optional)* — Open binary stream of the file in a good state, left open; `nullptr` opens and closes the
  file
- **`cache`** *(optional)* — Metadata of the file from [qrt_read_cache_init](#qrt_read_cache_init)

### Returns:
- Paths of the origin in CIR order; `[n_path]`

### See also:
- [qrt_file_append](#qrt_file_append) (writes the paths)
- [qrt_file_read](#qrt_file_read) (reads the paths as CIR data)

---
<a id="qrt_paths_from_scatterers"></a>
## qrt_paths_from_scatterers
Path objects of one CIR from spherical path data, for qrt_file_append

Not in: MATLAB, Python

- Converts the paths of one Tx-Rx pair (e.g. of [ssf_paths](geometry-based-stochastic-model.md#ssf_paths) with `drift` mobility) into [path](site-specific-simulation-tools.md#path) objects for
  [qrt_file_append](#qrt_file_append); one call per CIR, the objects of all CIRs of one origin appended together
- Interactions: LOS none; GR one reflection at `lbs_pos` (code 129, `n_ref` = 1); NLOS two point scatterers at
  `fbs_pos`, `lbs_pos` (code 193, `n_sct` = 2); `length` = polyline from the Tx to the last interaction. Counters
  and `length` exist on the returned objects only, files store the codes ([qrt_file_read_raw](#qrt_file_read_raw) returns 0)
- The path gain and the inverse free-space loss of the polyline are folded into the stored Jones matrices, so that
  [qrt_file_read](#qrt_file_read) (`uplink = false`, either `normalize_polmat`) returns sqrt(`path_gain`)·`polmat` per complex entry
  within float32 accuracy, the polyline as `path_length` and `path_type` as `n_interact`
- `tx_pos` must be the `orig_pos` of [qrt_file_append](#qrt_file_append) and `rx_pos` the CIR position of [qrt_file_init](#qrt_file_init), since
  [qrt_file_read](#qrt_file_read) recomputes the lengths from the stored positions; not checked
- Float32 storage: use local coordinates, shifted in x and y only (ground at z = 0); errors: see
  [QRT file format: precision](../../formats/qrt.md#qrt-precision)
- Memory: 80 to 105 bytes per path object at one frequency, e.g. 3 to 4 GB for 1000 UTs, 100 snapshots, 381 rays
- Throws for a path of length 0 in float32 (e.g. LOS with `tx_pos` = `rx_pos`) and for values beyond float32

### Declaration:
```cpp
std::vector<quadriga_lib::path> quadriga_lib::qrt_paths_from_scatterers(
    const double_list &center_freq,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::mat &fbs_pos,
    const arma::mat &lbs_pos,
    const arma::mat &path_gain,
    const arma::cube &polmat,
    const arma::u32_vec &path_type,
    arma::uword i_cir = 0);
```

### Inputs:
- **`center_freq`** — Frequencies in Hz, finite and > 0, 1 to 127 entries; `[n_freq]`
- **`tx_pos`** — Tx (origin) position in m, finite; `[3, 1]` or `[1, 3]`
- **`rx_pos`** — Rx position of the CIR in m, finite; `[3, 1]` or `[1, 3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m, finite (GR: `lbs_pos`);
  `[3, n_path]`
- **`path_gain`** — Path gains, linear, finite and >= 0; `[n_path, n_freq]`
- **`polmat`** — Polarization transfer matrices, finite; `[8, n_path, n_freq]`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `[n_path]`
- **`i_cir`** *(optional)* — CIR index, 0-based, stored in `path::i_cir`, < 2^32

### Returns:
- Path objects in input order, EM layout with `n_freq` frequencies; `[n_path]`

### See also:
- [qrt_file_append](#qrt_file_append) (writes the paths)
- [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) (ray data from the paths read back)

---
<a id="qrt_read_cache_init"></a>
## qrt_read_cache_init
Read the metadata of a QRT file into a cache for repeated reads

Not in: MATLAB, Python

- Reads the header of a QRT file once and precomputes the byte offsets of the path data, so that [qrt_file_read](#qrt_file_read) and
  [qrt_file_read_raw](#qrt_file_read_raw) with this cache and an open stream read only the requested data, 2 seeks and 4 reads per CIR
- Use it to read a file in parts, e.g. in chunks of CIRs or one origin after another (Example of [qrt_file_read](#qrt_file_read))
- The cache describes the file as it was read: create a new one after [qrt_file_append](#qrt_file_append)
- Throws `std::runtime_error` for a file that cannot be opened, is not a [QRT file](../../formats/qrt.md) of version 5 or 6,
  or is truncated or corrupt (counts or offsets beyond the file, two origins with one data block)

### Declaration:
```cpp
quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(
    const std::string &fn,
    std::ifstream *file = nullptr);
```

### Inputs:
- **`fn`** — File name; not used when `file` is given
- **`file`** *(optional)* — Open binary stream of the file in a good state, left open; `nullptr` opens and closes the
  file

### Returns:
- The cache, a `quadriga_lib::qrt_read_cache` with the members:
  | Member | Type | Description |
  | ------ | ---- | ----------- |
  | `version` | `int` | File version, 5 or 6 |
  | `n_orig`, `n_cir`, `n_dest`, `n_freq` | `unsigned` | Numbers of origin slots, CIRs, destinations (0 if the file lists none) and frequencies |
  | `freq` | `arma::fvec` | Frequencies in Hz as [qrt_file_parse](#qrt_file_parse) returns them; `[n_freq]` |
  | `cir_pos`, `cir_orientation` | `arma::fmat` | CIR positions in m and orientations in rad; `[n_cir, 3]` |
  | `orig_pos_all`, `orig_orientation` | `arma::fmat` | Origin positions in m and orientations in rad; `[n_orig, 3]` |
  | `orig_index` | `arma::u64_vec` | Byte offset of the data block of each origin, 0 = slot not written yet; `[n_orig]` |
  | `path_data_offset` | `arma::u64_vec` | Byte offset of the CIR index table of each origin, 0 = slot not written yet; `[n_orig]` |

### See also:
- [qrt_file_read](#qrt_file_read) (reads CIRs with the cache)
- [qrt_file_read_raw](#qrt_file_read_raw) (reads the paths of an origin with the cache)

---
<a id="quantize_delays"></a>
## quantize_delays
Map the path delays to a tap grid by two-tap interpolation

Also in: [MATLAB](../mex/channel-data-and-processing.md#quantize_delays) · [Python](../python/channel-data-and-processing.md#quantize_delays)

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

### Declaration:
```cpp
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> &coeff_re,
    const std::vector<arma::Cube<dtype>> &coeff_im,
    const std::vector<arma::Cube<dtype>> &delay,
    std::vector<arma::Cube<dtype>> &coeff_re_quant,
    std::vector<arma::Cube<dtype>> &coeff_im_quant,
    std::vector<arma::Cube<dtype>> &delay_quant,
    double tap_spacing = 5.0e-9,
    arma::uword max_taps = 48,
    double power_exponent = 1.0,
    int fix_taps = 0);
```

### Inputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_snap]` cubes
  `[n_rx, n_tx, n_path_s]`, at least one; `n_path_s` may differ between the snapshots, `n_rx` and `n_tx` (>= 1) not
- **`delay`** — Delays in s; `[n_snap]` cubes `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`, one of the two forms for
  all snapshots; finite, above `-tap_spacing / 2` and below `2^31` · `tap_spacing`
- **`tap_spacing`** *(optional)* — Tap spacing in s, > 0 and finite; 5 ns is a sampling rate of 200 MHz
- **`max_taps`** *(optional)* — Largest number of taps of a grid; 0 = unlimited
- **`power_exponent`** *(optional)* — Interpolation exponent $\alpha$, > 0
- **`fix_taps`** *(optional)* — Which coefficients share a grid:

  | Value | Grid |
  | ----- | ---- |
  | 0 | One grid per antenna pair and snapshot |
  | 1 | One grid for all antenna pairs and snapshots |
  | 2 | One grid per snapshot for all its antenna pairs; the grids of the snapshots differ |
  | 3 | One grid per antenna pair for all snapshots |

### Outputs:
- **`coeff_re_quant`** / **`coeff_im_quant`** — Real and imaginary part of the coefficients per tap; `[n_snap]` cubes
  `[n_rx, n_tx, n_taps]`
- **`delay_quant`** — Tap delays in s; `[n_snap]` cubes `[n_rx, n_tx, n_taps]`, or `[1, 1, n_taps]` for delays
  `[1, 1, n_path_s]` with `fix_taps` 1 or 2

### Notes:
- Negative delays: delays in `(-tap_spacing / 2, 0)` count as 0, all their power goes to tap 0 (the error is below
  `tap_spacing / 2`, as for rounding to the nearest tap)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](channel-generators.md#get_channels_spherical), [get_channels_planar](channel-generators.md#get_channels_planar) and
  [get_channels_multifreq](channel-generators.md#get_channels_multifreq)) are negative for element pairs that are closer to each other than the phase centres,
  by up to (half the Tx array size + half the Rx array size) / propagation speed. They stay above
  `-tap_spacing / 2` only while these half sizes add up to less than about 0.75 m for radio at 5 ns taps, or 3.6 mm
  for sound at 48 kHz taps, i.e. practically never for acoustic arrays: use absolute delays then
- The upper limit of `2^31` taps is e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz; run time and memory depend on the
  number of paths and taps, not on the delay values

### See also:
- [freq_response](#freq_response) (frequency response of the paths or taps)
- [calc_delay_spread](#calc_delay_spread) (delay spread, with an optional delay granularity)
