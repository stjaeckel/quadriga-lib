# Channel data and processing

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-channel-data-and-processing

| Function | Summary |
| --- | --- |
| [calc_angular_spread](#calc_angular_spread) | Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths |
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
| [qrt_file_parse](#qrt_file_parse) | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Read the paths of CIRs from a QRT file |
| [quantize_delays](#quantize_delays) | Map the path delays to a tap grid by two-tap interpolation |

A **channel** describes the propagation between a transmit and a receive antenna array: for each snapshot, a set of
propagation paths, each with a complex coefficient per antenna pair and a delay. The functions of this section store
channels in files, turn them into frequency responses, prepare them for link-level simulation, compute their statistics
and show the paths in 3D. The channels themselves come from the functions of section "Channel generators" or from
[get_channels_layout](network-layouts-and-mobility.md#get_channels_layout).

**What a channel holds** (a dict, keys in [hdf5_read_channel](#hdf5_read_channel)):

- The positions and orientations of the transmitter and the receiver, per snapshot or fixed
- The complex **coefficients** `coeff` and the **delays** `delay` of each snapshot, `(n_rx, n_tx, n_path)`
- Optionally, antenna-independent path data: path gain, path length, polarization transfer matrix, angles, scatterer
  positions and the interaction points of ray-traced paths
- Unstructured metadata: named values of any type

**In Python:**

- The functions are in the submodule `quadriga_lib.channel`, e.g. `quadriga_lib.channel.hdf5_read_channel`
- A channel is a dict with the member names of the C++ class `channel`, with one complex `coeff` for `coeff_re` and
  `coeff_im`; several channels are a list of dicts. A member with one entry per snapshot is a list of arrays, one per
  snapshot, or, with `stack = True` of [hdf5_read_channel](#hdf5_read_channel), one array with the snapshot on the last axis,
  zero-padded where the number of paths differs
- [hdf5_read_channel](#hdf5_read_channel) and [hdf5_write_channel](#hdf5_write_channel) take and return the unstructured data as a list `par` with one
  dict per channel, one key per dataset

**Typical workflow:**

1. Generate the channels, or read them: [hdf5_read_channel](#hdf5_read_channel) from a channel file, [qrt_file_read](#qrt_file_read) from the output
   of the QRT ray tracer
2. Optionally, map the delays to a tap grid with [quantize_delays](#quantize_delays), or join the segments of a track with
   [crossfade_segments](#crossfade_segments)
3. Compute the frequency response at the subcarriers with [freq_response](#freq_response), or with [freq_response_mf](#freq_response_mf) for
   coefficients sampled at several frequencies
4. Evaluate the channels: [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread), [calc_rician_k_factor](#calc_rician_k_factor), [calc_xpr](#calc_xpr)
5. Store them with [hdf5_write_channel](#hdf5_write_channel), or show the paths with [export_channel_obj](#export_channel_obj)

**Terms** (full list in the [Glossary](../cpp/general-usage-notes.md#glossary)):

- **Snapshot**: one sample of the channel in time or position; `n_snap` snapshots, each with its own number of paths
- **Path**: one propagation path from the transmitter to the receiver, e.g. the LOS path, a reflection or a ray of a
  cluster
- **Coefficient**: the complex amplitude $a$ of one path for one antenna pair, with the antenna patterns, the
  polarization and the path gain
- **Delay**: the propagation delay $\tau$ of a path in s, per antenna pair `(n_rx, n_tx, n_path)` or shared by all
  pairs `(1, 1, n_path)`
- **CIR**: the channel impulse response of one snapshot, its coefficients and delays
- **Frequency response**: the channel matrix at subcarrier $k$, $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$, with
  $f_k$ the offset of the subcarrier from the carrier (the absolute frequency for `_mf`)
- **Multi-frequency channel** (suffix `_mf`): coefficients sampled at several frequencies, the frequency on the 4th
  axis, as [channel.multifreq](channel-generators.md#channel.multifreq) produces them
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
| Path files (QRT) | [qrt_file_parse](#qrt_file_parse), [qrt_file_read](#qrt_file_read) |
| Frequency response | [freq_response](#freq_response), [freq_response_mf](#freq_response_mf) |
| Process | [quantize_delays](#quantize_delays), [crossfade_segments](#crossfade_segments) |
| Statistics | [calc_delay_spread](#calc_delay_spread), [calc_angular_spread](#calc_angular_spread), [calc_rician_k_factor](#calc_rician_k_factor), [calc_xpr](#calc_xpr) |
| Show | [export_channel_obj](#export_channel_obj) |

**Conventions.** Coefficients and delays of several snapshots are lists with one array per snapshot. Indices are
0-based.

---
<a id="calc_angular_spread"></a>
## calc_angular_spread
Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths

Also in: [C++](../cpp/channel-data-and-processing.md#calc_angular_spread) · [MATLAB](../mex/channel-data-and-processing.md#calc_angular_spread)

- For one angle, with the path powers $p_l$ normalized to sum 1 over the paths of a CIR (the second-moment definition
  of 3GPP TR 38.901):

  $$\sigma = \sqrt{\sum_l p_l \, \Delta_l^2}, \quad \Delta_l = \arg e^{j (\phi_l - \bar\phi)}$$

  with the circular mean $\bar\phi = \arg \sum_l p_l \, e^{j \phi_l}$
- `wrapping = False`: the spreads of the given azimuth and elevation angles; near the poles, the azimuth spread
  overstates the angular spread
- `wrapping = True`: the paths are first rotated so that their power-weighted mean direction (the mean of the unit
  vectors) lies at azimuth 0 on the equator; `calc_bank_angle` then also rotates them about that direction so that
  the azimuth spread is largest (principal axis of the power-weighted covariance of the rotated angles); `orientation`
  is the direction of an antenna that sees the paths at the rotated angles `phi` and `theta`
- `quantize_deg` > 0: the deviations $\Delta_l$ are rounded to multiples of `quantize_deg` and the powers of equal
  deviations summed, for azimuth and elevation separately

### Usage:
```python
azimuth_spread, elevation_spread, orientation, phi, theta = quadriga_lib.channel.calc_angular_spread( \
    az, el, powers, *, wrapping, calc_bank_angle, quantize_deg )
```

### Inputs:
- **`az`** — Path azimuths in rad; list of `n_cir` arrays `(n_path,)`, at least one CIR and one path per CIR; `n_path`
  may differ between the CIRs
- **`el`** — Path elevations in rad; list of `n_cir` arrays `(n_path,)`
- **`powers`** — Path powers, linear, with a positive sum per CIR; list of `n_cir` arrays `(n_path,)`
- **`wrapping`** — Rotate the paths to the equator before computing the spreads; default: `False`
- **`calc_bank_angle`** — Rotate them also about the mean direction, for the largest azimuth
  spread; only with `wrapping`; default: `False`
- **`quantize_deg`** — Quantization step of the angle deviations in deg; 0 = none, else finite and >= 1e-3; default: 0.0

### Outputs:
- **`azimuth_spread`** — RMS azimuth spread in rad; `(n_cir,)`
- **`elevation_spread`** — RMS elevation spread in rad; `(n_cir,)`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad of an antenna pointing at the mean
  direction (a positive tilt points down), zero without `wrapping`; `(3, n_cir)`
- **`phi`** — Path azimuths in rad in the frame of that antenna (as [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern)
  returns them), the input `az` without `wrapping`; list of `n_cir` arrays `(n_path,)`
- **`theta`** — Path elevations in rad in that frame, the input `el` without `wrapping`;
  list of `n_cir` arrays `(n_path,)`

### See also:
- [calc_delay_spread](#calc_delay_spread) (delay spread of the same paths)
- [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern) (antenna response in the rotated frame)

---
<a id="calc_delay_spread"></a>
## calc_delay_spread
Calculate the RMS delay spread of each CIR from its path delays and powers

Also in: [C++](../cpp/channel-data-and-processing.md#calc_delay_spread) · [MATLAB](../mex/channel-data-and-processing.md#calc_delay_spread)

- With the path powers $p_l$ normalized to sum 1 over the paths of a CIR, the mean delay and the RMS delay spread are

  $$\bar\tau = \sum_l p_l \, \tau_l, \qquad \sigma_\tau = \sqrt{\sum_l p_l \, (\tau_l - \bar\tau)^2}$$

- Paths more than `threshold_dB` below the strongest path of their CIR are left out
- With `granularity` > 0, the delays are first rounded to the nearest multiple of `granularity`, and the powers of
  the paths with the same rounded delay are summed
- A CIR without power gives a spread and a mean delay of 0

### Usage:
```python
ds, mean_delay = quadriga_lib.channel.calc_delay_spread( delays, powers, *, threshold_dB, granularity )
```

### Inputs:
- **`delays`** — Path delays in s; list of `n_cir` arrays `(n_path,)`, at least one CIR and one path per CIR; `n_path`
  may differ between the CIRs
- **`powers`** — Path powers, linear; list of `n_cir` arrays `(n_path,)`
- **`threshold_dB`** — Power range in dB below the strongest path that is kept, > 0; default: 100.0
- **`granularity`** — Spacing of the delay grid in s, finite and >= 0; 0 = no grid; with a grid, all
  delays must be finite and their span must be below $2^{53}$ grid steps; default: 0.0

### Outputs:
- **`ds`** — RMS delay spread $\sigma_\tau$ in s; `(n_cir,)`
- **`mean_delay`** — Mean delay $\bar\tau$ in s; `(n_cir,)`

### See also:
- [quantize_delays](#quantize_delays) (delays on a tap grid with interpolation)
- [calc_angular_spread](#calc_angular_spread) (angular spreads)

---
<a id="calc_rician_k_factor"></a>
## calc_rician_k_factor
Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power

Also in: [C++](../cpp/channel-data-and-processing.md#calc_rician_k_factor) · [MATLAB](../mex/channel-data-and-processing.md#calc_rician_k_factor)

- $K = P_\mathrm{LOS} / P_\mathrm{NLOS}$ with the sums of the path powers; LOS paths are those with a length
  `path_length` <= $d_\mathrm{TR}$ + `window_size`, $d_\mathrm{TR}$ the Tx-Rx distance, all others are NLOS
- Without NLOS power, $K$ is infinite, or 0 if the LOS power is zero too (also for a CIR without paths)

### Usage:
```python
kf, pg = quadriga_lib.channel.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, *, window_size )
```

### Inputs:
- **`powers`** — Path powers, linear; list of `n_cir` arrays `(n_path,)`, at least one CIR; `n_path` may differ
  between the CIRs
- **`path_length`** — Path lengths in m; list of `n_cir` arrays `(n_path,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `(3,)` (fixed) or `(3, n_cir)`
- **`window_size`** — LOS window in m, >= 0; default: 0.01

### Outputs:
- **`kf`** — K-factor, linear; `(n_cir,)`
- **`pg`** — Sum of all path powers, linear; `(n_cir,)`

### See also:
- [calc_xpr](#calc_xpr) (XPR with the same LOS window)
- [calc_delay_spread](#calc_delay_spread) (delay spread of the same paths)

---
<a id="calc_xpr"></a>
## calc_xpr
Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis

Also in: [C++](../cpp/channel-data-and-processing.md#calc_xpr) · [MATLAB](../mex/channel-data-and-processing.md#calc_xpr)

- Total-power ratio: the co- and cross-polar powers of the paths of a CIR, weighted by the path powers $p_l$, are
  summed, and the XPR is their ratio; LOS paths are left out unless `include_los` is set
- Entries of the polarization transfer matrix $M$ of a path, first index receive, second transmit polarization:
  rows 0-1 of `polmat` = $M_{vv}$, 2-3 = $M_{hv}$ (H rx, V tx), 4-5 = $M_{vh}$ (V rx, H tx), 6-7 = $M_{hh}$, each as
  real and imaginary part
- Circular basis: $M_{rt} = \mathbf{r}^T M \, \mathbf{t}$ (plain transpose), as [channel.spherical](channel-generators.md#channel.spherical)
  couples antenna patterns, with the same vectors at both ends, each in the $(e_\theta, e_\phi)$ frame of its antenna:
  $\mathbf{L} = (1, +j) / \sqrt{2}$ (LHCP) and $\mathbf{R} = (1, -j) / \sqrt{2}$ (RHCP) in (V, H), which gives

  $$M_{LL}, M_{RR} = \frac{1}{2} \left( M_{vv} - M_{hh} \pm j (M_{vh} + M_{hv}) \right)$$

  $$M_{LR}, M_{RL} = \frac{1}{2} \left( M_{vv} + M_{hh} \pm j (M_{hv} - M_{vh}) \right)$$

  (upper sign for the first entry)

- The names L and R follow the IEEE definition (rotation seen along the propagation direction) for the time
  convention $e^{+j 2\pi f t}$, which the coefficients $e^{-j 2\pi f \tau}$ of the library imply, with
  $e_\theta \times e_\phi$ the outward direction of the pattern
- Co-polar means the same circular vector at both ends: the LOS matrix $\mathrm{diag}(1, -1)$ of
  [channel.spherical](channel-generators.md#channel.spherical) is co-polar (handedness kept), $\mathrm{diag}(1, 1)$ (one mirrored component, as
  by a single reflection) is cross-polar

### Usage:
```python
xpr, pg = quadriga_lib.channel.calc_xpr( powers, polmat, path_length, tx_pos, rx_pos, *, include_los, \
    window_size )
```

### Inputs:
- **`powers`** — Path powers $p_l$, linear; list of `n_cir` arrays `(n_path,)`; `n_path` may differ between the CIRs
- **`polmat`** — Polarization transfer matrices, magnitudes below about 1e18; list of `n_cir` arrays `(8, n_path)`
- **`path_length`** — Path lengths in m; list of `n_cir` arrays `(n_path,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `(3,)` (fixed) or `(3, n_cir)`
- **`include_los`** — Include the LOS paths in the XPR; default: `False`
- **`window_size`** — LOS window in m, >= 0: paths with `path_length` <= $d_\mathrm{TR}$ +
  `window_size` are LOS, $d_\mathrm{TR}$ the Tx-Rx distance; default: 0.01

### Outputs:
- **`xpr`** — XPR, linear; `(n_cir, 6)`; columns:

  | Column | XPR |
  | ------ | --- |
  | 0 | Linear: $(P_{vv} + P_{hh}) / (P_{vh} + P_{hv})$ with $P_{rt} = \sum_l p_l \lvert M_{rt} \rvert^2$ |
  | 1 | V (receive V): $P_{vv} / P_{vh}$ |
  | 2 | H (receive H): $P_{hh} / P_{hv}$ |
  | 3 | Circular: $(P_{LL} + P_{RR}) / (P_{LR} + P_{RL})$ |
  | 4 | LHCP (receive L): $P_{LL} / P_{LR}$ |
  | 5 | RHCP (receive R): $P_{RR} / P_{RL}$ |

- **`pg`** — Path gain of all paths, LOS included,
  $\frac{1}{2} \sum_l p_l \sum_{r,t \in \{v,h\}} \lvert M_{rt} \rvert^2$; `(n_cir,)`

### Notes:
- A ratio with zero cross-polar power is infinite, or 0 if the co-polar power is zero too (also for a CIR without
  paths, which has `pg` = 0)
- A scaling of `polmat` does not change the XPR, but `pg`

### See also:
- [calc_rician_k_factor](#calc_rician_k_factor) (LOS and NLOS power with the same LOS window)
- [channel.spherical](channel-generators.md#channel.spherical) (polarization transfer matrices of the paths)

---
<a id="crossfade_segments"></a>
## crossfade_segments
Join the channels of consecutive segments of one link by a cross-fade

Also in: [C++](../cpp/channel-data-and-processing.md#crossfade_segments) · [MATLAB](../mex/channel-data-and-processing.md#crossfade_segments)

- Joins the "rays" channels of the segments of one link, e.g. of a track whose scenario or indoor state changes, into
  one channel; consecutive segments share the snapshots of a ramp, over which the clusters of the old segment are
  exchanged for those of the new one
- [get_channels_from_rays](network-layouts-and-mobility.md#get_channels_from_rays) joins the segments of tracks with this function, the ramp centred on the row of the change

### Usage:
```python
channel = quadriga_lib.channel.crossfade_segments( segments, *, n_overlap, output, par, stack )
```

### Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients and path gains (`"rays"`), the same
  number of paths in every snapshot, equal `n_rx`, `n_tx` and `center_freq` (at most one value); list of channel
  dicts, keys in [hdf5_read_channel](#hdf5_read_channel)
- **`n_overlap`** — Number of snapshots at the start of each segment that are also the last snapshots of the preceding
  segment (computed at the same positions); they form the ramp. The first entry is 0, at most the snapshots of the
  preceding segment after its own shared ones; `None`: the snapshots whose Tx and Rx positions match (within 1 µm), an
  error where the positions repeat (e.g. ends that do not move); `(n_segment,)`; default: None
- **`output`** — `"taps"` (the paths summed per tap index) or `"rays"`; default: `"taps"`
- **`par`** — Unstructured data of the segments: list of dicts, one per segment, with the keys `tap`, `path_type` and
  `cluster` (uint32, one value per path, see Notes); a list that is given must have one of these keys (other keys are
  ignored); `None` = all paths NLOS, each its own cluster; default: None
- **`stack`** — Snapshot layout of the channel as in [hdf5_read_channel](#hdf5_read_channel); default: `False`

### Outputs:
- **`channel`** — The joined channel, dict, keys in [hdf5_read_channel](#hdf5_read_channel)

### Notes:
- **Snapshots**: segment $k$ starts with the `n_overlap[k]` snapshots of the ramp, which are also the last ones of
  segment $k - 1$; the output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots. Outside the
  ramps the snapshots are those of one segment
- **Paths**: entries of `par`: `path_type` gives the LOS and GR paths (without it, tap index 0 of `tap` and a tap index
  3 held by one path, without index 4 or 5, as [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) numbers them), `cluster` the clusters, whose paths fade
  together (without it: the taps, without `tap`: each path)
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
  `lbs_pos` where all segments have them; `par`, `n_interact` and `interact_coord` are not carried
- **`par`**: the C++ channels carry these entries in their `par_data`; in Python they are the input `par`, as in
  [hdf5_write_channel](#hdf5_write_channel), with the entries of each segment as [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) returns them

### See also:
- [get_channels_from_rays](network-layouts-and-mobility.md#get_channels_from_rays) (joins the segments of tracks with this function)
- [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) (channels of one segment)

---
<a id="export_channel_obj"></a>
## export_channel_obj
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/channel-data-and-processing.md#export_channel_obj) · [MATLAB](../mex/channel-data-and-processing.md#export_channel_obj)

- Writes the paths of ray-traced channels as tubes to an OBJ file and their colors to an MTL file of the same name
  (`.mtl` instead of `.obj`), e.g. for Blender; existing files are overwritten
- Path gain in dB: with coefficients the power summed over all antenna pairs, else from `polmat` (sum of the squared
  entries) with the free-space gain of the length of the drawn path
- The gain, clipped to `[gain_min_dB, gain_max_dB]`, sets the color (64 steps of `colormap`) and the radius (linear
  from 0 at `gain_min_dB` to `radius_max` at `gain_max_dB`, at least `radius_min`); a path with a radius up to
  0.1 mm is written as a polyline
- Paths below `gain_min_dB` and paths with non-finite coordinates are left out

### Usage:
```python
quadriga_lib.channel.export_channel_obj( fn, chan, *, max_paths, gain_max_dB, gain_min_dB, colormap, \
    i_snap, radius_max, radius_min, n_edges )
```

### Inputs:
- **`fn`** — Path of the OBJ file, ending in `.obj`; str or path-like
- **`chan`** — Channel; dict with the keys of [hdf5_read_channel](#hdf5_read_channel), list or stacked layout; needs `rx_pos`,
  `tx_pos`, `center_freq`, `n_interact`, `interact_coord` and `coeff` with `delay`, or `polmat`; in the stacked
  layout, columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Number of the strongest paths per snapshot to export; 0 = all; default: 0
- **`gain_max_dB`** — Gain in dB of the largest radius and the last color, finite; default: -60.0
- **`gain_min_dB`** — Gain in dB of the first color, finite and below `gain_max_dB`; weaker paths are left out;
  default: -140.0
- **`colormap`** — Name of the colormap: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer;
  default: `"jet"`
- **`i_snap`** — 0-based indices of the snapshots to export; `(n,)`; default: `None` (all snapshots)
- **`radius_max`** — Tube radius in m at `gain_max_dB`, >= 0; default: 0.05
- **`radius_min`** — Smallest tube radius in m, >= 0; at most `radius_max` (a larger value is reduced to it);
  default: 0.01
- **`n_edges`** — Vertices of the tube cross-section, >= 3; default: 5

---
<a id="freq_response"></a>
## freq_response
Frequency response of a MIMO channel at the given subcarriers

Also in: [C++](../cpp/channel-data-and-processing.md#freq_response) · [MATLAB](../mex/channel-data-and-processing.md#freq_response)

- The frequency response of one snapshot at the subcarriers $k$: $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$ with the
  offset $f_k$ = `bandwidth` · `pilot_grid[k]` from the carrier
- The snapshots are list items, computed in parallel; the number of paths may differ between them
- For a multi-frequency channel and absolute carrier frequencies, use [freq_response_mf](#freq_response_mf)
- Precision: the phase is reduced to one period in double precision; sin / cos, coefficients and sums are single
  precision

### Usage:
```python
hmat = quadriga_lib.channel.freq_response( coeff, delay, pilot_grid, bandwidth, *, i_snap, use_kernel )
```

### Inputs:
- **`coeff`** — Complex coefficients; list of `n_snap` arrays `(n_rx, n_tx, n_path)`
- **`delay`** — Delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`
- **`pilot_grid`** — Subcarrier positions relative to the bandwidth, finite: 0 = carrier, 1 = carrier + `bandwidth`;
  `(n_carrier,)`; `np.linspace(0, 1, n)` gives `n` evenly spaced carriers
- **`bandwidth`** — Bandwidth in Hz, finite and >= 0
- **`i_snap`** — 0-based indices of the snapshots to compute; `(n_out,)`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (raises if
  unavailable); default: 0

### Output:
- **`hmat`** — Complex frequency response; complex128; `(n_rx, n_tx, n_carrier, n_out)`; the selected snapshots must
  have equal `n_rx` and `n_tx`

### See also:
- [freq_response_mf](#freq_response_mf) (multi-frequency channel, absolute carrier frequencies)
- [channel.spherical](channel-generators.md#channel.spherical) (coefficients and delays)
- [quantize_delays](#quantize_delays) (delays on a tap grid, for time-domain processing)

---
<a id="freq_response_mf"></a>
## freq_response_mf
Frequency response of a MIMO channel from coefficients sampled at several frequencies

Also in: [C++](../cpp/channel-data-and-processing.md#freq_response_mf) · [MATLAB](../mex/channel-data-and-processing.md#freq_response_mf)

- The frequency response of one snapshot of a multi-frequency channel at the absolute carrier frequencies
  $f_k$ = `freq_new[k]`: the coefficients of each path are interpolated from the input frequencies `freq` to $f_k$
  and the delay phase is applied there, $H_k = \sum_l \tilde a_l(f_k) \, e^{-j 2\pi f_k \tau_l}$
- Interpolation between the two neighbouring input frequencies: amplitude linear, phase along the shorter arc
  (SLERP); constant outside the range of `freq`
- `remove_delay_phase = True` interpolates the envelope $a_l(f_i) \, e^{j 2\pi f_i \tau_l}$ instead of $a_l(f_i)$, so
  that the delay phase the channel generators include is not interpolated
- The snapshots are list items, computed in parallel; the number of paths may differ between them

### Usage:
```python
hmat = quadriga_lib.channel.freq_response_mf( coeff, delay, freq, freq_new, *, remove_delay_phase, \
    i_snap, use_kernel )
```

### Inputs:
- **`coeff`** — Complex coefficients; list of `n_snap` arrays `(n_rx, n_tx, n_path, n_freq)`, for `n_freq = 1` also
  `(n_rx, n_tx, n_path)`
- **`delay`** — Delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_path)` or `(1, 1, n_path)` (a 4th dimension of
  length `n_freq` is allowed, only the first frequency is used)
- **`freq`** — Input frequencies in Hz, finite, strictly ascending; `(n_freq,)`
- **`freq_new`** — Output carrier frequencies in Hz, finite; `(n_carrier,)`
- **`remove_delay_phase`** — Remove the delay phase from the coefficients before the interpolation: `True` for the
  coefficients of [channel.multifreq](channel-generators.md#channel.multifreq) and [channel.spherical](channel-generators.md#channel.spherical), `False` for coefficients without delay phase;
  default: `True`
- **`i_snap`** — 0-based indices of the snapshots to compute; `(n_out,)`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

### Output:
- **`hmat`** — Complex frequency response; complex128; `(n_rx, n_tx, n_carrier, n_out)`; the selected snapshots must
  have equal `n_rx` and `n_tx`

### Notes:
- With `remove_delay_phase = True`, generate the channels with `use_absolute_delays = True`: their coefficients carry
  the phase of the absolute path length also when `delay` holds relative delays. With relative delays the phase of
  the Tx-Rx distance stays in the coefficients and is interpolated across `freq`, which gives wrong phases (the
  magnitudes stay right) unless the spacing of `freq` is far below $c$ / (Tx-Rx distance)
- One input frequency: the coefficients apply at all carriers; with `remove_delay_phase = True` the result is that of
  [freq_response](#freq_response) with `bandwidth` · `pilot_grid` = `freq_new - freq`
- Precision: the delay phase is reduced to one period in double precision (and removed in double precision with
  `remove_delay_phase`); sin / cos of the output phase and the interpolation are single
  precision, the sums double precision

### See also:
- [freq_response](#freq_response) (single-frequency coefficients, carriers relative to the bandwidth)
- [channel.multifreq](channel-generators.md#channel.multifreq) (multi-frequency channel)

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a given storage layout

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_create_file) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_create_file)

- Creates a channel file with `nx · ny · nz · nw` empty slots; the slot count is fixed, [hdf5_reshape_layout](#hdf5_reshape_layout) changes
  only the dimensions
- [hdf5_write_channel](#hdf5_write_channel) and [hdf5_write_dset](#hdf5_write_dset) create a missing file themselves, with the layout that just holds their
  slots; this function sets any other one
- Standard HDF5 readers open the file, see [HDF5 channel file format](../../formats/hdf5_channel.md); change it only with
  the functions of this section

### Usage:
```python
quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — File name, str or path-like; the file must not exist
- **`nx`** — Number of slots along x, at least 1; default: 65536
- **`ny`**, **`nz`**, **`nw`** — Number of slots along y, z and w, each at least 1; at most 4294967295 (2^32 - 1)
  slots in total; default: 1

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (writes channels into slots)
- [hdf5_read_layout](#hdf5_read_layout) (reads the layout and the occupied slots)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Read channel objects from one or several slots of an HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_read_channel) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_channel)

- Returns the channels of the selected slots that hold data as a list of dicts `chan`, and their unstructured data as
  a list of dicts `par` of the same length
- `ix`, `iy`, `iz` and `iw` select the slots along each dimension; the selection is every combination, in
  column-major order (`ix` fastest), and empty slots are skipped ([hdf5_read_layout](#hdf5_read_layout) tells which slots hold data)
- `i_snap` reads only the given snapshots of every selected slot (see Notes)

### Usage:
```python
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, *, i_snap, stack )
```

### Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot indices along x, y, z and w, 0-based, inside the layout; int or
  array; default: `None` (all slots of the dimension)
- **`i_snap`** — Snapshot indices, 0-based, any order, repeats allowed, below `n_snap` of every selected slot that
  has snapshots; `(n_out,)`; default: `None` (all snapshots)
- **`stack`** — `False`: members with one entry per snapshot are lists of arrays; `True`: one array with the
  snapshot on the last axis, zero-padded to the largest number of paths; default: `False`

### Outputs:
- **`chan`** — Channels, one per selected slot that holds data; list of dicts with the keys:
  | Key | Description | Shape `stack = False` | Shape `stack = True` |
  | --- | ----------- | --------------------- | -------------------- |
  | `name` | Channel name | str | str |
  | `tx_pos`, `rx_pos` | Tx and Rx positions in m | `(3, 1)` (fixed) or `(3, n_snap)` | as left |
  | `tx_orientation`, `rx_orientation` | Tx and Rx orientations (bank, tilt, heading) in rad | `(3, 1)` (fixed) or `(3, n_snap)` | as left |
  | `coeff` | Coefficients; complex128 | list of `(n_rx, n_tx, n_path)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay` | Delays in s | list of `(n_rx, n_tx, n_path)` or `(1, 1, n_path)` | `(n_rx, n_tx, n_path, n_snap)` or `(1, 1, n_path, n_snap)` |
  | `path_gain` | Path gain without antennas, linear | list of `(n_path,)` | `(n_path, n_snap)` |
  | `path_length` | Path length in m | list of `(n_path,)` | `(n_path, n_snap)` |
  | `polmat` | Polarization transfer matrix (Glossary) | list of `(8, n_path)` | `(8, n_path, n_snap)` |
  | `path_angles` | AOD, EOD, AOA, EOA in rad | list of `(n_path, 4)` | `(n_path, 4, n_snap)` |
  | `fbs_pos`, `lbs_pos` | First- and last-bounce scatterer positions in m | list of `(3, n_path)` | `(3, n_path, n_snap)` |
  | `n_interact` | Number of interaction points per path, 0 = LOS; int64 | list of `(n_path,)` | `(n_path, n_snap)` |
  | `interact_coord` | Interaction points of all paths, concatenated | list of `(3, sum(n_interact))` | `(3, max(sum(n_interact)), n_snap)` |
  | `center_freq` | Center frequency in Hz | `(1,)` or `(n_snap,)` | as left |
  | `initial_position` | Reference snapshot, 0-based | int | int |

  `n_path` may differ between the snapshots; `stack = True` zero-pads to the largest one. A key exists if the channel
  has the member
- **`par`** — Unstructured data, one dict per entry of `chan` (empty if the slot has none); keys are the dataset
  names without the prefix `par_`, values as [hdf5_read_dset](#hdf5_read_dset) returns them

### Notes:
- **Snapshot selection**: members with one entry per snapshot are sliced, also positions, orientations and
  `center_freq` stored per snapshot; values stored once and the unstructured data are kept. A slot without snapshots
  (only unstructured data) ignores `i_snap`
- **Round trip** with [hdf5_write_channel](#hdf5_write_channel): the values come back in single precision; empty orientations come back as
  `(3, 1)` zeros
- **Unstructured data**: the datasets of the slot with the prefix `par_`; datasets of another prefix are not read
- **Errors**: a missing file, an index outside the layout, an index in `i_snap` beyond the snapshots of a slot, or a
  file that is not a channel file of the [HDF5 channel file format](../../formats/hdf5_channel.md#hdf5-compatibility)

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (finds the occupied slots)
- [hdf5_write_channel](#hdf5_write_channel) (writes channels)

---
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read an unstructured dataset from a slot of an HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_read_dset) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_dset)

- Reads the dataset `prefix + name` of slot `(ix, iy, iz, iw)`, e.g. `par_carrier_frequency`
- The type follows the stored dataset: str for a string, a Python number for a scalar, an array `(n,)` for a 1D
  dataset (also with one element), else an array of the stored shape; dtypes float32, float64, int32, uint32, int64,
  uint64

### Usage:
```python
dset = quadriga_lib.channel.hdf5_read_dset( fn, name, ix, iy, iz, iw, *, prefix )
```

### Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`; str
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 0-based; default: 0
- **`prefix`** — Prefix of the dataset name; `''` reads `name` as given; str; default: `'par_'`

### Output:
- **`dset`** — The dataset; `None` if the slot is empty or outside the layout or has no such dataset

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (lists the datasets of a slot)
- [hdf5_write_dset](#hdf5_write_dset) (writes a dataset)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read the names of the unstructured datasets of a slot of an HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_read_dset_names) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_dset_names)

- Lists the datasets of slot `(ix, iy, iz, iw)` whose names start with `prefix`, without the prefix, in the order
  they were written

### Usage:
```python
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw, *, prefix )
```

### Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 0-based; default: 0
- **`prefix`** — Prefix of the listed names; `''` lists all datasets with their full names; str; default: `'par_'`

### Output:
- **`names`** — Dataset names; list of str, empty if the slot is empty or outside the layout or has none

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (reads a dataset by name)
- [hdf5_write_dset](#hdf5_write_dset) (writes a dataset)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of an HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_read_layout) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_read_layout)

- Returns the dimensions of the slot grid and the slots that hold data
- A missing file gives `[0, 0, 0, 0]`, so the function also tests whether a file exists; a file that is not a channel
  file raises an error

### Usage:
```python
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

### Input:
- **`fn`** — File name, str or path-like

### Outputs:
- **`storage_dims`** — Layout `[nx, ny, nz, nw]`; int64; `(4,)`
- **`has_data`** — 1 for the slots that were written, also with an empty channel, else 0; uint32; `(nx, ny, nz, nw)`

### See also:
- [hdf5_create_file](#hdf5_create_file) (creates a file with a given layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (changes the dimensions)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshape the storage layout of an existing HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_reshape_layout) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_reshape_layout)

- Sets new dimensions `[nx, ny, nz, nw]` with the slot count `nx · ny · nz · nw` of the file
- The data stays in place: the slot with the linear index `ix + nx * (iy + ny * (iz + nz * iw))` keeps its channel,
  which gets the slot index of that position in the new layout

### Usage:
```python
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`nx`** — New number of slots along x
- **`ny`**, **`nz`**, **`nw`** — New number of slots along y, z and w; default: 1

### See also:
- [hdf5_create_file](#hdf5_create_file) (sets the layout of a new file)
- [hdf5_read_layout](#hdf5_read_layout) (reads the layout)

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write channel objects to one or several slots of an HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_write_channel) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_write_channel)

- Stores `chan[k]` with the unstructured data `par[k]` in slot `(ix[k], iy[k], iz[k], iw[k])`; a scalar index
  applies to all channels
- A missing file is created with the layout that just holds the slots (the largest index + 1 per dimension);
  [hdf5_create_file](#hdf5_create_file) sets another layout
- A slot that holds data is replaced completely: its old channel and all its unstructured datasets, also those added
  with [hdf5_write_dset](#hdf5_write_dset), are removed
- All inputs are checked before anything is written, so a failing check leaves the file unchanged

### Usage:
```python
code = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

### Inputs:
- **`fn`** — File name, str or path-like
- **`chan`** — Channels to write, valid; a channel dict or a non-empty list of `n_chan` dicts, keys in
  [hdf5_read_channel](#hdf5_read_channel). Members with one entry per snapshot as a list of arrays or as one array with the snapshot on
  the last axis (`stack = True` of [hdf5_read_channel](#hdf5_read_channel)), detected per key; the coefficients as complex `coeff`
- **`par`** — Unstructured data; a dict or a list of `n_chan` dicts: a key is a dataset name (stored with the prefix
  `par_`), the values follow the type rules of [hdf5_write_dset](#hdf5_write_dset), `None` values are skipped; default: `None`
  (none)
- **`ix`** — Slot index along x, 0-based; int or `n_chan` entries; default: `None` (`0 ... n_chan - 1`)
- **`iy`**, **`iz`**, **`iw`** — Slot indices along y, z and w, 0-based; int or `n_chan` entries; default:
  `None` (0)

### Output:
- **`code`** — `0` if the slot was empty, `1` if its data was replaced; int64; `(n_chan,)`

### Notes:
- A channel without data and without `par` entries replaces the data of an occupied slot (it then reads as an empty
  channel, and [hdf5_read_layout](#hdf5_read_layout) reports it as written); it leaves an empty slot as it is and does not create a
  missing file
- The space of replaced data is not reused, so the file does not shrink; copying the channels into a new file reclaims
  it
- Files are not locked: one writer per file at a time, also for [hdf5_write_dset](#hdf5_write_dset) and [hdf5_reshape_layout](#hdf5_reshape_layout).
  Reading is safe while no one writes

### See also:
- [hdf5_read_channel](#hdf5_read_channel) (reads the channels back)
- [hdf5_create_file](#hdf5_create_file) (creates a file with a given layout)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write an unstructured dataset to a slot of an HDF5 channel file

Also in: [C++](../cpp/channel-data-and-processing.md#hdf5_write_dset) · [MATLAB](../mex/channel-data-and-processing.md#hdf5_write_dset)

- Adds the dataset `prefix + name` to slot `(ix, iy, iz, iw)`; everything else stored in the slot stays. An empty
  slot gets a channel without structured data
- A missing file is created with the layout `(ix + 1, iy + 1, iz + 1, iw + 1)`; [hdf5_create_file](#hdf5_create_file) sets another
  layout
- A dataset is written once: a dataset of the same name in the slot raises an error. A later [hdf5_write_channel](#hdf5_write_channel) to
  the slot removes it with the rest of the slot
- Invalid names and types raise an error before the file is changed
- [hdf5_read_channel](#hdf5_read_channel) returns the datasets with the prefix `par_` as the unstructured data `par`; [hdf5_read_dset](#hdf5_read_dset)
  reads each one back

### Usage:
```python
quadriga_lib.channel.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, *, prefix )
```

### Inputs:
- **`fn`** — File name, str or path-like
- **`name`** — Dataset name without the prefix, not empty; `prefix + name` holds only letters, digits and `_`, at
  most 65000 characters; str
- **`data`** — A str; a Python float (stored as float64), int or bool (stored as int64, a bool as 0 or 1); a numpy
  scalar, or a numpy array with 0 to 3 dimensions in any memory layout and byte order; dtypes float32, float64, int32,
  uint32, int64, uint64, other dtypes (e.g. int16, bool, complex) raise an error. A 1D array is stored as a vector and
  comes back with shape `(n,)`, also with one element; 2D and 3D arrays keep their shape, also `(n, 1)` and `(1, 1)`
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 0-based; inside the layout of an
  existing file; default: 0
- **`prefix`** — Prefix of the dataset name; `''` stores `name` without a prefix; str; default: `'par_'`

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (reads a dataset by name)
- [hdf5_read_dset_names](#hdf5_read_dset_names) (lists the datasets of a slot)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [C++](../cpp/channel-data-and-processing.md#qrt_file_parse) · [MATLAB](../mex/channel-data-and-processing.md#qrt_file_parse)

- Reads the header of a QRT file: counts, names, frequencies, the positions and orientations of the CIRs and the
  origins, and the file version; no path data
- A file without destinations reports one destination `RX` that owns all CIRs
- Raises an error for a file that cannot be opened, is not a [QRT file](../../formats/qrt.md) of version 5 or 6, or is
  truncated or corrupt

### Usage:
```python
n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

### Input:
- **`fn`** — File name, str or path-like

### Outputs:
- **`n_cir`** — Number of CIRs of each origin; int
- **`n_orig`** — Number of origin slots; int
- **`n_dest`** — Number of destinations, at least 1; int
- **`n_freq`** — Number of frequencies; int
- **`cir_offset`** — Number of CIRs before each destination, i.e. the 0-based index of its first CIR; int64;
  `(n_dest,)`
- **`orig_names`** — Origin names, empty for a slot not written yet; list of str of length `n_orig`
- **`dest_names`** — Destination names; list of str of length `n_dest`
- **`version`** — File version: 5 (EM) or 6 (scalar); int
- **`center_freq`** — Frequencies in Hz; `(n_freq,)`
- **`cir_pos`** — CIR positions in m; `(n_cir, 3)`
- **`cir_orientation`** — CIR orientations (bank, tilt, heading) in rad; `(n_cir, 3)`
- **`orig_pos`** — Origin positions in m, zeros for a slot not written yet; `(n_orig, 3)`
- **`orig_orientation`** — Origin orientations (bank, tilt, heading) in rad; `(n_orig, 3)`

### See also:
- [qrt_file_read](#qrt_file_read) (reads CIRs)

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read the paths of CIRs from a QRT file

Also in: [C++](../cpp/channel-data-and-processing.md#qrt_file_read) · [MATLAB](../mex/channel-data-and-processing.md#qrt_file_read)

- Reads the paths of a list of CIRs of one origin, with the positions and orientations of both ends: the inputs of
  [channel.spherical](channel-generators.md#channel.spherical) (`fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `polmat`) or [channel.planar](channel-generators.md#channel.planar) (the
  angles)
- Reads in chunks of 16 CIRs: besides the outputs, at most two chunks are held in memory
- `uplink` swaps the ends: the origin becomes the receiver (see Notes)

### Usage:
```python
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, \
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, \
    interact_type = quadriga_lib.channel.qrt_file_read( fn, i_cir, i_orig, *, uplink, normalize_polmat )
```

### Inputs:
- **`fn`** — File name, str or path-like
- **`i_cir`** — CIR indices, 0-based, below `n_cir`; `(n_out,)`; default: `None` (all CIRs of the file)
- **`i_orig`** — Origin index, 0-based, below `n_orig`; int; default: 0
- **`uplink`** — `False`: origin = Tx, destination = Rx; `True`: origin = Rx, destination = Tx; default: `False`
- **`normalize_polmat`** — Scaling of `polmat` and `path_gain`, bool (see Notes); default: `True`

### Outputs:
- **`center_freq`** — Frequencies in Hz; `(n_freq,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `(3, n_out)`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientations (bank, tilt, heading) in rad; `(3, n_out)`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m: the first and the last interaction
  point, the midpoint of Tx and Rx for a LOS path; list of `n_out` arrays `(3, n_path)`
- **`path_gain`** — Path gain, linear (see Notes); list of `n_out` arrays `(n_path, n_freq)`
- **`path_length`** — Length of the polyline from the Tx through the interaction points to the Rx in m; list of
  `n_out` arrays `(n_path,)`
- **`polmat`** — Polarization transfer matrix; list of `n_out` arrays `(8, n_path, n_freq)`; version 6: the
  complex scalar coefficient, `(2, n_path, n_freq)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure (at the Tx) and arrival (at the
  Rx) in rad; list of `n_out` arrays `(n_path,)`
- **`path_coord`** — Points of each path from the Tx through the interaction points to the Rx; list of `n_out` lists
  of `n_path` arrays `(3, n_interact + 2)`
- **`n_interact`** — Number of interaction points of each path, 0 = LOS; list of `n_out` int64 arrays `(n_path,)`
- **`coord`** — Interaction points of all paths, concatenated in path order; list of `n_out` arrays
  `(3, sum(n_interact))`
- **`interact_type`** — Interaction type codes (see [QRT file format: stored values](../../formats/qrt.md#qrt-stored-values)),
  concatenated like `coord`; list of `n_out` uint8 arrays `(sum(n_interact),)`

### Notes:
- **Path gain**: the file stores per path the coefficients with the interaction (material) losses; the reader adds the
  propagation loss of the path length $d$, for EM files (version 5) the free-space loss
  $\mathrm{PL} = 32.45 + 20 \log_{10}(f / \mathrm{GHz}) + 20 \log_{10}(d / \mathrm{m})$ dB, for scalar files (version 6)
  the spherical spreading and the air absorption $\alpha(f)$ of ISO 9613-1 at 20 °C, 50 % humidity and 1 atm,
  $\mathrm{PL} = 20 \log_{10}(d / \mathrm{m}) + \alpha(f) \, d$ dB. With $p$ the larger column power of the stored
  matrix (version 6: the power of the coefficient):
  | `normalize_polmat` | `polmat` | `path_gain` |
  | ------------------ | -------- | ----------- |
  | `False` | As stored | $10^{-\mathrm{PL}/10}$ |
  | `True` | Stored matrix divided by $\sqrt{p}$: largest column power 1 (uplink: row power) | $p \cdot 10^{-\mathrm{PL}/10}$ |

- **Uplink**: `polmat` is the transpose of the downlink matrix (reciprocal channel, no conjugation) with the same
  `path_gain`; positions, orientations, angles, `fbs_pos`, `lbs_pos` and `path_coord` swap their ends.
  `n_interact`, `coord` and `interact_type` keep the stored order from the origin to the destination, and the type
  codes describe that traversal
- An origin slot that is not written yet gives zero positions and orientations and no paths
- **Errors**: an index out of range raises `ValueError`; a file that cannot be opened, is not a
  [QRT file](../../formats/qrt.md) of version 5 or 6, or is truncated or corrupt (counts or offsets beyond the file, path
  blocks of different CIRs that overlap) raises `RuntimeError`. A message about one requested CIR starts with
  `Snapshot <k>: `, with `k` its position in `i_cir`

### See also:
- [qrt_file_parse](#qrt_file_parse) (metadata without path data)
- [channel.spherical](channel-generators.md#channel.spherical) (coefficients from the paths)
- [channel.planar](channel-generators.md#channel.planar) (coefficients from the angles)

---
<a id="quantize_delays"></a>
## quantize_delays
Map the path delays to a tap grid by two-tap interpolation

Also in: [C++](../cpp/channel-data-and-processing.md#quantize_delays) · [MATLAB](../mex/channel-data-and-processing.md#quantize_delays)

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

### Usage:
```python
coeff_q, delay_q = quadriga_lib.channel.quantize_delays( coeff, delay, *, tap_spacing, max_taps, \
    power_exponent, fix_taps, stack )
```

### Inputs:
- **`coeff`** — Complex coefficients; list of `n_snap` arrays `(n_rx, n_tx, n_path_s)`, at least one; `n_path_s` may
  differ between the snapshots, `n_rx` and `n_tx` (>= 1) not
- **`delay`** — Delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_path_s)` or `(1, 1, n_path_s)`, one of the two
  forms for all snapshots; finite, above `-tap_spacing / 2` and below `2^31` · `tap_spacing`
- **`tap_spacing`** — Tap spacing in s, > 0 and finite; 5 ns is a sampling rate of 200 MHz; default: 5e-9
- **`max_taps`** — Largest number of taps of a grid; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent $\alpha$, > 0; default: 1.0
- **`fix_taps`** — Which coefficients share a grid; default: 0

  | Value | Grid |
  | ----- | ---- |
  | 0 | One grid per antenna pair and snapshot |
  | 1 | One grid for all antenna pairs and snapshots |
  | 2 | One grid per snapshot for all its antenna pairs; the grids of the snapshots differ |
  | 3 | One grid per antenna pair for all snapshots |

- **`stack`** — `True`: the outputs as 4D arrays with the snapshots on the last axis; `False`: as lists with one array
  per snapshot; default: `False`

### Outputs:
- **`coeff_q`** — Complex coefficients per tap; complex128; list of `n_snap` arrays `(n_rx, n_tx, n_taps)`, or
  `(n_rx, n_tx, n_taps, n_snap)` with `stack`
- **`delay_q`** — Tap delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_taps)`, or `(n_rx, n_tx, n_taps, n_snap)`
  with `stack`; `(1, 1, n_taps)` and `(1, 1, n_taps, n_snap)` for delays `(1, 1, n_path_s)` with `fix_taps` 1 or 2

### Notes:
- Negative delays: delays in `(-tap_spacing / 2, 0)` count as 0, all their power goes to tap 0 (the error is below
  `tap_spacing / 2`, as for rounding to the nearest tap)
- Relative delays (`use_absolute_delays = False` in [channel.spherical](channel-generators.md#channel.spherical), [channel.planar](channel-generators.md#channel.planar) and
  [channel.multifreq](channel-generators.md#channel.multifreq)) are negative for element pairs that are closer to each other than the phase centres, by up
  to (half the Tx array size + half the Rx array size) / propagation speed. They stay above `-tap_spacing / 2` only
  while these half sizes add up to less than about 0.75 m for radio at 5 ns taps, or 3.6 mm for sound at 48 kHz
  taps, i.e. practically never for acoustic arrays: use absolute delays then
- The upper limit of `2^31` taps is e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz; run time and memory depend on the
  number of paths and taps, not on the delay values

### See also:
- [freq_response](#freq_response) (frequency response of the paths or taps)
- [calc_delay_spread](#calc_delay_spread) (delay spread, with an optional delay granularity)
