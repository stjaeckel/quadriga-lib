# Channel generators

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-channel-generators

| Function | Summary |
| --- | --- |
| [channel.ieee_indoor](#channel.ieee_indoor) | Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models |
| [channel.irs](#channel.irs) | Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface |
| [channel.multifreq](#channel.multifreq) | Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies |
| [channel.planar](#channel.planar) | Calculate MIMO channel coefficients and delays for planar wave propagation |
| [channel.spherical](#channel.spherical) | Calculate MIMO channel coefficients and delays for spherical wave propagation |
| [channel.tr38901](#channel.tr38901) | Generate the channels of a TR 38.901 scenario in one call |

The functions of this section compute the channel coefficients of a link from its propagation paths and the antenna
arrays at both ends. They apply the antenna patterns, the polarization and the motion of the antennas, and return the
coefficients and delays that the functions of section "Channel data and processing" work with. The coefficients are
given per port pair: `n_tx` and `n_rx` count the ports of the arrays, after their port weights.

**Generators:**

- [channel.spherical](#channel.spherical): paths given by their first- and last-bounce scatterers; each antenna element sees the path
  from its own position (spherical wavefronts), so it covers large arrays and short distances
- [channel.planar](#channel.planar): paths given by their departure and arrival angles; all elements of an array see the same
  direction (planar wavefronts)
- [channel.multifreq](#channel.multifreq): as [channel.spherical](#channel.spherical), at several frequencies with frequency-dependent antennas, path gains
  and polarization, for radio and acoustic waves
- [channel.irs](#channel.irs): paths over an intelligent reflecting surface (IRS), from two path sets that meet at the IRS array
- [channel.ieee_indoor](#channel.ieee_indoor): complete channels of the IEEE TGn, TGac, TGah and TGax indoor models
- [channel.tr38901](#channel.tr38901): complete channels of a 3GPP TR 38.901 scenario (UMa, UMi, RMa, SMa, InH, InF) in one call

The paths come from a ray tracer (e.g. [qrt_file_read](channel-data-and-processing.md#qrt_file_read)), from the geometry-based stochastic model, or from the user.
For complete channels of a network layout, see [get_channels_layout](network-layouts-and-mobility.md#get_channels_layout).

All functions of this section are in the submodule `quadriga_lib.channel`, without the prefix `get_channels_` of their
C++ and MATLAB names: `get_channels_spherical` is called as `quadriga_lib.channel.spherical`. The arrays are arrayant
dicts ([generate_antenna](antennas-transducers-and-patterns.md#generate_antenna)); the coefficients are returned as complex arrays.

**Terms** (full list in the [Glossary](../cpp/general-usage-notes.md#glossary)):

- **Path gain**: the linear power gain of a path without antennas, `(n_path,)`
- **Path length**: the length of a path from the Tx to the Rx phase center in m, which sets its delay
- **Polarization transfer matrix** `polmat`: the Jones matrix of each path, `(8, n_path)` with interleaved real and
  imaginary parts (Glossary, "Polarization")
- **Absolute delays**: `use_absolute_delays = True` keeps the LOS delay in all paths; otherwise the delays are
  relative to the LOS path
- **Vector form**: [channel.planar](#channel.planar) and [channel.spherical](#channel.spherical) take the path inputs of several snapshots as lists of
  arrays, one per snapshot, and return lists

---
<a id="channel.ieee_indoor"></a>
## channel.ieee_indoor
Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models

Also in: [C++](../cpp/channel-generators.md#get_channels_ieee_indoor) · [MATLAB](../mex/channel-generators.md#get_channels_ieee_indoor)

- Channels of the IEEE 802.11 indoor models A to F (TGn), with the TGac extensions (several users with offset angles,
  tap interpolation for shorter tap spacings), the floor loss of TGah below 1 GHz and the floor and wall loss of TGax
  above; one channel per user
- 2D model: azimuth angles only and motion in the horizontal plane; [channel.planar](#channel.planar) computes the coefficients with
  the AP and station arrays
- Static channels (`observation_time = 0`, one snapshot) or time-evolving channels with the Doppler shifts of the
  station and the environment

### Usage:
```python
chan = quadriga_lib.channel.ieee_indoor( ap_array, sta_array, channel_type, center_freq, *, \
    tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, \
    dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, \
    kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB, stack )
```

### Inputs:
- **`ap_array`**, **`sta_array`** — AP and station array, arrayant dict, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna); `n_ap` and
  `n_sta` ports
- **`channel_type`** — Model: `"A"`, `"B"`, `"C"`, `"D"`, `"E"` or `"F"`
- **`center_freq`** — Carrier frequency in Hz, finite, above 100 MHz; default: 5.25e9
- **`tap_spacing`** — Tap spacing in s, 10 ns divided by a power of 2; default: 10e-9
- **`n_users`** — Number of users, >= 1; default: 1
- **`observation_time`** — Observation time in s, finite, >= 0; 0: one snapshot; default: 0.0
- **`update_interval`** — Time between snapshots in s, finite, > 0; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h, finite; default: 0.0
- **`speed_env_kmh`** — Speed of the environment in km/h, finite; TGn 1.2, TGac 0.089; default: 1.2
- **`dist`** — Horizontal AP-station distance in m per user, finite, > 0; one value for all users or `(n_users,)`;
  default: `None` (4.99 m)
- **`n_floors`** — Floors between AP and station per user, integers 0 to 4; one value or `(n_users,)`; default: `None`
  (0)
- **`uplink`** — Return the uplink, with the station as Tx; default: `False`
- **`offset_angles_deg`** — Offset angles in degrees per user, finite; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  `(4, n_users)`; default: `None` (0 for one user, the pseudo-random offsets of TGac for several)
- **`n_subpath`** — Sub-paths per tap, >= 1; default: 20
- **`doppler_effect`** — Special Doppler (Notes): the mains frequency in Hz for models D and E, the vehicle speed in
  km/h for model F; finite, >= 0; 0 disables it; default: 50.0
- **`seed`** — Seed of the random numbers; -1: the system random device; other values give the same channels on all
  platforms; default: -1
- **`kf`** — Ricean K-factor of the first tap, linear, finite, >= 0; default: `nan` (model default, Notes)
- **`xpr_nlos`** — Cross-polarization ratio of the NLOS paths, linear, finite, > 0; default: `nan` (model default)
- **`sf_std_los_dB`**, **`sf_std_nlos_dB`** — Standard deviation of the shadow fading in dB below and above the
  breakpoint distance, finite, >= 0; default: `nan` (model default)
- **`dist_breakpoint`** — Breakpoint distance in m, finite, > 0; default: `nan` (model default)
- **`n_walls`** — Walls between AP and station per user (TGax), integers; one value or `(n_users,)`; default: `None`
  (0)
- **`wall_loss_dB`** — Loss of one wall in dB, finite; TGax: 5 or 7; default: 5.0
- **`stack`** — Snapshot layout of the channels as in [hdf5_read_channel](channel-data-and-processing.md#hdf5_read_channel); default: `False`

### Output:
- **`chan`** — Channels, list of `n_users` dicts, keys in [hdf5_read_channel](channel-data-and-processing.md#hdf5_read_channel): `name` (`"user_<i>"`), `center_freq`,
  `initial_position`, the positions and orientations (station positions `(3, n_snap)`), `coeff` and `delay`
  `(n_rx, n_tx, n_path)` per snapshot (downlink `(n_sta, n_ap, n_path)`) and `path_gain` `(n_path,)` per snapshot: the
  path powers without antennas, with path loss and shadow fading, whose mean over the snapshots is the power of the
  model

### Notes:
- **Model defaults** (`NaN` in the override inputs), with the LOS shadow fading 3 dB and the NLOS XPR 2 (3 dB) for all
  models; below 1 GHz (TGah) both shadow fading values are 1 dB lower:

  | Model | A | B | C | D | E | F |
  | --- | --- | --- | --- | --- | --- | --- |
  | KF, linear | 1 | 1 | 1 | 2 | 4 | 4 |
  | Breakpoint distance in m | 5 | 5 | 5 | 10 | 20 | 30 |
  | NLOS shadow fading in dB | 4 | 4 | 5 | 5 | 6 | 6 |

- **LOS**: a user has a LOS path below the breakpoint distance on the floor of the AP, and always when `kf` is given;
  the KF splits the power of the first tap between the LOS path and the first NLOS cluster
- **Paths**: one per tap of the model tables with non-zero power, plus the LOS path; each tap sums `n_subpath` sub-paths
  with Laplacian angular spreads. Below 10 ns `tap_spacing`, the taps are interpolated (TGac)
- **Path loss** in dB at the distance $d$ = `dist`: $20 \lg(4\pi f_c / c) + 20 \lg d$ below the breakpoint distance
  $d_\mathrm{BP}$, $20 \lg(4\pi f_c / c) + 20 \lg d_\mathrm{BP} + 35 \lg(d / d_\mathrm{BP})$ beyond, plus the floor
  loss (TGah below 1 GHz, TGax above) and `n_walls` times `wall_loss_dB`. The shadow fading uses the LOS value below
  the breakpoint, else the NLOS value; below 1 GHz on another floor, the TGah floor values replace it
- **Geometry**: the AP at the origin, the station at the distance `dist` in the direction of the AoD LOS offset and
  3 m higher per floor; floors add loss but no delay. The delays are relative to the LOS delay
- **Time evolution**: `1 + floor(observation_time / update_interval)` snapshots, `update_interval` apart; each sub-path
  turns by $e^{j 2\pi f_D t}$ with the Doppler shift $f_D$ of the station speed plus a random environment Doppler
  (bell-shaped spectrum, TGn). The station moves along its heading, toward the AP for an AoA LOS offset of 0; moving
  toward the AP or a scatterer gives a positive $f_D$, as [channel.spherical](#channel.spherical) along the returned station
  positions
- **Special Doppler** (`doppler_effect` > 0): models D and E modulate three taps with fluorescent light at the
  harmonics of the mains frequency, model F adds the reflection of a moving vehicle to one tap (TGn)
- **Uplink**: the coefficients and delays are the transposed downlink ones (reciprocal channel, no conjugation, the same
  Doppler shift); the positions and orientations are swapped, the station is the Tx

### See also:
- [channel.planar](#channel.planar) (computes the coefficients)
- [hdf5_write_channel](channel-data-and-processing.md#hdf5_write_channel) (store the channels)
- [IEEE 802.11-03/940r4 - TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc)
- [IEEE 802.11-09/0308r12 - TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc)
- [IEEE 802.11-11/0968r4 - TGah Channel Model](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx)
- [IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx)
- [S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025](http://quadriga-lib.org/publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf)

---
<a id="channel.irs"></a>
## channel.irs
Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface

Also in: [C++](../cpp/channel-generators.md#get_channels_irs) · [MATLAB](../mex/channel-generators.md#get_channels_irs)

- Paths of one link over an intelligent reflecting surface (IRS) with `n_irs` elements: segment 1 from the Tx to the
  IRS and segment 2 from the IRS to the Rx, each computed by [channel.spherical](#channel.spherical) with the IRS elements as Rx and Tx
  elements; each output path combines one path of each segment
- Each IRS element $k$ reflects once with its weight $w_k$, column `i_irs` of the IRS port weights (a codebook entry):
  per path pair, $\mathbf{H} = \mathbf{B} \, \mathrm{diag}(\mathbf{w}) \, \mathbf{A}$ with the element channels
  $\mathbf{A}$ of segment 1 `(n_irs, n_tx)` and $\mathbf{B}$ of segment 2 `(n_rx, n_irs)`
- The delay of a pair is the sum of its segment delays, each the $|w_k|^2$-weighted mean over the IRS elements; the
  departure angles come from segment 1, the arrival angles from segment 2, as $|w_k|^2$-weighted mean directions
- Kept are the path pairs whose strongest coefficient has a power above `threshold_dB`, or those set in `active_path`;
  the pair of segment-1 path $i_1$ and segment-2 path $i_2$ has the index $i_1 n_{\mathrm{path},2} + i_2$ in
  `active_path` and `active_path_out`, and the output keeps this order
- The outputs come in another order than in C++: `active_path_out` comes before the angles, which only `angles=True`
  returns

### Usage:
```python
coeff, delay, active_path_out = quadriga_lib.channel.irs( tx_array, rx_array, irs_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, \
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1, \
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2, center_freq, \
    *, i_irs, threshold_dB, use_absolute_delays, angles, irs_array_2, active_path, use_kernel )

# Departure and arrival angles as well
coeff, delay, active_path_out, aod, eod, aoa, eoa = quadriga_lib.channel.irs( ..., angles=True )
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, arrayant dict, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna); `n_tx` and `n_rx` ports
- **`irs_array`** — IRS array with `n_irs` elements, arrayant dict, valid, with port weights `(n_irs, n_codebook)` that
  hold the reflection weights; its patterns serve both segments unless `irs_array_2` is given
- **`tx_pos`**, **`rx_pos`**, **`irs_pos`** — Tx, Rx and IRS position in m, finite; `(3,)`
- **`tx_orientation`**, **`rx_orientation`**, **`irs_orientation`** — Orientations, Euler angles (bank, tilt, heading)
  in rad, finite; `(3,)`
- **`fbs_pos_1`**, **`lbs_pos_1`** — First- and last-bounce scatterer positions of the segment-1 paths in m;
  `(3, n_path_1)`
- **`path_gain_1`** — Path gains of segment 1, linear power; `(n_path_1,)`
- **`path_length_1`** — Path lengths of segment 1 in m between the Tx and IRS phase centres, as in
  [channel.spherical](#channel.spherical); `(n_path_1,)`
- **`polmat_1`** — Jones matrices of segment 1, the IRS receiving (section description); `(8, n_path_1)`
- **`fbs_pos_2`**, **`lbs_pos_2`**, **`path_gain_2`**, **`path_length_2`**, **`polmat_2`** — The same for segment 2
  from the IRS to the Rx, the IRS transmitting; `n_path_2` paths
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0, as in
  [channel.spherical](#channel.spherical); default: 0.0
- **`i_irs`** — Codebook entry, 0-based column of the IRS port weights, below `n_codebook`; default: 0
- **`threshold_dB`** — Minimum power in dB of the strongest coefficient of a kept path pair; not used with
  `active_path`; default: -140.0
- **`use_absolute_delays`** — Keep the delay of the direct Tx-Rx distance in all delays; `False`: delays relative to it
  (Notes); default: `False`
- **`angles`** — Also return the angles; default: `False`
- **`irs_array_2`** — Patterns of the IRS for segment 2 (asymmetric IRS), arrayant dict, valid, `n_irs` elements; its
  port weights and element positions are not used; default: `None` (`irs_array`)
- **`active_path`** — Path pairs to keep, replacing `threshold_dB`; bool or numeric (nonzero = keep);
  `(n_path_1 * n_path_2,)`; default: `None` (`threshold_dB`)
- **`use_kernel`** — Kernel of both segments, as in [channel.spherical](#channel.spherical); `int`; default: 0

### Outputs:
- **`coeff`** — Coefficients, complex; `(n_rx, n_tx, n_path_irs)` with the number of kept path pairs `n_path_irs`
- **`delay`** — Delays in s; `(n_rx, n_tx, n_path_irs)`
- **`active_path_out`** — Kept path pairs, for `active_path` of a later call; bool; `(n_path_1 * n_path_2,)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure and arrival in rad, GCS;
  `(n_rx, n_tx, n_path_irs)`; only with `angles=True`

### Notes:
- **IRS port weights**: an `irs_array` without `port_weights_re` / `port_weights_im` raises here, unlike in the other
  functions, where they mean identity; the IRS elements are not combined into ports
- **Segments**: path lengths, LOS paths and the per-element geometry as in [channel.spherical](#channel.spherical); a segment without
  paths (all its path inputs empty) gives the outputs `(n_rx, n_tx, 0)`
- **Relative delays**: the delay of the direct Tx-Rx distance (not via the IRS) is subtracted; delays can be slightly
  negative when the IRS is close to the Tx-Rx line

### See also:
- [channel.spherical](#channel.spherical) (the channel of each segment)

---
<a id="channel.multifreq"></a>
## channel.multifreq
Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies

Also in: [C++](../cpp/channel-generators.md#get_channels_multifreq) · [MATLAB](../mex/channel-generators.md#get_channels_multifreq)

- As [channel.spherical](#channel.spherical), at the output frequencies `freq_new`, with multi-frequency arrays (frequency-dependent
  patterns and port weights), path gains and Jones matrices; for radio waves and, with `propagation_speed` = 343 m/s
  and scalar `polmat`, for sound
- The geometry (element-pair path lengths, angles, LOS path) is computed once; at each output frequency, the patterns
  and port weights of the arrays and `path_gain` and `polmat` are interpolated between their two neighbouring entries
- The coefficients follow the formula of [channel.spherical](#channel.spherical) with $f_c$ = `freq_new[k]` and $c$ =
  `propagation_speed`; the frequency is the last axis of the outputs, a multi-frequency channel

### Usage:
```python
coeff, delay = quadriga_lib.channel.multifreq( tx_array, rx_array, tx_pos, tx_orientation, rx_pos, \
    rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, freq, freq_new, *, \
    use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel )
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx multi-frequency array, arrayant dict, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna): 4D
  patterns with the frequency on the last axis and `center_freq` `(n,)`, or 3D patterns for one frequency; port
  weights 2D (all frequencies) or 3D `(n_elements, n_ports, n)`; frequencies in any order; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `(3,)`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; `(3,)`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `(3, n_path)`
- **`path_gain`** — Path gains at the frequencies `freq`, linear power; `(n_path, n_freq)`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres, as in [channel.spherical](#channel.spherical); `(n_path,)`
- **`polmat`** — Jones matrices at the frequencies `freq` (section description): `(8, n_path, n_freq)`, or
  `(2, n_path, n_freq)` for a scalar coefficient (real and imaginary part, sound)
- **`freq`** — Sample frequencies of `path_gain` and `polmat` in Hz, in any order; `(n_freq,)`, at least one
- **`freq_new`** — Output frequencies in Hz; `(n_freq_new,)`, at least one
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `False`: delays relative to it;
  default: `False`
- **`add_fake_los_path`** — Put the LOS path into an added slot 0, with zero power if there is none; default: `False`
- **`propagation_speed`** — Wave speed in m/s, > 0; default: 299792458.0
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC, 2 = AVX2
  (single-precision arithmetic; raises if unavailable); `int`; default: 0

### Outputs:
- **`coeff`** — Coefficients, complex; `(n_rx, n_tx, n_out, n_freq_new)`, `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`
- **`delay`** — Delays in s; `(n_rx, n_tx, n_out, n_freq_new)`

### Notes:
- **Interpolation**: `path_gain` linear; patterns, port weights and `polmat` per complex value, the amplitude linear and
  the phase along the shorter arc (linear for nearly opposite phases); constant outside the range of each grid
- **Scalar `polmat`**: only the $F_\theta$ patterns are used,
  $a_l = \sqrt{g_l} \, F_{\theta,r} \, m_l \, F_{\theta,t} \, e^{-j 2\pi f d / c}$ with the complex scalar $m_l$
- **Paths**: NLOS path length, LOS path, `add_fake_los_path` and 0 paths as in [channel.spherical](#channel.spherical)
- **Relative delays**: negative delays as in [channel.spherical](#channel.spherical), with $c$ = `propagation_speed`; for sound, this
  affects practically every array, so use absolute delays for [quantize_delays](channel-data-and-processing.md#quantize_delays)
- **Delay phase**: the coefficients carry the phase of the absolute path length also with relative delays;
  [freq_response_mf](channel-data-and-processing.md#freq_response_mf) with `remove_delay_phase = True` therefore needs `use_absolute_delays = True`

### See also:
- [channel.spherical](#channel.spherical) (one frequency)
- [freq_response_mf](channel-data-and-processing.md#freq_response_mf) (frequency response of a multi-frequency channel)
- [join_arrays](antennas-transducers-and-patterns.md#join_arrays) (multi-frequency arrays)
- [generate_speaker](antennas-transducers-and-patterns.md#generate_speaker) (loudspeaker arrays)

---
<a id="channel.planar"></a>
## channel.planar
Calculate MIMO channel coefficients and delays for planar wave propagation

Also in: [C++](../cpp/channel-generators.md#get_channels_planar) · [MATLAB](../mex/channel-generators.md#get_channels_planar)

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; a path is given by its
  departure and arrival angles, which all elements of an array share (planar wavefronts)
- Path $l$ of the element pair $(t, r)$ has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \quad \tau_l = d / c, \quad d = \ell_l + d_t + d_r$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  at the path angles, the path length $\ell_l$ between the phase centres (used as given), the distances $d_t$, $d_r$ of
  the elements from the wavefront planes through the phase centres (negative for elements closer to the other end of the
  path), the center frequency $f_c$ and the speed of light $c$; the port weights then combine the elements into ports
- The positions enter only through the Tx-Rx distance (relative delays, LOS path)
- Vector form: with lists of arrays as path inputs, one per snapshot, the outputs are lists; the snapshots are computed
  in parallel
- Path data of several frequencies: `path_gain`, `polmat` and `center_freq` may hold `n_freq` frequencies, of which
  `i_freq` selects one (Python only, since the C++ function takes the data of one frequency)

### Usage:
```python
coeff, delay, rx_Doppler = quadriga_lib.channel.planar( tx_array, rx_array, tx_pos, tx_orientation, rx_pos, \
    rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, polmat, center_freq, *, \
    use_absolute_delays, add_fake_los_path, use_kernel, i_freq )
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, arrayant dict with 3D patterns, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna);
  `n_tx` and `n_rx` ports; a 4D (multi-frequency) dict raises `ValueError` (use [channel.multifreq](#channel.multifreq))
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `(3,)`; vector form: `(3,)` or `(3, 1)` (all
  snapshots) or `(3, n_snap)`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; shapes as `tx_pos`
- **`aod`**, **`eod`** — Azimuth and elevation of departure in rad, GCS; `(n_path,)`; vector form: lists with `n_snap`
  entries
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in rad, GCS (the direction from the Rx toward the incoming
  path); `(n_path,)`; vector form: lists
- **`path_gain`** — Path gains, linear power; `(n_path,)` or `(n_path, n_freq)`; vector form: list
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; `(n_path,)`; vector form: list
- **`polmat`** — Jones matrices, real and imaginary parts interleaved (section description); `(8, n_path)` or
  `(8, n_path, n_freq)`; vector form: list
- **`center_freq`** — Center frequency in Hz, finite and >= 0, `(n_freq,)`; 0 or `None` gives the coefficients at f = 0
  (Notes); default: `None`
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `False`: delays relative to it
  (Notes); default: `False`
- **`add_fake_los_path`** — Put the LOS path into an added slot 0, with zero power if there is none (Notes); default:
  `False`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC, 2 = AVX2
  (single-precision arithmetic; raises if unavailable); `int`; default: 0
- **`i_freq`** — 0-based frequency index into `path_gain`, `polmat` and `center_freq`, below `n_freq`; default: 0

### Outputs:
- **`coeff`** — Coefficients, complex; `(n_rx, n_tx, n_out)` with `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`; vector form: lists of `n_snap` such arrays, as all outputs
- **`delay`** — Delays in s; `(n_rx, n_tx, n_out)`
- **`rx_Doppler`** — Doppler weight of each path for an Rx moving along its local x-axis: the cosine of the angle
  between this axis and the arrival direction; the Doppler shift at the speed $v$ is $v f_c / c$ times this weight,
  positive when the Rx moves toward the incoming path; `(n_out,)`

### Notes:
- **LOS path** (only with `add_fake_los_path`): the strongest path whose `path_length` is within 0.1 mm of the Tx-Rx
  distance; the angles are not checked
- **`add_fake_los_path`**: slot 0 holds the LOS path, with zero power (and `rx_Doppler` 0) if there is none; a LOS path
  among the inputs moves to slot 0 and leaves a zero-power path in its own slot (input index + 1); the other paths
  keep their order
- **0 paths** (all path inputs empty): outputs `(n_rx, n_tx, 0)`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays; element pairs can then get
  negative delays, as in [channel.spherical](#channel.spherical)
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays of a port pair are means over its element pairs, weighted by the port-weight powers
- **Errors and threads**: as in [channel.spherical](#channel.spherical)

### See also:
- [channel.spherical](#channel.spherical) (paths given by scatterers, spherical wavefronts)
- [channel.ieee_indoor](#channel.ieee_indoor) (IEEE indoor channels, computed with this function)
- [freq_response](channel-data-and-processing.md#freq_response) (frequency response of the coefficients)
- [quantize_delays](channel-data-and-processing.md#quantize_delays) (delays on a tap grid)

---
<a id="channel.spherical"></a>
## channel.spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

Also in: [C++](../cpp/channel-generators.md#get_channels_spherical) · [MATLAB](../mex/channel-generators.md#get_channels_spherical)

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; every element sees a path
  from its own position (spherical wavefronts), so delays and phases follow the geometry of each element pair
- A path runs from the Tx over its first-bounce scatterer (FBS) and last-bounce scatterer (LBS) to the Rx; the patterns
  of both arrays are interpolated at the departure and arrival angles of each element pair ([interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern))
- Path $l$ of an element pair at the distance $d$ along the path has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \qquad \tau_l = d / c$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  of the Rx and Tx element, the center frequency $f_c$ and the speed of light $c$; the port weights then combine the
  elements into ports
- Vector form: with lists of arrays as path inputs, one per snapshot, the outputs are lists; the snapshots are computed
  in parallel
- Path data of several frequencies: `path_gain`, `polmat` and `center_freq` may hold `n_freq` frequencies, of which
  `i_freq` selects one (Python only, since the C++ function takes the data of one frequency)

### Usage:
```python
coeff, delay = quadriga_lib.channel.spherical( tx_array, rx_array, tx_pos, tx_orientation, rx_pos, \
    rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, center_freq, *, \
    use_absolute_delays, add_fake_los_path, angles, use_kernel, i_freq )

# Departure and arrival angles as well
coeff, delay, aod, eod, aoa, eoa = quadriga_lib.channel.spherical( ..., angles=True )
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, arrayant dict with 3D patterns, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna);
  `n_tx` and `n_rx` ports; a 4D (multi-frequency) dict raises `ValueError` (use [channel.multifreq](#channel.multifreq))
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `(3,)`; vector form: `(3,)` or `(3, 1)` (all
  snapshots) or `(3, n_snap)`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; shapes as `tx_pos`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `(3, n_path)`; vector form: lists with
  `n_snap` entries, as [qrt_file_read](channel-data-and-processing.md#qrt_file_read) returns them
- **`path_gain`** — Path gains, linear power; `(n_path,)` or `(n_path, n_freq)`; vector form: list
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; a value below the Tx-FBS-LBS-Rx polyline
  (e.g. 0) gives the polyline length (Notes); `(n_path,)`; vector form: list
- **`polmat`** — Jones matrices, real and imaginary parts interleaved (section description); `(8, n_path)` or
  `(8, n_path, n_freq)`; vector form: list
- **`center_freq`** — Center frequency in Hz, finite and >= 0, `(n_freq,)`; 0 or `None` gives the coefficients at f = 0
  (Notes); default: `None`
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `False`: delays relative to it
  (Notes); default: `False`
- **`add_fake_los_path`** — Put the LOS path into an added slot 0, with zero power if there is none (Notes); default:
  `False`
- **`angles`** — Also return the angles; `False` saves their memory and, with port weights, the time to compute them for
  the ports; default: `False`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto (GENERIC in this
  version), 1 = GENERIC, 2 = AVX2 (single-precision arithmetic; raises if unavailable); `int`; default: 0
- **`i_freq`** — 0-based frequency index into `path_gain`, `polmat` and `center_freq`, below `n_freq`; default: 0

### Outputs:
- **`coeff`** — Coefficients, complex; `(n_rx, n_tx, n_out)` with `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`; vector form: lists of `n_snap` such arrays, as all outputs
- **`delay`** — Delays in s; `(n_rx, n_tx, n_out)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure and arrival in rad, GCS;
  `(n_rx, n_tx, n_out)`; only with `angles=True`

### Notes:
- **NLOS path length**: the length between the phase centres is $\max(\ell, d_\mathrm{poly})$ with `path_length`
  $\ell$ and the length $d_\mathrm{poly}$ of the Tx-FBS-LBS-Rx polyline; the excess, e.g. from more than two
  interactions as in [qrt_file_read](channel-data-and-processing.md#qrt_file_read), is added to the FBS-LBS segment
- **LOS path**: the path whose length is closest to the Tx-Rx distance, within 0.1 mm; its element pairs use the
  direct line between the elements instead of the FBS and LBS
- **`add_fake_los_path`**: slot 0 holds the LOS path, with zero power if there is none; a LOS path among the inputs moves
  to slot 0 and leaves a zero-power path in its own slot (input index + 1); the other paths keep their order
- **0 paths** (all path inputs empty): outputs `(n_rx, n_tx, 0)`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays. Element pairs closer to each other
  than the phase centres then get negative delays, down to about $-(D_t + D_r) / (2c)$ with the array sizes $D$;
  [quantize_delays](channel-data-and-processing.md#quantize_delays) rejects delays at or below `-tap_spacing / 2`, so large arrays need absolute delays
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays and angles of a port pair are means over its element pairs, weighted by the
  port-weight powers (the angles via the mean direction)
- **Errors**: an error in the path data names its snapshot (`Snapshot <i>: `, 0-based, the lowest failing one), also
  without the vector form (`Snapshot 0: `); invalid arrays and `use_kernel` raise without this prefix
- **Threads**: the GIL is released during the computation; the function keeps references to its input arrays, but
  other threads must not write into them meanwhile. List entries that are Fortran-contiguous float64 arrays are read
  without a copy

### See also:
- [channel.planar](#channel.planar) (paths given by angles, planar wavefronts)
- [channel.multifreq](#channel.multifreq) (several frequencies)
- [channel.irs](#channel.irs) (paths over an IRS)
- [qrt_file_read](channel-data-and-processing.md#qrt_file_read) (path data of many snapshots)
- [freq_response](channel-data-and-processing.md#freq_response) (frequency response of the coefficients)
- [quantize_delays](channel-data-and-processing.md#quantize_delays) (delays on a tap grid)

---
<a id="channel.tr38901"></a>
## channel.tr38901
Generate the channels of a TR 38.901 scenario in one call

Also in: [C++](../cpp/channel-generators.md#get_channels_tr38901) · [MATLAB](../mex/channel-generators.md#get_channels_tr38901)

- Channels between the UTs of a drop and the BSs of a 3GPP TR 38.901 scenario (UMa, UMi, RMa, SMa, InH, InF), with the
  given BS and UT antenna arrays; by default one channel per UT and frequency, to its serving BS sector
- The scenario sets the BS layout and the drop of the UTs as in TR 38.901 (table in Notes); optional inputs change
  single values, the time evolution and which links are returned
- Built on the layout functions ([generate_layout](network-layouts-and-mobility.md#generate_layout), [init_link_states](network-layouts-and-mobility.md#init_link_states), [attach_terminals](network-layouts-and-mobility.md#attach_terminals), [generate_rays](network-layouts-and-mobility.md#generate_rays),
  [get_channels_from_rays](network-layouts-and-mobility.md#get_channels_from_rays)); `layout` returns the simulated layout, to continue there

### Usage:
```python
chan, gain_dB, layout, index = quadriga_lib.channel.tr38901( bs_array, ut_array, scenario, center_freq, *, n_ut, \
    seed, spacing, grid, area, bs, ut, indoor, time_grid, los, output, attach, links, link_gain, uplink, \
    o2i_low, config_dirs, stack )
```

### Inputs:
- **`bs_array`** — BS antenna array, arrayant dict, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna); every sector uses it, turned to the sector heading
- **`ut_array`** — UT antenna array, arrayant dict, keys in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna)
- **`scenario`** — `"UMa"`, `"UMi"`, `"RMa"`, `"SMa"`, `"InH_Open"`, `"InH_Mixed"`, `"InF_SL"`, `"InF_DL"`,
  `"InF_SH"`, `"InF_DH"` or `"InF_HH"`; also with the prefix `"38.901_"`
- **`center_freq`** — Carrier frequencies in Hz, ascending; one set of channels per frequency; default: 3.5e9;
  `(n_freq,)`
- **`n_ut`** — UTs per cell (per BS sector), >= 1; default: 1
- **`seed`** — Seed of the random numbers: the same seed gives the same drop and channels; -1 = a random seed,
  which `layout["seed"]` returns; default: -1

Optional changes to the scenario layout (defaults in Notes). Give only the values to change; nan keeps a default.
[generate_layout](network-layouts-and-mobility.md#generate_layout) describes all values:

- **`spacing`** — Inter-site distance in m, e.g. `[200]`; indoor: BS spacing in x and y; default: None
- **`grid`** — Rings of sites around the centre site, e.g. `[0]` = one site, `[2]` = 19 sites; the
  number of sectors as third value, e.g. `[0, nan, 1]`; indoor: BSs in x and y; default: None
- **`area`** — Hall size in m, length, width, height (InH, InF); default: None
- **`bs`** — BS height in m, e.g. `[30]`; then the mechanical downtilt and an offset of the sector
  headings in degrees; default: None
- **`ut`** — UT height and minimum distance to the BS in m, speed in m/s, e.g. `[nan, nan, 30 / 3.6]` =
  all UTs at 30 km/h; then the orientation (bank, tilt, heading) in degrees; default: None
- **`indoor`** — Share of the UTs in buildings, e.g. `[0.5]` (UMa, UMi, RMa, SMa); then the range of floors
  and the floor height; default: None

Simulation settings:

- **`time_grid`** — `[T, dt]` in s: snapshots at 0, dt, 2 dt, … up to T, while the UTs move; `dt`
  defaults to 1 ms, e.g. `[0.1]` = 101 snapshots over 100 ms; `None` = one snapshot (static channel); default: None
- **`los`** — `"auto"`: LOS or NLOS as drawn from the LOS probability of the scenario; `"los"` or `"nlos"`: that state for
  all links, also of the UTs in buildings and to the other sectors, so it can change the serving sector (`"InF_HH"` is
  always LOS); default: `"auto"`
- **`output`** — `"taps"`: the rays of a cluster combined into taps; `"rays"`: every ray as a path, with angles
  and path gains; default: `"taps"`
- **`attach`** — How the serving sector is chosen: `"large_scale"` (coupling gain) or `"rsrp"` (Notes); default: `"large_scale"`
- **`links`** — Which links are returned: `"serving"`, the serving sector of each UT; `"site"`, all sectors of its
  serving site; `"all"`, all BS sectors, e.g. for interference studies (Outputs); default: `"serving"`
- **`link_gain`** — `False`: without path loss, building loss and shadow fading, so that the power is that of the
  antennas and the small-scale fading; `gain_dB` and the serving sector still include them; default: `True`
- **`uplink`** — `True`: the uplink, with the UT as Tx (Notes); default: `False`
- **`o2i_low`** — Share of buildings with the low-loss model of the building penetration loss, in [0, 1];
  nan = the TR 38.901 value (50 % in UMa and UMi); not for InH, InF; default: nan
- **`config_dirs`** — Folders with own scenario files (`.conf`), searched before the shipped ones; default: None
- **`stack`** — Snapshot layout of the channels as in [hdf5_read_channel](channel-data-and-processing.md#hdf5_read_channel); default: `False`

### Outputs:
- **`chan`** — Channels as channel dicts, keys in [hdf5_read_channel](channel-data-and-processing.md#hdf5_read_channel); what taps and rays hold: [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel)
  - `"serving"`: one channel per UT and frequency, all UTs at the first frequency, then at the next: `chan[i_ut + i_freq * n_ut_total]`, 0-based
  - `"site"`, `"all"`: one channel per BS sector, UT and frequency; the sectors of the first UT one after the other,
    then those of the next UT, then the next frequency; `index` names the BS sector, UT and frequency of each channel
  - UTs in drop order: the first `n_ut` were dropped in cell 0 (site BS1, sector 0), the next `n_ut` in cell 1, …;
    the serving sector is the one with the highest gain, often not the cell of the drop. The name says it too, e.g.
    `"BS5:0_UT11"` = site BS5, sector 0, UT 11
  - Coefficients and delays lists of `n_snap` arrays `(n_ut_port, n_bs_port, n_tap)` (one array with `stack`) (downlink; `uplink` swaps the first two)
- **`gain_dB`** — Coupling gain or RSRP in dB of each UT to each BS sector, e.g. for the SIR; `(n_ut_total, n_bs_node)`
- **`layout`** — The simulated layout: BS and UT positions, link states, pairing; keys in [layout.csv_read](network-layouts-and-mobility.md#layout.csv_read)
- **`index`** — BS sector (node), UT and frequency of each channel, 0-based; int64; `(3, n_channel)`

### Example:
```python
import numpy as np
import quadriga_lib as qd
bs = qd.arrayant.generate_antenna( "3gpp", center_freq=3.5e9, panel=[4, 4, 1, 1] )
ut = qd.arrayant.generate_antenna( "xpol" )
chan, gain_dB, layout, index = qd.channel.tr38901( bs, ut, "UMa", 3.5e9, n_ut=10, seed=1 )    # 19 sites
chan, gain_dB, layout, index = qd.channel.tr38901( bs, ut, "UMi", 3.5e9, n_ut=10, seed=1, grid=[0], time_grid=[0.1] )
H = qd.channel.freq_response( chan[0]["coeff"], chan[0]["delay"], np.linspace(0, 1, 64), 20e6 )   # 64 subcarriers
```

### Notes:
- **Scenario defaults** (TR 38.901 Tables 7.2-1 to 7.2-5 and 7.8-7); the UTs are at 1.5 m (InH 1 m, InF_HH 3 m),
  in buildings on their floor:

  | Scenario | BS layout | BS height | UTs indoors | UT speed |
  | --- | --- | --- | --- | --- |
  | UMa | 19 sites × 3 sectors, ISD 500 m, wrap-around | 25 m | 80 % | 3 km/h |
  | UMi | 19 sites × 3 sectors, ISD 200 m, wrap-around | 10 m | 80 % | 3 km/h |
  | RMa | 19 sites × 3 sectors, ISD 1732 m, wrap-around | 35 m | 50 %, the others in cars | 3 km/h, cars 40 km/h |
  | SMa | 19 sites × 3 sectors, ISD 1299 m, wrap-around | 35 m | 80 %, the others in cars | 3 km/h, cars 40 km/h |
  | InH_Open, InH_Mixed | Hall 120 × 50 m, 12 BSs 20 m apart | 3 m | all | 3 km/h |
  | InF_SL, InF_DH | Hall 120 × 60 × 10 m, 18 BSs 20 m apart | 1.5 m (SL), 8 m (DH) | all | 3 km/h |
  | InF_DL, InF_SH, InF_HH | Hall 300 × 150 × 10 m, 18 BSs 50 m apart | 1.5 m (DL), 8 m (SH, HH) | all | 3 km/h |

  `n_ut` UTs are dropped in each cell (`n_ut_total` = `n_ut` × the number of BS sectors); with the wrap-around, the
  outer cells see the interference of a large network, also with one site
- **Serving sector**: the BS sector with the highest coupling gain (`"large_scale"`, TR 38.901 Table 7.8-1) or RSRP
  (`"rsrp"`, Table 7.8-2) at the first frequency, see [attach_terminals](network-layouts-and-mobility.md#attach_terminals); `"rsrp"` and `links` = `"all"` generate
  the rays of all links, the others only those to the serving sites
- **Time evolution**: the UTs move from their drop position along their direction at their speed; delays and angles of
  the clusters stay, the phases follow the motion (mobility `"phase"` of [get_channels_from_rays](network-layouts-and-mobility.md#get_channels_from_rays))
- **Several frequencies**: the clusters come from the same random values at all frequencies, scaled by the spreads of
  each frequency (method `"38.901_MF"`, TR 38.901 clause 7.6.5)
- **Uplink**: the transposed downlink (reciprocal channel, the same Doppler shift), with the two ends swapped in all
  members and the name; `index` and `gain_dB` keep the downlink meaning. This is the uplink of a TDD system; for FDD,
  add the uplink frequency to `center_freq`
- **Memory**: about 90 kB per snapshot for the 57 UTs of the default UMa drop with 2 × 2 ports (16 to 25 taps);
  `links` = `"all"` multiplies it by the number of BS sectors; `"site"` by the sectors per site (3)

### See also:
- [hdf5_read_channel](channel-data-and-processing.md#hdf5_read_channel) (the keys of the channels)
- [freq_response](channel-data-and-processing.md#freq_response) (frequency response of the channels at given subcarriers)
- [generate_layout](network-layouts-and-mobility.md#generate_layout) (the scenario layouts and their inputs in full)
- [get_channels_layout](network-layouts-and-mobility.md#get_channels_layout) (channels of any layout)
- [channel.ieee_indoor](#channel.ieee_indoor) (the IEEE indoor models)
