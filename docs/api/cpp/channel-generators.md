# Channel generators

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-channel-generators

| Function | Summary |
| --- | --- |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models |
| [get_channels_irs](#get_channels_irs) | Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface |
| [get_channels_multifreq](#get_channels_multifreq) | Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies |
| [get_channels_planar](#get_channels_planar) | Calculate MIMO channel coefficients and delays for planar wave propagation |
| [get_channels_spherical](#get_channels_spherical) | Calculate MIMO channel coefficients and delays for spherical wave propagation |

The functions of this section compute the channel coefficients of a link from its propagation paths and the antenna
arrays at both ends. They apply the antenna patterns, the polarization and the motion of the antennas, and return the
coefficients and delays that the functions of section "Channel data and processing" work with. The coefficients are
given per port pair: `n_tx` and `n_rx` count the ports of the arrays, after their port weights.

**Generators:**

- [get_channels_spherical](#get_channels_spherical): paths given by their first- and last-bounce scatterers; each antenna element sees the path
  from its own position (spherical wavefronts), so it covers large arrays and short distances
- [get_channels_planar](#get_channels_planar): paths given by their departure and arrival angles; all elements of an array see the same
  direction (planar wavefronts)
- [get_channels_multifreq](#get_channels_multifreq): as [get_channels_spherical](#get_channels_spherical), at several frequencies with frequency-dependent antennas,
  path gains and polarization, for radio and acoustic waves
- [get_channels_irs](#get_channels_irs): paths over an intelligent reflecting surface (IRS), from two path sets that meet at the IRS array
- [get_channels_ieee_indoor](#get_channels_ieee_indoor): complete channels of the IEEE TGn, TGac, TGah and TGax indoor models

The paths come from a ray tracer (e.g. [qrt_file_read](channel-data-and-processing.md#qrt_file_read)), from the geometry-based stochastic model, or from the user.
For complete channels of a network layout, see [get_channels_layout](network-layouts-and-mobility.md#get_channels_layout).

**Terms** (full list in the [Glossary](general-usage-notes.md#glossary)):

- **Path gain**: the linear power gain of a path without antennas, `[n_path]`
- **Path length**: the length of a path from the Tx to the Rx phase center in m, which sets its delay
- **Polarization transfer matrix** `polmat`: the Jones matrix of each path, `[8, n_path]` with interleaved real and
  imaginary parts (Glossary, "Polarization")
- **Absolute delays**: `use_absolute_delays = true` keeps the LOS delay in all paths; otherwise the delays are
  relative to the LOS path
- **Vector form**: [get_channels_planar](#get_channels_planar) and [get_channels_spherical](#get_channels_spherical) have same-name overloads that compute
  several snapshots at once, one entry per snapshot

---
<a id="get_channels_ieee_indoor"></a>
## get_channels_ieee_indoor
Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models

Also in: [MATLAB](../mex/channel-generators.md#get_channels_ieee_indoor) · [Python](../python/channel-generators.md#channel.ieee_indoor)

- Channels of the IEEE 802.11 indoor models A to F (TGn), with the TGac extensions (several users with offset angles,
  tap interpolation for shorter tap spacings), the floor loss of TGah below 1 GHz and the floor and wall loss of TGax
  above; one channel per user
- 2D model: azimuth angles only and motion in the horizontal plane; [get_channels_planar](#get_channels_planar) computes the coefficients
  with the AP and station arrays
- Static channels (`observation_time = 0`, one snapshot) or time-evolving channels with the Doppler shifts of the
  station and the environment

### Declaration:
```cpp
std::vector<quadriga_lib::channel<double>> quadriga_lib::get_channels_ieee_indoor(
    const quadriga_lib::arrayant<double> &ap_array,
    const quadriga_lib::arrayant<double> &sta_array,
    const std::string &channel_type,
    double center_freq = 5.25e9,
    double tap_spacing = 10.0e-9,
    arma::uword n_users = 1,
    double observation_time = 0.0,
    double update_interval = 1.0e-3,
    double speed_station_kmh = 0.0,
    double speed_env_kmh = 1.2,
    const arma::vec &dist = arma::vec(),
    const uint_list &n_floors = {},
    bool uplink = false,
    const arma::mat &offset_angles_deg = arma::mat(),
    arma::uword n_subpath = 20,
    double doppler_effect = 50.0,
    arma::sword seed = -1,
    double kf = NAN,
    double xpr_nlos = NAN,
    double sf_std_los_dB = NAN,
    double sf_std_nlos_dB = NAN,
    double dist_breakpoint = NAN,
    const uint_list &n_walls = {},
    double wall_loss_dB = 5.0);
```

### Inputs:
- **`ap_array`**, **`sta_array`** — AP and station array, valid ([arrayant](antennas-transducers-and-patterns.md#arrayant)); `n_ap` and `n_sta` ports
- **`channel_type`** — Model: `"A"`, `"B"`, `"C"`, `"D"`, `"E"` or `"F"`
- **`center_freq`** *(optional)* — Carrier frequency in Hz, finite, above 100 MHz
- **`tap_spacing`** *(optional)* — Tap spacing in s, 10 ns divided by a power of 2
- **`n_users`** *(optional)* — Number of users, >= 1
- **`observation_time`** *(optional)* — Observation time in s, finite, >= 0; 0: one snapshot
- **`update_interval`** *(optional)* — Time between snapshots in s, finite, > 0
- **`speed_station_kmh`** *(optional)* — Station speed in km/h, finite
- **`speed_env_kmh`** *(optional)* — Speed of the environment in km/h, finite; TGn 1.2, TGac 0.089
- **`dist`** *(optional)* — Horizontal AP-station distance in m per user, finite, > 0; one value for all users or
  `[n_users]`; empty: 4.99 m
- **`n_floors`** *(optional)* — Floors between AP and station per user, 0 to 4; one value or `[n_users]`; empty: 0
- **`uplink`** *(optional)* — Return the uplink, with the station as Tx
- **`offset_angles_deg`** *(optional)* — Offset angles in degrees per user, finite; rows: AoD LOS, AoD NLOS, AoA LOS,
  AoA NLOS; `[4, n_users]`; empty: 0 for one user, the pseudo-random offsets of TGac for several
- **`n_subpath`** *(optional)* — Sub-paths per tap, >= 1
- **`doppler_effect`** *(optional)* — Special Doppler (Notes): the mains frequency in Hz for models D and E, the vehicle
  speed in km/h for model F; finite, >= 0; 0 disables it
- **`seed`** *(optional)* — Seed of the random numbers; -1: the system random device; other values give the same
  channels on all platforms
- **`kf`** *(optional)* — Ricean K-factor of the first tap, linear, finite, >= 0; NaN: model default (Notes)
- **`xpr_nlos`** *(optional)* — Cross-polarization ratio of the NLOS paths, linear, finite, > 0; NaN: model default
- **`sf_std_los_dB`**, **`sf_std_nlos_dB`** *(optional)* — Standard deviation of the shadow fading in dB below and above
  the breakpoint distance, finite, >= 0; NaN: model default
- **`dist_breakpoint`** *(optional)* — Breakpoint distance in m, finite, > 0; NaN: model default
- **`n_walls`** *(optional)* — Walls between AP and station per user (TGax); one value or `[n_users]`; empty: 0
- **`wall_loss_dB`** *(optional)* — Loss of one wall in dB, finite; TGax: 5 or 7

### Returns:
- One [channel](channel-data-and-processing.md#channel) per user, `[n_users]`, with `name` (`"user_<i>"`), `center_freq`, the positions and orientations
  (station positions `[3, n_snap]`), and per snapshot `coeff_re`, `coeff_im`, `delay` `[n_rx, n_tx, n_path]` (downlink
  `[n_sta, n_ap, n_path]`) and `path_gain` `[n_path]`: the path powers without antennas, with path loss and shadow
  fading, whose mean over the snapshots is the power of the model

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
  toward the AP or a scatterer gives a positive $f_D$, as [get_channels_spherical](#get_channels_spherical) along the returned station
  positions
- **Special Doppler** (`doppler_effect` > 0): models D and E modulate three taps with fluorescent light at the
  harmonics of the mains frequency, model F adds the reflection of a moving vehicle to one tap (TGn)
- **Uplink**: the coefficients and delays are the transposed downlink ones (reciprocal channel, no conjugation, the same
  Doppler shift); the positions and orientations are swapped, the station is the Tx

### See also:
- [get_channels_planar](#get_channels_planar) (computes the coefficients)
- [IEEE 802.11-03/940r4 - TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc)
- [IEEE 802.11-09/0308r12 - TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc)
- [IEEE 802.11-11/0968r4 - TGah Channel Model](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx)
- [IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx)
- [S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025](http://quadriga-lib.org/publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf)

---
<a id="get_channels_irs"></a>
## get_channels_irs
Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface

Also in: [MATLAB](../mex/channel-generators.md#get_channels_irs) · [Python](../python/channel-generators.md#channel.irs)

- Paths of one link over an intelligent reflecting surface (IRS) with `n_irs` elements: segment 1 from the Tx to the
  IRS and segment 2 from the IRS to the Rx, each computed by [get_channels_spherical](#get_channels_spherical) with the IRS elements as Rx and
  Tx elements; each output path combines one path of each segment
- Each IRS element $k$ reflects once with its weight $w_k$, column `i_irs` of the IRS port weights (a codebook entry):
  per path pair, $\mathbf{H} = \mathbf{B} \, \mathrm{diag}(\mathbf{w}) \, \mathbf{A}$ with the element channels
  $\mathbf{A}$ of segment 1 `[n_irs, n_tx]` and $\mathbf{B}$ of segment 2 `[n_rx, n_irs]`
- The delay of a pair is the sum of its segment delays, each the $|w_k|^2$-weighted mean over the IRS elements; the
  departure angles come from segment 1, the arrival angles from segment 2, as $|w_k|^2$-weighted mean directions
- Kept are the path pairs whose strongest coefficient has a power above `threshold_dB`, or those set in `active_path`;
  the pair of segment-1 path $i_1$ and segment-2 path $i_2$ has the index $i_1 n_{\mathrm{path},2} + i_2$ in
  `active_path` and `active_path_out`, and the output keeps this order

### Declaration:
```cpp
void quadriga_lib::get_channels_irs(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const quadriga_lib::arrayant<dtype> &irs_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Mat<dtype> &irs_pos,
    const arma::Mat<dtype> &irs_orientation,
    const arma::Mat<dtype> &fbs_pos_1,
    const arma::Mat<dtype> &lbs_pos_1,
    const arma::Col<dtype> &path_gain_1,
    const arma::Col<dtype> &path_length_1,
    const arma::Mat<dtype> &polmat_1,
    const arma::Mat<dtype> &fbs_pos_2,
    const arma::Mat<dtype> &lbs_pos_2,
    const arma::Col<dtype> &path_gain_2,
    const arma::Col<dtype> &path_length_2,
    const arma::Mat<dtype> &polmat_2,
    arma::Cube<dtype> &coeff_re,
    arma::Cube<dtype> &coeff_im,
    arma::Cube<dtype> &delay,
    arma::uword i_irs = 0,
    double threshold_dB = -140.0,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    arma::Cube<dtype> *aod = nullptr,
    arma::Cube<dtype> *eod = nullptr,
    arma::Cube<dtype> *aoa = nullptr,
    arma::Cube<dtype> *eoa = nullptr,
    const quadriga_lib::arrayant<dtype> &irs_array_2 = quadriga_lib::arrayant<dtype>(),
    const std::vector<bool> &active_path = std::vector<bool>(),
    std::vector<bool> *active_path_out = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, valid ([arrayant](antennas-transducers-and-patterns.md#arrayant)); `n_tx` and `n_rx` ports
- **`irs_array`** — IRS array with `n_irs` elements, valid, with port weights `[n_irs, n_codebook]` that hold the
  reflection weights; its patterns serve both segments unless `irs_array_2` is given
- **`tx_pos`**, **`rx_pos`**, **`irs_pos`** — Tx, Rx and IRS position in m, finite; `[3, 1]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`**, **`irs_orientation`** — Orientations, Euler angles (bank, tilt, heading)
  in rad, finite; `[3, 1]` or `[1, 3]`
- **`fbs_pos_1`**, **`lbs_pos_1`** — First- and last-bounce scatterer positions of the segment-1 paths in m;
  `[3, n_path_1]`
- **`path_gain_1`** — Path gains of segment 1, linear power; `[n_path_1]`
- **`path_length_1`** — Path lengths of segment 1 in m between the Tx and IRS phase centres, as in
  [get_channels_spherical](#get_channels_spherical); `[n_path_1]`
- **`polmat_1`** — Jones matrices of segment 1, the IRS receiving ([Glossary](general-usage-notes.md#glossary), "Polarization");
  `[8, n_path_1]`
- **`fbs_pos_2`**, **`lbs_pos_2`**, **`path_gain_2`**, **`path_length_2`**, **`polmat_2`** — The same for segment 2
  from the IRS to the Rx, the IRS transmitting; `n_path_2` paths
- **`i_irs`** *(optional)* — Codebook entry, 0-based column of the IRS port weights, below `n_codebook`
- **`threshold_dB`** *(optional)* — Minimum power in dB of the strongest coefficient of a kept path pair; not used with
  `active_path`
- **`center_freq`** *(optional)* — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0, as in
  [get_channels_spherical](#get_channels_spherical)
- **`use_absolute_delays`** *(optional)* — Keep the delay of the direct Tx-Rx distance in all delays; `false`: delays
  relative to it (Notes)
- **`irs_array_2`** *(optional)* — Patterns of the IRS for segment 2 (asymmetric IRS), valid, `n_irs` elements; its
  port weights and element positions are not used; empty: `irs_array`
- **`active_path`** *(optional)* — Path pairs to keep, replacing `threshold_dB`; empty: `threshold_dB`;
  `[n_path_1 * n_path_2]`
- **`use_kernel`** *(optional)* — Kernel of both segments, as in [get_channels_spherical](#get_channels_spherical)

### Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_path_irs]` with the
  number of kept path pairs `n_path_irs`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path_irs]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** *(optional)* — Kept path pairs, for `active_path` of a later call or [combine_irs_coord](site-specific-simulation-tools.md#combine_irs_coord);
  `[n_path_1 * n_path_2]`

### Notes:
- **IRS port weights**: empty port weights throw here, unlike in the other functions, where they mean identity; the
  IRS elements are not combined into ports
- **Segments**: path lengths, LOS paths and the per-element geometry as in [get_channels_spherical](#get_channels_spherical); a segment
  without paths (all its path inputs empty) gives the outputs `[n_rx, n_tx, 0]`
- **Relative delays**: the delay of the direct Tx-Rx distance (not via the IRS) is subtracted; delays can be slightly
  negative when the IRS is close to the Tx-Rx line

### See also:
- [get_channels_spherical](#get_channels_spherical) (the channel of each segment)
- [combine_irs_coord](site-specific-simulation-tools.md#combine_irs_coord) (interaction coordinates of the kept path pairs)

---
<a id="get_channels_multifreq"></a>
## get_channels_multifreq
Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies

Also in: [MATLAB](../mex/channel-generators.md#get_channels_multifreq) · [Python](../python/channel-generators.md#channel.multifreq)

- As [get_channels_spherical](#get_channels_spherical), at the output frequencies `freq_new`, with multi-frequency arrays (frequency-dependent
  patterns and port weights), path gains and Jones matrices; for radio waves and, with `propagation_speed` = 343 m/s and
  scalar `polmat`, for sound
- The geometry (element-pair path lengths, angles, LOS path) is computed once; at each output frequency, the patterns
  and port weights of the arrays and `path_gain` and `polmat` are interpolated between their two neighbouring entries
- The coefficients follow the formula of [get_channels_spherical](#get_channels_spherical) with $f_c$ = `freq_new[k]` and $c$ =
  `propagation_speed`; the output has one cube per output frequency, a multi-frequency channel

### Declaration:
```cpp
void quadriga_lib::get_channels_multifreq(
    const std::vector<arrayant<dtype>> &tx_array,
    const std::vector<arrayant<dtype>> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Mat<dtype> &fbs_pos,
    const arma::Mat<dtype> &lbs_pos,
    const arma::Mat<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Cube<dtype> &polmat,
    const arma::Col<dtype> &freq,
    const arma::Col<dtype> &freq_new,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    double propagation_speed = 299792458.0,
    std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx multi-frequency array, valid ([is_valid](antennas-transducers-and-patterns.md#is_valid)), at least one entry, in any
  frequency order; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `[3, 1]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; `[3, 1]` or `[1, 3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`
- **`path_gain`** — Path gains at the frequencies `freq`, linear power; `[n_path, n_freq]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres, as in [get_channels_spherical](#get_channels_spherical);
  `[n_path]`
- **`polmat`** — Jones matrices at the frequencies `freq` ([Glossary](general-usage-notes.md#glossary), "Polarization"):
  `[8, n_path, n_freq]`, or `[2, n_path, n_freq]` for a scalar coefficient (real and imaginary part, sound)
- **`freq`** — Sample frequencies of `path_gain` and `polmat` in Hz, in any order; `[n_freq]`, at least one
- **`freq_new`** — Output frequencies in Hz; `[n_freq_new]`, at least one
- **`use_absolute_delays`** *(optional)* — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative
  to it
- **`add_fake_los_path`** *(optional)* — Put the LOS path into an added slot 0, with zero power if there is none
- **`propagation_speed`** *(optional)* — Wave speed in m/s, > 0
- **`use_kernel`** *(optional)* — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC,
  2 = AVX2 (single-precision arithmetic, also for `double`; throws if unavailable)

### Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_freq_new]` cubes
  `[n_rx, n_tx, n_out]`, `n_out = n_path`, or `n_path + 1` with `add_fake_los_path`
- **`delay`** — Delays in s; sizes as `coeff_re`
- **`coeff`** *(optional)* — Complex coefficients, replacing `coeff_re` / `coeff_im` (then left empty)

### Notes:
- **Interpolation**: `path_gain` linear; patterns, port weights and `polmat` per complex value, the amplitude linear and
  the phase along the shorter arc (linear for nearly opposite phases); constant outside the range of each grid
- **Scalar `polmat`**: only the $F_\theta$ patterns are used,
  $a_l = \sqrt{g_l} \, F_{\theta,r} \, m_l \, F_{\theta,t} \, e^{-j 2\pi f d / c}$ with the complex scalar $m_l$
- **Paths**: NLOS path length, LOS path, `add_fake_los_path` and 0 paths as in [get_channels_spherical](#get_channels_spherical)
- **Relative delays**: negative delays as in [get_channels_spherical](#get_channels_spherical), with $c$ = `propagation_speed`; for sound, this
  affects practically every array, so use absolute delays for [quantize_delays](channel-data-and-processing.md#quantize_delays)
- **Delay phase**: the coefficients carry the phase of the absolute path length also with relative delays;
  [freq_response_mf](channel-data-and-processing.md#freq_response_mf) with `remove_delay_phase = true` therefore needs `use_absolute_delays = true`

### See also:
- [get_channels_spherical](#get_channels_spherical) (one frequency)
- [freq_response_mf](channel-data-and-processing.md#freq_response_mf) (frequency response of a multi-frequency channel)
- [join_arrays](antennas-transducers-and-patterns.md#join_arrays) (multi-frequency arrays)
- [generate_speaker](antennas-transducers-and-patterns.md#generate_speaker) (loudspeaker arrays)

---
<a id="get_channels_planar"></a>
## get_channels_planar
Calculate MIMO channel coefficients and delays for planar wave propagation

Also in: [MATLAB](../mex/channel-generators.md#get_channels_planar) · [Python](../python/channel-generators.md#channel.planar)

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; a path is given by its
  departure and arrival angles, which all elements of an array share (planar wavefronts)
- Path $l$ of the element pair $(t, r)$ has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \quad \tau_l = d / c, \quad d = \ell_l + d_t + d_r$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  at the path angles, the path length $\ell_l$ between the phase centres (used as given), the distances $d_t$, $d_r$ of
  the elements from the wavefront planes through the phase centres (`dist` of [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern)), the center
  frequency $f_c$ and the speed of light $c$; the port weights then combine the elements into ports
- The positions enter only through the Tx-Rx distance (relative delays, LOS path)
- Vector form: the path inputs and the outputs have one entry per snapshot; the snapshots are computed in parallel

### Declarations:
```cpp
void quadriga_lib::get_channels_planar(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Col<dtype> &aod,
    const arma::Col<dtype> &eod,
    const arma::Col<dtype> &aoa,
    const arma::Col<dtype> &eoa,
    const arma::Col<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Mat<dtype> &polmat,
    arma::Cube<dtype> &coeff_re,
    arma::Cube<dtype> &coeff_im,
    arma::Cube<dtype> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    arma::Col<dtype> *rx_Doppler = nullptr,
    int use_kernel = 0);

void quadriga_lib::get_channels_planar(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const std::vector<arma::Col<dtype>> &aod,
    const std::vector<arma::Col<dtype>> &eod,
    const std::vector<arma::Col<dtype>> &aoa,
    const std::vector<arma::Col<dtype>> &eoa,
    const std::vector<arma::Col<dtype>> &path_gain,
    const std::vector<arma::Col<dtype>> &path_length,
    const std::vector<arma::Mat<dtype>> &polmat,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    std::vector<arma::Col<dtype>> *rx_Doppler = nullptr,
    std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, valid ([arrayant](antennas-transducers-and-patterns.md#arrayant)); `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `[3, 1]` or `[1, 3]`; vector form: `[3, 1]` (all
  snapshots) or `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`aod`**, **`eod`** — Azimuth and elevation of departure in rad, GCS; `[n_path]`
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in rad, GCS (the direction from the Rx toward the incoming
  path); `[n_path]`
- **`path_gain`** — Path gains, linear power; `[n_path]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; `[n_path]`
- **`polmat`** — Jones matrices, real and imaginary parts interleaved ([Glossary](general-usage-notes.md#glossary),
  "Polarization"); `[8, n_path]`
- **`center_freq`** *(optional)* — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes)
- **`use_absolute_delays`** *(optional)* — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative
  to it (Notes)
- **`add_fake_los_path`** *(optional)* — Put the LOS path into an added slot 0, with zero power if there is none (Notes)
- **`use_kernel`** *(optional)* — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC,
  2 = AVX2 (single-precision arithmetic, also for `double`; throws if unavailable)

### Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_out]` with
  `n_out = n_path`, or `n_path + 1` with `add_fake_los_path`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`rx_Doppler`** *(optional)* — Doppler weight of each path for an Rx moving along its local x-axis: the cosine of the
  angle between this axis and the arrival direction; the Doppler shift at the speed $v$ is $v f_c / c$ times this
  weight, positive when the Rx moves toward the incoming path; `[n_out]`
- **`coeff`** *(optional)* — Vector form: complex coefficients, replacing `coeff_re` / `coeff_im` (then left empty)

### Notes:
- **LOS path** (only with `add_fake_los_path`): the strongest path whose `path_length` is within 0.1 mm of the Tx-Rx
  distance; the angles are not checked
- **`add_fake_los_path`**: slot 0 holds the LOS path, with zero power (and `rx_Doppler` 0) if there is none; a LOS path
  among the inputs moves to slot 0 and leaves a zero-power path in its own slot (input index + 1); the other paths
  keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays; element pairs can then get
  negative delays, as in [get_channels_spherical](#get_channels_spherical)
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays of a port pair are means over its element pairs, weighted by the port-weight powers
- **Vector form**: as in [get_channels_spherical](#get_channels_spherical)

### See also:
- [get_channels_spherical](#get_channels_spherical) (paths given by scatterers, spherical wavefronts)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (IEEE indoor channels, computed with this function)
- [freq_response](channel-data-and-processing.md#freq_response) (frequency response of the coefficients)
- [quantize_delays](channel-data-and-processing.md#quantize_delays) (delays on a tap grid)

---
<a id="get_channels_spherical"></a>
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

Also in: [MATLAB](../mex/channel-generators.md#get_channels_spherical) · [Python](../python/channel-generators.md#channel.spherical)

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; every element sees a path
  from its own position (spherical wavefronts), so delays and phases follow the geometry of each element pair
- A path runs from the Tx over its first-bounce scatterer (FBS) and last-bounce scatterer (LBS) to the Rx; the patterns
  of both arrays are interpolated at the departure and arrival angles of each element pair ([interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern))
- Path $l$ of an element pair at the distance $d$ along the path has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \qquad \tau_l = d / c$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  of the Rx and Tx element, the center frequency $f_c$ and the speed of light $c$; the port weights then combine the
  elements into ports
- Vector form: the path inputs and the outputs have one entry per snapshot; the snapshots are computed in parallel

### Declarations:
```cpp
void quadriga_lib::get_channels_spherical(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Mat<dtype> &fbs_pos,
    const arma::Mat<dtype> &lbs_pos,
    const arma::Col<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Mat<dtype> &polmat,
    arma::Cube<dtype> &coeff_re,
    arma::Cube<dtype> &coeff_im,
    arma::Cube<dtype> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    arma::Cube<dtype> *aod = nullptr,
    arma::Cube<dtype> *eod = nullptr,
    arma::Cube<dtype> *aoa = nullptr,
    arma::Cube<dtype> *eoa = nullptr,
    int use_kernel = 0);

void quadriga_lib::get_channels_spherical(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const std::vector<arma::Mat<dtype>> &fbs_pos,
    const std::vector<arma::Mat<dtype>> &lbs_pos,
    const std::vector<arma::Col<dtype>> &path_gain,
    const std::vector<arma::Col<dtype>> &path_length,
    const std::vector<arma::Mat<dtype>> &polmat,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    std::vector<arma::Cube<dtype>> *aod = nullptr,
    std::vector<arma::Cube<dtype>> *eod = nullptr,
    std::vector<arma::Cube<dtype>> *aoa = nullptr,
    std::vector<arma::Cube<dtype>> *eoa = nullptr,
    std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, valid ([arrayant](antennas-transducers-and-patterns.md#arrayant)); `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `[3, 1]` or `[1, 3]`; vector form: `[3, 1]` (all
  snapshots) or `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`
- **`path_gain`** — Path gains, linear power; `[n_path]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; a value below the Tx-FBS-LBS-Rx polyline
  (e.g. 0) gives the polyline length (Notes); `[n_path]`
- **`polmat`** — Jones matrices, real and imaginary parts interleaved ([Glossary](general-usage-notes.md#glossary),
  "Polarization"); `[8, n_path]`
- **`center_freq`** *(optional)* — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes)
- **`use_absolute_delays`** *(optional)* — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative
  to it (Notes)
- **`add_fake_los_path`** *(optional)* — Put the LOS path into an added slot 0, with zero power if there is none (Notes)
- **`use_kernel`** *(optional)* — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto
  (GENERIC in this version), 1 = GENERIC, 2 = AVX2 (single-precision arithmetic, also for `double`; throws if
  unavailable)

### Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_out]` with
  `n_out = n_path`, or `n_path + 1` with `add_fake_los_path`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_out]`
- **`coeff`** *(optional)* — Vector form: complex coefficients, replacing `coeff_re` / `coeff_im` (then left empty)

### Notes:
- **NLOS path length**: the length between the phase centres is $\max(\ell, d_\mathrm{poly})$ with `path_length`
  $\ell$ and the length $d_\mathrm{poly}$ of the Tx-FBS-LBS-Rx polyline; the excess, e.g. from more than two
  interactions as in [qrt_file_read](channel-data-and-processing.md#qrt_file_read) or [coord2path](site-specific-simulation-tools.md#coord2path), is added to the FBS-LBS segment
- **LOS path**: the path whose length is closest to the Tx-Rx distance, within 0.1 mm; its element pairs use the
  direct line between the elements instead of the FBS and LBS
- **`add_fake_los_path`**: slot 0 holds the LOS path, with zero power if there is none; a LOS path among the inputs moves
  to slot 0 and leaves a zero-power path in its own slot (input index + 1); the other paths keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays. Element pairs closer to each other
  than the phase centres then get negative delays, down to about $-(D_t + D_r) / (2c)$ with the array sizes $D$;
  [quantize_delays](channel-data-and-processing.md#quantize_delays) rejects delays at or below `-tap_spacing / 2`, so large arrays need absolute delays
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays and angles of a port pair are means over its element pairs, weighted by the
  port-weight powers (the angles via the mean direction)
- **Vector form**: `n_snap` is the length of the path inputs; entries of the output vectors that already have the output
  size are written in place. An error names the lowest failing snapshot (`Snapshot <i>: `, 0-based); invalid arrays and
  `use_kernel` throw without this prefix, also for `n_snap = 0`

### See also:
- [get_channels_planar](#get_channels_planar) (paths given by angles, planar wavefronts)
- [get_channels_multifreq](#get_channels_multifreq) (several frequencies)
- [get_channels_irs](#get_channels_irs) (paths over an IRS)
- [qrt_file_read](channel-data-and-processing.md#qrt_file_read) (path data of many snapshots)
- [freq_response](channel-data-and-processing.md#freq_response) (frequency response of the coefficients)
- [quantize_delays](channel-data-and-processing.md#quantize_delays) (delays on a tap grid)
