# Channel generation functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-channel-generation-functions

| Function | Summary |
| --- | --- |
| [channel.ieee_indoor](#channel.ieee_indoor) | Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models |
| [channel.irs](#channel.irs) | Calculate MIMO channel coefficients for IRS-assisted communication |
| [channel.multifreq](#channel.multifreq) | Compute channel coefficients for spherical waves across multiple frequencies |
| [channel.planar](#channel.planar) | Calculate MIMO channel coefficients for planar wave paths |
| [channel.spherical](#channel.spherical) | Calculate MIMO channel coefficients and delays for spherical wave propagation |

---
<a id="channel.ieee_indoor"></a>
## channel.ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

Also in: [C++](../cpp/channel-generation-functions.md#get_channels_ieee_indoor) · [MATLAB](../mex/channel-generation-functions.md#get_channels_ieee_indoor)

- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions
- 2D model: azimuth angles and planar motion only, no elevation
- Supported channel types: `A, B, C, D, E, F` (TGn definitions)
- MU-MIMO supported (`n_users > 1`) with per-user distances/floors and optional angle offsets per TGac
- Time-evolving channels via `observation_time`, `update_rate`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Default KF (linear): A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0; applied to first tap only; breakpoint ignored when `KF_linear >= 0`
- Default XPR NLOS: 2 (3 dB); default SF LOS: 3 dB; default SF NLOS: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB
- Default breakpoint distance: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- Floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz; floors add attenuation
  (and the TGah shadow fading below 1 GHz) but no delay; the station is placed 3 m per floor above the AP
- Taps with zero power in the model tables are removed; the number of paths does not depend on distance, floors or walls
- Time evolution: each sub-path is rotated by `exp(+j 2 pi f_D t)`, i.e. a station moving towards the AP or a scatterer
  sees a positive Doppler shift `f_D`, as the channel of [channel.spherical](#channel.spherical) along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- NAN or negative value for any override parameter restores the model default, except `XPR_NLOS_linear` (see below)
- Per-snapshot data (`coeff`, `delay`, `path_gain`) is returned as a list of per-snapshot arrays, 
  or stacked into one array along an appended snapshot axis when `stack = True`

### Usage:
```python
chan = quadriga_lib.channel.ieee_indoor( ap_array, sta_array, ChannelType, *, CarrierFreq_Hz, \
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, \
   Dist_m, n_floors, uplink, offset_angles, n_subpath, Doppler_effect, seed, \
   KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m, n_walls, wall_loss, stack=False )
```

### Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see arrayant-[generate](array-antenna-functions.md#generate)
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see arrayant-[generate](array-antenna-functions.md#generate)
- **`ChannelType`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency (finite); default: 5.25e9
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** *(optional)* — Channel observation time in seconds; must be finite; default: 0
- **`update_rate`** *(optional)* — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** *(optional)* — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`Dist_m`** *(optional)* — TX-to-RX distance(s) (finite); `(n_users,)` or `(1,)`; default: 4.99
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `(n_users,)` or `(1,)`; default: 0
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: false
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `(4, n_users)`; default: [] (auto-generate)
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** *(optional)* — RNG seed; any other value than `-1` gives the same channels on all platforms; `-1` uses the system random device; default: -1
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default; Inf throws; default: NAN
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN restores model default; must otherwise be finite and > 0 (0, negative and Inf throw); default: NAN
- **`SF_std_dB_LOS`** *(optional)* — Overrides LOS shadow fading std in dB (applied when d < dBP); NAN restores model default; Inf throws; default: NAN
- **`SF_std_dB_NLOS`** *(optional)* — Overrides NLOS shadow fading std in dB (applied when d >= dBP); NAN restores model default; Inf throws; default: NAN
- **`dBP_m`** *(optional)* — Overrides breakpoint distance; NAN or negative restores model default; default: NAN
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `(n_users,)` or `(1,)`; default: 0
- **`wall_loss`** *(optional)* — Penetration loss for a single wall; TGax defines 5 or 7; default: 5
- **`stack`** *(optional)* — If `True`, per-snapshot data is stacked into a single array along an appended 
  snapshot axis instead of being returned as a list of per-snapshot arrays; default: False

### Output:
- **`chan`** — List of length `n_users` containing dictionaries of channel data. Only the keys
  listed below are populated; all other channel keys are omitted by this generator.
  | Key                | Description                                                              | Shape `stack = False`            | Shape `stack = True`           |
  | ------------------ | ------------------------------------------------------------------------ | -------------------------------- | ------------------------------ |
  | `name`             | Channel name                                                             | str                              | str                            |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `coeff`            | Channel coefficients, complex valued                                     | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`            | `(n_path, n_snap)`             |
  | `center_freq` | Center Frequency in Hz                                                   | `(1,)`                           | `(1,)`                         |
  | `initial_position` | 0-based index of the reference snapshot                                  | int                              | int                            |

See also:
- [IEEE 802.11-03/940r4 - TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc)
- [IEEE 802.11-09/0308r12 - TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc)
- [IEEE 802.11-11/0968r4 - TGah Channel Model](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx)
- [IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx)
- [S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025](http://quadriga-lib.org/publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf)
- [hdf5_write_channel](channel-functions.md#hdf5_write_channel) (for writing channel data to a HDF5 file)
- [hdf5_read_channel](channel-functions.md#hdf5_read_channel) (for reading channel data to a HDF5 file)

---
<a id="channel.irs"></a>
## channel.irs
Calculate MIMO channel coefficients for IRS-assisted communication

Also in: [C++](../cpp/channel-generation-functions.md#get_channels_irs) · [MATLAB](../mex/channel-generation-functions.md#get_channels_irs)

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- The IRS coupling matrix holds the IRS weights and must be given: an `irs_array` or `irs_array_2` dict without
  `coupling_re` / `coupling_im` raises, unlike in the other functions, where it stands for identity
- Polarization coupling is applied via the 8-row transfer matrices `polmat_1`, `polmat_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)
- Each segment is computed by [channel.spherical](#channel.spherical): NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`; the pair
  of segment-1 path `i_1` and segment-2 path `i_2` has index `i_1 · n_path_2 + i_2` in `active_path` and
  `active_path_out`, and the output keeps this order
- A segment without paths (all its path inputs empty) gives empty outputs `(n_rx, n_tx, 0)`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual
- If `use_absolute_delays == False`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

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
- **`tx_array`** — Transmit arrayant dict; `n_tx` = number of ports after element coupling; see [generate](array-antenna-functions.md#generate)
- **`rx_array`** — Receive arrayant dict; `n_rx` = number of ports after element coupling; see [generate](array-antenna-functions.md#generate)
- **`irs_array`** — IRS arrayant dict (TX-facing side) with `n_irs` elements; the coupling matrix holds the IRS
  weights, one codebook entry per column, `(n_irs, n_codebook)`; see [generate](array-antenna-functions.md#generate)
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`irs_pos`** — IRS position in Cartesian coordinates; `(3,)`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `(3, n_path_1)`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `(3, n_path_1)`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `(n_path_1,)`
- **`path_length_1`** — Total path lengths from TX to IRS phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `(n_path_1,)`
- **`polmat_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `(8, n_path_1)`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `(3, n_path_2)`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `(3, n_path_2)`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `(n_path_2,)`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `(n_path_2,)`
- **`polmat_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `(8, n_path_2)`
- **`center_freq`** — Center frequency in Hz; 0 gives the coefficients at f = 0 (see above); default: `0.0`
- **`i_irs`** — IRS codebook port index, 0-based (one port per column of the IRS coupling matrix); default: `0`
- **`threshold_dB`** — Gain threshold in dB; path combinations below this are discarded; default: `-140.0`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`angles`** — If `True`, also return the departure and arrival angles; `False` saves their memory; default: `False`
- **`irs_array_2`** — Second IRS arrayant dict for the RX-facing side (asymmetric IRS), with the same number of
  elements and the same coupling size as `irs_array`; default: `None` (`irs_array` for both sides)
- **`active_path`** — Mask selecting active path pairs, bool or numeric (nonzero = active); overrides `threshold_dB`
  when not empty; `(n_path_1 · n_path_2,)`; default: `None`
- **`use_kernel`** — Kernel of both segments, as in [channel.spherical](#channel.spherical): 0 = auto, 1 = GENERIC, 2 = AVX2 (single
  precision; raises if unavailable); auto selects GENERIC in this version; `int`; default: 0

### Outputs:
- **`coeff`** — Complex channel coefficients; `(n_rx, n_tx, n_path_irs)`
- **`delay`** — Propagation delays in seconds; `(n_rx, n_tx, n_path_irs)`
- **`active_path_out`** — Mask of the path combinations in the output (can be passed as `active_path` to a later
  call); `bool`; `(n_path_1 · n_path_2,)`
- **`aod`** — Azimuth of departure in rad; `(n_rx, n_tx, n_path_irs)`; only with `angles=True`
- **`eod`** — Elevation of departure in rad; same shape; only with `angles=True`
- **`aoa`** — Azimuth of arrival in rad; same shape; only with `angles=True`
- **`eoa`** — Elevation of arrival in rad; same shape; only with `angles=True`

### See also:
- [channel.spherical](#channel.spherical) (single-segment spherical-wave channel)
- [channel.planar](#channel.planar) (single-segment planar-wave channel)
- [generate](array-antenna-functions.md#generate) (antenna array generator)

---
<a id="channel.multifreq"></a>
## channel.multifreq
Compute channel coefficients for spherical waves across multiple frequencies

Also in: [C++](../cpp/channel-generation-functions.md#get_channels_multifreq) · [MATLAB](../mex/channel-generation-functions.md#get_channels_multifreq)

- Multi-frequency extension of [channel.spherical](#channel.spherical) with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (`tx_array['center_freq']`), RX array (`rx_array['center_freq']`), input samples (`freq`), and output (`freq_new`)
- Antenna pattern fields may be 3D (single-frequency, clamped for all output frequencies) or 4D (multi-frequency, 4th axis = frequency)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `polmat` is interpolated via SLERP per complex entry pair to preserve phase
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs); pass 3D coupling `(n_elem, n_ports, n_freq)` for per-frequency coupling
- Extrapolation clamps to the nearest frequency entry on all four grids
- `polmat` accepts 8 rows (full polarimetric: the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx),
  rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry
- `propagation_speed` supports EM (speed of light, default) and acoustic (343 m/s) simulations
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 0 holds the LOS path
  (zero power if no LOS path was found; LOS = the path whose length is closest to the TX-RX distance, within 0.1 mm);
  a LOS path found in the input is moved to slot 0 and leaves a zero-power path in its original slot
  (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0, n_freq_new)`; with
  `add_fake_los_path = True` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [quantize_delays](channel-functions.md#quantize_delays) rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [baseband_freq_response](channel-functions.md#baseband_freq_response) with multi-frequency input and `remove_delay_phase = True` removes the phase of `delay`
  before it interpolates across `freq`, so it needs channels generated with `use_absolute_delays = True`; with
  relative delays, the phase of the TX-RX distance stays in the coefficients, is interpolated across the frequency
  grid and gives wrong phases

### Usage:
```python
coeff, delay = quadriga_lib.channel.multifreq( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, \
    freq, freq_new, *, use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel )
```

### Inputs:
- **`tx_array`** — Multi-frequency TX arrayant dict; pattern fields 3D `(n_el, n_az, n_elem)` or 4D `(n_el, n_az, n_elem, n_freq)`; see [generate](array-antenna-functions.md#generate);
  4D input requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`rx_array`** — RX arrayant dict; same format as `tx_array`
- **`tx_pos`** — Transmitter position, finite; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `(3,)`
- **`rx_pos`** — Receiver position, finite; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `(3,)`
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`
- **`path_gain`** — Linear-scale path gains, one column per input frequency; `(n_path, n_freq)`
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `(n_path,)`
- **`polmat`** — Polarization transfer matrix, row order as above; `(8, n_path, n_freq)` full pol or
  `(2, n_path, n_freq)` scalar pressure
- **`freq`** — Input sample frequencies for `path_gain` and `polmat`; `(n_freq,)`
- **`freq_new`** — Target output frequencies; `(n_freq_new,)`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`propagation_speed`** — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; raises if
  unavailable); auto selects GENERIC in this version, see [interpolate](array-antenna-functions.md#interpolate); `int`; default: 0

### Outputs:
- **`coeff`** — Complex channel coefficients; `(n_ports_rx, n_ports_tx, n_path_out, n_freq_new)`
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path_out, n_freq_new)`

### See also:
- [channel.spherical](#channel.spherical) (single-frequency equivalent)
- [generate_speaker](array-antenna-functions.md#generate_speaker) (acoustic source construction)

---
<a id="channel.planar"></a>
## channel.planar
Calculate MIMO channel coefficients for planar wave paths

Also in: [C++](../cpp/channel-generation-functions.md#get_channels_planar) · [MATLAB](../mex/channel-generation-functions.md#get_channels_planar)

- Computes complex channel coefficients and delays for all TX/RX port pairs across `n_path` propagation
  paths under the planar wave assumption.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based; the input angles are not used for LOS detection.
- Polarization coupling is applied via the 8-row transfer matrix `polmat`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `i_freq` selects the frequency index that is processed; a single frequency is returned per call.
- `center_freq = None` or a selected entry of `0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- If `use_absolute_delays == False`, the straight-line TX-RX delay between the phase centres (LOS delay) is subtracted
  from all paths; per-element delays of pairs that are closer to each other than the phase centres are then
  negative, down to about -(half the TX array size + half the RX array size) / speed of light; [quantize_delays](channel-functions.md#quantize_delays) rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.
- If `add_fake_los_path == True`, one output path is always added (output size `n_path + 1`): slot 0 holds the LOS
  path, with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the
  TX-RX distance; it is moved to slot 0 and its original slot (input index + 1) holds a zero-power path. All other
  paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0)`; with `add_fake_los_path = True` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- Errors in the path data of a snapshot start with `Snapshot <i>: ` (0-based), also for a single snapshot
  (`Snapshot 0: `); antenna and `use_kernel` errors have no prefix.

### Batch mode:
- If `aod` is a `list` of arrays, the function switches to batch mode and processes `n_snap` snapshots in one call,
  the snapshots in parallel (OpenMP in C++).
- The path inputs `aod`, `eod`, `aoa`, `eoa`, `path_gain`, `path_length` and `polmat` are then lists of length `n_snap`;
  list entries that are Fortran-contiguous float64 arrays are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_snap)` arrays with one column per snapshot; a `(3,)` array or a
  single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_snap`, one entry per snapshot.
- The GIL is released while processing, so the function can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

### Usage:
```python
coeff, delay, rx_Doppler = quadriga_lib.channel.planar( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, polmat, \
    center_freq, *, use_absolute_delays, add_fake_los_path, use_kernel, i_freq )
```

### Inputs:
- **`tx_array`** — Transmit arrayant dict with single-frequency (3D) patterns, see [generate](array-antenna-functions.md#generate); a 4D dict raises
  `ValueError` (use [channel.multifreq](#channel.multifreq))
- **`rx_array`** — Receive arrayant dict with single-frequency (3D) patterns; as `tx_array`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`, or `(3, n_snap)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`, or `(3, n_snap)` in batch mode
- **`aod`** — Departure azimuth angles in rad; `(n_path,)`, or a list of length `n_snap` in batch mode
- **`eod`** — Departure elevation angles in rad; `(n_path,)`, or a list in batch mode
- **`aoa`** — Arrival azimuth angles in rad; `(n_path,)`, or a list in batch mode
- **`eoa`** — Arrival elevation angles in rad; `(n_path,)`, or a list in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; `(n_path,)`, or a list in batch mode
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the
  coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single precision; raises if
  unavailable); auto selects GENERIC in this version, see [interpolate](array-antenna-functions.md#interpolate); `int`; default: 0
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `polmat` and `center_freq` that is
  processed; Python only, since the C++ function takes single-frequency path data; default: `0`

### Outputs:
- **`coeff`** — Complex channel coefficients; `(n_ports_rx, n_ports_tx, n_path)`
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = toward path, negative = away; `(n_path,)`

### See also:
- [channel.spherical](#channel.spherical) (spherical wave variant)
- [channel.ieee_indoor](#channel.ieee_indoor) (IEEE compliant channels using `channel.planar` internally)
- [generate](array-antenna-functions.md#generate) (antenna array generator)

---
<a id="channel.spherical"></a>
## channel.spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

Also in: [C++](../cpp/channel-generation-functions.md#get_channels_spherical) · [MATLAB](../mex/channel-generation-functions.md#get_channels_spherical)

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using the spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `polmat`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [qrt_file_read](channel-functions.md#qrt_file_read) with more than two interactions) is added to the FBS-LBS segment. Delays and phases follow this
  length; the per-element offsets follow the FBS / LBS geometry.
- Path data is given for `n_freq` frequencies; single-frequency data uses `n_freq = 1`.
- The scalar argument `i_freq` selects the frequency index that is processed; a single frequency is returned per call.
- `center_freq = None` or a selected entry of `0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- If `use_absolute_delays == False`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [quantize_delays](channel-functions.md#quantize_delays) maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == True`, one output path is always added (`n_path + 1` outputs): slot 0 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 0 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `(n_rx, n_tx, 0)`; with `add_fake_los_path = True` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair (also per snapshot in batch
  mode).
- Errors in the path data of a snapshot start with `Snapshot <i>: ` (0-based), also for a single snapshot
  (`Snapshot 0: `); antenna and `use_kernel` errors have no prefix.

### Batch mode:
- If `fbs_pos` is a `list` of arrays, the function switches to batch mode and processes `n_snap` snapshots
  in one call, the snapshots in parallel (OpenMP in C++).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `polmat` are then lists of length `n_snap`,
  matching the output format of [qrt_file_read](channel-functions.md#qrt_file_read); list entries that are Fortran-contiguous float64 arrays (as
  returned by [qrt_file_read](channel-functions.md#qrt_file_read)) are read in place, other entries are copied.
- The position and orientation inputs are `(3, n_snap)` arrays with one column per snapshot; a `(3,)` array
  or a single-column `(3, 1)` array is applied to all snapshots.
- All outputs are returned as lists of length `n_snap`, one entry per snapshot.
- Snapshots without any path produce an empty output entry with `n_path = 0`, or only the zero-power LOS path
  when `add_fake_los_path` is set (as for a single snapshot).
- The GIL is released while processing, so the function can be combined with Python threads. The function keeps
  references to all input arrays until it returns, but other threads must not write into them meanwhile.

### Usage:
```python
coeff, delay = quadriga_lib.channel.spherical( tx_array, rx_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, \
    center_freq, *, use_absolute_delays, add_fake_los_path, angles, use_kernel, i_freq )

# Departure and arrival angles as well
coeff, delay, aod, eod, aoa, eoa = quadriga_lib.channel.spherical( ..., angles=True )
```

### Inputs:
- **`tx_array`** — Transmit arrayant dict with single-frequency (3D) patterns, see [generate](array-antenna-functions.md#generate); a 4D dict raises
  `ValueError` (use [channel.multifreq](#channel.multifreq))
- **`rx_array`** — Receive arrayant dict with single-frequency (3D) patterns; as `tx_array`
- **`tx_pos`** — Transmitter position in Cartesian coordinates, finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_pos`** — Receiver position in Cartesian coordinates, finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `(3,)`, or `(3, n_snap)` in batch mode
- **`fbs_pos`** — First-bounce scatterer positions; `(3, n_path)`, or a list of length `n_snap` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `(3, n_path)`, or a list of length `n_snap` in batch mode
- **`path_gain`** — Path gains in linear scale; `(n_path, n_freq)`, or a list of length `n_snap` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `(n_path,)`, or a list of length `n_snap` in batch mode
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `(8, n_path, n_freq)`, or a list of length `n_snap`
  in batch mode
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`; an entry of `0` or `None` gives the coefficients at f = 0 (see above); default: `None`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`add_fake_los_path`** — If `True`, always adds one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; default: `False`
- **`angles`** — If `True`, also return the departure and arrival angles; `False` saves their memory and, with element
  coupling, the time to convert them to the ports; default: `False`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto, 1 = GENERIC,
  2 = AVX2 (single precision; raises if unavailable); auto selects GENERIC in this version, see [interpolate](array-antenna-functions.md#interpolate);
  `int`; default: 0
- **`i_freq`** — Frequency index (0-based) selecting the slice of `path_gain`, `polmat` and `center_freq` that is
  processed; Python only, since the C++ function takes single-frequency path data; default: `0`

### Outputs:
- **`coeff`** — Complex channel coefficients; `(n_ports_rx, n_ports_tx, n_path)`
- **`delay`** — Propagation delays in seconds; `(n_ports_rx, n_ports_tx, n_path)`
- **`aod`** — Azimuth of departure in the GCS, in rad; `(n_ports_rx, n_ports_tx, n_path)`; only with `angles=True`
- **`eod`** — Elevation of departure in the GCS, in rad; same shape; only with `angles=True`
- **`aoa`** — Azimuth of arrival in the GCS, in rad; same shape; only with `angles=True`
- **`eoa`** — Elevation of arrival in the GCS, in rad; same shape; only with `angles=True`
- In batch mode, every output is a list of length `n_snap` whose entries have the shapes given above.

### See also:
- [channel.multifreq](#channel.multifreq) (multi-frequency extension)
- [channel.planar](#channel.planar) (planar wave variant)
- [qrt_file_read](channel-functions.md#qrt_file_read) (produces the list format used by batch mode)
