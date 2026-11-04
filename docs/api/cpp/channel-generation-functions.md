# Channel generation functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-channel-generation-functions

| Function | Summary |
| --- | --- |
| [get_channels_ieee_indoor](#get_channels_ieee_indoor) | Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models |
| [get_channels_irs](#get_channels_irs) | Calculate MIMO channel coefficients for IRS-assisted communication |
| [get_channels_multifreq](#get_channels_multifreq) | Compute channel coefficients for spherical waves across multiple frequencies |
| [get_channels_planar](#get_channels_planar) | Calculate MIMO channel coefficients for planar wave paths |
| [get_channels_spherical](#get_channels_spherical) | Calculate MIMO channel coefficients and delays for spherical wave propagation |

---
<a id="get_channels_ieee_indoor"></a>
## get_channels_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

Also in: [MATLAB](../mex/channel-generation-functions.md#get_channels_ieee_indoor) · [Python](../python/channel-generation-functions.md#get_ieee_indoor)

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
  sees a positive Doppler shift `f_D`, as the channel of [get_channels_spherical](#get_channels_spherical) along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- NAN or negative value for any override parameter restores the model default, except `XPR_NLOS_linear` (see below)

### Declaration:
```cpp
std::vector<quadriga_lib::channel<double>> quadriga_lib::get_channels_ieee_indoor(
    const quadriga_lib::arrayant<double> &ap_array,
    const quadriga_lib::arrayant<double> &sta_array,
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
    double dBP_m = NAN,
    arma::uvec n_walls = {0},
    double wall_loss = 5.0);
```

### Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see [arrayant](array-antenna-class.md#arrayant)
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see [arrayant](array-antenna-class.md#arrayant)
- **`ChannelType`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency (finite)
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`
- **`observation_time`** *(optional)* — Channel observation time in seconds; must be finite
- **`update_rate`** *(optional)* — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`
- **`speed_station_kmh`** *(optional)* — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`
- **`Dist_m`** *(optional)* — TX-to-RX distance(s) (finite); `[n_users]` or `[1]`
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `[n_users]` or `[1]`
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; empty uses TGac auto-defaults for `n_users > 1`; `[4, n_users]`
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); `0.0` disables
- **`seed`** *(optional)* — RNG seed for repeatability; `-1` uses the system random device
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN restores model default; must otherwise be finite and > 0 (0, negative and Inf throw)
- **`SF_std_dB_LOS`** *(optional)* — Overrides LOS shadow fading std in dB (applied when d < dBP); NAN restores model default
- **`SF_std_dB_NLOS`** *(optional)* — Overrides NLOS shadow fading std in dB (applied when d >= dBP); NAN restores model default
- **`dBP_m`** *(optional)* — Overrides breakpoint distance; NAN or negative restores model default
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `[n_users]` or `[1]`
- **`wall_loss`** *(optional)* — Penetration loss for a single wall; TGax defines 5.0 (default) or 7.0

### Returns:
- `std::vector<quadriga_lib::channel<double>>` of length `n_users`; each entry is one user's channel realization with direction set by `uplink`

### See also:
- [get_channels_planar](#get_channels_planar) (used internally to compute MIMO coefficients per user)
- [arrayant](array-antenna-class.md#arrayant) (antenna array type for ap_array and sta_array)
- [IEEE 802.11-03/940r4 - TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc)
- [IEEE 802.11-09/0308r12 - TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc)
- [IEEE 802.11-11/0968r4 - TGah Channel Model](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx)
- [IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx)

---
<a id="get_channels_irs"></a>
## get_channels_irs
Calculate MIMO channel coefficients for IRS-assisted communication

Also in: [MATLAB](../mex/channel-generation-functions.md#get_channels_irs) · Not in: Python

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)
- Each segment is computed by [get_channels_spherical](#get_channels_spherical): NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`
- A segment without paths (all its path inputs empty) gives empty outputs `[n_rx, n_tx, 0]`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- Optional `irs_array_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- Setting `center_frequency = 0.0` disables phase computation
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

### Declaration:
```cpp
std::vector<bool> quadriga_lib::get_channels_irs(
    const quadriga_lib::arrayant<dtype> *tx_array,
    const quadriga_lib::arrayant<dtype> *rx_array,
    const quadriga_lib::arrayant<dtype> *irs_array,
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
    const quadriga_lib::arrayant<dtype> *irs_array_2 = nullptr,
    const std::vector<bool> *active_path = nullptr);
```

### Inputs:
- **`tx_array`** — Transmit antenna array with `n_tx` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`rx_array`** — Receive antenna array with `n_rx` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`irs_array`** — IRS antenna array (TX-facing side) with `n_irs` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`Tx, Ty, Tz`** — Transmitter position in Cartesian coordinates
- **`Tb, Tt, Th`** — Transmitter orientation as Euler angles (bank, tilt, heading)
- **`Rx, Ry, Rz`** — Receiver position in Cartesian coordinates
- **`Rb, Rt, Rh`** — Receiver orientation as Euler angles (bank, tilt, heading)
- **`Ix, Iy, Iz`** — IRS position in Cartesian coordinates
- **`Ib, It, Ih`** — IRS orientation as Euler angles (bank, tilt, heading)
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `[n_path_1]`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `[n_path_2]`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `[8, n_path_2]`
- **`i_irs`** *(optional)* — IRS codebook port index
- **`threshold_dB`** *(optional)* — Gain threshold in dB; path combinations below this are discarded
- **`center_frequency`** *(optional)* — Center frequency; set to `0` to skip phase computation
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component
- **`irs_array_2`** *(optional)* — Second IRS antenna array for the RX-facing side; enables asymmetric IRS patterns; see [arrayant](array-antenna-class.md#arrayant)
- **`active_path`** *(optional)* — Bitmask selecting active path pairs; overrides `threshold_dB`; `[n_path_1 * n_path_2]`

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_irs]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path_irs]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path_irs]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path_irs]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path_irs]`

### Returns:
- Boolean mask of length `n_path_1 * n_path_2` indicating which path combinations were included in the output

### See also:
- [combine_irs_coord](site-specific-simulation-tools.md#combine_irs_coord) (coordinate setup for IRS geometry)
- [get_channels_spherical](#get_channels_spherical) (single-segment spherical-wave channel)
- [get_channels_planar](#get_channels_planar) (single-segment planar-wave channel)
- [arrayant](array-antenna-class.md#arrayant) (antenna array class)

---
<a id="get_channels_multifreq"></a>
## get_channels_multifreq
Compute channel coefficients for spherical waves across multiple frequencies

Also in: [MATLAB](../mex/channel-generation-functions.md#get_channels_multifreq) · [Python](../python/channel-generation-functions.md#get_channels_multifreq)

- Multi-frequency extension of [get_channels_spherical](#get_channels_spherical) with frequency-dependent antenna patterns, path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (from `tx_array[i].center_frequency`), RX array, input samples (`freq_in`), and output (`freq_out`)
- TX/RX patterns interpolated per output frequency via SLERP with linear fallback (same as [arrayant_interpolate_multi](array-antenna-functions.md#arrayant_interpolate_multi))
- `path_gain` interpolated linearly; `M` interpolated via SLERP per complex entry pair to preserve phase
- Extrapolation clamps to nearest frequency entry on all four grids
- `propagation_speed` supports EM (speed of light, default) and acoustic (~343 m/s) simulations
- `M` accepts 8 rows (full polarimetric: the 2x2 Jones matrix, rows = RX polarisation V/H, columns = TX polarisation
  V/H, column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)) or 2 rows (scalar pressure: Re, Im of the (V rx, V tx) entry only)
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS) is added to the
  FBS-LBS segment. Delays and phases follow this length; the per-element offsets follow the FBS / LBS geometry
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm
- Coupling matrices interpolated across frequencies per complex entry (SLERP for complex pairs), identical to antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`: with `add_fake_los_path`, slot 0 holds the LOS path
  (zero power if no LOS path was found); a LOS path found in the input is moved to slot 0 and leaves a zero-power path
  in its original slot (input index + 1); all other paths keep their order, shifted by one slot
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]` per output frequency; with
  `add_fake_los_path = true` they hold only this zero-power LOS path, with the LOS delays per antenna pair
- If `use_absolute_delays` is false, the LOS delay between the TX and RX phase centres is subtracted from all paths;
  per-element delays can then be negative, down to about -(half the TX array size + half the RX array size) /
  `propagation_speed` (both arrays endfire). [quantize_delays](channel-functions.md#quantize_delays) rejects delays at or below `-tap_spacing / 2`. For
  acoustics (`propagation_speed` about 343 m/s) that happens for practically every array: at 48 kHz taps, half sizes
  adding up to more than about 3.6 mm (a 5 cm microphone pair already exceeds it); for radio at 5 ns taps, about
  0.75 m. Use `use_absolute_delays = true` in that case
- The coefficients always carry the phase of the absolute path length, also when `use_absolute_delays` is false.
  [baseband_freq_response_multi](channel-functions.md#baseband_freq_response_multi) with `remove_delay_phase = true` removes the phase of `delay` before it
  interpolates across `freq_in`, so it needs channels generated with `use_absolute_delays = true`; with relative
  delays, the phase of the TX-RX distance stays in the coefficients, is interpolated across the `freq_in` grid and
  gives wrong phases

### Declaration:
```cpp
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
    dtype propagation_speed = dtype(299792458.0));
```

### Inputs:
- **`tx_array`** — Multi-frequency TX arrayant vector; all entries must pass [arrayant_is_valid_multi](array-antenna-functions.md#arrayant_is_valid_multi)
- **`rx_array`** — Multi-frequency RX arrayant vector; all entries must pass [arrayant_is_valid_multi](array-antenna-functions.md#arrayant_is_valid_multi)
- **`Tx, Ty, Tz`** — TX position in Cartesian coordinates
- **`Tb, Tt, Th`** — TX orientation, Euler angles (bank, tilt, heading)
- **`Rx, Ry, Rz`** — RX position in Cartesian coordinates
- **`Rb, Rt, Rh`** — RX orientation, Euler angles  (bank, tilt, heading)
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Linear-scale path gains; `[n_path, n_freq_in]`
- **`path_length`** — Absolute TX-to-RX path lengths (between the phase centres); values shorter than the
  TX-FBS-LBS-RX polyline (e.g. 0) use the polyline length; `[n_path]`
- **`M`** — Polarization transfer matrix, row order as above; `[8, n_path, n_freq_in]` (full pol) or
  `[2, n_path, n_freq_in]` (scalar)
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `[n_freq_in]`
- **`freq_out`** — Target output frequencies; `[n_freq_out]`
- **`use_absolute_delays`** *(optional)* — Include LOS delay in all paths if true
- **`add_fake_los_path`** *(optional)* — Always add one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot
- **`propagation_speed`** *(optional)* — Wave speed [m/s]; use ~343.0 for acoustics

### Outputs:
- **`coeff_re`** — Real part of coefficients; vector length `n_freq_out`, each cube `[n_rx_ports, n_tx_ports, n_path_out]`
- **`coeff_im`** — Imaginary part; same structure as `coeff_re`
- **`delay`** — Propagation delays [s]; same structure as `coeff_re`

### See also:
- [get_channels_spherical](#get_channels_spherical) (single-frequency equivalent)
- [arrayant_interpolate_multi](array-antenna-functions.md#arrayant_interpolate_multi) (underlying pattern interpolation)
- [arrayant_concat_multi](array-antenna-functions.md#arrayant_concat_multi) (building multi-frequency arrays)
- [generate_speaker](array-antenna-functions.md#generate_speaker) (acoustic source construction)

---
<a id="get_channels_planar"></a>
## get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

Also in: [MATLAB](../mex/channel-generation-functions.md#get_channels_planar) · [Python](../python/channel-generation-functions.md#get_channels_planar)

- Computes complex channel coefficients and delays for all TX/RX element pairs across `n_path` propagation paths.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based (angles ignored).
- Polarization transfer matrix `M` must be normalized; rows are interleaved real/imag components.
- If `add_fake_los_path` is true, one output path is always added (output size `n_path+1`): slot 0 holds the LOS path,
  with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the TX-RX
  distance; it is moved to slot 0 and its original slot (input index + 1) holds a zero-power path. All other paths
  keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- Polarization coupling: `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation V/H),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- Setting `center_frequency = 0` disables phase calculation (delays still computed).
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance (between the phase centres) from all path
  lengths before converting to delay; per-element delays of pairs that are closer to each other than the phase
  centres are then negative, down to about -(half the TX array size + half the RX array size) / speed of light; [quantize_delays](channel-functions.md#quantize_delays) rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.

### Declaration:
```cpp
void quadriga_lib::get_channels_planar(
    const quadriga_lib::arrayant<dtype> *tx_array,
    const quadriga_lib::arrayant<dtype> *rx_array,
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

### Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` elements
- **`rx_array`** — Receive antenna array; `n_rx` elements
- **`Tx, Ty, Tz`** — Transmitter position
- **`Tb, Tt, Th`** — Transmitter orientation: bank, tilt, heading
- **`Rx, Ry, Rz`** — Receiver position
- **`Rb, Rt, Rh`** — Receiver orientation: bank, tilt, heading
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path]`
- **`path_length`** — Path lengths from TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`
- **`center_frequency`** *(optional)* — Center frequency; 0 disables phase calculation
- **`use_absolute_delays`** *(optional)* — Include LOS delay offset in all paths
- **`add_fake_los_path`** *(optional)* — Always add one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** *(optional)* — Doppler weights for moving RX; positive = moving toward path, negative = away; `[n_path(+1)]`

### See also:
- [get_channels_spherical](#get_channels_spherical) (spherical wave variant accounting for per-element angle differences)
- [get_channels_ieee_indoor](#get_channels_ieee_indoor) (for generating IEEE compliant channels using `get_channels_planar` internally)
- [baseband_freq_response](channel-functions.md#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](channel-functions.md#quantize_delays) (for mapping delays to a fixed grid)
- [arrayant](array-antenna-class.md#arrayant) (antenna array class)

---
<a id="get_channels_spherical"></a>
## get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

Also in: [MATLAB](../mex/channel-generation-functions.md#get_channels_spherical) · [Python](../python/channel-generation-functions.md#get_channels_spherical)

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths, using spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `M`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [coord2path](site-specific-simulation-tools.md#coord2path) or [qrt_file_read](channel-functions.md#qrt_file_read) with more than two interactions) is added to the FBS-LBS segment. Delays and
  phases follow this length; the per-element offsets follow the FBS / LBS geometry (spherical waves).
- If `center_frequency == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [quantize_delays](channel-functions.md#quantize_delays) maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == true`, one output path is always added (`n_path + 1` outputs): slot 0 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 0 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair.

### Declaration:
```cpp
void quadriga_lib::get_channels_spherical(
    const quadriga_lib::arrayant<dtype> *tx_array,
    const quadriga_lib::arrayant<dtype> *rx_array,
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
    arma::Cube<dtype> *eoa = nullptr,
    bool use_avx2 = false);
```

### Inputs:
- **`tx_array`** — Transmit antenna array with `n_tx` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`rx_array`** — Receive antenna array with `n_rx` elements; see [arrayant](array-antenna-class.md#arrayant)
- **`Tx, Ty, Tz`** — Transmitter position in Cartesian coordinates
- **`Tb, Tt, Th`** — Transmitter orientation as Euler angles (bank, tilt, heading)
- **`Rx, Ry, Rz`** — Receiver position in Cartesian coordinates
- **`Rb, Rt, Rh`** — Receiver orientation as Euler angles (bank, tilt, heading)
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path]`
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `[n_path]`
- **`M`** — Polarization transfer matrix, 2x2 Jones matrix column-major, Re/Im interleaved: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx); `[8, n_path]`
- **`center_frequency`** *(optional)* — Center frequency; set to `0` to skip phase computation
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component
- **`add_fake_los_path`** *(optional)* — If `true`, always adds one output path; slot 0 holds the LOS path (zero power
  if none was found); a LOS path found in the input leaves a zero-power path in its original slot
- **`use_avx2`** *(optional)* — If `true`, use AVX2 for antenna interpolation; faster, but less accurate; ignored when not supported

### Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path(+1)]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path(+1)]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path(+1)]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path(+1)]`

### See also:
- [get_channels_planar](#get_channels_planar) (planar wave variant)
- [get_channels_multifreq](#get_channels_multifreq) (multi-freq counterpart)
- [get_channels_irs](#get_channels_irs) (for IRS-assisted communication)
- [arrayant](array-antenna-class.md#arrayant) (antenna array class)
- [baseband_freq_response](channel-functions.md#baseband_freq_response) (for calculating the frequency response)
- [quantize_delays](channel-functions.md#quantize_delays) (for mapping delays to a fixed grid)
