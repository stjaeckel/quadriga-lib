// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "quadriga_lib.hpp"
#include "ieee_channel_model_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

## Description:
- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions.
- 2D model: no elevation angles are used; azimuth angles and planar motion are considered.
- Supports channel model types `A, B, C, D, E, F` (as defined by TGn) via `ChannelType`.
- Can generate MU-MIMO channels (`n_users > 1`) with per-user distances/floors and optional angle offsets according to TGac
- Optional time evolution via `observation_time`, `update_rate`, and mobility parameters.

## Declaration:
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

## Arguments:
- `const arrayant<double> **ap_array**` (input)<br>
  Access point array antenna with `n_tx` elements (= ports after element coupling).

- `const arrayant<double> **sta_array**` (input)<br>
  Mobile station array antenna with `n_rx` elements (= ports after element coupling).

- `std::string **ChannelType**` (input)<br>
  Channel model type as defined by TGn. Supported: `A, B, C, D, E, F`.

- `double **CarrierFreq_Hz** = 5.25e9` (optional input)<br>
  Carrier frequency in Hz.

- `double **tap_spacing_s** = 10.0e-9` (optional input)<br>
  Tap spacing in seconds. Must be equal to `10 ns / 2^k` (TGn default = `10e-9`).

- `arma::uword **n_users** = 1` (optional input)<br>
  Number of users (only for TGac, TGah). Output vector length equals `n_users`.

- `double **observation_time** = 0.0` (optional input)<br>
  Channel observation time in seconds. `0.0` creates a static channel.

- `double **update_rate** = 1.0e-3` (optional input)<br>
  Channel update interval in seconds (only relevant when `observation_time > 0`).

- `double **speed_station_kmh** = 0.0` (optional input)<br>
  Station movement speed in km/h. Movement direction is `AoA_offset`. Only relevant when `observation_time > 0`.

- `double **speed_env_kmh** = 1.2` (optional input)<br>
  Environment movement speed in km/h. Default `1.2` for TGn, use `0.089` for TGac. Only relevant when
  `observation_time > 0`.

- `arma::vec **Dist_m** = {4.99}` (optional input)<br>
  TX-to-RX distance(s) in meters. Length `n_users` or length `1` (same distance for all users). Size
  `[n_users]` or `[1]`.

- `arma::uvec **n_floors** = {0}` (optional input)<br>
  Number of floors for TGah model (per user), up to 4 floors. Length `n_users` or length `1`. Size
  `[n_users]` or `[1]`.

- `bool **uplink** = false` (optional input)<br>
  Channel direction flag. Default is downlink; set to `true` to generate reverse (uplink) direction.

- `arma::mat **offset_angles** = {}` (optional input)<br>
  Offset angles in degree for MU-MIMO channels. Empty uses model defaults (TGac auto for `n_users > 1`).
  Size `[4, n_users]` with rows: `AoD LOS, AoD NLOS, AoA LOS, AoA NLOS`.

- `arma::uword **n_subpath** = 20` (optional input)<br>
  Number of sub-paths per path/cluster used for Laplacian angular spread mapping.

- `double **Doppler_effect** = 50.0` (optional input)<br>
  Special Doppler effects: models `D, E` (fluorescent lights, value = mains freq.) and `F` (moving
  vehicle speed in km/h). Use `0.0` to disable.

- `arma::sword **seed** = -1` (optional input)<br>
  Numeric seed for repeatability. `-1` disables the fixed seed and uses the system random device.

- `double **KF_linear** = NAN` (optional input)<br>
  Overwrites the model-specific KF-value. If this parameter is NAN (default) or negative, model defaults are used:
  A/B/C (KF = 1 for d < dBP, 0 otherwise); D (KF = 2 for d < dBP, 0 otherwise); E/F (KF = 4 for d < dBP, 0 otherwise). 
  KF is applied to the first tap only. Breakpoint distance is ignored for `KF_linear >= 0`.

- `double **XPR_NLOS_linear** = NAN` (optional input)<br>
  Overwrites the model-specific Cross-polarization ratio. If this parameter is NAN (default) or negative, 
  the model default of 2 (3 dB) is used. XPR is applied to all NLOS taps.

- `double **SF_std_dB_LOS** = NAN` (optional input)<br>
  Overwrites the model-specific shadow fading for LOS channels. If this parameter is NAN (default), 
  the model default of 3 dB is used. `SF_std_dB_LOS` is applied to all LOS channels, where the 
  AP-STA distance d < dBP.

- `double **SF_std_dB_NLOS** = NAN` (optional input)<br>
  Overwrites the model-specific shadow fading for LOS channels. If this parameter is NAN (default), 
  the model defaults are A/B: 4 dB, C/D: 5 dB, E/F: 6 dB. `SF_std_dB_NLOS` is applied to all NLOS channels, 
  where the AP-STA distance d >= dBP.

- `double **dBP_m** = NAN` (optional input)<br>
  Overwrites the model-specific breakpoint distance. If this parameter is NAN (default) or negative, 
  the model defaults are A/B/C: 5 m, D: 10 m, E: 20 m, F: 30 m.
 
## Returns:
- `std::vector<quadriga_lib::channel<double>>` (output)<br>
  Vector of channel objects with length `n_users`. Each entry contains the generated indoor channel
  realization for one user (including direction determined by `uplink`).
MD!*/

std::vector<quadriga_lib::channel<double>>
quadriga_lib::get_channels_ieee_indoor(const quadriga_lib::arrayant<double> &ap_array,  // Access point array antenna with 'n_tx' elements (= ports after element coupling)
                                       const quadriga_lib::arrayant<double> &sta_array, // Mobile station array antenna with 'n_rx' elements (= ports after element coupling)
                                       std::string ChannelType,                         // Channel Model Type (A, B, C, D, E, F) as defined by TGn
                                       double CarrierFreq_Hz,                           // Carrier frequency in Hz
                                       double tap_spacing_s,                            // Taps spacing in seconds, must be equal to 10 ns divided by a power of 2, TGn = 10e-9
                                       arma::uword n_users,                             // Number of user (only for TGac, TGah)
                                       double observation_time,                         // Channel observation time in seconds (0.0 = static channel)
                                       double update_rate,                              // Channel update interval in seconds
                                       double speed_station_kmh,                        // Movement speed of the station in km/h (optional feature, default = 0), movement direction = AoA_offset
                                       double speed_env_kmh,                            // Movement speed of the environment in km/h (default = 1.2 for TGn) use 0.089 for TGac
                                       arma::vec Dist_m,                                // Distance between TX and TX in meters, length n_users or length 1 (if same for all users)
                                       arma::uvec n_floors,                             // Number of floors for the TGah model, adjusted for each user, up to 4 floors, length n_users or length 1 (if same for all users)
                                       bool uplink,                                     // Default channel direction is downlink, set uplink to true to get reverse direction
                                       arma::mat offset_angles,                         // Offset angles in degree for MU-MIMO channels, empty (TGac auto for n_users > 1), Size: [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS
                                       arma::uword n_subpath,                           // Number of sub-paths per path and cluster for Laplacian AS mapping
                                       double Doppler_effect,                           // Special Doppler effects in models D, E (fluorescent lights, value = mains freq.) and F (moving vehicle speed in kmh), use 0.0 to disable
                                       arma::sword seed,                                // Numeric seed, optional, value -1 disabled seed and uses system random device
                                       double KF_linear,                                // Overwrites the default KF (linear scale)
                                       double XPR_NLOS_linear,                          // Overwrites the default Cross-polarization ratio (linear scale) for NLOS paths
                                       double SF_std_dB_LOS,                            // Overwrites the default Shadow Fading STD for LOS channels in dB
                                       double SF_std_dB_NLOS,                           // Overwrites the default Shadow Fading STD for NLOS channels in dB
                                       double dBP_m)                                    // Overwrites the default breakpoint distance in meters

{
    // Check if the antennas are valid
    auto error_message = ap_array.is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Transmit antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }
    error_message = sta_array.is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Receive antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    if (observation_time < 0.0)
        throw std::invalid_argument("Observation time cannot be negative.");

    if (update_rate <= 0.0)
        throw std::invalid_argument("Update rate cannot be 0 or negative.");

    // Get the number of snapshots
    arma::uword n_snap = 1 + (arma::uword)std::floor(observation_time / update_rate);
    long long n_snap_ll = (long long)n_snap;

    const double V_PI = 3.141592653589793;  // PI
    const double V_2PI = 6.283185307179586; // 2 * PI
    const double V_C = 299792458.0;         // Speed of light

    // Generate IEEE model parameters
    // This validates all other inputs and throws an error if invalid input was given
    arma::mat rx_pos, rx_orientation;     // Size: [3, n_users]
    std::vector<arma::mat> aod, aoa, pow; // Size: [n_subpath, n_path_out], per user
    std::vector<arma::vec> delay;         // Length: n_path_out
    std::vector<arma::cube> M;            // Size: Size: [8, n_subpath, n_path_out]

    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M,
                         ChannelType, CarrierFreq_Hz, tap_spacing_s, n_users, Dist_m, n_floors,
                         offset_angles, n_subpath, seed, 
                         KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m);

    arma::uword n_tx = ap_array.n_ports();
    arma::uword n_rx = sta_array.n_ports();

    arma::uword bw_factor = (arma::uword)std::round(10.0e-9 / tap_spacing_s); // bandwidth expansion factor

    // ---------- HELPER FUNCTIONS ----------

    auto find_index = [](size_t n_data, const double *data, double val) -> size_t
    {
        for (size_t i_data = 0; i_data < n_data; ++i_data)
            if (data[i_data] >= val)
                return i_data;
        return n_data;
    };

    // Initialize RNG
    static thread_local std::mt19937_64 rng;
    if (seed != -1)
        rng.seed(seed + 1); // Use different seed as for "qd_ieee_indoor_param"
    else
    {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        rng.seed(seq);
    }

    // Fisher–Yates shuffle using std::mt19937_64 RNG
    auto shuffle_uvec = [&](const arma::uvec &in) -> arma::uvec
    {
        arma::uvec out = in; // copy
        for (arma::uword i = out.n_elem; i > 1; --i)
        {
            std::uniform_int_distribution<arma::uword> dist(0, i - 1);
            arma::uword j = dist(rng);
            std::swap(out[i - 1], out[j]);
        }
        return out;
    };

    std::uniform_real_distribution<double> U(0.0, 1.0);

    // ---------- CHANNEL GENERATION ----------

    // Initialize output
    std::vector<quadriga_lib::channel<double>> result;
    result.reserve(n_users);

    // Default transmitter location and orientation
    const double Tx = 0.0, Ty = 0.0, Tz = 0.0;
    const double Tb = 0.0, Tt = 0.0, Th = 0.0;

    // Get a cross-polarized probe antenna to compute the antenna-neutral power after complex-sum of the sub-paths
    auto probe_antenna = quadriga_lib::generate_arrayant_xpol(15.0);
    arma::uword n_probe = probe_antenna.n_elements();

    // Constants for Doppler calculation
    const double v_station = speed_station_kmh / 3.6;      // Speed of the station
    const double v_env = speed_env_kmh / 3.6;              // Speed of the environment
    const double v_spike = Doppler_effect / 3.6;           // Moving vehicle reflection in model F
    const double lambda = V_C / CarrierFreq_Hz;            // Wavelength [m]
    const double f_d_env = v_env / lambda;                 // Max env Doppler (≈3 Hz at 2.4 GHz)
    const double f_spike = v_spike / lambda;               // Spike center frequency (model F)
    const double gamma_bell = f_d_env / 3.0;               // scale parameter (A=9 → 3*gamma = f_d)
    const double gamma_spike = f_spike / std::sqrt(90000); // Narrow spike bandwidth

    // Switches for Doppler effects
    bool use_fluor = Doppler_effect > 0.0 && (ChannelType == "D" || ChannelType == "E") && (n_snap > 0);
    bool use_vehicle = Doppler_effect > 0.0 && (ChannelType == "F") && (n_snap > 0);

    // Doppler components due to fluorescent lights
    arma::vec g(n_snap);
    if (use_fluor)
    {
        const double f_m = Doppler_effect; // Mains frequency, e.g. 50 Hz
        const double f0 = 2.0 * f_m;       // 100 Hz
        const double f1 = 6.0 * f_m;       // 300 Hz
        const double f2 = 10.0 * f_m;      // 500 Hz

        // Target harmonic amplitudes (from TGn, in dB, convert to linear)
        const double A0_lin = 1.0;                          // 0 dB
        const double A1_lin = std::pow(10.0, -15.0 / 20.0); // -15 dB
        const double A2_lin = std::pow(10.0, -20.0 / 20.0); // -20 dB

        // Number of "lamps" to approximate realistic jitter
        const int N_lamp = 10;
        // Small frequency jitter (Hz) around each harmonic
        const double sigma_f = 0.3; // tweak 0.2–0.5 Hz as you like

        std::normal_distribution<double> Nf(0.0, sigma_f);
        std::uniform_real_distribution<double> Uphase(0.0, 2.0 * V_PI);

        double dt = update_rate;

        for (int n = 0; n < N_lamp; ++n)
        {
            // jittered freqs for this lamp
            double f0n = f0 + Nf(rng);
            double f1n = f1 + Nf(rng);
            double f2n = f2 + Nf(rng);

            double phi0 = Uphase(rng);
            double phi1 = Uphase(rng);
            double phi2 = Uphase(rng);

            // scale per lamp so total amplitude across lamps ~ A*_lin
            double s0 = A0_lin / std::sqrt(N_lamp);
            double s1 = A1_lin / std::sqrt(N_lamp);
            double s2 = A2_lin / std::sqrt(N_lamp);

            for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
            {
                double t = dt * double(i_snap);

                g[i_snap] += s0 * std::cos(V_2PI * f0n * t + phi0) +
                             s1 * std::cos(V_2PI * f1n * t + phi1) +
                             s2 * std::cos(V_2PI * f2n * t + phi2);
            }
        }
    }

    // Compute MIMO channel coefficients for each user
    for (arma::uword i_user = 0; i_user < n_users; ++i_user)
    {
        // Get number of paths for the current user, may vary depending on LOS state
        arma::uword n_path_out = aod[i_user].n_cols;

        // All elevation angles are set to 0 (2D Model)
        arma::vec eox = arma::vec(n_subpath * n_path_out, arma::fill::zeros);

        // Receiver position and orientation
        double Rx = rx_pos(0, i_user), Ry = rx_pos(1, i_user), Rz = rx_pos(2, i_user);
        double Rb = 0.0, Rt = 0.0, Rh = rx_orientation(2, i_user);

        // Get angles, path powers and polarization transfer matrix for the current user
        const arma::vec aod_user = arma::vec(aod[i_user].memptr(), n_subpath * n_path_out, false, true);
        const arma::vec aoa_user = arma::vec(aoa[i_user].memptr(), n_subpath * n_path_out, false, true);
        const arma::vec path_gain_user = arma::vec(pow[i_user].memptr(), n_subpath * n_path_out, false, true);
        const arma::mat M_user = arma::mat(M[i_user].memptr(), 8, n_subpath * n_path_out, false, true);

        // Determine the the path indices that should be modified with Doppler effects based on IEEE 802.11-03/940r4
        // For TGac interpolation, assume that all interpolated taps are effected as well (maintains total interference power)
        arma::uvec mod_paths;
        if (use_fluor)
        {
            // Determine the the path indices that should be modulated baes on IEEE 802.11-03/940r4
            // For TGac interpolation, assume that all interpolated taps are effected as well (maintains total interference power)
            if (ChannelType == "D")
            {
                mod_paths.zeros(3 * bw_factor);
                mod_paths[0] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 139.999e-9) + 1;             // Tap 2 of cluster 2 @ 140 ns
                mod_paths[bw_factor] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 199.999e-9) + 1;     // Tap 4 of cluster 2 @ 200 ns
                mod_paths[2 * bw_factor] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 289.999e-9) + 1; // Tap 6 of cluster 2 @ 290 ns
                for (arma::uword i_path_interpol = 1; i_path_interpol < bw_factor; ++i_path_interpol)
                {
                    mod_paths[i_path_interpol] = mod_paths[i_path_interpol - 1] + 2;                                 // 2 clusters @ 140 ns
                    mod_paths[bw_factor + i_path_interpol] = mod_paths[bw_factor + i_path_interpol - 1] + 2;         // 2 clusters @ 200 ns
                    mod_paths[2 * bw_factor + i_path_interpol] = mod_paths[2 * bw_factor + i_path_interpol - 1] + 2; // 2 clusters @ 200 ns
                }
            }
            else if (ChannelType == "E")
            {
                mod_paths.zeros(3 * bw_factor);
                mod_paths[0] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 19.999e-9);              // Tap 3 of cluster 1 @ 20 ns
                mod_paths[bw_factor] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 49.999e-9);      // Tap 5 of cluster 1 @ 50 ns
                mod_paths[2 * bw_factor] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 109.999e-9); // Tap 7 of cluster 2 @ 110 ns
                for (arma::uword i_path_interpol = 1; i_path_interpol < bw_factor; ++i_path_interpol)
                {
                    mod_paths[i_path_interpol] = mod_paths[i_path_interpol - 1] + 1;                                 // 1 cluster @ 20 ns
                    mod_paths[bw_factor + i_path_interpol] = mod_paths[bw_factor + i_path_interpol - 1] + 2;         // 2 clusters @ 50 ns
                    mod_paths[2 * bw_factor + i_path_interpol] = mod_paths[2 * bw_factor + i_path_interpol - 1] + 2; // 2 clusters @ 110 ns
                }
            }
        }

        if (use_vehicle && ChannelType == "F")
        {
            mod_paths.zeros(bw_factor);
            mod_paths[0] = find_index(delay[i_user].n_elem, delay[i_user].memptr(), 19.999e-9); // Tap 3 of cluster 1 @ 20 ns
            for (arma::uword i_path_interpol = 1; i_path_interpol < bw_factor; ++i_path_interpol)
                mod_paths[i_path_interpol] = mod_paths[i_path_interpol - 1] + 1; // 1 cluster @ 20 ns
        }

        // Sanity check mod_paths to avoid downstream out-of range issues and construct boolean mask
        // This should not happen with the default parameter tables, but users might edit them and recompile the project, causing seg-faults.
        std::vector<bool> mod_paths_mask(n_path_out, false);
        if (!mod_paths.empty())
        {
            for (const auto &v : mod_paths)
                if (v < n_path_out)
                    mod_paths_mask[v] = true;
                else
                    throw std::invalid_argument("Could not map special Doppler effect to a path index.");
        }

        // Obtain the subpaths that are reflected from the vehicle in model F
        // - Split the vehicle-reflected taps into 2 groups 1) reflected from vehicle (33%) and 2) reflected from static objects (67%)
        // - Ratio between vehicle reflected paths and the rests should be 2-4 dB (~33% reflected from vehicle)
        std::vector<std::vector<bool>> vehicle_reflected; // Nested boolean mask, initially empty
        if (use_vehicle)
        {
            vehicle_reflected.resize(n_path_out); // Initialize as empty vectors

            const double dB_target = 3.0; // Ps - Pv ≈ 3 dB
            const int N_TRIES = 30;

            for (arma::uword i_mod_path = 0; i_mod_path < mod_paths.n_elem; ++i_mod_path)
            {
                arma::uword i_path = mod_paths[i_mod_path];
                arma::vec pow_subpath = pow[i_user].unsafe_col(i_path);
                double P_tot = arma::accu(pow_subpath);

                arma::uvec base_idx = arma::regspace<arma::uvec>(0, n_subpath - 1), best_idx;
                double best_err = INFINITY;

                // number of vehicle subpaths ≈ 1/3 of total (at least 1)
                arma::uword k = std::max<arma::uword>(1, (n_subpath + 1) / 3);

                for (int it = 0; it < N_TRIES; ++it)
                {
                    arma::uvec idx = shuffle_uvec(base_idx);
                    arma::uvec veh_idx = idx.head(k);

                    double Pv = arma::accu(pow_subpath(veh_idx));
                    double Ps = P_tot - Pv;
                    if (Pv <= 0.0 || Ps <= 0.0)
                        continue;

                    double dB = 10.0 * std::log10(Ps / Pv); // static - vehicle in dB
                    double err = std::abs(dB - dB_target);

                    if (err < best_err)
                    {
                        best_err = err;
                        best_idx = veh_idx;
                    }
                }

                // fallback: if all tries were degenerate, just take the strongest tap
                if (best_idx.is_empty())
                {
                    arma::uword max_i = pow_subpath.index_max();
                    best_idx = arma::uvec(1);
                    best_idx(0) = max_i;
                }

                vehicle_reflected[i_path].resize(n_subpath, false);
                for (const auto &v : best_idx)
                    if (v < n_subpath)
                        vehicle_reflected[i_path][v] = true;
                    else
                        throw std::invalid_argument("Could not map vehicle refection to subpath.");
            }
        }

        // Generate random Doppler shifts from the environment
        std::vector<double> f_env(n_subpath * n_path_out);
        for (arma::uword j_sub = 0; j_sub < f_env.size(); ++j_sub)
        {
            if (use_vehicle)
            {
                arma::uword i_path = j_sub / n_subpath; // which tap this subpath belongs to
                arma::uword i_sub = j_sub % n_subpath;  // subpath index
                bool from_vehicle = !vehicle_reflected[i_path].empty() && vehicle_reflected[i_path][i_sub];

                if (from_vehicle)
                {
                    double f = 0.0;
                    for (int iter = 0; iter < 10; ++iter) // rejection sampling for truncated Cauchy
                    {
                        double u = U(rng);
                        f = gamma_spike * std::tan(V_PI * (u - 0.5)); // Cauchy draw

                        if (std::abs(f) <= 0.1 * f_spike)
                            break; // accept sample
                        // else: redraw
                    }
                    f_env[j_sub] = f_spike + f; // Hz
                    continue;
                }
            }

            // Bell shape Doppler spectrum
            double f = 0.0;
            for (int iter = 0; iter < 10; ++iter) // rejection sampling for truncated Cauchy
            {
                double u = U(rng);
                f = gamma_bell * std::tan(V_PI * (u - 0.5)); // Cauchy draw

                if (std::abs(f) <= 5.0 * f_d_env)
                    break; // accept sample
                // else: redraw
            }
            f_env[j_sub] = f; // Hz
        }

        // Get the reference power without antennas by summing over all sub-paths
        arma::vec pow_reference = arma::sum(pow[i_user], 0).t();

        // Compute path length from the path delays and Distance
        // Rz is set to 3 * n_floors in case of TGah model, this adds the number of floors to the distance
        arma::vec path_length(n_subpath * n_path_out);
        double Dist = std::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
        double *p_length = path_length.memptr();
        for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
        {
            double len = Dist + delay[i_user].at(i_path) * V_C;
            for (arma::uword i_sub = 0; i_sub < n_subpath; ++i_sub)
                p_length[i_path * n_subpath + i_sub] = len;
        }

        // Variables
        arma::cube coeff_re, coeff_im, coeff_delay; // Size: [n_tx, n_rx, n_sub * n_path]
        arma::vec Doppler_Jakes;                    // Length: [n_sub * n_path]

        // Compute coefficients for the first snapshot for the probe antenna
        quadriga_lib::get_channels_planar(&probe_antenna, &probe_antenna, Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, 0.0, Rb, Rt, Rh,
                                          &aod_user, &eox, &aoa_user, &eox, &path_gain_user, &path_length, &M_user,
                                          &coeff_re, &coeff_im, &coeff_delay, CarrierFreq_Hz, false, false, &Doppler_Jakes);

        // Combine and restructure coefficients
        arma::cx_cube coeff_cplx; // Size: [n_tx * n_rx, n_sub, n_path]
        quadriga_lib::complex_cast(arma::cube(coeff_re.memptr(), n_probe * n_probe, n_subpath, n_path_out, false, true),
                                   arma::cube(coeff_im.memptr(), n_probe * n_probe, n_subpath, n_path_out, false, true),
                                   coeff_cplx);

        // Calculate the power after summing over the sub-paths
        arma::mat pow_dynamic(n_path_out, n_snap, arma::fill::zeros);
        std::complex<double> *p_coeff_cplx = coeff_cplx.memptr();
        double *p_Doppler_Jakes = Doppler_Jakes.memptr();

#pragma omp parallel for
        for (long long i_snap = 0; i_snap < n_snap_ll; ++i_snap)
            for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
            {
                // Power over all MIMO sublinks
                double pow_path = 0.0;
                for (arma::uword i_link = 0; i_link < n_probe * n_probe; ++i_link)
                {
                    // Complex sum over all sub-paths with applied Doppler drift
                    std::complex<double> cf_sub_sum(0.0, 0.0);
                    for (arma::uword i_sub = 0; i_sub < n_subpath; ++i_sub)
                    {
                        arma::uword j_sub = i_path * n_subpath + i_sub;           // Linear subpath index
                        arma::uword j_coeff = j_sub * n_probe * n_probe + i_link; // Linear coefficient index

                        double mu_j = p_Doppler_Jakes[j_sub];          // cos(az)*cos(el) in [-1,1]
                        double f_stat_j = (v_station / lambda) * mu_j; // Jakes
                        double f_env_j = f_env[j_sub];                 // TGn bell/Cauchy
                        double f_tot_j = f_stat_j + f_env_j;
                        double t_snap = double(i_snap) * update_rate;

                        double phase = -V_2PI * f_tot_j * t_snap;
                        std::complex<double> phasor{std::cos(phase), std::sin(phase)};

                        std::complex<double> cf = p_coeff_cplx[j_coeff] * phasor;
                        cf_sub_sum += cf;
                    }
                    // Add powers of all MIMO sublinks
                    pow_path += cf_sub_sum.real() * cf_sub_sum.real() + cf_sub_sum.imag() * cf_sub_sum.imag();
                }

                // Half the path power due to dual-polarized excitation of the channel
                pow_path *= 0.5;

                // Average over all snapshots
                pow_path /= (double)n_snap;

                // Add powers of all snapshots
                pow_dynamic(i_path, i_snap) = pow_path;
            }

        arma::vec amplitude_scale = arma::sqrt(pow_reference / arma::sum(pow_dynamic, 1));
        double *p_amplitude_scale = amplitude_scale.memptr();

        // Fluorescent modulation
        double alpha_fluor = 0.0;
        if (use_fluor)
        {
            // RNG for IEEE 802.11-03/940r4, eq. (29)
            std::normal_distribution<double> N_ic(0.0203, 0.0107);

            double X = N_ic(rng); // Draw I/C = X^2, with X ~ N(0.0203, 0.0107^2)
            double IC = X * X;    // Linear
            double C_total = 0.0; // Total carrier energy C over all snapshots & all paths
            double denom = 0.0;   // Denominator: sum |g[n]|^2 * |c_p[n]|^2 over modulated taps

            for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
            {
                double scale2 = p_amplitude_scale[i_path] * p_amplitude_scale[i_path];
                for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
                {
                    double p = pow_dynamic(i_path, i_snap) * scale2; // |c_p(t)|^2
                    C_total += p;

                    if (mod_paths_mask[i_path])
                    {
                        double g2 = g[i_snap] * g[i_snap]; // |g(t)|^2
                        denom += g2 * p;
                    }
                }
            }

            if (denom > 0.0)
                alpha_fluor = std::sqrt(IC * C_total / denom);
        }

        // Compute coefficients for the first snapshot for the actual antennas
        quadriga_lib::get_channels_planar(&ap_array, &sta_array, Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, 0.0, Rb, Rt, Rh,
                                          &aod_user, &eox, &aoa_user, &eox, &path_gain_user, &path_length, &M_user,
                                          &coeff_re, &coeff_im, &coeff_delay, CarrierFreq_Hz, false, false, &Doppler_Jakes);

        // Combine and restructure coefficients
        quadriga_lib::complex_cast(arma::cube(coeff_re.memptr(), n_rx * n_tx, n_subpath, n_path_out, false, true),
                                   arma::cube(coeff_im.memptr(), n_rx * n_tx, n_subpath, n_path_out, false, true),
                                   coeff_cplx);

        // Build channel object for the current user
        quadriga_lib::channel<double> chan_user;
        chan_user.name = "user_" + std::to_string(i_user);
        chan_user.center_frequency = {CarrierFreq_Hz};

        // Calculate virtual RX positions based on the direction and distance vector
        double cx = std::cos(Rh), cy = std::sin(Rh);
        chan_user.rx_pos = arma::mat(3, n_snap);
        for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
        {
            double d = v_station * update_rate * (double)i_snap;
            double *p = chan_user.rx_pos.colptr(i_snap);

            p[0] = Rx + cx * d;
            p[1] = Ry + cy * d;
            p[2] = Rz;
        }

        if (uplink)
        {
            chan_user.tx_pos = std::move(chan_user.rx_pos);
            chan_user.rx_pos = arma::vec({Tx, Ty, Tz});
            chan_user.rx_orientation = arma::vec({Tb, Tt, Th});
            chan_user.tx_orientation = arma::vec({Rb, Rt, Rh});
        }
        else
        {
            chan_user.tx_pos = arma::vec({Tx, Ty, Tz});
            chan_user.tx_orientation = arma::vec({Tb, Tt, Th});
            chan_user.rx_orientation = arma::vec({Rb, Rt, Rh});
        }

        chan_user.coeff_re.resize(n_snap);
        chan_user.coeff_im.resize(n_snap);
        chan_user.delay.resize(n_snap);
        chan_user.path_gain.resize(n_snap);

        // Allocate memory
        for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
        {
            if (uplink)
            {
                chan_user.coeff_re[i_snap].set_size(n_tx, n_rx, n_path_out);
                chan_user.coeff_im[i_snap].set_size(n_tx, n_rx, n_path_out);
                chan_user.delay[i_snap].set_size(n_tx, n_rx, n_path_out);
            }
            else
            {
                chan_user.coeff_re[i_snap].set_size(n_rx, n_tx, n_path_out);
                chan_user.coeff_im[i_snap].set_size(n_rx, n_tx, n_path_out);
                chan_user.delay[i_snap].set_size(n_rx, n_tx, n_path_out);
            }
        }

        // Extract delays (all sub-paths and all snapshots share same delay)
        if (n_subpath > 1)
        {
            for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                std::memcpy(coeff_delay.slice_memptr(i_path), coeff_delay.slice_memptr(i_path * n_subpath), n_rx * n_tx * sizeof(double));
            coeff_delay.resize(n_rx, n_tx, n_path_out);
        }

        // Downlink to Uplink conversion
        if (uplink)
        {
            arma::cube tmp = std::move(coeff_delay);
            coeff_delay = arma::cube(n_tx, n_rx, n_path_out, arma::fill::none);
            for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                coeff_delay.slice(i_path) = tmp.slice(i_path).t();
        }

        // Process actual coefficients
        p_coeff_cplx = coeff_cplx.memptr();

#pragma omp parallel
        {
            // Allocate coefficient storage per thread, reuse for iterations
            arma::cube coeff_snap_re, coeff_snap_im;
            if (uplink)
            {
                coeff_snap_re.set_size(n_tx, n_rx, n_path_out);
                coeff_snap_im.set_size(n_tx, n_rx, n_path_out);
            }
            else
            {
                coeff_snap_re.set_size(n_rx, n_tx, n_path_out);
                coeff_snap_im.set_size(n_rx, n_tx, n_path_out);
            }

#pragma omp for
            for (long long i_snap = 0; i_snap < n_snap_ll; ++i_snap)
            {
                // Access coefficient memory and initialize to 0
                double *p_cf_re = coeff_snap_re.memptr();
                double *p_cf_im = coeff_snap_im.memptr();
                std::fill_n(p_cf_re, coeff_snap_re.n_elem, 0.0);
                std::fill_n(p_cf_im, coeff_snap_im.n_elem, 0.0);

                for (arma::uword i_link = 0; i_link < n_rx * n_tx; ++i_link)
                    for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                    {
                        // Complex sum over all sub-paths with applied Doppler drift
                        std::complex<double> cf_sub_sum(0.0, 0.0);
                        for (arma::uword i_sub = 0; i_sub < n_subpath; ++i_sub)
                        {
                            arma::uword j_sub = i_path * n_subpath + i_sub;     // Linear subpath index
                            arma::uword j_coeff = j_sub * n_rx * n_tx + i_link; // Linear coefficient index

                            double mu_j = p_Doppler_Jakes[j_sub];          // cos(az)*cos(el) in [-1,1]
                            double f_stat_j = (v_station / lambda) * mu_j; // Jakes
                            double f_env_j = f_env[j_sub];                 // TGn bell/Cauchy
                            double f_tot_j = f_stat_j + f_env_j;
                            double t_snap = double(i_snap) * update_rate;

                            double phase = -V_2PI * f_tot_j * t_snap;
                            std::complex<double> phasor{std::cos(phase), std::sin(phase)};

                            std::complex<double> cf = p_coeff_cplx[j_coeff] * phasor;
                            cf_sub_sum += cf;
                        }

                        // Fluorescent modulation
                        if (use_fluor && mod_paths_mask[i_path])
                        {
                            double gval = g[i_snap]; // real
                            double a = alpha_fluor * gval;
                            std::complex<double> mod_factor(1.0 + a, 0.0);
                            cf_sub_sum *= mod_factor;
                        }

                        // Apply amplitude correction to all coefficients
                        cf_sub_sum *= p_amplitude_scale[i_path];

                        // Split into real and imaginary parts
                        if (uplink) // Conjugate transpose channel
                        {
                            arma::uword i_trans = (i_link / n_rx) + (i_link % n_rx) * n_tx;
                            arma::uword i_coeff = i_path * n_rx * n_tx + i_trans;
                            p_cf_re[i_coeff] = cf_sub_sum.real();
                            p_cf_im[i_coeff] = -cf_sub_sum.imag();
                        }
                        else
                        {
                            arma::uword i_coeff = i_path * n_rx * n_tx + i_link;
                            p_cf_re[i_coeff] = cf_sub_sum.real();
                            p_cf_im[i_coeff] = cf_sub_sum.imag();
                        }
                    }

                // write results into pre-sized vectors at this snapshot index
                size_t n_bytes = size_t(n_tx * n_rx * n_path_out) * sizeof(double);
                std::memcpy(chan_user.coeff_re[i_snap].memptr(), coeff_snap_re.memptr(), n_bytes);
                std::memcpy(chan_user.coeff_im[i_snap].memptr(), coeff_snap_im.memptr(), n_bytes);
                std::memcpy(chan_user.delay[i_snap].memptr(), coeff_delay.memptr(), n_bytes);

                // Store path loss
                double *p_pow_dynamic = pow_dynamic.colptr(i_snap);
                for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                    p_pow_dynamic[i_path] *= p_amplitude_scale[i_path] * p_amplitude_scale[i_path];

                chan_user.path_gain[i_snap] = pow_dynamic.col(i_snap); // copies arma::vec
            }
        }

        // Add user channel to result list
        result.push_back(chan_user);
    }

    return result;
}
