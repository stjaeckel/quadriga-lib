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

std::vector<quadriga_lib::channel<double>>
quadriga_lib::get_channels_ieee_indoor(const quadriga_lib::arrayant<double> &tx_array,
                                       const quadriga_lib::arrayant<double> &rx_array,
                                       std::string ChannelType,
                                       double CarrierFreq_Hz,
                                       double tap_spacing_s,
                                       arma::uword n_users,
                                       double observation_time,
                                       double update_rate,
                                       double speed_station_kmh,
                                       arma::vec Dist_m,
                                       arma::uvec n_floors,
                                       bool uplink,
                                       arma::mat offset_angles,
                                       arma::uword n_subpath,
                                       std::optional<arma::uword> seed)
{
    // Check if the antennas are valid
    auto error_message = tx_array.is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Transmit antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }
    error_message = rx_array.is_valid();
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

    // ---------- HELPER FUNCTIONS ----------

    const double V_2PI = 6.283185307179586; // 2 * PI
    const double V_C = 299792458.0;         // Speed of light

    auto linear_2_dB = [](double x)
    { return 10.0 * std::log10(x); };

    // Generate IEEE model parameters
    // This validates all other inputs and throws an error if invalid input was given
    arma::mat rx_pos, rx_orientation;     // Size: [3, n_users]
    std::vector<arma::mat> aod, aoa, pow; // Size: [n_subpath, n_path_out], per user
    std::vector<arma::vec> delay;         // Length: n_path_out
    std::vector<arma::cube> M;            // Size: Size: [8, n_subpath, n_path_out]

    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M,
                         ChannelType, CarrierFreq_Hz, tap_spacing_s, n_users, Dist_m, n_floors,
                         offset_angles, n_subpath, true, seed);

    arma::uword n_tx = tx_array.n_ports();
    arma::uword n_rx = rx_array.n_ports();

    // Initialize output
    std::vector<quadriga_lib::channel<double>> result;
    result.reserve(n_users);

    // Default transmitter location and orientation
    const double Tx = 0.0, Ty = 0.0, Tz = 0.0;
    const double Tb = 0.0, Tt = 0.0, Th = 0.0;

    // Get a cross-polarized probe antenna to compute the antenna-neutral power after complex-sum of the sub-paths
    auto probe_antenna = quadriga_lib::generate_arrayant_xpol(15.0);
    arma::uword n_probe = probe_antenna.n_elements();

    // Motion vector for Doppler calculation
    arma::vec dist_station = arma::linspace(0.0, double(n_snap - 1) * speed_station_kmh / 3.6 * update_rate, n_snap);
    auto wave_no = std::complex<double>(0.0, V_2PI * V_C / CarrierFreq_Hz);

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

        for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
        {
            double dist_station_snap = dist_station[i_snap]; // 0 for first snapshot

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

                        std::complex<double> phasor = std::exp(wave_no * p_Doppler_Jakes[j_sub] * dist_station_snap);
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
        }

        arma::vec amplitude_scale = arma::sqrt(pow_reference / arma::sum(pow_dynamic, 1));
        double *p_amplitude_scale = amplitude_scale.memptr();

        // Compute coefficients for the first snapshot for the actual antennas
        quadriga_lib::get_channels_planar(&tx_array, &rx_array, Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, 0.0, Rb, Rt, Rh,
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
        chan_user.tx_pos = arma::vec({Tx, Ty, Tz});
        chan_user.tx_orientation = arma::vec({Tb, Tt, Th});
        chan_user.rx_orientation = arma::vec({Rb, Rt, Rh});

        // Calculate virtual RX positions based on the direction and distance vector
        double cx = std::cos(Rh), cy = std::sin(Rh);
        chan_user.rx_pos = arma::mat(3, n_snap);
        for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
        {
            double d = dist_station[i_snap];
            double *p = chan_user.rx_pos.colptr(i_snap);

            p[0] = Rx + cx * d;
            p[1] = Ry + cy * d;
            p[2] = Rz;
        }

        // Extract delays (all sub-paths and all snapshots share same delay)
        if (n_subpath > 1)
        {
            for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                std::memcpy(coeff_delay.slice_memptr(i_path), coeff_delay.slice_memptr(i_path * n_subpath), n_rx * n_tx * sizeof(double));
            coeff_delay.resize(n_rx, n_tx, n_path_out);
        }

        // Process actual coefficients
        p_coeff_cplx = coeff_cplx.memptr();
        for (arma::uword i_snap = 0; i_snap < n_snap; ++i_snap)
        {
            double dist_station_snap = dist_station[i_snap];
            arma::cube coeff_snap_re(n_rx, n_tx, n_path_out, arma::fill::zeros);
            arma::cube coeff_snap_im(n_rx, n_tx, n_path_out, arma::fill::zeros);
            double *p_cf_re = coeff_snap_re.memptr();
            double *p_cf_im = coeff_snap_im.memptr();

            for (arma::uword i_link = 0; i_link < n_rx * n_tx; ++i_link)
                for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                {
                    // Complex sum over all sub-paths with applied Doppler drift
                    std::complex<double> cf_sub_sum(0.0, 0.0);
                    for (arma::uword i_sub = 0; i_sub < n_subpath; ++i_sub)
                    {
                        arma::uword j_sub = i_path * n_subpath + i_sub;     // Linear subpath index
                        arma::uword j_coeff = j_sub * n_rx * n_tx + i_link; // Linear coefficient index

                        std::complex<double> phasor = std::exp(wave_no * p_Doppler_Jakes[j_sub] * dist_station_snap);
                        std::complex<double> cf = p_coeff_cplx[j_coeff] * phasor;
                        cf_sub_sum += cf;
                    }

                    // Apply amplitude correction to all coefficients
                    cf_sub_sum *= p_amplitude_scale[i_path];

                    // Split into real and imaginary parts
                    arma::uword i_coeff = i_path * n_rx * n_tx + i_link;
                    p_cf_re[i_coeff] = cf_sub_sum.real();
                    p_cf_im[i_coeff] = cf_sub_sum.imag();
                }

            // Add coefficients to channel object
            chan_user.coeff_re.push_back(std::move(coeff_snap_re));
            chan_user.coeff_im.push_back(std::move(coeff_snap_im));
            chan_user.delay.push_back(coeff_delay);

            // Store path loss
            double *p_pow_dynamic = pow_dynamic.colptr(i_snap);
            for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
                p_pow_dynamic[i_path] *= p_amplitude_scale[i_path] * p_amplitude_scale[i_path];
            chan_user.path_gain.push_back(pow_dynamic.col(i_snap));
        }

        // Add user channel to result list
        result.push_back(chan_user);
    }

    return result;
}
