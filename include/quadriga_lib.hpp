// SPDX-License-Identifier: Apache-2.0
//
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

#ifndef quadriga_lib_H
#define quadriga_lib_H

#include <armadillo>
#include <string>
#include <vector>
#include <optional>

#include "quadriga_arrayant.hpp"
#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

#define QUADRIGA_LIB_VERSION v0_10_0

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "arma::uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(size_t), "arma::uword and size_t have different sizes");
static_assert(sizeof(unsigned long long) == sizeof(size_t), "unsigned long and size_t have different sizes");

namespace quadriga_lib
{
    // Returns the version number as a string in format (x.y.z)
    std::string quadriga_lib_version();

    // Returns the armadillo version used by quadriga-lib in format (x.y.z)
    std::string quadriga_lib_armadillo_version();

    // Check if AVX2 is supported
    bool quadriga_lib_has_AVX2();

    // Channel generation function for IEEE TGn, TGac, TGax and TGah indoor channel models
    // - Depends on arrayant and channel classes
    // - 2D model, no elevation angles
    std::vector<channel<double>>                                // Output: Vector of channel objects, length n_users
    get_channels_ieee_indoor(const arrayant<double> &ap_array,  // Access point array antenna with 'n_tx' elements (= ports after element coupling)
                             const arrayant<double> &sta_array, // Mobile station array antenna with 'n_rx' elements (= ports after element coupling)
                             std::string ChannelType,           // Channel Model Type (A, B, C, D, E, F) as defined by TGn
                             double CarrierFreq_Hz = 5.25e9,    // Carrier frequency in Hz
                             double tap_spacing_s = 10.0e-9,    // Taps spacing in seconds, must be equal to 10 ns divided by a power of 2, TGn = 10e-9
                             arma::uword n_users = 1,           // Number of user (only for TGac, TGah)
                             double observation_time = 0.0,     // Channel observation time in seconds (0.0 = static channel)
                             double update_rate = 1.0e-3,       // Channel update interval in seconds
                             double speed_station_kmh = 0.0,    // Movement speed of the station in km/h (optional feature, default = 0), movement direction = AoA_offset
                             double speed_env_kmh = 1.2,        // Movement speed of the environment in km/h (default = 1.2 for TGn) use 0.089 for TGac
                             arma::vec Dist_m = {4.99},         // Distance between TX and TX in meters, length n_users or length 1 (if same for all users)
                             arma::uvec n_floors = {0},         // Number of floors for the TGah model, adjusted for each user, up to 4 floors, length n_users or length 1 (if same for all users)
                             bool uplink = false,               // Default channel direction is downlink, set uplink to true to get reverse direction
                             arma::mat offset_angles = {},      // Offset angles in degree for MU-MIMO channels, empty (TGac auto for n_users > 1), Size: [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS
                             arma::uword n_subpath = 20,        // Number of sub-paths per path and cluster for Laplacian AS mapping
                             double Doppler_effect = 50.0,      // Special Doppler effects in models D, E (fluorescent lights, value = mains freq.) and F (moving vehicle speed in kmh), use 0.0 to disable
                             arma::sword seed = -1);            // Numeric seed, optional, value -1 disabled seed and uses system random device

}

#endif
