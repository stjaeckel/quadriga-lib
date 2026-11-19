// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qdlib_ieee_channel_model_functions_H
#define qdlib_ieee_channel_model_functions_H

#include <armadillo>
#include <string>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <cstddef>
#include <optional>

#include "qd_helpers_shared.hpp"

// Generate channel parameters for the IEEE indoor channel models as defined by TGn, TGac, TGah and TGax
// - Outputs non-zero paths and clusters where the max. n_path_out = n_cluster * n_path (+1) from the IEEE tables (+1 additional LOS cluster, if present)
// - Sub-path aod and aoa do not include AP or Station orientation
static void qd_ieee_indoor_param(arma::mat &rx_pos,                      // Output: Station positions, z = 3 m per floor, Size: [3, n_users]
                                 arma::mat &rx_orientation,              // Output: Station orientation in radians, Size: [3, n_users]
                                 std::vector<arma::mat> &aod,            // Output: Sub-path departure angles in radians, per user, Size: [n_subpath, n_path_out]
                                 std::vector<arma::mat> &aoa,            // Output: Sub-path arrival angles in radians, per user, Size: [n_subpath, n_path_out]
                                 std::vector<arma::mat> &pow,            // Output: Sub-path power, linear, relative to 0 dBm, per user, Size: [n_subpath, n_path_out]
                                 std::vector<arma::vec> &delay,          // Output: Path delays in seconds, per user, Length: n_path_out
                                 std::vector<arma::cube> &M,             // Output: Polarization transfer matrix, interleaved complex, col-major, per user, Size: [8, n_subpath, n_path_out]
                                 std::string ChannelType,                // Channel Model Type (A, B, C, D, E, F) as defined by TGn
                                 double CarrierFreq_Hz = 5.25e9,         // Carrier frequency in Hz
                                 double tap_spacing_s = 10.0e-9,         // Taps spacing in seconds, must be equal to 10 ns divided by a power of 2, TGn = 10e-9
                                 arma::uword n_users = 1,                // Number of users (only for TGac, TGah)
                                 arma::vec Dist_m = {4.99},              // Distance between TX and RX in meters, length n_users or length 1 (if same for all users)
                                 arma::uvec n_floors = {0},              // Number of floors for the TGah model, adjusted for each user, max. 4, length n_users or length 1 (if same for all users)
                                 arma::mat offset_angles = {},           // Offset angles in degree for MU-MIMO channels, empty (TGac auto for n_users > 1), Size: [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS
                                 arma::uword n_subpath = 20,             // Number of sub-paths per path and cluster for Laplacian AS mapping
                                 arma::sword seed = -1,                  // Numeric seed; -1 seeds from the system random device
                                 double KF_linear_overwrite = NAN,       // Overwrites the default KF (linear scale)
                                 double XPR_NLOS_linear_overwrite = NAN, // Overwrites the default cross-polarization ratio (linear scale) for NLOS paths
                                 double SF_std_dB_LOS_overwrite = NAN,   // Overwrites the default shadow fading STD for LOS channels in dB
                                 double SF_std_dB_NLOS_overwrite = NAN,  // Overwrites the default shadow fading STD for NLOS channels in dB
                                 double dBP_m_overwrite = NAN,           // Overwrites the default breakpoint distance in meters
                                 arma::uvec n_walls = {0},               // Number of walls per user TGax models; [n_users] or [1]
                                 double wall_loss = 5.0)                 // Penetration loss for a single wall; TGax defines 5.0 (default) or 7.0

{
    // Input validation
    if (n_users == 0)
        throw std::invalid_argument("Input 'n_users' cannot be 0.");

    if (n_subpath == 0)
        throw std::invalid_argument("Input 'n_subpath' cannot be 0.");

    if (Dist_m.n_elem == 0)
        Dist_m = {4.99};
    if (Dist_m.n_elem != 1 && Dist_m.n_elem != n_users)
        throw std::invalid_argument("Input 'dist' must have length 1 or n_users.");
    if (!Dist_m.is_finite())
        throw std::invalid_argument("Input 'dist' must be finite.");
    if (arma::any(Dist_m <= 0.0))
        throw std::invalid_argument("Input 'dist' must be larger than 0.");
    if (Dist_m.n_elem == 1 && n_users != 1)
        Dist_m = arma::vec(n_users, arma::fill::value(Dist_m[0]));

    if (n_floors.n_elem == 0)
        n_floors = {0};
    if (n_floors.n_elem != 1 && n_floors.n_elem != n_users)
        throw std::invalid_argument("Input 'n_floors' must have length 1 or n_users.");
    if (arma::any(n_floors > 4))
        throw std::invalid_argument("Input 'n_floors' cannot exceed 4.");
    if (n_floors.n_elem == 1 && n_users != 1)
        n_floors = arma::uvec(n_users, arma::fill::value(n_floors[0]));

    if (n_walls.n_elem == 0)
        n_walls = {0};
    if (n_walls.n_elem != 1 && n_walls.n_elem != n_users)
        throw std::invalid_argument("Input 'n_walls' must have length 1 or n_users.");
    if (n_walls.n_elem == 1 && n_users != 1)
        n_walls = arma::uvec(n_users, arma::fill::value(n_walls[0]));

    if (!std::isfinite(CarrierFreq_Hz))
        throw std::invalid_argument("Input 'center_freq' must be finite.");
    if (CarrierFreq_Hz <= 0.1e9)
        throw std::invalid_argument("Input 'center_freq' must be above 100 MHz.");

    // Tap spacing 10 ns / 2^k (k >= 0), relative tolerance 1e-9
    double tap_spacing_ns = tap_spacing_s * 1e9;
    double bw_factor_d = (std::isfinite(tap_spacing_ns) && tap_spacing_ns > 0.0) ? std::round(10.0 / tap_spacing_ns) : 0.0;
    if (!(bw_factor_d >= 1.0 && bw_factor_d <= 9007199254740992.0)) // 1 to 2^53
        throw std::invalid_argument("Input 'tap_spacing' must be equal to 10 ns divided by a power of 2.");
    arma::uword bw_factor = (arma::uword)bw_factor_d; // bandwidth expansion factor
    if ((bw_factor & (bw_factor - 1)) != 0 || std::abs(tap_spacing_ns * bw_factor_d - 10.0) > 1.0e-8)
        throw std::invalid_argument("Input 'tap_spacing' must be equal to 10 ns divided by a power of 2.");

    if (offset_angles.n_elem != 0 && (offset_angles.n_rows != 4 || offset_angles.n_cols != n_users))
        throw std::invalid_argument("Input 'offset_angles_deg' must be empty or have size [4, n_users].");
    if (!offset_angles.is_finite())
        throw std::invalid_argument("Input 'offset_angles_deg' must be finite.");
    if (!std::isfinite(wall_loss))
        throw std::invalid_argument("Input 'wall_loss_dB' must be finite.");

    // ---- Scenario parameters ----

    // Model parameters as in IEEE 802.11-03/940r4, Appendix C
    arma::mat power_clst_dB;      // Cluster power, rows = clusters, cols = paths
    arma::mat AoD_deg;            // Departure angles at the TX
    arma::mat ASD_deg;            // Angular spread at the TX
    arma::mat AoA_deg;            // Arrival angles at the RX
    arma::mat ASA_deg;            // Angular spread at the RX
    arma::vec delay_ns;           // Path delay in ns
    double dBP_m = 0.0;           // Path loss break point in meters
    double SF_std_dB_LOS = 3.0;   // Shadow Fading STD for LOS channels in dB
    double SF_std_dB_NLOS = 3.0;  // Shadow Fading STD for NLOS channels in dB
    double KF_linear = 1.0;       // Boost of LOS power at small distances below break point
    double XPR_NLOS_linear = 2.0; // Cross-polarization ratio (linear scale) for NLOS paths

    if (ChannelType == "A")
    {
        power_clst_dB = {0.0};
        AoD_deg = {45.0};
        ASD_deg = {40.0};
        AoA_deg = {45.0};
        ASA_deg = {40.0};
        delay_ns = {0.0};
        dBP_m = 5.0;
        SF_std_dB_NLOS = 4.0;
        KF_linear = 1.0; // 0 dB
    }
    else if (ChannelType == "B")
    {
        power_clst_dB = {{0.0, -5.4, -10.8, -16.2, -21.7, -INFINITY, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -3.2, -6.3, -9.4, -12.5, -15.6, -18.7, -21.8}};

        AoA_deg = {{4.3, 4.3, 4.3, 4.3, 4.3, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 118.4, 118.4, 118.4, 118.4, 118.4, 118.4, 118.4}};

        ASA_deg = {{14.4, 14.4, 14.4, 14.4, 14.4, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2}};

        AoD_deg = {{225.1, 225.1, 225.1, 225.1, 225.1, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 106.5, 106.5, 106.5, 106.5, 106.5, 106.5, 106.5}};

        ASD_deg = {{14.4, 14.4, 14.4, 14.4, 14.4, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4}};

        delay_ns = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};

        dBP_m = 5.0;
        SF_std_dB_NLOS = 4.0;
        KF_linear = 1.0; // 0 dB
    }
    else if (ChannelType == "C")
    {
        power_clst_dB = {{0.0, -2.1, -4.3, -6.5, -8.6, -10.8, -13.0, -15.2, -17.3, -19.5, -INFINITY, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -5.0, -7.2, -9.3, -11.5, -13.7, -15.8, -18.0, -20.2}};

        AoA_deg = {{290.3, 290.3, 290.3, 290.3, 290.3, 290.3, 290.3, 290.3, 290.3, 290.3, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 332.3, 332.3, 332.3, 332.3, 332.3, 332.3, 332.3, 332.3}};

        ASA_deg = {{24.6, 24.6, 24.6, 24.6, 24.6, 24.6, 24.6, 24.6, 24.6, 24.6, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.4, 22.4, 22.4, 22.4, 22.4, 22.4, 22.4, 22.4}};

        AoD_deg = {{13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4, 56.4}};

        ASD_deg = {{24.7, 24.7, 24.7, 24.7, 24.7, 24.7, 24.7, 24.7, 24.7, 24.7, 0.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5}};

        delay_ns = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 110.0, 140.0, 170.0, 200.0};

        dBP_m = 5.0;
        SF_std_dB_NLOS = 5.0;
        KF_linear = 1.0; // 0 dB
    }
    else if (ChannelType == "D")
    {
        power_clst_dB = {{0.0, -0.9, -1.7, -2.6, -3.5, -4.3, -5.2, -6.1, -6.9, -7.8, -9.0, -11.1, -13.7, -16.3, -19.3, -23.2, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -6.6, -9.5, -12.1, -14.7, -17.4, -21.9, -25.5, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -18.8, -23.2, -25.2, -26.7}};

        AoA_deg = {{158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 158.9, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 320.2, 320.2, 320.2, 320.2, 320.2, 320.2, 320.2, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 276.1, 276.1, 276.1, 276.1}};

        ASA_deg = {{27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 27.7, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.4, 31.4, 31.4, 31.4, 31.4, 31.4, 31.4, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 37.4, 37.4, 37.4, 37.4}};

        AoD_deg = {{332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 332.1, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 49.3, 49.3, 49.3, 49.3, 49.3, 49.3, 49.3, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 275.9, 275.9, 275.9, 275.9}};

        ASD_deg = {{27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 27.4, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.1, 32.1, 32.1, 32.1, 32.1, 32.1, 32.1, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.8, 36.8, 36.8, 36.8}};

        delay_ns = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 110.0, 140.0, 170.0, 200.0, 240.0, 290.0, 340.0, 390.0};

        dBP_m = 10.0;
        SF_std_dB_NLOS = 5.0;
        KF_linear = 2.0; // 3 dB
    }
    else if (ChannelType == "E")
    {
        power_clst_dB = {{-2.6, -3.0, -3.5, -3.9, -4.5, -5.6, -6.9, -8.2, -9.8, -11.7, -13.9, -16.1, -18.3, -20.5, -22.9, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -1.8, -3.2, -4.5, -5.8, -7.1, -9.9, -10.3, -14.3, -14.7, -18.7, -19.9, -22.4, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -7.9, -9.6, -14.2, -13.8, -18.6, -18.1, -22.8, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -20.6, -20.5, -20.7, -24.6}};

        AoA_deg = {{163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 163.7, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 251.8, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 182.0, 182.0, 182.0, 182.0}};

        ASA_deg = {{35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 35.8, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 37.4, 37.4, 37.4, 37.4, 37.4, 37.4, 37.4, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.3, 40.3, 40.3, 40.3}};

        AoD_deg = {{105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 105.6, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 293.1, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.9, 61.9, 61.9, 61.9, 61.9, 61.9, 61.9, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 275.7, 275.7, 275.7, 275.7}};

        ASD_deg = {{36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 36.1, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 42.5, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 38.7, 38.7, 38.7, 38.7}};

        delay_ns = {0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 110.0, 140.0, 180.0, 230.0, 280.0, 330.0, 380.0, 430.0, 490.0, 560.0, 640.0, 730.0};

        dBP_m = 20.0;
        SF_std_dB_NLOS = 6.0;
        KF_linear = 4.0; // 6 dB
    }
    else if (ChannelType == "F")
    {
        power_clst_dB = {{-3.3, -3.6, -3.9, -4.2, -4.6, -5.3, -6.2, -7.1, -8.2, -9.5, -11.0, -12.5, -14.3, -16.7, -19.9, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -1.8, -2.8, -3.5, -4.4, -5.3, -7.4, -7.0, -10.3, -10.4, -13.8, -15.7, -19.9, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -5.7, -6.7, -10.4, -9.6, -14.1, -12.7, -18.5, -INFINITY, -INFINITY, -INFINITY},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -8.8, -13.3, -18.7, -INFINITY, -16.3, -21.2},
                         {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -12.9, -14.2, -INFINITY, -INFINITY}};

        AoA_deg = {{315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 315.1, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 180.4, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 74.7, 74.7, 74.7, 74.7, 74.7, 74.7, 74.7, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 251.5, 251.5, 251.5, 0.0, 246.2, 246.2},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 68.5, 68.5, 0.0, 0.0}};

        ASA_deg = {{48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.6, 28.6, 28.6, 0.0, 38.2, 38.2},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.7, 30.7, 0.0, 0.0}};

        AoD_deg = {{56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 56.2, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 183.7, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 153.0, 153.0, 153.0, 153.0, 153.0, 153.0, 153.0, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 112.5, 112.5, 112.5, 0.0, 62.3, 62.3},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 291.0, 291.0, 0.0, 0.0}};

        ASD_deg = {{41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 41.6, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 55.2, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47.4, 47.4, 47.4, 47.4, 47.4, 47.4, 47.4, 0.0, 0.0, 0.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.2, 27.2, 27.2, 0.0, 38.0, 38.0},
                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33.0, 33.0, 0.0, 0.0}};

        delay_ns = {0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 110.0, 140.0, 180.0, 230.0, 280.0, 330.0, 400.0, 490.0, 600.0, 730.0, 880.0, 1050.0};

        dBP_m = 30.0;
        SF_std_dB_NLOS = 6.0;
        KF_linear = 4.0; // 6 dB
    }
    else
        throw std::invalid_argument("Input 'channel_type' must be one of A, B, C, D, E, F.");

    // Lower SF by 1 dB for frequencies below 1 GHz
    // See: IEEE 802.11-968r4 TGah Channel Model, Table 2
    if (CarrierFreq_Hz < 1.0e9)
        SF_std_dB_LOS -= 1.0, SF_std_dB_NLOS -= 1.0;

    // Override rules: NAN selects the model default; any other value must be finite and >= 0 (KF, SF spreads) or > 0
    // (XPR, breakpoint distance). An infinite KF removes all NLOS paths, an infinite SF spread gives NaN path powers,
    // XPR = 0 or Inf has no Jones matrix and a zero breakpoint gives log10(0).
    auto apply_override = [](double value, double model_default, bool positive, const char *name)
    {
        if (std::isnan(value))
            return model_default;
        if (!std::isfinite(value) || value < 0.0 || (positive && value == 0.0))
            throw std::invalid_argument(std::string("Input '") + name + "' must be NaN (model default) or finite and " +
                                        (positive ? "> 0." : ">= 0."));
        return value;
    };
    KF_linear = apply_override(KF_linear_overwrite, KF_linear, false, "kf");
    SF_std_dB_LOS = apply_override(SF_std_dB_LOS_overwrite, SF_std_dB_LOS, false, "sf_std_los_dB");
    SF_std_dB_NLOS = apply_override(SF_std_dB_NLOS_overwrite, SF_std_dB_NLOS, false, "sf_std_nlos_dB");
    XPR_NLOS_linear = apply_override(XPR_NLOS_linear_overwrite, XPR_NLOS_linear, true, "xpr_nlos");
    dBP_m = apply_override(dBP_m_overwrite, dBP_m, true, "dist_breakpoint");

    // Get the number of paths and clusters
    arma::uword n_cluster = power_clst_dB.n_rows;
    arma::uword n_path = power_clst_dB.n_cols;

    // PDP tap interpolation for TGac
    // See: IEEE 802.11-09/0308r12, Section 2
    if (bw_factor > 1)
    {
        // Reserve memory for interpolated data
        arma::mat power_clst_dB_int(n_cluster, bw_factor * n_path);
        arma::mat AoA_deg_int(n_cluster, bw_factor * n_path);
        arma::mat ASA_deg_int(n_cluster, bw_factor * n_path);
        arma::mat AoD_deg_int(n_cluster, bw_factor * n_path);
        arma::mat ASD_deg_int(n_cluster, bw_factor * n_path);
        arma::vec delay_ns_int(bw_factor * n_path);

        arma::uword i_path_int = 0;
        for (arma::uword i_path = 0; i_path < n_path; ++i_path)
        {
            // Copy TGn data for current path
            std::memcpy(power_clst_dB_int.colptr(i_path_int), power_clst_dB.colptr(i_path), n_cluster * sizeof(double));
            std::memcpy(AoA_deg_int.colptr(i_path_int), AoA_deg.colptr(i_path), n_cluster * sizeof(double));
            std::memcpy(ASA_deg_int.colptr(i_path_int), ASA_deg.colptr(i_path), n_cluster * sizeof(double));
            std::memcpy(AoD_deg_int.colptr(i_path_int), AoD_deg.colptr(i_path), n_cluster * sizeof(double));
            std::memcpy(ASD_deg_int.colptr(i_path_int), ASD_deg.colptr(i_path), n_cluster * sizeof(double));
            delay_ns_int[i_path_int] = delay_ns[i_path];

            // Check if we need to interpolate
            bool no_interpolation_needed = true;
            for (arma::uword i_clst = 0; i_clst < n_cluster; ++i_clst)
                if (i_path < n_path - 1 && power_clst_dB(i_clst, i_path) > -200.0 && power_clst_dB(i_clst, i_path + 1) > -200.0)
                {
                    no_interpolation_needed = false;
                    break;
                }

            if (no_interpolation_needed)
            {
                ++i_path_int;
                continue;
            }

            // Interpolate
            for (arma::uword i_int = 1; i_int < bw_factor; ++i_int)
            {
                // Delays
                double d0 = delay_ns(i_path);
                double d1 = (i_path < n_path - 1) ? delay_ns(i_path + 1) : INFINITY;
                double dI = d0 + double(i_int) * 10.0 / double(bw_factor);
                delay_ns_int[i_path_int + i_int] = dI;

                // Power, angles and spread
                for (arma::uword i_clst = 0; i_clst < n_cluster; ++i_clst)
                {
                    double p0 = power_clst_dB(i_clst, i_path);
                    double p1 = (i_path < n_path - 1) ? power_clst_dB(i_clst, i_path + 1) : -INFINITY;
                    if (p0 > -200.0 && p1 > -200.0)
                    {
                        // Linear interpolation
                        double pI = p0 + (p1 - p0) * (dI - d0) / (d1 - d0);
                        power_clst_dB_int(i_clst, i_path_int + i_int) = pI;

                        // Copy Angles and spreads
                        AoA_deg_int(i_clst, i_path_int + i_int) = AoA_deg(i_clst, i_path);
                        ASA_deg_int(i_clst, i_path_int + i_int) = ASA_deg(i_clst, i_path);
                        AoD_deg_int(i_clst, i_path_int + i_int) = AoD_deg(i_clst, i_path);
                        ASD_deg_int(i_clst, i_path_int + i_int) = ASD_deg(i_clst, i_path);
                    }
                    else // Ignore rest
                    {
                        power_clst_dB_int(i_clst, i_path_int + i_int) = -INFINITY;
                        AoA_deg_int(i_clst, i_path_int + i_int) = 0.0;
                        ASA_deg_int(i_clst, i_path_int + i_int) = 0.0;
                        AoD_deg_int(i_clst, i_path_int + i_int) = 0.0;
                        ASD_deg_int(i_clst, i_path_int + i_int) = 0.0;
                    }
                }
            }
            i_path_int += bw_factor;
        }

        // Update TGn values
        power_clst_dB = power_clst_dB_int.head_cols(i_path_int);
        AoA_deg = AoA_deg_int.head_cols(i_path_int);
        ASA_deg = ASA_deg_int.head_cols(i_path_int);
        AoD_deg = AoD_deg_int.head_cols(i_path_int);
        ASD_deg = ASD_deg_int.head_cols(i_path_int);
        delay_ns = delay_ns_int.head(i_path_int);

        n_path = delay_ns.n_elem;
    }

    // ---- Helper functions ----

    // Random number generator; the draws use the portable helpers qd_rng_uniform and qd_rng_normal
    std::mt19937_64 rng;
    if (seed != -1)
        rng.seed(seed);
    else
    {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        rng.seed(seq);
    }

    const double V_PI = 3.141592653589793;
    const double V_2PI = 6.283185307179586;
    const double V_PI_2 = 1.570796326794897;

    auto dB_2_linear = [](double x)
    { return std::pow(10.0, x * 0.1); };

    auto rad = [&](double deg)
    {
        double r = deg * 0.017453292519943;
        r = std::fmod(r - V_PI, V_2PI);
        if (r < 0)
            r += V_2PI;
        return r - V_PI;
    };

    // ---- Model functions ----

    // Offset angle generation
    // see IEEE 802.11-09/0308r12, A.2 – MATLAB-Independent Implementation
    auto get_offset_angles = [](arma::uword seed, arma::uword nValues, double *ptr)
    {
        arma::uword a = 16807;      // (7^5)
        arma::uword m = 2147483647; // (2^31-1)
        double md = 1.0 / (double)m;

        for (arma::uword i = 0; i < nValues; ++i)
        {
            double rand_variant = (double)seed * md;
            ptr[i] = (rand_variant - 0.5) * 360.0;
            seed = (a * seed) % m;
        }
    };

    // Generate N random AoD/AoA angles (independent Laplace with rms AS_tx/AS_rx)
    // and assign joint powers w ∝ f_tx * f_rx
    auto gen_joint_angles = [&](double AS_tx_deg, double AS_rx_deg, std::size_t N,
                                double *aod, double *aoa, double *w)
    {
        if (N == 0)
            return;

        AS_tx_deg = std::max(0.0, AS_tx_deg);
        AS_rx_deg = std::max(0.0, AS_rx_deg);

        constexpr double tol = 1e-12;
        const bool tx_zero = AS_tx_deg <= tol;
        const bool rx_zero = AS_rx_deg <= tol;

        // Fast path: both zero => all rays at 0 with equal power
        if (tx_zero && rx_zero)
        {
            const double invN = 1.0 / double(N);
            for (std::size_t i = 0; i < N; ++i)
            {
                aod[i] = 0.0;
                aoa[i] = 0.0;
                w[i] = invN;
            }
            return;
        }

        // Uniform in the open interval (0, 1) from the upper 52 bits of the generator output: avoids log(0)
        auto u01 = [&]
        { return ((double)(rng() >> 12) + 0.5) * (1.0 / 4503599627370496.0); };

        auto laplace_icdf = [](double u, double b)
        { return (u < 0.5) ? b * std::log(2.0 * u) : -b * std::log(2.0 * (1.0 - u)); };

        auto laplace_logpdf = [](double x, double b)
        { return -std::abs(x) / b - std::log(2.0 * b); };

        const double bt = tx_zero ? 0.0 : AS_tx_deg / std::sqrt(2.0);
        const double br = rx_zero ? 0.0 : AS_rx_deg / std::sqrt(2.0);
        const double bt0 = tx_zero ? 0.0 : 1.5 * bt; // proposal scale
        const double br0 = rx_zero ? 0.0 : 1.5 * br; // proposal scale

        // Sample proposals
        for (std::size_t i = 0; i < N; ++i)
        {
            aod[i] = tx_zero ? 0.0 : laplace_icdf(u01(), bt0);
            aoa[i] = rx_zero ? 0.0 : laplace_icdf(u01(), br0);
        }

        // Weight recomputation in log-space (target pdfs), normalize
        auto recompute_weights = [&]
        {
            double maxlogw = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < N; ++i)
            {
                double lwt = tx_zero ? 0.0 : laplace_logpdf(aod[i], bt);
                double lwr = rx_zero ? 0.0 : laplace_logpdf(aoa[i], br);
                w[i] = lwt + lwr; // store logw temporarily
                if (w[i] > maxlogw)
                    maxlogw = w[i];
            }
            double sumw = 0.0;
            for (std::size_t i = 0; i < N; ++i)
            {
                w[i] = std::exp(w[i] - maxlogw); // now w holds linear weights
                sumw += w[i];
            }
            if (!(sumw > 0.0) || !std::isfinite(sumw))
            {
                const double invN = 1.0 / static_cast<double>(N);
                for (std::size_t i = 0; i < N; ++i)
                    w[i] = invN;
                return;
            }
            const double inv = 1.0 / sumw;
            for (std::size_t i = 0; i < N; ++i)
                w[i] *= inv;
        };

        recompute_weights();

        // Weighted RMS correction to hit target AS on each side
        auto rms_correct = [&](double *arr, bool side_zero, double AS)
        {
            if (side_zero)
                return;
            double m2 = 0.0;
            for (std::size_t i = 0; i < N; ++i)
                m2 += w[i] * arr[i] * arr[i];
            const double s = std::sqrt(std::max(0.0, m2));
            if (s > 0.0)
            {
                const double scale = AS / s;
                for (std::size_t i = 0; i < N; ++i)
                    arr[i] *= scale;
            }
        };

        rms_correct(aod, tx_zero, AS_tx_deg);
        rms_correct(aoa, rx_zero, AS_rx_deg);
        recompute_weights(); // Make weights consistent with the final (scaled) angles
        rms_correct(aod, tx_zero, AS_tx_deg);
        rms_correct(aoa, rx_zero, AS_rx_deg);
    };

    // NLOS Jones matrix (linear basis) with common random path phase.
    // Input: XPR (linear H↔V cross-pol ratio). Uses maximal quadrature (±90°) to include linear→circular leakage.
    auto get_Jones_matrix = [&](double XPR_lin,  // Cross-polarization ratio, linear scale
                                double *M_local) // Output: interleaved Re,Im; col-major: (J11,J21 ; J12,J22), rows rx, columns tx, length 8
    {
        if (!M_local || XPR_lin <= 0.0)
            return;

        const double alpha = std::sqrt(XPR_lin / (1.0 + XPR_lin)); // co-pol magnitude
        const double beta = std::sqrt(1.0 / (1.0 + XPR_lin));      // cross-pol magnitude

        const double phi0 = V_2PI * qd_rng_uniform(rng);  // common path phase
        const double theta = V_2PI * qd_rng_uniform(rng); // Shared absolute phase for both columns

        const double Delta = V_PI_2;            // ±90° for max lin→circ mixing
        const double theta_a = theta;           // J_11 (rx V, tx V)
        const double theta_c = theta_a - Delta; // J_21 (rx H, tx V)
        const double theta_d = theta;           // J_22 (rx H, tx H), same absolute phase as J_11
        const double theta_b = theta_d + Delta; // J_12 (rx V, tx H)

        const std::complex<double> g = std::polar(1.0, phi0);

        // J_11
        std::complex<double> J = g * std::polar(alpha, theta_a);
        M_local[0] = J.real();
        M_local[1] = J.imag();

        // J_21
        J = g * std::polar(beta, theta_c);
        M_local[2] = J.real();
        M_local[3] = J.imag();

        // J_12
        J = g * std::polar(beta, theta_b);
        M_local[4] = J.real();
        M_local[5] = J.imag();

        // J_22
        J = g * std::polar(alpha, theta_d);
        M_local[6] = J.real();
        M_local[7] = J.imag();
    };

    // ---- Initialize outputs ----

    arma::uword n_path_out = 1 + n_cluster * n_path; // +1 is for LOS steering matrix

    rx_pos.zeros(3, n_users);
    rx_orientation.zeros(3, n_users);

    aod.clear();
    aoa.clear();
    pow.clear();
    delay.clear();
    M.clear();

    for (arma::uword n = 0; n < n_users; ++n)
    {
        aod.push_back(arma::mat(n_subpath, n_path_out, arma::fill::zeros));
        aoa.push_back(arma::mat(n_subpath, n_path_out, arma::fill::zeros));
        pow.push_back(arma::mat(n_subpath, n_path_out, arma::fill::zeros));
        delay.push_back(arma::vec(n_path_out, arma::fill::zeros));
        M.push_back(arma::cube(8, n_subpath, n_path_out, arma::fill::zeros));
    }

    // ---- Model steps ----

    // Path loss for each user
    arma::vec Pathloss_dB(n_users, arma::fill::value(20.0 * std::log10(4.0 * V_PI * CarrierFreq_Hz / 3.0e8)));
    for (int i_user = 0; i_user < (int)n_users; ++i_user)
    {
        if (Dist_m[i_user] < dBP_m)
            Pathloss_dB[i_user] += 20.0 * std::log10(Dist_m[i_user]);
        else
            Pathloss_dB[i_user] += 20.0 * std::log10(dBP_m) + 35.0 * std::log10(Dist_m[i_user] / dBP_m);

        // Add floor attenuation
        if (CarrierFreq_Hz < 1e9) // TGah model
        {
            // See: IEEE 802.11-968r4 TGah Channel Model, Table 3
            if (n_floors[i_user] == 1)
                Pathloss_dB[i_user] += 12.9;
            else if (n_floors[i_user] == 2)
                Pathloss_dB[i_user] += 18.7;
            else if (n_floors[i_user] == 3)
                Pathloss_dB[i_user] += 24.4;
            else if (n_floors[i_user] == 4)
                Pathloss_dB[i_user] += 27.7;
        }
        else // TGax model
        {
            // See: IEEE 802.11-14/0882r4, Section 3.1.1 Indoor floor and wall penetration loss
            double nn = (double)n_floors[i_user];
            Pathloss_dB[i_user] += 18.3 * std::pow(nn, (nn + 2.0) / (nn + 1.0) - 0.46);
        }

        // Add wall attenuation
        // See: IEEE 802.11-14/0882r4, Section 3.1.1 Indoor floor and wall penetration loss
        Pathloss_dB[i_user] += wall_loss * (double)n_walls[i_user];
    }

    // Generate Shadow Fading for each user
    arma::vec Shadow_Fading_dB(n_users);
    for (arma::uword i_user = 0; i_user < n_users; ++i_user)
    {
        // Use LOS SF for distances below break point, and NLOS SF above
        double SF_std_dB = (Dist_m[i_user] < dBP_m) ? SF_std_dB_LOS : SF_std_dB_NLOS;

        // Overwrite SF std if user is on a different floor
        if (CarrierFreq_Hz < 1e9) // TGah model
        {
            // See: IEEE 802.11-968r4 TGah Channel Model, Table 3
            if (n_floors[i_user] == 1)
                SF_std_dB = 7.0;
            else if (n_floors[i_user] == 2)
                SF_std_dB = 2.8;
            else if (n_floors[i_user] == 3)
                SF_std_dB = 1.7;
            else if (n_floors[i_user] == 4)
                SF_std_dB = 1.5;
        }

        Shadow_Fading_dB[i_user] = qd_rng_normal(rng); // standard normal sample
        Shadow_Fading_dB[i_user] *= SF_std_dB;
    }

    // Generate offset angles for MU-MIMO
    // See: TGac IEEE 802.11-09/0308r12, Appendix A – Generation of Pseudorandom Per-User AoA and AoD Offsets for MU-MIMO Channel Model
    // For 1 user, use TGn angles (AoD_offset = AoA_offset = 0.0)
    arma::vec AoD_LOS_offset_deg = offset_angles.empty() ? arma::vec(n_users) : offset_angles.row(0).t();
    arma::vec AoD_NLOS_offset_deg = offset_angles.empty() ? arma::vec(n_users) : offset_angles.row(1).t();
    arma::vec AoA_LOS_offset_deg = offset_angles.empty() ? arma::vec(n_users) : offset_angles.row(2).t();
    arma::vec AoA_NLOS_offset_deg = offset_angles.empty() ? arma::vec(n_users) : offset_angles.row(3).t();

    if (n_users != 1 && offset_angles.empty())
    {
        get_offset_angles(608341199, n_users, AoD_LOS_offset_deg.memptr());
        get_offset_angles(1468335517, n_users, AoD_NLOS_offset_deg.memptr());
        get_offset_angles(266639588, n_users, AoA_LOS_offset_deg.memptr());
        get_offset_angles(115415752, n_users, AoA_NLOS_offset_deg.memptr());
    }

    // User loop (only for TGac, TGn has only one user)
    for (int i_user = 0; i_user < (int)n_users; ++i_user)
    {
        // Calculate relative RX position based on the AoD_LOS_offset_deg and Dist_m
        double ang = rad(AoD_LOS_offset_deg[i_user]);
        double *pos = rx_pos.colptr(i_user);
        pos[0] = std::cos(ang) * Dist_m[i_user]; // x-position
        pos[1] = std::sin(ang) * Dist_m[i_user]; // y-position
        pos[2] = 3.0 * (double)n_floors[i_user]; // floor indicator, assuming floor height of 3 meters

        // Calculate the RX heading angles
        // i.e. how does the RX need to be oriented (Euler angles) so that it sees the TX at the given AoA_LOS_offset_deg
        double yaw_deg = AoD_LOS_offset_deg[i_user] + 180.0 - AoA_LOS_offset_deg[i_user];
        double *orientation = rx_orientation.colptr(i_user);
        orientation[2] = rad(yaw_deg); // heading angle

        // The PDP for all paths and clusters, initialized to 1
        arma::vec pdp_linear(n_path_out, arma::fill::ones);

        // Add LOS path for small distances below break-point, only when RX is on the same floor
        bool has_los = (Dist_m[i_user] < dBP_m && n_floors[i_user] == 0) || !std::isnan(KF_linear_overwrite);
        double K_los = has_los ? KF_linear / (KF_linear + 1.0) : 0.0;
        double K_nlos = has_los ? 1.0 / (KF_linear + 1.0) : 1.0;
        pdp_linear[0] = K_los * dB_2_linear(power_clst_dB[0]);

        // Generate LOS path
        // - This will generate the MIMO steering matrix in the MIMO processing step
        pow[i_user].at(0, 0) = 1.0; // Only 1 non-zero sub-path for the LOS
        aod[i_user].col(0).fill(AoD_LOS_offset_deg[i_user]);
        aoa[i_user].col(0).fill(AoD_LOS_offset_deg[i_user] + 180.0);
        M[i_user].at(0, 0, 0) = 1.0;
        M[i_user].at(6, 0, 0) = -1.0;

        // Generate NLOS paths
        for (arma::uword i_path = 0; i_path < n_path; ++i_path)
            for (arma::uword i_clst = 0; i_clst < n_cluster; ++i_clst)
            {
                arma::uword path_index = i_path * n_cluster + i_clst + 1;

                // Generate NLOS subpath angles and powers with a Laplacian PAS
                gen_joint_angles(ASD_deg(i_clst, i_path), ASA_deg(i_clst, i_path), n_subpath,
                                 aod[i_user].colptr(path_index),
                                 aoa[i_user].colptr(path_index),
                                 pow[i_user].colptr(path_index));

                // Apply NLOS offset angles for TGac MU-MIMO to all subpaths
                aod[i_user].col(path_index) += AoD_deg.at(i_clst, i_path) + AoD_NLOS_offset_deg[i_user];
                aoa[i_user].col(path_index) += AoA_deg.at(i_clst, i_path) + AoA_NLOS_offset_deg[i_user] + yaw_deg;

                // Calculate cluster-powers
                pdp_linear[path_index] = dB_2_linear(power_clst_dB(i_clst, i_path));
                if (path_index == 1) // Reduce power of the 1st NLOS cluster if LOS path is present
                    pdp_linear[path_index] *= K_nlos;

                // Get the polarization transfer matrices M for each sub-path
                // This also assigns a random phase to each subpath
                for (arma::uword i_sub = 0; i_sub < n_subpath; ++i_sub)
                    get_Jones_matrix(XPR_NLOS_linear, M[i_user].slice_colptr(path_index, i_sub));

                // Set delays
                delay[i_user].at(path_index) = delay_ns[i_path] * 1.0e-9;
            }

        // Transform to radians
        aod[i_user].transform(rad);
        aoa[i_user].transform(rad);

        // Normalize cluster powers to unit-sum-power; PL and SF are applied after the selection of the paths
        pdp_linear /= arma::accu(pdp_linear);
        const double large_scale_gain = dB_2_linear(-Pathloss_dB[i_user] + Shadow_Fading_dB[i_user]);

        // Remove zero-power paths from the output and scale path powers to include PDP, PL and SF
        // The threshold applies to the normalized PDP, so the number of paths does not depend on PL, SF, floors or walls
        arma::uword j_path = 0; // Current non-zero path index, starts at 0
        for (arma::uword i_path = 0; i_path < n_path_out; ++i_path)
        {
            // Get current path power, can be 0
            double p_path = pdp_linear[i_path];

            // Keep only non-zero paths
            if (p_path > 1.0e-14)
            {
                p_path *= large_scale_gain;

                // Scale sub-path powers to contain the path power
                double *vi = pow[i_user].colptr(i_path);
                double *vo = pow[i_user].colptr(j_path);
                for (arma::uword i_sub = 0; i_sub < n_subpath; ++i_sub)
                    vo[i_sub] = vi[i_sub] * p_path;

                // Copy sub-path angles and path delay
                if (i_path != j_path)
                {
                    vi = aod[i_user].colptr(i_path);
                    vo = aod[i_user].colptr(j_path);
                    std::memcpy(vo, vi, n_subpath * sizeof(double));

                    vi = aoa[i_user].colptr(i_path);
                    vo = aoa[i_user].colptr(j_path);
                    std::memcpy(vo, vi, n_subpath * sizeof(double));

                    vi = M[i_user].slice_memptr(i_path);
                    vo = M[i_user].slice_memptr(j_path);
                    std::memcpy(vo, vi, 8 * n_subpath * sizeof(double));

                    delay[i_user].at(j_path) = delay[i_user].at(i_path);
                }
                ++j_path;
            }
        }

        // Resize outputs to correct number of nonzero paths
        if (j_path != n_path_out)
        {
            pow[i_user].resize(n_subpath, j_path);
            aod[i_user].resize(n_subpath, j_path);
            aoa[i_user].resize(n_subpath, j_path);
            M[i_user].resize(8, n_subpath, j_path);
            delay[i_user].resize(j_path);
        }
    }
}

#endif
