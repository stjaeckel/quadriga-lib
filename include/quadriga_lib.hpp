// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_lib_H
#define quadriga_lib_H

#define QUADRIGA_LIB_VERSION_STR "0.12.0"

#include <armadillo>
#include <string>
#include <vector>
#include <array>
#include <memory>

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "arma::uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(size_t), "arma::uword and size_t have different sizes");
static_assert(sizeof(unsigned long long) == sizeof(size_t), "unsigned long and size_t have different sizes");

// OpenMP-safe exception handling
// Declare `bool` + `std::string` before the parallel region, pass them in,
// and rethrow a runtime_error after the region if the flag was set.
#define OMP_SAFE_CALL(error_flag, error_msg, ...)                 \
    try                                                           \
    {                                                             \
        __VA_ARGS__;                                              \
    }                                                             \
    catch (const std::exception &e)                               \
    {                                                             \
        _Pragma("omp critical") if (!(error_flag))                \
        {                                                         \
            (error_msg) = e.what();                               \
            (error_flag) = true;                                  \
        }                                                         \
    }                                                             \
    catch (...)                                                   \
    {                                                             \
        _Pragma("omp critical") if (!(error_flag))                \
        {                                                         \
            (error_msg) = "Unknown exception in parallel region"; \
            (error_flag) = true;                                  \
        }                                                         \
    }

#include "quadriga_math.hpp"
#include "quadriga_arrayant.hpp"
#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

namespace quadriga_lib
{
    // Returns the version number as a string in format (x.y.z)
    std::string quadriga_lib_version();

    // Returns the armadillo version used by quadriga-lib in format (x.y.z)
    std::string quadriga_lib_armadillo_version();

    // Check if AVX2 is supported
    bool quadriga_lib_has_AVX2();

    // Check if CUDA is supported
    bool quadriga_lib_has_CUDA();

    // Channel generation function for IEEE TGn, TGac, TGax and TGah indoor channel models
    // - Returns vector of channel objects, length n_users
    // - Depends on arrayant and channel classes
    // - 2D model, no elevation angles
    std::vector<channel<double>> get_channels_ieee_indoor(
        const arrayant<double> &ap_array,  // Access point array antenna with 'n_tx' elements (= ports after element coupling)
        const arrayant<double> &sta_array, // Mobile station array antenna with 'n_rx' elements (= ports after element coupling)
        std::string ChannelType,           // Channel Model Type (A, B, C, D, E, F) as defined by TGn
        double CarrierFreq_Hz = 5.25e9,    // Carrier frequency in Hz
        double tap_spacing_s = 10.0e-9,    // Taps spacing in seconds, must be equal to 10 ns divided by a power of 2, TGn = 10e-9
        arma::uword n_users = 1,           // Number of user (only for TGac, TGah)
        double observation_time = 0.0,     // Channel observation time in seconds (0.0 = static channel)
        double update_rate = 1.0e-3,       // Channel update interval in seconds
        double speed_station_kmh = 0.0,    // Movement speed of the station in km/h (optional feature, default = 0), movement direction = AoA_offset
        double speed_env_kmh = 1.2,        // Movement speed of the environment in km/h (default = 1.2 for TGn) use 0.089 for TGac
        arma::vec Dist_m = {4.99},         // Distance between TX and RX in meters, length n_users or length 1 (if same for all users)
        arma::uvec n_floors = {0},         // Number of floors for the TGah model, adjusted for each user, up to 4 floors, length n_users or length 1 (if same for all users)
        bool uplink = false,               // Default channel direction is downlink, set uplink to true to get reverse direction
        arma::mat offset_angles = {},      // Offset angles in degree for MU-MIMO channels, empty (TGac auto for n_users > 1), Size: [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS
        arma::uword n_subpath = 20,        // Number of sub-paths per path and cluster for Laplacian AS mapping
        double Doppler_effect = 50.0,      // Special Doppler effects in models D, E (fluorescent lights, value = mains freq.) and F (moving vehicle speed in kmh), use 0.0 to disable
        arma::sword seed = -1,             // Numeric seed, optional, value -1 disabled seed and uses system random device
        double KF_linear = NAN,            // Overwrites the default KF (linear scale)
        double XPR_NLOS_linear = NAN,      // Overwrites the default Cross-polarization ratio (linear scale) for NLOS paths
        double SF_std_dB_LOS = NAN,        // Overwrites the default Shadow Fading STD for LOS channels in dB
        double SF_std_dB_NLOS = NAN,       // Overwrites the default Shadow Fading STD for NLOS channels in dB
        double dBP_m = NAN,                // Overwrites the default breakpoint distance in meters
        arma::uvec n_walls = {0},          // Number of walls per user TGax models; [n_users] or [1]
        double wall_loss = 5.0);           // Penetration loss for a single wall; TGax defines 5.0 (default) or 7.0

    // Initialize rays; float only; returns n_ray; min 190 byte per ray
    arma::uword ray_init(
        arma::uword n_ray_target,                         // Target number of rays
        arma::uword n_freq,                               // Number of frequencies
        float Ox, float Oy, float Oz,                     // Origin position
        float max_path_length,                            // Maximum path length
        arma::fmat *orig = nullptr,                       // Ray origins in GCS, [n_ray, 3]
        arma::fmat *dest = nullptr,                       // Ray destinations in GCS, [n_ray, 3]
        arma::fmat *trivec = nullptr,                     // Beam wavefront vertices relative to origin, [n_ray, 9]
        arma::fmat *tridir = nullptr,                     // Vertex-ray directions, [n_ray, 9]
        arma::Col<short> *mtl_ind_prev = nullptr,         // Previous medium (0 = outside), [n_ray]
        arma::Col<short> *mtl_ind_current = nullptr,      // Current medium (0 = outside), [n_ray]
        arma::Col<short> *mtl_ind_buffer = nullptr,       // Next-transition buffer (0 = empty), [n_ray]
        arma::fmat *path_dir_prev = nullptr,              // Physical ray direction, [n_ray, 3]
        arma::fmat *acc_dist = nullptr,                   // Accumulated in-layer distance, [n_ray, 2]
        std::vector<quadriga_lib::path> *paths = nullptr, // Path data storage, 64 byte + overflow, [n_ray]
        const arma::fmat *mesh = nullptr,                 // Optional: faces of the triangular mesh for sphere size detection, [ n_mesh, 9 ]
        const arma::u32_vec *sub_mesh_index = nullptr,    // Optional: Sub-mesh index, 0-based, [n_sub]
        const arma::fmat *rx_points = nullptr,            // Receive points in 3D Space, Size: [n_points, 3]
        bool scalar_mode = false);                        // Switch for EM mode or scalar mode

    // Flag rays for subdivision
    std::vector<bool> ray_subdivide_flag(
        const arma::fmat &mesh,                       // Faces of the triangular mesh, [n_mesh, 9]
        const arma::fmat &orig,                       // Ray origins in GCS, [n_ray, 3]
        const arma::fmat &dest,                       // Ray destinations in GCS, [n_ray, 3]
        const arma::u32_vec &fbs_ind,                 // 1-based FBS face index, 0 = no hit, [n_ray]
        const arma::fmat &trivec,                     // Beam wavefront vertices relative to ray origin, [n_ray, 9]
        const arma::fmat &tridir,                     // Vertex-ray directions, Cartesian [n_ray, 9]
        const std::vector<quadriga_lib::path> &paths, // Path data storage, [n_ray]
        const arma::Col<short> &mtl_ind_current,      // Current medium (0 = outside), [n_ray]
        uint8_t max_no_interactions = 20,             // Total number of interactions per ray, 0-255
        uint8_t max_no_subdivisions = 2,              // Number of subdivisions, 0-255
        float subdivision_tolerance_m = 3.0f);        // Max. beam edge length before subdivision

    // Progress rays to next iteration step
    // - Retuns number of rays in new launch configuration
    // - Termination conditions: below min gain, reached destination
    // - Rays exceeding any of their assigned limits (reflections, transmissions, subdivision) are terminated
    // - Launch configuration is updated in-place
    std::array<unsigned, 4> ray_progress(
        const arma::fmat &mesh,                                              // Faces of the triangular mesh, [n_mesh, 9]
        const arma::uvec &mtl_ind,                                           // 1-based material index per face, [n_mesh]
        const std::unordered_map<std::string, std::vector<float>> &mtl_prop, // Material properties
        const arma::fvec &center_frequency,                                  // Center frequencies in Hz, [n_freq]
        float Ox, float Oy, float Oz,                                        // Global origin position
        arma::fmat &orig,                                                    // Ray origins in GCS, updated in-place, [n_ray, 3]
        arma::fmat &dest,                                                    // Ray destinations in GCS, updated in-place, [n_ray, 3]
        arma::Col<short> &mtl_ind_prev,                                      // Previous medium (0 = outside), updated in-place, [n_ray]
        arma::Col<short> &mtl_ind_current,                                   // Current medium (0 = outside), updated in-place, [n_ray]
        arma::Col<short> &mtl_ind_buffer,                                    // Next-transition buffer (0 = empty), updated in-place, [n_ray]
        arma::fmat &path_dir_prev,                                           // Physical ray direction, updated in-place, [n_ray, 3]
        arma::fmat &acc_dist,                                                // Accumulated in-layer distance, updated in-place, [n_ray, 2]
        std::vector<quadriga_lib::path> &paths,                              // Path data storage, 64 byte + overflow, updated in-place, [n_ray]
        arma::fmat *trivec = nullptr,                                        // Optional: Beam wavefront vertices relative to origin, updated in-place, [n_ray, 9]
        arma::fmat *tridir = nullptr,                                        // Optional: Vertex-ray directions, updated in-place, Cartesian [n_ray, 9]
        const arma::u32_vec *sub_mesh_index = nullptr,                       // Optional: Sub-mesh index, 0-based, [n_sub]
        const arma::fmat *aabb = nullptr,                                    // Optional: Bounding box matrix; [n_sub, 6]
        uint8_t max_no_interactions = 20,                                    // Total number of interactions per ray, 0-255
        uint8_t max_no_reflections = 10,                                     // Number of reflections, 0-255
        uint8_t max_no_transmissions = 10,                                   // Number of transmissions / refractions, 0-255
        uint8_t max_no_subdivisions = 2,                                     // Number of subdivisions, 0-255
        float min_gain_dB = -140.0f,                                         // Minimum gain below which a path is terminated
        float subdivision_tolerance_m = 3.0f,                                // Max. beam edge length before subdivision
        float thin_slab_threshold = 0.15f,                                   // Resolve threshold on the round-trip in-slab amplitude
        bool refraction_mode = true,                                         // Switch for straight-path / refraction
        bool scalar_mode = false,                                            // Switch for EM mode or scalar mode
        const arma::u32_vec *no_interact_in = nullptr,                       // Optional: Externally computed intersection count per ray, skips the internal intersector, [n_ray]
        const arma::u32_vec *fbs_ind_in = nullptr,                           // Optional: Externally computed 1-based FBS face index, 0 = no hit, [n_ray]
        const arma::u32_vec *sbs_ind_in = nullptr,                           // Optional: Externally computed 1-based SBS face index, 0 = no hit, [n_ray]
        const std::vector<bool> *subdiv_flag_in = nullptr);                  // Optional: List of beam to subdivide, [n_ray]

    // Compute the committed paths (paths that hit a receiver)
    // - Adds committed paths to the provided paths_commit vector
    // - Returns the number of newly committed paths
    // - Only includes points that are not shaded by the mesh at the next FBS location
    // - Drops rays that fall below the minimum gain (@ center_frequency[0]) or exceed the maximum length
    arma::uword ray_commit(
        const std::vector<quadriga_lib::path> &paths,                        // In-flight path data storage, [n_ray]
        std::vector<quadriga_lib::path> &paths_commit,                       // Committed data storage, extended by [n_ray_commit]
        const arma::fmat &mesh,                                              // Faces of the triangular mesh, [n_mesh, 9]
        const std::unordered_map<std::string, std::vector<float>> &mtl_prop, // Material properties
        const arma::fvec &center_frequency,                                  // Center frequencies in Hz, [n_freq]
        const arma::fmat &orig,                                              // Ray origins in GCS, [n_ray, 3]
        const arma::u32_vec &fbs_ind,                                        // 1-based FBS face index, 0 = no hit, [n_ray]
        const arma::fmat &trivec,                                            // Beam wavefront vertices relative to ray origin, [n_ray, 9]
        const arma::fmat &tridir,                                            // Vertex-ray directions, Cartesian [n_ray, 9]
        const arma::Col<short> &mtl_ind_current,                             // Current medium (0 = outside), [n_ray]
        const arma::fmat &points,                                            // Receive points in 3D Space, Size: [n_points, 3]
        const arma::u32_vec *sub_cloud_index = nullptr,                      // Sub-cloud index, 0-based, Length: [n_sub]
        const arma::u32_vec *point_index = nullptr,                          // Mapping "points" to "pointsR", 1-based, Length: [ n_pointsR ]
        const std::vector<bool> *subdiv_flag_in = nullptr,                   // Optional: List of beam to subdivide, [n_ray], NULL/empty = none
        float max_path_length = 10e3,                                        // Maximum path length
        float min_gain_dB = -140.0f,                                         // Minimum gain below which a path is terminated
        uint8_t min_no_segments = 0,                                         // Minimum number of segment required to commit a path
        bool ignore_direct_path = false);                                    // Flag to ignore all paths that have no reflections (covered separately by calc_diffraction_gain)

}

#endif