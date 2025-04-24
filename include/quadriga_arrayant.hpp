// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#ifndef quadriga_arrayant_H
#define quadriga_arrayant_H

#include <armadillo>
#include <string>

namespace quadriga_lib
{
    template <typename dtype> // float or double
    class arrayant
    {
    public:
        std::string name = "empty"; // Name of the array antenna object

        // Electric field of the array antenna elements in polar-spheric coordinates: Size [n_elevation, n_azimuth, n_elements]
        arma::Cube<dtype> e_theta_re;                // Vertical component of the electric field, real part
        arma::Cube<dtype> e_theta_im;                // Vertical component of the electric field, imaginary part
        arma::Cube<dtype> e_phi_re;                  // Horizontal component of the electric field, real part
        arma::Cube<dtype> e_phi_im;                  // Horizontal component of the electric field, imaginary part
        arma::Col<dtype> azimuth_grid;               // Azimuth angles in pattern (theta) in [rad], between -pi and pi, sorted
        arma::Col<dtype> elevation_grid;             // Elevation angles in pattern (phi) in [rad], between -pi/2 and pi/2, sorted
        arma::Mat<dtype> element_pos;                // Element positions (optional), Size: Empty or [3, n_elements]
        arma::Mat<dtype> coupling_re;                // Coupling matrix, real part (optional), Size: [n_elements, n_ports]
        arma::Mat<dtype> coupling_im;                // Coupling matrix, imaginary part (optional), Size: [n_elements, n_ports]
        dtype center_frequency = dtype(299792458.0); // Center frequency in [Hz] (optional)
        bool read_only = false;                      // Prevent member functions from writing to the properties
        dtype *check_ptr[9];                         // Data pointers for quick validation
        arrayant() {};                               // Default constructor

        // Functions to determine the size of the array antenna properties
        arma::uword n_elevation() const; // Number of elevation angles
        arma::uword n_azimuth() const;   // Number of azimuth angles
        arma::uword n_elements() const;  // Number of antenna elements
        arma::uword n_ports() const;     // Number of ports (after coupling of elements)

        // Adds elements of "new_arrayant" to the current one
        // - Returns a new arrayant that is a combination of the two
        // - Throws an error if grids don't match
        arrayant<dtype> append(const arrayant<dtype> *new_arrayant) const;

        // Calculate the directivity of an antenna element in dBi
        dtype calc_directivity_dBi(arma::uword i_element) const;

        // Calculates a virtual pattern of the given array by applying coupling and element positions
        // Calling this function without an argument updates the arrayant properties inplace
        void combine_pattern(arrayant<dtype> *output = nullptr);

        // Creates a copy of the array antenna object
        arrayant<dtype> copy() const;

        // Copy antenna elements, enlarges array size if needed (0-based indices)
        void copy_element(arma::uword source, arma::uvec destination);
        void copy_element(arma::uword source, arma::uword destination);

        // Export the antenna pattern geometry to a Wavefront OBJ file, e.g. for visualization in Blender
        // Supported colormaps: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
        void export_obj_file(std::string fn,                   // Filename of the OBJ file
                             dtype directivity_range = 30.0,   // Directivity range of the antenna pattern visualization in dB
                             std::string colormap = "jet",     // Colormap for the visualization
                             dtype object_radius = 1.0,        // Radius in meters of the exported object
                             arma::uword icosphere_n_div = 4,  // Map pattern to an Icosphere with given number of subdivisions
                             arma::uvec i_element = {}) const; // Antenna element indices, 0-based, empty = export all

        // Interpolation of the antenna pattern
        // - Takes azimuth and elevation angles as input and calculates the complex-valued antenna response
        // - "n_out" corresponds to the number of antenna elements
        // - "orientation" and "element_pos_i" are optional inputs
        void interpolate(const arma::Mat<dtype> *azimuth,                 // Azimuth angles [rad],                                  Size [1, n_ang] or [n_out, n_ang]
                         const arma::Mat<dtype> *elevation,               // Elevation angles [rad],                                Size [1, n_ang] or [n_out, n_ang]
                         arma::Mat<dtype> *V_re,                          // Interpolated vertical (e_theta) field, real part,      Size [n_out, n_ang]
                         arma::Mat<dtype> *V_im,                          // Interpolated vertical (e_theta) field, imaginary part, Size [n_out, n_ang]
                         arma::Mat<dtype> *H_re,                          // Interpolated horizontal (e_phi) field, real part,      Size [n_out, n_ang]
                         arma::Mat<dtype> *H_im,                          // Interpolated horizontal (e_phi) field, imaginary part, Size [n_out, n_ang]
                         arma::uvec i_element = {},                       // Element indices, 0-based, optional,                    Vector of length "n_out"
                         const arma::Cube<dtype> *orientation = nullptr,  // Orientation (bank, tilt, head) in [rad], optional,     Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
                         const arma::Mat<dtype> *element_pos_i = nullptr, // Alternative element positions, optional,               Size [3, n_out] or Empty (= use element_pos from arrayant object)
                         arma::Mat<dtype> *dist = nullptr,                // Projected element distances for phase offset,          Size [n_out, n_ang]
                         arma::Mat<dtype> *azimuth_loc = nullptr,         // Azimuth angles [rad] in local antenna coordinates,     Size [n_out, n_ang]
                         arma::Mat<dtype> *elevation_loc = nullptr,       // Elevation angles [rad] in local antenna coordinates,   Size [n_out, n_ang]
                         arma::Mat<dtype> *gamma = nullptr) const;        // Polarization rotation angles in [rad],                 Size [n_out, n_ang]

        // Write array antenna object and layout to QDANT file, returns id in file
        unsigned qdant_write(std::string fn, unsigned id = 0, arma::Mat<unsigned> layout = arma::Mat<unsigned>()) const;

        // Remove zeros from the pattern data. Changes that size of the pattern.
        // Calling this function without an argument updates the arrayant properties inplace
        void remove_zeros(arrayant<dtype> *output = nullptr);

        // Reset the size to zero (the arrayant object will contain no data)
        void reset();

        // Rotating antenna patterns (adjusts sampling grid if needed, e.g. for parabolic antennas)
        // Usage: 0: Rotate both (pattern+polarization), 1: Rotate only pattern, 2: Rotate only polarization, 3: as (0), but w/o grid adjusting
        // Calling this function without the argument "output" updates the arrayant properties inplace
        void rotate_pattern(dtype x_deg = 0.0, dtype y_deg = 0.0, dtype z_deg = 0.0,
                            unsigned usage = 0, unsigned element = -1, arrayant<dtype> *output = nullptr);

        // Change the size of an arrayant, without explicitly preserving data
        // - "element_pos" is set to zero, "coupling_re/im" is set to the identity matrix
        // - Data in other properties may contain garbage
        // - Only performs a size update if exisiting size is different from new size
        // - Returns error when read-only
        void set_size(arma::uword n_elevation, arma::uword n_azimuth, arma::uword n_elements, arma::uword n_ports);

        // Validate integrity
        std::string is_valid(bool quick_check = true) const; // Returns an empty string if arrayant object is valid or an error message otherwise
        std::string validate();                              // Same, but sets the "valid" property in the objet and initializes the element positions and coupling matrix
    };

    // Read array antenna object and layout from QDANT file
    template <typename dtype>
    arrayant<dtype> qdant_read(std::string fn, unsigned id = 1, arma::Mat<unsigned> *layout = nullptr);

    // Generate : Isotropic radiator, vertical polarization, 1 deg resolution
    // Usage example: auto ant = quadriga_lib::generate_omni<float>();
    template <typename dtype>
    arrayant<dtype> generate_arrayant_omni(dtype res = 1.0);

    // Generate : Cross-polarized isotropic radiator, 1 deg resolution
    template <typename dtype>
    arrayant<dtype> generate_arrayant_xpol(dtype res = 1.0);

    // Generate : Short dipole radiating with vertical polarization, 1 deg resolution
    template <typename dtype>
    arrayant<dtype> generate_arrayant_dipole(dtype res = 1.0);

    // Generate : Half-wave dipole radiating with vertical polarization, 1 deg resolution
    template <typename dtype>
    arrayant<dtype> generate_arrayant_half_wave_dipole(dtype res = 1.0);

    // Generate : An antenna with a custom 3dB beam with (in degree)
    template <typename dtype>
    arrayant<dtype> generate_arrayant_custom(dtype az_3dB = 90.0, dtype el_3db = 90.0, dtype rear_gain_lin = 0.0, dtype res = 1.0);

    // Generate : Antenna model for the 3GPP-NR channel model
    // Polarization indicator:
    //   1. K=1, vertical polarization only
    //   2. K=1, H/V polarized elements
    //   3. K=1, +/-45 degree polarized elements
    //   4. K=M, vertical polarization only
    //   5. K=M, H/V polarized elements
    //   6. K=M, +/-45 degree polarized elements
    // Custom pattern: It is possible to provide a custom pattern, having 1 or more elements.
    // Values for coupling, element positions and center frequency of the custom pattern are ignored.
    template <typename dtype>
    arrayant<dtype> generate_arrayant_3GPP(arma::uword M = 1,                        // Number of vertical elements
                                           arma::uword N = 1,                        // Number of horizontal elements
                                           dtype center_freq = 299792458.0,          // The center frequency in [Hz]
                                           unsigned pol = 1,                         // Polarization indicator
                                           dtype tilt = 0.0,                         // The electric downtilt angle in [deg] for pol = 4,5,6
                                           dtype spacing = 0.5,                      // Element spacing in [λ]
                                           arma::uword Mg = 1,                       // Number of nested panels in a column (Mg)
                                           arma::uword Ng = 1,                       // Number of nested panels in a row (Ng)
                                           dtype dgv = 0.5,                          // Panel spacing in vertical direction (dg,V) in [λ]
                                           dtype dgh = 0.5,                          // Panel spacing in horizontal direction (dg,H) in [λ]
                                           const arrayant<dtype> *pattern = nullptr, // Optional custom per-element pattern
                                           dtype res = 1.0);                         // Resolution in degree, ignored if pattern is given

    // Calculate channel coefficients for spherical waves
    // - Interpolates the transmit antenna pattern (including orientation and polarization)
    // - Interpolates the receive antenna pattern (including orientation and polarization)
    // - Calculates the channel response and delays in time domain for each MIMO sub-link (including antenna element coupling)
    // - LOS-Path identification: path-length must equal the TX-RX 3D-distance within 0.1 mm (use double precision for satellite links!)
    // - For the LOS-Path, FBS and LBS must be on the direct line between Tx and Rx (i.e. FBS-Pos = TX-Pos, LBS-Pos = RX-Pos)
    // - LOS-Path delays and amplitudes are calculated including the individual element positions
    // - If path length is shorter than the shortest possible path (TX > FBS > LBS > RX), shortest possible path is used for delay calculation
    // - Polarization transfer matrix must be given by 8 interleaved complex values, (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)
    // - Polarization transfer matrix must be normalized (i.e., not include the path gain)
    template <typename dtype>
    void get_channels_spherical(const arrayant<dtype> *tx_array,     // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
                                const arrayant<dtype> *rx_array,     // Receive array antenna with 'n_rx' elements (= ports after element coupling)
                                dtype Tx, dtype Ty, dtype Tz,        // Transmitter position in Cartesian coordinates
                                dtype Tb, dtype Tt, dtype Th,        // Transmitter orientation (bank, tilt, head) in [rad]
                                dtype Rx, dtype Ry, dtype Rz,        // Receiver position in Cartesian coordinates
                                dtype Rb, dtype Rt, dtype Rh,        // Receiver orientation (bank, tilt, head) in [rad]
                                const arma::Mat<dtype> *fbs_pos,     // First-bounce scatterer positions, matrix of size [3, n_path]
                                const arma::Mat<dtype> *lbs_pos,     // Last-bounce scatterer positions, matrix of size [3, n_path]
                                const arma::Col<dtype> *path_gain,   // Path gain (linear scale), vector of length [n_path]
                                const arma::Col<dtype> *path_length, // Absolute path length from TX to RX phase center, vector of length [n_path]
                                const arma::Mat<dtype> *M,           // Polarization transfer matrix, matrix of size [8, n_path]
                                arma::Cube<dtype> *coeff_re,         // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *coeff_im,         // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *delay,            // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path]
                                dtype center_frequency = dtype(0.0), // Center frequency in [Hz]; a value of 0 disables phase calculation in coefficients
                                bool use_absolute_delays = false,    // Option: If true, the LOS delay is included for all paths
                                bool add_fake_los_path = false,      // Option: Add a zero-power LOS path in case where no LOS path was present
                                arma::Cube<dtype> *aod = nullptr,    // Optional output: Azimuth of Departure angles in [rad], Size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *eod = nullptr,    // Optional output: Elevation of Departure angles in [rad], Size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *aoa = nullptr,    // Optional output: Azimuth of Arrival angles in [rad], Size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *eoa = nullptr);   // Optional output: Elevation of Arrival angles in [rad], Size [n_rx, n_tx, n_path]

    // Calculate channels for Intelligent Reflective Surfaces (IRS)
    // - IRS is provided as a 3rd array antenna model comprised of 'n_irs' elements
    // - Requires 2 channel segments: (1) TX -> IRS and (2) IRS -> RX
    // - Assumes a low-rank channel where the IRS creates additional multipath components
    // - The IRS weights (phase shifts) are provided as Coupling matrix of the IRS array (column vectors)
    // - Multiple columns in the Coupling matrix can be seen as multiple IRS codebook entries. Only one entry can be selected
    // - Generates n_path_irs output paths where n_path_irs <= n_path_1 * n_path_2
    // - By default the 'irs_array' for both, the receive (TX-IRS) and receive (IRS-RX) direction.
    // - If different IRS patterns are required for the receive (IRS-RX) direction, a second IRS array can be supplied as optional input
    // - Returns a vector of bools indicating which of the n_path_1 * n_path_2 paths are in the output
    template <typename dtype>                                                           // Supported types: float or double
    std::vector<bool> get_channels_irs(const arrayant<dtype> *tx_array,                 // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
                                       const arrayant<dtype> *rx_array,                 // Receive array antenna with 'n_rx' elements (= ports after element coupling)
                                       const arrayant<dtype> *irs_array,                // IRS array with 'n_irs' elements (only one port can be selected for channel generation)
                                       dtype Tx, dtype Ty, dtype Tz,                    // Transmitter position in Cartesian coordinates
                                       dtype Tb, dtype Tt, dtype Th,                    // Transmitter orientation (bank, tilt, head) in [rad]
                                       dtype Rx, dtype Ry, dtype Rz,                    // Receiver position in Cartesian coordinates
                                       dtype Rb, dtype Rt, dtype Rh,                    // Receiver orientation (bank, tilt, head) in [rad]
                                       dtype Ix, dtype Iy, dtype Iz,                    // IRS position in Cartesian coordinates
                                       dtype Ib, dtype It, dtype Ih,                    // IRS orientation (bank, tilt, head) in [rad]
                                       const arma::Mat<dtype> *fbs_pos_1,               // First-bounce scatterer positions of TX-IRS paths, matrix of size [3, n_path_1]
                                       const arma::Mat<dtype> *lbs_pos_1,               // Last-bounce scatterer positions of TX-IRS paths, matrix of size [3, n_path_1]
                                       const arma::Col<dtype> *path_gain_1,             // Path gain (linear scale) of TX-IRS paths, vector of length [n_path_1]
                                       const arma::Col<dtype> *path_length_1,           // Absolute path length from TX to IRS phase center, vector of length [n_path_1]
                                       const arma::Mat<dtype> *M_1,                     // Polarization transfer matrix of TX-IRS paths, matrix of size [8, n_path_1]
                                       const arma::Mat<dtype> *fbs_pos_2,               // First-bounce scatterer positions of IRS-RX paths, matrix of size [3, n_path_2]
                                       const arma::Mat<dtype> *lbs_pos_2,               // Last-bounce scatterer positions of IRS-RX paths, matrix of size [3, n_path_2]
                                       const arma::Col<dtype> *path_gain_2,             // Path gain (linear scale) of IRS-RX paths, vector of length [n_path_2]
                                       const arma::Col<dtype> *path_length_2,           // Absolute path length from TX to IRS phase center, vector of length [n_path_2]
                                       const arma::Mat<dtype> *M_2,                     // Polarization transfer matrix of IRS-RX paths, matrix of size [8, n_path_2]
                                       arma::Cube<dtype> *coeff_re,                     // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path_irs]
                                       arma::Cube<dtype> *coeff_im,                     // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path_irs]
                                       arma::Cube<dtype> *delay,                        // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path_irs]
                                       arma::uword i_irs = 0,                           // IRS codebook entry (= port number in the IRS array)
                                       dtype threshold_dB = -140.0,                     // Threshold in dB relative to path gain below which paths are discarded from the output
                                       dtype center_frequency = 0.0,                    // Center frequency in [Hz]; a value of 0 disables phase calculation in coefficients
                                       bool use_absolute_delays = false,                // Option: If true, the LOS delay is included for all paths
                                       arma::Cube<dtype> *aod = nullptr,                // Optional output: Azimuth of Departure angles in [rad], Size [n_rx, n_tx, n_path_irs]
                                       arma::Cube<dtype> *eod = nullptr,                // Optional output: Elevation of Departure angles in [rad], Size [n_rx, n_tx, n_path_irs]
                                       arma::Cube<dtype> *aoa = nullptr,                // Optional output: Azimuth of Arrival angles in [rad], Size [n_rx, n_tx, n_path_irs]
                                       arma::Cube<dtype> *eoa = nullptr,                // Optional output: Elevation of Arrival angles in [rad], Size [n_rx, n_tx, n_path_irs]
                                       const arrayant<dtype> *irs_array_2 = nullptr,    // Optional input: Transmit IRS for the second segment with 'n_irs' elements
                                       const std::vector<bool> *active_path = nullptr); // Optional input: Switch to activate individual paths, ignores threshold_dB

    // Calculate channel coefficients for planar waves
    // - Interpolates the transmit antenna pattern (including orientation and polarization)
    // - Interpolates the receive antenna pattern (including orientation and polarization)
    // - Calculates the channel response and delays in time domain for each MIMO sub-link (including antenna element coupling)
    // - LOS-Path identification: path-length must equal the TX-RX 3D-distance within 0.1 mm
    // - Angles are ignored for LOS path identification
    // - Polarization transfer matrix must be given by 8 interleaved complex values, (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH)
    // - Polarization transfer matrix must be normalized (i.e., not include the path gain)
    // - Option to calculate the Doppler weights from orientation (+1 Moves towards path, -1 moves away from path)
    template <typename dtype>
    void get_channels_planar(const arrayant<dtype> *tx_array,         // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
                             const arrayant<dtype> *rx_array,         // Receive array antenna with 'n_rx' elements (= ports after element coupling)
                             dtype Tx, dtype Ty, dtype Tz,            // Transmitter position in Cartesian coordinates
                             dtype Tb, dtype Tt, dtype Th,            // Transmitter orientation (bank, tilt, head) in [rad]
                             dtype Rx, dtype Ry, dtype Rz,            // Receiver position in Cartesian coordinates
                             dtype Rb, dtype Rt, dtype Rh,            // Receiver orientation (bank, tilt, head) in [rad]
                             const arma::Col<dtype> *aod,             // Departure azimuth angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *eod,             // Departure elevation angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *aoa,             // Arrival azimuth angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *eoa,             // Arrival elevation angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *path_gain,       // Path gain (linear scale), vector of length [n_path]
                             const arma::Col<dtype> *path_length,     // Absolute path length from TX to RX phase center, vector of length [n_path]
                             const arma::Mat<dtype> *M,               // Polarization transfer matrix, matrix of size [8, n_path]
                             arma::Cube<dtype> *coeff_re,             // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path(+1)]
                             arma::Cube<dtype> *coeff_im,             // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path(+1)]
                             arma::Cube<dtype> *delay,                // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path(+1)]
                             dtype center_frequency = dtype(0.0),     // Center frequency in [Hz]; a value of 0.0 disables phase calculation in coefficients
                             bool use_absolute_delays = false,        // Option: If true, the LOS delay is included for all paths
                             bool add_fake_los_path = false,          // Option: Add a zero-power LOS path in case where no LOS path was present
                             arma::Col<dtype> *rx_Doppler = nullptr); // Optional output: Doppler weights for moving RX, vector of length 'n_path(+1)'

}

#endif
