// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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
#include <any>

#define QUADRIGA_LIB_VERSION v0_1_8

namespace quadriga_lib
{
    // Returns the version number as a string in format (x.y.z)
    std::string quadriga_lib_version();
    std::string get_HDF5_version();
    void print_lib_versions();

    inline namespace QUADRIGA_LIB_VERSION // Maintain ABI compatibility
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
            arrayant(){};                                // Default constructor

            // Functions to determine the size of the array antenna properties
            unsigned n_elevation() const; // Number of elevation angles
            unsigned n_azimuth() const;   // Number of azimuth angles
            unsigned n_elements() const;  // Number of antenna elements
            unsigned n_ports() const;     // Number of ports (after coupling of elements)

            // Calculate the directivity of an antenna element in dBi
            dtype calc_directivity_dBi(unsigned element) const;

            // Calculates a virtual pattern of the given array by applying coupling and element positions
            // Calling this function without an argument updates the arrayant properties inplace
            void combine_pattern(arrayant<dtype> *output = NULL);

            // Creates a copy of the array antenna object
            arrayant<dtype> copy() const;

            // Copy antenna elements, enlarges array size if needed (0-based indices)
            void copy_element(unsigned source, arma::Col<unsigned> destination);
            void copy_element(unsigned source, unsigned destination);

            // Interpolation of the antenna pattern
            void interpolate(const arma::Mat<dtype> azimuth,       // Azimuth angles [rad],                                  Size [1, n_ang] or [n_out, n_ang]
                             const arma::Mat<dtype> elevation,     // Elevation angles for interpolation in [rad],           Size [1, n_ang] or [n_out, n_ang]
                             const arma::Col<unsigned> i_element,  // Element indices, 1-based,                              Vector of length "n_out"
                             const arma::Cube<dtype> orientation,  // Orientation (bank, tilt, head) in [rad],               Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
                             const arma::Mat<dtype> element_pos_i, // Alternative element positions, optional,               Size [3, n_out] or Empty (= use element_pos from arrayant object)
                             arma::Mat<dtype> *V_re,               // Interpolated vertical (e_theta) field, real part,      Size [n_out, n_ang]
                             arma::Mat<dtype> *V_im,               // Interpolated vertical (e_theta) field, imaginary part, Size [n_out, n_ang]
                             arma::Mat<dtype> *H_re,               // Interpolated horizontal (e_phi) field, real part,      Size [n_out, n_ang]
                             arma::Mat<dtype> *H_im,               // Interpolated horizontal (e_phi) field, imaginary part, Size [n_out, n_ang]
                             arma::Mat<dtype> *dist,               // Projected element distances for phase offset,          Size [n_out, n_ang]
                             arma::Mat<dtype> *azimuth_loc,        // Azimuth angles [rad] in local antenna coordinates,     Size [n_out, n_ang]
                             arma::Mat<dtype> *elevation_loc,      // Elevation angles [rad] in local antenna coordinates,   Size [n_out, n_ang]
                             arma::Mat<dtype> *gamma) const;       // Polarization rotation angles in [rad],                 Size [n_out, n_ang]

            void interpolate(const arma::Mat<dtype> azimuth,                 // Azimuth angles [rad],                        Size [1, n_ang] or [n_out, n_ang]
                             const arma::Mat<dtype> elevation,               // Elevation angles for interpolation in [rad], Size [1, n_ang] or [n_out, n_ang]
                             const arma::Cube<dtype> orientation,            // Orientation (bank, tilt, head) in [rad],     Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
                             arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im, // Interpolated vertical (e_theta) field,       Size [n_out, n_ang]
                             arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im, // Interpolated horizontal (e_phi) field,       Size [n_out, n_ang]
                             arma::Mat<dtype> *dist) const;                  // Projected element distances,                 Size [n_out, n_ang]

            // Write array antenna object and layout to QDANT file, returns id in file
            unsigned qdant_write(std::string fn, unsigned id = 0, arma::Mat<unsigned> layout = arma::Mat<unsigned>()) const;

            // Remove zeros from the pattern data. Changes that size of the pattern.
            // Calling this function without an argument updates the arrayant properties inplace
            void remove_zeros(arrayant<dtype> *output = NULL);

            // Reset the size to zero (the arrayant object will contain no data)
            void reset();

            // Rotating antenna patterns (adjusts sampling grid if needed, e.g. for parabolic antennas)
            // Usage: 0: Rotate both (pattern+polarization), 1: Rotate only pattern, 2: Rotate only polarization, 3: as (0), but w/o grid adjusting
            // Calling this function without the argument "output" updates the arrayant properties inplace
            void rotate_pattern(dtype x_deg = 0.0, dtype y_deg = 0.0, dtype z_deg = 0.0,
                                unsigned usage = 0, unsigned element = -1, arrayant<dtype> *output = NULL);

            // Change the size of an arrayant, without explicitly preserving data
            // - "element_pos" is set to zero, "coupling_re/im" is set to the identity matrix
            // - Data in other properties may contain garbage
            // - Only performs a size update if exisiting size is different from new size
            // - Returns error when read-only
            void set_size(unsigned n_elevation, unsigned n_azimuth, unsigned n_elements, unsigned n_ports);

            // Validate integrity
            std::string is_valid(bool quick_check = true) const; // Returns an empty string if arrayant object is valid or an error message otherwise
            std::string validate();                              // Same, but sets the "valid" property in the objet and initializes the element positions and coupling matrix
        };

        // Class for storing and managing channel data (+ metadata)
        // Note: "n_path" is different for each snapshot
        template <typename dtype> // float or double
        class channel
        {
        public:
            std::string name = "empty";                      // Name of the channel object
            dtype center_frequency = dtype(299792458.0);     // Center frequency in [Hz]
            arma::Mat<dtype> tx_pos;                         // Transmitter positions, matrix of size [3, n_snap] or [3, 1]
            arma::Mat<dtype> rx_pos;                         // Receiver positions, matrix of size [3, n_snap]
            arma::Mat<dtype> tx_orientation;                 // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
            arma::Mat<dtype> rx_orientation;                 // Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []
            std::vector<arma::Cube<dtype>> coeff_re;         // Channel coefficients, real part, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
            std::vector<arma::Cube<dtype>> coeff_im;         // Channel coefficients, imaginary part, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
            std::vector<arma::Cube<dtype>> delay;            // Path delays in seconds, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
            std::vector<arma::Col<dtype>> path_gain;         // Path gain before antenna patterns, vector (n_snap) of vectors of length [n_path]
            std::vector<arma::Col<dtype>> path_length;       // Absolute path length from TX to RX phase center, vector (n_snap) of vectors of length [n_path]
            std::vector<arma::Mat<dtype>> path_polarization; // Polarization transfer function, vector (n_snap) of matrices of size [8, n_path], interleaved complex
            std::vector<arma::Mat<dtype>> path_angles;       // Departure and arrival angles, vector (n_snap) of matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
            std::vector<arma::Cube<dtype>> path_coord;       // Interaction coordinates, NAN-padded, vector (n_snap) of tensors of size [3, n_coord, n_path]
            std::vector<std::string> par_names;              // Names of unstructured data fields
            std::vector<std::any> par_data;                  // Unstructured data of types {string, float, double, int, long int, arma::Col<dtype>, arma::Mat<dtype>, arma::Cube<dtype>}
            int initial_position = 0;                        // Index of reference position, values between 0 and n_snap-1 (mainly used internally)
            channel(){};                                     // Default constructor

            arma::uword n_snap() const; // Number of snapshots
            arma::uword n_rx() const;   // Number of receive antennas in coefficient matrix, returns 0 if there are no coefficients
            arma::uword n_tx() const;   // Number of transmit antennas in coefficient matrix, returns 0 if there are no coefficients
            arma::uvec n_path() const;  // Number of paths per snapshot (arma::uword)

            // Validate integrity
            std::string is_valid() const; // Returns an empty string if channel object is valid or an error message otherwise

            // Save data to HDF file
            // - All data is stored in single precision
            // - Data is added to the end of the file
            // - The index is updated to include to the newly written data
            // - If file does not exist, a new file will be created with (nx = 65535, ny = 1, nz = 1, nw = 1)
            void hdf5_write(std::string fn, unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0) const;
        };
    }

    // Read array antenna object and layout from QDANT file
    template <typename dtype>
    arrayant<dtype> qdant_read(std::string fn, unsigned id = 1, arma::Mat<unsigned> *layout = NULL);

    // Generate : Isotropic radiator, vertical polarization, 1 deg resolution
    // Usage example: auto ant = quadriga_lib::generate_omni<float>();
    template <typename dtype>
    arrayant<dtype> generate_arrayant_omni();

    // Generate : Cross-polarized isotropic radiator, 1 deg resolution
    template <typename dtype>
    arrayant<dtype> generate_arrayant_xpol();

    // Generate : Short dipole radiating with vertical polarization, 1 deg resolution
    template <typename dtype>
    arrayant<dtype> generate_arrayant_dipole();

    // Generate : Half-wave dipole radiating with vertical polarization, 1 deg resolution
    template <typename dtype>
    arrayant<dtype> generate_arrayant_half_wave_dipole();

    // Generate : An antenna with a custom 3dB beam with (in degree)
    template <typename dtype>
    arrayant<dtype> generate_arrayant_custom(dtype az_3dB = 90.0, dtype el_3db = 90.0, dtype rear_gain_lin = 0.0);

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
    arrayant<dtype> generate_arrayant_3GPP(unsigned M = 1,                         // Number of vertical elements
                                           unsigned N = 1,                         // Number of horizontal elements
                                           dtype center_freq = 299792458.0,        // The center frequency in [Hz]
                                           unsigned pol = 1,                       // Polarization indicator
                                           dtype tilt = 0.0,                       // The electric downtilt angle in [deg] for pol = 4,5,6
                                           dtype spacing = 0.5,                    // Element spacing in [λ]
                                           unsigned Mg = 1,                        // Number of nested panels in a column (Mg)
                                           unsigned Ng = 1,                        // Number of nested panels in a row (Ng)
                                           dtype dgv = 0.5,                        // Panel spacing in vertical direction (dg,V) in [λ]
                                           dtype dgh = 0.5,                        // Panel spacing in horizontal direction (dg,H) in [λ]
                                           const arrayant<dtype> *pattern = NULL); // Optional custom per-element pattern

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
                                arma::Cube<dtype> *aod = NULL,       // Optional output: Azimuth of Departure angles in [rad], Size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *eod = NULL,       // Optional output: Elevation of Departure angles in [rad], Size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *aoa = NULL,       // Optional output: Azimuth of Arrival angles in [rad], Size [n_rx, n_tx, n_path]
                                arma::Cube<dtype> *eoa = NULL);      // Optional output: Elevation of Arrival angles in [rad], Size [n_rx, n_tx, n_path]

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
    void get_channels_planar(const arrayant<dtype> *tx_array,      // Transmit array antenna with 'n_tx' elements (= ports after element coupling)
                             const arrayant<dtype> *rx_array,      // Receive array antenna with 'n_rx' elements (= ports after element coupling)
                             dtype Tx, dtype Ty, dtype Tz,         // Transmitter position in Cartesian coordinates
                             dtype Tb, dtype Tt, dtype Th,         // Transmitter orientation (bank, tilt, head) in [rad]
                             dtype Rx, dtype Ry, dtype Rz,         // Receiver position in Cartesian coordinates
                             dtype Rb, dtype Rt, dtype Rh,         // Receiver orientation (bank, tilt, head) in [rad]
                             const arma::Col<dtype> *aod,          // Departure azimuth angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *eod,          // Departure elevation angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *aoa,          // Arrival azimuth angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *eoa,          // Arrival elevation angles in [rad], vector of length 'n_path'
                             const arma::Col<dtype> *path_gain,    // Path gain (linear scale), vector of length [n_path]
                             const arma::Col<dtype> *path_length,  // Absolute path length from TX to RX phase center, vector of length [n_path]
                             const arma::Mat<dtype> *M,            // Polarization transfer matrix, matrix of size [8, n_path]
                             arma::Cube<dtype> *coeff_re,          // Output: Channel coefficients, real part, tensor of size [n_rx, n_tx, n_path(+1)]
                             arma::Cube<dtype> *coeff_im,          // Output: Channel coefficients, imaginary part, tensor of size [n_rx, n_tx, n_path(+1)]
                             arma::Cube<dtype> *delay,             // Output: Propagation delay in seconds, tensor of size [n_rx, n_tx, n_path(+1)]
                             dtype center_frequency = dtype(0.0),  // Center frequency in [Hz]; a value of 0.0 disables phase calculation in coefficients
                             bool use_absolute_delays = false,     // Option: If true, the LOS delay is included for all paths
                             bool add_fake_los_path = false,       // Option: Add a zero-power LOS path in case where no LOS path was present
                             arma::Col<dtype> *rx_Doppler = NULL); // Optional output: Doppler weights for moving RX, vector of length 'n_path(+1)'

    // Create a new channel HDF file and set the index to to given storage layout
    void hdf5_create(std::string fn, unsigned nx = 65535, unsigned ny = 1, unsigned nz = 1, unsigned nw = 1);

    // Write unstructured data to hdf5 file
    // - Scalar types: string, unsigned, int, long long, unsigned long long, float, double
    // - Armadillo Vectors, Matrices and Cubes with types: unsigned, int, uword, sword, float, double
    // - Vector types: Armadillo Row and Col types are mapped to 1D storage type - reading will be arma::Col
    // - Parameter name may only contain letters and numbers and the underscore "_"
    // - Unsupported data types will cause an error
    void hdf5_write_unstructured(std::string fn, std::string par_name, const std::any *par_data,
                                 unsigned ix = 0, unsigned iy = 0, unsigned iz = 0, unsigned iw = 0);
}

#endif
