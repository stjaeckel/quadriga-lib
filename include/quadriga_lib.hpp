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

#define QUADRIGA_LIB_VERSION v0_1_6

namespace quadriga_lib
{
    // Returns the quadriga_lib version number as a string in format (x.y.z)
    std::string quadriga_lib_version();

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
            int valid = -1;                              // Indicator of data integrity (-1 = unknown, 0 = ERROR, 1 = OK)
            bool read_only = false;                      // Prevent member functions from writing to the properties
            arrayant(){};                                // Default constructor

            // Functions to determine the size of the array antenna properties
            unsigned n_elevation(); // Number of elevation angles
            unsigned n_azimuth();   // Number of azimuth angles
            unsigned n_elements();  // Number of antenna elements
            unsigned n_ports();     // Number of ports (after coupling of elements)

            // Write array antenna object and layout to QDANT file, returns id in file
            unsigned qdant_write(std::string fn, unsigned id, arma::Mat<unsigned> layout);
            unsigned qdant_write(std::string fn, unsigned id);
            unsigned qdant_write(std::string fn);

            // Change the size of an arrayant, without explicitly preserving data
            // - "element_pos" is set to zero, "coupling_re/im" is set to the identity matrix
            // - Only performs a size update if exisiting size is different from new size
            // - Returns error when read-only
            void set_size(unsigned n_elevation, unsigned n_azimuth, unsigned n_elements, unsigned n_ports);

            // Reset the size to zero (the arrayant object will contain no data)
            void reset();

            // Creates a copy of the array antenna object
            arrayant<dtype> copy() const;

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
                             arma::Mat<dtype> *gamma);             // Polarization rotation angles in [rad],                 Size [n_out, n_ang]

            void interpolate(const arma::Mat<dtype> azimuth,                 // Azimuth angles [rad],                        Size [1, n_ang] or [n_out, n_ang]
                             const arma::Mat<dtype> elevation,               // Elevation angles for interpolation in [rad], Size [1, n_ang] or [n_out, n_ang]
                             const arma::Cube<dtype> orientation,            // Orientation (bank, tilt, head) in [rad],     Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
                             arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im, // Interpolated vertical (e_theta) field,       Size [n_out, n_ang]
                             arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im, // Interpolated horizontal (e_phi) field,       Size [n_out, n_ang]
                             arma::Mat<dtype> *dist);                        // Projected element distances,                 Size [n_out, n_ang]

            // Copy antenna elements, enlarges array size if needed (0-based indices)
            void copy_element(unsigned source, arma::Col<unsigned> destination);
            void copy_element(unsigned source, unsigned destination);

            // Calculates a virtual pattern of the given array by applying coupling and element positions
            // Calling this function without an argument updates the arrayant properties inplace
            void combine_pattern(arrayant<dtype> *output = NULL);

            // Rotating antenna patterns (adjusts sampling grid if needed, e.g. for parabolic antennas)
            // Usage: 0: Rotate both (pattern+polarization), 1: Rotate only pattern, 2: Rotate only polarization, 3: as (0), but w/o grid adjusting
            // Calling this function without the argument "output" updates the arrayant properties inplace
            void rotate_pattern(dtype x_deg = 0.0, dtype y_deg = 0.0, dtype z_deg = 0.0,
                                unsigned usage = 0, unsigned element = -1, arrayant<dtype> *output = NULL);

            // Remove zeros from the pattern
            // Calling this function without an argument updates the arrayant properties inplace
            void remove_zeros(arrayant<dtype> *output = NULL);

            // Calculate the directivity of an antenna element in dBi
            dtype calc_directivity_dBi(unsigned element);

            // Validates integrity, returns error message and sets 'valid' property accordingly
            std::string validate();
        };
    }

    // Read array antenna object and layout from QDANT file
    template <typename dtype>
    arrayant<dtype> qdant_read(std::string fn, unsigned id = 1, arma::Mat<unsigned> *layout = NULL);

    // Generate : Isotropic radiator, vertical polarization, 1 deg resolution
    // Usage example: auto ant = quadriga_lib::generate_omni<float>();
    template <typename dtype>
    arrayant<dtype> generate_arrayant_omni();

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

}

#endif
