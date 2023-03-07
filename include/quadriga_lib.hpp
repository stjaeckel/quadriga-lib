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

using namespace std;
using namespace arma;

#define QUADRIGA_LIB_VERSION v0_1_3

namespace quadriga_lib
{
    // Returns the quadriga_lib version number as a string in format (x.y.z)
    string quadriga_lib_version();

    inline namespace QUADRIGA_LIB_VERSION // Maintain ABI compatibility
    {
        template <typename dtype> // float or double
        class arrayant
        {
        public:
            string name = "empty";                                                // Name of the array antenna object
            Cube<dtype> e_theta_re;                                               // Horizontal component of the electric field, real part
            Cube<dtype> e_theta_im;                                               // Horizontal component of the electric field, imaginary part
            Cube<dtype> e_phi_re;                                                 // Vertical component of the electric field, real part
            Cube<dtype> e_phi_im;                                                 // Vertical component of the electric field, imaginary part
            Col<dtype> azimuth_grid;                                              // Azimuth angles in pattern (theta) in [rad]
            Col<dtype> elevation_grid;                                            // Elevation angles in pattern (phi) in [rad]
            Mat<dtype> element_pos;                                               // Element positions (optional)
            Mat<dtype> coupling_re;                                               // Coupling matrix, real part (optional)
            Mat<dtype> coupling_im;                                               // Coupling matrix, imaginary part (optional)
            dtype center_frequency = dtype(299792448.0);                          // Center frequency in [Hz] (optional)
            int valid = -1;                                                       // Indicator of data integrity (-1 = unknown, 0 = ERROR, 1 = OK)
            arrayant(){};                                                         // Default constructor
            arrayant(string qdant_file_name, unsigned id, Mat<unsigned> *layout); // Read array antenna object and layout from QDANT file
            arrayant(string qdant_file_name, unsigned id);                        // Read array antenna object from QDANT file
            unsigned n_elevation();                                               // Number of elevation angles
            unsigned n_azimuth();                                                 // Number of azimuth angles
            unsigned n_elements();                                                // Number of antenna elements
            unsigned n_ports();                                                   // Number of ports (after coupling of elements)
            string validate();                                                    // Validates integrity, returns error message and sets 'valid'
        };
    }
}

#endif
