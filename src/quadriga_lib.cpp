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

#include <stdexcept>
#include "quadriga_lib.hpp"
#include "qd_arrayant_qdant.hpp"

#define AUX(x) #x
#define STRINGIFY(x) AUX(x)

// Returns the arrayant_lib version number as a string
std::string quadriga_lib::quadriga_lib_version()
{
    std::string str = STRINGIFY(QUADRIGA_LIB_VERSION);
    std::size_t found = str.find_first_of("_");
    str.replace(found, 1, ".");
    found = str.find_first_of("_");
    str.replace(found, 1, ".");
    str = str.substr(1, str.length());
    return str;
}

// ARRAYANT Constructor : Read from file
template <typename dtype>
quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::arrayant(std::string fn, unsigned id, arma::Mat<unsigned> *layout)
{
    // Call private function to read the data from file
    std::string error_message = qd_arrayant_qdant_read(fn, id,
                                                       &name, &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                                       &azimuth_grid, &elevation_grid, &element_pos,
                                                       &coupling_re, &coupling_im, &center_frequency, layout);
    // Throw parsing errors
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Throw validation errors
    error_message = validate();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());
}

// ARRAYANT Constructor : Read from file
template <typename dtype>
quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::arrayant(std::string fn, unsigned id)
{
    arma::Mat<unsigned> layout;
    *this = quadriga_lib::arrayant<dtype>(fn, id, &layout);
    layout.reset();
}

// ARRAYANT : Write to QDANT file
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::qdant_write(std::string fn, unsigned id, arma::Mat<unsigned> layout)
{
    // Check if arrayant object is valid
    std::string error_message = "";
    if (valid != 0 || valid != 1)
        error_message = validate();
    else if (valid == 0)
        error_message = "Array antenna object is invalid";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    unsigned id_in_file = 0;
    error_message = qd_arrayant_qdant_write(fn, id, &name, &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                            &azimuth_grid, &elevation_grid, &element_pos,
                                            &coupling_re, &coupling_im, &center_frequency, &layout,
                                            &id_in_file);

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return id_in_file;
}

// ARRAYANT METHODS : Return number of elevation angles, azimuth angles and elemets
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_elevation()
{
    return unsigned(e_theta_re.n_rows);
}
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_azimuth()
{
    return unsigned(e_theta_re.n_cols);
}
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_elements()
{
    return unsigned(e_theta_re.n_slices);
}
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_ports()
{
    if (coupling_re.empty() && coupling_im.empty())
        return unsigned(e_theta_re.n_slices);
    else if (coupling_re.empty())
        return unsigned(coupling_im.n_cols);
    else
        return unsigned(coupling_re.n_cols);
}

// ARRAYANT METHOD : Validates correctness of the member functions
template <typename dtype>
std::string quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::validate()
{
    valid = 0;
    if (e_theta_re.n_elem == 0 || e_theta_im.n_elem == 0 || e_phi_re.n_elem == 0 || e_phi_im.n_elem == 0 || azimuth_grid.n_elem == 0 || elevation_grid.n_elem == 0)
        return "Missing data for any of: e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid";

    arma::uword n_elevation = e_theta_re.n_rows;
    arma::uword n_azimuth = e_theta_re.n_cols;
    arma::uword n_elements = e_theta_re.n_slices;

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        return "Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.";

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        return "Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.";

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        return "Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.";

    if (azimuth_grid.n_elem != n_azimuth)
        return "Number of elements in 'azimuth_grid' does not match number of columns in pattern data.";

    if (elevation_grid.n_elem != n_elevation)
        return "Number of elements in 'elevation_grid' does not match number of rows in pattern data.";

    int error_code = 0;
    auto fnc_az = [&error_code](dtype &val)
    {
        if (val < dtype(-3.1415930) || val > dtype(3.1415930))
            error_code = 5;
    };
    azimuth_grid.for_each(fnc_az);
    auto fnc_el = [&error_code](dtype &val)
    {
        if (val < dtype(-1.5707965) || val > dtype(1.5707965))
            error_code = 6;
    };
    elevation_grid.for_each(fnc_el);

    valid = error_code;
    if (error_code == 5)
        return "Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).";

    if (error_code == 6)
        return "Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).";

    if (!azimuth_grid.is_sorted())
        return "Values of 'azimuth_grid' must be sorted in ascending order.";

    if (!elevation_grid.is_sorted())
        return "Values of 'elevation_grid' must be sorted in ascending order.";

    if (!element_pos.empty() && (element_pos.n_rows != 3 || element_pos.n_cols != n_elements))
        return "Size of 'element_pos' must be either empty or match [3, n_elements]";

    if (!coupling_re.empty() && coupling_re.n_rows != n_elements)
        return "'Coupling' must be a matrix with rows equal to number of elements";

    if (coupling_re.empty() && !coupling_im.empty())
        return "Imaginary part of coupling matrix (phase component) defined without real part (absolute component)";

    if (!coupling_im.empty() && (coupling_im.n_rows != n_elements && coupling_im.n_cols != coupling_re.n_rows))
        return "'Coupling' must be a matrix with rows equal to number of elements and columns equal to number of ports";

    valid = 1;
    return "";
}

template class quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<float>;
template class quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<double>;
