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

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <iostream>
#include <string>
#include <any>

TEST_CASE("HDF - Minimal Test")
{
    // Test antenna
    double center_frequency = 21.0e6;
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.copy_element(0, 1);
    ant.copy_element(0, 2);
    ant.element_pos(1, 0) = -1.0;
    ant.element_pos(1, 1) = 0.0;
    ant.element_pos(1, 2) = 1.0;
    ant.e_theta_re.slice(1) *= 2;
    ant.e_theta_re.slice(1) *= 3;

    // Generate a new channel object
    auto c = quadriga_lib::channel<double>();
    c.name = "Very cool test channel object with a long name!!!";
    c.center_frequency = center_frequency;

    c.tx_pos = arma::mat(3, 1);
    c.tx_pos(2, 0) = 1.0;

    c.rx_pos = arma::mat(3, 2);
    c.rx_pos(0, 0) = 20.0;
    c.rx_pos(1, 0) = 0.0;
    c.rx_pos(2, 0) = 1.0;
    c.rx_pos(0, 1) = 20.0;
    c.rx_pos(1, 1) = 1.0;
    c.rx_pos(2, 1) = 1.0;

    c.tx_orientation = arma::mat(3, 1);
    c.rx_orientation = arma::mat(3, 1);

    c.path_gain.push_back(arma::vec(2));
    c.path_gain[0](0) = 1.0;
    c.path_gain[0](1) = 0.25;

    c.path_polarization.push_back(arma::mat(8, 2));
    c.path_polarization[0](0, 0) = 1.0;
    c.path_polarization[0](0, 1) = 1.0;
    c.path_polarization[0](6, 0) = -1.0;
    c.path_polarization[0](6, 1) = -1.0;

    c.path_coord.push_back(arma::cube(3, 4, 2, arma::fill::value(arma::datum::nan)));
    c.path_coord[0](0, 0, 1) = 0.0;
    c.path_coord[0](1, 0, 1) = 10.0;
    c.path_coord[0](2, 0, 1) = 11.0;

    arma::mat fbs_pos, lbs_pos;

    // Add empty elements for storing new coefficients
    c.path_length.push_back(arma::vec());
    c.path_angles.push_back(arma::mat());
    c.coeff_re.push_back(arma::cube());
    c.coeff_im.push_back(arma::cube());
    c.delay.push_back(arma::cube());

    unsigned s = 0; // Snapshot index
    quadriga_tools::coord2path(c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s),
                               &c.path_coord[s], &c.path_length[s], &fbs_pos, &lbs_pos, &c.path_angles[s]);

    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.tx_orientation(0), c.tx_orientation(1), c.tx_orientation(2),
                                                 c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s), c.rx_orientation(0), c.rx_orientation(1), c.rx_orientation(2),
                                                 &fbs_pos, &fbs_pos, &c.path_gain[s], &c.path_length[s], &c.path_polarization[s],
                                                 &c.coeff_re[s], &c.coeff_im[s], &c.delay[s], center_frequency, true, false);

    // Second snapshot
    c.path_gain.push_back(c.path_gain[0]);
    c.path_polarization.push_back(c.path_polarization[0]);
    c.path_coord.push_back(c.path_coord[0]);
    c.path_length.push_back(arma::vec());
    c.path_angles.push_back(arma::mat());
    c.coeff_re.push_back(arma::cube());
    c.coeff_im.push_back(arma::cube());
    c.delay.push_back(arma::cube());

    c.coeff_re[0].print();

    s = 1;
    quadriga_tools::coord2path(c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s),
                               &c.path_coord[s], &c.path_length[s], &fbs_pos, &lbs_pos, &c.path_angles[s]);

    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.tx_orientation(0), c.tx_orientation(1), c.tx_orientation(2),
                                                 c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s), c.rx_orientation(0), c.rx_orientation(1), c.rx_orientation(2),
                                                 &fbs_pos, &fbs_pos, &c.path_gain[s], &c.path_length[s], &c.path_polarization[s],
                                                 &c.coeff_re[s], &c.coeff_im[s], &c.delay[s], center_frequency, true, false);

    std::remove("test.hdf5");
    quadriga_lib::hdf5_create("test.hdf5", 10);
    c.hdf5_write("test.hdf5", 1);

    // Test all supported par types
    { // String
        auto t = std::string("Bla Bla Bla");
        c.par_names.push_back("string");
        c.par_data.push_back(t);
    }

    { // For unsigned
        auto t = unsigned(13);
        c.par_names.push_back("scalar_unsigned");
        c.par_data.push_back(t);
    }

    { // For int
        auto t = int(-13);
        c.par_names.push_back("scalar_int");
        c.par_data.push_back(t);
    }

    { // For long long (64 bit integer)
        auto t = (long long)10000000000LL;
        c.par_names.push_back("scalar_llong");
        c.par_data.push_back(t);
    }

    { // For unsigned long long (64 bit unsigned integer)
        auto t = (unsigned long long)10000000000ULL;
        c.par_names.push_back("scalar_ullong");
        c.par_data.push_back(t);
    }

    { // Float
        auto t = 3.14f;
        c.par_names.push_back("scalar_float");
        c.par_data.push_back(t);
    }

    { // Double
        auto t = 6.28;
        c.par_names.push_back("scalar_double");
        c.par_data.push_back(t);
    }

    { // For row, unsigned
        auto t = arma::Row<unsigned>(14, arma::fill::ones);
        c.par_names.push_back("row_unsigned");
        c.par_data.push_back(t);
    }

    { // For col, unsigned
        auto t = arma::Col<unsigned>(13, arma::fill::randn);
        c.par_names.push_back("col_unsigned");
        c.par_data.push_back(t);
    }

    { // For row, int
        auto t = arma::Row<int>(14, arma::fill::ones);
        c.par_names.push_back("row_int");
        c.par_data.push_back(t);
    }

    { // For col, int
        auto t = arma::Col<int>(13, arma::fill::randn);
        c.par_names.push_back("col_int");
        c.par_data.push_back(t);
    }

    { // For row, uword (64 bit unsigned integer)
        auto t = arma::urowvec(7, arma::fill::ones);
        c.par_names.push_back("row_uword");
        c.par_data.push_back(t);
    }

    { // For col, uword (64 bit unsigned integer)
        auto t = arma::ucolvec(8, arma::fill::randn);
        c.par_names.push_back("col_uword");
        c.par_data.push_back(t);
    }

    { // For row, sword (64 bit integer)
        auto t = arma::irowvec(7, arma::fill::ones);
        c.par_names.push_back("row_sword");
        c.par_data.push_back(t);
    }

    { // For col, sword (64 bit integer)
        auto t = arma::icolvec(8, arma::fill::randn);
        c.par_names.push_back("col_sword");
        c.par_data.push_back(t);
    }

    { // For row, float
        auto t = arma::frowvec(12, arma::fill::randn);
        c.par_names.push_back("row_float");
        c.par_data.push_back(t);
    }

    { // For col, float
        auto t = arma::Col<float>(13, arma::fill::randn);
        c.par_names.push_back("col_float");
        c.par_data.push_back(t);
    }

    { // For row, double
        auto t = arma::rowvec(10, arma::fill::randn);
        c.par_names.push_back("row_double");
        c.par_data.push_back(t);
    }

    { // For col, double
        auto t = arma::Col<double>(11, arma::fill::randn);
        c.par_names.push_back("col_double");
        c.par_data.push_back(t);
    }

    { // For Mat, unsigned
        auto t = arma::Mat<unsigned>(4, 3);
        for (unsigned i = 0; i < t.n_elem; i++)
            t.at(i) = i;
        c.par_names.push_back("mat_unsigned");
        c.par_data.push_back(t);
    }

    { // For Mat, int
        auto t = arma::Mat<int>(4, 3);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = -i;
        c.par_names.push_back("mat_int");
        c.par_data.push_back(t);
    }

    { // For Mat, uword (64 bit unsigned integer)
        auto t = arma::umat(5, 6);
        for (long long i = 0; i < t.n_elem; i++)
            t.at(i) = i;
        c.par_names.push_back("mat_uword");
        c.par_data.push_back(t);
    }

    { // For Mat, sword (64 bit integer)
        auto t = arma::imat(5, 6);
        for (long long i = 0; i < t.n_elem; i++)
            t.at(i) = -i;
        c.par_names.push_back("mat_sword");
        c.par_data.push_back(t);
    }

    { // For Mat, float
        auto t = arma::fmat(3, 6);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = float(i) + 0.1 * float(i);
        c.par_names.push_back("mat_float");
        c.par_data.push_back(t);
    }

    { // For Mat, double
        auto t = arma::mat(4, 6);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = double(i) + 1e-10 * double(i);
        c.par_names.push_back("mat_double");
        c.par_data.push_back(t);
    }

    c.hdf5_write("test.hdf5", 7);

    //    quadriga_lib::hdf5_write_unstructured("test.hdf5",c.par_names[0], &c.par_data[0]);
}

// TEST_CASE("HDF - Test Bench")
// {
//     // Get typenames of armadillo types, different for each compiler
//     std::any x;
//     x = arma::Mat<double>();
//     const auto type_arma_Mat_double = x.type().name();
//     x = arma::Mat<unsigned>();
//     const auto type_arma_Mat_unsigned = x.type().name();

//     //  std::cout << type_arma_Mat_double << ", " << type_arma_Mat_unsigned << std::endl;

//     // Create a vector of dynamic armadillo types
//     std::vector<std::any> v;
//     for (unsigned n = 0; n < 5; n++)
//     {
//         auto w = arma::Mat<double>(1, 5);
//         w.at(0) = n;
//         v.push_back(w);
//     }

//     auto w = arma::Mat<unsigned>(1, 3);
//     w.at(1) = 1999;
//     v.push_back(w);

//     // Return values, depending on their type
//     for (long unsigned i = 0; i < v.size(); i++)
//     {
//         if (v[i].type().name() == type_arma_Mat_double) // Returns reference
//         {
//             auto *w = std::any_cast<arma::Mat<double>>(&v[i]);
//             //  (*w).print();
//         }

//         // if (v[i].type().name() == type_arma_Mat_unsigned) // returns copy
//         //     std::any_cast<arma::Mat<unsigned>>(v[i]).print();
//     }

//     // std::cout << v.capacity() << std::endl;

//     // std::any w = arma::Mat<unsigned>(5, 5);

//     // std::any_cast<arma::Mat<unsigned>>(w).print();

//     // std::cout << w.type().name() << std::endl;

//     // auto c = quadriga_lib::channel<float>();
//     // std::cout << c.name << ", " << c.version << std::endl;
// }