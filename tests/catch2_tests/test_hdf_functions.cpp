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
#include <iostream>

#include "quadriga_lib.hpp"

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
    CHECK(c.empty());

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

    // The second path has 1 interaction with the environment
    c.no_interact.push_back(arma::u32_vec(2));
    c.no_interact[0](1) = 1; // 2nd path

    // List of interactions = length 1
    c.interact_coord.push_back(arma::mat(3, 1));
    c.interact_coord[0](0, 0) = 0.0;
    c.interact_coord[0](1, 0) = 10.0;
    c.interact_coord[0](2, 0) = 11.0;

    arma::mat fbs_pos, lbs_pos;

    // Add empty elements for storing new coefficients
    c.path_length.push_back(arma::vec());
    c.path_angles.push_back(arma::mat());
    c.coeff_re.push_back(arma::cube());
    c.coeff_im.push_back(arma::cube());
    c.delay.push_back(arma::cube());

    unsigned s = 0; // Snapshot index
    quadriga_lib::coord2path(c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s),
                               &c.no_interact[s], &c.interact_coord[s], &c.path_length[s], &fbs_pos, &lbs_pos, &c.path_angles[s]);

    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.tx_orientation(0), c.tx_orientation(1), c.tx_orientation(2),
                                                 c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s), c.rx_orientation(0), c.rx_orientation(1), c.rx_orientation(2),
                                                 &fbs_pos, &fbs_pos, &c.path_gain[s], &c.path_length[s], &c.path_polarization[s],
                                                 &c.coeff_re[s], &c.coeff_im[s], &c.delay[s], center_frequency, true, false);

    // Second snapshot
    c.path_gain.push_back(c.path_gain[0]);
    c.path_polarization.push_back(c.path_polarization[0]);
    c.no_interact.push_back(c.no_interact[0]);
    c.interact_coord.push_back(c.interact_coord[0]);
    c.path_length.push_back(arma::vec());
    c.path_angles.push_back(arma::mat());
    c.coeff_re.push_back(arma::cube());
    c.coeff_im.push_back(arma::cube());
    c.delay.push_back(arma::cube());

    s = 1;
    quadriga_lib::coord2path(c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s),
                               &c.no_interact[s], &c.interact_coord[s], &c.path_length[s], &fbs_pos, &lbs_pos, &c.path_angles[s]);

    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 c.tx_pos(0), c.tx_pos(1), c.tx_pos(2), c.tx_orientation(0), c.tx_orientation(1), c.tx_orientation(2),
                                                 c.rx_pos(0, s), c.rx_pos(1, s), c.rx_pos(2, s), c.rx_orientation(0), c.rx_orientation(1), c.rx_orientation(2),
                                                 &fbs_pos, &fbs_pos, &c.path_gain[s], &c.path_length[s], &c.path_polarization[s],
                                                 &c.coeff_re[s], &c.coeff_im[s], &c.delay[s], center_frequency, true, false);

    std::remove("test.hdf5");
    quadriga_lib::hdf5_create("test.hdf5", 10);
    c.hdf5_write("test.hdf5", 1);

    auto ChanelDims = quadriga_lib::hdf5_read_layout("test.hdf5");
    CHECK(ChanelDims.n_elem == 4);
    CHECK(ChanelDims.at(0) == 10);
    CHECK(ChanelDims.at(1) == 1);
    CHECK(ChanelDims.at(2) == 1);
    CHECK(ChanelDims.at(3) == 1);

    // Test all supported par types
    { // String - 0
        auto t = std::string("Bla Bla Bla");
        c.par_names.push_back("string");
        c.par_data.push_back(t);
    }

    { // Float - 1
        auto t = 3.14f;
        c.par_names.push_back("scalar_float");
        c.par_data.push_back(t);
    }

    { // Double - 2
        auto t = 6.28;
        c.par_names.push_back("scalar_double");
        c.par_data.push_back(t);
    }

    { // For unsigned long long (64 bit unsigned integer) - 3
        auto t = (unsigned long long)12341100000ULL;
        c.par_names.push_back("scalar_ullong");
        c.par_data.push_back(t);
    }

    { // For long long (64 bit integer) - 4
        auto t = (long long)-98760000000LL;
        c.par_names.push_back("scalar_llong");
        c.par_data.push_back(t);
    }

    { // For unsigned - 5
        auto t = unsigned(13);
        c.par_names.push_back("scalar_unsigned");
        c.par_data.push_back(t);
    }

    { // For int - 6
        auto t = int(-13);
        c.par_names.push_back("scalar_int");
        c.par_data.push_back(t);
    }

    { // For Mat, float - 7
        auto t = arma::fmat(3, 6);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = float(i) + 0.1 * float(i);
        c.par_names.push_back("mat_float");
        c.par_data.push_back(t);
    }

    { // For Mat, double - 8
        auto t = arma::mat(4, 6);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = double(i) + 1e-10 * double(i);
        c.par_names.push_back("mat_double");
        c.par_data.push_back(t);
    }

    { // For Mat, uword (64 bit unsigned integer) - 9
        auto t = arma::umat(5, 6);
        for (long long i = 0; i < t.n_elem; i++)
            t.at(i) = i;
        c.par_names.push_back("mat_uword");
        c.par_data.push_back(t);
    }

    { // For Mat, sword (64 bit integer) - 10
        auto t = arma::imat(5, 6);
        for (long long i = 0; i < t.n_elem; i++)
            t.at(i) = -i;
        c.par_names.push_back("mat_sword");
        c.par_data.push_back(t);
    }

    { // For Mat, unsigned - 11
        auto t = arma::Mat<unsigned>(4, 3);
        for (unsigned i = 0; i < t.n_elem; i++)
            t.at(i) = i;
        c.par_names.push_back("mat_unsigned");
        c.par_data.push_back(t);
    }

    { // For Mat, int - 12
        auto t = arma::Mat<int>(4, 3);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = -i;
        c.par_names.push_back("mat_int");
        c.par_data.push_back(t);
    }

    { // For Cube, float - 13
        auto t = arma::fcube(3, 6, 2);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = float(i) + 0.1 * float(i);
        c.par_names.push_back("cube_float");
        c.par_data.push_back(t);
    }

    { // For Cube, double - 14
        auto t = arma::cube(4, 6, 3);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = double(i) + 1e-10 * double(i);
        c.par_names.push_back("cube_double");
        c.par_data.push_back(t);
    }

    { // For Cube, uword (64 bit unsigned integer) - 15
        auto t = arma::ucube(5, 6, 2);
        for (long long i = 0; i < t.n_elem; i++)
            t.at(i) = i;
        c.par_names.push_back("cube_uword");
        c.par_data.push_back(t);
    }

    { // For Cube, sword (64 bit integer) - 16
        auto t = arma::icube(5, 6, 3);
        for (long long i = 0; i < t.n_elem; i++)
            t.at(i) = -i;
        c.par_names.push_back("cube_sword");
        c.par_data.push_back(t);
    }

    { // For Cube, unsigned - 17
        auto t = arma::Cube<unsigned>(4, 3, 2);
        for (unsigned i = 0; i < t.n_elem; i++)
            t.at(i) = i;
        c.par_names.push_back("cube_unsigned");
        c.par_data.push_back(t);
    }

    { // For Cube, int - 18
        auto t = arma::Cube<int>(4, 3, 5);
        for (int i = 0; i < t.n_elem; i++)
            t.at(i) = -i;
        c.par_names.push_back("cube_int");
        c.par_data.push_back(t);
    }

    { // For col, float - 19
        auto t = arma::Col<float>(13, arma::fill::randn);
        c.par_names.push_back("col_float");
        c.par_data.push_back(t);
    }

    { // For col, double - 20
        auto t = arma::Col<double>(11, arma::fill::randn);
        c.par_names.push_back("col_double");
        c.par_data.push_back(t);
    }

    { // For col, uword (64 bit unsigned integer) - 21
        auto t = arma::ucolvec(8);
        for (auto i = 0ULL; i < t.n_elem; ++i)
            t.at(i) = 221ULL * i;
        c.par_names.push_back("col_uword");
        c.par_data.push_back(t);
    }

    { // For col, sword (64 bit integer) - 22
        auto t = arma::icolvec(18);
        for (auto i = 0ULL; i < t.n_elem; ++i)
            t.at(i) = -13LL * (long long)i;
        c.par_names.push_back("col_sword");
        c.par_data.push_back(t);
    }

    { // For col, unsigned - 23
        auto t = arma::Col<unsigned>(13);
        for (auto i = 0ULL; i < t.n_elem; ++i)
            t.at(i) = 11 * (unsigned)i;
        c.par_names.push_back("col_unsigned");
        c.par_data.push_back(t);
    }

    { // For col, int - 24
        auto t = arma::Col<int>(13);
        for (auto i = 0ULL; i < t.n_elem; ++i)
            t.at(i) = -13 * (int)i;
        c.par_names.push_back("col_int");
        c.par_data.push_back(t);
    }

    { // For row, float - 25
        auto t = arma::frowvec(12, arma::fill::randn);
        c.par_names.push_back("row_float");
        c.par_data.push_back(t);
    }

    { // For row, double - 26
        auto t = arma::rowvec(10, arma::fill::randn);
        c.par_names.push_back("row_double");
        c.par_data.push_back(t);
    }

    { // For row, uword (64 bit unsigned integer) - 27
        auto t = arma::urowvec(7);
        for (auto i = 0ULL; i < t.n_elem; i++)
            t.at(i) = 111ULL * i;
        c.par_names.push_back("row_uword");
        c.par_data.push_back(t);
    }

    { // For row, sword (64 bit integer) - 28
        auto t = arma::irowvec(8);
        for (auto i = 0ULL; i < t.n_elem; i++)
            t.at(i) = -17LL * (long long)i;
        c.par_names.push_back("row_sword");
        c.par_data.push_back(t);
    }

    { // For row, unsigned - 29
        auto t = arma::Row<unsigned>(14);
        for (auto i = 0ULL; i < t.n_elem; i++)
            t.at(i) = 3 * (unsigned)i;
        c.par_names.push_back("row_unsigned");
        c.par_data.push_back(t);
    }

    { // For row, int - 30
        auto t = arma::Row<int>(12);
        for (auto i = 0ULL; i < t.n_elem; i++)
            t.at(i) = -4 * (int)i;
        c.par_names.push_back("row_int");
        c.par_data.push_back(t);
    }

    c.hdf5_write("test.hdf5", 7);

    // Read names of the unstructured data fields
    std::vector<std::string> names;
    quadriga_lib::hdf5_read_dset_names("test.hdf5", &names, 7);

    // Check if all fields are included in the HDF file and that the correct names are returned
    for (auto &hdf_name : names)
    {
        bool included = false;
        for (auto &par_name : c.par_names)
            included = (par_name == hdf_name) ? true : included;
        CHECK(included);
    }

    // Write all unstructured data fields individually to slot [0]
    for (auto i = 0ULL; i < c.par_names.size(); i++)
        quadriga_lib::hdf5_write_dset("test.hdf5", c.par_names[i], &c.par_data[i]);

    // Read channelIDs
    arma::Col<unsigned> channelID;
    ChanelDims = quadriga_lib::hdf5_read_layout("test.hdf5", &channelID);
    CHECK(channelID.at(0) == 3);
    CHECK(channelID.at(1) == 1);
    CHECK(channelID.at(7) == 2);
    CHECK(channelID.at(5) == 0);

    // Read names of the unstructured data fields
    std::vector<std::string> names2;
    quadriga_lib::hdf5_read_dset_names("test.hdf5", &names2, 0);

    // Check if all fields are included in the HDF file and that the correct names are returned
    for (auto &hdf_name : names2)
    {
        bool included = false;
        for (auto &par_name : c.par_names)
            included = (par_name == hdf_name) ? true : included;
        CHECK(included);
    }

    // Read string
    std::any value;
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "string");
    CHECK(quadriga_lib::any_type_id(&value) == 9);
    CHECK(std::any_cast<std::string>(value) == "Bla Bla Bla");

    // Read float
    void *dataptr;
    unsigned long long dims[3];
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "scalar_float", 7);
    CHECK(quadriga_lib::any_type_id(&value, dims, &dataptr) == 10);
    CHECK(*(float *)dataptr == 3.14f);

    // Read double
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "scalar_double");
    CHECK(quadriga_lib::any_type_id(&value, dims, &dataptr) == 11);
    CHECK(std::any_cast<double>(value) == 6.28);

    // Read ULL
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "scalar_ullong");
    CHECK(quadriga_lib::any_type_id(&value, dims, &dataptr) == 12);
    CHECK(*(unsigned long long int *)dataptr == 12341100000ULL);

    // Read LL
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "scalar_llong");
    CHECK(quadriga_lib::any_type_id(&value, dims, &dataptr) == 13);
    CHECK(*(long long int *)dataptr == -98760000000LL);

    // Read unsigned
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "scalar_unsigned", 7);
    CHECK(quadriga_lib::any_type_id(&value, dims, &dataptr) == 14);
    CHECK(*(unsigned *)dataptr == 13);

    // Read int
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "scalar_int");
    CHECK(quadriga_lib::any_type_id(&value, dims, &dataptr) == 15);
    CHECK(*(int *)dataptr == -13);

    // Mat, float
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "mat_float");
    CHECK(quadriga_lib::any_type_id(&value) == 20);
    CHECK(arma::approx_equal(std::any_cast<arma::fmat>(value), std::any_cast<arma::fmat>(c.par_data[7]), "absdiff", 1e-14));

    // Mat, double
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "mat_double");
    CHECK(quadriga_lib::any_type_id(&value) == 21);
    CHECK(arma::approx_equal(std::any_cast<arma::mat>(value), std::any_cast<arma::mat>(c.par_data[8]), "absdiff", 1e-14));

    // Mat, uword
    arma::umat cmp;
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "mat_uword");
    CHECK(quadriga_lib::any_type_id(&value) == 22);
    cmp = std::any_cast<arma::umat>(value) == std::any_cast<arma::umat>(c.par_data[9]);
    CHECK(arma::all(arma::vectorise(cmp)));

    // Mat, sword
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "mat_sword");
    CHECK(quadriga_lib::any_type_id(&value) == 23);
    cmp = std::any_cast<arma::imat>(value) == std::any_cast<arma::imat>(c.par_data[10]);
    CHECK(arma::all(arma::vectorise(cmp)));

    // Mat, unsigned
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "mat_unsigned");
    CHECK(quadriga_lib::any_type_id(&value) == 24);
    cmp = std::any_cast<arma::u32_mat>(value) == std::any_cast<arma::Mat<unsigned>>(c.par_data[11]);
    CHECK(arma::all(arma::vectorise(cmp)));

    // Mat, int
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "mat_int");
    CHECK(quadriga_lib::any_type_id(&value) == 25);
    cmp = std::any_cast<arma::s32_mat>(value) == std::any_cast<arma::Mat<int>>(c.par_data[12]);
    CHECK(arma::all(arma::vectorise(cmp)));

    // Cube, float
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "cube_float");
    CHECK(quadriga_lib::any_type_id(&value) == 30);
    CHECK(arma::approx_equal(std::any_cast<arma::fcube>(value), std::any_cast<arma::fcube>(c.par_data[13]), "absdiff", 1e-14));

    // Cube, double
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "cube_double", 7);
    CHECK(quadriga_lib::any_type_id(&value) == 31);
    CHECK(arma::approx_equal(std::any_cast<arma::cube>(value), std::any_cast<arma::cube>(c.par_data[14]), "absdiff", 1e-14));

    //  Cube, uword
    arma::ucube cmp_cube;
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "cube_uword");
    CHECK(quadriga_lib::any_type_id(&value) == 32);
    cmp_cube = std::any_cast<arma::ucube>(value) == std::any_cast<arma::ucube>(c.par_data[15]);
    CHECK(arma::all(arma::vectorise(cmp_cube)));

    // Cube, sword
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "cube_sword");
    CHECK(quadriga_lib::any_type_id(&value) == 33);
    cmp_cube = std::any_cast<arma::icube>(value) == std::any_cast<arma::icube>(c.par_data[16]);
    CHECK(arma::all(arma::vectorise(cmp_cube)));

    // Cube, unsigned
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "cube_unsigned");
    CHECK(quadriga_lib::any_type_id(&value) == 34);
    cmp_cube = std::any_cast<arma::u32_cube>(value) == std::any_cast<arma::Cube<unsigned>>(c.par_data[17]);
    CHECK(arma::all(arma::vectorise(cmp_cube)));

    // Cube, int
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "cube_int");
    CHECK(quadriga_lib::any_type_id(&value) == 35);
    cmp_cube = std::any_cast<arma::s32_cube>(value) == std::any_cast<arma::Cube<int>>(c.par_data[18]);
    CHECK(arma::all(arma::vectorise(cmp_cube)));

    // Col, float
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "col_float");
    CHECK(quadriga_lib::any_type_id(&value) == 40);
    CHECK(arma::approx_equal(std::any_cast<arma::fvec>(value), std::any_cast<arma::Col<float>>(c.par_data[19]), "absdiff", 1e-14));

    // Col, double
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "col_double");
    CHECK(quadriga_lib::any_type_id(&value) == 41);
    CHECK(arma::approx_equal(std::any_cast<arma::vec>(value), std::any_cast<arma::vec>(c.par_data[20]), "absdiff", 1e-14));

    // Col, uword
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "col_uword");
    CHECK(quadriga_lib::any_type_id(&value) == 42);
    CHECK(arma::all(std::any_cast<arma::uvec>(value) == std::any_cast<arma::uvec>(c.par_data[21])));

    // Col, sword
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "col_sword");
    CHECK(quadriga_lib::any_type_id(&value) == 43);
    CHECK(arma::all(std::any_cast<arma::icolvec>(value) == std::any_cast<arma::ivec>(c.par_data[22])));

    // Col, unsigned
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "col_unsigned");
    CHECK(quadriga_lib::any_type_id(&value) == 44);
    CHECK(arma::all(std::any_cast<arma::u32_vec>(value) == std::any_cast<arma::Col<unsigned>>(c.par_data[23])));

    // Col, int
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "col_int");
    CHECK(quadriga_lib::any_type_id(&value) == 45);
    CHECK(arma::all(std::any_cast<arma::s32_vec>(value) == std::any_cast<arma::Col<int>>(c.par_data[24])));

    // Row, float
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "row_float");
    CHECK(quadriga_lib::any_type_id(&value) == 40);
    CHECK(arma::approx_equal(std::any_cast<arma::fvec>(value), std::any_cast<arma::Row<float>>(c.par_data[25]).as_col(), "absdiff", 1e-14));

    // Row, double
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "row_double", 7);
    CHECK(quadriga_lib::any_type_id(&value) == 41);
    CHECK(arma::approx_equal(std::any_cast<arma::vec>(value), std::any_cast<arma::rowvec>(c.par_data[26]).as_col(), "absdiff", 1e-14));

    // Row, uword
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "row_uword");
    CHECK(quadriga_lib::any_type_id(&value) == 42);
    CHECK(arma::all(std::any_cast<arma::uvec>(value) == std::any_cast<arma::u64_rowvec>(c.par_data[27]).as_col()));

    // Row, sword
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "row_sword");
    CHECK(quadriga_lib::any_type_id(&value) == 43);
    CHECK(arma::all(std::any_cast<arma::icolvec>(value) == std::any_cast<arma::irowvec>(c.par_data[28]).as_col()));

    // Row, unsigned
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "row_unsigned");
    CHECK(quadriga_lib::any_type_id(&value) == 44);
    CHECK(arma::all(std::any_cast<arma::u32_vec>(value) == std::any_cast<arma::Row<unsigned>>(c.par_data[29]).as_col()));

    // Row, int
    value = quadriga_lib::hdf5_read_dset("test.hdf5", "row_int");
    CHECK(quadriga_lib::any_type_id(&value) == 45);
    CHECK(arma::all(std::any_cast<arma::s32_vec>(value) == std::any_cast<arma::Row<int>>(c.par_data[30]).as_col()));

    // Read channel from file
    auto d = quadriga_lib::hdf5_read_channel<double>("test.hdf5", 7);
    CHECK(!d.empty());           // Contains data?
    CHECK(d.is_valid().empty()); // Data is valid?

    CHECK(d.name == c.name);
    CHECK(d.center_frequency(0) == c.center_frequency(0));
    CHECK(arma::approx_equal(d.tx_pos, c.tx_pos, "absdiff", 1e-14));
    CHECK(arma::approx_equal(d.rx_pos, c.rx_pos, "absdiff", 1e-14));
    CHECK(arma::approx_equal(d.tx_orientation, c.tx_orientation, "absdiff", 1e-14));
    CHECK(arma::approx_equal(d.rx_orientation, c.rx_orientation, "absdiff", 1e-14));

    // Storage as float reduces precision
    for (auto i = 0ULL; i < c.coeff_re.size(); ++i)
    {
        CHECK(arma::approx_equal(d.coeff_re[i], c.coeff_re[i], "absdiff", 1e-6));
        CHECK(arma::approx_equal(d.coeff_im[i], c.coeff_im[i], "absdiff", 1e-6));
        CHECK(arma::approx_equal(d.delay[i], c.delay[i], "absdiff", 1e-14));
        CHECK(arma::approx_equal(d.path_gain[i], c.path_gain[i], "absdiff", 1e-14));
        CHECK(arma::approx_equal(d.path_length[i], c.path_length[i], "absdiff", 2e-6));
        CHECK(arma::approx_equal(d.path_polarization[i], c.path_polarization[i], "absdiff", 1e-14));
        CHECK(arma::approx_equal(d.path_angles[i], c.path_angles[i], "absdiff", 1e-6));
        CHECK(arma::approx_equal(d.interact_coord[i], c.interact_coord[i], "absdiff", 1e-6));
    }

    // Check if the unstructured data fields are in the same order as written
    CHECK(d.par_names.size() == c.par_names.size());
    for (auto i = 0ULL; i < c.par_names.size(); ++i)
    {
        CHECK(c.par_names[i] == d.par_names[i]);
        int a = quadriga_lib::any_type_id(&c.par_data[i]);
        int b = quadriga_lib::any_type_id(&d.par_data[i]);
        CHECK((a > 0 && b > 0));
        if (a < 50)
            CHECK(a == b);
        else // Conversion from row to col
            CHECK(b == a - 10);
    }

    std::remove("test.hdf5");
}
