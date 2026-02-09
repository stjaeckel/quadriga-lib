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

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"

TEST_CASE("QRT Reader")
{
    std::string fn = "tests/data/test.qrt";

    arma::uword no_orig, no_cir, no_dest, no_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;

    quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &cir_offset, &orig_names, &dest_names, &version);

    REQUIRE(no_orig == 3ULL);
    REQUIRE(no_dest == 2ULL);
    REQUIRE(no_cir == 7ULL);
    REQUIRE(no_freq == 1ULL);
    REQUIRE(version == 4);
    REQUIRE(cir_offset.n_elem == no_dest);

    CHECK(cir_offset[0] == 0ULL);
    CHECK(cir_offset[1] == 1ULL);

    REQUIRE(orig_names.size() == 3);
    REQUIRE(dest_names.size() == 2);

    CHECK(orig_names[0] == "TX1");
    CHECK(orig_names[1] == "TX2");
    CHECK(orig_names[2] == "TX3");

    CHECK(dest_names[0] == "RX1");
    CHECK(dest_names[1] == "RX2");

    arma::vec tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa;
    arma::cube M;
    arma::mat fbs_pos, lbs_pos, path_gain;
    arma::vec path_length, center_frequency;
    std::vector<arma::mat> path_coord;

    // Check positions and orientations of first link
    quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &center_frequency, &tx_pos, &tx_orientation,
                                        &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos, &path_gain,
                                        &path_length, &M, &aod, &eod, &aoa, &eoa, &path_coord);

    CHECK(center_frequency[0] == 3.75e9);

    arma::vec T = {-12.9607, 59.6906, 2.0};
    CHECK(arma::approx_equal(tx_pos, T, "absdiff", 1.5e-4));

    T = {0.0, 0.0, 0.0};
    CHECK(arma::approx_equal(tx_orientation, T, "absdiff", 1.5e-4));

    T = {-8.83498, 57.1893, 1.0};
    CHECK(arma::approx_equal(rx_pos, T, "absdiff", 1.5e-4));

    // Check positions and orientations of second link
    quadriga_lib::qrt_file_read<double>(fn, 1, 1, true, &center_frequency, &tx_pos, &tx_orientation,
                                        &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos, &path_gain,
                                        &path_length, &M, &aod, &eod, &aoa, &eoa, &path_coord);

    T = {-2.67888, 60.257, 2.0};
    CHECK(arma::approx_equal(tx_pos, T, "absdiff", 1.5e-4));

    T = {0.0, 0.0, arma::datum::pi};
    CHECK(arma::approx_equal(tx_orientation, T, "absdiff", 1.5e-4));

    T = {-5.86144, 53.8124, 1.0};
    CHECK(arma::approx_equal(rx_pos, T, "absdiff", 1.5e-4));

    T = {0.0, 0.0, 1.2753};
    CHECK(arma::approx_equal(rx_orientation, T, "absdiff", 1.5e-4));

    // Test uplink
    arma::mat B, C;
    quadriga_lib::qrt_file_read<double>(fn, 1, 1, false, &center_frequency, &tx_pos, &tx_orientation,
                                        &rx_pos, &rx_orientation, &B, &C, &path_gain,
                                        &path_length, &M, &aod, &eod);

    T = {-5.86144, 53.8124, 1.0};
    CHECK(arma::approx_equal(tx_pos, T, "absdiff", 1.5e-4));

    T = {0.0, 0.0, 1.2753};
    CHECK(arma::approx_equal(tx_orientation, T, "absdiff", 1.5e-4));

    T = {-2.67888, 60.257, 2.0};
    CHECK(arma::approx_equal(rx_pos, T, "absdiff", 1.5e-4));

    T = {0.0, 0.0, arma::datum::pi};
    CHECK(arma::approx_equal(rx_orientation, T, "absdiff", 1.5e-4));

    CHECK(arma::approx_equal(fbs_pos, C, "absdiff", 1.5e-4));
    CHECK(arma::approx_equal(lbs_pos, B, "absdiff", 1.5e-4));
    CHECK(arma::approx_equal(aod, aoa, "absdiff", 1.5e-4));
    CHECK(arma::approx_equal(eod, eoa, "absdiff", 1.5e-4));
}

TEST_CASE("QRT Reader v5")
{
    std::string fn = "tests/data/test_v5.qrt";

    arma::uword no_orig, no_cir, no_dest, no_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;

    quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &cir_offset, &orig_names, &dest_names, &version);

    CHECK(version == 5);
    CHECK(no_freq == 2);

    arma::cube M;
    arma::mat path_gain;
    arma::vec center_frequency;

    // Check positions and orientations of first link
    quadriga_lib::qrt_file_read<double>(fn, 1, 0, true, &center_frequency, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, &path_gain,
                                        nullptr, &M, nullptr, nullptr, nullptr, nullptr, nullptr, 0);

    REQUIRE(center_frequency.n_elem == 2);
    CHECK(center_frequency[0] == 1.0e9);
    CHECK(center_frequency[1] == 1.5e9);

    CHECK(M.n_rows == 8);
    CHECK(M.n_slices == 2);

    CHECK(path_gain.n_cols == 2);

    arma::vec A(M.slice_colptr(0, 0), 8);
    arma::vec T = {0.1131, 0, 0, 0, 0, 0, -0.1131, 0};
    CHECK(arma::approx_equal(A, T, "absdiff", 1.5e-4));

    arma::vec B(M.slice_colptr(1, 0), 8);
    T = {0.0866, 0, 0, 0, 0, 0, -0.0866, 0};
    CHECK(arma::approx_equal(B, T, "absdiff", 1.5e-4));

    // Check positions and orientations of first link
    quadriga_lib::qrt_file_read<double>(fn, 1, 0, true, &center_frequency, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, &path_gain,
                                        nullptr, &M);

    T = {1, 0, 0, 0, 0, 0, -1, 0};
    arma::vec C(M.slice_colptr(1, 0), 8);
    CHECK(arma::approx_equal(C, T, "absdiff", 1.5e-4));
}
