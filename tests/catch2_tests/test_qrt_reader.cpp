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

    arma::fvec fGHz;
    arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;

    // Check parser
    quadriga_lib::qrt_file_parse(fn, &no_cir, &no_orig, &no_dest, &no_freq, &cir_offset, &orig_names, &dest_names, &version,
                                 &fGHz, &cir_pos, &cir_orientation, &orig_pos, &orig_orientation);

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

    REQUIRE(fGHz.n_elem == 1);
    CHECK(fGHz[0] == 3.75f);

    REQUIRE(cir_pos.n_rows == 7);
    REQUIRE(cir_pos.n_cols == 3);
    arma::fvec F = {-8.8350, -5.8614, -5.5750, -5.3516, -5.2809, -5.3739, -5.5913};
    CHECK(arma::approx_equal(cir_pos.col(0), F, "absdiff", 1.5e-4));

    F = {57.1893, 53.8124, 54.7705, 55.7446, 56.7405, 57.7351, 58.7106};
    CHECK(arma::approx_equal(cir_pos.col(1), F, "absdiff", 1.5e-4));

    F = {1, 1, 1, 1, 1, 1, 1};
    CHECK(arma::approx_equal(cir_pos.col(2), F, "absdiff", 1.5e-4));

    F = {0, 0, 0, 0, 0, 0, 0};
    CHECK(arma::approx_equal(cir_orientation.col(0), F, "absdiff", 1.5e-4));
    CHECK(arma::approx_equal(cir_orientation.col(1), F, "absdiff", 1.5e-4));

    F = {0, 1.2753, 1.2802, 1.3454, 1.5000, 1.6640, 1.7901};
    CHECK(arma::approx_equal(cir_orientation.col(2), F, "absdiff", 1.5e-4));

    F = {-12.9607, -2.6789, -8.1167};
    CHECK(arma::approx_equal(orig_pos.col(0), F, "absdiff", 1.5e-4));

    F = {59.6906, 60.2570, 51.9636};
    CHECK(arma::approx_equal(orig_pos.col(1), F, "absdiff", 1.5e-4));

    F = {2, 2, 2};
    CHECK(arma::approx_equal(orig_pos.col(2), F, "absdiff", 1.5e-4));

    F = {0, 0, 0};
    CHECK(arma::approx_equal(orig_orientation.col(0), F, "absdiff", 1.5e-4));
    CHECK(arma::approx_equal(orig_orientation.col(1), F, "absdiff", 1.5e-4));

    F = {0, 3.1416, 1.5708};
    CHECK(arma::approx_equal(orig_orientation.col(2), F, "absdiff", 1.5e-4));

    // Check Cache
    std::ifstream stream(fn, std::ios::in | std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init("", &stream);

    REQUIRE(cache.version == 4);
    REQUIRE(cache.no_orig == 3);
    REQUIRE(cache.no_dest == 2);
    REQUIRE(cache.no_cir == 7);
    REQUIRE(cache.no_freq == 1);

    CHECK(arma::approx_equal(cache.freq, fGHz, "absdiff", 1.5e-10));
    CHECK(arma::approx_equal(cache.cir_pos, cir_pos, "absdiff", 1.5e-10));
    CHECK(arma::approx_equal(cache.cir_orientation, cir_orientation, "absdiff", 1.5e-10));
    CHECK(arma::approx_equal(cache.orig_pos_all, orig_pos, "absdiff", 1.5e-10));
    CHECK(arma::approx_equal(cache.orig_orientation, orig_orientation, "absdiff", 1.5e-10));

    arma::u64_vec U = {249, 6243, 12762};
    CHECK(arma::all(cache.orig_index == U));

    U += 1 + 3 + 4 + 4;
    CHECK(arma::all(cache.path_data_offset == U));

    for (int xxx = 0; xxx < 2; ++xxx)
    {
        // Read data of first link
        arma::vec tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa;
        arma::cube M;
        arma::mat fbs_pos, lbs_pos, path_gain;
        arma::vec path_length, center_frequency;
        std::vector<arma::mat> path_coord;

        // Check positions and orientations of first link
        // Normalization: path gain = FSPL, Jones matrix M = material losses
        if (xxx == 0) // Open / close file
            quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &center_frequency, &tx_pos, &tx_orientation,
                                                &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos, &path_gain,
                                                &path_length, &M, &aod, &eod, &aoa, &eoa, &path_coord, 0);
        else // Use cache
            quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &center_frequency, &tx_pos, &tx_orientation,
                                                &rx_pos, &rx_orientation, &fbs_pos, &lbs_pos, &path_gain,
                                                &path_length, &M, &aod, &eod, &aoa, &eoa, &path_coord, 0,
                                                nullptr, nullptr, &stream, &cache);

        CHECK(center_frequency[0] == 3.75e9);
        arma::vec O = {-12.9607, 59.6906, 2.0};
        CHECK(arma::approx_equal(tx_pos, O, "absdiff", 1.5e-4));
        arma::vec T = {0.0, 0.0, 0.0};
        CHECK(arma::approx_equal(tx_orientation, T, "absdiff", 1.5e-4));
        CHECK(arma::approx_equal(rx_orientation, T, "absdiff", 1.5e-4));
        arma::vec R = {-8.83498, 57.1893, 1.0};
        CHECK(arma::approx_equal(rx_pos, R, "absdiff", 1.5e-4));
        REQUIRE(fbs_pos.n_cols == 19);

        // Path 0: O > FBS > LBS > R
        arma::vec FBS = {-0.2000, 58.2234, 0.8380};
        CHECK(arma::approx_equal(fbs_pos.col(0), FBS, "absdiff", 1.5e-4));
        arma::vec LBS = {-3.9123, 57.7967, 0.5000};
        CHECK(arma::approx_equal(lbs_pos.col(0), LBS, "absdiff", 1.5e-4));
        double d1 = arma::norm(FBS - O);   // O to FBS
        double d2 = arma::norm(LBS - FBS); // FBS to LBS
        double d3 = arma::norm(R - LBS);   // LBS to R
        double total_length = d1 + d2 + d3;

        // Calculate angles for path 0
        arma::vec dir_tx = (FBS - O) / d1;
        double aod_calc = std::atan2(dir_tx[1], dir_tx[0]);
        double eod_calc = std::asin(dir_tx[2]);

        arma::vec dir_rx = (R - LBS) / d3;
        double aoa_calc = std::atan2(dir_rx[1], dir_rx[0]) + M_PI;
        double eoa_calc = -std::asin(dir_rx[2]);

        // Calculate FSPL as linear gain: (lambda / (4*pi*d))^2
        double wavelength = 299792458.0 / center_frequency[0]; // c/f
        double path_gain_calc = wavelength / (4.0 * M_PI * total_length);
        path_gain_calc = path_gain_calc * path_gain_calc; // Square it

        REQUIRE(path_length.n_elem == 19);
        REQUIRE(path_gain.n_elem == 19);
        CHECK(std::abs(path_length[0] - total_length) < 1.5e-4);
        CHECK(std::abs(aod[0] - aod_calc) < 1.5e-4);
        CHECK(std::abs(eod[0] - eod_calc) < 1.5e-4);
        CHECK(std::abs(aoa[0] - aoa_calc) < 1.5e-4);
        CHECK(std::abs(eoa[0] - eoa_calc) < 1.5e-4);
        CHECK(std::abs(path_gain[0] - path_gain_calc) / path_gain_calc < 1e-3); // Relative error

        // Path 17: O > FBS > R (single bounce)
        FBS = {-9.88096, 57.83151626586914, 0.5};
        CHECK(arma::approx_equal(fbs_pos.col(17), FBS, "absdiff", 1.5e-4));
        d1 = arma::norm(FBS - O); // O to FBS
        d3 = arma::norm(R - FBS); // FBS to R
        total_length = d1 + d3;

        // Calculate angles for path 17
        dir_tx = (FBS - O) / d1;
        aod_calc = std::atan2(dir_tx[1], dir_tx[0]);
        eod_calc = std::asin(dir_tx[2]);

        dir_rx = (R - FBS) / d3;
        aoa_calc = std::atan2(dir_rx[1], dir_rx[0]) + M_PI;
        eoa_calc = -std::asin(dir_rx[2]);

        // Calculate FSPL as linear gain
        path_gain_calc = wavelength / (4.0 * M_PI * total_length);
        path_gain_calc = path_gain_calc * path_gain_calc;

        CHECK(std::abs(path_length[17] - total_length) < 1.5e-4);
        CHECK(std::abs(aod[17] - aod_calc) < 1.5e-4);
        CHECK(std::abs(eod[17] - eod_calc) < 1.5e-4);
        CHECK(std::abs(aoa[17] - aoa_calc) < 1.5e-4);
        CHECK(std::abs(eoa[17] - eoa_calc) < 1.5e-4);
        CHECK(std::abs(path_gain[17] - path_gain_calc) / path_gain_calc < 1e-3);

        // Path 18: O > R (LOS path)
        d1 = arma::norm(R - O); // O to R (LOS path)

        // Calculate angles for path 18
        dir_tx = (R - O) / d1;
        aod_calc = std::atan2(dir_tx[1], dir_tx[0]);
        eod_calc = std::asin(dir_tx[2]);

        // For LOS, arrival is opposite of departure
        aoa_calc = std::atan2(dir_tx[1], dir_tx[0]) + M_PI;
        eoa_calc = -std::asin(dir_tx[2]);

        // Calculate FSPL as linear gain
        path_gain_calc = wavelength / (4.0 * M_PI * d1);
        path_gain_calc = path_gain_calc * path_gain_calc;

        CHECK(std::abs(path_length[18] - d1) < 1.5e-4);
        CHECK(std::abs(aod[18] - aod_calc) < 1.5e-4);
        CHECK(std::abs(eod[18] - eod_calc) < 1.5e-4);
        CHECK(std::abs(aoa[18] - aoa_calc) < 1.5e-4);
        CHECK(std::abs(eoa[18] - eoa_calc) < 1.5e-4);
        CHECK(std::abs(path_gain[18] - path_gain_calc) / path_gain_calc < 1e-3);

        arma::cube M1;
        arma::mat path_gain1;

        // Normalization: path gain = path power, Jones matrix M = stronger column has power of 2
        if (xxx == 0) // Open / close file
            quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, nullptr, &path_gain1,
                                                nullptr, &M1, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
        else
            quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, nullptr, &path_gain1,
                                                nullptr, &M1, nullptr, nullptr, nullptr, nullptr, nullptr, 1,
                                                nullptr, nullptr, &stream, &cache);

        REQUIRE(M.n_rows == 8);
        REQUIRE(M.n_cols == 19);
        REQUIRE(M.n_slices == 1);
        REQUIRE(M1.n_rows == 8);
        REQUIRE(M1.n_cols == 19);
        REQUIRE(M1.n_slices == 1);
        REQUIRE(path_gain1.n_elem == 19);

        // For each path, calculate effective power and compare normalizations
        for (arma::uword i = 0; i < 19; ++i)
        {
            // Extract Jones matrix for path i (4 complex values = 8 real values)
            // Layout: [Re(H_VV), Im(H_VV), Re(H_VH), Im(H_VH), Re(H_HV), Im(H_HV), Re(H_HH), Im(H_HH)]
            arma::cx_mat H(2, 2);
            H(0, 0) = arma::cx_double(M(0, i, 0), M(1, i, 0)); // VV
            H(0, 1) = arma::cx_double(M(2, i, 0), M(3, i, 0)); // VH
            H(1, 0) = arma::cx_double(M(4, i, 0), M(5, i, 0)); // HV
            H(1, 1) = arma::cx_double(M(6, i, 0), M(7, i, 0)); // HH

            arma::cx_mat H1(2, 2);
            H1(0, 0) = arma::cx_double(M1(0, i, 0), M1(1, i, 0));
            H1(0, 1) = arma::cx_double(M1(2, i, 0), M1(3, i, 0));
            H1(1, 0) = arma::cx_double(M1(4, i, 0), M1(5, i, 0));
            H1(1, 1) = arma::cx_double(M1(6, i, 0), M1(7, i, 0));

            // Normalization 0: effective_power = path_gain * ||H||_F^2 / 2
            // Frobenius norm squared = sum of squared magnitudes
            // Division by 2 because power is averaged over two polarizations
            double H_power = std::abs(H(0, 0)) * std::abs(H(0, 0)) +
                             std::abs(H(0, 1)) * std::abs(H(0, 1)) +
                             std::abs(H(1, 0)) * std::abs(H(1, 0)) +
                             std::abs(H(1, 1)) * std::abs(H(1, 1));
            double effective_power_0 = path_gain[i] * H_power / 2.0;

            // Normalization 1: effective_power = path_gain1 * ||H1||_F^2 / 2
            // H1 has normalized columns where stronger column has power of 1.0
            double col0_power = std::abs(H1(0, 0)) * std::abs(H1(0, 0)) +
                                std::abs(H1(1, 0)) * std::abs(H1(1, 0));
            double col1_power = std::abs(H1(0, 1)) * std::abs(H1(0, 1)) +
                                std::abs(H1(1, 1)) * std::abs(H1(1, 1));
            double max_col_power = std::max(col0_power, col1_power);
            CHECK(std::abs(max_col_power - 1.0) < 1.5e-3); // Verify normalization

            double H1_power = std::abs(H1(0, 0)) * std::abs(H1(0, 0)) +
                              std::abs(H1(0, 1)) * std::abs(H1(0, 1)) +
                              std::abs(H1(1, 0)) * std::abs(H1(1, 0)) +
                              std::abs(H1(1, 1)) * std::abs(H1(1, 1));
            double effective_power_1 = path_gain1[i] * H1_power / 2.0;

            // Compare: both normalizations should give same effective power
            double rel_error = std::abs(effective_power_0 - effective_power_1) / effective_power_1;
            CHECK(rel_error < 1e-3);
        }

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
