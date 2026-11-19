// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// qrt_file_parse, qrt_read_cache_init and qrt_file_read on the QRT files in tests/data (test.qrt: version 4,
// 3 origins, 2 destinations, 7 CIRs; test_v5.qrt: two frequencies). Oracles: positions and orientations stored in
// the files, path lengths, angles and free-space gain recomputed from the interaction points, uplink as the
// reciprocal of downlink, list reads against single-CIR reads, and the error and aliasing rules.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "quadriga_channel.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

// A missing file is a file problem (std::runtime_error); an index outside the file's layout an argument error
TEST_CASE("QRT Reader - missing file and index outside the file")
{
    const std::string missing = "tests/data/test_qrt_missing_file.qrt";
    CHECK_THROWS_AS(quadriga_lib::qrt_file_parse(missing), std::runtime_error);
    CHECK_THROWS_AS(quadriga_lib::qrt_read_cache_init(missing), std::runtime_error);
    CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(missing), std::runtime_error);

    arma::uword n_cir = 0;
    quadriga_lib::qrt_file_parse("tests/data/test.qrt", &n_cir);
    CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>("tests/data/test.qrt", {n_cir}), std::invalid_argument);
}

TEST_CASE("QRT Reader")
{
    std::string fn = "tests/data/test.qrt";

    arma::uword n_orig, n_cir, n_dest, n_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;

    arma::fvec center_freq_f;
    arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;

    quadriga_lib::qrt_file_parse(fn, &n_cir, &n_orig, &n_dest, &n_freq, &cir_offset, &orig_names, &dest_names, &version,
                                 &center_freq_f, &cir_pos, &cir_orientation, &orig_pos, &orig_orientation);

    REQUIRE(n_orig == 3ULL);
    REQUIRE(n_dest == 2ULL);
    REQUIRE(n_cir == 7ULL);
    REQUIRE(n_freq == 1ULL);
    REQUIRE(version == 4);
    REQUIRE(cir_offset.n_elem == n_dest);

    CHECK(cir_offset[0] == 0ULL);
    CHECK(cir_offset[1] == 1ULL);

    REQUIRE(orig_names.size() == 3);
    REQUIRE(dest_names.size() == 2);

    CHECK(orig_names[0] == "TX1");
    CHECK(orig_names[1] == "TX2");
    CHECK(orig_names[2] == "TX3");

    CHECK(dest_names[0] == "RX1");
    CHECK(dest_names[1] == "RX2");

    REQUIRE(center_freq_f.n_elem == 1);
    CHECK(center_freq_f[0] == 3.75e9f);

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

    std::ifstream stream(fn, std::ios::in | std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init("", &stream);

    REQUIRE(cache.version == 4);
    REQUIRE(cache.n_orig == 3);
    REQUIRE(cache.n_dest == 2);
    REQUIRE(cache.n_cir == 7);
    REQUIRE(cache.n_freq == 1);

    CHECK(arma::approx_equal(cache.freq, center_freq_f, "reldiff", 1.0e-7)); // v4 file stores GHz, the cache holds Hz
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
        // Read data of one link (one CIR), entry 0 of the per-CIR outputs
        arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;
        arma::vec aod, eod, aoa, eoa, path_length, center_freq;
        arma::cube M;
        arma::mat fbs_pos, lbs_pos, path_gain;
        auto read = [&](arma::uword i_cir, arma::uword i_orig, bool uplink, int normalize_polmat, bool use_cache)
        {
            std::vector<arma::mat> v_fbs, v_lbs, v_pg;
            std::vector<arma::vec> v_pl, v_aod, v_eod, v_aoa, v_eoa;
            std::vector<arma::cube> v_M;
            std::vector<std::vector<arma::mat>> v_pc;
            quadriga_lib::qrt_file_read<double>(fn, {i_cir}, i_orig, uplink, &center_freq, &tx_pos, &tx_orientation,
                                                &rx_pos, &rx_orientation, &v_fbs, &v_lbs, &v_pg, &v_pl, &v_M,
                                                &v_aod, &v_eod, &v_aoa, &v_eoa, &v_pc, normalize_polmat, nullptr, nullptr,
                                                nullptr, use_cache ? &stream : nullptr, use_cache ? &cache : nullptr);
            REQUIRE(v_fbs.size() == 1);
            fbs_pos = v_fbs[0], lbs_pos = v_lbs[0], path_gain = v_pg[0], path_length = v_pl[0], M = v_M[0];
            aod = v_aod[0], eod = v_eod[0], aoa = v_aoa[0], eoa = v_eoa[0];
        };

        // First link, normalize_polmat = 0: path gain = FSPL, Jones matrix M = material losses
        read(0, 0, false, 0, xxx == 1); // xxx = 0: open / close file, 1: use cache

        CHECK(std::abs(center_freq[0] - 3.75e9) < 1.0e-7 * 3.75e9); // v4 stores GHz, converted to Hz in float
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

        arma::vec dir_tx = (FBS - O) / d1;
        double aod_calc = std::atan2(dir_tx[1], dir_tx[0]);
        double eod_calc = std::asin(dir_tx[2]);

        arma::vec dir_rx = (R - LBS) / d3;
        double aoa_calc = std::atan2(dir_rx[1], dir_rx[0]) + M_PI;
        double eoa_calc = -std::asin(dir_rx[2]);

        // Calculate FSPL as linear gain: (lambda / (4*pi*d))^2
        double wavelength = 299792458.0 / center_freq[0];
        double path_gain_calc = wavelength / (4.0 * M_PI * total_length);
        path_gain_calc = path_gain_calc * path_gain_calc;

        REQUIRE(path_length.n_elem == 19);
        REQUIRE(path_gain.n_elem == 19);
        CHECK(std::abs(path_length[0] - total_length) < 1.5e-4);
        CHECK(std::abs(aod[0] - aod_calc) < 1.5e-4);
        CHECK(std::abs(eod[0] - eod_calc) < 1.5e-4);
        CHECK(std::abs(aoa[0] - aoa_calc) < 1.5e-4);
        CHECK(std::abs(eoa[0] - eoa_calc) < 1.5e-4);
        CHECK(std::abs(path_gain[0] - path_gain_calc) / path_gain_calc < 1e-3);

        // Path 17: O > FBS > R (single bounce)
        FBS = {-9.88096, 57.83151626586914, 0.5};
        CHECK(arma::approx_equal(fbs_pos.col(17), FBS, "absdiff", 1.5e-4));
        d1 = arma::norm(FBS - O); // O to FBS
        d3 = arma::norm(R - FBS); // FBS to R
        total_length = d1 + d3;

        dir_tx = (FBS - O) / d1;
        aod_calc = std::atan2(dir_tx[1], dir_tx[0]);
        eod_calc = std::asin(dir_tx[2]);

        dir_rx = (R - FBS) / d3;
        aoa_calc = std::atan2(dir_rx[1], dir_rx[0]) + M_PI;
        eoa_calc = -std::asin(dir_rx[2]);

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

        dir_tx = (R - O) / d1;
        aod_calc = std::atan2(dir_tx[1], dir_tx[0]);
        eod_calc = std::asin(dir_tx[2]);

        // For LOS, arrival is opposite of departure
        aoa_calc = std::atan2(dir_tx[1], dir_tx[0]) + M_PI;
        eoa_calc = -std::asin(dir_tx[2]);

        path_gain_calc = wavelength / (4.0 * M_PI * d1);
        path_gain_calc = path_gain_calc * path_gain_calc;

        CHECK(std::abs(path_length[18] - d1) < 1.5e-4);
        CHECK(std::abs(aod[18] - aod_calc) < 1.5e-4);
        CHECK(std::abs(eod[18] - eod_calc) < 1.5e-4);
        CHECK(std::abs(aoa[18] - aoa_calc) < 1.5e-4);
        CHECK(std::abs(eoa[18] - eoa_calc) < 1.5e-4);
        CHECK(std::abs(path_gain[18] - path_gain_calc) / path_gain_calc < 1e-3);

        // normalize_polmat = 1: path gain = path power, the stronger column of M has power 1
        const arma::cube M0 = M;
        const arma::mat path_gain0 = path_gain;
        read(0, 0, false, 1, xxx == 1);
        const arma::cube M1 = M;
        const arma::mat path_gain1 = path_gain;
        M = M0, path_gain = path_gain0;

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
            // Rows (re, im): 0-1 VV, 2-3 HV, 4-5 VH, 6-7 HH (rx, tx)
            arma::cx_mat H(2, 2);
            H(0, 0) = arma::cx_double(M(0, i, 0), M(1, i, 0)); // VV
            H(0, 1) = arma::cx_double(M(2, i, 0), M(3, i, 0)); // HV
            H(1, 0) = arma::cx_double(M(4, i, 0), M(5, i, 0)); // VH
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

        // Second link
        read(1, 1, false, 1, false);

        T = {-2.67888, 60.257, 2.0};
        CHECK(arma::approx_equal(tx_pos, T, "absdiff", 1.5e-4));

        T = {0.0, 0.0, arma::datum::pi};
        CHECK(arma::approx_equal(tx_orientation, T, "absdiff", 1.5e-4));

        T = {-5.86144, 53.8124, 1.0};
        CHECK(arma::approx_equal(rx_pos, T, "absdiff", 1.5e-4));

        T = {0.0, 0.0, 1.2753};
        CHECK(arma::approx_equal(rx_orientation, T, "absdiff", 1.5e-4));

        // Uplink
        const arma::cube M_dl = M;
        const arma::mat path_gain_dl = path_gain;
        const arma::mat fbs_dl = fbs_pos, lbs_dl = lbs_pos;
        const arma::vec aoa_dl = aoa, eoa_dl = eoa;
        read(1, 1, true, 1, false);
        const arma::mat B = fbs_pos, C = lbs_pos;
        fbs_pos = fbs_dl, lbs_pos = lbs_dl, aoa = aoa_dl, eoa = eoa_dl;

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

        // Reciprocity: the uplink M is the plain transpose of the downlink M, without conjugation.
        // Rows (re, im): 0-1 VV, 2-3 HV, 4-5 VH, 6-7 HH (rx, tx); the transpose swaps HV and VH.
        // This link has paths with complex, non-symmetric M, so a conjugation or a missing swap shows.
        REQUIRE(M.n_rows == 8);
        REQUIRE(M.n_cols == M_dl.n_cols);
        REQUIRE(M.n_slices == M_dl.n_slices);
        arma::cube M_T = M_dl;
        M_T.rows(2, 3) = M_dl.rows(4, 5);
        M_T.rows(4, 5) = M_dl.rows(2, 3);
        CHECK(arma::approx_equal(M, M_T, "absdiff", 1e-12));
        CHECK(arma::approx_equal(path_gain, path_gain_dl, "reldiff", 1e-12));
    }
}

TEST_CASE("QRT Reader v5")
{
    std::string fn = "tests/data/test_v5.qrt";

    arma::uword n_orig, n_cir, n_dest, n_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;

    quadriga_lib::qrt_file_parse(fn, &n_cir, &n_orig, &n_dest, &n_freq, &cir_offset, &orig_names, &dest_names, &version);

    CHECK(version == 5);
    CHECK(n_freq == 2);

    std::vector<arma::cube> v_M;
    std::vector<arma::mat> v_pg;
    arma::vec center_freq;

    // CIR 1, normalize_polmat = 0: M as stored, one slice per frequency
    quadriga_lib::qrt_file_read<double>(fn, {1}, 0, false, &center_freq, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, &v_pg,
                                        nullptr, &v_M, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    arma::cube M = v_M.at(0);
    arma::mat path_gain = v_pg.at(0);

    REQUIRE(center_freq.n_elem == 2);
    CHECK(center_freq[0] == 1.0e9);
    CHECK(center_freq[1] == 1.5e9);
    CHECK(arma::approx_equal(quadriga_lib::qrt_read_cache_init(fn).freq, arma::fvec({1.0e9f, 1.5e9f}), "reldiff", 1.0e-7f)); // Hz in the cache

    CHECK(M.n_rows == 8);
    CHECK(M.n_slices == 2);

    CHECK(path_gain.n_cols == 2);

    arma::vec A(M.slice_colptr(0, 0), 8);
    arma::vec T = {0.1131, 0, 0, 0, 0, 0, -0.1131, 0};
    CHECK(arma::approx_equal(A, T, "absdiff", 1.5e-4));

    arma::vec B(M.slice_colptr(1, 0), 8);
    T = {0.0866, 0, 0, 0, 0, 0, -0.0866, 0};
    CHECK(arma::approx_equal(B, T, "absdiff", 1.5e-4));

    // Default normalize_polmat = 1: the stronger column of M has power 1
    quadriga_lib::qrt_file_read<double>(fn, {1}, 0, false, &center_freq, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, &v_pg,
                                        nullptr, &v_M);
    M = v_M.at(0);

    T = {1, 0, 0, 0, 0, 0, -1, 0};
    arma::vec C(M.slice_colptr(1, 0), 8);
    CHECK(arma::approx_equal(C, T, "absdiff", 1.5e-4));
}

TEST_CASE("QRT Reader - angles without path length")
{
    // Aod, eod, aoa, eoa must be returned when requested alone (without path_gain or path_length)
    std::string fn = "tests/data/test_v5.qrt";

    std::vector<arma::vec> aod1(1, arma::vec{12345.0}), eod1 = aod1, aoa1 = aod1, eoa1 = aod1; // stale input values
    quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, &aod1, &eod1, &aoa1, &eoa1);

    std::vector<arma::vec> aod2, eod2, aoa2, eoa2, path_length;
    quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &path_length, nullptr, &aod2, &eod2, &aoa2, &eoa2);

    REQUIRE(path_length.at(0).n_elem > 0);
    REQUIRE(aod2.at(0).n_elem == path_length[0].n_elem);
    REQUIRE(aod1.at(0).n_elem == aod2[0].n_elem);
    REQUIRE(eod1.at(0).n_elem == eod2.at(0).n_elem);
    REQUIRE(aoa1.at(0).n_elem == aoa2.at(0).n_elem);
    REQUIRE(eoa1.at(0).n_elem == eoa2.at(0).n_elem);
    CHECK(arma::approx_equal(aod1[0], aod2[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(eod1[0], eod2[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(aoa1[0], aoa2[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(eoa1[0], eoa2[0], "absdiff", 1e-12));
}

TEST_CASE("QRT Reader - angles with 0 paths")
{
    // An origin without paths returns empty angle vectors (no memcpy from an empty matrix)
    std::string fn = "tests/data/test_read_zero_paths.qrt";

    arma::fvec freq = {3.75e9f};
    arma::fmat cir_pos(2, 3, arma::fill::zeros);
    arma::fmat cir_orient;
    std::vector<std::string> dest_names = {"RX"};
    arma::u32_vec cir_offset = {0u};
    quadriga_lib::qrt_file_init<float>(fn, freq, cir_pos, cir_orient, dest_names, cir_offset, 1, false);

    // One path in CIR 1, none in CIR 0
    quadriga_lib::path p(1, 1, false);
    p.i_cir = 1;
    float *c = p.coord(0);
    c[0] = 5.0f, c[1] = 5.0f, c[2] = 0.0f;
    float *m = p.polmat(0);
    for (unsigned k = 0; k < 8; ++k)
        m[k] = (k == 0 || k == 6) ? 0.5f : 0.0f;
    p.set_interaction_type_codes({10});
    std::vector<quadriga_lib::path> paths = {p};
    quadriga_lib::qrt_file_append<float>(fn, paths, arma::fvec{1.0f, 1.0f, 1.0f}, {}, "TX0");

    // CIR 1 has one path
    std::vector<arma::vec> aod1;
    quadriga_lib::qrt_file_read<double>(fn, {1}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, &aod1);
    CHECK(aod1.at(0).n_elem == 1);

    std::vector<arma::vec> aod(1, arma::vec{1.0}), eod = aod, aoa = aod, eoa = aod, path_length = aod;
    quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, &aod, &eod, &aoa, &eoa);
    CHECK(aod.at(0).n_elem == 0);
    CHECK(eod.at(0).n_elem == 0);
    CHECK(aoa.at(0).n_elem == 0);
    CHECK(eoa.at(0).n_elem == 0);

    aod[0] = {1.0};
    quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &path_length, nullptr, &aod, &eod, &aoa, &eoa);
    CHECK(path_length.at(0).n_elem == 0);
    CHECK(aod.at(0).n_elem == 0);

    std::remove(fn.c_str());
}

TEST_CASE("QRT Reader - index of 2^32 or more")
{
    // I_cir and i_orig are compared as 64-bit values, with and without the cache
    for (std::string fn : {"tests/data/test.qrt", "tests/data/test_v5.qrt"})
    {
        std::ifstream f(fn, std::ios::binary);
        auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
        const arma::uword big = 1ULL << 32;
        for (arma::uword i_cir : {big, big + 1ULL, (arma::uword)cache.n_cir})
        {
            arma::mat rx_pos;
            CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(fn, {i_cir}, 0, false, nullptr, nullptr, nullptr, &rx_pos),
                            std::invalid_argument);
            CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(fn, {i_cir}, 0, false, nullptr, nullptr, nullptr, &rx_pos,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, 1, nullptr, nullptr, nullptr, &f, &cache),
                            std::invalid_argument);
        }
        for (arma::uword i_orig : {big, (arma::uword)cache.n_orig})
        {
            arma::mat rx_pos;
            CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(fn, {0}, i_orig, false, nullptr, nullptr, nullptr, &rx_pos),
                            std::invalid_argument);
            CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(fn, {0}, i_orig, false, nullptr, nullptr, nullptr, &rx_pos,
                                                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                nullptr, nullptr, nullptr, nullptr, 1, nullptr, nullptr, nullptr, &f, &cache),
                            std::invalid_argument);
        }
        if (cache.version != 4)
        {
            CHECK_THROWS_AS(quadriga_lib::qrt_file_read_raw(fn, big), std::invalid_argument);
            CHECK_THROWS_AS(quadriga_lib::qrt_file_read_raw(fn, big, &f, &cache), std::invalid_argument);
        }
    }
}

TEST_CASE("QRT Reader - normalize_polmat must be 0 or 1")
{
    // Other values of normalize_polmat throw, with and without the cache
    std::string fn = "tests/data/test_v5.qrt";
    std::ifstream f(fn, std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
    for (int nM : {-1, 2, 5})
    {
        std::vector<arma::cube> M;
        CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr, &M, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, nM),
                        std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr, &M, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, nM, nullptr, nullptr, nullptr, &f, &cache),
                        std::invalid_argument);
    }
    for (int nM : {0, 1})
    {
        std::vector<arma::cube> M;
        quadriga_lib::qrt_file_read<double>(fn, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, &M, nullptr, nullptr, nullptr, nullptr, nullptr, nM);
        CHECK(M.at(0).n_cols > 0);
    }
}

TEST_CASE("QRT Reader - outputs that alias other outputs or the cache")
{
    std::string fn = "tests/data/test_v5.qrt";
    std::ifstream f(fn, std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);

    // Reads tx_pos, rx_pos, fbs_pos, lbs_pos, aoa, and coord
    using fvecs = std::vector<arma::fvec>;
    using fmats = std::vector<arma::fmat>;
    auto read = [&](arma::fmat *tx, arma::fmat *rx, fmats *fbs, fmats *lbs, fvecs *aoa, fmats *coord,
                    arma::fvec *fc = nullptr)
    {
        quadriga_lib::qrt_file_read<float>(fn, {0}, 0, false, fc, tx, nullptr, rx, nullptr, fbs, lbs, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, aoa, nullptr, nullptr, 1, nullptr, coord, nullptr, &f, &cache);
    };
    arma::fmat tx, rx;
    arma::fvec fc;
    fvecs aoa;
    fmats fbs, lbs, coord;
    CHECK_NOTHROW(read(&tx, &rx, &fbs, &lbs, &aoa, &coord, &fc));

    // Two outputs are the same object
    CHECK_THROWS_AS(read(&tx, &tx, &fbs, &lbs, &aoa, &coord), std::invalid_argument);
    CHECK_THROWS_AS(read(&tx, &rx, &fbs, &fbs, &aoa, &coord), std::invalid_argument);
    CHECK_THROWS_AS(read(&tx, &rx, &fbs, &lbs, &aoa, &fbs), std::invalid_argument);

    // An output uses the memory of another output
    tx.set_size(3, 1);
    arma::fmat tx_view(tx.memptr(), 3, 1, false, false);
    CHECK_THROWS_AS(read(&tx, &tx_view, &fbs, &lbs, &aoa, &coord), std::invalid_argument);

    // An output is part of the cache
    CHECK_THROWS_AS(read(&tx, &rx, &fbs, &lbs, &aoa, &coord, &cache.freq), std::invalid_argument);
    CHECK_THROWS_AS(read(&cache.cir_pos, &rx, &fbs, &lbs, &aoa, &coord), std::invalid_argument);

    // qrt_file_parse: two outputs are the same object
    arma::uword n1 = 0, n2 = 0;
    arma::fmat p1, p2;
    CHECK_NOTHROW(quadriga_lib::qrt_file_parse(fn, &n1, &n2));
    CHECK_THROWS_AS(quadriga_lib::qrt_file_parse(fn, &n1, &n1), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::qrt_file_parse(fn, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 nullptr, &p1, nullptr, &p1),
                    std::invalid_argument);
    std::vector<std::string> names;
    CHECK_THROWS_AS(quadriga_lib::qrt_file_parse(fn, nullptr, nullptr, nullptr, nullptr, nullptr, &names, &names),
                    std::invalid_argument);
}

// All outputs of qrt_file_read for a list of CIRs
struct qrt_read_out
{
    arma::vec cf;
    arma::mat tp, to, rp, ro;
    std::vector<arma::mat> fbs, lbs, pg;
    std::vector<arma::vec> pl, aod, eod, aoa, eoa;
    std::vector<arma::cube> M;
    std::vector<std::vector<arma::mat>> pc;
    std::vector<arma::u32_vec> ni;
    std::vector<arma::fmat> co;
    std::vector<std::vector<uint8_t>> it;

    void read(const std::string &fn, const arma::uvec &i_cir, arma::uword i_orig, bool uplink,
              std::ifstream *f = nullptr, const quadriga_lib::qrt_read_cache *cache = nullptr)
    {
        quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, uplink, &cf, &tp, &to, &rp, &ro, &fbs, &lbs, &pg, &pl, &M,
                                            &aod, &eod, &aoa, &eoa, &pc, 1, &ni, &co, &it, f, cache);
    }
};

TEST_CASE("QRT Reader - a list read equals single-CIR reads")
{
    // Entry k of a list read is the CIR i_cir[k] read alone; any order and repeats are allowed
    for (std::string fn : {"tests/data/test.qrt", "tests/data/test_v5.qrt"})
    {
        std::ifstream f(fn, std::ios::binary);
        auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
        const arma::uword n_cir = cache.n_cir;
        for (arma::uword i_orig = 0; i_orig < cache.n_orig; ++i_orig)
            for (bool uplink : {false, true})
            {
                INFO(fn << ", origin " << i_orig << ", uplink " << uplink);
                qrt_read_out all, sel;
                all.read(fn, {}, i_orig, uplink); // Empty list = all CIRs, without a cache
                const arma::uvec i_sel = {n_cir - 1, 0, n_cir - 1};
                sel.read(fn, i_sel, i_orig, uplink, &f, &cache);
                REQUIRE(all.fbs.size() == n_cir);
                REQUIRE(all.tp.n_cols == n_cir);
                REQUIRE(sel.M.size() == 3);
                for (arma::uword k = 0; k < n_cir + 3; ++k)
                {
                    const arma::uword i = k < n_cir ? k : i_sel[k - n_cir];
                    const qrt_read_out &L = k < n_cir ? all : sel;
                    const arma::uword j = k < n_cir ? k : k - n_cir;
                    qrt_read_out one;
                    one.read(fn, {i}, i_orig, uplink, &f, &cache);
                    CHECK(arma::approx_equal(L.cf, one.cf, "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.tp.col(j), one.tp, "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.to.col(j), one.to, "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.rp.col(j), one.rp, "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.ro.col(j), one.ro, "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.fbs[j], one.fbs[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.lbs[j], one.lbs[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.pg[j], one.pg[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.pl[j], one.pl[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.M[j], one.M[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.aod[j], one.aod[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.eod[j], one.eod[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.aoa[j], one.aoa[0], "absdiff", 0.0));
                    CHECK(arma::approx_equal(L.eoa[j], one.eoa[0], "absdiff", 0.0));
                    REQUIRE(L.pc[j].size() == one.pc[0].size());
                    for (size_t p = 0; p < one.pc[0].size(); ++p)
                        CHECK(arma::approx_equal(L.pc[j][p], one.pc[0][p], "absdiff", 0.0));
                    CHECK(arma::all(L.ni[j] == one.ni[0]));
                    CHECK(arma::approx_equal(L.co[j], one.co[0], "absdiff", 0.0f));
                    CHECK(L.it[j] == one.it[0]);
                }
            }
    }
}

TEST_CASE("QRT Reader - empty list and an index out of range")
{
    // A file without CIRs: all CIRs = none; positions [3, 0], per-CIR outputs empty, frequencies read
    qrt_read_out z;
    z.pg.resize(2);
    z.read("tests/data/test_zero_cir.qrt", {}, 0, false);
    CHECK(z.cf.n_elem > 0);
    CHECK(z.tp.n_rows == 3);
    CHECK(z.tp.n_cols == 0);
    CHECK(z.pg.empty());
    CHECK(z.it.empty());
    CHECK_THROWS_AS(z.read("tests/data/test_zero_cir.qrt", {}, 5, false), std::invalid_argument); // origin 5 of 1

    // An index out of range throws before any output is written
    std::string fn = "tests/data/test_v5.qrt";
    arma::uword n_cir = 0;
    quadriga_lib::qrt_file_parse(fn, &n_cir);
    qrt_read_out o;
    o.pg.resize(5);
    o.tp.ones(2, 2);
    CHECK_THROWS_AS(o.read(fn, {0, n_cir}, 0, false), std::invalid_argument);
    CHECK_THROWS_WITH(o.read(fn, {0, n_cir}, 0, false), Catch::Matchers::StartsWith("Snapshot 1: "));
    CHECK(o.pg.size() == 5);
    CHECK(o.tp.n_elem == 4);

    // A file error while reading names the position of the CIR in the list: the last CIR block is truncated
    const std::string fn_t = "tests/data/test_qrt_truncated_list.qrt";
    {
        std::ifstream in(fn, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::ofstream out(fn_t, std::ios::binary);
        out.write(data.data(), (std::streamsize)data.size() - 10);
    }
    qrt_read_out t;
    CHECK_NOTHROW(t.read(fn_t, {0}, 0, false));
    CHECK_THROWS_AS(t.read(fn_t, {0, 0, n_cir - 1}, 0, false), std::runtime_error);
    CHECK_THROWS_WITH(t.read(fn_t, {0, 0, n_cir - 1}, 0, false), Catch::Matchers::StartsWith("Snapshot 2: "));
    std::remove(fn_t.c_str());
}

TEST_CASE("QRT Reader - v4 frequency of each origin")
{
    // Version 4 stores the frequency in each origin block; give origin 1 another one (2.5 GHz, stored in GHz)
    const std::string fn = "tests/data/test_qrt_v4_freq.qrt";
    {
        std::ifstream in("tests/data/test.qrt", std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const float f_GHz = 2.5f;
        std::memcpy(&data[6243 + 1 + 3], &f_GHz, sizeof(float)); // orig_index of origin 1, name length byte, "TX2"
        std::ofstream out(fn, std::ios::binary);
        out.write(data.data(), (std::streamsize)data.size());
    }
    std::ifstream f(fn, std::ios::binary);
    auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
    REQUIRE(cache.orig_freq.n_elem == 3);
    CHECK(cache.orig_freq(0) == 3.75e9f);
    CHECK(cache.orig_freq(1) == 2.5e9f);
    CHECK(cache.freq(0) == 3.75e9f); // the first origin

    // The path gain follows the free-space loss at the origin's frequency: (3.75 / 2.5)^2 = 2.25 times that of the
    // unpatched file for origin 1, unchanged for origin 0
    for (arma::uword i_orig : {0, 1})
    {
        qrt_read_out a, b, ref;
        a.read(fn, {0, 3}, i_orig, false);
        b.read(fn, {0, 3}, i_orig, false, &f, &cache);
        ref.read("tests/data/test.qrt", {0, 3}, i_orig, false);
        CHECK(a.cf(0) == (i_orig == 0 ? (double)3.75e9f : (double)2.5e9f));
        CHECK(arma::approx_equal(a.cf, b.cf, "absdiff", 0.0));
        for (size_t k = 0; k < 2; ++k)
        {
            CHECK(arma::approx_equal(a.pg[k], b.pg[k], "absdiff", 0.0));
            CHECK(arma::approx_equal(a.pg[k], ref.pg[k] * (i_orig == 0 ? 1.0 : 2.25), "reldiff", 1e-6));
        }
    }
    f.close();
    std::remove(fn.c_str());
}
