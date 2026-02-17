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

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <filesystem>

// QDANT is XML text, so roundtrip precision is limited by text formatting.
// Use a tolerance that accounts for this.
static const double tol_d = 1.0e-5;
static const float tol_f = 1.0e-4f;

TEST_CASE("QDANT Multi - Write/Read roundtrip with speaker model")
{
    std::string fn = "test_speaker_roundtrip.qdant";

    // Generate a small speaker model: 5 frequencies, coarse grid
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "hemisphere",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 5);

    // Write to file
    REQUIRE_NOTHROW(quadriga_lib::qdant_write_multi(fn, spk));
    CHECK(std::filesystem::exists(fn));

    // Read back
    arma::u32_mat layout;
    auto spk_read = quadriga_lib::qdant_read_multi<double>(fn, &layout);

    // Same number of entries
    REQUIRE(spk_read.size() == spk.size());

    // Layout should be [n_freq, 1] with entries 1...n_freq
    REQUIRE(layout.n_rows == 5);
    REQUIRE(layout.n_cols == 1);
    for (arma::uword i = 0; i < 5; ++i)
        CHECK(layout(i, 0) == (unsigned)(i + 1));

    // Compare each frequency entry
    for (size_t i = 0; i < spk.size(); ++i)
    {
        INFO("Frequency index: " << i);

        // Grid sizes must match exactly
        CHECK(spk_read[i].n_azimuth() == spk[i].n_azimuth());
        CHECK(spk_read[i].n_elevation() == spk[i].n_elevation());
        CHECK(spk_read[i].n_elements() == spk[i].n_elements());
        CHECK(spk_read[i].n_ports() == spk[i].n_ports());

        // Center frequency
        CHECK(std::abs(spk_read[i].center_frequency - spk[i].center_frequency) < tol_d);

        // Name
        CHECK(spk_read[i].name == spk[i].name);

        // Pattern data
        CHECK(arma::approx_equal(spk_read[i].e_theta_re, spk[i].e_theta_re, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].e_theta_im, spk[i].e_theta_im, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].e_phi_re, spk[i].e_phi_re, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].e_phi_im, spk[i].e_phi_im, "absdiff", tol_d));

        // Grids
        CHECK(arma::approx_equal(spk_read[i].azimuth_grid, spk[i].azimuth_grid, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].elevation_grid, spk[i].elevation_grid, "absdiff", tol_d));

        // Element positions
        CHECK(arma::approx_equal(spk_read[i].element_pos, spk[i].element_pos, "absdiff", tol_d));

        // Coupling
        CHECK(arma::approx_equal(spk_read[i].coupling_re, spk[i].coupling_re, "absdiff", tol_d));
        CHECK(arma::approx_equal(spk_read[i].coupling_im, spk[i].coupling_im, "absdiff", tol_d));
    }

    // Cleanup
    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Single entry roundtrip")
{
    std::string fn = "test_speaker_single.qdant";

    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    quadriga_lib::qdant_write_multi(fn, spk);

    arma::u32_mat layout;
    auto spk_read = quadriga_lib::qdant_read_multi<double>(fn, &layout);

    REQUIRE(spk_read.size() == 1);
    REQUIRE(layout.n_rows == 1);
    REQUIRE(layout.n_cols == 1);
    CHECK(layout(0, 0) == 1);

    CHECK(std::abs(spk_read[0].center_frequency - 1000.0) < tol_d);
    CHECK(arma::approx_equal(spk_read[0].e_theta_re, spk[0].e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Float template roundtrip")
{
    std::string fn = "test_speaker_float.qdant";

    arma::fvec freqs = {200.0f, 2000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("horn", 0.05f, 80.0f, 12000.0f,
                                                      12.0f, 12.0f, 85.0f, "monopole",
                                                      0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);
    REQUIRE(spk.size() == 2);

    quadriga_lib::qdant_write_multi(fn, spk);
    auto spk_read = quadriga_lib::qdant_read_multi<float>(fn);

    REQUIRE(spk_read.size() == 2);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        CHECK(std::abs(spk_read[i].center_frequency - spk[i].center_frequency) < tol_f);
        CHECK(arma::approx_equal(spk_read[i].e_theta_re, spk[i].e_theta_re, "absdiff", tol_f));
    }

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Overwrite existing file")
{
    std::string fn = "test_speaker_overwrite.qdant";

    // Write a 3-entry file
    arma::vec freqs3 = {100.0, 500.0, 1000.0};
    auto spk3 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                        12.0, 12.0, 85.0, "monopole",
                                                        0.0, 0.0, 0.0, 0.15, 0.25, freqs3, 10.0);
    quadriga_lib::qdant_write_multi(fn, spk3);

    auto read3 = quadriga_lib::qdant_read_multi<double>(fn);
    REQUIRE(read3.size() == 3);

    // Overwrite with a 2-entry file
    arma::vec freqs2 = {2000.0, 8000.0};
    auto spk2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                        12.0, 12.0, 85.0, "monopole",
                                                        0.0, 0.0, 0.0, 0.15, 0.25, freqs2, 10.0);
    quadriga_lib::qdant_write_multi(fn, spk2);

    auto read2 = quadriga_lib::qdant_read_multi<double>(fn);

    // Should have 2 entries, not 3 or 5
    REQUIRE(read2.size() == 2);
    CHECK(std::abs(read2[0].center_frequency - 2000.0) < tol_d);
    CHECK(std::abs(read2[1].center_frequency - 8000.0) < tol_d);

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - All driver/radiation types survive roundtrip")
{
    std::string fn = "test_speaker_all_types.qdant";

    std::vector<std::string> drivers = {"piston", "horn", "omni"};
    std::vector<std::string> radiations = {"monopole", "hemisphere", "dipole", "cardioid"};
    arma::vec freqs = {500.0, 2000.0};

    for (const auto &drv : drivers)
    {
        for (const auto &rad : radiations)
        {
            INFO("driver=" << drv << " radiation=" << rad);

            auto spk = quadriga_lib::generate_speaker<double>(drv, 0.05, 80.0, 12000.0,
                                                               12.0, 12.0, 85.0, rad,
                                                               0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

            quadriga_lib::qdant_write_multi(fn, spk);
            auto spk_read = quadriga_lib::qdant_read_multi<double>(fn);

            REQUIRE(spk_read.size() == spk.size());

            for (size_t i = 0; i < spk.size(); ++i)
            {
                CHECK(arma::approx_equal(spk_read[i].e_theta_re, spk[i].e_theta_re, "absdiff", tol_d));
                CHECK(arma::approx_equal(spk_read[i].e_theta_im, spk[i].e_theta_im, "absdiff", tol_d));
            }

            std::filesystem::remove(fn);
        }
    }
}

TEST_CASE("QDANT Multi - Roundtrip preserves frequency ordering")
{
    std::string fn = "test_speaker_ordering.qdant";

    // Non-uniform frequency spacing
    arma::vec freqs = {63.0, 250.0, 1000.0, 4000.0, 16000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 30.0, 20000.0,
                                                       12.0, 12.0, 85.0, "hemisphere",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::qdant_write_multi(fn, spk);
    auto spk_read = quadriga_lib::qdant_read_multi<double>(fn);

    REQUIRE(spk_read.size() == 5);

    // Frequencies should come back in the same order
    for (size_t i = 0; i < 5; ++i)
        CHECK(std::abs(spk_read[i].center_frequency - freqs[i]) < tol_d);

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Read with existing qdant_read for individual entries")
{
    std::string fn = "test_speaker_compat.qdant";

    arma::vec freqs = {100.0, 1000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "hemisphere",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::qdant_write_multi(fn, spk);

    // Read individual entries with the existing single-entry qdant_read
    for (size_t i = 0; i < spk.size(); ++i)
    {
        unsigned id = (unsigned)(i + 1);
        auto ant = quadriga_lib::qdant_read<double>(fn, id);

        CHECK(std::abs(ant.center_frequency - spk[i].center_frequency) < tol_d);
        CHECK(arma::approx_equal(ant.e_theta_re, spk[i].e_theta_re, "absdiff", tol_d));
        CHECK(ant.name == spk[i].name);
    }

    // Compare: qdant_read_multi result should match individual reads
    auto spk_multi = quadriga_lib::qdant_read_multi<double>(fn);
    REQUIRE(spk_multi.size() == 3);

    for (size_t i = 0; i < 3; ++i)
    {
        auto ant_single = quadriga_lib::qdant_read<double>(fn, (unsigned)(i + 1));
        CHECK(arma::approx_equal(spk_multi[i].e_theta_re, ant_single.e_theta_re, "absdiff", 1e-12));
        CHECK(std::abs(spk_multi[i].center_frequency - ant_single.center_frequency) < 1e-12);
    }

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Write with auto-generated frequencies")
{
    std::string fn = "test_speaker_auto_freq.qdant";

    // Use default (auto third-octave) frequency generation
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 200.0, 8000.0,
                                                       12.0, 12.0, 85.0, "hemisphere");
    CHECK(spk.size() > 5); // Should have many third-octave bands

    quadriga_lib::qdant_write_multi(fn, spk);
    auto spk_read = quadriga_lib::qdant_read_multi<double>(fn);

    REQUIRE(spk_read.size() == spk.size());

    // Spot-check first and last entries
    CHECK(std::abs(spk_read.front().center_frequency - spk.front().center_frequency) < tol_d);
    CHECK(std::abs(spk_read.back().center_frequency - spk.back().center_frequency) < tol_d);
    CHECK(arma::approx_equal(spk_read.front().e_theta_re, spk.front().e_theta_re, "absdiff", tol_d));
    CHECK(arma::approx_equal(spk_read.back().e_theta_re, spk.back().e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Write empty vector throws")
{
    std::string fn = "test_speaker_empty.qdant";
    std::vector<quadriga_lib::arrayant<double>> empty_vec;

    REQUIRE_THROWS_AS(quadriga_lib::qdant_write_multi(fn, empty_vec), std::invalid_argument);

    // File should not have been created
    CHECK(!std::filesystem::exists(fn));
}

TEST_CASE("QDANT Multi - Write empty filename throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    REQUIRE_THROWS_AS(quadriga_lib::qdant_write_multi("", spk), std::invalid_argument);
}

TEST_CASE("QDANT Multi - Read empty filename throws")
{
    REQUIRE_THROWS_AS(quadriga_lib::qdant_read_multi<double>(""), std::invalid_argument);
}

TEST_CASE("QDANT Multi - Read non-existent file throws")
{
    REQUIRE_THROWS_AS(quadriga_lib::qdant_read_multi<double>("does_not_exist_12345.qdant"), std::invalid_argument);
}

TEST_CASE("QDANT Multi - Read without layout pointer")
{
    std::string fn = "test_speaker_no_layout.qdant";

    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::qdant_write_multi(fn, spk);

    // Read without requesting layout (nullptr default)
    auto spk_read = quadriga_lib::qdant_read_multi<double>(fn);

    REQUIRE(spk_read.size() == 2);
    CHECK(std::abs(spk_read[0].center_frequency - 500.0) < tol_d);
    CHECK(std::abs(spk_read[1].center_frequency - 2000.0) < tol_d);

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Non-speaker arrayant roundtrip")
{
    // Verify that write/read_multi works with any arrayant, not just speaker models
    std::string fn = "test_generic_multi.qdant";

    std::vector<quadriga_lib::arrayant<double>> vec;

    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    omni.center_frequency = 1000.0;
    vec.push_back(omni);

    auto dipole = quadriga_lib::generate_arrayant_dipole<double>(10.0);
    dipole.center_frequency = 2000.0;
    vec.push_back(dipole);

    auto hw = quadriga_lib::generate_arrayant_half_wave_dipole<double>(10.0);
    hw.center_frequency = 3000.0;
    vec.push_back(hw);

    quadriga_lib::qdant_write_multi(fn, vec);
    auto vec_read = quadriga_lib::qdant_read_multi<double>(fn);

    REQUIRE(vec_read.size() == 3);

    CHECK(vec_read[0].name == "omni");
    CHECK(vec_read[1].name == "dipole");
    CHECK(vec_read[2].name == "half-wave-dipole");

    CHECK(std::abs(vec_read[0].center_frequency - 1000.0) < tol_d);
    CHECK(std::abs(vec_read[1].center_frequency - 2000.0) < tol_d);
    CHECK(std::abs(vec_read[2].center_frequency - 3000.0) < tol_d);

    CHECK(arma::approx_equal(vec_read[0].e_theta_re, omni.e_theta_re, "absdiff", tol_d));
    CHECK(arma::approx_equal(vec_read[1].e_theta_re, dipole.e_theta_re, "absdiff", tol_d));
    CHECK(arma::approx_equal(vec_read[2].e_theta_re, hw.e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}

TEST_CASE("QDANT Multi - Large number of entries")
{
    std::string fn = "test_speaker_many.qdant";

    // Generate many frequency samples via auto third-octave over a wide range
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 30.0, 16000.0,
                                                       12.0, 12.0, 85.0, "monopole",
                                                       0.0, 0.0, 0.0, 0.15, 0.25, {}, 10.0);

    CHECK(spk.size() >= 20); // Expect ~28 third-octave bands

    quadriga_lib::qdant_write_multi(fn, spk);

    arma::u32_mat layout;
    auto spk_read = quadriga_lib::qdant_read_multi<double>(fn, &layout);

    REQUIRE(spk_read.size() == spk.size());
    REQUIRE(layout.n_rows == (arma::uword)spk.size());
    REQUIRE(layout.n_cols == 1);

    // Verify layout IDs are sequential 1..N
    for (arma::uword i = 0; i < layout.n_rows; ++i)
        CHECK(layout(i, 0) == (unsigned)(i + 1));

    // Spot-check middle entry
    size_t mid = spk.size() / 2;
    CHECK(std::abs(spk_read[mid].center_frequency - spk[mid].center_frequency) < tol_d);
    CHECK(arma::approx_equal(spk_read[mid].e_theta_re, spk[mid].e_theta_re, "absdiff", tol_d));

    std::filesystem::remove(fn);
}
