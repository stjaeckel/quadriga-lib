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

// Grid at 10 deg resolution: no_az = 37, no_el = 19
// Azimuth:   linspace(-pi, pi, 37)  → ia=0: -180°, ia=9: -90°, ia=18: 0°, ia=27: 90°, ia=36: 180°
// Elevation: linspace(-pi/2, pi/2, 19) → ie=0: -90°, ie=9: 0°, ie=18: 90°
// Forward (+x axis): ie=9, ia=18
// Rear (-x axis):    ie=9, ia=36
// Side +y (az=90°):  ie=9, ia=27
// Side -y (az=-90°): ie=9, ia=9
// Top (el=90°):      ie=18, ia=18

TEST_CASE("Speaker - Input validation")
{
    // Bad driver type
    REQUIRE_THROWS_AS(quadriga_lib::generate_speaker<float>("banana"), std::invalid_argument);

    // Bad radiation type
    REQUIRE_THROWS_AS(quadriga_lib::generate_speaker<float>("omni", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f, "banana"),
                      std::invalid_argument);
}

TEST_CASE("Speaker - Default call, basic structure")
{
    auto spk = quadriga_lib::generate_speaker<double>();

    // Auto-generated third-octave bands should produce multiple frequency samples
    CHECK(spk.size() > 10);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        // Default angular resolution is 5 deg
        CHECK(spk[i].n_azimuth() == 73);
        CHECK(spk[i].n_elevation() == 37);
        CHECK(spk[i].n_elements() == 1);
        CHECK(spk[i].n_ports() == 1);
        CHECK(spk[i].name == "speaker_piston");
        CHECK(spk[i].center_frequency > 0.0);

        // Validate arrayant structure
        std::string err = spk[i].is_valid(false);
        CHECK(err.empty());
    }
}

TEST_CASE("Speaker - Custom frequencies and angular resolution")
{
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    REQUIRE(spk.size() == 4);
    CHECK(spk[0].n_azimuth() == 37);
    CHECK(spk[0].n_elevation() == 19);

    // Verify center frequencies match input
    CHECK(std::abs(spk[0].center_frequency - 100.0) < 1e-6);
    CHECK(std::abs(spk[1].center_frequency - 500.0) < 1e-6);
    CHECK(std::abs(spk[2].center_frequency - 1000.0) < 1e-6);
    CHECK(std::abs(spk[3].center_frequency - 5000.0) < 1e-6);
}

TEST_CASE("Speaker - Float template works")
{
    arma::fvec freqs = {1000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("omni", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                      "monopole", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);

    REQUIRE(spk.size() == 1);
    CHECK(spk[0].n_azimuth() == 37);
    CHECK(spk[0].n_elevation() == 19);
    CHECK(std::abs(spk[0].center_frequency - 1000.0f) < 1e-3f);

    // On-axis value should be near 1.0 (passband, default sensitivity)
    float on_axis = spk[0].e_theta_re(9, 18, 0);
    CHECK(std::abs(on_axis - 1.0f) < 0.01f);
}

TEST_CASE("Speaker - Omni + Monopole: flat pattern")
{
    // Omni driver + monopole radiation = uniform pattern at all angles
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    // At 1000 Hz with default params, freq_response ≈ 1.0, sens_lin = 1.0
    // Every grid point should have the same amplitude ≈ 1.0
    double on_axis = spk[0].e_theta_re(9, 18, 0);  // az=0, el=0
    double rear = spk[0].e_theta_re(9, 36, 0);      // az=180, el=0
    double side = spk[0].e_theta_re(9, 27, 0);      // az=90, el=0
    double top = spk[0].e_theta_re(18, 18, 0);      // az=0, el=90

    CHECK(std::abs(on_axis - 1.0) < 0.001);
    CHECK(std::abs(rear - 1.0) < 0.001);
    CHECK(std::abs(side - 1.0) < 0.001);
    CHECK(std::abs(top - 1.0) < 0.001);

    // Check the entire pattern is uniform
    double val_max = spk[0].e_theta_re.max();
    double val_min = spk[0].e_theta_re.min();
    CHECK(std::abs(val_max - val_min) < 0.001);

    // Imaginary part and phi components should be zero
    CHECK(spk[0].e_theta_im.max() < 1e-10);
    CHECK(spk[0].e_phi_re.max() < 1e-10);
    CHECK(spk[0].e_phi_im.max() < 1e-10);
}

TEST_CASE("Speaker - Omni + Dipole: figure-8 pattern")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "dipole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    // Dipole: R = |cos(theta_off)|, rear has phase flip
    // On-axis (ct=1): positive, amplitude = freq_response ≈ 1.0
    double on_axis = spk[0].e_theta_re(9, 18, 0);
    CHECK(std::abs(on_axis - 1.0) < 0.001);

    // Rear (ct=-1): negative sign (phase flip), amplitude = 1.0
    double rear = spk[0].e_theta_re(9, 36, 0);
    CHECK(std::abs(rear + 1.0) < 0.001); // Should be -1.0

    // Side (ct=0): null
    double side = spk[0].e_theta_re(9, 27, 0);
    CHECK(std::abs(side) < 1e-10);

    // Also check opposite side
    double side_neg = spk[0].e_theta_re(9, 9, 0);
    CHECK(std::abs(side_neg) < 1e-10);

    // Top (az=0, el=90): ct = cos(90)*cos(0) = 0 → null
    double top = spk[0].e_theta_re(18, 18, 0);
    CHECK(std::abs(top) < 1e-10);
}

TEST_CASE("Speaker - Omni + Cardioid: null at rear")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "cardioid", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    // Cardioid: R = 0.5*(1 + cos(theta_off))
    // On-axis (ct=1): R = 1.0
    double on_axis = spk[0].e_theta_re(9, 18, 0);
    CHECK(std::abs(on_axis - 1.0) < 0.001);

    // Side (ct=0): R = 0.5
    double side = spk[0].e_theta_re(9, 27, 0);
    CHECK(std::abs(side - 0.5) < 0.001);

    // Rear (ct=-1): R = 0.0
    double rear = spk[0].e_theta_re(9, 36, 0);
    CHECK(std::abs(rear) < 1e-10);

    // Pattern should be non-negative everywhere (no phase flip in cardioid)
    CHECK(spk[0].e_theta_re.min() >= -1e-10);
}

TEST_CASE("Speaker - Omni + Hemisphere: frequency-dependent front/back ratio")
{
    // Baffle params: width=0.15, height=0.25
    // baffle_mean = sqrt(0.15*0.25) = sqrt(0.0375) = 0.19365 m
    // f_baffle = 344 / (pi * 0.19365) = 565.5 Hz
    double f_baffle = 344.0 / (arma::datum::pi * std::sqrt(0.15 * 0.25));

    arma::vec freqs = {50.0, 1000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 20.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 3);

    // --- 50 Hz: well below f_baffle, nearly omnidirectional ---
    // alpha = (50/565.5)^2 / (1 + (50/565.5)^2) = 0.00781 / 1.00781 = 0.00775
    double alpha_50 = std::pow(50.0 / f_baffle, 2.0) / (1.0 + std::pow(50.0 / f_baffle, 2.0));
    double rear_50_expected = (1.0 - alpha_50); // ct=-1: (1-a) + a*0.5*(1-1) = (1-a)
    double on_axis_50 = spk[0].e_theta_re(9, 18, 0);
    double rear_50 = spk[0].e_theta_re(9, 36, 0);
    double ratio_50 = rear_50 / on_axis_50;
    CHECK(ratio_50 > 0.99); // Nearly uniform at low freq

    // --- 1000 Hz: above f_baffle, noticeable front/back difference ---
    double alpha_1k = std::pow(1000.0 / f_baffle, 2.0) / (1.0 + std::pow(1000.0 / f_baffle, 2.0));
    double on_axis_1k = spk[1].e_theta_re(9, 18, 0);
    double rear_1k = spk[1].e_theta_re(9, 36, 0);
    double ratio_1k = rear_1k / on_axis_1k;
    double expected_ratio_1k = 1.0 - alpha_1k;
    CHECK(std::abs(ratio_1k - expected_ratio_1k) < 0.001);

    // On-axis should always be unity (times freq_response)
    CHECK(on_axis_1k > 0.99); // Near passband center

    // --- 10000 Hz: well above f_baffle, strong front/back difference ---
    double alpha_10k = std::pow(10000.0 / f_baffle, 2.0) / (1.0 + std::pow(10000.0 / f_baffle, 2.0));
    double on_axis_10k = spk[2].e_theta_re(9, 18, 0);
    double rear_10k = spk[2].e_theta_re(9, 36, 0);
    double ratio_10k = rear_10k / on_axis_10k;

    CHECK(ratio_10k < 0.05); // Very small rear radiation at high freq
    CHECK(ratio_10k > 0.0);  // But not zero (asymptotic)
}

TEST_CASE("Speaker - Piston + Monopole: low freq near-omni")
{
    // At 200 Hz, radius=0.05: ka = 2*pi*200/344 * 0.05 = 0.183
    // Very small ka → pattern is nearly omnidirectional
    arma::vec freqs = {200.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);
    double side = spk[0].e_theta_re(9, 27, 0);
    double rear = spk[0].e_theta_re(9, 36, 0);

    // All should be close to on-axis value (small ka → nearly uniform)
    CHECK(std::abs(side / on_axis - 1.0) < 0.01);
    CHECK(std::abs(rear / on_axis - 1.0) < 0.01);
}

TEST_CASE("Speaker - Piston + Monopole: high freq directional")
{
    // At 10000 Hz, radius=0.05: ka = 2*pi*10000/344 * 0.05 = 9.13
    // Large ka → strongly directional
    arma::vec freqs = {10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);
    double side = spk[0].e_theta_re(9, 27, 0);

    // On-axis should be the maximum
    CHECK(on_axis > 0.0);

    // Side (90° off-axis) should be much smaller than on-axis
    CHECK(std::abs(side / on_axis) < 0.15);

    // Pattern should be rotationally symmetric: az=90° and el=90° should give similar values
    // (both are 90° off-axis from +x)
    double top = spk[0].e_theta_re(18, 18, 0); // el=90, az=0: cos_off = cos(90)*cos(0) = 0 → 90° off axis
    CHECK(std::abs(side - top) < 1e-6);

    // On-axis is maximum of entire pattern
    double pat_max = spk[0].e_theta_re.max();
    CHECK(std::abs(on_axis - pat_max) < 1e-10);
}

TEST_CASE("Speaker - Piston directivity increases with frequency")
{
    // Compare on-axis to side ratio at two frequencies
    arma::vec freqs = {500.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 2);

    // At 500 Hz (ka=0.457): nearly omni
    double ratio_500 = spk[0].e_theta_re(9, 27, 0) / spk[0].e_theta_re(9, 18, 0);

    // At 5000 Hz (ka=4.57): noticeably directional
    double ratio_5k = spk[1].e_theta_re(9, 27, 0) / spk[1].e_theta_re(9, 18, 0);

    // Higher frequency → more directional → lower side/on-axis ratio
    CHECK(ratio_500 > ratio_5k);
    CHECK(ratio_500 > 0.95);  // Nearly omni at 500 Hz
    CHECK(ratio_5k < 0.6);    // Notably directional at 5 kHz
}

TEST_CASE("Speaker - Horn + Monopole: controlled directivity")
{
    // Horn with radius=0.05 → auto horn_control_freq ≈ 1094.7 Hz
    // Auto coverage: 90° horizontal, 60° vertical
    double f_ctrl = 344.0 / (2.0 * arma::datum::pi * 0.05);

    arma::vec freqs = {100.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 2);

    // --- 100 Hz: well below horn_control_freq → nearly omnidirectional ---
    double blend_100 = std::pow(100.0 / f_ctrl, 2.0) / (1.0 + std::pow(100.0 / f_ctrl, 2.0));
    double on_axis_100 = spk[0].e_theta_re(9, 18, 0);
    double side_100 = spk[0].e_theta_re(9, 27, 0);

    // At side: D = (1-blend) + blend*0 = (1-blend), so ratio = (1-blend)
    double expected_side_ratio = 1.0 - blend_100;
    CHECK(std::abs(side_100 / on_axis_100 - expected_side_ratio) < 0.001);
    CHECK(side_100 / on_axis_100 > 0.99); // Nearly omni

    // --- 5000 Hz: well above horn_control_freq → controlled directivity ---
    double blend_5k = std::pow(5000.0 / f_ctrl, 2.0) / (1.0 + std::pow(5000.0 / f_ctrl, 2.0));
    double on_axis_5k = spk[1].e_theta_re(9, 18, 0);
    double side_5k = spk[1].e_theta_re(9, 27, 0);
    double rear_5k = spk[1].e_theta_re(9, 36, 0);

    // Side and rear should be very low at high freq (horn D_h=0 for az>=90°)
    double expected_side_5k = (1.0 - blend_5k); // Only the omni residual
    CHECK(std::abs(side_5k / on_axis_5k - expected_side_5k) < 0.002);
    CHECK(side_5k / on_axis_5k < 0.06); // Very directional

    // Rear should equal side (both have D_h = 0 since cos(az) <= 0)
    CHECK(std::abs(rear_5k - side_5k) < 1e-6);
}

TEST_CASE("Speaker - Horn coverage angles check")
{
    // Horn with explicit 90° horizontal, 60° vertical coverage
    // horn_exp_h = log(0.5)/log(cos(45°)) = 2.0
    // At half-angle (45°): cos^2(45°) = 0.5 amplitude
    // We test at very high frequency where blend → 1

    arma::vec freqs = {10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 90.0, 60.0, 500.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);

    // At az=40° (ia=22), el=0°: inside half-angle (45°) → amplitude > 0.5 of on-axis
    double at_40deg = spk[0].e_theta_re(9, 22, 0);
    double amp_ratio_40 = at_40deg / on_axis;
    CHECK(amp_ratio_40 > 0.5);

    // At az=50° (ia=23), el=0°: outside half-angle → amplitude < 0.5 of on-axis
    double at_50deg = spk[0].e_theta_re(9, 23, 0);
    double amp_ratio_50 = at_50deg / on_axis;
    CHECK(amp_ratio_50 < 0.5);
}

TEST_CASE("Speaker - Horn auto-derivation of parameters")
{
    // When horn params are 0, they should be auto-derived from radius
    // horn_control_freq = c / (2*pi*radius) = 344 / (2*pi*0.1) = 547.4 Hz
    // hor_coverage = 90°, ver_coverage = 60°

    arma::vec freqs = {5000.0};

    // Call with all zeros (auto-derive)
    auto spk_auto = quadriga_lib::generate_speaker<double>("horn", 0.1, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                            "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Call with explicit values that match the auto-derived ones
    double expected_ctrl = 344.0 / (2.0 * arma::datum::pi * 0.1);
    auto spk_explicit = quadriga_lib::generate_speaker<double>("horn", 0.1, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                                "monopole", 90.0, 60.0, expected_ctrl, 0.15, 0.25, freqs, 10.0);

    REQUIRE(spk_auto.size() == 1);
    REQUIRE(spk_explicit.size() == 1);

    // Patterns should be identical
    CHECK(arma::approx_equal(spk_auto[0].e_theta_re.slice(0), spk_explicit[0].e_theta_re.slice(0), "absdiff", 1e-10));
}

TEST_CASE("Speaker - Frequency response: -3 dB at cutoff frequencies")
{
    // At cutoff frequency, Butterworth response gives -3 dB (amplitude = 1/sqrt(2))
    // Use omni + monopole to isolate the frequency response from directivity
    arma::vec freqs = {80.0, 1000.0, 12000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 3);

    // On-axis values (all grid points are equal for omni+monopole)
    double val_at_80 = spk[0].e_theta_re(9, 18, 0);
    double val_at_1k = spk[1].e_theta_re(9, 18, 0);
    double val_at_12k = spk[2].e_theta_re(9, 18, 0);

    // Passband (1000 Hz) should be near 1.0
    CHECK(std::abs(val_at_1k - 1.0) < 0.001);

    // At lower cutoff (80 Hz): -3 dB → amplitude = 1/sqrt(2) of passband
    CHECK(std::abs(val_at_80 / val_at_1k - 1.0 / std::sqrt(2.0)) < 0.001);

    // At upper cutoff (12000 Hz): -3 dB → amplitude = 1/sqrt(2) of passband
    CHECK(std::abs(val_at_12k / val_at_1k - 1.0 / std::sqrt(2.0)) < 0.001);
}

TEST_CASE("Speaker - Frequency response: rolloff slope")
{
    // At one octave below lower_cutoff (40 Hz), 12 dB/oct rolloff:
    // ratio_low = 80/40 = 2, pow(2, 4) = 16, H = 1/sqrt(1+16) = 1/sqrt(17) = 0.2425
    arma::vec freqs = {40.0, 1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 2);

    double val_at_40 = spk[0].e_theta_re(9, 18, 0);
    double val_at_1k = spk[1].e_theta_re(9, 18, 0);

    double expected_ratio = 1.0 / std::sqrt(1.0 + std::pow(2.0, 4.0)); // 1/sqrt(17)
    CHECK(std::abs(val_at_40 / val_at_1k - expected_ratio) < 0.001);
}

TEST_CASE("Speaker - Frequency response: different rolloff slopes")
{
    // 24 dB/oct = 4th order Butterworth (n = 24/6 = 4)
    // At lower_cutoff: H = 1/sqrt(1 + 1^8) = 1/sqrt(2) = 0.7071 (same -3dB point)
    // At one octave below (40 Hz): ratio=2, pow(2, 8) = 256, H = 1/sqrt(257) = 0.0624
    arma::vec freqs = {40.0, 80.0, 1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 24.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 3);

    double val_40 = spk[0].e_theta_re(9, 18, 0);
    double val_80 = spk[1].e_theta_re(9, 18, 0);
    double val_1k = spk[2].e_theta_re(9, 18, 0);

    // -3dB at cutoff still holds
    CHECK(std::abs(val_80 / val_1k - 1.0 / std::sqrt(2.0)) < 0.001);

    // Steeper rolloff: one octave below should be much lower than with 12 dB/oct
    double expected_ratio_24 = 1.0 / std::sqrt(1.0 + std::pow(2.0, 8.0)); // 1/sqrt(257)
    CHECK(std::abs(val_40 / val_1k - expected_ratio_24) < 0.001);
}

TEST_CASE("Speaker - Sensitivity scaling")
{
    arma::vec freqs = {1000.0};

    // Default sensitivity: 85 dB → sens_lin = 1.0
    auto spk_85 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                          "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Higher sensitivity: 95 dB → sens_lin = 10^(10/20) = sqrt(10) ≈ 3.1623
    auto spk_95 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 95.0,
                                                          "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    REQUIRE(spk_85.size() == 1);
    REQUIRE(spk_95.size() == 1);

    double val_85 = spk_85[0].e_theta_re(9, 18, 0);
    double val_95 = spk_95[0].e_theta_re(9, 18, 0);

    CHECK(std::abs(val_85 - 1.0) < 0.001);
    CHECK(std::abs(val_95 - std::sqrt(10.0)) < 0.001);
    CHECK(std::abs(val_95 / val_85 - std::sqrt(10.0)) < 0.001);
}

TEST_CASE("Speaker - Piston + Dipole combined")
{
    // Combines piston beaming with figure-8 radiation
    arma::vec freqs = {5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "dipole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);
    double side = spk[0].e_theta_re(9, 27, 0);
    double rear = spk[0].e_theta_re(9, 36, 0);

    // On-axis should be positive and maximum
    CHECK(on_axis > 0.0);

    // Side (az=90°): dipole null regardless of piston directivity
    CHECK(std::abs(side) < 1e-10);

    // Rear: dipole gives |ct|=1 but phase flip, piston gives 2*J1(ka*0)/ka*0 = 1
    // (rear is also on the piston axis with st = sin(180°) = 0)
    // Wait: at rear, ct = -1 but st = sin(theta_off) = sin(180°) = 0
    // So piston D = piston_directivity(ka*0) = 1.0, dipole R = |-1| = 1 with flip
    CHECK(std::abs(rear + on_axis) < 0.001); // Equal magnitude, opposite sign
}

TEST_CASE("Speaker - Piston + Cardioid combined")
{
    arma::vec freqs = {5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "cardioid", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);
    double rear = spk[0].e_theta_re(9, 36, 0);

    // On-axis: piston=1, cardioid=1 → positive
    CHECK(on_axis > 0.0);

    // Rear: cardioid null → zero regardless of piston
    CHECK(std::abs(rear) < 1e-10);

    // All values should be non-negative (cardioid has no phase flip)
    CHECK(spk[0].e_theta_re.min() >= -1e-10);
}

TEST_CASE("Speaker - Horn + Cardioid combined")
{
    arma::vec freqs = {5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "cardioid", 90.0, 60.0, 500.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);
    double rear = spk[0].e_theta_re(9, 36, 0);

    // On-axis should be positive
    CHECK(on_axis > 0.0);

    // Rear: cardioid null → zero
    CHECK(std::abs(rear) < 1e-10);

    // No negative values
    CHECK(spk[0].e_theta_re.min() >= -1e-10);
}

TEST_CASE("Speaker - Horn + Hemisphere combined")
{
    arma::vec freqs = {5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 90.0, 60.0, 500.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    double on_axis = spk[0].e_theta_re(9, 18, 0);
    double side = spk[0].e_theta_re(9, 27, 0);
    double rear = spk[0].e_theta_re(9, 36, 0);

    // On-axis should be the largest value
    CHECK(on_axis > side);
    CHECK(on_axis > rear);

    // Rear should be small but non-zero (hemisphere doesn't null out)
    CHECK(rear > 0.0);
    CHECK(rear < on_axis * 0.1);

    // No negative values
    CHECK(spk[0].e_theta_re.min() >= -1e-10);
}

TEST_CASE("Speaker - All 12 driver x radiation combinations produce valid output")
{
    std::vector<std::string> drivers = {"piston", "horn", "omni"};
    std::vector<std::string> radiations = {"monopole", "hemisphere", "dipole", "cardioid"};

    arma::vec freqs = {500.0, 2000.0};

    for (const auto &drv : drivers)
    {
        for (const auto &rad : radiations)
        {
            auto spk = quadriga_lib::generate_speaker<double>(drv, 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                               rad, 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

            REQUIRE(spk.size() == 2);

            for (size_t i = 0; i < spk.size(); ++i)
            {
                // Validate structure
                std::string err = spk[i].is_valid(false);
                INFO("driver=" << drv << " radiation=" << rad << " freq_idx=" << i << " error=" << err);
                CHECK(err.empty());

                // Name should reflect driver type
                CHECK(spk[i].name == "speaker_" + drv);

                // On-axis should be positive for all combinations
                double on_axis = spk[i].e_theta_re(9, 18, 0);
                CHECK(on_axis > 0.0);

                // Pattern should contain no NaN or Inf
                CHECK(spk[i].e_theta_re.is_finite());
                CHECK(spk[i].e_theta_im.is_finite());
            }
        }
    }
}

TEST_CASE("Speaker - Piston rotational symmetry")
{
    // Piston pattern should be rotationally symmetric around the forward axis
    arma::vec freqs = {3000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 1);

    // Points at the same off-axis angle should have the same value
    // az=90°, el=0° and az=0°, el=90° both give 90° off-axis from +x
    double at_az90 = spk[0].e_theta_re(9, 27, 0);   // az=90, el=0
    double at_el90 = spk[0].e_theta_re(18, 18, 0);  // az=0, el=90
    CHECK(std::abs(at_az90 - at_el90) < 1e-6);

    // az=-90°, el=0° should also match
    double at_az_neg90 = spk[0].e_theta_re(9, 9, 0);
    CHECK(std::abs(at_az90 - at_az_neg90) < 1e-6);

    // az=0°, el=-90° should also match
    double at_el_neg90 = spk[0].e_theta_re(0, 18, 0);
    CHECK(std::abs(at_az90 - at_el_neg90) < 1e-6);

    // Check at 30° off-axis in two different directions
    // az=30° (ia=21), el=0° (ie=9): cos_off = cos(0)*cos(30°) = 0.866
    // az=0° (ia=18), el=30° (ie=12): cos_off = cos(30°)*cos(0°) = 0.866
    double at_az30 = spk[0].e_theta_re(9, 21, 0);
    double at_el30 = spk[0].e_theta_re(12, 18, 0);
    CHECK(std::abs(at_az30 - at_el30) < 1e-6);
}

TEST_CASE("Speaker - Third-octave auto-generation covers passband")
{
    // Auto-generated frequencies should cover the range [lower_cutoff, upper_cutoff] and a bit beyond
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 200.0, 8000.0, 12.0, 12.0, 85.0,
                                                       "monopole");

    // Check that we have a reasonable number of samples
    CHECK(spk.size() >= 10);
    CHECK(spk.size() <= 40);

    // First frequency should be at or below lower_cutoff
    CHECK(spk.front().center_frequency <= 200.0);

    // Last frequency should be at or above upper_cutoff
    CHECK(spk.back().center_frequency >= 8000.0);

    // Frequencies should be monotonically increasing
    for (size_t i = 1; i < spk.size(); ++i)
        CHECK(spk[i].center_frequency > spk[i - 1].center_frequency);
}
