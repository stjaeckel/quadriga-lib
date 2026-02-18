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
    double on_axis = spk[0].e_theta_re(9, 18, 0); // az=0, el=0
    double rear = spk[0].e_theta_re(9, 36, 0);    // az=180, el=0
    double side = spk[0].e_theta_re(9, 27, 0);    // az=90, el=0
    double top = spk[0].e_theta_re(18, 18, 0);    // az=0, el=90

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
    CHECK(ratio_500 > 0.95); // Nearly omni at 500 Hz
    CHECK(ratio_5k < 0.6);   // Notably directional at 5 kHz
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
    double at_az90 = spk[0].e_theta_re(9, 27, 0);  // az=90, el=0
    double at_el90 = spk[0].e_theta_re(18, 18, 0); // az=0, el=90
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

// ================================================================================================
// is_valid_multi tests
// ================================================================================================

TEST_CASE("Speaker - is_valid_multi: generated speaker passes")
{
    // A freshly generated speaker model should always pass validation
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());

    // Quick check should also pass
    err = quadriga_lib::arrayant_is_valid_multi(spk, true);
    CHECK(err.empty());
}

TEST_CASE("Speaker - is_valid_multi: all driver/radiation combos pass")
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
            std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
            INFO("driver=" << drv << " radiation=" << rad << " error=" << err);
            CHECK(err.empty());
        }
    }
}

TEST_CASE("Speaker - is_valid_multi: float template passes")
{
    arma::fvec freqs = {500.0f, 2000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("piston", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                     "hemisphere", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - is_valid_multi: single entry passes")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - is_valid_multi: empty vector fails")
{
    std::vector<quadriga_lib::arrayant<double>> empty_vec;
    std::string err = quadriga_lib::arrayant_is_valid_multi(empty_vec);
    CHECK(!err.empty());
    CHECK(err.find("empty") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: mismatched azimuth grid size fails")
{
    // Generate two speakers with different angular resolutions
    arma::vec freqs = {1000.0};
    auto spk5 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
    auto spk10 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                        "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Combine them into one vector: entry 0 at 5°, entry 1 at 10°
    std::vector<quadriga_lib::arrayant<double>> mixed;
    mixed.push_back(spk5[0]);
    mixed.push_back(spk10[0]);

    std::string err = quadriga_lib::arrayant_is_valid_multi(mixed, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("elevation") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: mismatched elevation grid size fails")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Create a copy with a different elevation grid
    std::vector<quadriga_lib::arrayant<double>> vec;
    vec.push_back(spk[0]);

    quadriga_lib::arrayant<double> bad = spk[0];
    // Resize elevation grid to a different size (also resize pattern data to keep individual validity)
    arma::uword new_n_el = 10;
    bad.elevation_grid = arma::linspace<arma::vec>(-arma::datum::pi * 0.5, arma::datum::pi * 0.5, new_n_el);
    bad.e_theta_re.zeros(new_n_el, bad.n_azimuth(), 1);
    bad.e_theta_im.zeros(new_n_el, bad.n_azimuth(), 1);
    bad.e_phi_re.zeros(new_n_el, bad.n_azimuth(), 1);
    bad.e_phi_im.zeros(new_n_el, bad.n_azimuth(), 1);
    vec.push_back(bad);

    std::string err = quadriga_lib::arrayant_is_valid_multi(vec, false);

    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("elevation") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: shifted azimuth grid values fails")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Shift the azimuth grid of entry 1 by a tiny amount
    spk[1].azimuth_grid[0] += 0.001;

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("Azimuth grid values") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: shifted elevation grid values fails")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Shift the elevation grid of entry 1
    spk[1].elevation_grid[5] += 0.001;

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("Elevation grid values") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: mismatched element count fails")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Create a second entry with 2 elements by duplicating the pattern slice
    quadriga_lib::arrayant<double> two_elem = spk[0];
    arma::uword n_el = two_elem.n_elevation();
    arma::uword n_az = two_elem.n_azimuth();
    two_elem.e_theta_re = arma::join_slices(two_elem.e_theta_re, two_elem.e_theta_re);
    two_elem.e_theta_im = arma::join_slices(two_elem.e_theta_im, two_elem.e_theta_im);
    two_elem.e_phi_re = arma::join_slices(two_elem.e_phi_re, two_elem.e_phi_re);
    two_elem.e_phi_im = arma::join_slices(two_elem.e_phi_im, two_elem.e_phi_im);
    two_elem.element_pos.zeros(3, 2);
    two_elem.coupling_re.ones(2, 1);
    two_elem.coupling_im.zeros(2, 1);

    std::vector<quadriga_lib::arrayant<double>> vec;
    vec.push_back(spk[0]);
    vec.push_back(two_elem);

    std::string err = quadriga_lib::arrayant_is_valid_multi(vec, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("elements") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: modified element position fails")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Move the element position in entry 1
    spk[1].element_pos(0, 0) = 1.5;

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("position") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: mismatched coupling shape fails")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Change coupling to have 2 columns (2 ports) in entry 1
    spk[1].coupling_re.ones(1, 2);
    spk[1].coupling_im.zeros(1, 2);

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("oupling") != std::string::npos); // "Coupling" with capital or lowercase
}

TEST_CASE("Speaker - is_valid_multi: invalid individual entry caught")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Corrupt entry 1: clear the pattern data
    spk[1].e_theta_re.reset();

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: invalid first entry caught")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Corrupt entry 0
    spk[0].e_theta_re.reset();

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 0") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: error at middle entry")
{
    // Three entries, corrupt the middle one's grid
    arma::vec freqs = {200.0, 1000.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 3);

    // Shift elevation grid of the middle entry
    spk[1].elevation_grid[0] += 0.01;

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(!err.empty());
    CHECK(err.find("Entry 1") != std::string::npos);
    CHECK(err.find("Elevation grid values") != std::string::npos);
}

TEST_CASE("Speaker - is_valid_multi: different pattern values are allowed")
{
    // Pattern data SHOULD differ between frequencies; only structural properties must match
    arma::vec freqs = {500.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Patterns should actually differ (piston beams more at higher freq)
    CHECK(!arma::approx_equal(spk[0].e_theta_re, spk[1].e_theta_re, "absdiff", 1e-3));

    // But multi-validation should still pass
    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - is_valid_multi: different center frequencies are allowed")
{
    arma::vec freqs = {100.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    CHECK(std::abs(spk[0].center_frequency - spk[1].center_frequency) > 1000.0);

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

// ================================================================================================
// arrayant_set_element_pos_multi tests
// ================================================================================================

TEST_CASE("Speaker - set_element_pos_multi: update all elements")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk.size() == 2);

    arma::mat pos = arma::mat({1.5, -0.3, 0.7}).t();
    REQUIRE_NOTHROW(quadriga_lib::arrayant_set_element_pos_multi(spk, pos));

    for (size_t i = 0; i < spk.size(); ++i)
    {
        REQUIRE(spk[i].element_pos.n_rows == 3);
        REQUIRE(spk[i].element_pos.n_cols == 1);
        CHECK(std::abs(spk[i].element_pos(0, 0) - 1.5) < 1e-12);
        CHECK(std::abs(spk[i].element_pos(1, 0) + 0.3) < 1e-12);
        CHECK(std::abs(spk[i].element_pos(2, 0) - 0.7) < 1e-12);
    }
}

TEST_CASE("Speaker - set_element_pos_multi: update selected elements")
{
    // Build a 2-element model by concatenating two single-element speakers
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto spk = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    REQUIRE(spk[0].n_elements() == 2);

    // Update only element 1 (second element, 0-based)
    arma::mat pos1 = arma::mat({0.0, 0.0, 0.12}).t();
    arma::uvec idx = {1};
    quadriga_lib::arrayant_set_element_pos_multi(spk, pos1, idx);

    CHECK(std::abs(spk[0].element_pos(0, 0)) < 1e-12); // Element 0 unchanged
    CHECK(std::abs(spk[0].element_pos(1, 0)) < 1e-12);
    CHECK(std::abs(spk[0].element_pos(2, 0)) < 1e-12);
    CHECK(std::abs(spk[0].element_pos(0, 1)) < 1e-12); // Element 1 updated
    CHECK(std::abs(spk[0].element_pos(1, 1)) < 1e-12);
    CHECK(std::abs(spk[0].element_pos(2, 1) - 0.12) < 1e-12);
}

TEST_CASE("Speaker - set_element_pos_multi: empty vector throws")
{
    std::vector<quadriga_lib::arrayant<double>> empty_vec;
    arma::mat pos = arma::mat({0.0, 0.0, 0.0}).t();
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_set_element_pos_multi(empty_vec, pos), std::invalid_argument);
}

TEST_CASE("Speaker - set_element_pos_multi: wrong row count throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    arma::mat bad_pos(2, 1, arma::fill::zeros); // 2 rows instead of 3
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_set_element_pos_multi(spk, bad_pos), std::invalid_argument);
}

TEST_CASE("Speaker - set_element_pos_multi: wrong column count throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    arma::mat bad_pos(3, 5, arma::fill::zeros); // 5 columns but only 1 element
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_set_element_pos_multi(spk, bad_pos), std::invalid_argument);
}

TEST_CASE("Speaker - set_element_pos_multi: out-of-range index throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    arma::mat pos = arma::mat({0.0, 0.0, 0.0}).t();
    arma::uvec idx = {5}; // Only 1 element, index 5 is out of range
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_set_element_pos_multi(spk, pos, idx), std::invalid_argument);
}

TEST_CASE("Speaker - set_element_pos_multi: float template")
{
    arma::fvec freqs = {1000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("omni", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                     "monopole", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);
    arma::fmat pos = arma::fmat({0.5f, -0.5f, 1.0f}).t();
    REQUIRE_NOTHROW(quadriga_lib::arrayant_set_element_pos_multi(spk, pos));
    CHECK(std::abs(spk[0].element_pos(0, 0) - 0.5f) < 1e-6f);
}

// ================================================================================================
// arrayant_concat_multi tests
// ================================================================================================

TEST_CASE("Speaker - concat_multi: basic two-driver concatenation")
{
    arma::vec freqs = {500.0, 2000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    auto combined = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    REQUIRE(combined.size() == 2);
    for (size_t i = 0; i < combined.size(); ++i)
    {
        CHECK(combined[i].n_elements() == 2);
        CHECK(combined[i].n_ports() == 2);
        CHECK(combined[i].n_azimuth() == drv1[i].n_azimuth());
        CHECK(combined[i].n_elevation() == drv1[i].n_elevation());
        CHECK(std::abs(combined[i].center_frequency - drv1[i].center_frequency) < 1e-12);
    }
}

TEST_CASE("Speaker - concat_multi: pattern data preserved")
{
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    auto combined = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    // Element 0 pattern should match drv1
    CHECK(arma::approx_equal(combined[0].e_theta_re.slice(0), drv1[0].e_theta_re.slice(0), "absdiff", 1e-12));
    CHECK(arma::approx_equal(combined[0].e_theta_im.slice(0), drv1[0].e_theta_im.slice(0), "absdiff", 1e-12));

    // Element 1 pattern should match drv2
    CHECK(arma::approx_equal(combined[0].e_theta_re.slice(1), drv2[0].e_theta_re.slice(0), "absdiff", 1e-12));
    CHECK(arma::approx_equal(combined[0].e_theta_im.slice(1), drv2[0].e_theta_im.slice(0), "absdiff", 1e-12));
}

TEST_CASE("Speaker - concat_multi: element positions preserved")
{
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Set positions before concatenation
    arma::mat pos1 = arma::mat({0.0, 0.0, -0.10}).t();
    arma::mat pos2 = arma::mat({0.0, 0.0, 0.10}).t();
    quadriga_lib::arrayant_set_element_pos_multi(drv1, pos1);
    quadriga_lib::arrayant_set_element_pos_multi(drv2, pos2);

    auto combined = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    REQUIRE(combined[0].element_pos.n_cols == 2);
    CHECK(std::abs(combined[0].element_pos(2, 0) + 0.10) < 1e-12); // drv1 z = -0.10
    CHECK(std::abs(combined[0].element_pos(2, 1) - 0.10) < 1e-12); // drv2 z = +0.10
}

TEST_CASE("Speaker - concat_multi: block-diagonal coupling matrix")
{
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    auto combined = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    // Coupling should be 2x2 identity (block-diagonal of two 1x1 ones)
    REQUIRE(combined[0].coupling_re.n_rows == 2);
    REQUIRE(combined[0].coupling_re.n_cols == 2);
    CHECK(std::abs(combined[0].coupling_re(0, 0) - 1.0) < 1e-12);
    CHECK(std::abs(combined[0].coupling_re(0, 1)) < 1e-12);
    CHECK(std::abs(combined[0].coupling_re(1, 0)) < 1e-12);
    CHECK(std::abs(combined[0].coupling_re(1, 1) - 1.0) < 1e-12);
}

TEST_CASE("Speaker - concat_multi: output passes arrayant_is_valid_multi")
{
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("piston", 0.08, 50.0, 3000.0, 12.0, 24.0, 87.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);

    auto combined = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    std::string err = quadriga_lib::arrayant_is_valid_multi(combined, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - concat_multi: three-driver chained concatenation")
{
    arma::vec freqs = {500.0, 2000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.10, 30.0, 200.0, 12.0, 24.0, 92.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 3000.0, 12.0, 24.0, 87.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv3 = quadriga_lib::generate_speaker<double>("horn", 0.025, 1500.0, 20000.0, 24.0, 12.0, 95.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    auto two_way = quadriga_lib::arrayant_concat_multi(drv1, drv2);
    auto three_way = quadriga_lib::arrayant_concat_multi(two_way, drv3);

    REQUIRE(three_way.size() == 2);
    for (size_t i = 0; i < three_way.size(); ++i)
    {
        CHECK(three_way[i].n_elements() == 3);
        CHECK(three_way[i].n_ports() == 3);
    }

    // Coupling should be 3x3 identity
    CHECK(std::abs(three_way[0].coupling_re(0, 0) - 1.0) < 1e-12);
    CHECK(std::abs(three_way[0].coupling_re(1, 1) - 1.0) < 1e-12);
    CHECK(std::abs(three_way[0].coupling_re(2, 2) - 1.0) < 1e-12);
    CHECK(std::abs(three_way[0].coupling_re(0, 2)) < 1e-12);
    CHECK(std::abs(three_way[0].coupling_re(2, 0)) < 1e-12);

    std::string err = quadriga_lib::arrayant_is_valid_multi(three_way, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - concat_multi: mismatched frequency count throws")
{
    arma::vec freqs2 = {500.0, 2000.0};
    arma::vec freqs3 = {500.0, 2000.0, 8000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs2, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs3, 10.0);

    REQUIRE_THROWS_AS(quadriga_lib::arrayant_concat_multi(drv1, drv2), std::invalid_argument);
}

TEST_CASE("Speaker - concat_multi: mismatched angular resolution throws")
{
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    REQUIRE_THROWS_AS(quadriga_lib::arrayant_concat_multi(drv1, drv2), std::invalid_argument);
}

TEST_CASE("Speaker - concat_multi: mismatched center frequency throws")
{
    arma::vec freqs1 = {1000.0};
    arma::vec freqs2 = {1001.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs1, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs2, 10.0);

    REQUIRE_THROWS_AS(quadriga_lib::arrayant_concat_multi(drv1, drv2), std::invalid_argument);
}

TEST_CASE("Speaker - concat_multi: float template")
{
    arma::fvec freqs = {1000.0f};
    auto drv1 = quadriga_lib::generate_speaker<float>("omni", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                      "monopole", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);
    auto drv2 = quadriga_lib::generate_speaker<float>("omni", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                      "monopole", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);

    auto combined = quadriga_lib::arrayant_concat_multi(drv1, drv2);
    REQUIRE(combined.size() == 1);
    CHECK(combined[0].n_elements() == 2);
    CHECK(combined[0].n_ports() == 2);
}

TEST_CASE("Speaker - concat_multi: documentation example 2-way speaker")
{
    // Reproduce the example from the documentation
    arma::vec freqs = {100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0};

    // Generate woofer: 6.5" piston, sealed box, 50-3000 Hz passband
    auto woofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);

    // Generate tweeter: 1" dome, sealed box, 1500-20000 Hz passband
    auto tweeter = quadriga_lib::generate_speaker<double>(
        "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);

    // Position the drivers: woofer centered, tweeter 12 cm above
    arma::mat woofer_pos = arma::mat({0.0, 0.0, 0.0}).t();
    arma::mat tweeter_pos = arma::mat({0.0, 0.0, 0.12}).t();
    quadriga_lib::arrayant_set_element_pos_multi(woofer, woofer_pos);
    quadriga_lib::arrayant_set_element_pos_multi(tweeter, tweeter_pos);

    // Combine into a single 2-way speaker model
    auto speaker_2way = quadriga_lib::arrayant_concat_multi(woofer, tweeter);

    // Validate structure
    REQUIRE(speaker_2way.size() == 7);
    for (size_t i = 0; i < speaker_2way.size(); ++i)
    {
        CHECK(speaker_2way[i].n_elements() == 2);
        CHECK(speaker_2way[i].n_ports() == 2);
        CHECK(std::abs(speaker_2way[i].center_frequency - freqs[i]) < 1e-6);
    }

    // Element positions correct
    CHECK(std::abs(speaker_2way[0].element_pos(2, 0)) < 1e-12);        // woofer z = 0
    CHECK(std::abs(speaker_2way[0].element_pos(2, 1) - 0.12) < 1e-12); // tweeter z = 0.12

    // Crossover behavior: at 100 Hz, woofer dominates (tweeter below its passband)
    double woofer_100 = speaker_2way[0].e_theta_re(9, 18, 0);  // Woofer on-axis at 100 Hz
    double tweeter_100 = speaker_2way[0].e_theta_re(9, 18, 1); // Tweeter on-axis at 100 Hz
    CHECK(woofer_100 > tweeter_100 * 10.0);                    // Woofer >> tweeter at 100 Hz

    // At 10000 Hz, tweeter dominates (woofer above its passband)
    double woofer_10k = speaker_2way[6].e_theta_re(9, 18, 0);
    double tweeter_10k = speaker_2way[6].e_theta_re(9, 18, 1);
    CHECK(tweeter_10k > woofer_10k * 2.0); // Tweeter >> woofer at 10 kHz

    // Multi-validation passes
    std::string err = quadriga_lib::arrayant_is_valid_multi(speaker_2way, false);
    CHECK(err.empty());
}

// ================================================================================================
// arrayant_copy_element_multi tests
// ================================================================================================

TEST_CASE("Speaker - copy_element_multi: single destination")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk[0].n_elements() == 1);

    // Copy element 0 to element 1 (enlarges)
    quadriga_lib::arrayant_copy_element_multi(spk, 0, (arma::uword)1);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        CHECK(spk[i].n_elements() == 2);
        CHECK(spk[i].n_ports() == 2);

        // Pattern data should be identical
        CHECK(arma::approx_equal(spk[i].e_theta_re.slice(0), spk[i].e_theta_re.slice(1), "absdiff", 1e-12));
        CHECK(arma::approx_equal(spk[i].e_theta_im.slice(0), spk[i].e_theta_im.slice(1), "absdiff", 1e-12));
        CHECK(arma::approx_equal(spk[i].e_phi_re.slice(0), spk[i].e_phi_re.slice(1), "absdiff", 1e-12));
        CHECK(arma::approx_equal(spk[i].e_phi_im.slice(0), spk[i].e_phi_im.slice(1), "absdiff", 1e-12));

        // Element positions should be identical
        CHECK(arma::approx_equal(spk[i].element_pos.col(0), spk[i].element_pos.col(1), "absdiff", 1e-12));
    }
}

TEST_CASE("Speaker - copy_element_multi: multiple destinations")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE(spk[0].n_elements() == 1);

    arma::uvec dest = {1, 2, 3};
    quadriga_lib::arrayant_copy_element_multi(spk, 0, dest);

    REQUIRE(spk[0].n_elements() == 4);
    REQUIRE(spk[0].n_ports() == 4);

    // All 4 elements should have identical patterns
    for (arma::uword e = 1; e < 4; ++e)
        CHECK(arma::approx_equal(spk[0].e_theta_re.slice(0), spk[0].e_theta_re.slice(e), "absdiff", 1e-12));
}

TEST_CASE("Speaker - copy_element_multi: coupling matrix structure after enlargement")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Original: 1 element, 1 port, coupling = [1]
    REQUIRE(spk[0].coupling_re.n_rows == 1);
    REQUIRE(spk[0].coupling_re.n_cols == 1);

    arma::uvec dest = {1, 2};
    quadriga_lib::arrayant_copy_element_multi(spk, 0, dest);

    // After: 3 elements, 3 ports (original port + 2 new ones)
    REQUIRE(spk[0].coupling_re.n_rows == 3);
    REQUIRE(spk[0].coupling_re.n_cols == 3);

    // Should be identity-like: diagonal = 1, off-diagonal = 0
    CHECK(std::abs(spk[0].coupling_re(0, 0) - 1.0) < 1e-12);
    CHECK(std::abs(spk[0].coupling_re(1, 1) - 1.0) < 1e-12);
    CHECK(std::abs(spk[0].coupling_re(2, 2) - 1.0) < 1e-12);
    CHECK(std::abs(spk[0].coupling_re(0, 1)) < 1e-12);
    CHECK(std::abs(spk[0].coupling_re(1, 0)) < 1e-12);
    CHECK(std::abs(spk[0].coupling_re(0, 2)) < 1e-12);
    CHECK(std::abs(spk[0].coupling_re(2, 0)) < 1e-12);
}

TEST_CASE("Speaker - copy_element_multi: consistent across frequencies")
{
    arma::vec freqs = {200.0, 1000.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::arrayant_copy_element_multi(spk, 0, (arma::uword)1);

    // All frequencies should have 2 elements
    for (size_t i = 0; i < spk.size(); ++i)
    {
        CHECK(spk[i].n_elements() == 2);
        CHECK(spk[i].n_ports() == 2);
    }

    // Patterns differ between frequencies (piston beams more at high freq)
    // but within each frequency, element 0 and 1 are identical
    for (size_t i = 0; i < spk.size(); ++i)
        CHECK(arma::approx_equal(spk[i].e_theta_re.slice(0), spk[i].e_theta_re.slice(1), "absdiff", 1e-12));
}

TEST_CASE("Speaker - copy_element_multi: overwrite existing element")
{
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto spk = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    REQUIRE(spk[0].n_elements() == 2);

    // Element 0 (piston) and element 1 (omni) should differ
    CHECK(!arma::approx_equal(spk[0].e_theta_re.slice(0), spk[0].e_theta_re.slice(1), "absdiff", 1e-3));

    // Overwrite element 1 with element 0
    quadriga_lib::arrayant_copy_element_multi(spk, 0, (arma::uword)1);

    // Now they should be identical
    CHECK(arma::approx_equal(spk[0].e_theta_re.slice(0), spk[0].e_theta_re.slice(1), "absdiff", 1e-12));
}

TEST_CASE("Speaker - copy_element_multi: valid after operation")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    arma::uvec dest = {1, 2, 3};
    quadriga_lib::arrayant_copy_element_multi(spk, 0, dest);

    // Individual validation
    for (size_t i = 0; i < spk.size(); ++i)
    {
        std::string err = spk[i].is_valid(false);
        CHECK(err.empty());
    }

    // Multi-validation (element_pos are identical since copy_element copies positions too)
    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - copy_element_multi: empty vector throws")
{
    std::vector<quadriga_lib::arrayant<double>> empty_vec;
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_copy_element_multi(empty_vec, 0, (arma::uword)1), std::invalid_argument);
}

TEST_CASE("Speaker - copy_element_multi: empty destination throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    arma::uvec empty_dest;
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_copy_element_multi(spk, 0, empty_dest), std::invalid_argument);
}

TEST_CASE("Speaker - copy_element_multi: source out of range throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    // Only 1 element, source index 5 is out of range
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_copy_element_multi(spk, 5, (arma::uword)1), std::invalid_argument);
}

TEST_CASE("Speaker - copy_element_multi: float template")
{
    arma::fvec freqs = {1000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("omni", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                     "monopole", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);

    quadriga_lib::arrayant_copy_element_multi(spk, 0, (arma::uword)1);

    CHECK(spk[0].n_elements() == 2);
    CHECK(arma::approx_equal(spk[0].e_theta_re.slice(0), spk[0].e_theta_re.slice(1), "absdiff", 1e-6f));
}

TEST_CASE("Speaker - copy_element_multi: set positions after copy")
{
    // Common workflow: copy elements then reposition them
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                      "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    arma::uvec dest = {1, 2};
    quadriga_lib::arrayant_copy_element_multi(spk, 0, dest);
    REQUIRE(spk[0].n_elements() == 3);

    // Reposition all three elements
    arma::mat positions(3, 3, arma::fill::zeros);
    positions(2, 0) = -0.15;
    positions(2, 1) = 0.0;
    positions(2, 2) = 0.15;
    quadriga_lib::arrayant_set_element_pos_multi(spk, positions);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        CHECK(std::abs(spk[i].element_pos(2, 0) + 0.15) < 1e-12);
        CHECK(std::abs(spk[i].element_pos(2, 1)) < 1e-12);
        CHECK(std::abs(spk[i].element_pos(2, 2) - 0.15) < 1e-12);
    }

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - copy_element_multi: documentation example line array")
{
    // Reproduce the documentation example
    arma::vec freqs = {500.0, 1000.0, 2000.0, 5000.0};
    auto driver = quadriga_lib::generate_speaker<double>(
        "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
        0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Replicate element 0 to create elements 1, 2, 3
    arma::uvec dest = {1, 2, 3};
    quadriga_lib::arrayant_copy_element_multi(driver, 0, dest);

    // Now set the individual positions (vertical line array, 15 cm spacing)
    arma::mat positions(3, 4, arma::fill::zeros);
    positions(2, 0) = -0.225; // z = -22.5 cm
    positions(2, 1) = -0.075; // z = -7.5 cm
    positions(2, 2) = 0.075;  // z = +7.5 cm
    positions(2, 3) = 0.225;  // z = +22.5 cm
    quadriga_lib::arrayant_set_element_pos_multi(driver, positions);

    // Validate structure
    REQUIRE(driver.size() == 4);
    for (size_t i = 0; i < driver.size(); ++i)
    {
        CHECK(driver[i].n_elements() == 4);
        CHECK(driver[i].n_ports() == 4);

        // Positions correct at all frequencies
        CHECK(std::abs(driver[i].element_pos(2, 0) + 0.225) < 1e-12);
        CHECK(std::abs(driver[i].element_pos(2, 1) + 0.075) < 1e-12);
        CHECK(std::abs(driver[i].element_pos(2, 2) - 0.075) < 1e-12);
        CHECK(std::abs(driver[i].element_pos(2, 3) - 0.225) < 1e-12);

        // All elements have the same pattern (copied from element 0)
        for (arma::uword e = 1; e < 4; ++e)
            CHECK(arma::approx_equal(driver[i].e_theta_re.slice(0), driver[i].e_theta_re.slice(e), "absdiff", 1e-12));
    }

    // Multi-validation passes
    std::string err = quadriga_lib::arrayant_is_valid_multi(driver, false);
    CHECK(err.empty());

    // Write and read back to verify QDANT roundtrip works with enlarged arrays
    quadriga_lib::qdant_write_multi("test_line_array.qdant", driver);
    auto driver_read = quadriga_lib::qdant_read_multi<double>("test_line_array.qdant");

    REQUIRE(driver_read.size() == driver.size());
    for (size_t i = 0; i < driver.size(); ++i)
    {
        CHECK(driver_read[i].n_elements() == 4);
        CHECK(driver_read[i].n_ports() == 4);
        CHECK(std::abs(driver_read[i].element_pos(2, 3) - 0.225) < 1e-4);
    }

    std::filesystem::remove("test_line_array.qdant");
}

// ================================================================================================
// arrayant_rotate_pattern_multi tests
// ================================================================================================

TEST_CASE("Speaker - rotate_pattern_multi: z-rotation shifts on-axis direction")
{
    // A piston facing +x rotated 90° around z should now face +y
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    double on_axis_before = spk[0].e_theta_re(9, 18, 0); // az=0, el=0 → +x
    double side_before = spk[0].e_theta_re(9, 27, 0);     // az=90, el=0 → +y
    CHECK(on_axis_before > side_before);                    // Piston beams forward

    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 90.0, 1);

    // After 90° z-rotation: peak should now be at az=90° (+y)
    double at_x_after = spk[0].e_theta_re(9, 18, 0);  // az=0
    double at_y_after = spk[0].e_theta_re(9, 27, 0);   // az=90
    CHECK(at_y_after > at_x_after);

    // The new peak at az=90° should match the original on-axis value
    CHECK(std::abs(at_y_after - on_axis_before) < 0.01);
}

TEST_CASE("Speaker - rotate_pattern_multi: 360° rotation is identity")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 90.0, 60.0, 500.0, 0.15, 0.25, freqs, 10.0);

    // Save original patterns
    std::vector<arma::cube> orig_theta_re, orig_theta_im;
    for (size_t i = 0; i < spk.size(); ++i)
    {
        orig_theta_re.push_back(spk[i].e_theta_re);
        orig_theta_im.push_back(spk[i].e_theta_im);
    }

    // Rotate 360° around z
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 360.0, 1);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        CHECK(arma::approx_equal(spk[i].e_theta_re, orig_theta_re[i], "absdiff", 0.01));
        CHECK(arma::approx_equal(spk[i].e_theta_im, orig_theta_im[i], "absdiff", 0.01));
    }
}

TEST_CASE("Speaker - rotate_pattern_multi: consistent across frequencies")
{
    arma::vec freqs = {200.0, 1000.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 45.0, 1);

    // All frequencies should have the same grid (grid not adjusted)
    for (size_t i = 1; i < spk.size(); ++i)
    {
        CHECK(arma::approx_equal(spk[i].azimuth_grid, spk[0].azimuth_grid, "absdiff", 1e-12));
        CHECK(arma::approx_equal(spk[i].elevation_grid, spk[0].elevation_grid, "absdiff", 1e-12));
    }

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - rotate_pattern_multi: grid preserved (no adjustment)")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    arma::vec az_orig = spk[0].azimuth_grid;
    arma::vec el_orig = spk[0].elevation_grid;
    arma::uword n_az_orig = spk[0].n_azimuth();
    arma::uword n_el_orig = spk[0].n_elevation();

    quadriga_lib::arrayant_rotate_pattern_multi(spk, 10.0, 20.0, 30.0, 1);

    // Grid dimensions and values must be unchanged
    CHECK(spk[0].n_azimuth() == n_az_orig);
    CHECK(spk[0].n_elevation() == n_el_orig);
    CHECK(arma::approx_equal(spk[0].azimuth_grid, az_orig, "absdiff", 1e-12));
    CHECK(arma::approx_equal(spk[0].elevation_grid, el_orig, "absdiff", 1e-12));
}

TEST_CASE("Speaker - rotate_pattern_multi: usage 0 pattern+polarization")
{
    arma::vec freqs = {1000.0};
    auto spk0 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                        "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto spk1 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                        "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Usage 0 rotates both pattern and polarization
    quadriga_lib::arrayant_rotate_pattern_multi(spk0, 0.0, 0.0, 45.0, 0);
    // Usage 1 rotates only pattern
    quadriga_lib::arrayant_rotate_pattern_multi(spk1, 0.0, 0.0, 45.0, 1);

    // e_theta_re should be identical (same pattern rotation)
    CHECK(arma::approx_equal(spk0[0].e_theta_re, spk1[0].e_theta_re, "absdiff", 0.01));

    // But e_phi_re may differ because usage 0 also rotates polarization
    // For a speaker with only e_theta data, the difference is in e_phi
    // (usage 0 may mix theta into phi via polarization rotation)
}

TEST_CASE("Speaker - rotate_pattern_multi: usage 2 polarization only")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    arma::cube theta_orig = spk[0].e_theta_re;

    // Usage 2: only polarization, pattern should be unchanged
    // For a purely theta-polarized source, rotating polarization redistributes energy
    // but the total power per angle should remain the same
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 45.0, 2);

    // Check that total power at each grid point is preserved
    arma::cube power_orig = theta_orig % theta_orig;
    arma::cube power_after = spk[0].e_theta_re % spk[0].e_theta_re +
                             spk[0].e_phi_re % spk[0].e_phi_re;

    CHECK(arma::approx_equal(power_orig.slice(0), power_after.slice(0), "absdiff", 1e-6));
}

TEST_CASE("Speaker - rotate_pattern_multi: selected elements only")
{
    arma::vec freqs = {1000.0};
    auto drv1 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                        "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto drv2 = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                        "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    auto spk = quadriga_lib::arrayant_concat_multi(drv1, drv2);

    arma::mat elem0_before = spk[0].e_theta_re.slice(0);
    arma::mat elem1_before = spk[0].e_theta_re.slice(1);

    // Rotate only element 1
    arma::uvec idx = {1};
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 90.0, 1, idx);

    // Element 0 should be unchanged
    CHECK(arma::approx_equal(spk[0].e_theta_re.slice(0), elem0_before, "absdiff", 1e-12));

    // Element 1 should be different
    CHECK(!arma::approx_equal(spk[0].e_theta_re.slice(1), elem1_before, "absdiff", 0.01));
}

TEST_CASE("Speaker - rotate_pattern_multi: multiple selected elements")
{
    arma::vec freqs = {1000.0};
    auto drv = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    quadriga_lib::arrayant_copy_element_multi(drv, 0, arma::uvec({1, 2}));
    REQUIRE(drv[0].n_elements() == 3);

    arma::mat elem2_before = drv[0].e_theta_re.slice(2);

    // Rotate elements 0 and 1 but not 2
    arma::uvec idx = {0, 1};
    quadriga_lib::arrayant_rotate_pattern_multi(drv, 0.0, 0.0, 90.0, 1, idx);

    // Element 2 should be unchanged
    CHECK(arma::approx_equal(drv[0].e_theta_re.slice(2), elem2_before, "absdiff", 1e-12));

    // Elements 0 and 1 should be identical to each other (both rotated the same way)
    CHECK(arma::approx_equal(drv[0].e_theta_re.slice(0), drv[0].e_theta_re.slice(1), "absdiff", 1e-12));
}

TEST_CASE("Speaker - rotate_pattern_multi: omni pattern stays omni after rotation")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    double val_before = spk[0].e_theta_re(9, 18, 0);

    // Any rotation of an omnidirectional pattern should stay omnidirectional
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 30.0, 45.0, 60.0, 1);

    double val_max = spk[0].e_theta_re.max();
    double val_min = spk[0].e_theta_re.min();
    CHECK(std::abs(val_max - val_min) < 0.01);
    CHECK(std::abs(val_max - val_before) < 0.01);
}

TEST_CASE("Speaker - rotate_pattern_multi: y-rotation tilts beam")
{
    // Piston facing +x, tilt 90° around y → should now face -z (el=-90°)
    arma::vec freqs = {5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    double on_axis_before = spk[0].e_theta_re(9, 18, 0); // az=0, el=0

    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 90.0, 0.0, 1);

    // After tilt: peak should be near el=-90° (ie=0)
    double at_bottom = spk[0].e_theta_re(0, 18, 0);  // az=0, el=-90°
    double at_front = spk[0].e_theta_re(9, 18, 0);    // az=0, el=0

    CHECK(at_bottom > at_front);
    CHECK(std::abs(at_bottom - on_axis_before) < 0.05);
}

TEST_CASE("Speaker - rotate_pattern_multi: empty vector throws")
{
    std::vector<quadriga_lib::arrayant<double>> empty_vec;
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(empty_vec, 0.0, 0.0, 45.0, 1), std::invalid_argument);
}

TEST_CASE("Speaker - rotate_pattern_multi: invalid usage throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 45.0, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 45.0, 4), std::invalid_argument);
}

TEST_CASE("Speaker - rotate_pattern_multi: out-of-range element throws")
{
    arma::vec freqs = {1000.0};
    auto spk = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "monopole", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);
    arma::uvec idx = {5};
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 45.0, 1, idx), std::invalid_argument);
}

TEST_CASE("Speaker - rotate_pattern_multi: float template")
{
    arma::fvec freqs = {1000.0f};
    auto spk = quadriga_lib::generate_speaker<float>("piston", 0.05f, 80.0f, 12000.0f, 12.0f, 12.0f, 85.0f,
                                                      "monopole", 0.0f, 0.0f, 0.0f, 0.15f, 0.25f, freqs, 10.0f);

    float on_axis_before = spk[0].e_theta_re(9, 18, 0);
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0f, 0.0f, 90.0f, 1);
    float at_y_after = spk[0].e_theta_re(9, 27, 0);

    CHECK(std::abs(at_y_after - on_axis_before) < 0.05f);
}

TEST_CASE("Speaker - rotate_pattern_multi: valid after rotation")
{
    arma::vec freqs = {200.0, 1000.0, 5000.0};
    auto spk = quadriga_lib::generate_speaker<double>("horn", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 90.0, 60.0, 500.0, 0.15, 0.25, freqs, 10.0);

    quadriga_lib::arrayant_rotate_pattern_multi(spk, 15.0, 30.0, 45.0, 1);

    for (size_t i = 0; i < spk.size(); ++i)
    {
        std::string err = spk[i].is_valid(false);
        CHECK(err.empty());
        CHECK(spk[i].e_theta_re.is_finite());
    }

    std::string err = quadriga_lib::arrayant_is_valid_multi(spk, false);
    CHECK(err.empty());
}

TEST_CASE("Speaker - rotate_pattern_multi: two opposite rotations cancel")
{
    arma::vec freqs = {500.0, 2000.0};
    auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0,
                                                       "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Save original
    std::vector<arma::cube> orig;
    for (size_t i = 0; i < spk.size(); ++i)
        orig.push_back(spk[i].e_theta_re);

    // Rotate then un-rotate
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, 45.0, 1);
    quadriga_lib::arrayant_rotate_pattern_multi(spk, 0.0, 0.0, -45.0, 1);

    // Should approximately recover original (interpolation introduces small errors)
    for (size_t i = 0; i < spk.size(); ++i)
        CHECK(arma::approx_equal(spk[i].e_theta_re, orig[i], "absdiff", 0.05));
}

TEST_CASE("Speaker - rotate_pattern_multi: documentation example 2-way speaker")
{
    // Reproduce the documentation example
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto woofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto tweeter = quadriga_lib::generate_speaker<double>(
        "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);

    REQUIRE(speaker.size() == 5);
    REQUIRE(speaker[0].n_elements() == 2);

    // Save woofer pattern at 1 kHz before rotation
    arma::mat woofer_1k_before = speaker[2].e_theta_re.slice(0);

    // Rotate entire speaker 30 degrees around the z-axis (heading)
    quadriga_lib::arrayant_rotate_pattern_multi(speaker, 0.0, 0.0, 30.0, 1);

    // Both elements should have been rotated
    CHECK(!arma::approx_equal(speaker[2].e_theta_re.slice(0), woofer_1k_before, "absdiff", 0.01));

    // Save tweeter pattern at 10 kHz before selective rotation
    arma::mat tweeter_10k_before = speaker[4].e_theta_re.slice(1);
    arma::mat woofer_10k_before = speaker[4].e_theta_re.slice(0);

    // Tilt only the tweeter (element 1) upward by 10 degrees
    arma::uvec tweeter_idx = {1};
    quadriga_lib::arrayant_rotate_pattern_multi(speaker, 0.0, 10.0, 0.0, 1, tweeter_idx);

    // Woofer (element 0) should be unchanged
    CHECK(arma::approx_equal(speaker[4].e_theta_re.slice(0), woofer_10k_before, "absdiff", 1e-12));

    // Tweeter (element 1) should have changed
    CHECK(!arma::approx_equal(speaker[4].e_theta_re.slice(1), tweeter_10k_before, "absdiff", 0.01));

    // Validation passes
    std::string err = quadriga_lib::arrayant_is_valid_multi(speaker, false);
    CHECK(err.empty());
}