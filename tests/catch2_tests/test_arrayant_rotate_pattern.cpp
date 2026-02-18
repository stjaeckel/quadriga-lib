// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

// ================================================================================================
// Helper: compute total radiated power (sum of |E_theta|^2 + |E_phi|^2 over all angles and elements)
// This should be preserved by any rotation (energy conservation)
// ================================================================================================
template <typename dtype>
dtype total_power(const quadriga_lib::arrayant<dtype> &ant)
{
    dtype power = dtype(0.0);
    for (arma::uword s = 0; s < ant.e_theta_re.n_slices; ++s)
    {
        const dtype *tr = ant.e_theta_re.slice_memptr(s);
        const dtype *ti = ant.e_theta_im.slice_memptr(s);
        const dtype *pr = ant.e_phi_re.slice_memptr(s);
        const dtype *pi = ant.e_phi_im.slice_memptr(s);
        arma::uword n = ant.e_theta_re.n_rows * ant.e_theta_re.n_cols;
        for (arma::uword j = 0; j < n; ++j)
            power += tr[j] * tr[j] + ti[j] * ti[j] + pr[j] * pr[j] + pi[j] * pi[j];
    }
    return power;
}

// ================================================================================================
// Helper: maximum absolute difference between two arrayant pattern fields
// ================================================================================================
template <typename dtype>
dtype max_pattern_diff(const quadriga_lib::arrayant<dtype> &a, const quadriga_lib::arrayant<dtype> &b)
{
    // Must have same grid sizes for meaningful comparison
    if (a.n_elevation() != b.n_elevation() || a.n_azimuth() != b.n_azimuth() || a.n_elements() != b.n_elements())
        return dtype(1.0e30); // Signal mismatch

    dtype max_diff = dtype(0.0);
    for (arma::uword s = 0; s < a.e_theta_re.n_slices; ++s)
    {
        arma::uword n = a.e_theta_re.n_rows * a.e_theta_re.n_cols;
        const dtype *atr = a.e_theta_re.slice_memptr(s), *btr = b.e_theta_re.slice_memptr(s);
        const dtype *ati = a.e_theta_im.slice_memptr(s), *bti = b.e_theta_im.slice_memptr(s);
        const dtype *apr = a.e_phi_re.slice_memptr(s), *bpr = b.e_phi_re.slice_memptr(s);
        const dtype *api = a.e_phi_im.slice_memptr(s), *bpi = b.e_phi_im.slice_memptr(s);
        for (arma::uword j = 0; j < n; ++j)
        {
            dtype d;
            d = std::abs(atr[j] - btr[j]);
            max_diff = d > max_diff ? d : max_diff;
            d = std::abs(ati[j] - bti[j]);
            max_diff = d > max_diff ? d : max_diff;
            d = std::abs(apr[j] - bpr[j]);
            max_diff = d > max_diff ? d : max_diff;
            d = std::abs(api[j] - bpi[j]);
            max_diff = d > max_diff ? d : max_diff;
        }
    }
    return max_diff;
}

// ================================================================================================
// Section 1: rotate_pattern - Input validation and error handling
// ================================================================================================

TEST_CASE("Rotate Pattern - Invalid usage parameter throws")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    REQUIRE_THROWS_AS(ant.rotate_pattern(0.0, 0.0, 0.0, 5), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.rotate_pattern(0.0, 0.0, 0.0, 6), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.rotate_pattern(0.0, 0.0, 0.0, 100), std::invalid_argument);
}

TEST_CASE("Rotate Pattern - Element index out of bounds throws")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>(); // 1 element
    REQUIRE(ant.n_elements() == 1);

    // Element index 1 is out of bounds for a 1-element antenna (0-based)
    REQUIRE_THROWS_AS(ant.rotate_pattern(10.0, 0.0, 0.0, 0, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.rotate_pattern(10.0, 0.0, 0.0, 0, 99), std::invalid_argument);
}

TEST_CASE("Rotate Pattern - Read-only inplace throws")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.read_only = true;

    // Inplace rotation (output == nullptr) on read-only should throw
    REQUIRE_THROWS_AS(ant.rotate_pattern(10.0, 0.0, 0.0, 3), std::invalid_argument);

    // But writing to an output should NOT throw
    quadriga_lib::arrayant<double> out;
    REQUIRE_NOTHROW(ant.rotate_pattern(10.0, 0.0, 0.0, 3, (unsigned)-1, &out));
}

TEST_CASE("Rotate Pattern - Valid usage values accepted")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    for (unsigned u = 0; u <= 4; ++u)
        REQUIRE_NOTHROW(ant.rotate_pattern(10.0, 0.0, 0.0, u));
}

// ================================================================================================
// Section 2: rotate_pattern - Identity and near-identity rotations
// ================================================================================================

TEST_CASE("Rotate Pattern - Zero rotation is identity (omni, double)")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    auto ref = ant.copy();

    // Rotate by (0,0,0) with usage 3 (no grid adjustment) - should be exact identity
    ant.rotate_pattern(0.0, 0.0, 0.0, 3);

    CHECK(ant.n_elevation() == ref.n_elevation());
    CHECK(ant.n_azimuth() == ref.n_azimuth());
    CHECK(arma::approx_equal(ant.e_theta_re, ref.e_theta_re, "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.e_theta_im, ref.e_theta_im, "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.e_phi_re, ref.e_phi_re, "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.e_phi_im, ref.e_phi_im, "absdiff", 1e-14));
}

TEST_CASE("Rotate Pattern - Zero rotation is identity (omni, float)")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0f, 0.0f, 0.0f, 3);

    CHECK(arma::approx_equal(ant.e_theta_re, ref.e_theta_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_phi_re, ref.e_phi_re, "absdiff", 1e-6));
}

TEST_CASE("Rotate Pattern - Zero rotation is identity (dipole)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0, 0.0, 0.0, 3);

    CHECK(arma::approx_equal(ant.e_theta_re, ref.e_theta_re, "absdiff", 1e-13));
    CHECK(arma::approx_equal(ant.e_phi_re, ref.e_phi_re, "absdiff", 1e-13));
}

TEST_CASE("Rotate Pattern - 360 degree z-rotation approximates identity (usage 3, omni)")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    auto ref = ant.copy();

    // Usage 3 = rotate both pattern+polarization, no grid adjustment
    ant.rotate_pattern(0.0, 0.0, 360.0, 3);

    // After 360° rotation and interpolation round-trip, expect small numerical error
    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern - 360 degree x-rotation approximates identity (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(360.0, 0.0, 0.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-4); // x-rotation has more interpolation artifacts near poles
}

TEST_CASE("Rotate Pattern - 360 degree y-rotation approximates identity (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0, 360.0, 0.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-4);
}

TEST_CASE("Rotate Pattern - 360 degree z-rotation approximates identity (usage 3, dipole)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0, 0.0, 360.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern - 360 degree z-rotation approximates identity (usage 3, xpol)")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0, 0.0, 360.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern - 360 degree z-rotation approximates identity (usage 4, dipole)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    // Usage 4 = rotate only pattern, no grid adjustment
    ant.rotate_pattern(0.0, 0.0, 360.0, 4);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

// ================================================================================================
// Section 3: rotate_pattern - Rotation composition
// ================================================================================================

TEST_CASE("Rotate Pattern - 90 + 270 = 360 z-rotation (usage 3)")
{
    auto ant_360 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant_comp = ant_360.copy();

    ant_360.rotate_pattern(0.0, 0.0, 360.0, 3);
    ant_comp.rotate_pattern(0.0, 0.0, 90.0, 3);
    ant_comp.rotate_pattern(0.0, 0.0, 270.0, 3);

    // Both should be close to identity; check they're close to each other
    double diff = max_pattern_diff(ant_360, ant_comp);
    CHECK(diff < 1e-4); // Two interpolation round-trips accumulate more error
}

TEST_CASE("Rotate Pattern - 180 + 180 = 360 z-rotation (usage 3)")
{
    auto ant_360 = quadriga_lib::generate_arrayant_custom<double>(30.0, 30.0);
    auto ant_comp = ant_360.copy();

    ant_360.rotate_pattern(0.0, 0.0, 360.0, 3);
    ant_comp.rotate_pattern(0.0, 0.0, 180.0, 3);
    ant_comp.rotate_pattern(0.0, 0.0, 180.0, 3);

    double diff = max_pattern_diff(ant_360, ant_comp);
    CHECK(diff < 1e-4);
}

// ================================================================================================
// Section 4: rotate_pattern - Omni antenna invariance under rotation
// ================================================================================================

TEST_CASE("Rotate Pattern - Omni pattern invariant under z-rotation (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    auto ref = ant.copy();

    // An isotropic radiator with vertical polarization should have constant e_theta_re = 1.0
    // After z-rotation with usage 3 (pattern+pol, no grid adj), it should remain constant
    ant.rotate_pattern(0.0, 0.0, 45.0, 3);

    // For a pure z-rotation of an omni with only e_theta, the theta/phi decomposition changes
    // but total power at each angle should remain 1.0
    arma::uword n_el = ant.n_elevation(), n_az = ant.n_azimuth();
    for (arma::uword ia = 0; ia < n_az; ++ia)
    {
        for (arma::uword ie = 0; ie < n_el; ++ie)
        {
            double tr = ant.e_theta_re.at(ie, ia, 0);
            double ti = ant.e_theta_im.at(ie, ia, 0);
            double pr = ant.e_phi_re.at(ie, ia, 0);
            double pi = ant.e_phi_im.at(ie, ia, 0);
            double pwr = tr * tr + ti * ti + pr * pr + pi * pi;
            CHECK(std::abs(pwr - 1.0) < 1e-4);
        }
    }
}

TEST_CASE("Rotate Pattern - Omni pattern invariant under z-rotation (usage 4, pattern only)")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    auto ref = ant.copy();

    // Usage 4 = only pattern rotation, no grid adjustment
    // For omni, any rotation should not change the magnitude pattern
    ant.rotate_pattern(0.0, 0.0, 90.0, 4);

    // e_theta_re should remain ~1.0, everything else ~0.0 (since omni has no azimuth dependence)
    CHECK(arma::approx_equal(ant.e_theta_re, ref.e_theta_re, "absdiff", 1e-6));
}

// ================================================================================================
// Section 5: rotate_pattern - Directivity preservation
// ================================================================================================

TEST_CASE("Rotate Pattern - Directivity preserved after rotation (dipole)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    double dir_before = ant.calc_directivity_dBi(0);

    ant.rotate_pattern(30.0, 45.0, 60.0, 0); // Full rotation with grid adjustment

    double dir_after = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir_before - dir_after) < 0.1);
}

TEST_CASE("Rotate Pattern - Directivity preserved after rotation (3GPP)")
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<double>();
    double dir_before = ant.calc_directivity_dBi(0);

    ant.rotate_pattern(15.0, 0.0, 45.0, 0);

    double dir_after = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir_before - dir_after) < 0.3); // Grid resampling may cause small change
}

TEST_CASE("Rotate Pattern - Directivity preserved after rotation (custom beam)")
{
    auto ant = quadriga_lib::generate_arrayant_custom<double>(30.0, 30.0);
    double dir_before = ant.calc_directivity_dBi(0);

    ant.rotate_pattern(0.0, 0.0, 90.0, 0);

    double dir_after = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir_before - dir_after) < 0.2);
}

// ================================================================================================
// Section 6: rotate_pattern - Usage modes
// ================================================================================================

TEST_CASE("Rotate Pattern - Usage 2 only changes polarization, not power pattern")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();
    auto ref = ant.copy();

    // Usage 2 = rotate only polarization
    ant.rotate_pattern(0.0, 0.0, 45.0, 2);

    // Total power at each angle should be preserved
    for (arma::uword ie = 0; ie < ant.n_elevation(); ++ie)
    {
        for (arma::uword ia = 0; ia < ant.n_azimuth(); ++ia)
        {
            for (arma::uword s = 0; s < ant.n_elements(); ++s)
            {
                double pwr_ref = ref.e_theta_re.at(ie, ia, s) * ref.e_theta_re.at(ie, ia, s) +
                                 ref.e_theta_im.at(ie, ia, s) * ref.e_theta_im.at(ie, ia, s) +
                                 ref.e_phi_re.at(ie, ia, s) * ref.e_phi_re.at(ie, ia, s) +
                                 ref.e_phi_im.at(ie, ia, s) * ref.e_phi_im.at(ie, ia, s);
                double pwr_rot = ant.e_theta_re.at(ie, ia, s) * ant.e_theta_re.at(ie, ia, s) +
                                 ant.e_theta_im.at(ie, ia, s) * ant.e_theta_im.at(ie, ia, s) +
                                 ant.e_phi_re.at(ie, ia, s) * ant.e_phi_re.at(ie, ia, s) +
                                 ant.e_phi_im.at(ie, ia, s) * ant.e_phi_im.at(ie, ia, s);
                CHECK(std::abs(pwr_ref - pwr_rot) < 1e-10);
            }
        }
    }
}

TEST_CASE("Rotate Pattern - Usage 2 preserves grid")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(30.0, 20.0, 10.0, 2);

    // Usage 2 should not change the sampling grid
    CHECK(arma::approx_equal(ant.azimuth_grid, ref.azimuth_grid, "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.elevation_grid, ref.elevation_grid, "absdiff", 1e-14));
}

TEST_CASE("Rotate Pattern - Usage 3 and 4 preserve grid size")
{
    auto ant3 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant4 = ant3.copy();
    arma::uword n_el = ant3.n_elevation(), n_az = ant3.n_azimuth();

    // Usage 3 = rotate both, no grid adjustment
    ant3.rotate_pattern(20.0, 30.0, 40.0, 3);
    CHECK(ant3.n_elevation() == n_el);
    CHECK(ant3.n_azimuth() == n_az);

    // Usage 4 = rotate pattern only, no grid adjustment
    ant4.rotate_pattern(20.0, 30.0, 40.0, 4);
    CHECK(ant4.n_elevation() == n_el);
    CHECK(ant4.n_azimuth() == n_az);
}

// ================================================================================================
// Section 7: rotate_pattern - Output pointer vs inplace
// ================================================================================================

TEST_CASE("Rotate Pattern - Output pointer gives same result as inplace (usage 3)")
{
    auto ant_inplace = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant_source = ant_inplace.copy();
    quadriga_lib::arrayant<double> ant_output;

    ant_inplace.rotate_pattern(15.0, 25.0, 35.0, 3);
    ant_source.rotate_pattern(15.0, 25.0, 35.0, 3, (unsigned)-1, &ant_output);

    double diff = max_pattern_diff(ant_inplace, ant_output);
    CHECK(diff < 1e-14);
    CHECK(arma::approx_equal(ant_inplace.azimuth_grid, ant_output.azimuth_grid, "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant_inplace.elevation_grid, ant_output.elevation_grid, "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant_inplace.element_pos, ant_output.element_pos, "absdiff", 1e-14));
}

TEST_CASE("Rotate Pattern - Output pointer gives same result as inplace (usage 2)")
{
    auto ant_inplace = quadriga_lib::generate_arrayant_xpol<double>();
    auto ant_source = ant_inplace.copy();
    quadriga_lib::arrayant<double> ant_output;

    ant_inplace.rotate_pattern(10.0, 20.0, 30.0, 2);
    ant_source.rotate_pattern(10.0, 20.0, 30.0, 2, (unsigned)-1, &ant_output);

    double diff = max_pattern_diff(ant_inplace, ant_output);
    CHECK(diff < 1e-14);
}

// ================================================================================================
// Section 8: rotate_pattern - Single element rotation
// ================================================================================================

TEST_CASE("Rotate Pattern - Single element rotation only affects that element")
{
    // Use a directive 2-element antenna (3GPP H/V pol) so that z-rotation actually
    // changes the sampled pattern values. The xpol antenna is spatially uniform (constant
    // 1.0 across all angles), so rotation just samples the same constant and appears unchanged.
    auto ant = quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3e8, 2); // H/V polarized, 2 elements
    REQUIRE(ant.n_elements() == 2);
    auto ref = ant.copy();

    // Rotate only element 0 with usage 3 (pattern+pol, no grid adjust)
    ant.rotate_pattern(0.0, 0.0, 45.0, 3, 0);

    // Element 1 should be unchanged
    CHECK(arma::approx_equal(ant.e_theta_re.slice(1), ref.e_theta_re.slice(1), "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.e_theta_im.slice(1), ref.e_theta_im.slice(1), "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(1), ref.e_phi_re.slice(1), "absdiff", 1e-14));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(1), ref.e_phi_im.slice(1), "absdiff", 1e-14));

    // Element 0 should be different from original (directive pattern rotated by 45°)
    bool theta_changed = !arma::approx_equal(ant.e_theta_re.slice(0), ref.e_theta_re.slice(0), "absdiff", 1e-6);
    bool phi_changed = !arma::approx_equal(ant.e_phi_re.slice(0), ref.e_phi_re.slice(0), "absdiff", 1e-6);
    CHECK((theta_changed || phi_changed));
}

TEST_CASE("Rotate Pattern - Single element with output pointer (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<double>(); // 2 elements
    quadriga_lib::arrayant<double> out;

    // Rotate element 1 to output
    ant.rotate_pattern(0.0, 0.0, 90.0, 3, 1, &out);

    // Output should have 1 element (the rotated one)
    CHECK(out.n_elements() == 1);
    CHECK(out.n_elevation() == ant.n_elevation());
    CHECK(out.n_azimuth() == ant.n_azimuth());
}

// ================================================================================================
// Section 9: rotate_pattern - Element position rotation
// ================================================================================================

TEST_CASE("Rotate Pattern - Element positions are rotated correctly (z-rotation)")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();
    // Set a known element position
    ant.element_pos.zeros(3, 2);
    ant.element_pos(0, 0) = 1.0; // Element 0 at (1, 0, 0)
    ant.element_pos(1, 1) = 1.0; // Element 1 at (0, 1, 0)
    ant.validate();

    // 90° z-rotation: (1,0,0) -> (0,1,0), (0,1,0) -> (-1,0,0)
    ant.rotate_pattern(0.0, 0.0, 90.0, 3);

    CHECK(std::abs(ant.element_pos(0, 0) - 0.0) < 1e-10);
    CHECK(std::abs(ant.element_pos(1, 0) - 1.0) < 1e-10);
    CHECK(std::abs(ant.element_pos(2, 0) - 0.0) < 1e-10);

    CHECK(std::abs(ant.element_pos(0, 1) + 1.0) < 1e-10);
    CHECK(std::abs(ant.element_pos(1, 1) - 0.0) < 1e-10);
    CHECK(std::abs(ant.element_pos(2, 1) - 0.0) < 1e-10);
}

TEST_CASE("Rotate Pattern - Element positions unchanged for usage 2")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();
    ant.element_pos.zeros(3, 2);
    ant.element_pos(0, 0) = 1.0;
    ant.element_pos(1, 1) = 2.0;
    ant.validate();
    auto ref_pos = ant.element_pos;

    // Usage 2 only rotates polarization - positions should not change
    ant.rotate_pattern(30.0, 45.0, 60.0, 2);

    CHECK(arma::approx_equal(ant.element_pos, ref_pos, "absdiff", 1e-14));
}

// ================================================================================================
// Section 10: rotate_pattern - Specific known rotation results
// ================================================================================================

TEST_CASE("Rotate Pattern - Dipole 90 deg x-rotation flips pattern to horizontal (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();

    // A vertical dipole has e_theta pattern ~ cos(elevation), e_phi = 0
    // 90° x-rotation (bank) tilts it - the peak should shift
    double dir_before = ant.calc_directivity_dBi(0);
    ant.rotate_pattern(90.0, 0.0, 0.0, 0); // Full rotation with grid adjustment

    double dir_after = ant.calc_directivity_dBi(0);
    // Directivity should be preserved
    CHECK(std::abs(dir_before - dir_after) < 0.2);
}

TEST_CASE("Rotate Pattern - Float precision (custom beam, usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    auto ref = ant.copy();

    ant.rotate_pattern(0.0f, 0.0f, 360.0f, 3);

    float diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-3f); // Float has less precision
}

// ================================================================================================
// Section 11: rotate_pattern - Grid adjustment tests (usage 0 and 1)
// ================================================================================================

TEST_CASE("Rotate Pattern - Usage 0 may adjust grid for non-uniform antenna")
{
    // Create a custom antenna with deliberately non-uniform grid
    quadriga_lib::arrayant<double> ant;
    ant.azimuth_grid = {-1.0, -0.3, 0.0, 0.3, 1.0};
    ant.elevation_grid = {-0.5, 0.0, 0.5};
    arma::uword n_el = 3, n_az = 5;
    ant.e_theta_re.ones(n_el, n_az, 1);
    ant.e_theta_im.zeros(n_el, n_az, 1);
    ant.e_phi_re.zeros(n_el, n_az, 1);
    ant.e_phi_im.zeros(n_el, n_az, 1);
    ant.validate();

    // Usage 0 should detect non-uniform grid and resamples
    ant.rotate_pattern(0.0, 0.0, 45.0, 0);

    // Grid should have been adjusted (different size)
    // At minimum, it shouldn't crash
    CHECK(ant.n_elevation() > 0);
    CHECK(ant.n_azimuth() > 0);
    CHECK(ant.is_valid().empty());
}

TEST_CASE("Rotate Pattern - Usage 0 preserves directivity for non-uniform grid")
{
    quadriga_lib::arrayant<double> ant;
    ant.azimuth_grid = {-1.0, -0.3, 0.0, 0.3, 1.0};
    ant.elevation_grid = {-0.5, 0.0, 0.5};
    arma::uword n_el = 3, n_az = 5;
    ant.e_theta_re.ones(n_el, n_az, 1);
    ant.e_theta_im.zeros(n_el, n_az, 1);
    ant.e_phi_re.zeros(n_el, n_az, 1);
    ant.e_phi_im.zeros(n_el, n_az, 1);
    ant.validate();

    double dir_before = ant.calc_directivity_dBi(0);
    ant.rotate_pattern(0.0, 0.0, 45.0, 0);
    double dir_after = ant.calc_directivity_dBi(0);

    CHECK(std::abs(dir_before - dir_after) < 1.0); // Grid resampling can cause some deviation
}

// ================================================================================================
// Section 12: rotate_pattern - Consistency between usage modes
// ================================================================================================

TEST_CASE("Rotate Pattern - Usage 0 and 3 give same result for uniform grid")
{
    auto ant0 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant3 = ant0.copy();

    // For a 1-degree uniform grid, update_grid should be false, so usage 0 and 3 should be identical
    ant0.rotate_pattern(15.0, 20.0, 25.0, 0);
    ant3.rotate_pattern(15.0, 20.0, 25.0, 3);

    // They should produce identical results if no grid adjustment was triggered
    if (ant0.n_elevation() == ant3.n_elevation() && ant0.n_azimuth() == ant3.n_azimuth())
    {
        double diff = max_pattern_diff(ant0, ant3);
        CHECK(diff < 1e-14);
    }
}

TEST_CASE("Rotate Pattern - Usage 1 and 4 give same result for uniform grid")
{
    auto ant1 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant4 = ant1.copy();

    ant1.rotate_pattern(10.0, 15.0, 20.0, 1);
    ant4.rotate_pattern(10.0, 15.0, 20.0, 4);

    if (ant1.n_elevation() == ant4.n_elevation() && ant1.n_azimuth() == ant4.n_azimuth())
    {
        double diff = max_pattern_diff(ant1, ant4);
        CHECK(diff < 1e-14);
    }
}

// ================================================================================================
// Section 13: rotate_pattern - Symmetry tests
// ================================================================================================

TEST_CASE("Rotate Pattern - Forward and inverse rotation recovers original (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_custom<double>(30.0, 60.0);
    auto ref = ant.copy();

    // Rotate forward then backward
    // Note: Euler rotation composition is NOT simply negating angles due to non-commutativity
    // But for a single-axis rotation, negation works:
    ant.rotate_pattern(0.0, 0.0, 45.0, 3);
    ant.rotate_pattern(0.0, 0.0, -45.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-4); // Two interpolation round-trips
}

TEST_CASE("Rotate Pattern - Multiple small rotations accumulate correctly (usage 3)")
{
    auto ant_single = quadriga_lib::generate_arrayant_custom<double>(30.0, 30.0);
    auto ant_multi = ant_single.copy();

    // Single 90° rotation
    ant_single.rotate_pattern(0.0, 0.0, 90.0, 3);

    // Ten 9° rotations
    for (int i = 0; i < 10; ++i)
        ant_multi.rotate_pattern(0.0, 0.0, 9.0, 3);

    // Should give similar results, but multi-step accumulates interpolation error
    double dir_single = ant_single.calc_directivity_dBi(0);
    double dir_multi = ant_multi.calc_directivity_dBi(0);
    CHECK(std::abs(dir_single - dir_multi) < 0.5);
}

// ================================================================================================
// Section 14: rotate_pattern - Edge cases
// ================================================================================================

TEST_CASE("Rotate Pattern - Very small rotation angle")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0, 0.0, 1e-10, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern - Large rotation angle (720 degrees)")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    auto ref = ant.copy();

    ant.rotate_pattern(0.0, 0.0, 720.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern - Negative rotation angles")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    // -360° should also be near-identity
    ant.rotate_pattern(0.0, 0.0, -360.0, 3);

    double diff = max_pattern_diff(ref, ant);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern - All three axes simultaneously (usage 3)")
{
    auto ant = quadriga_lib::generate_arrayant_custom<double>(30.0, 30.0);
    double dir_before = ant.calc_directivity_dBi(0);

    ant.rotate_pattern(30.0, 45.0, 60.0, 3);

    double dir_after = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir_before - dir_after) < 0.3);
    CHECK(ant.is_valid().empty());
}

// ================================================================================================
// Section 15: rotate_pattern - Pattern validity after rotation
// ================================================================================================

TEST_CASE("Rotate Pattern - Result is always valid (multiple usage modes)")
{
    auto ant0 = quadriga_lib::generate_arrayant_3GPP<double>();
    auto ant1 = ant0.copy(), ant2 = ant0.copy(), ant3 = ant0.copy(), ant4 = ant0.copy();

    ant0.rotate_pattern(20.0, 30.0, 40.0, 0);
    CHECK(ant0.is_valid().empty());

    ant1.rotate_pattern(20.0, 30.0, 40.0, 1);
    CHECK(ant1.is_valid().empty());

    ant2.rotate_pattern(20.0, 30.0, 40.0, 2);
    CHECK(ant2.is_valid().empty());

    ant3.rotate_pattern(20.0, 30.0, 40.0, 3);
    CHECK(ant3.is_valid().empty());

    ant4.rotate_pattern(20.0, 30.0, 40.0, 4);
    CHECK(ant4.is_valid().empty());
}

// ================================================================================================
// Section 16: rotate_pattern - Multi-element antennas
// ================================================================================================

TEST_CASE("Rotate Pattern - 3GPP 2x2 array rotation preserves all elements")
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3e9, 1);
    REQUIRE(ant.n_elements() == 4);

    auto ref = ant.copy();
    ant.rotate_pattern(0.0, 0.0, 45.0, 3);

    // All elements should have been rotated
    for (arma::uword s = 0; s < 4; ++s)
    {
        bool changed = !arma::approx_equal(ant.e_theta_re.slice(s), ref.e_theta_re.slice(s), "absdiff", 1e-6) ||
                       !arma::approx_equal(ant.e_phi_re.slice(s), ref.e_phi_re.slice(s), "absdiff", 1e-6);
        CHECK(changed);
    }
}

TEST_CASE("Rotate Pattern - Rotating all elements individually matches rotating all at once")
{
    auto ant_all = quadriga_lib::generate_arrayant_xpol<double>();
    auto ant_ind = ant_all.copy();
    REQUIRE(ant_all.n_elements() == 2);

    // Rotate all at once (usage 3)
    ant_all.rotate_pattern(0.0, 0.0, 30.0, 3);

    // Rotate each element individually (usage 3)
    ant_ind.rotate_pattern(0.0, 0.0, 30.0, 3, 0);
    ant_ind.rotate_pattern(0.0, 0.0, 30.0, 3, 1);

    double diff = max_pattern_diff(ant_all, ant_ind);
    CHECK(diff < 1e-14);
}

// ================================================================================================
// Section 17: arrayant_rotate_pattern_multi - Basic tests
// ================================================================================================

TEST_CASE("Rotate Pattern Multi - Empty vector throws")
{
    std::vector<quadriga_lib::arrayant<double>> vec;
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 0.0, 45.0, 0, {}),
                      std::invalid_argument);
}

TEST_CASE("Rotate Pattern Multi - Invalid usage throws")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    std::vector<quadriga_lib::arrayant<double>> vec = {ant, ant};

    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 0.0, 45.0, 3, {}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 0.0, 45.0, 4, {}),
                      std::invalid_argument);
}

TEST_CASE("Rotate Pattern Multi - Element index out of range throws")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>(); // 1 element
    std::vector<quadriga_lib::arrayant<double>> vec = {ant, ant};
    arma::uvec bad_elem = {0, 1}; // Element 1 is out of range

    REQUIRE_THROWS_AS(quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 0.0, 45.0, 0, bad_elem),
                      std::invalid_argument);
}

TEST_CASE("Rotate Pattern Multi - Single entry behaves like single rotate_pattern")
{
    auto ant_single = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant_multi = ant_single.copy();

    // Single: use internal usage 3 (which is what multi maps usage 0 to)
    ant_single.rotate_pattern(10.0, 20.0, 30.0, 3);

    // Multi with usage 0 -> maps to internal usage 3
    std::vector<quadriga_lib::arrayant<double>> vec = {ant_multi};
    quadriga_lib::arrayant_rotate_pattern_multi(vec, 10.0, 20.0, 30.0, 0, {});

    double diff = max_pattern_diff(ant_single, vec[0]);
    CHECK(diff < 1e-14);
}

TEST_CASE("Rotate Pattern Multi - All entries in vector are rotated")
{
    // arrayant_is_valid_multi requires all entries to have matching grid dimensions,
    // so we use same-resolution antennas with different directivities
    auto ant1 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant2 = quadriga_lib::generate_arrayant_custom<double>(30.0, 30.0); // Same 1° grid, different pattern
    auto ref1 = ant1.copy();
    auto ref2 = ant2.copy();

    REQUIRE(ant1.n_elevation() == ant2.n_elevation());
    REQUIRE(ant1.n_azimuth() == ant2.n_azimuth());

    std::vector<quadriga_lib::arrayant<double>> vec = {ant1, ant2};
    // Use y-rotation: the dipole is azimuthally symmetric around z, so z-rotation alone
    // won't change its sampled pattern. A y-rotation (tilt) breaks that symmetry.
    quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 45.0, 0.0, 0, {});

    // Both should be different from reference
    double diff1 = max_pattern_diff(ref1, vec[0]);
    double diff2 = max_pattern_diff(ref2, vec[1]);
    CHECK(diff1 > 1e-3);
    CHECK(diff2 > 1e-3);

    // Both should still be valid
    CHECK(vec[0].is_valid().empty());
    CHECK(vec[1].is_valid().empty());
}

TEST_CASE("Rotate Pattern Multi - 360 degree rotation approximates identity")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<double>();
    auto ref = ant.copy();

    std::vector<quadriga_lib::arrayant<double>> vec = {ant};
    quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 0.0, 360.0, 0, {});

    double diff = max_pattern_diff(ref, vec[0]);
    CHECK(diff < 1e-6);
}

TEST_CASE("Rotate Pattern Multi - Usage mapping: 0->3, 1->4, 2->2")
{
    // Verify that multi usage 0 behaves like single usage 3
    auto ant_u0 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant_u3 = ant_u0.copy();

    std::vector<quadriga_lib::arrayant<double>> vec0 = {ant_u0};
    quadriga_lib::arrayant_rotate_pattern_multi(vec0, 15.0, 25.0, 35.0, 0, {});
    ant_u3.rotate_pattern(15.0, 25.0, 35.0, 3);

    double diff03 = max_pattern_diff(vec0[0], ant_u3);
    CHECK(diff03 < 1e-14);

    // Verify that multi usage 1 behaves like single usage 4
    auto ant_u1 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant_u4 = ant_u1.copy();

    std::vector<quadriga_lib::arrayant<double>> vec1 = {ant_u1};
    quadriga_lib::arrayant_rotate_pattern_multi(vec1, 15.0, 25.0, 35.0, 1, {});
    ant_u4.rotate_pattern(15.0, 25.0, 35.0, 4);

    double diff14 = max_pattern_diff(vec1[0], ant_u4);
    CHECK(diff14 < 1e-14);

    // Verify that multi usage 2 behaves like single usage 2
    auto ant_u2m = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant_u2s = ant_u2m.copy();

    std::vector<quadriga_lib::arrayant<double>> vec2 = {ant_u2m};
    quadriga_lib::arrayant_rotate_pattern_multi(vec2, 15.0, 25.0, 35.0, 2, {});
    ant_u2s.rotate_pattern(15.0, 25.0, 35.0, 2);

    double diff22 = max_pattern_diff(vec2[0], ant_u2s);
    CHECK(diff22 < 1e-14);
}

TEST_CASE("Rotate Pattern Multi - Per-element rotation")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<double>(); // 2 elements
    auto ref = ant.copy();

    arma::uvec elem = {0}; // Only rotate element 0
    std::vector<quadriga_lib::arrayant<double>> vec = {ant};
    quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0, 0.0, 45.0, 0, elem);

    // Element 1 should be unchanged
    CHECK(arma::approx_equal(vec[0].e_theta_re.slice(1), ref.e_theta_re.slice(1), "absdiff", 1e-14));
    CHECK(arma::approx_equal(vec[0].e_phi_re.slice(1), ref.e_phi_re.slice(1), "absdiff", 1e-14));
}

TEST_CASE("Rotate Pattern Multi - Directivity preserved across all entries")
{
    auto ant1 = quadriga_lib::generate_arrayant_dipole<double>();
    auto ant2 = quadriga_lib::generate_arrayant_custom<double>(30.0, 30.0);
    double dir1_before = ant1.calc_directivity_dBi(0);
    double dir2_before = ant2.calc_directivity_dBi(0);

    std::vector<quadriga_lib::arrayant<double>> vec = {ant1, ant2};
    quadriga_lib::arrayant_rotate_pattern_multi(vec, 20.0, 30.0, 40.0, 0, {});

    double dir1_after = vec[0].calc_directivity_dBi(0);
    double dir2_after = vec[1].calc_directivity_dBi(0);
    CHECK(std::abs(dir1_before - dir1_after) < 0.3);
    CHECK(std::abs(dir2_before - dir2_after) < 0.3);
}

TEST_CASE("Rotate Pattern Multi - Float precision")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<float>();
    auto ref = ant.copy();

    std::vector<quadriga_lib::arrayant<float>> vec = {ant};
    quadriga_lib::arrayant_rotate_pattern_multi(vec, 0.0f, 0.0f, 360.0f, 0, {});

    float diff = max_pattern_diff(ref, vec[0]);
    CHECK(diff < 1e-3f);
}

// ================================================================================================
// Section 18: rotate_pattern - Stress tests and combined scenarios
// ================================================================================================

TEST_CASE("Rotate Pattern - Half-wave dipole directivity preserved through rotation")
{
    auto ant = quadriga_lib::generate_arrayant_half_wave_dipole<double>();
    double dir_before = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir_before - 2.15) < 0.01);

    ant.rotate_pattern(45.0, 30.0, 60.0, 0);

    double dir_after = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir_after - 2.15) < 0.3);
}

TEST_CASE("Rotate Pattern - ULA antenna rotation")
{
    auto ant = quadriga_lib::generate_arrayant_ula<double>(4, 3e9);
    REQUIRE(ant.n_elements() == 4);

    double dir0 = ant.calc_directivity_dBi(0);
    ant.rotate_pattern(0.0, 0.0, 90.0, 0);

    double dir0_after = ant.calc_directivity_dBi(0);
    CHECK(std::abs(dir0 - dir0_after) < 0.3);
    CHECK(ant.is_valid().empty());
}
