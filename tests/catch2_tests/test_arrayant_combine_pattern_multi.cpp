// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <string>
#include <vector>

// ================================================================================================
// Helper: Build a simple multi-frequency arrayant vector
// Each entry has 1 element, 1 elevation sample, 2 azimuth samples (0 and pi)
// Pattern values scale with frequency index for easy verification
// ================================================================================================
static std::vector<quadriga_lib::arrayant<double>>
build_simple_multi(const arma::vec &center_freqs, bool polarimetric = false)
{
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < center_freqs.n_elem; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double scale = (double)(i + 1); // 1, 2, 3, ...
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.slice(0) = arma::mat({{scale, -scale}});
        ant.e_theta_im.zeros(1, 2, 1);
        if (polarimetric)
        {
            ant.e_phi_re.zeros(1, 2, 1);
            ant.e_phi_re.slice(0) = arma::mat({{-0.5 * scale, 0.5 * scale}});
            ant.e_phi_im.zeros(1, 2, 1);
        }
        else
        {
            ant.e_phi_re.zeros(1, 2, 1);
            ant.e_phi_im.zeros(1, 2, 1);
        }
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.element_pos.zeros(3, 1);
        ant.coupling_re.ones(1, 1);
        ant.coupling_im.zeros(1, 1);
        ant.center_frequency = center_freqs[i];
        ant.name = "entry_" + std::to_string(i);
        vec.push_back(ant);
    }
    return vec;
}

// ================================================================================================
// Helper: Build a 2-element vector with non-trivial coupling
// Each entry has 2 elements at zero position (no phase shift needed)
// Coupling combines both elements into 1 port via [1; 1] (sum)
// ================================================================================================
static std::vector<quadriga_lib::arrayant<double>>
build_2elem_multi(const arma::vec &center_freqs)
{
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < center_freqs.n_elem; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double s = (double)(i + 1);
        ant.e_theta_re.zeros(1, 2, 2);
        ant.e_theta_re(0, 0, 0) = s;        // Elem 0, az=0
        ant.e_theta_re(0, 1, 0) = -s;       // Elem 0, az=pi
        ant.e_theta_re(0, 0, 1) = 2.0 * s;  // Elem 1, az=0
        ant.e_theta_re(0, 1, 1) = -2.0 * s; // Elem 1, az=pi
        ant.e_theta_im.zeros(1, 2, 2);
        ant.e_phi_re.zeros(1, 2, 2);
        ant.e_phi_im.zeros(1, 2, 2);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.element_pos.zeros(3, 2);
        // [1; 1] coupling: sums both elements into a single port
        ant.coupling_re.ones(2, 1);
        ant.coupling_im.zeros(2, 1);
        ant.center_frequency = center_freqs[i];
        vec.push_back(ant);
    }
    return vec;
}

// ================================================================================================
// SECTION 1: Input validation
// ================================================================================================

TEST_CASE("Combine multi - Empty vector throws")
{
    std::vector<quadriga_lib::arrayant<double>> empty_vec;
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_combine_pattern_multi(empty_vec),
                      std::invalid_argument);
}

TEST_CASE("Combine multi - Invalid arrayant_vec caught by validation")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    vec[1].azimuth_grid = {0.0}; // Corrupt: grid size mismatch
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_combine_pattern_multi(vec),
                      std::invalid_argument);
}

TEST_CASE("Combine multi - Out-of-range azimuth_grid_new throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::vec bad_az = {-4.0, 0.0, 4.0}; // Outside [-pi, pi]
    REQUIRE_THROWS_AS(quadriga_lib::arrayant_combine_pattern_multi(vec, &bad_az),
                      std::invalid_argument);
}

TEST_CASE("Combine multi - Out-of-range elevation_grid_new throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::vec bad_el = {-2.0, 0.0, 2.0}; // Outside [-pi/2, pi/2]
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, &bad_el),
        std::invalid_argument);
}

TEST_CASE("Combine multi - Non-positive freq_grid_new throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::vec bad_freq = {0.0, 1000.0};
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &bad_freq),
        std::invalid_argument);

    arma::vec neg_freq = {-100.0, 1000.0};
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &neg_freq),
        std::invalid_argument);
}

TEST_CASE("Combine multi - Unsorted freq_grid_new produces correct results")
{
    arma::vec entry_freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(entry_freqs);

    // Sorted query
    arma::vec freq_sorted = {1200.0, 1800.0, 2500.0};
    auto out_sorted = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_sorted);

    // Same frequencies in scrambled order
    arma::vec freq_unsorted = {2500.0, 1200.0, 1800.0};
    auto out_unsorted = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_unsorted);

    REQUIRE(out_sorted.size() == 3);
    REQUIRE(out_unsorted.size() == 3);

    // Match by frequency: out_unsorted[0]↔out_sorted[2], [1]↔[0], [2]↔[1]
    CHECK(arma::approx_equal(out_unsorted[0].e_theta_re, out_sorted[2].e_theta_re, "absdiff", 1e-12));
    CHECK(arma::approx_equal(out_unsorted[1].e_theta_re, out_sorted[0].e_theta_re, "absdiff", 1e-12));
    CHECK(arma::approx_equal(out_unsorted[2].e_theta_re, out_sorted[1].e_theta_re, "absdiff", 1e-12));

    CHECK(std::abs(out_unsorted[0].center_frequency - 2500.0) < 1e-6);
    CHECK(std::abs(out_unsorted[1].center_frequency - 1200.0) < 1e-6);
    CHECK(std::abs(out_unsorted[2].center_frequency - 1800.0) < 1e-6);
}

TEST_CASE("Combine multi - Reversed freq_grid_new (descending)")
{
    arma::vec entry_freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(entry_freqs);

    arma::vec freq_desc = {2000.0, 1000.0, 500.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_desc);

    REQUIRE(out.size() == 3);
    // At exact match each output should equal that entry's single-freq combine_pattern
    auto ref0 = vec[2].combine_pattern(); // 2000 Hz
    auto ref1 = vec[1].combine_pattern(); // 1000 Hz
    auto ref2 = vec[0].combine_pattern(); //  500 Hz
    CHECK(arma::approx_equal(out[0].e_theta_re, ref0.e_theta_re, "absdiff", 1e-10));
    CHECK(arma::approx_equal(out[1].e_theta_re, ref1.e_theta_re, "absdiff", 1e-10));
    CHECK(arma::approx_equal(out[2].e_theta_re, ref2.e_theta_re, "absdiff", 1e-10));
}

TEST_CASE("Combine multi - Duplicate freq_grid_new entries produce identical outputs")
{
    arma::vec entry_freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(entry_freqs);

    arma::vec freq_dup = {1500.0, 1500.0, 1500.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_dup);

    REQUIRE(out.size() == 3);
    CHECK(arma::approx_equal(out[0].e_theta_re, out[1].e_theta_re, "absdiff", 1e-12));
    CHECK(arma::approx_equal(out[0].e_theta_re, out[2].e_theta_re, "absdiff", 1e-12));
    for (auto &o : out)
        CHECK(std::abs(o.center_frequency - 1500.0) < 1e-6);
}

// ================================================================================================
// SECTION 2: Output structure
// ================================================================================================

TEST_CASE("Combine multi - Output count defaults to input count")
{
    arma::vec freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(freqs);
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    CHECK(out.size() == 3);
}

TEST_CASE("Combine multi - Output count matches freq_grid_new")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    arma::vec freq_out = {500.0, 1500.0, 2500.0, 3500.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);
    CHECK(out.size() == 4);
}

TEST_CASE("Combine multi - Each output entry has coupling = identity, element_pos = zeros")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_2elem_multi(freqs); // n_elements=2, n_ports=1
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);

    REQUIRE(out.size() == 2);
    for (auto &o : out)
    {
        // Each output has n_ports = 1 element, identity coupling [1x1]
        CHECK(o.n_elements() == 1);
        CHECK(o.n_ports() == 1);
        CHECK(o.coupling_re.n_rows == 1);
        CHECK(o.coupling_re.n_cols == 1);
        CHECK(std::abs(o.coupling_re(0, 0) - 1.0) < 1e-12);
        CHECK(std::abs(o.coupling_im(0, 0)) < 1e-12);
        // Element position should be all zeros
        CHECK(arma::approx_equal(o.element_pos, arma::mat(3, 1, arma::fill::zeros),
                                 "absdiff", 1e-12));
    }
}

TEST_CASE("Combine multi - Each output entry has correct pattern dimensions")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_2elem_multi(freqs);
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);

    for (auto &o : out)
    {
        CHECK(o.e_theta_re.n_rows == 1);   // n_elevation
        CHECK(o.e_theta_re.n_cols == 2);   // n_azimuth
        CHECK(o.e_theta_re.n_slices == 1); // n_ports (after coupling)
        CHECK(o.azimuth_grid.n_elem == 2);
        CHECK(o.elevation_grid.n_elem == 1);
    }
}

TEST_CASE("Combine multi - Output passes is_valid")
{
    arma::vec freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(freqs);
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    for (auto &o : out)
        CHECK(o.is_valid(false).empty());
}

TEST_CASE("Combine multi - Output center_frequency matches requested freq grid")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);

    // Default: per-entry center_frequency in input order
    auto out_default = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out_default.size() == 2);
    CHECK(std::abs(out_default[0].center_frequency - 1000.0) < 1e-9);
    CHECK(std::abs(out_default[1].center_frequency - 2000.0) < 1e-9);

    // Custom: requested values
    arma::vec freq_out = {750.0, 1500.0, 2500.0};
    auto out_custom = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);
    REQUIRE(out_custom.size() == 3);
    CHECK(std::abs(out_custom[0].center_frequency - 750.0) < 1e-9);
    CHECK(std::abs(out_custom[1].center_frequency - 1500.0) < 1e-9);
    CHECK(std::abs(out_custom[2].center_frequency - 2500.0) < 1e-9);
}

TEST_CASE("Combine multi - Output name inherited from arrayant_vec[0]")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    vec[0].name = "my_array";
    vec[1].name = "other_name";
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    for (auto &o : out)
        CHECK(o.name == "my_array");
}

// ================================================================================================
// SECTION 3: Single entry vector — equivalent to single-freq combine_pattern
// ================================================================================================

TEST_CASE("Combine multi - Single entry matches single-freq combine_pattern")
{
    arma::vec freqs = {1.0e9};
    auto vec = build_simple_multi(freqs, true);
    auto ref = vec[0].combine_pattern();
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);

    REQUIRE(out.size() == 1);
    CHECK(arma::approx_equal(out[0].e_theta_re, ref.e_theta_re, "absdiff", 1e-10));
    CHECK(arma::approx_equal(out[0].e_theta_im, ref.e_theta_im, "absdiff", 1e-10));
    CHECK(arma::approx_equal(out[0].e_phi_re, ref.e_phi_re, "absdiff", 1e-10));
    CHECK(arma::approx_equal(out[0].e_phi_im, ref.e_phi_im, "absdiff", 1e-10));
}

TEST_CASE("Combine multi - Single entry, clamped query frequency uses that entry's pattern")
{
    arma::vec freqs = {1.0e9};
    auto vec = build_simple_multi(freqs);

    // Query at lots of different frequencies — all should clamp to entry 0
    arma::vec freq_out = {1.0e6, 1.0e9, 1.0e12};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);
    REQUIRE(out.size() == 3);

    // Field amplitudes should be identical (clamped), only the center_frequency changes
    for (arma::uword f = 0; f < 3; ++f)
    {
        CHECK(std::abs(out[f].e_theta_re(0, 0, 0) - 1.0) < 1e-10);
        CHECK(std::abs(out[f].e_theta_re(0, 1, 0) - (-1.0)) < 1e-10);
    }
}

// ================================================================================================
// SECTION 4: Exact frequency match — output matches single-freq combine_pattern per entry
// ================================================================================================

TEST_CASE("Combine multi - Default output matches per-entry single-freq combine_pattern")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs, true);
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);

    REQUIRE(out.size() == 3);
    for (arma::uword f = 0; f < 3; ++f)
    {
        auto ref = vec[f].combine_pattern();
        CHECK(arma::approx_equal(out[f].e_theta_re, ref.e_theta_re, "absdiff", 1e-10));
        CHECK(arma::approx_equal(out[f].e_theta_im, ref.e_theta_im, "absdiff", 1e-10));
        CHECK(arma::approx_equal(out[f].e_phi_re, ref.e_phi_re, "absdiff", 1e-10));
        CHECK(arma::approx_equal(out[f].e_phi_im, ref.e_phi_im, "absdiff", 1e-10));
    }
}

TEST_CASE("Combine multi - 2-element coupling sums elements correctly at exact freq")
{
    arma::vec freqs = {1000.0};
    auto vec = build_2elem_multi(freqs); // Each entry: elem0=s, elem1=2s, coupling=[1;1]

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 1);

    // Port output = elem0 + elem1 = s + 2s = 3s, with s=1 → 3.0 at az=0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 3.0) < 1e-10);
    CHECK(std::abs(out[0].e_theta_re(0, 1, 0) - (-3.0)) < 1e-10);
}

// ================================================================================================
// SECTION 5: Custom output grids
// ================================================================================================

TEST_CASE("Combine multi - Custom azimuth_grid_new used in output")
{
    // Build an entry with broad angular coverage
    double pi = arma::datum::pi;
    auto base = quadriga_lib::generate_arrayant_omni<double>();
    base.center_frequency = 1.0e9;
    std::vector<quadriga_lib::arrayant<double>> vec = {base};

    arma::vec az_new = {-pi / 2.0, 0.0, pi / 2.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec, &az_new);

    REQUIRE(out.size() == 1);
    CHECK(out[0].azimuth_grid.n_elem == 3);
    CHECK(arma::approx_equal(out[0].azimuth_grid, az_new, "absdiff", 1e-12));
    CHECK(out[0].e_theta_re.n_cols == 3);
}

TEST_CASE("Combine multi - Custom elevation_grid_new used in output")
{
    double pi = arma::datum::pi;
    auto base = quadriga_lib::generate_arrayant_omni<double>();
    base.center_frequency = 1.0e9;
    std::vector<quadriga_lib::arrayant<double>> vec = {base};

    arma::vec el_new = {-pi / 4.0, 0.0, pi / 4.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, &el_new);

    REQUIRE(out.size() == 1);
    CHECK(out[0].elevation_grid.n_elem == 3);
    CHECK(arma::approx_equal(out[0].elevation_grid, el_new, "absdiff", 1e-12));
    CHECK(out[0].e_theta_re.n_rows == 3);
}

TEST_CASE("Combine multi - All three grids custom")
{
    double pi = arma::datum::pi;
    auto base = quadriga_lib::generate_arrayant_omni<double>();
    base.center_frequency = 1.0e9;
    std::vector<quadriga_lib::arrayant<double>> vec = {base};

    arma::vec az_new = {-pi / 2.0, 0.0, pi / 2.0};
    arma::vec el_new = {-pi / 6.0, 0.0, pi / 6.0};
    arma::vec freq_new = {0.5e9, 1.0e9, 1.5e9};
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec, &az_new, &el_new, &freq_new);

    REQUIRE(out.size() == 3);
    for (auto &o : out)
    {
        CHECK(o.n_azimuth() == 3);
        CHECK(o.n_elevation() == 3);
        CHECK(o.e_theta_re.n_rows == 3);
        CHECK(o.e_theta_re.n_cols == 3);
    }
}

// ================================================================================================
// SECTION 6: Frequency interpolation (between input entries)
// ================================================================================================

TEST_CASE("Combine multi - Midpoint frequency interpolation (in-phase reals)")
{
    // Two entries with constant patterns; in-phase → SLERP collapses to linear amplitude blend.
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double val = (i == 0) ? 1.0 : 3.0;
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.fill(val);
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.element_pos.zeros(3, 1);
        ant.coupling_re.ones(1, 1);
        ant.coupling_im.zeros(1, 1);
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::vec freq_out = {1500.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);
    REQUIRE(out.size() == 1);

    // SLERP with in-phase → linear blend: 0.5*1 + 0.5*3 = 2.0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 2.0) < 1e-9);
}

TEST_CASE("Combine multi - Coupling SLERPed between bracketing entries")
{
    // 1 element, 1 port, but coupling magnitude changes with freq
    // Entry 0: coupling = 1.0 + 0j, freq = 1000
    // Entry 1: coupling = 3.0 + 0j, freq = 2000
    // Pattern is constant = 1.0 (real) at all entries.
    // At f=1500 (midpoint): coupling SLERP = 2.0 (in-phase real linear blend);
    //                       pattern SLERP = 1.0 (no variation); result = 2.0.
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.fill(1.0); // Constant pattern across entries
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.element_pos.zeros(3, 1);
        ant.coupling_re.set_size(1, 1);
        ant.coupling_re(0, 0) = (i == 0) ? 1.0 : 3.0;
        ant.coupling_im.zeros(1, 1);
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::vec freq_out = {1500.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    // result = coupling(1500) * pattern(1500) = 2.0 * 1.0 = 2.0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 2.0) < 1e-9);
}

TEST_CASE("Combine multi - Coupling magnitude SLERP equals nearest at quarter point")
{
    // Pattern is constant 1.0; coupling magnitudes |c0|=2, |c1|=6 in-phase reals.
    // At w=0.25: coupling = 0.75*2 + 0.25*6 = 3.0  → output = 3.0
    // At w=0.75: coupling = 0.25*2 + 0.75*6 = 5.0  → output = 5.0
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.fill(1.0);
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.element_pos.zeros(3, 1);
        ant.coupling_re.set_size(1, 1);
        ant.coupling_re(0, 0) = (i == 0) ? 2.0 : 6.0;
        ant.coupling_im.zeros(1, 1);
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::vec freq_out = {1250.0, 1750.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);
    REQUIRE(out.size() == 2);
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 3.0) < 1e-9);
    CHECK(std::abs(out[1].e_theta_re(0, 0, 0) - 5.0) < 1e-9);
}

// ================================================================================================
// SECTION 7: Extrapolation (clamping)
// ================================================================================================

TEST_CASE("Combine multi - Query below lowest center_freq clamps to first entry")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);

    arma::vec freq_out = {10.0}; // Way below lowest
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    auto ref = vec[0].combine_pattern();
    REQUIRE(out.size() == 1);
    CHECK(arma::approx_equal(out[0].e_theta_re, ref.e_theta_re, "absdiff", 1e-9));
    CHECK(arma::approx_equal(out[0].e_theta_im, ref.e_theta_im, "absdiff", 1e-9));
}

TEST_CASE("Combine multi - Query above highest center_freq clamps to last entry")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);

    arma::vec freq_out = {1.0e9}; // Way above highest
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    auto ref = vec[2].combine_pattern();
    REQUIRE(out.size() == 1);
    CHECK(arma::approx_equal(out[0].e_theta_re, ref.e_theta_re, "absdiff", 1e-9));
    CHECK(arma::approx_equal(out[0].e_theta_im, ref.e_theta_im, "absdiff", 1e-9));
}

// ================================================================================================
// SECTION 8: No-coupling (empty coupling matrices) case
// ================================================================================================

TEST_CASE("Combine multi - Empty coupling treated as identity")
{
    // Build a 2-element vector; first entry has empty coupling, but identity-shape inferred.
    // Per arrayant_is_valid_multi, all entries must have same coupling shape (or all empty).
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double s = (double)(i + 1);
        ant.e_theta_re.zeros(1, 2, 2);
        ant.e_theta_re(0, 0, 0) = s;
        ant.e_theta_re(0, 1, 0) = -s;
        ant.e_theta_re(0, 0, 1) = 2.0 * s;
        ant.e_theta_re(0, 1, 1) = -2.0 * s;
        ant.e_theta_im.zeros(1, 2, 2);
        ant.e_phi_re.zeros(1, 2, 2);
        ant.e_phi_im.zeros(1, 2, 2);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.element_pos.zeros(3, 2);
        // Leave coupling empty
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 2);

    // With identity coupling, output has 2 ports, each = the corresponding element's pattern.
    // Entry 0: elem0 = 1 (at az=0), elem1 = 2.
    CHECK(out[0].n_elements() == 2);
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 1.0) < 1e-10); // port 0 = elem 0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 1) - 2.0) < 1e-10); // port 1 = elem 1
}

// ================================================================================================
// SECTION 9: Multi-element pattern combination at a single output frequency
// ================================================================================================

TEST_CASE("Combine multi - Cross-frequency, multi-element combination")
{
    // 2-elem antenna, coupling [1;1] sums both elements.
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_2elem_multi(freqs);

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 2);

    // Entry 0: s=1 → elem0=1, elem1=2 → sum=3 at az=0
    // Entry 1: s=2 → elem0=2, elem1=4 → sum=6 at az=0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 3.0) < 1e-10);
    CHECK(std::abs(out[1].e_theta_re(0, 0, 0) - 6.0) < 1e-10);

    // n_ports = 1 → output has 1 element
    CHECK(out[0].n_elements() == 1);
    CHECK(out[1].n_elements() == 1);
}

TEST_CASE("Combine multi - Multi-port coupling produces multi-element output")
{
    // 2-elem antenna with 2 ports: port 0 = elem0, port 1 = elem1
    double pi = arma::datum::pi;
    arma::vec freqs = {1000.0};
    auto vec = build_2elem_multi(freqs);
    // Replace [1;1] coupling with [[1,0],[0,1]] (2 ports, identity)
    vec[0].coupling_re.set_size(2, 2);
    vec[0].coupling_re(0, 0) = 1.0;
    vec[0].coupling_re(0, 1) = 0.0;
    vec[0].coupling_re(1, 0) = 0.0;
    vec[0].coupling_re(1, 1) = 1.0;
    vec[0].coupling_im.zeros(2, 2);

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 1);
    CHECK(out[0].n_elements() == 2);                           // 2 ports
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 1.0) < 1e-10); // port 0 = elem 0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 1) - 2.0) < 1e-10); // port 1 = elem 1
}

// ================================================================================================
// SECTION 10: Element-position phase shift (frequency-dependent)
// ================================================================================================

TEST_CASE("Combine multi - Single elem at zero pos: result is real (no phase shift)")
{
    // Single element with constant real pattern at zero position.
    // Phase shift should be unity at any frequency.
    arma::vec freqs = {1.0e9, 5.0e9};
    auto vec = build_simple_multi(freqs);

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 2);
    for (auto &o : out)
    {
        CHECK(arma::all(arma::vectorise(arma::abs(o.e_theta_im)) < 1e-12));
    }
}

TEST_CASE("Combine multi - Element position introduces frequency-dependent phase")
{
    // Single-element antenna at x = lambda/4 at 1 GHz → phase shift = -2pi/lambda * (lambda/4) = -pi/2 at az=0.
    // At 1 GHz, lambda = c / 1e9 ≈ 0.2998 m, so x = 0.0749481 m.
    double pi = arma::datum::pi;
    const double C0 = 299792458.0;
    double f0 = 1.0e9;
    double lambda0 = C0 / f0;
    double x_pos = lambda0 / 4.0;

    quadriga_lib::arrayant<double> ant;
    ant.e_theta_re.ones(1, 1, 1); // Pattern = 1 + 0j at az=0, el=0
    ant.e_theta_im.zeros(1, 1, 1);
    ant.e_phi_re.zeros(1, 1, 1);
    ant.e_phi_im.zeros(1, 1, 1);
    ant.azimuth_grid = {0.0};
    ant.elevation_grid = {0.0};
    ant.element_pos.set_size(3, 1);
    ant.element_pos(0, 0) = x_pos;
    ant.element_pos(1, 0) = 0.0;
    ant.element_pos(2, 0) = 0.0;
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.center_frequency = f0;

    std::vector<quadriga_lib::arrayant<double>> vec = {ant};
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);

    REQUIRE(out.size() == 1);
    // Cross-check against single-freq combine_pattern (also negates dist)
    auto ref = ant.combine_pattern();
    // Allow a slightly looser tolerance: single-freq uses c=299792448 (typo), multi uses c=299792458.
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - ref.e_theta_re(0, 0, 0)) < 1e-5);
    CHECK(std::abs(out[0].e_theta_im(0, 0, 0) - ref.e_theta_im(0, 0, 0)) < 1e-5);

    // Direct check: phase = exp(-j * k * dist), where dist = -(x_pos * cos(el) * cos(az)) = -x_pos at az=el=0
    // So phase = exp(+j * k * x_pos) = exp(+j * pi/2) = (0, +1)
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 0.0) < 1e-5);
    CHECK(std::abs(out[0].e_theta_im(0, 0, 0) - 1.0) < 1e-5);
}

TEST_CASE("Combine multi - Doubling frequency doubles the phase shift")
{
    // Same antenna, output at f0 and 2*f0 → phase shift doubles at higher freq.
    const double C0 = 299792458.0;
    double f0 = 1.0e9;
    double lambda0 = C0 / f0;
    double x_pos = lambda0 / 8.0; // phase = -k*(-x) = +k*x → +pi/4 at f0, +pi/2 at 2*f0

    quadriga_lib::arrayant<double> ant;
    ant.e_theta_re.ones(1, 1, 1);
    ant.e_theta_im.zeros(1, 1, 1);
    ant.e_phi_re.zeros(1, 1, 1);
    ant.e_phi_im.zeros(1, 1, 1);
    ant.azimuth_grid = {0.0};
    ant.elevation_grid = {0.0};
    ant.element_pos.set_size(3, 1);
    ant.element_pos(0, 0) = x_pos;
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.center_frequency = f0;

    std::vector<quadriga_lib::arrayant<double>> vec = {ant};
    arma::vec freq_out = {f0, 2.0 * f0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    REQUIRE(out.size() == 2);

    // At f0: phase = +pi/4 → (cos(pi/4), sin(pi/4)) ≈ (0.7071, 0.7071)
    double rs2 = 1.0 / std::sqrt(2.0);
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - rs2) < 1e-5);
    CHECK(std::abs(out[0].e_theta_im(0, 0, 0) - rs2) < 1e-5);

    // At 2*f0: phase = +pi/2 → (0, 1)
    CHECK(std::abs(out[1].e_theta_re(0, 0, 0) - 0.0) < 1e-5);
    CHECK(std::abs(out[1].e_theta_im(0, 0, 0) - 1.0) < 1e-5);
}

TEST_CASE("Combine multi - Negated dist sign matches single-freq combine_pattern")
{
    // Critical regression test: ensure dist sign matches qd_arrayant_interpolate convention.
    // Use a position that produces a non-trivial phase, then compare to single-freq.
    const double C0 = 299792458.0;
    double f0 = 2.4e9;
    double lambda0 = C0 / f0;

    quadriga_lib::arrayant<double> ant;
    ant.e_theta_re.ones(1, 1, 1);
    ant.e_theta_im.zeros(1, 1, 1);
    ant.e_phi_re.zeros(1, 1, 1);
    ant.e_phi_im.zeros(1, 1, 1);
    ant.azimuth_grid = {0.0};
    ant.elevation_grid = {0.0};
    ant.element_pos.set_size(3, 1);
    ant.element_pos(0, 0) = lambda0 * 0.137; // odd value to ensure non-zero/non-cardinal phase
    ant.element_pos(1, 0) = lambda0 * 0.211;
    ant.element_pos(2, 0) = lambda0 * 0.064;
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.center_frequency = f0;

    std::vector<quadriga_lib::arrayant<double>> vec = {ant};
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    auto ref = ant.combine_pattern();

    REQUIRE(out.size() == 1);
    // Loose tolerance because of c=299792448 vs 299792458 difference; sign mismatch would be ~1.0
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - ref.e_theta_re(0, 0, 0)) < 1e-5);
    CHECK(std::abs(out[0].e_theta_im(0, 0, 0) - ref.e_theta_im(0, 0, 0)) < 1e-5);
}

// ================================================================================================
// SECTION 11: Frequency-dependent pattern and coupling combined
// ================================================================================================

TEST_CASE("Combine multi - GHz antenna with varying pattern and coupling")
{
    arma::vec ghz_freqs = {2.4e9, 2.45e9, 2.5e9};

    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < ghz_freqs.n_elem; ++i)
    {
        auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
        ant.center_frequency = ghz_freqs[i];
        double scale = 1.0 + 0.1 * (double)i;
        ant.e_theta_re *= scale;
        ant.e_theta_im *= scale;
        vec.push_back(ant);
    }

    // Query at WiFi channel frequencies — most are between input entries
    arma::vec freq_out = {2.412e9, 2.437e9, 2.462e9, 2.484e9};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    REQUIRE(out.size() == 4);
    for (auto &o : out)
    {
        CHECK(o.is_valid(false).empty());
        CHECK(o.e_theta_re.is_finite());
        CHECK(!o.e_theta_re.has_nan());
    }

    // On-axis gain should be monotonically non-decreasing (scale grows with freq, pattern in-phase)
    arma::uword n_el = out[0].n_elevation();
    arma::uword n_az = out[0].n_azimuth();
    double g0 = std::abs(out[0].e_theta_re(n_el / 2, n_az / 2, 0));
    double g3 = std::abs(out[3].e_theta_re(n_el / 2, n_az / 2, 0));
    CHECK(g3 >= g0 - 1e-9);
}

// ================================================================================================
// SECTION 12: Acoustic / generate_speaker workflow
// ================================================================================================

TEST_CASE("Combine multi - Single speaker generate_speaker workflow")
{
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>(
        "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
        0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    // Combine without changing the frequency grid — output should be 1:1 with input
    auto out = quadriga_lib::arrayant_combine_pattern_multi(spk);
    REQUIRE(out.size() == spk.size());

    // Each output entry has identity coupling and matches single-freq combine_pattern
    for (arma::uword f = 0; f < spk.size(); ++f)
    {
        auto ref = spk[f].combine_pattern();
        CHECK(arma::approx_equal(out[f].e_theta_re, ref.e_theta_re, "absdiff", 1e-9));
        CHECK(arma::approx_equal(out[f].e_theta_im, ref.e_theta_im, "absdiff", 1e-9));
        CHECK(std::abs(out[f].center_frequency - spk[f].center_frequency) < 1e-6);
    }
}

TEST_CASE("Combine multi - 2-way speaker workflow with arrayant_concat_multi")
{
    arma::vec entry_freqs = {100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 16000.0};
    auto woofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, entry_freqs, 10.0);
    auto tweeter = quadriga_lib::generate_speaker<double>(
        "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, entry_freqs, 10.0);
    auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);
    // speaker has 2 elements; default coupling is [[1,0],[0,1]] (2 ports)
    REQUIRE(speaker[0].n_elements() == 2);
    REQUIRE(speaker[0].coupling_re.n_cols == 2);

    // Combine at output frequencies that fall inside the input range
    arma::vec freq_out = {150.0, 750.0, 3000.0, 8000.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(speaker, nullptr, nullptr, &freq_out);

    REQUIRE(out.size() == 4);
    for (auto &o : out)
    {
        CHECK(o.is_valid(false).empty());
        CHECK(o.n_elements() == 2); // 2 ports → 2 output "elements"
        CHECK(o.e_theta_re.is_finite());
        CHECK(!o.e_theta_re.has_nan());
    }

    // At 150 Hz (within first bracket), woofer (port 0) should dominate over tweeter (port 1)
    arma::uword el_mid = out[0].n_elevation() / 2;
    arma::uword az_mid = out[0].n_azimuth() / 2;
    double w150 = std::sqrt(out[0].e_theta_re(el_mid, az_mid, 0) * out[0].e_theta_re(el_mid, az_mid, 0) +
                            out[0].e_theta_im(el_mid, az_mid, 0) * out[0].e_theta_im(el_mid, az_mid, 0));
    double t150 = std::sqrt(out[0].e_theta_re(el_mid, az_mid, 1) * out[0].e_theta_re(el_mid, az_mid, 1) +
                            out[0].e_theta_im(el_mid, az_mid, 1) * out[0].e_theta_im(el_mid, az_mid, 1));
    CHECK(w150 > t150 * 5.0);

    // At 8 kHz, tweeter should dominate
    double w8k = std::sqrt(out[3].e_theta_re(el_mid, az_mid, 0) * out[3].e_theta_re(el_mid, az_mid, 0) +
                           out[3].e_theta_im(el_mid, az_mid, 0) * out[3].e_theta_im(el_mid, az_mid, 0));
    double t8k = std::sqrt(out[3].e_theta_re(el_mid, az_mid, 1) * out[3].e_theta_re(el_mid, az_mid, 1) +
                           out[3].e_theta_im(el_mid, az_mid, 1) * out[3].e_theta_im(el_mid, az_mid, 1));
    CHECK(t8k > w8k * 1.5);
}

// ================================================================================================
// SECTION 13: Dense frequency sweep
// ================================================================================================

TEST_CASE("Combine multi - Dense sweep produces smooth finite output")
{
    arma::vec entry_freqs = {500.0, 1000.0, 2000.0, 5000.0};
    auto vec = build_simple_multi(entry_freqs);

    arma::vec freq_out = arma::linspace(100.0, 10000.0, 50);
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    REQUIRE(out.size() == 50);
    for (auto &o : out)
    {
        CHECK(o.e_theta_re.is_finite());
        CHECK(!o.e_theta_re.has_nan());
    }

    // At az=0: pattern values are scale=1,2,3,4 (in-phase real). Output should be monotone non-decreasing.
    for (arma::uword i = 1; i < 50; ++i)
        CHECK(out[i].e_theta_re(0, 0, 0) >= out[i - 1].e_theta_re(0, 0, 0) - 1e-9);
}

// ================================================================================================
// SECTION 14: Unsorted input center_frequencies still work
// ================================================================================================

TEST_CASE("Combine multi - Input arrayant_vec in unsorted frequency order")
{
    // Build entries in non-sorted order: 2000, 500, 1000
    arma::vec freqs_unsorted = {2000.0, 500.0, 1000.0};
    auto vec = build_simple_multi(freqs_unsorted);

    // Query at sorted output frequencies — should still produce sensible results
    arma::vec freq_out = {500.0, 1000.0, 2000.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    REQUIRE(out.size() == 3);
    // Output at f=500 should match vec[1] (which has center_frequency=500, scale=2)
    auto ref_500 = vec[1].combine_pattern();
    CHECK(arma::approx_equal(out[0].e_theta_re, ref_500.e_theta_re, "absdiff", 1e-9));
    // Output at f=2000 should match vec[0] (scale=1)
    auto ref_2000 = vec[0].combine_pattern();
    CHECK(arma::approx_equal(out[2].e_theta_re, ref_2000.e_theta_re, "absdiff", 1e-9));
}

// ================================================================================================
// SECTION 15: Single output frequency
// ================================================================================================

TEST_CASE("Combine multi - Single output frequency entry produces single output")
{
    arma::vec entry_freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(entry_freqs);

    arma::vec freq_out = {1500.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi<double>(vec, nullptr, nullptr, &freq_out);

    REQUIRE(out.size() == 1);
    CHECK(std::abs(out[0].center_frequency - 1500.0) < 1e-6);
}

// ================================================================================================
// SECTION 16: Float instantiation works
// ================================================================================================

TEST_CASE("Combine multi - Float instantiation")
{
    auto ant_f = quadriga_lib::generate_arrayant_omni<float>();
    ant_f.center_frequency = 1.0e9f;
    std::vector<quadriga_lib::arrayant<float>> vec = {ant_f};

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 1);
    CHECK(out[0].is_valid(false).empty());
}

// ================================================================================================
// SECTION 17: Polarimetric arrayant — both V and H preserved
// ================================================================================================

TEST_CASE("Combine multi - Polarimetric pattern preserves V and H")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs, true);

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 2);

    // At entry 0 (scale=1): e_theta_re at az=0 → 1, e_phi_re at az=0 → -0.5
    CHECK(std::abs(out[0].e_theta_re(0, 0, 0) - 1.0) < 1e-9);
    CHECK(std::abs(out[0].e_phi_re(0, 0, 0) - (-0.5)) < 1e-9);

    // At entry 1 (scale=2):
    CHECK(std::abs(out[1].e_theta_re(0, 0, 0) - 2.0) < 1e-9);
    CHECK(std::abs(out[1].e_phi_re(0, 0, 0) - (-1.0)) < 1e-9);
}

// ================================================================================================
// SECTION 18: Validation can be combined with custom grids
// ================================================================================================

TEST_CASE("Combine multi - Custom angular grid resamples pattern correctly")
{
    double pi = arma::datum::pi;
    // Use a omni antenna with constant pattern = 1.0 across full sphere
    auto base = quadriga_lib::generate_arrayant_omni<double>();
    base.center_frequency = 1.0e9;
    std::vector<quadriga_lib::arrayant<double>> vec = {base};

    arma::vec az_new = {-pi / 2.0, 0.0, pi / 2.0};
    arma::vec el_new = {0.0};
    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec, &az_new, &el_new);

    REQUIRE(out.size() == 1);
    // Omni has constant V-pol on-axis → e_theta_re ≈ 1 everywhere on equator (within numerical noise)
    for (arma::uword a = 0; a < 3; ++a)
        CHECK(std::abs(out[0].e_theta_re(0, a, 0) - 1.0) < 1e-6);
}

// ================================================================================================
// SECTION 19: Two-step equivalence (combine, then re-interpolate)
// ================================================================================================

TEST_CASE("Combine multi - Default freq grid recovers per-entry combine_pattern (multi-element)")
{
    arma::vec freqs = {1.0e9, 2.0e9, 3.0e9};
    auto vec = build_2elem_multi(freqs); // 2 elem, 1 port

    auto out = quadriga_lib::arrayant_combine_pattern_multi(vec);
    REQUIRE(out.size() == 3);

    for (arma::uword f = 0; f < 3; ++f)
    {
        auto ref = vec[f].combine_pattern();
        CHECK(arma::approx_equal(out[f].e_theta_re, ref.e_theta_re, "absdiff", 1e-9));
        CHECK(arma::approx_equal(out[f].e_theta_im, ref.e_theta_im, "absdiff", 1e-9));
    }
}
