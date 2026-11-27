// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Antenna array structure and operations: is_valid, the block-diagonal coupling of join_arrays, trim_pattern,
// copy_element and the output checks of calc_beamwidth. Oracles are hand-written grids and coupling
// matrices; the short-dipole directivity is 1.76 dBi (10 log10(1.5)).

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_arrayant.hpp"

#include <cmath>
#include <type_traits>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("Arrayant append")
{
    auto ant1 = quadriga_lib::generate_antenna<float>("dipole");
    auto ant2 = quadriga_lib::generate_antenna<float>("custom", 1.0, NAN, {10.0, 10.0});

    ant1.port_weights_re.at(0) = 2.0;
    ant2.port_weights_re.at(0) = 3.0;

    auto ant = quadriga_lib::join_arrays(ant1, ant2);

    arma::fmat T;
    T = {{2.0, 0.0}, {0.0, 3.0}};
    CHECK(arma::approx_equal(ant.port_weights_re, T, "absdiff", 1.0e-6f));

    float g0 = qd_directivity(ant, 0);
    float g1 = qd_directivity(ant, 1);

    CHECK(std::abs(g0 - 1.760964f) < 0.0001f);
    CHECK(std::abs(g1 - 25.627f) < 0.001f);
}

// is_valid always checks the data: a value changed in place is found although the data pointers are unchanged
TEST_CASE("Arrayant is_valid - Checks the data")
{
    auto a = quadriga_lib::generate_antenna<double>("omni", 30.0);
    a.azimuth_grid.memptr()[1] = 4.0; // Out of range, written in place
    CHECK(a.is_valid() == "Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).");

    // Empty element_pos and coupling are allowed
    quadriga_lib::arrayant<double> c;
    c.e_theta_re.ones(2, 3, 1), c.e_theta_im.zeros(2, 3, 1), c.e_phi_re.zeros(2, 3, 1), c.e_phi_im.zeros(2, 3, 1);
    c.azimuth_grid = {-1.0, 0.0, 1.0}, c.elevation_grid = {-0.5, 0.5};
    CHECK(c.is_valid().empty());

    // Range errors are reported before order errors, azimuth before elevation
    c.azimuth_grid = {1.0, 0.0, 4.0}; // Unsorted and out of range
    CHECK(c.is_valid() == "Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).");
    c.azimuth_grid = {1.0, 0.0, -1.0}, c.elevation_grid = {0.5, 2.0}; // Azimuth unsorted, elevation out of range
    CHECK(c.is_valid() == "Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");
    c.elevation_grid = {0.5, -0.5}; // Both unsorted
    CHECK(c.is_valid() == "Values of 'azimuth_grid' must be sorted in strictly ascending order.");
    c.azimuth_grid = {-1.0, 0.0, 1.0};
    CHECK(c.is_valid() == "Values of 'elevation_grid' must be sorted in strictly ascending order.");
    c.elevation_grid = {-0.5, std::nan("")};
    CHECK(c.is_valid() == "Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");
}

// Append must build the block-diagonal coupling [C1 0; 0 C2] (as join_arrays), also when the number
// of ports differs from the number of elements, and treat an empty port_weights_im as zeros.
TEST_CASE("Arrayant append - Block-diagonal coupling with fewer ports than elements")
{
    auto a = quadriga_lib::generate_antenna<double>("omni", 10.0);
    quadriga_lib::copy_element(a, 0, {1});
    a.port_weights_re = arma::mat(2, 1, arma::fill::ones); // 2 elements -> 1 port
    a.port_weights_im = arma::mat(arma::colvec{0.5, -0.5});

    auto b = quadriga_lib::generate_antenna<double>("xpol", 10.0); // 2 elements, 2 ports (identity)
    b.port_weights_re = {{2.0, 0.0}, {0.0, 3.0}};
    b.port_weights_im = {{0.0, 0.25}, {0.0, 0.0}};

    auto c = quadriga_lib::join_arrays(a, a);
    arma::mat T = {{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}};
    CHECK(arma::approx_equal(c.port_weights_re, T, "absdiff", 1e-14));
    T = {{0.5, 0.0}, {-0.5, 0.0}, {0.0, 0.5}, {0.0, -0.5}};
    CHECK(arma::approx_equal(c.port_weights_im, T, "absdiff", 1e-14));
    CHECK(c.n_ports() == 2);

    auto cm = quadriga_lib::join_arrays(std::vector<quadriga_lib::arrayant<double>>{a},
                                                  std::vector<quadriga_lib::arrayant<double>>{a});
    CHECK(arma::approx_equal(c.port_weights_re, cm[0].port_weights_re, "absdiff", 1e-14));
    CHECK(arma::approx_equal(c.port_weights_im, cm[0].port_weights_im, "absdiff", 1e-14));

    // Different port counts: 2x1 block followed by a 2x2 block
    auto d = quadriga_lib::join_arrays(a, b);
    T = {{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
    CHECK(arma::approx_equal(d.port_weights_re, T, "absdiff", 1e-14));
    T = {{0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {0.0, 0.0, 0.25}, {0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(d.port_weights_im, T, "absdiff", 1e-14));
    auto dm = quadriga_lib::join_arrays(std::vector<quadriga_lib::arrayant<double>>{a},
                                                  std::vector<quadriga_lib::arrayant<double>>{b});
    CHECK(arma::approx_equal(d.port_weights_re, dm[0].port_weights_re, "absdiff", 1e-14));
    CHECK(arma::approx_equal(d.port_weights_im, dm[0].port_weights_im, "absdiff", 1e-14));
    CHECK(d.is_valid().empty());
}

TEST_CASE("Arrayant append - Empty port_weights_im is treated as zeros")
{
    auto b = quadriga_lib::generate_antenna<double>("omni", 10.0);
    b.port_weights_re = arma::mat(1, 1, arma::fill::value(2.0));
    b.port_weights_im.reset();
    REQUIRE(b.is_valid().empty());

    quadriga_lib::arrayant<double> d;
    REQUIRE_NOTHROW(d = quadriga_lib::join_arrays(b, b));
    arma::mat T = {{2.0, 0.0}, {0.0, 2.0}};
    CHECK(arma::approx_equal(d.port_weights_re, T, "absdiff", 1e-14));
    CHECK(d.port_weights_im.n_rows == 2);
    CHECK(d.port_weights_im.n_cols == 2);
    CHECK(arma::all(arma::vectorise(d.port_weights_im) == 0.0));

    // Both couplings empty (allowed by is_valid): identity blocks
    auto e = quadriga_lib::generate_antenna<double>("xpol", 10.0);
    e.port_weights_re.reset(), e.port_weights_im.reset();
    REQUIRE(e.is_valid().empty());
    auto f = quadriga_lib::join_arrays(e, b);
    T = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}};
    CHECK(arma::approx_equal(f.port_weights_re, T, "absdiff", 1e-14));
    CHECK(arma::all(arma::vectorise(f.port_weights_im) == 0.0));

    // join_arrays with empty couplings
    std::vector<quadriga_lib::arrayant<double>> fm;
    REQUIRE_NOTHROW(fm = quadriga_lib::join_arrays(std::vector<quadriga_lib::arrayant<double>>{e},
                                                             std::vector<quadriga_lib::arrayant<double>>{b}));
    CHECK(arma::approx_equal(fm[0].port_weights_re, T, "absdiff", 1e-14));
    CHECK(arma::all(arma::vectorise(fm[0].port_weights_im) == 0.0));
}

// Remove_zeros and calc_directivity must not depend on the absolute scale of the pattern; an all-zero
// pattern is left unchanged.
TEST_CASE("Arrayant remove_zeros and calc_directivity - Low-level patterns")
{
    auto a = quadriga_lib::generate_antenna<double>("custom", 5.0, NAN, {90.0, 90.0}); // 37 x 73
    const double d_ref = qd_directivity(a, 0);
    quadriga_lib::arrayant<double> o_ref;
    o_ref = a, quadriga_lib::trim_pattern(o_ref);

    for (double s : {1e-6, 1e-8, 1e-12, 1e-30, 1e160, 1e300}) // power above the double range at 1e160
    {
        auto b = a;
        b.e_theta_re *= s;
        CHECK(std::abs(qd_directivity(b, 0) - d_ref) < 1e-9);
        quadriga_lib::arrayant<double> o;
        o = b, quadriga_lib::trim_pattern(o);
        CHECK(o.n_elevation() == o_ref.n_elevation());
        CHECK(o.n_azimuth() == o_ref.n_azimuth());
        CHECK(arma::approx_equal(o.e_theta_re, o_ref.e_theta_re * s, "reldiff", 1e-12));
    }

    // Float pattern scaled by 1e-8 (power 1e-16)
    auto af = quadriga_lib::generate_antenna<float>("custom", 5.0f, NAN, {90.0f, 90.0f});
    const float df_ref = qd_directivity(af, 0);
    quadriga_lib::arrayant<float> of_ref;
    of_ref = af, quadriga_lib::trim_pattern(of_ref);
    af.e_theta_re *= 1e-8f;
    CHECK(std::abs(qd_directivity(af, 0) - df_ref) < 1e-4f);
    quadriga_lib::arrayant<float> of;
    of = af, quadriga_lib::trim_pattern(of);
    CHECK(of.n_elevation() == of_ref.n_elevation());
    CHECK(of.n_azimuth() == of_ref.n_azimuth());

    // All-zero pattern: unchanged (in place and into an output), directivity 0
    auto z = quadriga_lib::generate_antenna<double>("omni", 10.0);
    z.e_theta_re.zeros();
    CHECK(qd_directivity(z, 0) == 0.0);
    quadriga_lib::arrayant<double> oz;
    oz = z, quadriga_lib::trim_pattern(oz);
    CHECK(oz.n_elevation() == z.n_elevation());
    CHECK(oz.n_azimuth() == z.n_azimuth());
    CHECK(arma::approx_equal(oz.azimuth_grid, z.azimuth_grid, "absdiff", 0.0));
    CHECK(oz.is_valid().empty());
    quadriga_lib::trim_pattern(z);
    CHECK(z.n_elevation() == 19);
    CHECK(z.n_azimuth() == 37);
    CHECK(z.is_valid().empty());
}

// remove_zeros keeps the name, and the azimuth spacing of the first and the last grid point wraps
// around the circle (previous point of the first = last point - 360 deg)
TEST_CASE("Arrayant remove_zeros - name and the azimuth spacing at the wrap-around")
{
    // Sector grid -30 ... 30 deg in 10 deg steps: the wrap-around gap from 30 to -30 deg is 300 deg. Power only at
    // -30 deg: its next neighbour (-20 deg) is kept, its previous neighbour across the 300 deg gap (30 deg) is not.
    const double d2r = arma::datum::pi / 180.0;
    quadriga_lib::arrayant<double> a;
    a.name = "sector";
    a.azimuth_grid = arma::regspace<arma::vec>(-30.0, 10.0, 30.0) * d2r;
    a.elevation_grid = arma::vec({-90.0, 0.0, 90.0}) * d2r;
    a.e_theta_re.zeros(3, 7, 1);
    a.e_theta_re.slice(0).col(0).fill(1.0);
    a.e_theta_im.zeros(3, 7, 1), a.e_phi_re.zeros(3, 7, 1), a.e_phi_im.zeros(3, 7, 1);
    a.element_pos.zeros(3, 1), a.port_weights_re.ones(1, 1), a.port_weights_im.zeros(1, 1);
    a.center_freq = 1.0e9;
    REQUIRE(a.is_valid().empty());

    quadriga_lib::arrayant<double> o = a;
    quadriga_lib::trim_pattern(o);
    CHECK(o.name == "sector");
    CHECK(arma::approx_equal(o.azimuth_grid, arma::vec({-30.0, -20.0}) * d2r, "absdiff", 1e-14));
    CHECK(o.n_elevation() == 3);
    CHECK(o.is_valid().empty());

    // Power only at 30 deg: the previous neighbour (20 deg) is kept, the next one across the gap (-30 deg) is not
    a.e_theta_re.zeros();
    a.e_theta_re.slice(0).col(6).fill(1.0);
    o = a, quadriga_lib::trim_pattern(o);
    CHECK(arma::approx_equal(o.azimuth_grid, arma::vec({20.0, 30.0}) * d2r, "absdiff", 1e-14));

    // In place: the name is kept
    quadriga_lib::trim_pattern(a);
    CHECK(a.name == "sector");
    CHECK(a.n_azimuth() == 2);
}

// Copy_element must reject an empty destination with a clear message.
TEST_CASE("Arrayant copy_element - Empty destination")
{
    auto a = quadriga_lib::generate_antenna<double>("omni", 10.0);

    try
    {
        quadriga_lib::copy_element(a, 0, arma::uvec());
        FAIL("no exception for an empty destination");
    }
    catch (const std::invalid_argument &e)
    {
        CHECK(std::string(e.what()).find("i_target") != std::string::npos);
    }
    CHECK(a.n_elements() == 1);

    REQUIRE_NOTHROW(quadriga_lib::copy_element(a, 0, {2}));
    CHECK(a.n_elements() == 3);

    // The largest index cannot be reached by resizing (index + 1 wraps to 0); it must throw, not write
    const arma::uword idx_max = std::numeric_limits<arma::uword>::max();
    CHECK_THROWS_AS(quadriga_lib::copy_element(a, 0, {idx_max}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::copy_element(a, 0, arma::uvec{1, idx_max}), std::invalid_argument);
    CHECK(a.n_elements() == 3);
}

// The grids must be finite, in range and strictly increasing. A NaN or two equal neighbours must be rejected by the
// full check and by interpolate, which otherwise returns zero or NaN patterns without an error.
// The same holds for the output grids of combine_pattern and apply_port_weights.
TEMPLATE_TEST_CASE("Arrayant - Full check rejects NaN and repeated grid values", "", float, double)
{
    using T = TestType;
    const T nan = std::numeric_limits<T>::quiet_NaN();

    // 3 elevation x 4 azimuth points, one element
    quadriga_lib::arrayant<T> a;
    a.e_theta_re.ones(3, 4, 1), a.e_theta_im.zeros(3, 4, 1), a.e_phi_re.zeros(3, 4, 1), a.e_phi_im.zeros(3, 4, 1);
    a.azimuth_grid = {T(-3.0), T(-1.0), T(1.0), T(3.0)};
    a.elevation_grid = {T(-1.0), T(0.0), T(1.0)};
    a.element_pos.zeros(3, 1), a.port_weights_re.ones(1, 1), a.port_weights_im.zeros(1, 1);
    REQUIRE(a.is_valid().empty());

    arma::Mat<T> az = {T(0.1)}, el = {T(0.2)}, vr, vi, hr, hi;

    // Rejected by both entry points; interpolate_pattern runs on a copy that was never validated
    auto rejected = [&](const quadriga_lib::arrayant<T> &b)
    {
        quadriga_lib::arrayant<T> c = b, d = b;
        CHECK(!c.is_valid().empty());
        CHECK_THROWS_AS(quadriga_lib::interpolate_pattern(d, az, el, vr, vi, hr, hi), std::invalid_argument);
    };

    SECTION("Controls: valid grids pass")
    {
        quadriga_lib::arrayant<T> b = a;
        CHECK(b.is_valid().empty());
        quadriga_lib::interpolate_pattern(b, az, el, vr, vi, hr, hi);
        CHECK(std::abs(vr(0) - T(1.0)) < T(1e-5));

        // A single grid point in both directions
        quadriga_lib::arrayant<T> s;
        s.e_theta_re.ones(1, 1, 1), s.e_theta_im.zeros(1, 1, 1), s.e_phi_re.zeros(1, 1, 1), s.e_phi_im.zeros(1, 1, 1);
        s.azimuth_grid = {T(0.0)}, s.elevation_grid = {T(0.0)};
        CHECK(s.is_valid().empty());
    }

    SECTION("NaN in azimuth_grid")
    {
        for (arma::uword i : {0, 2, 3})
        {
            quadriga_lib::arrayant<T> b = a;
            b.azimuth_grid(i) = nan;
            rejected(b);
        }
    }

    SECTION("NaN in elevation_grid")
    {
        for (arma::uword i : {0, 1, 2})
        {
            quadriga_lib::arrayant<T> b = a;
            b.elevation_grid(i) = nan;
            rejected(b);
        }
    }

    SECTION("Repeated and constant grid values")
    {
        quadriga_lib::arrayant<T> b = a;
        b.elevation_grid(2) = b.elevation_grid(1);
        rejected(b);

        b = a;
        b.azimuth_grid(1) = b.azimuth_grid(0);
        rejected(b);

        b = a;
        b.azimuth_grid.zeros();
        rejected(b);
    }

    SECTION("Unsorted grid")
    {
        quadriga_lib::arrayant<T> b = a;
        std::swap(b.azimuth_grid(1), b.azimuth_grid(2));
        rejected(b);
    }

    SECTION("Output grids of combine_pattern")
    {
        const std::vector<quadriga_lib::arrayant<T>> v = {a};
        arma::Col<T> az_ok = {T(-1.0), T(0.0), T(1.0)}, el_ok = {T(0.0)};
        CHECK_NOTHROW(quadriga_lib::apply_port_weights(a, az_ok, el_ok));

        arma::Col<T> az_nan = {T(-1.0), nan, T(1.0)}, el_nan = {nan};
        CHECK_THROWS_AS(quadriga_lib::apply_port_weights(a, az_nan, el_ok), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::apply_port_weights(a, az_ok, el_nan), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::apply_port_weights(v, az_nan, el_ok), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::apply_port_weights(v, az_ok, el_nan), std::invalid_argument);

        arma::Col<T> az_rep = {T(-1.0), T(-1.0), T(1.0)};
        CHECK_THROWS_AS(quadriga_lib::apply_port_weights(a, az_rep, el_ok), std::invalid_argument);
    }
}

TEST_CASE("Arrayant remove_zeros - non-finite field values")
{
    // A NaN in every azimuth column would remove all columns; an Inf would keep the pattern unchanged
    for (double v : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        auto a = quadriga_lib::generate_antenna<double>("omni", 10.0);
        a.e_theta_re.slice(0).row(3).fill(v);
        CHECK_THROWS_AS(quadriga_lib::trim_pattern(a), std::invalid_argument);
        CHECK(a.n_azimuth() == 37);
    }
}

TEST_CASE("Arrayant calc_beamwidth - outputs that are the same variable")
{
    auto a = quadriga_lib::generate_antenna<double>("custom", 10.0, NAN, {90.0, 60.0});
    arma::vec bw_az, bw_el, p_az, p_el, d;
    CHECK_NOTHROW(quadriga_lib::calc_beamwidth(a, {}, 3.0, &bw_az, &bw_el, &p_az, &p_el));
    CHECK_THROWS_AS(quadriga_lib::calc_beamwidth(a, {}, 3.0, &bw_az, &bw_az), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_beamwidth<double>(a, {}, 3.0, &bw_az, nullptr, nullptr, &bw_az), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_beamwidth(a, {}, 3.0, &a.azimuth_grid), std::invalid_argument);

    std::vector<quadriga_lib::arrayant<double>> v = {a, a};
    arma::mat m1, m2;
    CHECK_THROWS_AS(quadriga_lib::calc_beamwidth(v, {}, 3.0, &m1, &m1), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_beamwidth<double>(v, {}, 3.0, nullptr, &v[1].element_pos), std::invalid_argument);
}

// Index arguments take a single integer, brace lists and integer vectors of any type; a literal 0 is index 0 (#98)
TEST_CASE("Arrayant - Index arguments accept integers and integer vectors")
{
    const auto ant = quadriga_lib::generate_antenna<double>("xpol", 10.0);
    const arma::vec ref0 = quadriga_lib::calc_directivity(ant, {0});
    const arma::vec ref10 = quadriga_lib::calc_directivity(ant, {1, 0});
    REQUIRE(ref0.n_elem == 1);
    REQUIRE(ref10.n_elem == 2);

    arma::vec d;
    REQUIRE_NOTHROW(d = quadriga_lib::calc_directivity(ant, 0));
    CHECK(arma::approx_equal(d, ref0, "absdiff", 1e-12));

    const int i1 = 1, i0 = 0;
    const arma::uvec u10 = {1, 0};
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, 1u), ref10.head(1), "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, {i1, i0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, std::vector<int>{1, 0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, std::vector<size_t>{1, 0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, arma::ivec{1, 0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, arma::u32_vec{1, 0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, arma::urowvec{1, 0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, u10), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, arma::uvec{1, 0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, {arma::uword(1), 0}), ref10, "absdiff", 1e-12));
    CHECK(quadriga_lib::calc_directivity(ant, std::vector<int>{}).n_elem == 2); // Empty = all
    CHECK(quadriga_lib::calc_directivity(ant, arma::umat(0, 3)).n_elem == 2);

    const arma::vec sel = {0.0, 1.0};
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, arma::find(sel > 0.5)), ref10.head(1), "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, u10.head(1)), ref10.head(1), "absdiff", 1e-12));

    CHECK_THROWS_WITH(quadriga_lib::calc_directivity(ant, -1), "Index and count arguments must be non-negative integers.");
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, {-1}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, std::vector<int>{1, -2}), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, arma::umat(2, 2, arma::fill::zeros)), std::invalid_argument);


    // Floating-point values are accepted when they are non-negative integers
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, 1.0), ref10.head(1), "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, 1.0f), ref10.head(1), "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, {1.0, 0.0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, arma::vec{1.0, 0.0}), ref10, "absdiff", 1e-12));
    CHECK(arma::approx_equal(quadriga_lib::calc_directivity(ant, std::vector<double>{1.0, 0.0}), ref10, "absdiff", 1e-12));
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, 1.5), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, -1.0), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, arma::datum::nan), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_directivity(ant, arma::vec{0.0, arma::datum::inf}), std::invalid_argument);

    // A uvec is used in place; a count tuple takes a single integer
    CHECK(&quadriga_lib::uint_list(u10).vec() == &u10);
    const auto ula4 = quadriga_lib::generate_antenna<double>("ula", 10.0, 3.0e9, {}, 4);
    CHECK(ula4.n_elements() == quadriga_lib::generate_antenna<double>("ula", 10.0, 3.0e9, {}, {4}).n_elements());
    CHECK(ula4.n_elements() == 4);

    // A null pointer and bool are not accepted
    static_assert(!std::is_convertible<std::nullptr_t, quadriga_lib::uint_list>::value, "nullptr");
    static_assert(!std::is_convertible<bool, quadriga_lib::uint_list>::value, "bool");
}

TEST_CASE("double_list - Real-valued list arguments accept scalars and vectors of any number type")
{
    using quadriga_lib::double_list;
    auto v = [](const double_list &l) { return arma::vec(l.vec()); };
    const arma::vec ref = {1.0, 2.5};

    // Scalars, brace lists and vectors of any number type
    CHECK(arma::approx_equal(v(3.5e9), arma::vec{3.5e9}, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(2), arma::vec{2.0}, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(0), arma::vec{0.0}, "absdiff", 0.0)); // Not Col(const char *)
    CHECK(arma::approx_equal(v(2.5f), arma::vec{2.5}, "absdiff", 0.0));
    CHECK(arma::approx_equal(v({1, 2.5}), ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(v({1.0f, 2.5f}), ref, "absdiff", 0.0));
    int n = 3; // A variable, not a constant expression
    CHECK(arma::approx_equal(v({n}), arma::vec{3.0}, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(std::vector<float>{1.0f, 2.5f}), ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(arma::fvec{1.0f, 2.5f}), ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(arma::rowvec{1.0, 2.5}), ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(arma::ivec{1, 3}), arma::vec{1.0, 3.0}, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(arma::linspace(1.0, 2.5, 2)), ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(v(2.0 * ref), 2.0 * ref, "absdiff", 0.0));
    CHECK(v({}).is_empty());
    CHECK(v(arma::mat(0, 3)).is_empty());
    CHECK_THROWS_AS(v(arma::mat(2, 2, arma::fill::zeros)), std::invalid_argument);

    // NaN and Inf are kept; a vec is used in place; as<T>() converts
    const arma::vec x = v({arma::datum::nan, arma::datum::inf});
    CHECK((std::isnan(x[0]) && std::isinf(x[1])));
    CHECK(&double_list(ref).vec() == &ref);
    const arma::uword k = 2;
    CHECK(arma::approx_equal(v({k, 0, 3.5}), arma::vec{2.0, 0.0, 3.5}, "absdiff", 0.0)); // Mixed variables and literals
    int neg = -1;
    auto u = [](const quadriga_lib::uint_list &l) { return l.vec().n_elem; };
    CHECK_THROWS_AS(u({neg, k}), std::invalid_argument); // Each entry is checked
    const double_list from_temp(arma::vec{4.0}); // A temporary is owned, so the list outlives it
    CHECK(from_temp.vec()[0] == 4.0);
    const arma::fvec f = double_list(ref).as<float>();
    CHECK(arma::approx_equal(f, arma::fvec{1.0f, 2.5f}, "absdiff", 0.0f));

    // A null pointer and bool are not accepted
    static_assert(!std::is_convertible<std::nullptr_t, double_list>::value, "nullptr");
    static_assert(!std::is_convertible<bool, double_list>::value, "bool");

    // Through the API: a scalar or a float vector where a brace list was needed
    CHECK(arma::approx_equal(quadriga_lib::cube<double>(2.0), quadriga_lib::cube<double>({2.0}), "absdiff", 0.0));
    CHECK(arma::approx_equal(quadriga_lib::cube<double>(1.0, arma::fvec{0.5f, 0.0f, 0.0f}), quadriga_lib::cube<double>(1.0, {0.5, 0.0, 0.0}), "absdiff", 0.0));
    CHECK(arma::approx_equal(quadriga_lib::generate_layout("37.885_Highway", 0, 5.9e9).center_freq, arma::vec{5.9e9}, "absdiff", 0.0));
    const auto a = quadriga_lib::generate_antenna<double>("ula", 10.0, 3.0e9, {}, 4, 0.25);
    const auto b = quadriga_lib::generate_antenna<double>("ula", 10.0, 3.0e9, {}, 4, {0.25});
    CHECK(arma::approx_equal(a.element_pos, b.element_pos, "absdiff", 0.0));
    const arma::fmat pos = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    arma::fmat val1, val2;
    quadriga_lib::sos_randn(10.0, pos, val1, arma::fmat(), {}, "Comb256", 1);
    quadriga_lib::sos_randn(arma::fvec{10.0f}, pos, val2, arma::fmat(), {}, "Comb256", 1);
    CHECK(arma::approx_equal(val1, val2, "absdiff", 0.0f));
    const auto omni = quadriga_lib::generate_antenna<double>("omni", 10.0);
    const auto one_dir = quadriga_lib::apply_port_weights(omni, 0.0, 0.0); // One output direction
    CHECK((one_dir.n_azimuth() == 1 && one_dir.n_elevation() == 1));
    const auto spk = quadriga_lib::generate_speaker<float>("piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
                                                           NAN, NAN, NAN, 0.15, 0.25, 1000.0);
    CHECK((spk.size() == 1 && spk[0].center_freq == 1000.0f));
    const arma::cube c_re(1, 1, 1, arma::fill::ones), c_im(1, 1, 1, arma::fill::zeros), dl(1, 1, 1, arma::fill::value(1.0e-7));
    arma::cube h_re, h_im;
    quadriga_lib::freq_response(c_re, c_im, dl, 0.0, 1.0e8, &h_re, &h_im); // One carrier at fc: H = coefficient
    CHECK((h_re.n_elem == 1 && std::abs(h_re[0] - 1.0) < 1e-6 && std::abs(h_im[0]) < 1e-6));
    const arma::vec acf = {1.0, 0.5, 0.0}, dist = {0.0, 10.0, 20.0};
    CHECK(arma::approx_equal(quadriga_lib::sos_acfi(acf, dist, 10.0), arma::vec{0.5}, "absdiff", 1e-12));

    // Vector form: a scalar list is passed on to every snapshot
    std::vector<arma::cube> v_re = {c_re, c_re}, v_im = {c_im, c_im}, v_dl = {dl, dl}, hv_re, hv_im;
    quadriga_lib::freq_response(v_re, v_im, v_dl, 0.0, 1.0e8, &hv_re, &hv_im);
    CHECK((hv_re.size() == 2 && hv_re[1].n_elem == 1 && std::abs(hv_re[1][0] - 1.0) < 1e-6));

    // Templates copy a list before writing outputs: an output on the memory of the list gives the values of a normal call
    const arma::mat p2 = {{0.0, 0.0, 0.0}, {3.0, 4.0, 0.0}};
    arma::mat ref_val;
    quadriga_lib::sos_randn(arma::vec{10.0}, p2, ref_val, arma::mat(), {}, "Comb256", 7);
    arma::mat shared = {10.0}; // [1, 1]: the list and, after resizing, the output
    arma::vec dd_view(shared.memptr(), 1, false, true);
    quadriga_lib::sos_randn(dd_view, p2, shared, arma::mat(), {}, "Comb256", 7);
    CHECK(arma::approx_equal(shared, ref_val, "absdiff", 0.0));
}
