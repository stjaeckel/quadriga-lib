// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"

#include <cmath>
#include <stdexcept>

// Canonical 2 x 2 plane in the xy-plane (vertices at +/-1, z = 0), 2 triangles,
// identical to the plane() base and to the Blender default plane winding (f 2 3 1, f 2 4 3)
static arma::mat ref_plane()
{
    return arma::mat{{1.0, -1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0},
                     {1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0}};
}

// Apply scale -> rotate(Rz·Ry·Rx) -> translate to every vertex, scalar math only (no BLAS).
// The rotation is built by composing the three single-axis maps in order Rx, Ry, Rz, which
// equals multiplying each point by R = Rz·Ry·Rx. The plane has no thickness, so z is not scaled.
static arma::mat apply_transform(const arma::mat &ref,
                                 double sx, double sy,
                                 double a, double b, double c, // bank(x), tilt(y), heading(z)
                                 double lx, double ly, double lz)
{
    double ca = std::cos(a), sa = std::sin(a);
    double cb = std::cos(b), sb = std::sin(b);
    double cc = std::cos(c), sc = std::sin(c);

    arma::mat out(ref.n_rows, 9);
    for (arma::uword n = 0; n < ref.n_rows; ++n)
        for (arma::uword k = 0; k < 3; ++k)
        {
            double x = sx * ref(n, 3 * k);
            double y = sy * ref(n, 3 * k + 1);
            double z = ref(n, 3 * k + 2);

            double x1 = x, y1 = ca * y - sa * z, z1 = sa * y + ca * z;      // Rx
            double x2 = cb * x1 + sb * z1, y2 = y1, z2 = -sb * x1 + cb * z1; // Ry
            double x3 = cc * x2 - sc * y2, y3 = sc * x2 + cc * y2, z3 = z2;  // Rz

            out(n, 3 * k) = x3 + lx;
            out(n, 3 * k + 1) = y3 + ly;
            out(n, 3 * k + 2) = z3 + lz;
        }
    return out;
}

// Total surface area = sum of triangle areas (0.5 · norm of the edge cross product)
static double mesh_area(const arma::mat &m)
{
    double A = 0.0;
    for (arma::uword n = 0; n < m.n_rows; ++n)
    {
        double ux = m(n, 3) - m(n, 0), uy = m(n, 4) - m(n, 1), uz = m(n, 5) - m(n, 2);
        double vx = m(n, 6) - m(n, 0), vy = m(n, 7) - m(n, 1), vz = m(n, 8) - m(n, 2);
        double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
        A += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    return A;
}

// Check that every triangle normal (right-hand rule) equals the given unit vector
static bool normals_match(const arma::mat &m, double nx, double ny, double nz, double tol = 1e-12)
{
    for (arma::uword n = 0; n < m.n_rows; ++n)
    {
        double ux = m(n, 3) - m(n, 0), uy = m(n, 4) - m(n, 1), uz = m(n, 5) - m(n, 2);
        double vx = m(n, 6) - m(n, 0), vy = m(n, 7) - m(n, 1), vz = m(n, 8) - m(n, 2);
        double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
        double len = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (len < 1e-14)
            return false;
        if (std::abs(cx / len - nx) > tol || std::abs(cy / len - ny) > tol || std::abs(cz / len - nz) > tol)
            return false;
    }
    return true;
}

// Per-axis min/max over all vertices (columns x={0,3,6}, y={1,4,7}, z={2,5,8})
static arma::vec axis_min(const arma::mat &m)
{
    arma::vec r(3);
    r(0) = arma::vectorise(m.cols(arma::uvec{0, 3, 6})).min();
    r(1) = arma::vectorise(m.cols(arma::uvec{1, 4, 7})).min();
    r(2) = arma::vectorise(m.cols(arma::uvec{2, 5, 8})).min();
    return r;
}

static arma::vec axis_max(const arma::mat &m)
{
    arma::vec r(3);
    r(0) = arma::vectorise(m.cols(arma::uvec{0, 3, 6})).max();
    r(1) = arma::vectorise(m.cols(arma::uvec{1, 4, 7})).max();
    r(2) = arma::vectorise(m.cols(arma::uvec{2, 5, 8})).max();
    return r;
}

// Bounding-box center; invariant under rotation about the plane center, so equals "location"
static arma::vec bbox_center(const arma::mat &m)
{
    return 0.5 * (axis_min(m) + axis_max(m));
}

TEST_CASE("Plane - Default geometry")
{
    arma::mat m = quadriga_lib::plane<double>();

    REQUIRE(m.n_rows == 2);
    REQUIRE(m.n_cols == 9);

    // Identity transform reproduces the base plane exactly
    CHECK(arma::approx_equal(m, ref_plane(), "absdiff", 1e-12));

    // Centered at origin, half-extent 1 in x and y, flat in z, surface area 4 (2 x 2 quad)
    CHECK(arma::approx_equal(bbox_center(m), arma::vec{0.0, 0.0, 0.0}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_min(m), arma::vec{-1.0, -1.0, 0.0}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_max(m), arma::vec{1.0, 1.0, 0.0}, "absdiff", 1e-12));
    CHECK(std::abs(mesh_area(m) - 4.0) < 1e-9);

    // Blender winding: both normals point up (+z)
    CHECK(normals_match(m, 0.0, 0.0, 1.0));
}

TEST_CASE("Plane - Uniform scale")
{
    arma::mat m = quadriga_lib::plane<double>({2.0});

    CHECK(arma::approx_equal(m, 2.0 * ref_plane(), "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_min(m), arma::vec{-2.0, -2.0, 0.0}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_max(m), arma::vec{2.0, 2.0, 0.0}, "absdiff", 1e-12));

    // Area scales with the square of the linear factor: 4 · 2^2 = 16
    CHECK(std::abs(mesh_area(m) - 16.0) < 1e-9);
    CHECK(normals_match(m, 0.0, 0.0, 1.0));
}

TEST_CASE("Plane - Per-axis scale")
{
    arma::mat m = quadriga_lib::plane<double>({2.0, 3.0});

    arma::mat expected = ref_plane();
    for (arma::uword k = 0; k < 3; ++k)
    {
        expected.col(3 * k) *= 2.0;
        expected.col(3 * k + 1) *= 3.0;
    }

    CHECK(arma::approx_equal(m, expected, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_min(m), arma::vec{-2.0, -3.0, 0.0}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_max(m), arma::vec{2.0, 3.0, 0.0}, "absdiff", 1e-12));
    CHECK(std::abs(mesh_area(m) - 24.0) < 1e-9); // 4 x 6 quad
}

TEST_CASE("Plane - Scale with 3 elements ignores z")
{
    arma::mat m2 = quadriga_lib::plane<double>({2.0, 3.0});
    arma::mat m3 = quadriga_lib::plane<double>({2.0, 3.0, 4.0});

    CHECK(arma::approx_equal(m3, m2, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_min(m3), arma::vec{-2.0, -3.0, 0.0}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_max(m3), arma::vec{2.0, 3.0, 0.0}, "absdiff", 1e-12));
}

TEST_CASE("Plane - Translation")
{
    arma::vec loc = {5.0, 6.0, 7.0};
    arma::mat m = quadriga_lib::plane<double>({1.0}, {}, loc);

    arma::mat expected = ref_plane();
    for (arma::uword k = 0; k < 3; ++k)
    {
        expected.col(3 * k) += loc(0);
        expected.col(3 * k + 1) += loc(1);
        expected.col(3 * k + 2) += loc(2);
    }

    CHECK(arma::approx_equal(m, expected, "absdiff", 1e-12));
    CHECK(arma::approx_equal(bbox_center(m), loc, "absdiff", 1e-12));

    // Shape unchanged: extent still 2 x 2 and flat in z about the new center
    CHECK(arma::approx_equal(axis_max(m) - axis_min(m), arma::vec{2.0, 2.0, 0.0}, "absdiff", 1e-12));
}

TEST_CASE("Plane - Rotation about single axes")
{
    const double pi = arma::datum::pi;
    arma::mat ref = ref_plane();

    // Rz(pi/2): (x,y,z) -> (-y, x, z); plane stays in the xy-plane, normal still +z
    {
        arma::mat m = quadriga_lib::plane<double>({1.0}, {0.0, 0.0, pi / 2.0});
        arma::mat e(2, 9);
        for (arma::uword k = 0; k < 3; ++k)
        {
            e.col(3 * k) = -ref.col(3 * k + 1);
            e.col(3 * k + 1) = ref.col(3 * k);
            e.col(3 * k + 2) = ref.col(3 * k + 2);
        }
        CHECK(arma::approx_equal(m, e, "absdiff", 1e-12));
        CHECK(normals_match(m, 0.0, 0.0, 1.0));
    }

    // Rx(pi/2): (x,y,z) -> (x, -z, y); plane becomes vertical, normal turns to -y
    {
        arma::mat m = quadriga_lib::plane<double>({1.0}, {pi / 2.0, 0.0, 0.0});
        arma::mat e(2, 9);
        for (arma::uword k = 0; k < 3; ++k)
        {
            e.col(3 * k) = ref.col(3 * k);
            e.col(3 * k + 1) = -ref.col(3 * k + 2);
            e.col(3 * k + 2) = ref.col(3 * k + 1);
        }
        CHECK(arma::approx_equal(m, e, "absdiff", 1e-12));
        CHECK(arma::approx_equal(axis_min(m), arma::vec{-1.0, 0.0, -1.0}, "absdiff", 1e-12));
        CHECK(arma::approx_equal(axis_max(m), arma::vec{1.0, 0.0, 1.0}, "absdiff", 1e-12));
        CHECK(normals_match(m, 0.0, -1.0, 0.0));
    }

    // Ry(pi/2): (x,y,z) -> (z, y, -x); plane becomes vertical, normal turns to +x
    {
        arma::mat m = quadriga_lib::plane<double>({1.0}, {0.0, pi / 2.0, 0.0});
        arma::mat e(2, 9);
        for (arma::uword k = 0; k < 3; ++k)
        {
            e.col(3 * k) = ref.col(3 * k + 2);
            e.col(3 * k + 1) = ref.col(3 * k + 1);
            e.col(3 * k + 2) = -ref.col(3 * k);
        }
        CHECK(arma::approx_equal(m, e, "absdiff", 1e-12));
        CHECK(arma::approx_equal(axis_min(m), arma::vec{0.0, -1.0, -1.0}, "absdiff", 1e-12));
        CHECK(arma::approx_equal(axis_max(m), arma::vec{0.0, 1.0, 1.0}, "absdiff", 1e-12));
        CHECK(normals_match(m, 1.0, 0.0, 0.0));
    }
}

TEST_CASE("Plane - Combined rotation (Rz·Ry·Rx)")
{
    double a = 0.3, b = -0.7, c = 1.1; // bank (x), tilt (y), heading (z)

    arma::mat expected = apply_transform(ref_plane(), 1.0, 1.0, a, b, c, 0.0, 0.0, 0.0);
    arma::mat m = quadriga_lib::plane<double>({1.0}, {a, b, c});

    CHECK(arma::approx_equal(m, expected, "absdiff", 1e-12));

    // Rotation about the center preserves the bounding-box center and the area
    CHECK(arma::approx_equal(bbox_center(m), arma::vec{0.0, 0.0, 0.0}, "absdiff", 1e-12));
    CHECK(std::abs(mesh_area(m) - 4.0) < 1e-9);
}

TEST_CASE("Plane - Transform order (scale, rotate, translate)")
{
    const double pi = arma::datum::pi;
    double s = 2.0, c = pi / 2.0;
    arma::vec loc = {10.0, 0.0, 0.0};

    arma::mat expected = apply_transform(ref_plane(), s, s, 0.0, 0.0, c, loc(0), loc(1), loc(2));
    arma::mat m = quadriga_lib::plane<double>({s}, {0.0, 0.0, c}, loc);

    CHECK(arma::approx_equal(m, expected, "absdiff", 1e-12));

    // Order check: with rotate-before-translate the quad stays centered at loc.
    // Translate-before-rotate would move the center to Rz·loc = (0, 10, 0).
    CHECK(arma::approx_equal(bbox_center(m), loc, "absdiff", 1e-10));
}

TEST_CASE("Plane - Subdivision")
{
    for (arma::uword n_div = 1; n_div <= 3; ++n_div)
    {
        arma::mat m = quadriga_lib::plane<double>({1.0}, {}, {}, n_div);

        CHECK(m.n_rows == 2 * n_div * n_div);
        CHECK(m.n_cols == 9);

        // Subdivision preserves total surface area, the bounding box and the winding
        CHECK(std::abs(mesh_area(m) - 4.0) < 1e-9);
        CHECK(arma::approx_equal(axis_min(m), arma::vec{-1.0, -1.0, 0.0}, "absdiff", 1e-12));
        CHECK(arma::approx_equal(axis_max(m), arma::vec{1.0, 1.0, 0.0}, "absdiff", 1e-12));
        CHECK(arma::approx_equal(bbox_center(m), arma::vec{0.0, 0.0, 0.0}, "absdiff", 1e-12));
        CHECK(normals_match(m, 0.0, 0.0, 1.0));
    }
}

TEST_CASE("Plane - Subdivision with scale")
{
    arma::mat m = quadriga_lib::plane<double>({2.0}, {}, {}, 2);

    CHECK(m.n_rows == 8);
    CHECK(m.n_cols == 9);
    CHECK(std::abs(mesh_area(m) - 16.0) < 1e-9); // 4 · 2^2
    CHECK(arma::approx_equal(axis_min(m), arma::vec{-2.0, -2.0, 0.0}, "absdiff", 1e-12));
    CHECK(arma::approx_equal(axis_max(m), arma::vec{2.0, 2.0, 0.0}, "absdiff", 1e-12));
}

TEST_CASE("Plane - Float specialization")
{
    arma::fmat m = quadriga_lib::plane<float>();

    REQUIRE(m.n_rows == 2);
    REQUIRE(m.n_cols == 9);
    CHECK(arma::approx_equal(m, arma::conv_to<arma::fmat>::from(ref_plane()), "absdiff", 1e-5f));
}

TEST_CASE("Plane - Invalid inputs")
{
    // n_div = 0 is not allowed
    CHECK_THROWS_AS(quadriga_lib::plane<double>({1.0}, {}, {}, 0), std::invalid_argument);

    // scale must have 0, 1, 2, or 3 elements
    CHECK_THROWS_AS(quadriga_lib::plane<double>({1.0, 2.0, 3.0, 4.0}), std::invalid_argument);

    // rotation must have 0 or 3 elements
    CHECK_THROWS_AS(quadriga_lib::plane<double>({1.0}, {0.1, 0.2}), std::invalid_argument);

    // location must have 0 or 3 elements
    CHECK_THROWS_AS(quadriga_lib::plane<double>({1.0}, {}, {5.0}), std::invalid_argument);
}