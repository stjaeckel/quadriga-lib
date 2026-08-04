// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"

#include <cmath>
#include <vector>

namespace
{
    // Per-row distance from each row of M [n, 3] to point O
    arma::fvec dist_from(const arma::fmat &M, const arma::frowvec &O)
    {
        arma::fmat D = M.each_row() - O;
        return arma::sqrt(arma::sum(D % D, 1));
    }

    // Per-row Euclidean norm of a [n, 3] matrix
    arma::fvec row_norms(const arma::fmat &M)
    {
        return arma::sqrt(arma::sum(M % M, 1));
    }

    // Build a [n, 3] point matrix from a list of coordinates
    arma::fmat make_points(const std::vector<std::array<float, 3>> &pts)
    {
        arma::fmat P(pts.size(), 3, arma::fill::none);
        for (arma::uword i = 0; i < (arma::uword)pts.size(); ++i)
            P(i, 0) = pts[i][0], P(i, 1) = pts[i][1], P(i, 2) = pts[i][2];
        return P;
    }

    // Run ray_init at a fixed launch point of (0,0,0) and report the observed launch sphere size.
    // The icosphere places origins at face centroids, which lie slightly inside the sphere of
    // radius r0, so this returns a fixed fraction of r0 rather than r0 itself. Comparing two runs
    // with the same n_ray_target cancels that fraction exactly.
    float launch_radius(arma::uword n_ray_target, float max_path_length,
                        const arma::fmat *mesh, const arma::fmat *points)
    {
        arma::fmat orig;
        arma::uword n = quadriga_lib::ray_init(n_ray_target, 1, 0.0f, 0.0f, 0.0f, max_path_length,
                                               &orig, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, mesh, nullptr, points, false);
        REQUIRE(orig.n_rows == n);
        return row_norms(orig).max();
    }
}

TEST_CASE("ray_init - Ray count quantization")
{
    // n_ray = 20 * round(sqrt(n_ray_target / 20))^2, clamped to a minimum of one division (20 rays)
    struct Case
    {
        arma::uword target;
        arma::uword expected;
    };
    std::vector<Case> cases = {{0, 20}, {20, 20}, {80, 80}, {500, 500}, {1000, 980}};

    for (const auto &c : cases)
    {
        arma::fmat orig;
        arma::uword n = quadriga_lib::ray_init(c.target, 1, 0.0f, 0.0f, 0.0f, 5.0f,
                                               &orig, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, false);
        CHECK(n == c.expected);
        CHECK(orig.n_rows == c.expected);
    }
}

TEST_CASE("ray_init - Output shapes")
{
    arma::fmat orig, dest, trivec, tridir, path_dir, acc_dist;
    arma::Col<short> mtl_prev, mtl_current, mtl_buffer;
    std::vector<quadriga_lib::path> paths;

    arma::uword n = quadriga_lib::ray_init(500, 4, 0.0f, 0.0f, 0.0f, 5.0f,
                                           &orig, &dest, &trivec, &tridir,
                                           &mtl_prev, &mtl_current, &mtl_buffer, &path_dir, &acc_dist,
                                           &paths, nullptr, nullptr, nullptr, false);

    REQUIRE(n == 500);

    CHECK((orig.n_rows == n && orig.n_cols == 3));
    CHECK((dest.n_rows == n && dest.n_cols == 3));
    CHECK((path_dir.n_rows == n && path_dir.n_cols == 3));
    CHECK((trivec.n_rows == n && trivec.n_cols == 9));
    CHECK((tridir.n_rows == n && tridir.n_cols == 9));
    CHECK(paths.size() == n);

    // Material-state words: length n_ray, all zero (outside air, no flags)
    REQUIRE(mtl_prev.n_elem == n);
    REQUIRE(mtl_current.n_elem == n);
    REQUIRE(mtl_buffer.n_elem == n);
    CHECK(arma::all(mtl_prev == 0));
    CHECK(arma::all(mtl_current == 0));
    CHECK(arma::all(mtl_buffer == 0));

    // acc_dist encodes the ray_state_update contract: [n_ray, 2], col 0 refracted, col 1 geometric
    REQUIRE(acc_dist.n_rows == n);
    CHECK(acc_dist.n_cols == 2);
    CHECK(arma::accu(arma::abs(acc_dist)) == 0.0f);
}

TEST_CASE("ray_init - Geometry invariants (no mesh)")
{
    const arma::frowvec O = {10.0f, 20.0f, 30.0f};
    const float maxpath = 5.0f;

    arma::fmat orig, dest, path_dir;
    arma::uword n = quadriga_lib::ray_init(500, 1, O(0), O(1), O(2), maxpath,
                                           &orig, &dest, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, &path_dir, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, false);
    REQUIRE(n == 500);

    // Destinations lie at max_path_length from O
    arma::fvec dd = dist_from(dest, O);
    CHECK(arma::approx_equal(dd, arma::fvec(n, arma::fill::value(maxpath)), "absdiff", 1e-3f));

    // path_dir_prev holds unit vectors
    arma::fvec dn = row_norms(path_dir);
    CHECK(arma::approx_equal(dn, arma::fvec(n, arma::fill::ones), "absdiff", 1e-4f));

    // Origin and destination directions are consistent with path_dir_prev
    arma::fmat od = orig.each_row() - O;
    arma::fvec r0 = row_norms(od); // launch sphere radius per ray
    od.each_col() /= r0;
    CHECK(arma::approx_equal(od, path_dir, "absdiff", 1e-3f));

    arma::fmat dsd = (dest.each_row() - O) / maxpath;
    CHECK(arma::approx_equal(dsd, path_dir, "absdiff", 1e-3f));

    // Full-sphere coverage: directions span both extremes on every axis
    for (arma::uword c = 0; c < 3; ++c)
    {
        CHECK(path_dir.col(c).max() > 0.9f);
        CHECK(path_dir.col(c).min() < -0.9f);
    }
}

TEST_CASE("ray_init - max_path_length floor")
{
    // A non-positive path length is floored to 0.01 m
    arma::fmat orig, dest;
    arma::uword n = quadriga_lib::ray_init(80, 1, 0.0f, 0.0f, 0.0f, 0.0f,
                                           &orig, &dest, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, false);
    REQUIRE(n == 80);

    arma::fvec dd = dist_from(dest, arma::frowvec{0.0f, 0.0f, 0.0f});
    CHECK(arma::approx_equal(dd, arma::fvec(n, arma::fill::value(0.01f)), "absdiff", 1e-5f));

    // Origins sit inside the 1 cm launch sphere
    arma::fvec r0 = dist_from(orig, arma::frowvec{0.0f, 0.0f, 0.0f});
    CHECK(r0.max() <= 0.01f + 1e-6f);
    CHECK(r0.min() > 0.0f);
}

TEST_CASE("ray_init - Path initialization")
{
    const arma::uword n_freq = 10;

    auto run = [&](bool scalar_mode)
    {
        arma::fmat orig;
        std::vector<quadriga_lib::path> paths;
        arma::uword n = quadriga_lib::ray_init(80, n_freq, 0.0f, 0.0f, 0.0f, 5.0f,
                                               &orig, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, nullptr,
                                               &paths, nullptr, nullptr, nullptr, scalar_mode);
        REQUIRE(n == 80);
        REQUIRE(paths.size() == n);

        // Path length is initialized to the launch sphere radius (launch point at the origin)
        arma::fvec r0 = row_norms(orig);
        for (arma::uword i = 0; i < n; ++i)
        {
            CHECK(paths[i].n_seg() == 0);
            CHECK(paths[i].n_freq() == n_freq);
            CHECK(paths[i].is_scalar() == scalar_mode);
            CHECK(std::abs(paths[i].length() - r0(i)) < 1e-5f);
        }
    };

    SECTION("EM layout") { run(false); }
    SECTION("SCALAR layout") { run(true); }
}

TEST_CASE("ray_init - Optional (NULL) outputs")
{
    // Request only the origin
    arma::fmat orig;
    arma::uword n = quadriga_lib::ray_init(80, 4, 1.0f, 2.0f, 3.0f, 5.0f,
                                           &orig, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, false);
    REQUIRE(n == 80);
    REQUIRE((orig.n_rows == 80 && orig.n_cols == 3));

    // Request nothing but the ray count
    arma::uword m = quadriga_lib::ray_init(80, 4, 0.0f, 0.0f, 0.0f, 5.0f,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, false);
    CHECK(m == 80);
}

TEST_CASE("ray_init - Error cases")
{
    // Zero frequencies
    CHECK_THROWS_AS(quadriga_lib::ray_init(80, 0, 0.0f, 0.0f, 0.0f, 5.0f,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, false),
                    std::invalid_argument);

    // Mesh with the wrong number of columns
    arma::fmat bad_mesh(5, 8, arma::fill::zeros);
    CHECK_THROWS_AS(quadriga_lib::ray_init(80, 1, 0.0f, 0.0f, 0.0f, 5.0f,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, &bad_mesh, nullptr, nullptr, false),
                    std::invalid_argument);

    // Points with the wrong number of columns
    arma::fmat bad_points(4, 2, arma::fill::zeros);
    CHECK_THROWS_AS(quadriga_lib::ray_init(80, 1, 0.0f, 0.0f, 0.0f, 5.0f,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, &bad_points, false),
                    std::invalid_argument);

    // An empty point matrix is treated as absent, not as an error
    arma::fmat empty_points(0, 3, arma::fill::zeros);
    CHECK_NOTHROW(quadriga_lib::ray_init(80, 1, 0.0f, 0.0f, 0.0f, 5.0f,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr, nullptr, &empty_points, false));
}

TEST_CASE("ray_init - Launch sphere sizing with mesh")
{
    const float maxpath = 10.0f;
    const float h = 2.0f; // nearest obstacle face at 2 m from the origin
    arma::fmat mesh = quadriga_lib::cube<float>({}, {}, {3.0f, 0.0f, 0.0f});

    arma::fmat orig, dest, path_dir;
    arma::uword n = quadriga_lib::ray_init(180, 1, 0.0f, 0.0f, 0.0f, maxpath,
                                           &orig, &dest, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, &path_dir, nullptr, nullptr,
                                           &mesh, nullptr, nullptr, false);
    REQUIRE(n == 180);

    // Launch sphere radius, read back from the origins (launch point at the coordinate origin)
    const arma::frowvec O = {0.0f, 0.0f, 0.0f};
    arma::fvec r0 = dist_from(orig, O);

    // Launch sphere never exceeds 0.8 * nearest hit distance (inradius <= r0). Holds regardless
    // of intersect semantics.
    CHECK(r0.max() <= 0.8f * h + 1e-3f);
    CHECK(r0.min() > 0.0f);

    // Auto-sizing should enlarge the sphere well beyond the 1 cm default. This assumes the probe
    // registers hits (double-sided ray-triangle intersection); if it fails, check culling/winding.
    CHECK(r0.max() > 0.1f);

    // Core invariants still hold with a mesh present
    CHECK(arma::approx_equal(row_norms(path_dir), arma::fvec(n, arma::fill::ones), "absdiff", 1e-4f));
    arma::fvec dd = dist_from(dest, O);
    CHECK(arma::approx_equal(dd, arma::fvec(n, arma::fill::value(maxpath)), "absdiff", 1e-3f));
}

TEST_CASE("ray_init - Launch sphere sizing with points only")
{
    const float maxpath = 10.0f;

    // A single receive point at 2 m caps the launch sphere at 0.8 * 2 m
    arma::fmat P = make_points({{2.0f, 0.0f, 0.0f}});
    float r_2m = launch_radius(180, maxpath, nullptr, &P);

    CHECK(r_2m <= 0.8f * 2.0f + 1e-3f);
    CHECK(r_2m > 0.1f); // clearly above the 1 cm default

    // The receive point must never fall inside the launch sphere
    CHECK(r_2m < 2.0f);

    // Doubling the distance doubles the radius: the icosphere centroid factor is identical for
    // both runs, so the ratio is exact up to float rounding
    arma::fmat P4 = make_points({{0.0f, 4.0f, 0.0f}});
    float r_4m = launch_radius(180, maxpath, nullptr, &P4);
    CHECK(std::abs(r_4m / r_2m - 2.0f) < 1e-3f);

    // Only the nearest point matters, and direction is irrelevant
    arma::fmat P_many = make_points({{0.0f, 0.0f, -7.0f}, {2.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}});
    float r_many = launch_radius(180, maxpath, nullptr, &P_many);
    CHECK(std::abs(r_many - r_2m) < 1e-4f);
}

TEST_CASE("ray_init - Nearest of mesh and points wins")
{
    const float maxpath = 10.0f;
    arma::fmat mesh = quadriga_lib::cube<float>({}, {}, {3.0f, 0.0f, 0.0f}); // nearest face at 2 m

    float r_mesh_only = launch_radius(180, maxpath, &mesh, nullptr);

    // A receive point closer than the mesh takes over the sizing
    arma::fmat P_near = make_points({{-1.0f, 0.0f, 0.0f}}); // 1 m, half the mesh distance
    float r_point_wins = launch_radius(180, maxpath, &mesh, &P_near);
    CHECK(std::abs(r_point_wins / r_mesh_only - 0.5f) < 1e-3f);

    // A receive point farther than the mesh leaves the mesh-derived radius untouched
    arma::fmat P_far = make_points({{0.0f, -5.0f, 0.0f}});
    float r_mesh_wins = launch_radius(180, maxpath, &mesh, &P_far);
    CHECK(std::abs(r_mesh_wins - r_mesh_only) < 1e-4f);

    // Neither source is allowed to place an obstacle or a receiver inside the sphere
    CHECK(r_point_wins < 1.0f);
    CHECK(r_mesh_wins < 2.0f);
}

TEST_CASE("ray_init - Launch sphere fallback and clamp with points")
{
    // Every receive point beyond max_path_length falls back to the 1 cm default
    const float maxpath = 5.0f;
    arma::fmat P_far = make_points({{20.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -30.0f}});
    float r_fallback = launch_radius(80, maxpath, nullptr, &P_far);
    CHECK(r_fallback <= 0.01f + 1e-6f);
    CHECK(r_fallback > 0.0f);

    // A receive point that would produce a sub-centimeter radius is clamped to 0.01 m
    arma::fmat P_close = make_points({{0.005f, 0.0f, 0.0f}});
    float r_clamped = launch_radius(80, maxpath, nullptr, &P_close);
    CHECK(std::abs(r_clamped - r_fallback) < 1e-6f); // both sit at the 1 cm floor
}

TEST_CASE("ray_init - Points do not disturb the other outputs")
{
    const arma::frowvec O = {5.0f, -3.0f, 1.0f};
    const float maxpath = 8.0f;

    arma::fmat P = make_points({{5.0f, -3.0f, 4.0f}}); // 3 m above the launch point

    arma::fmat orig, dest, trivec, tridir, path_dir, acc_dist;
    arma::Col<short> mtl_prev, mtl_current, mtl_buffer;
    std::vector<quadriga_lib::path> paths;

    arma::uword n = quadriga_lib::ray_init(180, 2, O(0), O(1), O(2), maxpath,
                                           &orig, &dest, &trivec, &tridir,
                                           &mtl_prev, &mtl_current, &mtl_buffer, &path_dir, &acc_dist,
                                           &paths, nullptr, nullptr, &P, false);
    REQUIRE(n == 180);

    // Destinations still sit at max_path_length, directions are still unit length
    arma::fvec dd = dist_from(dest, O);
    CHECK(arma::approx_equal(dd, arma::fvec(n, arma::fill::value(maxpath)), "absdiff", 1e-3f));
    CHECK(arma::approx_equal(row_norms(path_dir), arma::fvec(n, arma::fill::ones), "absdiff", 1e-4f));

    // The enlarged launch sphere is reflected in the seeded path length
    arma::fvec r0 = dist_from(orig, O);
    CHECK(r0.max() > 0.1f);
    CHECK(r0.max() < 3.0f); // the receive point stays outside
    for (arma::uword i = 0; i < n; ++i)
        CHECK(std::abs(paths[i].length() - r0(i)) < 1e-4f);

    // State words and accumulators are unaffected
    CHECK(arma::all(mtl_current == 0));
    CHECK((acc_dist.n_rows == n && acc_dist.n_cols == 2));
    CHECK(arma::accu(arma::abs(acc_dist)) == 0.0f);
}