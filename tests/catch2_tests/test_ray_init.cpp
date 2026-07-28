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
                                               nullptr, nullptr, nullptr, false);
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
                                           &paths, nullptr, nullptr, false);

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

    // acc_dist encodes the ray_state_update contract: [n_ray, 2] (col 0 refracted, col 1 geometric).
    // NOTE: current ray_init uses acc_dist->zeros(n_ray), which yields [n_ray, 1]; the n_cols check
    // below is written against the intended contract and will fail until that becomes zeros(n_ray, 2).
    REQUIRE(acc_dist.n_rows == n);
    CHECK(acc_dist.n_cols == 2);
    CHECK(arma::accu(arma::abs(acc_dist)) == 0.0f); // fully zeroed regardless of column count
}

TEST_CASE("ray_init - Geometry invariants (no mesh)")
{
    const arma::frowvec O = {10.0f, 20.0f, 30.0f};
    const float maxpath = 5.0f;

    arma::fmat orig, dest, path_dir;
    arma::uword n = quadriga_lib::ray_init(500, 1, O(0), O(1), O(2), maxpath,
                                           &orig, &dest, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, &path_dir, nullptr,
                                           nullptr, nullptr, nullptr, false);
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
                                           nullptr, nullptr, nullptr, false);
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
                                               &paths, nullptr, nullptr, scalar_mode);
        REQUIRE(n == 80);
        REQUIRE(paths.size() == n);

        // Path length is initialized to the launch sphere radius (launch point at the origin)
        arma::fvec r0 = row_norms(orig);
        for (arma::uword i = 0; i < n; ++i)
        {
            CHECK(paths[i].n_seg() == 0);
            CHECK(paths[i].n_freq() == n_freq);
            CHECK(paths[i].is_scalar() == scalar_mode);
            CHECK(std::abs(paths[i].length - r0(i)) < 1e-5f);
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
                                           nullptr, nullptr, nullptr, false);
    REQUIRE(n == 80);
    REQUIRE((orig.n_rows == 80 && orig.n_cols == 3));

    // Request nothing but the ray count
    arma::uword m = quadriga_lib::ray_init(80, 4, 0.0f, 0.0f, 0.0f, 5.0f,
                             nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, false);
    CHECK(m == 80);
}

TEST_CASE("ray_init - Error cases")
{
    // Zero frequencies
    CHECK_THROWS_AS(quadriga_lib::ray_init(80, 0, 0.0f, 0.0f, 0.0f, 5.0f,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, false),
                    std::invalid_argument);

    // Mesh with the wrong number of columns
    arma::fmat bad_mesh(5, 8, arma::fill::zeros);
    CHECK_THROWS_AS(quadriga_lib::ray_init(80, 1, 0.0f, 0.0f, 0.0f, 5.0f,
                                           nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, &bad_mesh, nullptr, false),
                    std::invalid_argument);
}

TEST_CASE("ray_init - Launch sphere sizing with mesh")
{
    const float maxpath = 10.0f;
    const float h = 2.0f; // nearest obstacle face at 2 m from the origin
    arma::fmat mesh = quadriga_lib::cube<float>({},{},{3.0f, 0.0f, 0.0f});

    arma::fmat orig, dest, path_dir;
    arma::uword n = quadriga_lib::ray_init(180, 1, 0.0f, 0.0f, 0.0f, maxpath,
                                           &orig, &dest, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, &path_dir, nullptr, nullptr,
                                           &mesh, nullptr, false);
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