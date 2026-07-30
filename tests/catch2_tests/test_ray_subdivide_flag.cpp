// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for the public ray_subdivide_flag API.
//
// The function answers one boolean per ray, but that boolean is the conjunction of a continuous
// geometric term and three discrete state gates, so the suite pins them separately:
//
//   geometry   the longest edge of the beam-tube footprint on the first-bounce face
//   state      travelling outside a medium, below the subdivision limit, below the interaction limit
//
// The geometric term has a closed form for an axis-aligned face and a symmetric beam: a tube of
// half-angle a launched from a point source at distance L lands as an equilateral triangle of side
// sqrt(3) * L * tan(a), independent of the launch sphere radius. That is the primary oracle. Since
// the API only exposes a boolean, the continuous value is recovered by bisecting the tolerance,
// which also exercises the decision boundary itself.
//
// The second oracle is ray_mesh_interact's edge_lengthN. ray_subdivide_flag exists to be the single
// source of truth for the subdivision decision, and ray_progress no longer derives it from
// edge_lengthN, so the two implementations can drift apart silently. The cross-check here is what
// makes that drift loud. It also pins the property that made the split possible in the first place:
// the footprint depends only on where the vertex rays meet the face, so it is the same for
// reflection and for transmission, and the same for any material.
//
// Conventions:
//  - All cubes are the 2x2x2 Blender-style cube, translated by the constructor argument. A cube
//    centred at (5,0,0) therefore has faces at x = 4 (near) and x = 6 (far).
//  - The point source sits at the coordinate origin; rays are launched from a small launch sphere
//    of radius r0 = 0.1 m, matching what ray_init produces.
//  - Material columns are the obj_file_read names {a,b,c,d,att,attB,alpha,alphaB,fRef}.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr float FRQ = 10.0e9f;  // Reference frequency for the ray_mesh_interact cross-check
    constexpr float R0 = 0.1f;      // Launch sphere radius
    constexpr float FACE_X = 4.0f;  // Near face of a cube centred at (5,0,0)
    constexpr float SQRT3 = 1.7320508075688772f;

    // Euclidean length of a 3-element row. arma::norm dispatches to BLAS snrm2, which is not
    // linked for single precision here, so do it directly.
    float nrm3(const arma::frowvec &v) { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); }

    // Closed-form edge length of a symmetric tube of half-angle "deg" striking a face normal to the
    // ray at distance L from the point source. The launch sphere radius cancels: the vertex rays
    // leave the sphere at radius r0*sin(a) and reach L*tan(a) at the face regardless of r0.
    float analytic_edge(float deg, float L)
    {
        return SQRT3 * L * std::tan(deg * (float)(arma::datum::pi / 180.0));
    }

    // ---------------------------------------------------------------------------------------
    // Materials and scene
    // ---------------------------------------------------------------------------------------

    // Convert a per-face material matrix [n_face, 9] with columns {a,b,c,d,att,attB,alpha,
    // alphaB,fRef} into the (mtl_ind, mtl_prop-map) pair, deduplicating identical rows.
    void mtl_matrix_to_map(const arma::fmat &M, arma::uvec &mtl_ind,
                           std::unordered_map<std::string, std::vector<float>> &mtl_prop)
    {
        static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
        const arma::uword n_face = M.n_rows;

        mtl_ind.set_size(n_face);
        std::vector<arma::uword> uniq;
        for (arma::uword f = 0; f < n_face; ++f)
        {
            arma::uword m = 0;
            bool found = false;
            for (; m < uniq.size(); ++m)
                if (arma::approx_equal(M.row(f), M.row(uniq[m]), "absdiff", 0.0f))
                {
                    found = true;
                    break;
                }
            if (!found)
            {
                m = (arma::uword)uniq.size();
                uniq.push_back(f);
            }
            mtl_ind(f) = m + 1;
        }

        mtl_prop.clear();
        for (int c = 0; c < 9; ++c)
        {
            std::vector<float> col(uniq.size());
            for (size_t m = 0; m < uniq.size(); ++m)
                col[m] = M.at(uniq[m], c);
            mtl_prop[names[c]] = std::move(col);
        }
    }

    // Lossless dielectric, eps_r = 5, frequency independent
    arma::frowvec mtl_dielectric() { return {5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

    // ITU-style concrete: eps_r = 5.24, sigma = 0.0462 * f^0.7822 -> frequency dependent
    arma::frowvec mtl_concrete() { return {5.24f, 0.0f, 0.0462f, 0.7822f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

    struct Scene
    {
        arma::fmat mesh;
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<float>> mtl_prop;
    };

    Scene one_cube(float cx, float cy, float cz, const arma::frowvec &mtl = mtl_dielectric())
    {
        Scene S;
        S.mesh = quadriga_lib::cube<float>({}, {}, {cx, cy, cz});
        arma::fmat M = arma::repmat(mtl, (arma::uword)S.mesh.n_rows, 1);
        mtl_matrix_to_map(M, S.mtl_ind, S.mtl_prop);
        return S;
    }

    // ---------------------------------------------------------------------------------------
    // Beam set
    // ---------------------------------------------------------------------------------------

    struct Beams
    {
        arma::fmat orig, dest, trivec, tridir;
        arma::u32_vec fbs_ind, sbs_ind;
        std::vector<quadriga_lib::path> paths;
        arma::Col<short> cur;

        arma::uword n() const { return orig.n_rows; }

        std::vector<bool> flag(const Scene &S, float tol = 3.0f,
                               uint8_t max_int = 20, uint8_t max_sub = 2) const
        {
            return quadriga_lib::ray_subdivide_flag(S.mesh, orig, dest, fbs_ind, trivec, tridir,
                                                    paths, cur, max_int, max_sub, tol);
        }
    };

    // Build a beam set: n rays from the origin along the rows of "dirs" (need not be normalized),
    // each reaching "len" metres past the launch sphere, with a triangular tube of the given
    // half-angle whose vertices sit on the launch sphere.
    Beams make_beams(const arma::fmat &dirs_in, float len, const arma::fvec &half_deg)
    {
        const arma::uword n = dirs_in.n_rows;
        REQUIRE(half_deg.n_elem == n);

        Beams B;
        arma::fmat d = dirs_in;
        for (arma::uword i = 0; i < n; ++i)
            d.row(i) /= std::sqrt(arma::accu(d.row(i) % d.row(i)));

        B.orig = d * R0;
        B.dest = d * (R0 + len);
        B.trivec.zeros(n, 9);
        B.tridir.zeros(n, 9);
        B.cur.zeros(n);

        B.paths.resize(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            B.paths[i].init(0, 1, false);
            B.paths[i].length = R0;
            B.paths[i].iR = (unsigned)i;
        }

        for (arma::uword i = 0; i < n; ++i)
        {
            arma::frowvec k = d.row(i);
            arma::frowvec a = (std::abs(k(2)) < 0.9f) ? arma::frowvec{0.0f, 0.0f, 1.0f}
                                                      : arma::frowvec{1.0f, 0.0f, 0.0f};
            arma::frowvec u = arma::cross(a, k);
            u /= nrm3(u);
            arma::frowvec v = arma::cross(k, u);

            const float ha = half_deg(i) * (float)(arma::datum::pi / 180.0);
            for (int j = 0; j < 3; ++j)
            {
                const float phi = (float)(2.0 * arma::datum::pi * (double)j / 3.0);
                arma::frowvec vj = std::cos(ha) * k +
                                   std::sin(ha) * (std::cos(phi) * u + std::sin(phi) * v);
                vj /= nrm3(vj);

                arma::frowvec tv = R0 * vj - R0 * k; // vertex on the launch sphere, rel. to orig
                for (int c = 0; c < 3; ++c)
                {
                    B.tridir(i, 3 * j + c) = vj(c);
                    B.trivec(i, 3 * j + c) = tv(c);
                }
            }
        }
        return B;
    }

    // A single beam along +x
    Beams one_beam(float half_deg, float len = 10.0f)
    {
        arma::fmat d = {{1.0f, 0.0f, 0.0f}};
        return make_beams(d, len, arma::fvec{half_deg});
    }

    // Fill fbs_ind / sbs_ind from the intersector, exactly as a caller would
    void intersect(Beams &B, const Scene &S)
    {
        quadriga_lib::ray_triangle_intersect<float>(&B.orig, &B.dest, &S.mesh, nullptr, nullptr,
                                                    nullptr, &B.fbs_ind, &B.sbs_ind, nullptr, nullptr);
    }

    // Deterministic fan of beams that all strike the near face (x = 4) of a cube centred at
    // (5,0,0): ray i points at (4, y_i, z_i) with |y_i|, |z_i| <= 0.6, and carries a half-angle
    // spread over a decade so a single tolerance separates the set. Directions depend only on i,
    // so two fans of different length share their common prefix.
    void beam_fan(arma::uword n_total, arma::fmat &dirs, arma::fvec &half)
    {
        dirs.set_size(n_total, 3);
        half.set_size(n_total);
        for (arma::uword i = 0; i < n_total; ++i)
        {
            const double t = (double)(i + 1);
            const double u = std::fmod(0.7548776662466927 * t, 1.0); // frac(i / phi)
            const double v = std::fmod(0.5698402909980532 * t, 1.0); // frac(i / phi^2)
            const double w = std::fmod(0.4142135623730951 * t, 1.0); // frac(i * (sqrt(2) - 1))
            dirs(i, 0) = 4.0f;
            dirs(i, 1) = (float)(1.2 * u - 0.6);
            dirs(i, 2) = (float)(1.2 * v - 0.6);
            half(i) = (float)(0.05 + 1.95 * w); // 0.05 to 2.0 degrees
        }
    }

    // ---------------------------------------------------------------------------------------
    // Oracles
    // ---------------------------------------------------------------------------------------

    // Edge length as reported by ray_mesh_interact, for the whole set. interaction_type selects
    // reflection (0) or transmission (1) / refraction (2); the result must not depend on it.
    arma::fvec reference_edge(const Beams &B, const Scene &S, int interaction_type = 0)
    {
        arma::fvec edge;
        quadriga_lib::ray_mesh_interact<float>(interaction_type, FRQ, &B.orig, &B.dest, &S.mesh,
                                               &S.mtl_ind, &S.mtl_prop, &B.fbs_ind, &B.sbs_ind,
                                               &B.trivec, &B.tridir,
                                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, nullptr, &edge, nullptr,
                                               nullptr, nullptr, false);
        return edge;
    }

    // Recover the edge length of ray i through the public API by bisecting the tolerance. The API
    // exposes only a boolean, so this is the only way to observe the continuous quantity — and it
    // walks the decision boundary while doing so.
    float measured_edge(const Beams &B, const Scene &S, arma::uword i, float lo, float hi)
    {
        REQUIRE(B.flag(S, lo)[i] == true);   // below the edge: flagged
        REQUIRE(B.flag(S, hi)[i] == false);  // above the edge: not flagged

        for (int it = 0; it < 40; ++it)
        {
            const float mid = 0.5f * (lo + hi);
            if (mid <= lo || mid >= hi) // converged to adjacent floats
                break;
            if (B.flag(S, mid)[i])
                lo = mid;
            else
                hi = mid;
        }
        return 0.5f * (lo + hi);
    }
}

// ===========================================================================================
// Geometry
// ===========================================================================================

TEST_CASE("ray_subdivide_flag - edge length matches the closed form")
{
    // A tube of half-angle a from a point source lands on a face at distance L as an equilateral
    // triangle of side sqrt(3) * L * tan(a). The face at x = 4 is normal to the ray, so L = 4.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    for (float deg : {0.2f, 0.5f, 1.0f, 2.0f, 5.0f})
    {
        Beams B = one_beam(deg);
        intersect(B, S);
        REQUIRE(B.fbs_ind(0) != 0u);

        const float expect = analytic_edge(deg, FACE_X);
        const float got = measured_edge(B, S, 0, 0.25f * expect, 4.0f * expect);
        CHECK(std::abs(got - expect) < 1e-3f * expect);
    }
}

TEST_CASE("ray_subdivide_flag - edge length scales with propagation distance")
{
    // Same beam, faces at x = 4 and x = 8: the footprint is twice as wide.
    const float deg = 1.0f;

    Scene near_cube = one_cube(5.0f, 0.0f, 0.0f); // near face at x = 4
    Scene far_cube = one_cube(9.0f, 0.0f, 0.0f);  // near face at x = 8

    Beams A = one_beam(deg, 20.0f);
    intersect(A, near_cube);
    Beams C = one_beam(deg, 20.0f);
    intersect(C, far_cube);

    const float e_near = measured_edge(A, near_cube, 0, 0.01f, 1.0f);
    const float e_far = measured_edge(C, far_cube, 0, 0.01f, 1.0f);

    CHECK(std::abs(e_near - analytic_edge(deg, 4.0f)) < 1e-3f * e_near);
    CHECK(std::abs(e_far - analytic_edge(deg, 8.0f)) < 1e-3f * e_far);
    CHECK(std::abs(e_far / e_near - 2.0f) < 1e-2f);

    // A tolerance between the two separates them
    const float tol = 0.5f * (e_near + e_far);
    CHECK(A.flag(near_cube, tol)[0] == false);
    CHECK(C.flag(far_cube, tol)[0] == true);
}

TEST_CASE("ray_subdivide_flag - decision boundary")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Beams B = one_beam(1.0f);
    intersect(B, S);

    const float edge = analytic_edge(1.0f, FACE_X);

    CHECK(B.flag(S, 0.99f * edge)[0] == true);  // tolerance below the edge: split
    CHECK(B.flag(S, 1.01f * edge)[0] == false); // tolerance above the edge: keep

    // The comparison is strict, so a beam exactly at the tolerance is not split. Read the edge
    // back through the API first so the two sides agree to the last bit.
    const float measured = measured_edge(B, S, 0, 0.5f * edge, 2.0f * edge);
    CHECK(B.flag(S, std::nextafter(measured, 0.0f))[0] == true);
}

TEST_CASE("ray_subdivide_flag - a partial face hit always splits")
{
    // A vertex ray that runs parallel to the face, or points away from it, has no usable
    // intersection. The tube is then treated as infinitely wide and the beam is always split,
    // however tight the beam and however loose the tolerance.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    SECTION("vertex ray parallel to the face")
    {
        Beams B = one_beam(0.05f); // far below any tolerance used here
        intersect(B, S);
        REQUIRE(B.flag(S, 0.01f)[0] == false);

        B.tridir(0, 0) = 0.0f, B.tridir(0, 1) = 1.0f, B.tridir(0, 2) = 0.0f; // normal to x
        CHECK(B.flag(S, 1.0e6f)[0] == true);
    }
    SECTION("vertex ray pointing away from the face")
    {
        Beams B = one_beam(0.05f);
        intersect(B, S);
        REQUIRE(B.flag(S, 0.01f)[0] == false);

        B.tridir(0, 3) = -1.0f, B.tridir(0, 4) = 0.0f, B.tridir(0, 5) = 0.0f; // backwards
        CHECK(B.flag(S, 1.0e6f)[0] == true);
    }
}

TEST_CASE("ray_subdivide_flag - rays that miss the mesh are never flagged")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat d = {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    Beams B = make_beams(d, 10.0f, arma::fvec{2.0f, 2.0f, 2.0f});
    intersect(B, S);

    REQUIRE(B.fbs_ind(0) != 0u);
    REQUIRE(B.fbs_ind(1) == 0u);
    REQUIRE(B.fbs_ind(2) == 0u);

    auto f = B.flag(S, 0.01f); // tolerance low enough that any real hit would be flagged
    REQUIRE(f.size() == 3u);
    CHECK(f[0] == true);
    CHECK(f[1] == false);
    CHECK(f[2] == false);
}

// ===========================================================================================
// Cross-check against ray_mesh_interact
// ===========================================================================================

TEST_CASE("ray_subdivide_flag - agrees with the edge length reported by ray_mesh_interact")
{
    // ray_progress no longer derives the decision from edge_lengthN, so the two implementations
    // can drift apart without anything else failing. This is the guard against that.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(256, dirs, half);
    Beams B = make_beams(dirs, 10.0f, half);
    intersect(B, S);

    arma::fvec edge = reference_edge(B, S, 0);
    REQUIRE(edge.n_elem == B.n());
    REQUIRE(edge.is_finite()); // the fan is built so that every tube lands fully on the face

    // A tolerance halfway between two adjacent samples, never on one. edge_lengthN is rounded to
    // float on output while ray_subdivide_flag takes the decision in double, so a tolerance placed
    // exactly on a sample lets the two sides disagree by a single rounding step on that ray. The
    // fan spreads the edge lengths over a decade, so the gap between neighbours is many orders of
    // magnitude wider than the rounding and every ray stays clear of the boundary.
    arma::fvec sorted = arma::sort(edge);
    const arma::uword k = sorted.n_elem / 2;
    REQUIRE(k + 1 < sorted.n_elem);
    REQUIRE(sorted(k + 1) > sorted(k));

    const float tol = 0.5f * (sorted(k) + sorted(k + 1));
    REQUIRE(tol > sorted(k));
    REQUIRE(tol < sorted(k + 1));

    auto f = B.flag(S, tol);
    REQUIRE(f.size() == B.n());

    arma::uword n_true = 0;
    for (arma::uword i = 0; i < B.n(); ++i)
    {
        INFO("ray " << i << ": edge = " << edge(i) << ", tol = " << tol);
        CHECK(f[i] == (edge(i) > tol));
        n_true += f[i] ? 1u : 0u;
    }
    CHECK(n_true > 0);         // the comparison is not vacuous in either direction
    CHECK(n_true < B.n());
}

TEST_CASE("ray_subdivide_flag - the footprint does not depend on the interaction or the material")
{
    // The new vertex origins are where the vertex rays meet the face, which is the same point for
    // reflection, transmission and refraction; the interaction type only changes the outgoing
    // directions. This is the property that lets the decision live outside ray_mesh_interact.
    arma::fmat dirs;
    arma::fvec half;
    beam_fan(128, dirs, half);
    Beams B = make_beams(dirs, 10.0f, half);

    Scene D = one_cube(5.0f, 0.0f, 0.0f, mtl_dielectric());
    intersect(B, D);

    arma::fvec e_reflect = reference_edge(B, D, 0);
    arma::fvec e_transmit = reference_edge(B, D, 1);
    arma::fvec e_refract = reference_edge(B, D, 2);

    CHECK(arma::approx_equal(e_reflect, e_transmit, "absdiff", 0.0f));
    CHECK(arma::approx_equal(e_reflect, e_refract, "absdiff", 0.0f));

    // Same geometry, different material
    Scene C = one_cube(5.0f, 0.0f, 0.0f, mtl_concrete());
    CHECK(arma::approx_equal(e_reflect, reference_edge(B, C, 0), "absdiff", 0.0f));
    CHECK(arma::approx_equal(e_reflect, reference_edge(B, C, 2), "absdiff", 0.0f));

    // And the flag follows the same values regardless
    arma::fvec sorted = arma::sort(e_reflect);
    const float tol = sorted(sorted.n_elem / 2);
    auto fd = B.flag(D, tol);
    auto fc = B.flag(C, tol);
    for (arma::uword i = 0; i < B.n(); ++i)
        CHECK(fd[i] == fc[i]);
}

// ===========================================================================================
// State gates
// ===========================================================================================

TEST_CASE("ray_subdivide_flag - state gates")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    const float tol = 0.5f * analytic_edge(1.0f, FACE_X); // well below the edge: geometry says split

    SECTION("all gates open")
    {
        Beams B = one_beam(1.0f);
        intersect(B, S);
        CHECK(B.flag(S, tol)[0] == true);
    }
    SECTION("a ray inside a medium is never split")
    {
        Beams B = one_beam(1.0f);
        intersect(B, S);
        B.cur(0) = 1;
        CHECK(B.flag(S, tol)[0] == false);
    }
    SECTION("the medium word is compared whole, not masked")
    {
        // Bit 15 is the resolved flag, bits 0-14 the material. A ray outside a medium but carrying
        // the flag is left alone, matching what ray_progress did.
        Beams B = one_beam(1.0f);
        intersect(B, S);
        B.cur(0) = (short)0x8000;
        CHECK(B.flag(S, tol)[0] == false);
    }
    SECTION("the subdivision counter stops at max_no_subdivisions")
    {
        Beams B = one_beam(1.0f);
        intersect(B, S);

        B.paths[0].nSUB = 1;
        CHECK(B.flag(S, tol, 20, 2)[0] == true);
        B.paths[0].nSUB = 2;
        CHECK(B.flag(S, tol, 20, 2)[0] == false);
        B.paths[0].nSUB = 3;
        CHECK(B.flag(S, tol, 20, 2)[0] == false);
    }
    SECTION("the segment count stops at max_no_interactions")
    {
        Beams B = one_beam(1.0f);
        intersect(B, S);

        B.paths[0].init(2, 1, false);
        CHECK(B.flag(S, tol, 3, 2)[0] == true);
        B.paths[0].init(3, 1, false);
        CHECK(B.flag(S, tol, 3, 2)[0] == false);
        B.paths[0].init(4, 1, false);
        CHECK(B.flag(S, tol, 3, 2)[0] == false);
    }
    SECTION("max_no_subdivisions = 0 disables the whole function")
    {
        Beams B = one_beam(1.0f);
        intersect(B, S);
        auto f = B.flag(S, tol, 20, 0);
        REQUIRE(f.size() == 1u);
        CHECK(f[0] == false);
    }
    SECTION("max_no_interactions = 0 disables the whole function")
    {
        Beams B = one_beam(1.0f);
        intersect(B, S);
        auto f = B.flag(S, tol, 0, 2);
        REQUIRE(f.size() == 1u);
        CHECK(f[0] == false);
    }
}

TEST_CASE("ray_subdivide_flag - gates are applied per ray")
{
    // A mixed set: the gates must not leak between neighbouring rays.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(64, dirs, half);
    half.fill(2.0f); // every beam wide enough to be split on geometry alone

    Beams B = make_beams(dirs, 10.0f, half);
    intersect(B, S);

    for (arma::uword i = 0; i < B.n(); ++i)
        if (i % 4 == 1)
            B.cur(i) = 1;
        else if (i % 4 == 2)
            B.paths[i].nSUB = 2;
        else if (i % 4 == 3)
            B.paths[i].init(20, 1, false);

    auto f = B.flag(S, 0.01f, 20, 2);
    REQUIRE(f.size() == B.n());
    for (arma::uword i = 0; i < B.n(); ++i)
    {
        INFO("ray " << i);
        CHECK(f[i] == (i % 4 == 0));
    }
}

// ===========================================================================================
// Threading and determinism
// ===========================================================================================

TEST_CASE("ray_subdivide_flag - the parallel and serial paths agree")
{
    // The geometry loop only forks above an internal ray-count threshold. Two fans that share
    // their prefix therefore run the same rays through both paths.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat d_small, d_large;
    arma::fvec h_small, h_large;
    beam_fan(64, d_small, h_small);
    beam_fan(16384, d_large, h_large);

    Beams SMALL = make_beams(d_small, 10.0f, h_small);
    Beams LARGE = make_beams(d_large, 10.0f, h_large);
    intersect(SMALL, S);
    intersect(LARGE, S);

    const float tol = analytic_edge(1.0f, FACE_X); // roughly mid-range for the fan

    auto fs = SMALL.flag(S, tol);
    auto fl = LARGE.flag(S, tol);

    REQUIRE(fs.size() == 64u);
    REQUIRE(fl.size() == 16384u);
    for (arma::uword i = 0; i < 64; ++i)
    {
        INFO("ray " << i);
        CHECK(fs[i] == fl[i]);
    }

    // Both outcomes occur in the large set, so the comparison covers real work
    arma::uword n_true = 0;
    for (arma::uword i = 0; i < fl.size(); ++i)
        n_true += fl[i] ? 1u : 0u;
    CHECK(n_true > 0);
    CHECK(n_true < fl.size());

    // Repeated evaluation is bit-stable
    auto again = LARGE.flag(S, tol);
    CHECK(again == fl);
}

// ===========================================================================================
// Input validation
// ===========================================================================================

TEST_CASE("ray_subdivide_flag - input validation")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Beams ref = one_beam(1.0f);
    intersect(ref, S);

    { // the baseline must run
        Beams B = ref;
        CHECK_NOTHROW(B.flag(S));
    }

    SECTION("mesh")
    {
        Beams B = ref;
        Scene E = S;
        E.mesh.reset();
        CHECK_THROWS_AS(B.flag(E), std::invalid_argument);

        Scene W = S;
        W.mesh = arma::fmat(4, 8, arma::fill::zeros);
        CHECK_THROWS_AS(B.flag(W), std::invalid_argument);
    }
    SECTION("orig defines the ray count and must be non-empty and 3-wide")
    {
        Beams B = ref;
        B.orig.set_size(0, 3);
        CHECK_THROWS_AS(B.flag(S), std::invalid_argument);

        Beams C = ref;
        C.orig = arma::fmat(1, 4, arma::fill::zeros);
        CHECK_THROWS_AS(C.flag(S), std::invalid_argument);
    }
    SECTION("per-ray blocks must match the ray count")
    {
        {
            Beams B = ref;
            B.dest = arma::fmat(2, 3, arma::fill::zeros);
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
        {
            Beams B = ref;
            B.fbs_ind.set_size(2);
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
        {
            Beams B = ref;
            B.paths.resize(2);
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
        {
            Beams B = ref;
            B.cur.zeros(2);
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
    }
    SECTION("trivec and tridir must be [n_ray, 9]")
    {
        {
            Beams B = ref;
            B.trivec.reset();
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
        {
            Beams B = ref;
            B.tridir.reset();
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
        { // spherical tridir is not accepted here
            Beams B = ref;
            B.tridir = arma::fmat(1, 6, arma::fill::zeros);
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
        {
            Beams B = ref;
            B.trivec = arma::fmat(1, 6, arma::fill::zeros);
            CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
        }
    }
    SECTION("subdivision tolerance must be positive and finite")
    {
        for (float t : {0.0f, -1.0f,
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()})
        {
            Beams B = ref;
            CHECK_THROWS_AS(B.flag(S, t), std::invalid_argument);
        }
    }
    SECTION("face indices must lie within the mesh")
    {
        Beams B = ref;
        B.fbs_ind(0) = (arma::u32)S.mesh.n_rows + 1u;
        CHECK_THROWS_AS(B.flag(S), std::invalid_argument);
    }
    SECTION("a disabled limit still validates its inputs")
    {
        // The early return for max_no_subdivisions = 0 must not bypass the shape checks
        Beams B = ref;
        B.cur.zeros(2);
        CHECK_THROWS_AS(B.flag(S, 3.0f, 20, 0), std::invalid_argument);
    }
}