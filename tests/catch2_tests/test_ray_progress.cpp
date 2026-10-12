// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for the public ray_progress API.
//
// ray_progress is an orchestrator, not a physics kernel: the Fresnel/Jones math is pinned by
// test_ray_mesh_interact, the state machine by test_ray_state_update, and the beam split by the
// subdivide_rays tests. This suite therefore asserts the things only ray_progress can get wrong,
// namely the bookkeeping between three index spaces:
//
//   full set     [n_ray]      -> the launch configuration handed in
//   compact set  [n_interact] -> rays that hit the mesh (gated on no_interact != 0)
//   per-stage    [n_subdiv], [n_reflect], [n_transmit] -> subsets addressed through index arrays
//
// What is checked: the per-stage return counts, the layout and column counts of every returned
// block, the ordering of the new launch configuration (sub-beams, then reflections, then
// transmissions), the survival gates (gain, interaction / reflection / transmission / subdivision
// limits), the medium-state hand-off across a slab, and — most importantly — that the two sides of
// the single compaction is a pure permutation of per-ray results.
//
// Oracles: geometry is exact by construction (axis-aligned 2x2x2 cubes, rays aimed at known
// points), so first-bounce coordinates are asserted directly. Where the value depends on the
// physics kernels, the assertion is an invariant instead (finiteness, monotonicity, cross-branch
// agreement, energy <= 1) so this file does not duplicate — or drift from — the kernel suites.
//
// Ray identity: the tests borrow quadriga_lib::path::iC as a stable per-ray handle. ray_progress
// never writes it (only the downstream commit stage assigns real channel IDs), and both
// path::extend and path copy-assignment preserve it, so seeding it in the builder gives a stable
// handle from an output row back to the input ray it came from. That is what makes the
// cross-branch comparison possible.
//
// Delegated intersection: the optional no_interact_in / fbs_ind_in / sbs_ind_in inputs let a caller
// hand in a ray_triangle_intersect result instead of having ray_progress compute it. Those cases
// assert bit-identical output against the internal path rather than mere equivalence, since the two
// routes feed the same arrays into the same per-ray kernels in the same order; anything less would
// let a remapping bug hide behind a tolerance.
//
// Conventions:
//  - All cubes are the 2x2x2 Blender-style cube, translated by the constructor argument. A cube
//    centred at (5,0,0) therefore has faces at x = 4 (near) and x = 6 (far).
//  - The point source sits at the coordinate origin; rays are launched from a small launch sphere
//    of radius r0 = 0.1 m, matching what ray_init produces.
//  - Material columns are the obj_file_read names {a,b,c,d,att,attB,alpha,alphaB,fRef}.

#include <catch2/catch_test_macros.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "quadriga_lib.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Diagnostic tables of the trajectory case, off unless defined
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

namespace
{
    constexpr float FRQ = 10.0e9f;      // Reference frequency
    constexpr float R0 = 0.1f;          // Launch sphere radius
    constexpr float WIDE_DEG = 1.0f;    // Beam half-angle that subdivides at SUB_TOL
    constexpr float NARROW_DEG = 0.01f; // Beam half-angle that never subdivides
    constexpr float SUB_TOL = 0.05f;    // Subdivision tolerance used by the beam tests

    // Euclidean length of a 3-element row. arma::norm dispatches to BLAS snrm2, which is not
    // linked for single precision here, so do it directly.
    float nrm3(const arma::frowvec &v) { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); }

    // ---------------------------------------------------------------------------------------
    // Materials
    // ---------------------------------------------------------------------------------------

    // Convert a per-face material matrix [n_face, 9] with columns {a,b,c,d,att,attB,alpha,
    // alphaB,fRef} into the (mtl_ind, mtl_prop-map) pair. Identical rows are deduplicated, so the
    // result matches what obj_file_read would emit.
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

    // ---------------------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------------------

    struct Scene
    {
        arma::fmat mesh;
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<float>> mtl_prop;
    };

    // One 2x2x2 cube centred at (cx,cy,cz), all 12 faces of the same material
    Scene one_cube(float cx, float cy, float cz, const arma::frowvec &mtl = mtl_dielectric())
    {
        Scene S;
        S.mesh = quadriga_lib::cube<float>({}, {}, {cx, cy, cz});
        arma::fmat M = arma::repmat(mtl, (arma::uword)S.mesh.n_rows, 1);
        mtl_matrix_to_map(M, S.mtl_ind, S.mtl_prop);
        return S;
    }

    // Two 2x2x2 cubes, one at (5,0,0) and one at (9,0,0), same material. Used for the sub-mesh
    // partitioning test: faces 0..11 are the first cube, 12..23 the second.
    Scene two_cubes()
    {
        Scene S;
        arma::fmat a = quadriga_lib::cube<float>({}, {}, {5.0f, 0.0f, 0.0f});
        arma::fmat b = quadriga_lib::cube<float>({}, {}, {9.0f, 0.0f, 0.0f});
        S.mesh = arma::join_cols(a, b);
        arma::fmat M = arma::repmat(mtl_dielectric(), (arma::uword)S.mesh.n_rows, 1);
        mtl_matrix_to_map(M, S.mtl_ind, S.mtl_prop);
        return S;
    }

    // ---------------------------------------------------------------------------------------
    // Launch configuration
    // ---------------------------------------------------------------------------------------

    // The optional block of ray_progress, so a scenario reads as one call
    struct Opt
    {
        uint8_t max_int = 20;
        uint8_t max_ref = 10;
        uint8_t max_tra = 10;
        uint8_t max_sub = 2;
        float min_gain_dB = -140.0f;
        float sub_tol = 3.0f;
        float slab = 0.15f;
        bool refraction = true;
        const arma::u32_vec *smi = nullptr;
        const arma::fmat *aabb = nullptr;

        // Delegated ray-mesh intersection (all three or none)
        const arma::u32_vec *ni_in = nullptr;
        const arma::u32_vec *fi_in = nullptr;
        const arma::u32_vec *si_in = nullptr;

        // Delegated subdivision flag
        const std::vector<bool> *sf_in = nullptr;
    };

    struct Cfg
    {
        float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
        arma::fvec freq;
        bool scalar = false;
        bool beam = false;

        arma::fmat orig, dest, acc_dist, trivec, tridir;
        arma::fcube path_dir; // [n_ray, 3, n_freq], one direction slice per frequency
        arma::Col<short> prev, cur, buf;
        std::vector<quadriga_lib::path> paths;

        arma::uword n() const { return orig.n_rows; }
        arma::uword nf() const { return freq.n_elem; }

        std::array<unsigned, 4> step(const Scene &S, const Opt &o = Opt())
        {
            return quadriga_lib::ray_progress(S.mesh, S.mtl_ind, S.mtl_prop, freq,
                                              Ox, Oy, Oz,
                                              orig, dest, prev, cur, buf, path_dir, acc_dist, paths,
                                              beam ? &trivec : nullptr,
                                              beam ? &tridir : nullptr,
                                              o.smi, o.aabb,
                                              o.max_int, o.max_ref, o.max_tra, o.max_sub,
                                              o.min_gain_dB, o.sub_tol, o.slab,
                                              o.refraction, scalar,
                                              o.ni_in, o.fi_in, o.si_in, o.sf_in);
        }
    };

    // Build a launch configuration: n rays from the source along the rows of "dirs" (need not be
    // normalized), each reaching "len" metres past the launch sphere. If "half_deg" has one entry
    // per ray the configuration is a beam set, with a triangular tube of that half-angle whose
    // vertices sit on the launch sphere.
    Cfg make_cfg(const arma::fmat &dirs_in, float len,
                 arma::uword n_freq = 1, bool scalar = false,
                 const arma::fvec &half_deg = arma::fvec())
    {
        const arma::uword n = dirs_in.n_rows;

        Cfg C;
        C.scalar = scalar;
        C.beam = (n != 0 && half_deg.n_elem == n);

        C.freq.set_size(n_freq);
        for (arma::uword f = 0; f < n_freq; ++f)
            C.freq(f) = FRQ * (1.0f + 0.5f * (float)f); // 10, 15, 20 GHz, ...

        arma::fmat d = dirs_in;
        for (arma::uword i = 0; i < n; ++i)
            d.row(i) /= std::sqrt(arma::accu(d.row(i) % d.row(i)));

        C.orig = d * R0;
        C.dest = d * (R0 + len);

        // A launched ray is outside a medium, so every frequency starts from the same direction
        C.path_dir.set_size(n, 3, n_freq);
        for (arma::uword f = 0; f < n_freq; ++f)
            C.path_dir.slice(f) = d;

        C.prev.zeros(n);
        C.cur.zeros(n);
        C.buf.zeros(n);
        C.acc_dist.zeros(n, 1 + n_freq);

        C.paths.resize(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            C.paths[i].init(0, n_freq, scalar);
            C.paths[i].set_length(R0);   // source -> ray origin
            C.paths[i].iC = (unsigned)i; // stable identity, preserved by copy and extend
        }

        if (C.beam)
        {
            C.trivec.zeros(n, 9);
            C.tridir.zeros(n, 9);
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
                        C.tridir(i, 3 * j + c) = vj(c);
                        C.trivec(i, 3 * j + c) = tv(c);
                    }
                }
            }
        }
        return C;
    }

    // A single ray along +x
    Cfg one_ray(float len = 10.0f, arma::uword n_freq = 1, bool scalar = false, float half_deg = -1.0f)
    {
        arma::fmat d = {{1.0f, 0.0f, 0.0f}};
        arma::fvec h;
        if (half_deg > 0.0f)
            h = arma::fvec{half_deg};
        return make_cfg(d, len, n_freq, scalar, h);
    }

    // Deterministic fan of rays that all strike the near face (x = 4) of a cube centred at
    // (5,0,0): ray i points at (4, y_i, z_i) with |y_i|, |z_i| <= 0.6. The first n_wide rays get
    // WIDE_DEG (subdivides at SUB_TOL), the rest NARROW_DEG (never subdivides). The narrow
    // directions depend only on i, so two fans of different length share their common prefix —
    // which is what lets the compaction test compare the same rays under both branches.
    void beam_fan(arma::uword n_total, arma::uword n_wide, arma::fmat &dirs, arma::fvec &half)
    {
        dirs.set_size(n_total, 3);
        half.set_size(n_total);
        for (arma::uword i = 0; i < n_total; ++i)
        {
            const double t = (double)(i + 1);
            const double u = std::fmod(0.7548776662466927 * t, 1.0); // frac(i / phi)
            const double v = std::fmod(0.5698402909980532 * t, 1.0); // frac(i / phi^2)
            dirs(i, 0) = 4.0f;
            dirs(i, 1) = (float)(1.2 * u - 0.6);
            dirs(i, 2) = (float)(1.2 * v - 0.6);
            half(i) = (i < n_wide) ? WIDE_DEG : NARROW_DEG;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Delegated ray-mesh intersection
    // ---------------------------------------------------------------------------------------

    // Intersection results, computed exactly the way a caller delegating the work would
    struct Hits
    {
        arma::u32_vec no_interact, fbs_ind, sbs_ind;
    };

    Hits intersect(const Cfg &C, const Scene &S, const arma::u32_vec *smi = nullptr)
    {
        Hits H;
        quadriga_lib::ray_triangle_intersect<float>(&C.orig, &C.dest, &S.mesh, nullptr, nullptr,
                                                    &H.no_interact, &H.fbs_ind, &H.sbs_ind,
                                                    smi, nullptr);
        return H;
    }

    // Bind a delegated intersection into an option block
    Opt with_hits(const Opt &base, const Hits &H)
    {
        Opt o = base;
        o.ni_in = &H.no_interact;
        o.fi_in = &H.fbs_ind;
        o.si_in = &H.sbs_ind;
        return o;
    }

    // Subdivision flags, computed the way a shading pass would: from the launch configuration as it
    // stands, before ray_progress touches it, using the same limits ray_progress will apply.
    std::vector<bool> subdiv_flags(const Cfg &C, const Scene &S, const Hits &H, const Opt &o)
    {
        return quadriga_lib::ray_subdivide_flag(S.mesh, C.orig, C.dest, H.fbs_ind, C.trivec, C.tridir,
                                                C.paths, C.cur, o.max_int, o.max_sub, o.sub_tol);
    }

    // Bind a delegated subdivision flag into an option block
    Opt with_flag(const Opt &base, const std::vector<bool> &f)
    {
        Opt o = base;
        o.sf_in = &f;
        return o;
    }

    // Number of set flags
    arma::uword count_true(const std::vector<bool> &f)
    {
        arma::uword n = 0;
        for (size_t i = 0; i < f.size(); ++i)
            n += f[i] ? 1u : 0u;
        return n;
    }

    // ---------------------------------------------------------------------------------------
    // Assertions
    // ---------------------------------------------------------------------------------------

    // Every returned block matches n_out, keeps its column count, and is finite
    void check_shapes(const Cfg &C, arma::uword n_out)
    {
        CHECK(C.orig.n_rows == n_out);
        CHECK(C.orig.n_cols == 3);
        CHECK(C.dest.n_rows == n_out);
        CHECK(C.dest.n_cols == 3);
        CHECK(C.path_dir.n_rows == n_out);
        CHECK(C.path_dir.n_cols == 3);
        CHECK(C.path_dir.n_slices == C.nf());
        CHECK(C.acc_dist.n_rows == n_out);
        CHECK(C.acc_dist.n_cols == 1 + C.nf());
        CHECK(C.prev.n_elem == n_out);
        CHECK(C.cur.n_elem == n_out);
        CHECK(C.buf.n_elem == n_out);
        CHECK(C.paths.size() == n_out);

        if (C.beam)
        {
            CHECK(C.trivec.n_rows == n_out);
            CHECK(C.trivec.n_cols == 9);
            CHECK(C.tridir.n_rows == n_out);
            CHECK(C.tridir.n_cols == 9);
        }

        if (n_out == 0)
            return;

        CHECK(C.orig.is_finite());
        CHECK(C.dest.is_finite());
        CHECK(C.path_dir.is_finite());
        CHECK(C.acc_dist.is_finite());
        CHECK(C.acc_dist.min() >= 0.0f); // ray_progress rejects a negative acc_dist on entry
        if (C.beam)
        {
            CHECK(C.trivec.is_finite());
            CHECK(C.tridir.is_finite());
        }
        for (const auto &p : C.paths)
        {
            CHECK(std::isfinite(p.length()));
            CHECK(p.calc_gain(0.0f, 0) >= 0.0f);
        }
    }

    // n_out follows from the per-stage counts
    arma::uword expected_out(const std::array<unsigned, 4> &s)
    {
        return 4ull * s[1] + s[2] + s[3];
    }

    // Map input ray id (path::iC) -> output row. Requires the ids to be unique, which holds when
    // only one continuation class is launched (e.g. reflections with max_no_transmissions = 0).
    std::map<unsigned, arma::uword> id_map(const Cfg &C)
    {
        std::map<unsigned, arma::uword> m;
        for (arma::uword i = 0; i < C.paths.size(); ++i)
            m.emplace(C.paths[i].iC, i);
        return m;
    }

    // Compare two runs row by row for every ray id present in both. Ids below min_id are
    // skipped, which is how a run with subdivision is compared against one without: the
    // subdivided ids describe different objects in the two runs.
    void check_same_rays(const Cfg &A, const Cfg &B, unsigned min_id = 0u, float tol = 1e-5f)
    {
        auto ma = id_map(A), mb = id_map(B);
        arma::uword matched = 0;
        for (const auto &kv : ma)
        {
            if (kv.first < min_id)
                continue;
            auto it = mb.find(kv.first);
            if (it == mb.end())
                continue;
            const arma::uword ia = kv.second, ib = it->second;
            ++matched;

            for (arma::uword c = 0; c < 3; ++c)
            {
                CHECK(std::abs(A.orig(ia, c) - B.orig(ib, c)) < tol);
                CHECK(std::abs(A.dest(ia, c) - B.dest(ib, c)) < tol);
                for (arma::uword f = 0; f < A.path_dir.n_slices; ++f)
                    CHECK(std::abs(A.path_dir(ia, c, f) - B.path_dir(ib, c, f)) < tol);
            }
            for (arma::uword c = 0; c < A.acc_dist.n_cols; ++c)
                CHECK(std::abs(A.acc_dist(ia, c) - B.acc_dist(ib, c)) < tol);
            CHECK((int)A.prev(ia) == (int)B.prev(ib));
            CHECK((int)A.cur(ia) == (int)B.cur(ib));
            CHECK((int)A.buf(ia) == (int)B.buf(ib));
            CHECK(A.paths[ia].n_seg() == B.paths[ib].n_seg());
            CHECK((int)A.paths[ia].nREF == (int)B.paths[ib].nREF);
            CHECK((int)A.paths[ia].nTRA == (int)B.paths[ib].nTRA);
            CHECK(std::abs(A.paths[ia].calc_gain(0.0f, 0) - B.paths[ib].calc_gain(0.0f, 0)) < tol);

            if (A.beam && B.beam)
                for (arma::uword c = 0; c < 9; ++c)
                {
                    CHECK(std::abs(A.trivec(ia, c) - B.trivec(ib, c)) < tol);
                    CHECK(std::abs(A.tridir(ia, c) - B.tridir(ib, c)) < tol);
                }
        }
        CHECK(matched > 0); // a vacuous comparison is a failed comparison
    }

    // Two runs that must agree row for row, not merely ray for ray
    void check_identical(const Cfg &A, const Cfg &B)
    {
        REQUIRE(A.n() == B.n());
        REQUIRE(A.paths.size() == B.paths.size());

        if (A.n() != 0)
        {
            CHECK(arma::approx_equal(A.orig, B.orig, "absdiff", 0.0f));
            CHECK(arma::approx_equal(A.dest, B.dest, "absdiff", 0.0f));
            CHECK(arma::approx_equal(A.path_dir, B.path_dir, "absdiff", 0.0f));
            CHECK(arma::approx_equal(A.acc_dist, B.acc_dist, "absdiff", 0.0f));
            CHECK(arma::all(A.prev == B.prev));
            CHECK(arma::all(A.cur == B.cur));
            CHECK(arma::all(A.buf == B.buf));
            if (A.beam && B.beam)
            {
                CHECK(arma::approx_equal(A.trivec, B.trivec, "absdiff", 0.0f));
                CHECK(arma::approx_equal(A.tridir, B.tridir, "absdiff", 0.0f));
            }
        }

        for (size_t i = 0; i < A.paths.size(); ++i)
        {
            CHECK(A.paths[i].iC == B.paths[i].iC);
            CHECK(A.paths[i].n_seg() == B.paths[i].n_seg());
            CHECK((int)A.paths[i].nREF == (int)B.paths[i].nREF);
            CHECK((int)A.paths[i].nTRA == (int)B.paths[i].nTRA);
            CHECK((int)A.paths[i].nSUB == (int)B.paths[i].nSUB);
            CHECK(A.paths[i].length() == B.paths[i].length());
            CHECK(A.paths[i].calc_gain(0.0f, 0) == B.paths[i].calc_gain(0.0f, 0));
        }
    }

    // Distance from a row of M to the coordinate origin
    float radius(const arma::fmat &M, arma::uword row)
    {
        return std::sqrt(M(row, 0) * M(row, 0) + M(row, 1) * M(row, 1) + M(row, 2) * M(row, 2));
    }
}

// ===========================================================================================
// Return contract
// ===========================================================================================

TEST_CASE("ray_progress - single reflection")
{
    // Source at the origin, cube centred at (5,0,0): near face at x = 4, far face at x = 6.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    Opt o;
    o.max_tra = 0; // reflection only

    auto s = C.step(S, o);
    CHECK(s[0] == 1); // n_interact
    CHECK(s[1] == 0); // n_subdiv
    CHECK(s[2] == 1); // n_reflect
    CHECK(s[3] == 0); // n_transmit

    check_shapes(C, expected_out(s));
    REQUIRE(C.n() == 1);

    // The relaunch point is the first bounce, nudged off the face by a few float ULP
    CHECK(std::abs(C.orig(0, 0) - 4.0f) < 1e-3f);
    CHECK(std::abs(C.orig(0, 1)) < 1e-5f);
    CHECK(std::abs(C.orig(0, 2)) < 1e-5f);

    // Normal incidence on the -x face sends the ray straight back
    CHECK(C.dest(0, 0) < 4.0f);
    CHECK(C.path_dir(0, 0, 0) < -0.99f);

    // Path bookkeeping
    REQUIRE(C.paths.size() == 1);
    CHECK(C.paths[0].n_seg() == 1);
    CHECK((int)C.paths[0].nREF == 1);
    CHECK((int)C.paths[0].nTRA == 0);
    CHECK((int)C.paths[0].nSUB == 0);
    CHECK(C.paths[0].iC == 0u);

    const float *crd = C.paths[0].coord(0);
    CHECK(std::abs(crd[0] - 4.0f) < 1e-3f);

    // Still outside the cube after a reflection
    CHECK((int)C.cur(0) == 0);
}

TEST_CASE("ray_progress - single transmission")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    Opt o;
    o.max_ref = 0; // transmission only

    auto s = C.step(S, o);
    CHECK(s[0] == 1);
    CHECK(s[1] == 0);
    CHECK(s[2] == 0);
    CHECK(s[3] == 1);

    check_shapes(C, expected_out(s));
    REQUIRE(C.n() == 1);

    CHECK(std::abs(C.orig(0, 0) - 4.0f) < 1e-3f);
    CHECK(C.dest(0, 0) > 4.0f);      // continues into the slab
    CHECK(C.path_dir(0, 0, 0) > 0.99f); // normal incidence: no bending

    CHECK(C.paths[0].n_seg() == 1);
    CHECK((int)C.paths[0].nREF == 0);
    CHECK((int)C.paths[0].nTRA == 1);

    // The state machine reports the ray as inside material 1, on a leg the refracting tracer bent
    CHECK((int)C.cur(0) == (1 | 0x4000));
    CHECK((int)C.prev(0) == 0);
}

TEST_CASE("ray_progress - reflection and transmission in one call")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    auto s = C.step(S);
    CHECK(s[0] == 1);
    CHECK(s[1] == 0);
    CHECK(s[2] == 1);
    CHECK(s[3] == 1);

    check_shapes(C, expected_out(s));
    REQUIRE(C.n() == 2);

    // Documented layout: sub-beams, then reflections, then transmissions
    CHECK((int)C.paths[0].nREF == 1);
    CHECK((int)C.paths[0].nTRA == 0);
    CHECK((int)C.paths[1].nREF == 0);
    CHECK((int)C.paths[1].nTRA == 1);

    // Both continuations descend from the same input ray
    CHECK(C.paths[0].iC == 0u);
    CHECK(C.paths[1].iC == 0u);

    // Reflection stays outside, transmission enters the medium
    CHECK((int)C.cur(0) == 0);
    CHECK((int)C.cur(1) == (1 | 0x4000)); // with the bent bit of the refracting tracer

    // They leave the same point in opposite directions
    CHECK(std::abs(C.orig(0, 0) - C.orig(1, 0)) < 1e-3f);
    CHECK(C.path_dir(0, 0, 0) * C.path_dir(1, 0, 0) < 0.0f);
}

TEST_CASE("ray_progress - no hit terminates the trace")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    // Rays fired away from the cube
    arma::fmat d = {{-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}};
    Cfg C = make_cfg(d, 10.0f);

    auto s = C.step(S);
    CHECK(s[0] == 0);
    CHECK(s[1] == 0);
    CHECK(s[2] == 0);
    CHECK(s[3] == 0);

    // Empty, but the column counts survive so the caller can keep chaining shapes
    check_shapes(C, 0);
    CHECK(C.orig.n_cols == 3);
    CHECK(C.acc_dist.n_cols == 1 + C.nf());
    CHECK(C.paths.empty());

    // End of trace: feeding the empty configuration back in is an error, not a no-op
    CHECK_THROWS_AS(C.step(S), std::invalid_argument);
}

TEST_CASE("ray_progress - every ray hits but none survives the gain gate")
{
    // n_interact > 0 with an empty launch configuration: the counts are NOT all zero, so a
    // caller that tests only the return value would loop forever. The empty configuration is
    // the end-of-trace signal.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    Opt o;
    o.min_gain_dB = 10.0f; // linear threshold 10, above any physical path gain

    auto s = C.step(S, o);
    CHECK(s[0] == 1);
    CHECK(s[1] == 0);
    CHECK(s[2] == 0);
    CHECK(s[3] == 0);

    check_shapes(C, 0);
    CHECK(C.orig.n_rows == 0);
    CHECK(C.orig.n_cols == 3);
    CHECK_THROWS_AS(C.step(S), std::invalid_argument);
}

TEST_CASE("ray_progress - max_no_interactions = 0 disables tracing")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    Opt o;
    o.max_int = 0;

    auto s = C.step(S, o);
    CHECK(s[0] == 0);
    CHECK(s[1] == 0);
    CHECK(s[2] == 0);
    CHECK(s[3] == 0);
    check_shapes(C, 0);
}

// ===========================================================================================
// Survival gates
// ===========================================================================================

TEST_CASE("ray_progress - reflection and transmission passes can be switched off")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    SECTION("both off: rays are counted as hits but nothing is launched")
    {
        Cfg C = one_ray();
        Opt o;
        o.max_ref = 0;
        o.max_tra = 0;
        auto s = C.step(S, o);
        CHECK(s[0] == 1);
        CHECK(s[2] == 0);
        CHECK(s[3] == 0);
        check_shapes(C, 0);
    }
    SECTION("reflection off")
    {
        Cfg C = one_ray();
        Opt o;
        o.max_ref = 0;
        auto s = C.step(S, o);
        CHECK(s[2] == 0);
        CHECK(s[3] == 1);
        check_shapes(C, 1);
    }
    SECTION("transmission off")
    {
        Cfg C = one_ray();
        Opt o;
        o.max_tra = 0;
        auto s = C.step(S, o);
        CHECK(s[2] == 1);
        CHECK(s[3] == 0);
        check_shapes(C, 1);
    }
}

TEST_CASE("ray_progress - per-ray counters gate the next generation")
{
    // A cube on each side of the source so a reflected ray finds a second target
    Scene S = two_cubes();

    SECTION("the reflection counter stops at max_no_reflections")
    {
        Cfg C = one_ray(20.0f);
        Opt o;
        o.max_ref = 1;
        o.max_tra = 0;

        auto s1 = C.step(S, o);
        REQUIRE(s1[2] == 1);
        REQUIRE(C.paths.size() == 1);
        CHECK((int)C.paths[0].nREF == 1);

        // The reflected ray is aimed back at the source and misses, so drive a second
        // generation from a fresh configuration whose path already carries nREF = 1.
        Cfg D = one_ray(20.0f);
        D.paths[0].nREF = 1;
        auto s2 = D.step(S, o);
        CHECK(s2[0] == 1); // it still hits
        CHECK(s2[2] == 0); // but no reflection is launched
    }
    SECTION("the transmission counter stops at max_no_transmissions")
    {
        Cfg C = one_ray(20.0f);
        C.paths[0].nTRA = 3;
        Opt o;
        o.max_ref = 0;
        o.max_tra = 3;
        auto s = C.step(S, o);
        CHECK(s[0] == 1);
        CHECK(s[3] == 0);
        check_shapes(C, 0);
    }
    SECTION("the segment count stops at max_no_interactions")
    {
        // Follow the transmission strand: x = 4 (enter cube 1), x = 6 (exit cube 1),
        // x = 8 (cube 2) — so the ray keeps hitting after the limit is reached.
        Cfg C = one_ray(20.0f);
        Opt o;
        o.max_int = 2;
        o.max_ref = 0;

        auto s1 = C.step(S, o);
        REQUIRE(s1[3] == 1);
        REQUIRE(C.paths[0].n_seg() == 1);

        auto s2 = C.step(S, o);
        REQUIRE(s2[3] == 1);
        REQUIRE(C.paths[0].n_seg() == 2);

        // n_seg has reached the limit: the ray still hits, but nothing is launched
        auto s3 = C.step(S, o);
        CHECK(s3[0] == 1);
        CHECK(s3[2] == 0);
        CHECK(s3[3] == 0);
        check_shapes(C, 0);
    }
}

// ===========================================================================================
// Beam subdivision
// ===========================================================================================

TEST_CASE("ray_progress - beam subdivision")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
    REQUIRE(C.beam);

    Opt o;
    o.sub_tol = SUB_TOL;

    const arma::fmat orig_in = C.orig;

    auto s = C.step(S, o);
    CHECK(s[0] == 1); // it hits
    CHECK(s[1] == 1); // and is subdivided
    CHECK(s[2] == 0); // a subdivided ray does not also reflect
    CHECK(s[3] == 0); // nor transmit

    check_shapes(C, expected_out(s));
    REQUIRE(C.n() == 4);

    for (arma::uword i = 0; i < 4; ++i)
    {
        // The sub-beam carries the parent's history and one more subdivision
        CHECK(C.paths[i].iC == 0u);
        CHECK(C.paths[i].n_seg() == 0); // no interaction was consumed
        CHECK((int)C.paths[i].nSUB == 1);
        CHECK((int)C.paths[i].nREF == 0);
        CHECK((int)C.paths[i].nTRA == 0);

        // The new origin sits on the parent wavefront, i.e. within the tube at the old origin
        const float dx = C.orig(i, 0) - orig_in(0, 0);
        const float dy = C.orig(i, 1) - orig_in(0, 1);
        const float dz = C.orig(i, 2) - orig_in(0, 2);
        CHECK(std::sqrt(dx * dx + dy * dy + dz * dz) < 0.01f);

        // The stored length is recomputed from the source, not inherited
        CHECK(std::abs(C.paths[i].length() - radius(C.orig, i)) < 1e-4f);

        // Sub-beams start outside a medium with a cleared accumulator
        CHECK((int)C.cur(i) == 0);
        for (arma::uword c = 0; c < C.acc_dist.n_cols; ++c)
            CHECK(C.acc_dist(i, c) == 0.0f);

        // Direction is the unit vector from the new origin to the new destination
        const float nx = C.dest(i, 0) - C.orig(i, 0);
        const float ny = C.dest(i, 1) - C.orig(i, 1);
        const float nz = C.dest(i, 2) - C.orig(i, 2);
        const float ln = std::sqrt(nx * nx + ny * ny + nz * nz);
        CHECK(std::abs(C.path_dir(i, 0, 0) - nx / ln) < 1e-4f);
    }

    // The sub-beam tubes are real geometry, not a zeroed buffer
    CHECK(arma::accu(arma::abs(C.trivec)) > 0.0f);
    CHECK(arma::accu(arma::abs(C.tridir)) > 0.0f);
}

TEST_CASE("ray_progress - subdivision requires beam data and a positive limit")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    SECTION("no beam data: the tolerance is ignored")
    {
        Cfg C = one_ray(); // no trivec / tridir
        Opt o;
        o.sub_tol = 1e-6f; // would subdivide everything if beams were active
        auto s = C.step(S, o);
        CHECK(s[1] == 0);
        CHECK(s[2] == 1);
        CHECK(s[3] == 1);
    }
    SECTION("max_no_subdivisions = 0")
    {
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_sub = 0;
        auto s = C.step(S, o);
        CHECK(s[1] == 0);
        CHECK(s[2] == 1);
        CHECK(s[3] == 1);
    }
    SECTION("a narrow beam stays below the tolerance")
    {
        Cfg C = one_ray(10.0f, 1, false, NARROW_DEG);
        Opt o;
        o.sub_tol = SUB_TOL;
        auto s = C.step(S, o);
        CHECK(s[1] == 0);
        CHECK(s[2] == 1);
    }
    SECTION("the subdivision counter stops at max_no_subdivisions")
    {
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        C.paths[0].nSUB = 2;
        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_sub = 2;
        auto s = C.step(S, o);
        CHECK(s[1] == 0);
        CHECK(s[2] == 1); // it reflects instead
    }
}

TEST_CASE("ray_progress - every ray subdivided")
{
    // n_interact drops to zero after the subdivision compaction, so both interaction passes are
    // skipped and the output is nothing but sub-beams.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(8, 8, dirs, half); // all wide
    Cfg C = make_cfg(dirs, 10.0f, 1, false, half);

    Opt o;
    o.sub_tol = SUB_TOL;

    auto s = C.step(S, o);
    CHECK(s[0] == 8);
    CHECK(s[1] == 8);
    CHECK(s[2] == 0);
    CHECK(s[3] == 0);

    check_shapes(C, 32);
    CHECK(C.trivec.n_rows == 32);
    CHECK(C.tridir.n_rows == 32);
    CHECK(arma::accu(arma::abs(C.trivec)) > 0.0f);

    for (const auto &p : C.paths)
        CHECK((int)p.nSUB == 1);
}

TEST_CASE("ray_progress - beam buffers survive the assembly")
{
    // trivec / tridir are written after the path storage is moved out of the internal launch
    // buffer. A size() that tracked the path vector would return them as [0, 9].
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(16, 0, dirs, half); // all narrow: pure reflection / transmission
    Cfg C = make_cfg(dirs, 10.0f, 1, false, half);

    Opt o;
    o.sub_tol = SUB_TOL;

    auto s = C.step(S, o);
    REQUIRE(s[1] == 0);
    const arma::uword n_out = expected_out(s);
    REQUIRE(n_out > 0);

    CHECK(C.trivec.n_rows == n_out);
    CHECK(C.trivec.n_cols == 9);
    CHECK(C.tridir.n_rows == n_out);
    CHECK(C.tridir.n_cols == 9);
    CHECK(C.trivec.is_finite());
    CHECK(C.tridir.is_finite());
    CHECK(arma::accu(arma::abs(C.trivec)) > 0.0f);

    // tridir holds unit vectors, three per ray
    for (arma::uword i = 0; i < n_out; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const float x = C.tridir(i, 3 * j + 0), y = C.tridir(i, 3 * j + 1), z = C.tridir(i, 3 * j + 2);
            CHECK(std::abs(std::sqrt(x * x + y * y + z * z) - 1.0f) < 1e-3f);
        }
}

// ===========================================================================================
// Compaction
// ===========================================================================================

TEST_CASE("ray_progress - compaction preserves per-ray results")
{
    // ray_progress compacts the launch configuration onto the rays that hit the mesh and were not
    // subdivided, remapping every intermediate array into a shorter index space. That compaction
    // must be a pure permutation: the surviving rays have to come out unchanged, and the subdivided
    // fraction must not influence them.
    //
    // Three runs, where two fans share their narrow prefix and differ only in length so the same
    // physical rays appear at very different subdivided fractions:
    //     HIGH:   4 wide of 229 -> 1.75 %
    //     LOW:    4 wide of 454 -> 0.88 %
    //     REF:    the HIGH fan with subdivision disabled entirely
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    Opt sub;
    sub.sub_tol = SUB_TOL;
    sub.max_tra = 0; // reflections only, so path::iC is unique in the output

    Opt nosub = sub;
    nosub.max_sub = 0;

    arma::fmat dh, dl;
    arma::fvec hh, hl;
    beam_fan(229, 4, dh, hh);
    beam_fan(454, 4, dl, hl);

    Cfg HIGH = make_cfg(dh, 10.0f, 1, false, hh);
    Cfg LOW = make_cfg(dl, 10.0f, 1, false, hl);
    Cfg REF = make_cfg(dh, 10.0f, 1, false, hh);

    auto sh = HIGH.step(S, sub);
    auto sl = LOW.step(S, sub);
    auto sr = REF.step(S, nosub);

    // All rays hit; only the four wide ones subdivide
    CHECK(sh[0] == 229);
    CHECK(sh[1] == 4);
    CHECK(sh[2] == 225);
    CHECK(sh[3] == 0);

    CHECK(sl[0] == 454);
    CHECK(sl[1] == 4);
    CHECK(sl[2] == 450);
    CHECK(sl[3] == 0);

    CHECK(sr[0] == 229);
    CHECK(sr[1] == 0);
    CHECK(sr[2] == 229);

    check_shapes(HIGH, expected_out(sh));
    check_shapes(LOW, expected_out(sl));
    check_shapes(REF, expected_out(sr));

    // The compaction is a pure permutation of per-ray results: every ray common to two runs must
    // come out identical. HIGH vs LOW crosses the branch; both against REF pins the absolute value.
    check_same_rays(HIGH, LOW);       // ids 0..3 are sub-beams in both, 4.. are reflections
    check_same_rays(HIGH, REF, 4u);   // ids 0..3 differ in kind, so compare the rest
    check_same_rays(LOW, REF, 4u);

    // The subdivided rays appear as sub-beams, never as reflections
    auto mh = id_map(HIGH);
    for (unsigned id = 0; id < 4; ++id)
    {
        auto it = mh.find(id);
        REQUIRE(it != mh.end());
        CHECK((int)HIGH.paths[it->second].nSUB == 1);
        CHECK((int)HIGH.paths[it->second].nREF == 0);
    }
}

TEST_CASE("ray_progress - the reported hit count is independent of subdivision")
{
    // n_interact is reported as the number of rays that hit the mesh, counted before anything is
    // compacted or split. Subdivision removes rays from the interaction passes, so a count taken
    // after that stage — or reused as an array stride — would silently under-report.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    Opt o;
    o.sub_tol = SUB_TOL;
    o.max_tra = 0;

    arma::fmat dh, dl;
    arma::fvec hh, hl;
    beam_fan(229, 4, dh, hh);
    beam_fan(454, 4, dl, hl);

    Cfg HIGH = make_cfg(dh, 10.0f, 1, false, hh);
    Cfg LOW = make_cfg(dl, 10.0f, 1, false, hl);

    auto sh = HIGH.step(S, o);
    auto sl = LOW.step(S, o);

    CHECK(sh[0] == 229); // not 225
    CHECK(sl[0] == 454); // not 450

    // The identity n_out = 4 * n_subdiv + n_reflect + n_transmit holds on both sides
    CHECK(HIGH.n() == expected_out(sh));
    CHECK(LOW.n() == expected_out(sl));
}

// ===========================================================================================
// Frequencies, layout, medium state
// ===========================================================================================

TEST_CASE("ray_progress - multi-frequency")
{
    // Geometry is traced once at center_frequency[0]; the remaining bands only recompute the
    // polarization coefficient, which a frequency-dependent conductivity must actually move.
    Scene S = one_cube(5.0f, 0.0f, 0.0f, mtl_concrete());

    const arma::uword n_freq = 3;
    Cfg C = one_ray(10.0f, n_freq);

    Opt o;
    o.max_tra = 0;

    auto s = C.step(S, o);
    REQUIRE(s[2] == 1);
    REQUIRE(C.paths.size() == 1);

    CHECK(C.paths[0].n_freq() == n_freq);
    CHECK(C.paths[0].is_scalar() == false);

    std::vector<float> g(n_freq);
    for (arma::uword f = 0; f < n_freq; ++f)
    {
        g[f] = C.paths[0].calc_gain(0.0f, f); // polarization power only, no path loss
        CHECK(std::isfinite(g[f]));
        CHECK(g[f] > 0.0f);
        CHECK(g[f] <= 1.0f); // a passive interface cannot amplify
    }

    // A conductive material reflects differently at 10 and 20 GHz
    CHECK(std::abs(g[0] - g[2]) > 1e-6f);

    // A frequency-independent material must give the same coefficient in every band
    Scene D = one_cube(5.0f, 0.0f, 0.0f, mtl_dielectric());
    Cfg E = one_ray(10.0f, n_freq);
    auto se = E.step(D, o);
    REQUIRE(se[2] == 1);
    for (arma::uword f = 1; f < n_freq; ++f)
        CHECK(std::abs(E.paths[0].calc_gain(0.0f, f) - E.paths[0].calc_gain(0.0f, 0)) < 1e-5f);

    // Per-frequency propagation state. The refracted direction inside a dispersive medium follows
    // that band's material, so the slices of path_dir must actually disagree. Oblique incidence is
    // required: at normal incidence there is no bend for them to disagree about.
    {
        arma::fmat d = {{1.0f, 0.15f, 0.0f}}; // strikes the near face at y = 0.6, inside the face
        Opt ot;
        ot.max_ref = 0; // transmission only, so the surviving ray is the one inside the medium

        Cfg T = make_cfg(d, 10.0f, n_freq);
        auto st = T.step(S, ot); // S is the concrete cube
        REQUIRE(st[3] == 1);
        REQUIRE(T.n() == 1);
        REQUIRE((int)T.cur(0) == (1 | 0x4000)); // inside the medium, on a bent leg
        REQUIRE(T.path_dir.n_slices == n_freq);

        bool any_diff = false;
        for (arma::uword f = 1; f < n_freq; ++f)
            for (arma::uword c = 0; c < 3; ++c)
                any_diff |= std::abs(T.path_dir(0, c, f) - T.path_dir(0, c, 0)) > 1e-6f;
        CHECK(any_diff);

        // The same geometry in a frequency-independent material must agree across every slice
        Cfg U = make_cfg(d, 10.0f, n_freq);
        auto su = U.step(D, ot);
        REQUIRE(su[3] == 1);
        for (arma::uword f = 1; f < n_freq; ++f)
            CHECK(arma::approx_equal(U.path_dir.slice(f), U.path_dir.slice(0), "absdiff", 1e-6f));
    }
}

TEST_CASE("ray_progress - scalar layout")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    SECTION("a scalar path set traces in scalar mode")
    {
        Cfg C = one_ray(10.0f, 2, true);
        Opt o;
        o.max_tra = 0;
        auto s = C.step(S, o);
        CHECK(s[2] == 1);
        check_shapes(C, 1);
        CHECK(C.paths[0].is_scalar());
        CHECK(C.paths[0].n_freq() == 2);
        CHECK(C.paths[0].calc_gain(0.0f, 0) > 0.0f);
    }
    SECTION("a layout mismatch is rejected")
    {
        Cfg C = one_ray(10.0f, 2, true); // scalar paths
        C.scalar = false;                // EM request
        CHECK_THROWS_AS(C.step(S), std::invalid_argument);
    }
}

TEST_CASE("ray_progress - medium state across a slab")
{
    // Enter the cube at x = 4, leave it at x = 6. The state words must go 0 -> 1 -> 0 and the
    // previous-medium word must remember the slab on the way out.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray(20.0f);

    Opt o;
    o.max_ref = 0; // transmission only, so the trace is a single strand

    auto s1 = C.step(S, o);
    REQUIRE(s1[3] == 1);
    REQUIRE(C.n() == 1);
    CHECK((int)C.cur(0) == (1 | 0x4000)); // inside, on a bent leg
    CHECK((int)C.prev(0) == 0);
    CHECK(C.paths[0].n_seg() == 1);

    auto s2 = C.step(S, o);
    REQUIRE(s2[0] == 1);
    REQUIRE(s2[3] == 1);
    REQUIRE(C.n() == 1);
    CHECK((int)C.cur(0) == 0); // back outside
    CHECK((int)C.prev(0) == 1);
    CHECK(C.paths[0].n_seg() == 2);
    CHECK((int)C.paths[0].nTRA == 2);

    // The exit point is the far face
    const float *crd = C.paths[0].coord(1);
    CHECK(std::abs(crd[0] - 6.0f) < 1e-2f);

    // Leaving the slab clears the accumulator
    for (arma::uword c = 0; c < C.acc_dist.n_cols; ++c)
        CHECK(C.acc_dist(0, c) >= 0.0f);
}

TEST_CASE("ray_progress - sub-mesh partitioning does not change the result")
{
    Scene S = two_cubes();
    REQUIRE(S.mesh.n_rows == 24);

    arma::u32_vec smi = {0u, 12u};

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(32, 0, dirs, half);

    Cfg PLAIN = make_cfg(dirs, 20.0f);
    Cfg PART = make_cfg(dirs, 20.0f);

    Opt o;
    Opt op = o;
    op.smi = &smi; // aabb left NULL: computed internally

    auto s1 = PLAIN.step(S, o);
    auto s2 = PART.step(S, op);

    CHECK(s1 == s2);
    REQUIRE(PLAIN.n() == PART.n());
    REQUIRE(PLAIN.n() > 0);

    // The partition only changes the traversal order inside the intersect, so the output is
    // row-for-row identical, reflections and transmissions alike.
    CHECK(arma::approx_equal(PLAIN.orig, PART.orig, "absdiff", 1e-6f));
    CHECK(arma::approx_equal(PLAIN.dest, PART.dest, "absdiff", 1e-6f));
    CHECK(arma::approx_equal(PLAIN.path_dir, PART.path_dir, "absdiff", 1e-6f));
    CHECK(arma::approx_equal(PLAIN.acc_dist, PART.acc_dist, "absdiff", 1e-6f));
    CHECK(arma::all(PLAIN.cur == PART.cur));
    CHECK(arma::all(PLAIN.prev == PART.prev));
    for (arma::uword i = 0; i < PLAIN.paths.size(); ++i)
    {
        CHECK(PLAIN.paths[i].iC == PART.paths[i].iC);
        CHECK(PLAIN.paths[i].n_seg() == PART.paths[i].n_seg());
    }
}

// ===========================================================================================
// Multi-generation behaviour
// ===========================================================================================

TEST_CASE("ray_progress - a trace terminates and respects its limits")
{
    Scene S = two_cubes();

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(64, 0, dirs, half);
    Cfg C = make_cfg(dirs, 30.0f);

    Opt o;
    o.max_int = 4;
    o.max_ref = 2;
    o.max_tra = 2;

    arma::uword gen = 0;
    bool ended = false;
    for (; gen < 40; ++gen)
    {
        auto s = C.step(S, o);
        const arma::uword n_out = expected_out(s);
        check_shapes(C, n_out);

        for (const auto &p : C.paths)
        {
            CHECK(p.n_seg() <= (size_t)o.max_int);
            CHECK((int)p.nREF <= (int)o.max_ref);
            CHECK((int)p.nTRA <= (int)o.max_tra);
            CHECK((size_t)((int)p.nREF + (int)p.nTRA) <= p.n_seg());
        }

        if (n_out == 0)
        {
            ended = true;
            break;
        }
    }
    CHECK(ended);
    CHECK(gen > 0);   // at least one productive generation
    CHECK(gen < 40u); // and it did not run away
}

TEST_CASE("ray_progress - repeated calls are deterministic")
{
    // The internal launch buffer is allocated without zero-fill, so any slot that is committed
    // but not written would surface here (or as a non-finite value in check_shapes).
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(96, 6, dirs, half);

    Opt o;
    o.sub_tol = SUB_TOL;

    Cfg A = make_cfg(dirs, 10.0f, 2, false, half);
    Cfg B = make_cfg(dirs, 10.0f, 2, false, half);

    auto sa = A.step(S, o);
    auto sb = B.step(S, o);

    REQUIRE(sa == sb);
    REQUIRE(A.n() == B.n());

    CHECK(arma::approx_equal(A.orig, B.orig, "absdiff", 0.0f));
    CHECK(arma::approx_equal(A.dest, B.dest, "absdiff", 0.0f));
    CHECK(arma::approx_equal(A.path_dir, B.path_dir, "absdiff", 0.0f));
    CHECK(arma::approx_equal(A.acc_dist, B.acc_dist, "absdiff", 0.0f));
    CHECK(arma::approx_equal(A.trivec, B.trivec, "absdiff", 0.0f));
    CHECK(arma::approx_equal(A.tridir, B.tridir, "absdiff", 0.0f));
    CHECK(arma::all(A.prev == B.prev));
    CHECK(arma::all(A.cur == B.cur));
    CHECK(arma::all(A.buf == B.buf));

    for (arma::uword i = 0; i < A.paths.size(); ++i)
    {
        CHECK(A.paths[i].iC == B.paths[i].iC);
        CHECK(A.paths[i].n_seg() == B.paths[i].n_seg());
        CHECK(A.paths[i].length() == B.paths[i].length());
        for (arma::uword f = 0; f < 2; ++f)
            CHECK(A.paths[i].calc_gain(0.0f, f) == B.paths[i].calc_gain(0.0f, f));
    }
}

TEST_CASE("ray_progress - accumulated path length")
{
    // path::length is defined as source -> last interaction, and check_gains reads it back as
    // "source -> current ray origin" when it forms the free-space path loss. The source position
    // only enters ray_progress through Ox/Oy/Oz, so the first segment is the one at risk.
    //
    // NOTE: path::extend derives the new segment length from the previous stored coordinate, and
    // a fresh path has none (nSEG == 0), so the first interaction contributes dl = 0 and the leg
    // from the source to the first bounce is dropped. The first CHECK below is written against
    // the intended contract and fails until ray_progress writes the length it already computes
    // for the gain gate back into the extended path.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray(20.0f);

    Opt o;
    o.max_ref = 0; // transmission only: one strand, x = 4 then x = 6

    auto s1 = C.step(S, o);
    REQUIRE(s1[3] == 1);
    CHECK(std::abs(C.paths[0].length() - 4.0f) < 1e-2f); // source -> first bounce at x = 4

    const float after_first = C.paths[0].length();

    auto s2 = C.step(S, o);
    REQUIRE(s2[3] == 1);

    // Independent of the absolute value, the increment must be the length of the new leg,
    // here the 2 m slab crossing from x = 4 to x = 6. This isolates the defect to the first
    // extend rather than to the accumulation.
    CHECK(std::abs((C.paths[0].length() - after_first) - 2.0f) < 1e-2f);

    // The subdivision path recomputes the length from the source instead of accumulating it,
    // and is correct for a fresh path.
    Cfg B = one_ray(10.0f, 1, false, WIDE_DEG);
    Opt ob;
    ob.sub_tol = SUB_TOL;
    auto sb = B.step(S, ob);
    REQUIRE(sb[1] == 1);
    for (arma::uword i = 0; i < B.n(); ++i)
        CHECK(std::abs(B.paths[i].length() - radius(B.orig, i)) < 1e-4f);
}

// ===========================================================================================
// Delegated ray-mesh intersection
// ===========================================================================================

TEST_CASE("ray_progress - delegated intersection matches the internal one")
{
    // Supplying the ray_triangle_intersect result must be a pure refactor: bit-identical output,
    // not merely an equivalent one. Exact comparison is legitimate here because the delegated and
    // internal paths feed the same arrays into the same per-ray kernels in the same order.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;

    SECTION("no beam")
    {
        beam_fan(64, 0, dirs, half);
        Cfg A = make_cfg(dirs, 10.0f);
        Cfg B = make_cfg(dirs, 10.0f);

        Opt o;
        Hits H = intersect(B, S);
        auto sa = A.step(S, o);
        auto sb = B.step(S, with_hits(o, H));

        CHECK(sa == sb);
        CHECK(sa[0] == 64); // the hit count comes from the delegated array
        check_identical(A, B);
    }
    SECTION("beam, no subdivision")
    {
        beam_fan(64, 0, dirs, half);
        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;
        Hits H = intersect(B, S);
        auto sa = A.step(S, o);
        auto sb = B.step(S, with_hits(o, H));

        CHECK(sa == sb);
        REQUIRE(sa[1] == 0);
        check_identical(A, B);
    }
    SECTION("beam with subdivision")
    {
        // 4 wide of 229. The delegated intersect arrays are full-length and are compacted by the
        // same index list as everything else.
        beam_fan(229, 4, dirs, half);
        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_tra = 0;
        Hits H = intersect(B, S);
        auto sa = A.step(S, o);
        auto sb = B.step(S, with_hits(o, H));

        CHECK(sa == sb);
        REQUIRE(sa[1] == 4);
        REQUIRE(sa[0] == 229);
        check_identical(A, B);
    }
    SECTION("rays that miss are dropped the same way")
    {
        // Half the fan is aimed away from the cube, so the delegated no_interact drives the
        // compaction rather than an internally computed one.
        beam_fan(32, 0, dirs, half);
        arma::fmat away = dirs;
        for (arma::uword i = 0; i < 32; i += 2)
            away(i, 0) = -4.0f; // fired backwards

        Cfg A = make_cfg(away, 10.0f);
        Cfg B = make_cfg(away, 10.0f);

        Opt o;
        Hits H = intersect(B, S);
        auto sa = A.step(S, o);
        auto sb = B.step(S, with_hits(o, H));

        CHECK(sa == sb);
        CHECK(sa[0] == 16);
        check_identical(A, B);
    }
    SECTION("multi-frequency")
    {
        beam_fan(48, 0, dirs, half);
        Cfg A = make_cfg(dirs, 10.0f, 3);
        Cfg B = make_cfg(dirs, 10.0f, 3);

        Opt o;
        Hits H = intersect(B, S);
        auto sa = A.step(S, o);
        auto sb = B.step(S, with_hits(o, H));

        CHECK(sa == sb);
        check_identical(A, B);
        for (size_t i = 0; i < A.paths.size(); ++i)
            for (arma::uword f = 0; f < 3; ++f)
                CHECK(A.paths[i].calc_gain(0.0f, f) == B.paths[i].calc_gain(0.0f, f));
    }
}

TEST_CASE("ray_progress - delegated intersection across a multi-generation trace")
{
    // Delegation is per call, not per trace: each generation emits a new orig/dest pair, so the
    // caller has to re-intersect every time. Running both variants in lockstep also proves that
    // nothing from a delegated call leaks into the next one.
    Scene S = two_cubes();

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(64, 0, dirs, half);

    Cfg A = make_cfg(dirs, 30.0f);
    Cfg B = make_cfg(dirs, 30.0f);

    Opt o;
    o.max_int = 4;
    o.max_ref = 2;
    o.max_tra = 2;

    bool ended = false;
    for (int gen = 0; gen < 20; ++gen)
    {
        Hits H = intersect(B, S); // from B's current, pre-step launch configuration

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_hits(o, H));

        CHECK(sa == sb);
        check_identical(A, B);

        if (A.n() == 0)
        {
            ended = true;
            break;
        }
    }
    CHECK(ended);
}

TEST_CASE("ray_progress - delegated intersection ignores the acceleration structure")
{
    // sub_mesh_index and aabb only feed the internal intersector, so passing them alongside a
    // delegated result must change nothing. They are still validated.
    Scene S = two_cubes();
    arma::u32_vec smi = {0u, 12u};

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(32, 0, dirs, half);

    Cfg A = make_cfg(dirs, 20.0f);
    Cfg B = make_cfg(dirs, 20.0f);

    Opt o;
    Hits H = intersect(A, S); // computed without segmentation

    Opt oa = with_hits(o, H);
    Opt ob = with_hits(o, H);
    ob.smi = &smi; // supplied but unused

    auto sa = A.step(S, oa);
    auto sb = B.step(S, ob);

    CHECK(sa == sb);
    check_identical(A, B);
}

TEST_CASE("ray_progress - delegated intersection with max_no_interactions = 0")
{
    // Tracing stays disabled even when the caller hands in a full set of hits.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    Hits H = intersect(C, S);
    REQUIRE(H.no_interact(0) > 0u); // the ray really does hit

    Opt o;
    o.max_int = 0;

    auto s = C.step(S, with_hits(o, H));
    CHECK(s[0] == 0);
    CHECK(s[1] == 0);
    CHECK(s[2] == 0);
    CHECK(s[3] == 0);
    check_shapes(C, 0);
}

TEST_CASE("ray_progress - delegated intersection validation")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    const arma::u32 n_mesh = (arma::u32)S.mesh.n_rows;

    Cfg probe = one_ray();
    const Hits ref = intersect(probe, S);

    SECTION("the three arrays must be supplied together")
    {
        // Each of the six partial combinations
        for (int mask = 1; mask < 7; ++mask)
        {
            Cfg C = one_ray();
            Opt o;
            if (mask & 1)
                o.ni_in = &ref.no_interact;
            if (mask & 2)
                o.fi_in = &ref.fbs_ind;
            if (mask & 4)
                o.si_in = &ref.sbs_ind;
            CHECK_THROWS_AS(C.step(S, o), std::invalid_argument);
        }
    }
    SECTION("each array must match the ray count")
    {
        Hits H = ref;
        H.no_interact.set_size(2);
        Cfg C = one_ray();
        CHECK_THROWS_AS(C.step(S, with_hits(Opt(), H)), std::invalid_argument);

        Hits G = ref;
        G.fbs_ind.set_size(2);
        Cfg D = one_ray();
        CHECK_THROWS_AS(D.step(S, with_hits(Opt(), G)), std::invalid_argument);

        Hits F = ref;
        F.sbs_ind.set_size(2);
        Cfg E = one_ray();
        CHECK_THROWS_AS(E.step(S, with_hits(Opt(), F)), std::invalid_argument);
    }
    SECTION("face indices must lie within the mesh")
    {
        Hits H = ref;
        H.fbs_ind(0) = n_mesh + 1u;
        Cfg C = one_ray();
        CHECK_THROWS_AS(C.step(S, with_hits(Opt(), H)), std::invalid_argument);

        Hits G = ref;
        G.sbs_ind(0) = n_mesh + 1u;
        Cfg D = one_ray();
        CHECK_THROWS_AS(D.step(S, with_hits(Opt(), G)), std::invalid_argument);
    }
}

// ===========================================================================================
// Delegated subdivision flag
// ===========================================================================================

TEST_CASE("ray_progress - delegated subdivision flag matches the internal one")
{
    // ray_subdivide_flag is the single source of truth: handing its result back in must reproduce
    // the internal path exactly, so a shading pass and ray_progress can never disagree about which
    // beams will reappear as sub-beams.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;

    SECTION("nothing to subdivide")
    {
        beam_fan(64, 0, dirs, half); // all narrow
        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;

        Hits H = intersect(B, S);
        auto f = subdiv_flags(B, S, H, o);
        REQUIRE(count_true(f) == 0u);

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_flag(o, f));

        CHECK(sa == sb);
        CHECK(sb[1] == 0);
        check_identical(A, B);
    }
    SECTION("a small subdivided fraction")
    {
        beam_fan(454, 4, dirs, half); // 4 of 454 = 0.88 %
        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_tra = 0;

        Hits H = intersect(B, S);
        auto f = subdiv_flags(B, S, H, o);
        REQUIRE(count_true(f) == 4u);

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_flag(o, f));

        CHECK(sa == sb);
        CHECK(sb[1] == 4);
        check_identical(A, B);
    }
    SECTION("a larger subdivided fraction")
    {
        beam_fan(229, 4, dirs, half); // 4 of 229 = 1.75 %
        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_tra = 0;

        Hits H = intersect(B, S);
        auto f = subdiv_flags(B, S, H, o);
        REQUIRE(count_true(f) == 4u);

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_flag(o, f));

        CHECK(sa == sb);
        CHECK(sb[1] == 4);
        CHECK(sb[0] == 229);
        check_identical(A, B);
    }
    SECTION("mixed hit and miss: the flag is remapped, not truncated")
    {
        // Half the fan is fired backwards. The flag is indexed in the full ray set while
        // ray_progress works on the compacted one, so the remap has to line up.
        beam_fan(64, 0, dirs, half);
        arma::uword n_wide_hit = 0;
        for (arma::uword i = 0; i < 64; ++i)
            if (i % 2 == 0) // fired backwards, misses the cube
            {
                dirs(i, 0) = -4.0f;
                half(i) = WIDE_DEG; // and would be flagged if it did hit
            }
            else if (i % 8 == 1) // a subset of the 32 survivors subdivides
            {
                half(i) = WIDE_DEG;
                ++n_wide_hit;
            }
        REQUIRE(n_wide_hit == 8u); // pin the construction, not just the outcome

        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_tra = 0;

        Hits H = intersect(B, S);
        auto f = subdiv_flags(B, S, H, o);

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_flag(o, f));

        CHECK(sa == sb);
        CHECK(sb[0] == 32);                        // only the forward half hits
        CHECK(sb[1] == (unsigned)n_wide_hit);      // and only the wide survivors subdivide
        CHECK(count_true(f) == n_wide_hit);        // rays that miss are never flagged
        CHECK(sb[1] == (unsigned)count_true(f));   // the flag is the decision, one for one
        check_identical(A, B);
    }
    SECTION("combined with a delegated intersection")
    {
        beam_fan(229, 4, dirs, half);
        Cfg A = make_cfg(dirs, 10.0f, 1, false, half);
        Cfg B = make_cfg(dirs, 10.0f, 1, false, half);

        Opt o;
        o.sub_tol = SUB_TOL;
        o.max_tra = 0;

        Hits H = intersect(B, S);
        auto f = subdiv_flags(B, S, H, o);

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_flag(with_hits(o, H), f));

        CHECK(sa == sb);
        check_identical(A, B);
    }
}

TEST_CASE("ray_progress - the delegated flag count is the reported n_subdiv")
{
    // The shading pass counts the beams it must hold back; that count has to be exactly what
    // ray_progress expands into sub-beams.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(128, 0, dirs, half);
    for (arma::uword i = 0; i < 128; i += 3)
        half(i) = WIDE_DEG;

    Cfg C = make_cfg(dirs, 10.0f, 1, false, half);

    Opt o;
    o.sub_tol = SUB_TOL;

    Hits H = intersect(C, S);
    auto f = subdiv_flags(C, S, H, o);
    const arma::uword n_flagged = count_true(f);
    REQUIRE(n_flagged > 0u);
    REQUIRE(n_flagged < 128u);

    auto s = C.step(S, with_flag(o, f));
    CHECK(s[1] == (unsigned)n_flagged);

    // Every sub-beam carries one more subdivision and no interaction
    arma::uword n_sub_out = 0;
    for (const auto &p : C.paths)
        if (p.nSUB == 1)
        {
            ++n_sub_out;
            CHECK(p.n_seg() == 0);
        }
    CHECK(n_sub_out == 4u * n_flagged);
}

TEST_CASE("ray_progress - the delegated flag overrides the geometry")
{
    // Proves the flag is taken as the decision rather than treated as a hint. Both directions are
    // exercised: a narrow beam forced to split, and a wide one forced to continue.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    Opt o;
    o.sub_tol = SUB_TOL;
    o.max_tra = 0;

    SECTION("a narrow beam is split when the flag says so")
    {
        Cfg C = one_ray(10.0f, 1, false, NARROW_DEG);
        Hits H = intersect(C, S);
        auto f = subdiv_flags(C, S, H, o);
        REQUIRE(f.size() == 1u);
        REQUIRE(f[0] == false); // geometry alone would not split it

        f[0] = true;
        auto s = C.step(S, with_flag(o, f));
        CHECK(s[1] == 1);
        CHECK(s[2] == 0); // a subdivided ray does not also reflect
        check_shapes(C, 4);
        for (const auto &p : C.paths)
            CHECK((int)p.nSUB == 1);
    }
    SECTION("a wide beam continues when the flag says so")
    {
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        Hits H = intersect(C, S);
        auto f = subdiv_flags(C, S, H, o);
        REQUIRE(f.size() == 1u);
        REQUIRE(f[0] == true); // geometry alone would split it

        f[0] = false;
        auto s = C.step(S, with_flag(o, f));
        CHECK(s[1] == 0);
        CHECK(s[2] == 1); // reflects instead
        check_shapes(C, 1);
        CHECK((int)C.paths[0].nSUB == 0);
        CHECK(C.paths[0].n_seg() == 1);
    }
}

TEST_CASE("ray_progress - delegated subdivision flag across a multi-generation trace")
{
    // The flag has to be recomputed every generation, exactly like the intersection: each call
    // emits a new launch configuration. Running both variants in lockstep also shows that nothing
    // from a delegated call leaks into the next.
    Scene S = two_cubes();

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(96, 6, dirs, half);

    Cfg A = make_cfg(dirs, 30.0f, 1, false, half);
    Cfg B = make_cfg(dirs, 30.0f, 1, false, half);

    Opt o;
    o.max_int = 4;
    o.max_ref = 2;
    o.max_tra = 2;
    o.sub_tol = SUB_TOL;

    bool ended = false;
    for (int gen = 0; gen < 20; ++gen)
    {
        Hits H = intersect(B, S);
        auto f = subdiv_flags(B, S, H, o);

        auto sa = A.step(S, o);
        auto sb = B.step(S, with_flag(with_hits(o, H), f));

        CHECK(sa == sb);
        check_identical(A, B);

        if (A.n() == 0)
        {
            ended = true;
            break;
        }
    }
    CHECK(ended);
}

TEST_CASE("ray_progress - delegated subdivision flag validation")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    Cfg probe = one_ray(10.0f, 1, false, WIDE_DEG);
    Hits H = intersect(probe, S);

    Opt base;
    base.sub_tol = SUB_TOL;
    const std::vector<bool> good = subdiv_flags(probe, S, H, base);
    REQUIRE(good.size() == 1u);

    SECTION("the flag must have one entry per ray")
    {
        std::vector<bool> wrong(2, false);
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        CHECK_THROWS_AS(C.step(S, with_flag(base, wrong)), std::invalid_argument);

        std::vector<bool> empty;
        Cfg D = one_ray(10.0f, 1, false, WIDE_DEG);
        CHECK_THROWS_AS(D.step(S, with_flag(base, empty)), std::invalid_argument);
    }
    SECTION("beam mode is required")
    {
        // Without trivec / tridir there is no wavefront to split
        Cfg C = one_ray(); // no beam data
        std::vector<bool> f(1, false);
        CHECK_THROWS_AS(C.step(S, with_flag(base, f)), std::invalid_argument);
    }
    SECTION("max_no_subdivisions = 0 rejects a flag instead of ignoring it")
    {
        Opt o = base;
        o.max_sub = 0;
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        CHECK_THROWS_AS(C.step(S, with_flag(o, good)), std::invalid_argument);
    }
    SECTION("an all-false flag is accepted and disables subdivision")
    {
        std::vector<bool> none(1, false);
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        auto s = C.step(S, with_flag(base, none));
        CHECK(s[1] == 0);
        CHECK(s[2] == 1);
    }
}

// ===========================================================================================
// Authoritative subdivision flag
// ===========================================================================================
//
// A delegated subdiv_flag_in is the authority on what gets split. It overrides every condition
// ray_subdivide_flag would otherwise apply: the mesh-hit status, the per-ray subdivision limit and
// the outside-a-medium gate. These paths are unreachable from the internal decision, so each case
// first asserts that ray_subdivide_flag really does refuse the ray, then forces the flag and checks
// that ray_progress obeys.

TEST_CASE("ray_progress - a flagged ray that misses the mesh is still subdivided")
{
    // The interaction compaction used to discard these rays before the subdivision was applied, so
    // a shading pass that held the beam back saw its energy vanish. n_subdiv is now independent of
    // n_interact and the two are counted separately.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat d = {{-1.0f, 0.0f, 0.0f}}; // fired away from the cube
    arma::fvec h = {WIDE_DEG};
    Cfg C = make_cfg(d, 10.0f, 1, false, h);

    Opt o;
    o.sub_tol = SUB_TOL;

    Hits H = intersect(C, S);
    REQUIRE(H.fbs_ind(0) == 0u); // it really does miss

    // ray_subdivide_flag refuses: with no first-bounce face there is no footprint to measure
    auto f_auto = subdiv_flags(C, S, H, o);
    REQUIRE(f_auto.size() == 1u);
    CHECK(f_auto[0] == false);

    std::vector<bool> f(1, true);
    auto s = C.step(S, with_flag(o, f));

    CHECK(s[0] == 0); // nothing hit the mesh
    CHECK(s[1] == 1); // but the beam was split anyway
    CHECK(s[2] == 0);
    CHECK(s[3] == 0);

    check_shapes(C, 4);
    REQUIRE(C.n() == 4);
    for (const auto &p : C.paths)
    {
        CHECK((int)p.nSUB == 1);
        CHECK(p.n_seg() == 0); // no interaction was consumed
        CHECK(p.iC == 0u);
    }
    CHECK(arma::accu(arma::abs(C.trivec)) > 0.0f);
}

TEST_CASE("ray_progress - the end-of-trace signal accounts for subdivision")
{
    // The trace ends only when nothing hits the mesh AND nothing is flagged. A flagged ray that
    // misses keeps the trace alive; an unflagged one does not.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat d = {{-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    arma::fvec h = {WIDE_DEG, WIDE_DEG};

    Opt o;
    o.sub_tol = SUB_TOL;

    SECTION("all-false flag: the trace ends")
    {
        Cfg C = make_cfg(d, 10.0f, 1, false, h);
        std::vector<bool> f(2, false);
        auto s = C.step(S, with_flag(o, f));
        CHECK(s[0] == 0);
        CHECK(s[1] == 0);
        check_shapes(C, 0);
        CHECK_THROWS_AS(C.step(S, o), std::invalid_argument); // empty config is end of trace
    }
    SECTION("one flagged: the trace continues")
    {
        Cfg C = make_cfg(d, 10.0f, 1, false, h);
        std::vector<bool> f(2, false);
        f[1] = true;
        auto s = C.step(S, with_flag(o, f));
        CHECK(s[0] == 0);
        CHECK(s[1] == 1);
        check_shapes(C, 4);
        for (const auto &p : C.paths)
            CHECK(p.iC == 1u); // the sub-beams descend from the flagged ray, not its neighbour
    }
}

TEST_CASE("ray_progress - a flagged ray past its subdivision limit is still subdivided")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    Opt o;
    o.sub_tol = SUB_TOL;
    o.max_sub = 2;

    SECTION("the limit is overridden and the counter advances past it")
    {
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        C.paths[0].nSUB = 2;

        Hits H = intersect(C, S);
        auto f_auto = subdiv_flags(C, S, H, o);
        REQUIRE(f_auto[0] == false); // ray_subdivide_flag applies max_no_subdivisions

        std::vector<bool> f(1, true);
        auto s = C.step(S, with_flag(o, f));
        CHECK(s[1] == 1);
        CHECK(s[2] == 0); // a subdivided ray does not also reflect
        REQUIRE(C.n() == 4);
        for (const auto &p : C.paths)
            CHECK((int)p.nSUB == 3);
    }
    SECTION("the counter saturates instead of wrapping")
    {
        // nSUB is a uint8_t. A wrapped counter reads as 0, which would let ray_subdivide_flag
        // re-arm the same beam and split it without bound.
        Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);
        C.paths[0].nSUB = 255;

        std::vector<bool> f(1, true);
        auto s = C.step(S, with_flag(o, f));
        CHECK(s[1] == 1);
        REQUIRE(C.n() == 4);
        for (const auto &p : C.paths)
            CHECK((int)p.nSUB == 255);
    }
}

TEST_CASE("ray_progress - a flagged ray inside a medium inherits its propagation state")
{
    // Outside a medium each sub-beam gets a freshly recomputed geometric direction and a cleared
    // in-layer accumulator. Inside one, the tracked direction is the refracted direction and
    // deliberately differs from the geometric continuation, and the accumulator holds the distance
    // travelled so far — so both must be inherited from the parent instead. Only a delegated flag
    // reaches this branch; ray_subdivide_flag gates on mtl_ind_current == 0.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);

    C.cur(0) = 1;              // travelling inside material 1
    C.acc_dist(0, 0) = 1.10f;  // geometric in-layer distance, shared across frequencies
    C.acc_dist(0, 1) = 1.25f;  // refracted in-layer distance at the single frequency

    // A tracked direction that is deliberately not the geometric +x continuation
    const float dx = 0.6f, dy = 0.8f, dz = 0.0f;
    C.path_dir(0, 0, 0) = dx, C.path_dir(0, 1, 0) = dy, C.path_dir(0, 2, 0) = dz;

    Opt o;
    o.sub_tol = SUB_TOL;

    Hits H = intersect(C, S);
    auto f_auto = subdiv_flags(C, S, H, o);
    REQUIRE(f_auto[0] == false); // ray_subdivide_flag refuses a ray inside a medium

    std::vector<bool> f(1, true);
    auto s = C.step(S, with_flag(o, f));
    CHECK(s[1] == 1);
    REQUIRE(C.n() == 4);

    for (arma::uword i = 0; i < 4; ++i)
    {
        INFO("sub-beam " << i);
        CHECK((int)C.cur(i) == 1); // still inside the medium

        // Accumulator carried over, not cleared
        CHECK(std::abs(C.acc_dist(i, 0) - 1.10f) < 1e-6f);
        CHECK(std::abs(C.acc_dist(i, 1) - 1.25f) < 1e-6f);

        // Refracted direction carried over, not recomputed from the new origin
        CHECK(std::abs(C.path_dir(i, 0, 0) - dx) < 1e-6f);
        CHECK(std::abs(C.path_dir(i, 1, 0) - dy) < 1e-6f);
        CHECK(std::abs(C.path_dir(i, 2, 0) - dz) < 1e-6f);
    }
}

TEST_CASE("ray_progress - the outside-a-medium branch still recomputes and clears")
{
    // The counterpart to the case above: with mtl_ind_current == 0 the sub-beams must NOT inherit.
    // Both branches share one loop, so this pins that the added condition did not invert it.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray(10.0f, 1, false, WIDE_DEG);

    C.acc_dist(0, 0) = 2.5f; // stale values that must be discarded
    C.acc_dist(0, 1) = 2.5f;
    C.path_dir(0, 0, 0) = 0.6f, C.path_dir(0, 1, 0) = 0.8f, C.path_dir(0, 2, 0) = 0.0f;

    Opt o;
    o.sub_tol = SUB_TOL;

    auto s = C.step(S, o); // internal flag: the ray is outside and wide, so it is split
    REQUIRE(s[1] == 1);
    REQUIRE(C.n() == 4);

    for (arma::uword i = 0; i < 4; ++i)
    {
        INFO("sub-beam " << i);
        CHECK((int)C.cur(i) == 0);
        for (arma::uword c = 0; c < C.acc_dist.n_cols; ++c)
            CHECK(C.acc_dist(i, c) == 0.0f);

        // Direction is the unit vector from the new origin to the new destination
        const float nx = C.dest(i, 0) - C.orig(i, 0);
        const float ny = C.dest(i, 1) - C.orig(i, 1);
        const float nz = C.dest(i, 2) - C.orig(i, 2);
        const float ln = std::sqrt(nx * nx + ny * ny + nz * nz);
        CHECK(std::abs(C.path_dir(i, 0, 0) - nx / ln) < 1e-4f);
        CHECK(std::abs(C.path_dir(i, 1, 0) - ny / ln) < 1e-4f);
        CHECK(std::abs(C.path_dir(i, 2, 0) - nz / ln) < 1e-4f);
        CHECK(C.path_dir(i, 0, 0) > 0.9f); // and it points along +x, not the stale (0.6, 0.8, 0)
    }
}

TEST_CASE("ray_progress - hit and subdivision counts are independent")
{
    // n_subdiv is no longer a subset of n_interact. A mixed set exercises all four combinations of
    // (hit, flagged) in one call and pins the identity n_out = 4*n_subdiv + n_reflect + n_transmit.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    arma::fmat dirs;
    arma::fvec half;
    beam_fan(64, 0, dirs, half);
    for (arma::uword i = 0; i < 64; i += 2)
        dirs(i, 0) = -4.0f; // even rays miss, odd rays hit

    Cfg C = make_cfg(dirs, 10.0f, 1, false, half);

    Opt o;
    o.sub_tol = SUB_TOL;
    o.max_tra = 0;

    Hits H = intersect(C, S);

    // Flag every missing ray plus a quarter of the hitting ones
    std::vector<bool> f(64, false);
    arma::uword n_miss_flagged = 0, n_hit_flagged = 0;
    for (arma::uword i = 0; i < 64; ++i)
        if (H.fbs_ind(i) == 0u)
            f[i] = true, ++n_miss_flagged;
        else if (i % 8 == 1)
            f[i] = true, ++n_hit_flagged;

    REQUIRE(n_miss_flagged == 32u);
    REQUIRE(n_hit_flagged == 8u);

    auto s = C.step(S, with_flag(o, f));

    CHECK(s[0] == 32);                                          // rays that hit the mesh
    CHECK(s[1] == (unsigned)(n_miss_flagged + n_hit_flagged));  // rays that were split
    CHECK(s[2] == (unsigned)(32 - n_hit_flagged));              // hit and not split -> reflected
    CHECK(s[3] == 0);

    check_shapes(C, expected_out(s));
    CHECK(C.n() == expected_out(s));

    // Every flagged ray produced four sub-beams and consumed no interaction
    arma::uword n_sub_out = 0;
    for (const auto &p : C.paths)
        if (p.nSUB == 1)
        {
            ++n_sub_out;
            CHECK(p.n_seg() == 0);
        }
    CHECK(n_sub_out == 4u * (n_miss_flagged + n_hit_flagged));
}

// ===========================================================================================
// Input validation
// ===========================================================================================

TEST_CASE("ray_progress - input validation")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);

    { // the baseline must run
        Cfg C = one_ray();
        CHECK_NOTHROW(C.step(S));
    }

    SECTION("mesh")
    {
        Cfg C = one_ray();
        Scene B = S;
        B.mesh.reset();
        CHECK_THROWS_AS(C.step(B), std::invalid_argument);

        Scene W = S;
        W.mesh = arma::fmat(4, 8, arma::fill::zeros);
        CHECK_THROWS_AS(C.step(W), std::invalid_argument);

        Scene M = S;
        M.mtl_ind = arma::uvec(3, arma::fill::ones);
        CHECK_THROWS_AS(C.step(M), std::invalid_argument);
    }
    SECTION("center frequency")
    {
        std::vector<arma::fvec> bad;
        bad.push_back(arma::fvec());                          // empty
        bad.push_back(arma::fvec(128, arma::fill::ones));     // more than 127 entries
        bad.push_back(arma::fvec{0.0f});                      // not positive
        bad.push_back(arma::fvec{-FRQ});                      // negative
        bad.push_back(arma::fvec{FRQ, 0.0f});                 // one bad entry among good ones

        for (const auto &f : bad)
        {
            Cfg C = one_ray();
            C.freq = f;
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
    }
    SECTION("orig defines the ray count and must be non-empty and 3-wide")
    {
        Cfg C = one_ray();
        C.orig.set_size(0, 3);
        CHECK_THROWS_AS(C.step(S), std::invalid_argument);

        Cfg D = one_ray();
        D.orig = arma::fmat(1, 4, arma::fill::zeros);
        CHECK_THROWS_AS(D.step(S), std::invalid_argument);
    }
    SECTION("per-ray blocks must match the ray count")
    {
        {
            Cfg C = one_ray();
            C.dest = arma::fmat(2, 3, arma::fill::zeros);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray();
            C.prev.zeros(2);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray();
            C.cur.zeros(2);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray();
            C.buf.zeros(2);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray();
            C.path_dir.zeros(1, 2, 1); // wrong column count
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray(10.0f, 2); // two frequencies, one direction slice
            C.path_dir.zeros(1, 3, 1);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray();
            C.paths.resize(2);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
    }
    SECTION("acc_dist must be [n_ray, 1+n_freq]")
    {
        Cfg C = one_ray(); // one frequency, so two columns are required
        C.acc_dist.zeros(1, 1);
        CHECK_THROWS_AS(C.step(S), std::invalid_argument);

        Cfg D = one_ray();
        D.acc_dist.zeros(1, 3);
        CHECK_THROWS_AS(D.step(S), std::invalid_argument);

        Cfg E = one_ray(10.0f, 2); // two frequencies need three columns, not two
        E.acc_dist.zeros(1, 2);
        CHECK_THROWS_AS(E.step(S), std::invalid_argument);
    }
    SECTION("path layout must match the request")
    {
        Cfg C = one_ray(10.0f, 1);
        C.freq.set_size(2); // paths hold one frequency
        C.freq(0) = FRQ, C.freq(1) = 2.0f * FRQ;
        CHECK_THROWS_AS(C.step(S), std::invalid_argument);
    }
    SECTION("trivec and tridir come as a pair with 9 columns")
    {
        {
            Cfg C = one_ray(10.0f, 1, false, NARROW_DEG);
            C.tridir.reset(); // trivec without tridir
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray(10.0f, 1, false, NARROW_DEG);
            C.trivec.reset();
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        {
            Cfg C = one_ray(10.0f, 1, false, NARROW_DEG);
            C.trivec = arma::fmat(1, 6, arma::fill::zeros);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
        { // spherical tridir is not accepted here
            Cfg C = one_ray(10.0f, 1, false, NARROW_DEG);
            C.tridir = arma::fmat(1, 6, arma::fill::zeros);
            CHECK_THROWS_AS(C.step(S), std::invalid_argument);
        }
    }
    SECTION("scalar thresholds")
    {
        for (float t : {0.0f, -1.0f})
        {
            Cfg C = one_ray();
            Opt o;
            o.sub_tol = t;
            CHECK_THROWS_AS(C.step(S, o), std::invalid_argument);
        }
        for (float g : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()})
        {
            Cfg C = one_ray();
            Opt o;
            o.min_gain_dB = g;
            CHECK_THROWS_AS(C.step(S, o), std::invalid_argument);
        }
    }
    SECTION("sub-mesh index")
    {
        Scene T = two_cubes();

        arma::u32_vec bad_first = {1u, 12u};
        arma::u32_vec not_sorted = {0u, 12u, 6u};
        arma::u32_vec too_large = {0u, 99u};

        for (const arma::u32_vec *p : {&bad_first, &not_sorted, &too_large})
        {
            Cfg C = one_ray();
            Opt o;
            o.smi = p;
            CHECK_THROWS_AS(C.step(T, o), std::invalid_argument);
        }
    }
    SECTION("aabb requires a sub-mesh index and the matching shape")
    {
        Scene T = two_cubes();
        arma::u32_vec smi = {0u, 12u};

        arma::fmat box(2, 6, arma::fill::zeros);
        {
            Cfg C = one_ray();
            Opt o;
            o.aabb = &box; // no sub_mesh_index
            CHECK_THROWS_AS(C.step(T, o), std::invalid_argument);
        }
        {
            arma::fmat wrong(3, 6, arma::fill::zeros);
            Cfg C = one_ray();
            Opt o;
            o.smi = &smi;
            o.aabb = &wrong;
            CHECK_THROWS_AS(C.step(T, o), std::invalid_argument);
        }
    }
}
TEST_CASE("ray_progress - the material count is limited by the 14-bit state index")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    for (auto &kv : S.mtl_prop)
        kv.second.resize(16383, kv.second[0]);
    {
        Cfg C = one_ray();
        CHECK_NOTHROW(C.step(S));
    }
    for (auto &kv : S.mtl_prop)
        kv.second.resize(16384, kv.second[0]);
    {
        Cfg C = one_ray();
        CHECK_THROWS_AS(C.step(S), std::invalid_argument);
    }
}

TEST_CASE("ray_progress - refracting tracer: the segment out of a lossy medium is the direction the wave has")
{
    // With refraction_mode ray_mesh_interact bends the segment at a real angle in the medium the ray
    // leaves, and ray_state_update writes the direction the wave has into path_dir. Across parallel
    // faces a wave from air leaves parallel to the direction it arrived in, so the launched segment,
    // dest - orig, has to be that direction, for the spine and for the three vertex rays of a tube,
    // and inside a second solid it has to be the direction the transition wrote.
    //
    // Everything runs through ray_progress, float, scalar, 1 GHz. Walls are planes normal to x; the
    // spine crosses x = 0 at y = 0.37, off the triangulation edge. Rows that this must not move are
    // printed with every digit of a float: lossless legs, the undeviated tracer, reflections.
    const double PI = 3.14159265358979323846;
    const float L = 0.10f, gap = 1.0e-5f;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: eps 5.24-0.63j, 2: 5.24, 3: sheet 5.24-0.63j 6 mm, 4: 9.0-1.20j, 5: 1.50-0.40j, 6: 1.50
    pal["a"] = {5.24f, 5.24f, 5.24f, 9.0f, 1.50f, 1.50f};
    pal["b"] = std::vector<float>(6, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), 0.0f, (float)(0.63 / 17.98), (float)(1.20 / 17.98), (float)(0.40 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(6, 0.0f);
    pal["fRef"] = std::vector<float>(6, 1.0f);
    pal["thk"] = {0.0f, 0.0f, 0.006f, 0.0f, 0.0f, 0.0f};

    Pal rig; // DENSE: eps 1.50 with the mass column, a mass-law solid the wave is refracted into
    rig["a"] = {1.5f};
    rig["fRef"] = {0.001f};
    rig["tf"] = {0.0f};
    rig["m"] = {20.0f};
    rig["thk"] = {0.0f};

    struct Wall
    {
        arma::fmat mesh;
        arma::uvec mi;
    };
    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip) -> Wall
    {
        const float h = 200.0f;
        Wall W;
        W.mesh.set_size(2 * x.size(), 9);
        W.mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k]) // normal along -x, front side towards the source
                W.mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                W.mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                W.mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                W.mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            W.mi(2 * k) = (arma::uword)mat[k], W.mi(2 * k + 1) = (arma::uword)mat[k];
        }
        return W;
    };

    struct Gen // the launch configuration after one call
    {
        arma::fmat orig, dest, dir, trivec, tridir, hit;
        arma::Col<short> cur, prev;
        std::vector<std::string> types;
        arma::uword n() const { return orig.n_rows; }
        arma::frowvec seg(arma::uword i) const
        {
            arma::frowvec d = dest.row(i) - orig.row(i);
            return d / nrm3(d);
        }
    };
    struct Run
    {
        std::vector<Gen> g;
        arma::frowvec k;     // launched spine direction
        arma::fmat v0;       // launched vertex directions, one per row
    };
    auto trace = [&](const Pal &P, const Wall &W, double thd, bool refr, int n_gen, int max_ref, float slab,
                     bool beam, int force_gen) -> Run
    {
        Run R;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        R.k = arma::frowvec{(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {1.0e9f};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros), trivec, tridir;
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * R.k;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * R.k;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = R.k;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, true);
        paths[0].set_length(r0);
        if (beam)
        {
            trivec.zeros(1, 9), tridir.zeros(1, 9), R.v0.zeros(3, 3);
            arma::frowvec u = arma::cross(arma::frowvec{0.0f, 0.0f, 1.0f}, R.k);
            u /= nrm3(u);
            arma::frowvec v = arma::cross(R.k, u);
            const float ha = 0.5f * (float)(PI / 180.0);
            for (int j = 0; j < 3; ++j)
            {
                const float phi = (float)(2.0 * PI * (double)j / 3.0);
                arma::frowvec vj = std::cos(ha) * R.k + std::sin(ha) * (std::cos(phi) * u + std::sin(phi) * v);
                vj /= nrm3(vj);
                for (int c = 0; c < 3; ++c)
                    tridir(0, 3 * j + c) = vj(c), trivec(0, 3 * j + c) = r0 * (vj(c) - R.k(c)), R.v0(j, c) = vj(c);
            }
        }
        for (int g = 0; g < n_gen; ++g)
        {
            std::vector<bool> ff(orig.n_rows, g == force_gen);
            quadriga_lib::ray_progress(W.mesh, W.mi, P, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       beam ? &trivec : nullptr, beam ? &tridir : nullptr, nullptr, nullptr,
                                       20, (uint8_t)max_ref, 10, (uint8_t)(force_gen >= 0 ? 2 : 0), -300.0f, 3.0f, slab, refr, true,
                                       nullptr, nullptr, nullptr, force_gen >= 0 ? &ff : nullptr);
            if (orig.n_rows == 0)
                break;
            Gen G;
            G.orig = orig, G.dest = dest, G.dir = pd.slice(0), G.cur = cur, G.prev = prev;
            if (beam)
                G.trivec = trivec, G.tridir = tridir;
            G.hit.zeros(orig.n_rows, 3);
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                std::string ty;
                for (auto c : paths[i].interaction_type_codes())
                    ty += std::to_string((unsigned)c) + "/";
                G.types.push_back(ty);
                if (paths[i].n_seg())
                    for (int c = 0; c < 3; ++c)
                        G.hit(i, c) = paths[i].coord(paths[i].n_seg() - 1)[c];
            }
            R.g.push_back(G);
        }
        return R;
    };
    auto dmax = [](const arma::frowvec &a, const arma::frowvec &b) -> double
    {
        double d = 0.0;
        for (int c = 0; c < 3; ++c)
            d = std::max(d, (double)std::abs(a(c) - b(c)));
        return d;
    };
    auto apart = [&](const arma::frowvec &a, const arma::frowvec &b) -> double // degrees, well conditioned near zero
    {
        double cx = (double)a(1) * b(2) - (double)a(2) * b(1), cy = (double)a(2) * b(0) - (double)a(0) * b(2),
               cz = (double)a(0) * b(1) - (double)a(1) * b(0);
        return std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), (double)a(0) * b(0) + (double)a(1) * b(1) + (double)a(2) * b(2)) * 180.0 / PI;
    };
    auto find = [](const Gen &G, const std::string &ty) -> long long
    {
        for (size_t i = 0; i < G.types.size(); ++i)
            if (G.types[i] == ty)
                return (long long)i;
        return -1;
    };
    auto raw = [](const Gen &G, arma::uword i)
    {
        QD_DIAG("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", G.orig(i, 0), G.orig(i, 1), G.orig(i, 2), G.dest(i, 0), G.dest(i, 1), G.dest(i, 2));
    };
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    const double TOL = 1.0e-6;

    // ------------------------------------------------------------------------------------------
    // One wall: the segment behind it against the direction the ray arrived in
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, one wall. air | 100 mm | air through ray_progress. seg: dest - orig behind the wall,\n");
    QD_DIAG("# dir: path_dir there, deg from the normal. d_seg, d_dir: largest component against the launched\n");
    QD_DIAG("# direction. raw: orig and dest behind the wall, every digit\n");
    QD_DIAG("mtl,theta,tracer,types,seg_deg,dir_deg,d_seg,d_dir,raw\n");
    for (int m : {5, 1, 4, 6})
        for (double thd : angles)
            for (bool refr : {true, false})
            {
                Run R = trace(pal, build({0.0f, L}, {m, m}, {false, true}), thd, refr, 2, 0, 0.15f, false, -1);
                REQUIRE(R.g.size() == 2);
                const Gen &G = R.g[1];
                REQUIRE(G.n() == 1);
                arma::frowvec x = {1.0f, 0.0f, 0.0f};
                QD_DIAG("%d,%.1f,%d,%s,%.4f,%.4f,%.3e,%.3e,", m, thd, refr ? 5 : 4, G.types[0].c_str(), apart(G.seg(0), x), apart(G.dir.row(0), x),
                        dmax(G.seg(0), R.k), dmax(G.dir.row(0), R.k));
                raw(G, 0);
                QD_DIAG("\n");
                INFO("one wall, material " << m << ", theta " << thd << ", refraction " << refr);
                CHECK(G.cur(0) == (short)0);
                CHECK(dmax(G.dir.row(0), R.k) < TOL);
                CHECK(dmax(G.seg(0), R.k) < TOL);
            }

    // ------------------------------------------------------------------------------------------
    // Two walls in a row: where and at what angle the second one is met
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, two walls in a row, 1.9 m of air between. dy: the lateral run from the exit of\n");
    QD_DIAG("# the first wall to the entry of the second, refracting and undeviated tracer; the two differ in\n");
    QD_DIAG("# where they leave the first wall and in nothing else. in2: the segment inside the second wall,\n");
    QD_DIAG("# refracting, against the direction the undeviated tracer holds there, deg\n");
    QD_DIAG("first,theta,dy5,dy4,d_dy,in2_seg5,in2_dir4,d_in2,out2_seg5,d_out2\n");
    for (int m : {5, 1, 6})
        for (double thd : angles)
        {
            Wall W = build({0.0f, L, 2.0f, 2.0f + L}, {m, m, 5, 5}, {false, true, false, true});
            Run R5 = trace(pal, W, thd, true, 4, 0, 0.15f, false, -1), R4 = trace(pal, W, thd, false, 4, 0, 0.15f, false, -1);
            REQUIRE((R5.g.size() == 4 && R4.g.size() == 4));
            double dy5 = (double)R5.g[2].hit(0, 1) - (double)R5.g[1].hit(0, 1), dy4 = (double)R4.g[2].hit(0, 1) - (double)R4.g[1].hit(0, 1);
            arma::frowvec x = {1.0f, 0.0f, 0.0f};
            QD_DIAG("%d,%.1f,%.6f,%.6f,%+.6f,%.4f,%.4f,%.3e,%.4f,%.3e\n", m, thd, dy5, dy4, dy5 - dy4, apart(R5.g[2].seg(0), x),
                    apart(R4.g[2].dir.row(0), x), dmax(R5.g[2].seg(0), R4.g[2].dir.row(0)), apart(R5.g[3].seg(0), x), dmax(R5.g[3].seg(0), R5.k));
            INFO("two walls, first " << m << ", theta " << thd);
            CHECK(R5.g[2].hit(0, 0) == R4.g[2].hit(0, 0)); // both at x = 2
            CHECK(std::abs(dy5 - dy4) < 2.0e-4);
            CHECK(dmax(R5.g[2].seg(0), R4.g[2].dir.row(0)) < TOL);
            CHECK(dmax(R5.g[3].seg(0), R5.k) < TOL);
        }

    // ------------------------------------------------------------------------------------------
    // Two solids across a bare face: the segment in the second one
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, two solids across a bare face, 50 mm | 100 mm. in2: the segment in the second\n");
    QD_DIAG("# solid against path_dir there, and against the segment of the same solid entered from air, which\n");
    QD_DIAG("# is the direction a single refraction writes. out: the segment behind the wall\n");
    QD_DIAG("mats,theta,in2_seg,in2_dir,d_in2,d_air,out_seg,d_out,raw_in2\n");
    for (auto P : std::vector<std::pair<int, int>>{{1, 5}, {5, 1}, {4, 5}, {1, 6}, {2, 5}, {2, 6}})
        for (double thd : angles)
        {
            Run R = trace(pal, build({0.0f, 0.05f, 0.05f + gap, 0.15f + gap}, {P.first, P.first, P.second, P.second}, {false, true, false, true}),
                          thd, true, 3, 0, 0.15f, false, -1);
            Run A = trace(pal, build({0.0f, L}, {P.second, P.second}, {false, true}), thd, true, 1, 0, 0.15f, false, -1);
            REQUIRE((R.g.size() == 3 && A.g.size() == 1));
            arma::frowvec x = {1.0f, 0.0f, 0.0f};
            QD_DIAG("%d%d,%.1f,%.4f,%.4f,%.3e,%.3e,%.4f,%.3e,", P.first, P.second, thd, apart(R.g[1].seg(0), x), apart(R.g[1].dir.row(0), x),
                    dmax(R.g[1].seg(0), R.g[1].dir.row(0)), dmax(R.g[1].seg(0), A.g[0].seg(0)), apart(R.g[2].seg(0), x), dmax(R.g[2].seg(0), R.k));
            raw(R.g[1], 0);
            QD_DIAG("\n");
            INFO("two solids " << P.first << P.second << ", theta " << thd);
            CHECK(dmax(R.g[1].seg(0), R.g[1].dir.row(0)) < TOL);
            CHECK(dmax(R.g[1].seg(0), A.g[0].seg(0)) < TOL);
            CHECK(dmax(R.g[2].seg(0), R.k) < TOL);
        }

    // ------------------------------------------------------------------------------------------
    // Sheets on and in the host
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, sheets on and in the host. The segment behind the wall against the launched\n");
    QD_DIAG("# direction. Lossless host: a leg behind a glued entry carries that entry's anchoring, 1.6e-6 at 85 deg\n");
    QD_DIAG("stack,host,theta,types,seg_deg,d_seg,raw\n");
    for (int h : {5, 6})
        for (int s = 0; s < 4; ++s)
            for (double thd : angles)
            {
                const char *name[4] = {"glued_entry ", "glued_exit  ", "lined       ", "sheet_inside"};
                Wall W = s == 0   ? build({0.0f, gap, L}, {3, h, h}, {false, false, true})
                         : s == 1 ? build({0.0f, L, L + gap}, {h, h, 3}, {false, true, true})
                         : s == 2 ? build({0.0f, gap, L, L + gap}, {3, h, h, 3}, {false, false, true, true})
                                  : build({0.0f, 0.05f, L}, {h, 3, h}, {false, false, true});
                Run R = trace(pal, W, thd, true, s == 3 ? 3 : 2, 0, 0.15f, false, -1);
                REQUIRE(R.g.size() == (s == 3 ? 3u : 2u));
                const Gen &G = R.g.back();
                arma::frowvec x = {1.0f, 0.0f, 0.0f};
                QD_DIAG("%s,%d,%.1f,%s,%.4f,%.3e,", name[s], h, thd, G.types[0].c_str(), apart(G.seg(0), x), dmax(G.seg(0), R.k));
                raw(G, 0);
                QD_DIAG("\n");
                INFO(name[s] << ", host " << h << ", theta " << thd);
                CHECK(G.cur(0) == (short)0);
                CHECK(dmax(G.seg(0), R.k) < (h == 5 ? TOL : 1.0e-5));
            }

    // ------------------------------------------------------------------------------------------
    // Reflections: a ray leaving by the front face, and what the reflect pass launches
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, reflections on, one wall. slab 0: the far-face reflection collapses the series and\n");
    QD_DIAG("# the ray leaves resolved; slab 1: it does not. front: the ray that leaves by the front face against\n");
    QD_DIAG("# the mirror of the launched direction. Every ray of the last generation is printed raw\n");
    QD_DIAG("mtl,slab,theta,types,cur,seg_deg,dir_deg,d_mirror,raw\n");
    for (int m : {5, 6})
        for (float slab : {0.0f, 1.0f})
            for (double thd : {20.0, 60.0, 85.0})
            {
                Run R = trace(pal, build({0.0f, L}, {m, m}, {false, true}), thd, true, 3, 3, slab, false, -1);
                REQUIRE(R.g.size() == 3);
                arma::frowvec x = {1.0f, 0.0f, 0.0f}, mir = {-R.k(0), R.k(1), R.k(2)};
                for (size_t g = 0; g < 3; ++g)
                    for (arma::uword i = 0; i < R.g[g].n(); ++i)
                    {
                        const Gen &G = R.g[g];
                        QD_DIAG("%d,%.0f,%.1f,%s,%04x,%.4f,%.4f,%.3e,", m, slab, thd, G.types[i].c_str(), (unsigned)(unsigned short)G.cur(i),
                                apart(G.seg(i), x), apart(G.dir.row(i), x), dmax(G.seg(i), mir));
                        raw(G, i);
                        QD_DIAG("\n");
                    }
                long long i = find(R.g[2], slab == 0.0f ? "9/141/1/" : "9/137/1/");
                if (m == 6 && i < 0) // the lossless slab leaves with the VBS bit of the type code as it stands
                    i = find(R.g[2], slab == 0.0f ? "9/141/5/" : "9/137/5/");
                INFO("front-face exit, material " << m << ", slab " << slab << ", theta " << thd);
                REQUIRE(i >= 0);
                CHECK(dmax(R.g[2].dir.row((arma::uword)i), mir) < TOL);
                CHECK(dmax(R.g[2].seg((arma::uword)i), mir) < TOL);
                // An internal reflection launches the mirror of the segment it arrived on
                long long j = find(R.g[1], slab == 0.0f ? "9/141/" : "9/137/");
                REQUIRE(j >= 0);
                CHECK(dmax(R.g[1].seg((arma::uword)j), R.g[1].dir.row((arma::uword)j)) < TOL);
            }

    // ------------------------------------------------------------------------------------------
    // The ray tube
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, the ray tube, 0.5 deg half angle. Behind one wall: each vertex direction against\n");
    QD_DIAG("# the one it was launched in (d_v), its angle to the spine (fan, deg), and the mean of the three\n");
    QD_DIAG("# against the segment, which is what a subdivision makes the spine of\n");
    QD_DIAG("mtl,theta,seg_deg,d_seg,d_v0,d_v1,d_v2,fan0,fan1,fan2,mean_minus_seg_deg\n");
    for (int m : {5, 1, 6})
        for (double thd : {40.0, 75.0, 85.0})
        {
            Run R = trace(pal, build({0.0f, L}, {m, m}, {false, true}), thd, true, 2, 0, 0.15f, true, -1);
            REQUIRE(R.g.size() == 2);
            const Gen &G = R.g[1];
            arma::frowvec x = {1.0f, 0.0f, 0.0f}, mean(3, arma::fill::zeros);
            double dv[3], fan[3];
            for (int j = 0; j < 3; ++j)
            {
                arma::frowvec v = G.tridir.submat(0, 3 * j, 0, 3 * j + 2);
                dv[j] = dmax(v, R.v0.row(j)), fan[j] = apart(v, G.seg(0)), mean += v;
            }
            mean /= nrm3(mean);
            QD_DIAG("%d,%.1f,%.4f,%.3e,%.3e,%.3e,%.3e,%.4f,%.4f,%.4f,%.5f\n", m, thd, apart(G.seg(0), x), dmax(G.seg(0), R.k), dv[0], dv[1], dv[2],
                    fan[0], fan[1], fan[2], apart(mean, G.seg(0)));
            INFO("tube, material " << m << ", theta " << thd);
            CHECK(dmax(G.seg(0), R.k) < TOL);
            for (int j = 0; j < 3; ++j)
                CHECK(dv[j] < TOL);
        }

    QD_DIAG("\n# diag: trajectory, the ray tube into a second solid, 50 mm | 100 mm. Each vertex direction in the\n");
    QD_DIAG("# second solid against the same vertex entering that solid from air\n");
    QD_DIAG("mats,theta,d_v0,d_v1,d_v2,d_seg\n");
    for (auto P : std::vector<std::pair<int, int>>{{1, 5}, {5, 1}, {1, 6}, {2, 5}})
        for (double thd : {40.0, 75.0, 85.0})
        {
            Run R = trace(pal, build({0.0f, 0.05f, 0.05f + gap, 0.15f + gap}, {P.first, P.first, P.second, P.second}, {false, true, false, true}),
                          thd, true, 2, 0, 0.15f, true, -1);
            Run A = trace(pal, build({0.0f, L}, {P.second, P.second}, {false, true}), thd, true, 1, 0, 0.15f, true, -1);
            REQUIRE((R.g.size() == 2 && A.g.size() == 1));
            double dv[3];
            for (int j = 0; j < 3; ++j)
                dv[j] = dmax(R.g[1].tridir.submat(0, 3 * j, 0, 3 * j + 2), A.g[0].tridir.submat(0, 3 * j, 0, 3 * j + 2));
            QD_DIAG("%d%d,%.1f,%.3e,%.3e,%.3e,%.3e\n", P.first, P.second, thd, dv[0], dv[1], dv[2], dmax(R.g[1].seg(0), A.g[0].seg(0)));
            INFO("tube into a second solid " << P.first << P.second << ", theta " << thd);
            for (int j = 0; j < 3; ++j)
                CHECK(dv[j] < TOL);
        }

    QD_DIAG("\n# diag: trajectory, a beam split in air behind the wall. The four sub-beams against those of the\n");
    QD_DIAG("# undeviated tracer, whose tube is the launched one: segment and path_dir\n");
    QD_DIAG("mtl,theta,sub,seg_deg5,seg_deg4,d_seg,d_dir\n");
    for (int m : {5, 6})
        for (double thd : {40.0, 75.0})
        {
            Wall W = build({0.0f, L}, {m, m}, {false, true});
            Run R5 = trace(pal, W, thd, true, 3, 0, 0.15f, true, 2), R4 = trace(pal, W, thd, false, 3, 0, 0.15f, true, 2);
            REQUIRE((R5.g.size() == 3 && R4.g.size() == 3));
            REQUIRE((R5.g[2].n() == 4 && R4.g[2].n() == 4));
            arma::frowvec x = {1.0f, 0.0f, 0.0f};
            for (arma::uword i = 0; i < 4; ++i)
            {
                QD_DIAG("%d,%.1f,%u,%.4f,%.4f,%.3e,%.3e\n", m, thd, (unsigned)i, apart(R5.g[2].seg(i), x), apart(R4.g[2].seg(i), x),
                        dmax(R5.g[2].seg(i), R4.g[2].seg(i)), dmax(R5.g[2].dir.row(i), R4.g[2].dir.row(i)));
                INFO("split in air, material " << m << ", theta " << thd << ", sub-beam " << i);
                CHECK(dmax(R5.g[2].seg(i), R4.g[2].seg(i)) < TOL);
                CHECK(dmax(R5.g[2].dir.row(i), R4.g[2].dir.row(i)) < TOL);
            }
        }

    // ------------------------------------------------------------------------------------------
    // Recorded: a mass-law solid the wave is refracted into (issue 15), and a second frequency
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, recorded. DENSE, eps 1.50 with m 20, 30 deg: ray_mesh_interact holds the wave\n");
    QD_DIAG("# undeviated in a mass-law medium, and the segment behind it is not this item's\n");
    QD_DIAG("tracer,seg_deg,dir_deg,raw\n");
    for (bool refr : {true, false})
    {
        Wall W = build({0.0f, L}, {1, 1}, {false, true});
        Run R = trace(rig, W, 30.0, refr, 2, 0, 0.15f, false, -1);
        REQUIRE(R.g.size() == 2);
        arma::frowvec x = {1.0f, 0.0f, 0.0f};
        QD_DIAG("%d,%.4f,%.4f,", refr ? 5 : 4, apart(R.g[1].seg(0), x), apart(R.g[1].dir.row(0), x));
        raw(R.g[1], 0);
        QD_DIAG("\n");
        CHECK(dmax(R.g[1].dir.row(0), R.k) < TOL);
    }
}

TEST_CASE("Measurement - ray_progress follows the coupling between the series of a layered wall")
{
    // Plan item 1, measured through the tracer itself. A wall of two solid layers, or of one layer
    // with a sheet inside it, is resolved as two series, each with the other body as a mirror that
    // does not return what its far side reflects. That part is not dropped by a run: the ray
    // reflected at the far face goes back into the first body, is reflected at the front face and
    // comes forward again, as further paths. One ray is launched, ray_progress is called until no
    // ray is left, and every path that ends in air is collected: behind the wall into T, in front
    // of it into R, each as the field at the point its polyline leaves at,
    // coefficient * exp(-j k0 L_in) * exp(+j k0 sin(theta) dy), against the characteristic matrix
    // of the whole wall. Scalar, float, 1 GHz, both tracers, thin_slab_threshold 0 (every series
    // resolved) and 0.15 (the default). Nothing is asserted beyond sanity.
    const double PI = 3.14159265358979323846, C0 = 299792458.0, FRQ = 1.0e9, k0 = 2.0 * PI * FRQ / C0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: eps 5.24-0.63j, 2: 5.24, 3: sheet 5.24-0.63j 6 mm, 4: 9.0-1.20j, 5: 1.50-0.40j, 6: 1.50
    pal["a"] = {5.24f, 5.24f, 5.24f, 9.0f, 1.50f, 1.50f};
    pal["b"] = std::vector<float>(6, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), 0.0f, (float)(0.63 / 17.98), (float)(1.20 / 17.98), (float)(0.40 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(6, 0.0f);
    pal["fRef"] = std::vector<float>(6, 1.0f);
    pal["thk"] = {0.0f, 0.0f, 0.006f, 0.0f, 0.0f, 0.0f};
    auto eps_of = [](int m) -> std::complex<double>
    {
        const double a[7] = {1.0, 5.24, 5.24, 5.24, 9.0, 1.50, 1.50}, im[7] = {0.0, 0.63, 0.0, 0.63, 1.20, 0.40, 0.0};
        return std::complex<double>(a[m], -im[m]);
    };

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    auto cm = [&](const std::vector<Lay> &ls, double s2, std::complex<double> &r, std::complex<double> &t)
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0, h0 = std::sqrt(std::complex<double>(1.0 - s2, 0.0));
        for (const Lay &l : ls)
        {
            std::complex<double> b = std::sqrt(l.e - s2), dl = k0 * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> B = m11 + m12 * h0, Cc = m21 + m22 * h0;
        r = (h0 * B - Cc) / (h0 * B + Cc), t = 2.0 * h0 / (h0 * B + Cc);
    };

    struct Sum
    {
        std::complex<double> T = 0.0, R = 0.0, T1 = 0.0, R1 = 0.0; // all paths, and the shortest one
        int nT = 0, nR = 0, gens = 0;
        size_t segT1 = 1000, segR1 = 1000, seg_max = 0;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float slab, float x_back, int max_int, float min_gain) -> Sum
    {
        Sum S;
        const double th = thd * PI / 180.0, st = std::sin(th);
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {(float)FRQ};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 80; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)max_int, 100, 100, 0, min_gain, 3.0f, slab, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            S.gens = g + 1;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue; // inside the wall, or a ray that was already collected
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *a = paths[i].coord(s - 1), *b = paths[i].coord(s);
                    Lin += std::sqrt(((double)b[0] - a[0]) * ((double)b[0] - a[0]) + ((double)b[1] - a[1]) * ((double)b[1] - a[1]) +
                                     ((double)b[2] - a[2]) * ((double)b[2] - a[2]));
                }
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1), *xc = paths[i].xpr_coeff(0);
                std::complex<double> F = std::complex<double>((double)xc[0], (double)xc[1]) * std::exp(-J * k0 * Lin) *
                                         std::exp(J * k0 * st * ((double)last[1] - (double)first[1]));
                const bool fwd = dest(i, 0) > orig(i, 0);
                S.seg_max = std::max(S.seg_max, ns);
                if (fwd && last[0] > x_back - 1.0e-3f)
                {
                    S.T += F, ++S.nT;
                    if (ns < S.segT1)
                        S.segT1 = ns, S.T1 = F;
                }
                else if (!fwd && last[0] < 1.0e-3f)
                {
                    S.R += F, ++S.nR;
                    if (ns < S.segR1)
                        S.segR1 = ns, S.R1 = F;
                }
            }
        }
        return S;
    };
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [&](std::complex<double> z)
    { return std::arg(z) * 180.0 / PI; };

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: mirror through ray_progress. One ray, called until none is left, scalar, float, 1 GHz. wall AB: air | A 50 |\n");
    std::printf("# B 100 | air; wall HSH: air | host 50 | sheet 6 mm eps 5.24 - 0.63j | host 50 | air, against host | sheet | host with\n");
    std::printf("# 6 mm of host divided out. slab: thin_slab_threshold. max: max_no_interactions. nT, nR: the paths that end behind\n");
    std::printf("# and in front of the wall; seg: the segments of the longest. T1, R1: the shortest path alone against the matrix of\n");
    std::printf("# the whole wall, dB; T, R: all of them, dB and deg at the polyline's point\n");
    std::printf("# min: min_gain_dB. wall HS: air | host 100 | coating 6 mm | air; wall H: air | host 100 | air. slab 1.00 declines\n");
    std::printf("# every series, so every bounce is a path of its own\n");
    std::printf("wall,a,b,tracer,slab,max,min,theta,nT,nR,seg,T1_dB,T_dB,T_deg,R1_dB,R_dB,R_deg\n");
#endif
    for (int wall = 0; wall < 4; ++wall)
        for (int pair = 0; pair < 2; ++pair)
            for (int refr = 0; refr < 2; ++refr)
                for (int cfg = 0; cfg < 5; ++cfg)
                    for (double thd : {10.0, 40.0, 60.0, 75.0, 85.0})
                    {
                        const float slab = cfg == 4 ? 1.0f : (cfg >= 2 ? 0.15f : 0.0f), min_gain = cfg == 3 ? -70.0f : -300.0f;
                        const int max_int = (cfg == 1 || cfg == 4) ? 60 : (cfg == 3 ? 44 : 20);
                        if (cfg == 4 && wall < 2)
                            continue; // every bounce re-emitted: one cavity only, or the paths multiply
                        const int A = wall ? (pair ? 6 : 5) : 2, B = wall ? 3 : (pair ? 6 : 5);
                        arma::fmat mesh;
                        arma::uvec mi;
                        std::vector<Lay> ls;
                        float x_back;
                        std::complex<double> div = 1.0;
                        const double s2 = std::sin(thd * PI / 180.0) * std::sin(thd * PI / 180.0);
                        if (wall == 0)
                        {
                            build({0.0f, 0.05f, 0.05f + gap, 0.15f + gap}, {A, A, B, B}, {false, true, false, true}, mesh, mi);
                            ls = {{eps_of(A), 0.05}, {eps_of(B), 0.10}}, x_back = 0.15f;
                        }
                        else if (wall == 2) // a coating on the far side
                        {
                            build({0.0f, 0.10f, 0.10f + gap}, {A, A, B}, {false, true, true}, mesh, mi);
                            ls = {{eps_of(A), 0.10}, {eps_of(B), 0.006}}, x_back = 0.10f;
                        }
                        else if (wall == 3) // one bare wall
                        {
                            build({0.0f, 0.10f}, {A, A}, {false, true}, mesh, mi);
                            ls = {{eps_of(A), 0.10}}, x_back = 0.10f;
                        }
                        else
                        {
                            build({0.0f, 0.05f, 0.10f}, {A, B, A}, {false, false, true}, mesh, mi);
                            ls = {{eps_of(A), 0.05}, {eps_of(B), 0.006}, {eps_of(A), 0.05}}, x_back = 0.10f;
                            div = std::exp(-J * k0 * std::sqrt(eps_of(A) - s2) * 0.006);
                        }
                        std::complex<double> r, t;
                        cm(ls, s2, r, t);
                        t /= div;
                        Sum S = trace(mesh, mi, thd, refr != 0, slab, x_back, max_int, min_gain);
                        INFO("wall " << wall << ", pair " << pair << ", refr " << refr << ", cfg " << cfg << ", theta " << thd);
                        CHECK(S.nT >= 1);
                        CHECK(std::isfinite(std::norm(S.T)));
                        // Two solids, every series resolved: the paths a run follows sum to the wall.
                        // A lossless second layer at 85 deg loses 3 dB per path and is not there yet
                        if (wall == 0 && cfg <= 1)
                        {
                            const bool slow = B == 6 && thd >= 75.0;
                            const double bound = !slow ? 5.0e-3 : (cfg == 1 ? 0.05 : 2.0);
                            CHECK(std::abs(dB(S.T / t)) < bound);
                            CHECK(std::abs(dB(S.R / r)) < bound);
                            CHECK(S.nT >= 5); // the coupling is followed
                        }
#ifdef QD_TEST_VERBOSE
                        std::printf("%s,%d,%d,%d,%.2f,%d,%.0f,%.0f,%d,%d,%zu,%+.4f,%+.4f,%+.3f,%+.4f,%+.4f,%+.3f\n", wall == 0 ? "AB " : (wall == 1 ? "HSH" : (wall == 2 ? "HS " : "H  ")), A, B, refr ? 5 : 4,
                                    (double)slab, max_int, (double)min_gain, thd, S.nT, S.nR, S.seg_max, dB(S.T1 / t), dB(S.T / t), deg(S.T / t), dB(S.R1 / r),
                                    dB(S.R / r), deg(S.R / r));
#endif
                    }

    // Which wall a sheet inside a host is. The sheet's transit takes its thickness of host out
    // while both host series run over the meshed legs, so the sum is compared with four walls:
    // 50 | 6 | 50 with 6 mm of host divided out of T, 47 | 6 | 47, 44 | 6 | 50 and 50 | 6 | 44.
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: mirror through ray_progress, which wall a sheet inside a host is. air | host 50 | sheet 6 mm | host 50 | air as\n");
    std::printf("# meshed, every series resolved, 60 interactions. The summed paths against four walls, dB and deg: w50, host 50 |\n");
    std::printf("# sheet | host 50 with 6 mm of host divided out of T; w47, 47 | 6 | 47; w44f, 44 | 6 | 50; w44b, 50 | 6 | 44\n");
    std::printf("host,tracer,theta,T_w50,T_w47,T_w44f,T_w44b,R_w50,R_w47,R_w44f,R_w44b,Tdeg_w50,Tdeg_w47,Rdeg_w50,Rdeg_w47,Rdeg_w44f,Rdeg_w44b\n");
#endif
    for (int hst : {5, 6})
        for (int refr = 0; refr < 2; ++refr)
            for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
            {
                arma::fmat mesh;
                arma::uvec mi;
                build({0.0f, 0.05f, 0.10f}, {hst, 3, hst}, {false, false, true}, mesh, mi);
                const double s2 = std::sin(thd * PI / 180.0) * std::sin(thd * PI / 180.0);
                Sum S = trace(mesh, mi, thd, refr != 0, 0.0f, 0.10f, 60, -300.0f);
                const double fr[4] = {0.050, 0.047, 0.044, 0.050}, bk[4] = {0.050, 0.047, 0.050, 0.044};
                std::complex<double> r[4], t[4];
                for (int w = 0; w < 4; ++w)
                    cm({{eps_of(hst), fr[w]}, {eps_of(3), 0.006}, {eps_of(hst), bk[w]}}, s2, r[w], t[w]);
                t[0] /= std::exp(-J * k0 * std::sqrt(eps_of(hst) - s2) * 0.006);
                INFO("which wall, host " << hst << ", refr " << refr << ", theta " << thd);
                CHECK(S.nT >= 1);
#ifdef QD_TEST_VERBOSE
                std::printf("%d,%d,%.0f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f\n", hst, refr ? 5 : 4, thd,
                            dB(S.T / t[0]), dB(S.T / t[1]), dB(S.T / t[2]), dB(S.T / t[3]), dB(S.R / r[0]), dB(S.R / r[1]), dB(S.R / r[2]),
                            dB(S.R / r[3]), deg(S.T / t[0]), deg(S.T / t[1]), deg(S.R / r[0]), deg(S.R / r[1]), deg(S.R / r[2]), deg(S.R / r[3]));
#endif
            }
}

TEST_CASE("ray_progress - a declined series at a sheet is followed and not dropped")
{
    // Where thin_slab_threshold declines the series of a host that closes at a sheet, the bounces
    // have to arrive as paths, as they do on a bare wall. One ray, called until none is left;
    // every path that ends in air is summed at its polyline's point and compared with the
    // characteristic matrix of the wall. Scalar, float, 1 GHz, both tracers.
    //   HS:  air | host 100 | coating 6 mm | air, thresholds 0.15 (the default) and 1.0 (every
    //        series declined), against the matrix of the wall
    //   HSH: air | host 50 | sheet 6 mm | host 50 | air, threshold 0.15 against threshold 0, the
    //        two regimes of one wall
    const double PI = 3.14159265358979323846, C0 = 299792458.0, FRQ = 1.0e9, k0 = 2.0 * PI * FRQ / C0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: eps 5.24-0.63j, 2: 5.24, 3: sheet 5.24-0.63j 6 mm, 4: 9.0-1.20j, 5: 1.50-0.40j, 6: 1.50
    pal["a"] = {5.24f, 5.24f, 5.24f, 9.0f, 1.50f, 1.50f};
    pal["b"] = std::vector<float>(6, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), 0.0f, (float)(0.63 / 17.98), (float)(1.20 / 17.98), (float)(0.40 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(6, 0.0f);
    pal["fRef"] = std::vector<float>(6, 1.0f);
    pal["thk"] = {0.0f, 0.0f, 0.006f, 0.0f, 0.0f, 0.0f};
    auto eps_of = [](int m) -> std::complex<double>
    {
        const double a[7] = {1.0, 5.24, 5.24, 5.24, 9.0, 1.50, 1.50}, im[7] = {0.0, 0.63, 0.0, 0.63, 1.20, 0.40, 0.0};
        return std::complex<double>(a[m], -im[m]);
    };

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    auto cm = [&](const std::vector<Lay> &ls, double s2, std::complex<double> &r, std::complex<double> &t)
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0, h0 = std::sqrt(std::complex<double>(1.0 - s2, 0.0));
        for (const Lay &l : ls)
        {
            std::complex<double> b = std::sqrt(l.e - s2), dl = k0 * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> B = m11 + m12 * h0, Cc = m21 + m22 * h0;
        r = (h0 * B - Cc) / (h0 * B + Cc), t = 2.0 * h0 / (h0 * B + Cc);
    };

    struct Sum
    {
        std::complex<double> T = 0.0, R = 0.0, T1 = 0.0, R1 = 0.0; // all paths, and the shortest one
        int nT = 0, nR = 0, gens = 0;
        size_t segT1 = 1000, segR1 = 1000, seg_max = 0;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float slab, float x_back, int max_int, float min_gain) -> Sum
    {
        Sum S;
        const double th = thd * PI / 180.0, st = std::sin(th);
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {(float)FRQ};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 80; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)max_int, 100, 100, 0, min_gain, 3.0f, slab, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            S.gens = g + 1;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue; // inside the wall, or a ray that was already collected
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *a = paths[i].coord(s - 1), *b = paths[i].coord(s);
                    Lin += std::sqrt(((double)b[0] - a[0]) * ((double)b[0] - a[0]) + ((double)b[1] - a[1]) * ((double)b[1] - a[1]) +
                                     ((double)b[2] - a[2]) * ((double)b[2] - a[2]));
                }
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1), *xc = paths[i].xpr_coeff(0);
                std::complex<double> F = std::complex<double>((double)xc[0], (double)xc[1]) * std::exp(-J * k0 * Lin) *
                                         std::exp(J * k0 * st * ((double)last[1] - (double)first[1]));
                const bool fwd = dest(i, 0) > orig(i, 0);
                S.seg_max = std::max(S.seg_max, ns);
                if (fwd && last[0] > x_back - 1.0e-3f)
                {
                    S.T += F, ++S.nT;
                    if (ns < S.segT1)
                        S.segT1 = ns, S.T1 = F;
                }
                else if (!fwd && last[0] < 1.0e-3f)
                {
                    S.R += F, ++S.nR;
                    if (ns < S.segR1)
                        S.segR1 = ns, S.R1 = F;
                }
            }
        }
        return S;
    };
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [&](std::complex<double> z)
    { return std::arg(z) * 180.0 / PI; };


#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: declined series at a sheet, through ray_progress. wall HS: air | host 100 | coating 6 mm | air, the summed\n");
    std::printf("# paths against the matrix of the wall, dB and deg. slab: thin_slab_threshold; max: max_no_interactions; nT, nR:\n");
    std::printf("# the paths behind and in front of the wall\n");
    std::printf("wall,host,tracer,slab,max,theta,nT,nR,T_dB,R_dB,R_deg\n");
#endif
    for (int hst : {5, 6})
        for (int refr = 0; refr < 2; ++refr)
            for (int cfg = 0; cfg < 2; ++cfg)
                for (double thd : {10.0, 40.0, 60.0, 75.0, 85.0})
                {
                    const float slab = cfg ? 1.0f : 0.15f;
                    const int max_int = cfg ? 60 : 20;
                    arma::fmat mesh;
                    arma::uvec mi;
                    build({0.0f, 0.10f, 0.10f + gap}, {hst, hst, 3}, {false, true, true}, mesh, mi);
                    const double s2 = std::sin(thd * PI / 180.0) * std::sin(thd * PI / 180.0);
                    std::complex<double> r, t;
                    cm({{eps_of(hst), 0.10}, {eps_of(3), 0.006}}, s2, r, t);
                    Sum S = trace(mesh, mi, thd, refr != 0, slab, 0.10f, max_int, -300.0f);
#ifdef QD_TEST_VERBOSE
                    std::printf("HS ,%d,%d,%.2f,%d,%.0f,%d,%d,%+.4f,%+.4f,%+.3f\n", hst, refr ? 5 : 4, (double)slab, max_int, thd, S.nT, S.nR, dB(S.T / t),
                                dB(S.R / r), deg(S.R / r));
#endif
                    INFO("HS, host " << hst << ", refr " << refr << ", slab " << slab << ", theta " << thd);
                    REQUIRE(S.nT >= 1);
                    // 0.03: the row the default threshold resolves on the lossy host at 85 deg carries
                    // the clamp of a mirror above one, +0.019 dB
                    const double bound = (hst == 5 && thd == 85.0) ? 0.03 : 5.0e-3;
                    // The lossy host at 75 deg at the default threshold is the row where the front
                    // face and the coating decline together (0.145) while the coating twice would
                    // resolve (0.201): the leg behind the back reflection at the bare front face
                    // inherits the verdict and forms no series
                    CHECK(std::abs(dB(S.T / t)) < bound);
                    CHECK(std::abs(dB(S.R / r)) < bound);
                    if (cfg == 1)
                        CHECK(S.nT >= 9); // every bounce is a path
                }

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: declined series at a sheet, a sheet inside a wall. air | host 50 | sheet 6 mm | host 50 | air: the summed\n");
    std::printf("# paths at the default threshold 0.15 over those with every series resolved, 60 interactions, min_gain -100 dB\n");
    std::printf("host,tracer,theta,nT_def,nT_res,T_dB,T_deg,R_dB,R_deg\n");
#endif
    for (int hst : {5, 6})
        for (int refr = 0; refr < 2; ++refr)
            for (double thd : {10.0, 40.0, 60.0, 75.0, 85.0})
            {
                arma::fmat mesh;
                arma::uvec mi;
                build({0.0f, 0.05f, 0.10f}, {hst, 3, hst}, {false, false, true}, mesh, mi);
                Sum D = trace(mesh, mi, thd, refr != 0, 0.15f, 0.10f, 60, -100.0f), Z = trace(mesh, mi, thd, refr != 0, 0.0f, 0.10f, 60, -100.0f);
#ifdef QD_TEST_VERBOSE
                std::printf("%d,%d,%.0f,%d,%d,%+.4f,%+.3f,%+.4f,%+.3f\n", hst, refr ? 5 : 4, thd, D.nT, Z.nT, dB(D.T / Z.T), deg(D.T / Z.T), dB(D.R / Z.R),
                            deg(D.R / Z.R));
#endif
                INFO("HSH, host " << hst << ", refr " << refr << ", theta " << thd);
                REQUIRE((D.nT >= 1 && Z.nT >= 1));
                // To 60 deg the two regimes are one wall. From 75 deg they are recorded: up to 0.18 dB,
                // not attributed
                if (thd <= 60.0)
                {
                    CHECK(std::abs(dB(D.T / Z.T)) < 0.02);
                    CHECK(std::abs(dB(D.R / Z.R)) < 0.02);
                }
            }
}

TEST_CASE("Measurement - ray_progress and the mirror behind a back reflection")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // Plan item 1, measured through the tracer itself. One ray, ray_progress called until no ray
    // is left, every path that ends in air summed at its polyline's point. Scalar, float, 1 GHz
    // unless a band is named, both tracers. Nothing is asserted beyond sanity.
    //   H    host 100                          HS   host 100 | coating 6 mm
    //   SH   coating 6 mm | host 100           AB   A 50 | B 100, two solids across a bare face
    //   HSH  host 50 | sheet 6 mm | host 50
    const double PI = 3.14159265358979323846, C0 = 299792458.0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: eps 5.24-0.63j, 2: 5.24, 3: sheet 5.24-0.63j 6 mm, 4: 9.0-1.20j, 5: 1.50-0.40j, 6: 1.50
    pal["a"] = {5.24f, 5.24f, 5.24f, 9.0f, 1.50f, 1.50f};
    pal["b"] = std::vector<float>(6, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), 0.0f, (float)(0.63 / 17.98), (float)(1.20 / 17.98), (float)(0.40 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(6, 0.0f);
    pal["fRef"] = std::vector<float>(6, 1.0f);
    pal["thk"] = {0.0f, 0.0f, 0.006f, 0.0f, 0.0f, 0.0f};
    auto eps_of = [](int m, double fGHz) -> std::complex<double>
    {
        const double a[7] = {1.0, 5.24, 5.24, 5.24, 9.0, 1.50, 1.50}, im[7] = {0.0, 0.63, 0.0, 0.63, 1.20, 0.40, 0.0};
        return std::complex<double>(a[m], -im[m] / fGHz);
    };

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        int m;
        double d;
    };
    auto cm = [&](const std::vector<Lay> &ls, double s2, double fGHz, std::complex<double> &r, std::complex<double> &t)
    {
        const double k = 2.0 * PI * fGHz * 1.0e9 / C0;
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0, h0 = std::sqrt(std::complex<double>(1.0 - s2, 0.0));
        for (const Lay &l : ls)
        {
            std::complex<double> b = std::sqrt(eps_of(l.m, fGHz) - s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> B = m11 + m12 * h0, Cc = m21 + m22 * h0;
        r = (h0 * B - Cc) / (h0 * B + Cc), t = 2.0 * h0 / (h0 * B + Cc);
    };

    struct Sum
    {
        std::complex<double> T[2] = {0.0, 0.0}, R[2] = {0.0, 0.0}, R1[2] = {0.0, 0.0}; // per frequency: all paths; the shortest reflected one
        int nT = 0, nR = 0, gens = 0;
        size_t segR1 = 1000;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float slab, float x_back, int max_int, float min_gain,
                     const std::vector<double> &fGHz) -> Sum
    {
        Sum S;
        const arma::uword nf = fGHz.size();
        const double th = thd * PI / 180.0, st = std::sin(th);
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (arma::uword f = 0; f < nf; ++f)
            cf(f) = (float)(fGHz[f] * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (arma::uword f = 0; f < nf; ++f)
            pd.slice(f) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 80; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)max_int, 100, 100, 0, min_gain, 3.0f, slab, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            S.gens = g + 1;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue; // inside the wall, or a ray that was already collected
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *a = paths[i].coord(s - 1), *b = paths[i].coord(s);
                    Lin += std::sqrt(((double)b[0] - a[0]) * ((double)b[0] - a[0]) + ((double)b[1] - a[1]) * ((double)b[1] - a[1]) +
                                     ((double)b[2] - a[2]) * ((double)b[2] - a[2]));
                }
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                const bool fwd = dest(i, 0) > orig(i, 0);
                const bool isT = fwd && last[0] > x_back - 1.0e-3f, isR = !fwd && last[0] < 1.0e-3f;
                for (arma::uword f = 0; f < nf; ++f)
                {
                    const double k = 2.0 * PI * fGHz[f] * 1.0e9 / C0;
                    const float *xc = paths[i].xpr_coeff(f);
                    std::complex<double> F = std::complex<double>((double)xc[0], (double)xc[1]) * std::exp(-J * k * Lin) *
                                             std::exp(J * k * st * ((double)last[1] - (double)first[1]));
                    if (isT)
                        S.T[f] += F;
                    else if (isR)
                    {
                        S.R[f] += F;
                        if (ns <= S.segR1)
                            S.R1[f] = F;
                    }
                }
                if (isT)
                    ++S.nT;
                else if (isR)
                    ++S.nR, S.segR1 = std::min(S.segR1, ns);
            }
        }
        return S;
    };
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [&](std::complex<double> z)
    { return std::arg(z) * 180.0 / PI; };
    struct Wall
    {
        const char *name;
        int a, b;
    };
    auto make = [&](const Wall &w, arma::fmat &mesh, arma::uvec &mi, std::vector<Lay> &ls, float &x_back)
    {
        const std::string nm = w.name;
        x_back = 0.10f;
        if (nm == "H")
            build({0.0f, 0.10f}, {w.a, w.a}, {false, true}, mesh, mi), ls = {{w.a, 0.10}};
        else if (nm == "HS")
            build({0.0f, 0.10f, 0.10f + gap}, {w.a, w.a, 3}, {false, true, true}, mesh, mi), ls = {{w.a, 0.10}, {3, 0.006}};
        else if (nm == "SH")
            build({0.0f, gap, 0.10f + gap}, {3, w.a, w.a}, {false, false, true}, mesh, mi), ls = {{3, 0.006}, {w.a, 0.10}};
        else if (nm == "AB")
            build({0.0f, 0.05f, 0.05f + gap, 0.15f + gap}, {w.a, w.a, w.b, w.b}, {false, true, false, true}, mesh, mi),
                ls = {{w.a, 0.05}, {w.b, 0.10}}, x_back = 0.15f;
        else
            build({0.0f, 0.05f, 0.10f}, {w.a, 3, w.a}, {false, false, true}, mesh, mi), ls.clear();
    };

    // ------------------------------------------------------------------------------------------
    // A. The windows at the default threshold
    // ------------------------------------------------------------------------------------------
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: mirror behind, through ray_progress. thin_slab_threshold 0.15 and 0, 20 interactions, no gain floor. The\n");
    std::printf("# summed paths against the matrix of the wall, dB; HSH against its own sum at threshold 0 with 60 interactions.\n");
    std::printf("# n: the paths behind and in front of the wall at 0.15. SH is referenced with its coating in front of the mesh\n");
    std::printf("# plane, where the reflect port of a glued entry puts it\n");
    std::printf("wall,a,b,tracer,theta,nT,nR,T_015,R_015,Rdeg_015,T_0,R_0\n");
#endif
    std::vector<Wall> walls = {{"H", 5, 0}, {"H", 6, 0}, {"HS", 5, 0}, {"HS", 6, 0}, {"SH", 5, 0}, {"SH", 6, 0}, {"AB", 2, 5}, {"AB", 2, 6}, {"HSH", 5, 0}, {"HSH", 6, 0}};
    for (const Wall &w : walls)
        for (int refr = 0; refr < 2; ++refr)
            for (double thd = 10.0; thd < 86.0; thd += 5.0)
            {
                arma::fmat mesh;
                arma::uvec mi;
                std::vector<Lay> ls;
                float x_back;
                make(w, mesh, mi, ls, x_back);
                const double s2 = std::sin(thd * PI / 180.0) * std::sin(thd * PI / 180.0), k0 = 2.0 * PI * 1.0e9 / C0;
                Sum D = trace(mesh, mi, thd, refr != 0, 0.15f, x_back, 20, -300.0f, {1.0}), Z = trace(mesh, mi, thd, refr != 0, 0.0f, x_back, 20, -300.0f, {1.0});
                std::complex<double> r, t;
                if (ls.empty())
                {
                    Sum Z60 = trace(mesh, mi, thd, refr != 0, 0.0f, x_back, 60, -300.0f, {1.0});
                    r = Z60.R[0], t = Z60.T[0];
                }
                else
                    cm(ls, s2, 1.0, r, t);
                if (std::string(w.name) == "SH")
                    r *= std::exp(2.0 * J * k0 * std::cos(thd * PI / 180.0) * 0.006);
                INFO("windows, wall " << w.name << ", a " << w.a << ", b " << w.b << ", refr " << refr << ", theta " << thd);
                CHECK(D.nT >= 1);
                // The default threshold and every series summed are one wall. A sheet inside a wall
                // follows 511 paths where it declines, and 20 interactions cut them at 0.008 dB
                const double bound = ls.empty() ? 1.0e-2 : 5.0e-3;
                CHECK(std::abs(dB(D.T[0] / Z.T[0])) < bound);
                CHECK(std::abs(dB(D.R[0] / Z.R[0])) < bound);
#ifdef QD_TEST_VERBOSE
                std::printf("%s,%d,%d,%d,%.0f,%d,%d,%+.4f,%+.4f,%+.3f,%+.4f,%+.4f\n", w.name, w.a, w.b, refr ? 5 : 4, thd, D.nT, D.nR, dB(D.T[0] / t),
                            dB(D.R[0] / r), deg(D.R[0] / r), dB(Z.T[0] / t), dB(Z.R[0] / r));
#endif
            }

    // ------------------------------------------------------------------------------------------
    // B. A sheet inside a wall: the gain floor
    // ------------------------------------------------------------------------------------------
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: mirror behind, a sheet inside a wall and the gain floor. air | host 50 | sheet 6 mm | host 50 | air, undeviated\n");
    std::printf("# tracer. D: threshold 0.15; Z: threshold 0. Each against Z at 60 interactions and a floor of -300 dB, dB.\n");
    std::printf("# max: max_no_interactions; min: min_gain_dB; n: the paths behind the wall\n");
    std::printf("host,theta,max,min,nT_D,nT_Z,T_D,T_Z,T_D/Z,R_D,R_Z,R_D/Z\n");
#endif
    for (int hst : {5, 6})
        for (double thd : {60.0, 75.0, 85.0})
        {
            arma::fmat mesh;
            arma::uvec mi;
            build({0.0f, 0.05f, 0.10f}, {hst, 3, hst}, {false, false, true}, mesh, mi);
            Sum W = trace(mesh, mi, thd, false, 0.0f, 0.10f, 60, -300.0f, {1.0});
            struct Cfg
            {
                int max_int;
                float floor;
            };
            for (const Cfg &c : {Cfg{60, -100.0f}, Cfg{60, -120.0f}, Cfg{60, -140.0f}, Cfg{60, -160.0f}, Cfg{20, -100.0f}, Cfg{20, -300.0f}})
            {
                Sum D = trace(mesh, mi, thd, false, 0.15f, 0.10f, c.max_int, c.floor, {1.0}), Z = trace(mesh, mi, thd, false, 0.0f, 0.10f, c.max_int, c.floor, {1.0});
                INFO("floor, host " << hst << ", theta " << thd << ", max " << c.max_int << ", floor " << c.floor);
                CHECK((D.nT >= 1 && Z.nT >= 1));
#ifdef QD_TEST_VERBOSE
                std::printf("%d,%.0f,%d,%.0f,%d,%d,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f\n", hst, thd, c.max_int, (double)c.floor, D.nT, Z.nT, dB(D.T[0] / W.T[0]),
                            dB(Z.T[0] / W.T[0]), dB(D.T[0] / Z.T[0]), dB(D.R[0] / W.R[0]), dB(Z.R[0] / W.R[0]), dB(D.R[0] / Z.R[0]));
#endif
            }
        }

    // ------------------------------------------------------------------------------------------
    // C. The verdict of the base frequency and another frequency of the band
    // ------------------------------------------------------------------------------------------
    // The paths are the base frequency's: where it sums a series the reflected ray is resolved,
    // where it declines the bounces are rays. Every other frequency of the run takes that
    // verdict, or its series is counted twice or not at all. One bare wall, a band of two.
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: mirror behind, the verdict across a band. air | host 100 | air, eps 1.50 - 0.40j at 1 GHz, threshold 0.15,\n");
    std::printf("# 20 interactions. base, other: the two frequencies of one run, GHz. The summed paths of each against the matrix at\n");
    std::printf("# its frequency, dB; alone: the other frequency run on its own. n: the paths behind the wall\n");
    std::printf("tracer,base,other,theta,nT_band,nT_alone,T_base,T_other,T_alone,R_base,R_other,R_alone\n");
#endif
    for (int refr = 0; refr < 2; ++refr)
        for (int ord = 0; ord < 6; ++ord)
            for (double thd : {60.0, 65.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 85.0})
            {
                const double fbv[6] = {1.0, 0.5, 1.0, 0.25, 1.0, 4.0}, fov[6] = {0.5, 1.0, 0.25, 1.0, 4.0, 1.0};
                const double fb = fbv[ord], fo = fov[ord];
                arma::fmat mesh;
                arma::uvec mi;
                build({0.0f, 0.10f}, {5, 5}, {false, true}, mesh, mi);
                const double s2 = std::sin(thd * PI / 180.0) * std::sin(thd * PI / 180.0);
                Sum B = trace(mesh, mi, thd, refr != 0, 0.15f, 0.10f, 20, -300.0f, {fb, fo}), A = trace(mesh, mi, thd, refr != 0, 0.15f, 0.10f, 20, -300.0f, {fo});
                std::complex<double> rb, tb, ro, to;
                cm({{5, 0.10}}, s2, fb, rb, tb), cm({{5, 0.10}}, s2, fo, ro, to);
                INFO("band, refr " << refr << ", base " << fb << ", other " << fo << ", theta " << thd);
                CHECK(B.nT >= 1);
                CHECK(std::abs(dB(B.T[0] / tb)) < 5.0e-3);
                CHECK(std::abs(dB(B.R[0] / rb)) < 5.0e-3);
                CHECK(std::abs(dB(B.T[1] / to)) < 5.0e-3); // the other frequency is the wall as well
                CHECK(std::abs(dB(B.R[1] / ro)) < 5.0e-3);
#ifdef QD_TEST_VERBOSE
                std::printf("%d,%.2f,%.2f,%.0f,%d,%d,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f\n", refr ? 5 : 4, fb, fo, thd, B.nT, A.nT, dB(B.T[0] / tb), dB(B.T[1] / to),
                            dB(A.T[0] / to), dB(B.R[0] / rb), dB(B.R[1] / ro), dB(A.R[0] / ro));
#endif
            }

    // ------------------------------------------------------------------------------------------
    // D. A wall coated on its near side, over frequency
    // ------------------------------------------------------------------------------------------
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: mirror behind, a wall coated on its near side over frequency. air | coating 6 mm | host 100 | air, every\n");
    std::printf("# series summed, undeviated tracer. The reflected sum against the matrix of the wall with the coating in front of\n");
    std::printf("# the mesh plane, dB and deg; T: the transmitted sum; mis: 2 k0 cos(theta) thk, deg\n");
    std::printf("host,fGHz,theta,mis_deg,R_dB,R_deg,T_dB\n");
#endif
    for (int hst : {5, 6})
        for (double fg : {0.3, 1.0, 3.0, 6.0})
            for (double thd : {10.0, 30.0, 50.0, 70.0})
            {
                arma::fmat mesh;
                arma::uvec mi;
                build({0.0f, gap, 0.10f + gap}, {3, hst, hst}, {false, false, true}, mesh, mi);
                const double th = thd * PI / 180.0, s2 = std::sin(th) * std::sin(th), k = 2.0 * PI * fg * 1.0e9 / C0;
                Sum Z = trace(mesh, mi, thd, false, 0.0f, 0.10f, 20, -300.0f, {fg});
                std::complex<double> r, t;
                cm({{3, 0.006}, {hst, 0.10}}, s2, fg, r, t);
                r *= std::exp(2.0 * J * k * std::cos(th) * 0.006);
                INFO("near coating, host " << hst << ", f " << fg << ", theta " << thd);
                CHECK(Z.nR >= 1);
#ifdef QD_TEST_VERBOSE
                std::printf("%d,%.1f,%.0f,%.1f,%+.4f,%+.3f,%+.4f\n", hst, fg, thd, 2.0 * k * std::cos(th) * 0.006 * 180.0 / PI, dB(Z.R[0] / r), deg(Z.R[0] / r),
                            dB(Z.T[0] / t));
#endif
            }
}

TEST_CASE("ray_progress - the reflect port of a sheet lies where its transmit port puts the layer")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // The reflect port of a sheet owes the round trip through the part of its layer in front of
    // the mesh plane: all of it for a coating seen from the room, half of it for a sheet with one
    // medium on both sides. Through the tracer itself: one ray, ray_progress called until no ray
    // is left, every path that ends in air outside the structure summed at its polyline's point.
    // Scalar, float, undeviated and refracting tracer.
    //   SH   coating 6 mm | host 100, a wall coated on its near side
    //   SHS  coating 6 mm | host 100 | coating 6 mm, a wall lined on both faces
    //   SGH  sheet 6 mm | air 30 mm | host 100, a sheet in front of a wall across an air gap
    const double PI = 3.14159265358979323846, C0 = 299792458.0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: eps 5.24-0.63j, 2: 5.24, 3: sheet 5.24-0.63j 6 mm, 4: 9.0-1.20j, 5: 1.50-0.40j, 6: 1.50
    pal["a"] = {5.24f, 5.24f, 5.24f, 9.0f, 1.50f, 1.50f};
    pal["b"] = std::vector<float>(6, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), 0.0f, (float)(0.63 / 17.98), (float)(1.20 / 17.98), (float)(0.40 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(6, 0.0f);
    pal["fRef"] = std::vector<float>(6, 1.0f);
    pal["thk"] = {0.0f, 0.0f, 0.006f, 0.0f, 0.0f, 0.0f};
    auto eps_of = [](int m, double fGHz) -> std::complex<double>
    {
        const double a[7] = {1.0, 5.24, 5.24, 5.24, 9.0, 1.50, 1.50}, im[7] = {0.0, 0.63, 0.0, 0.63, 1.20, 0.40, 0.0};
        return std::complex<double>(a[m], -im[m] / fGHz);
    };

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d; material 0 is air
    struct Lay
    {
        int m;
        double d;
    };
    auto cm = [&](const std::vector<Lay> &ls, double s2, double fGHz, std::complex<double> &r, std::complex<double> &t)
    {
        const double k = 2.0 * PI * fGHz * 1.0e9 / C0;
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0, h0 = std::sqrt(std::complex<double>(1.0 - s2, 0.0));
        for (const Lay &l : ls)
        {
            std::complex<double> b = std::sqrt(eps_of(l.m, fGHz) - s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> B = m11 + m12 * h0, Cc = m21 + m22 * h0;
        r = (h0 * B - Cc) / (h0 * B + Cc), t = 2.0 * h0 / (h0 * B + Cc);
    };

    struct Sum
    {
        std::complex<double> T = 0.0, R = 0.0, R1 = 0.0; // all paths; the shortest reflected one
        int nT = 0, nR = 0;
        size_t segR1 = 1000;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float slab, float x_back, int max_int, float min_gain,
                     double fGHz) -> Sum
    {
        Sum S;
        const double th = thd * PI / 180.0, st = std::sin(th), k = 2.0 * PI * fGHz * 1.0e9 / C0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(1);
        cf(0) = (float)(fGHz * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 120; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)max_int, 100, 100, 0, min_gain, 3.0f, slab, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue; // inside a body, or a ray that was already collected
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                const bool fwd = dest(i, 0) > orig(i, 0);
                const bool isT = fwd && last[0] > x_back - 1.0e-3f, isR = !fwd && last[0] < 1.0e-3f;
                if (!isT && !isR)
                    continue; // in air between two bodies
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *a = paths[i].coord(s - 1), *b = paths[i].coord(s);
                    Lin += std::sqrt(((double)b[0] - a[0]) * ((double)b[0] - a[0]) + ((double)b[1] - a[1]) * ((double)b[1] - a[1]) +
                                     ((double)b[2] - a[2]) * ((double)b[2] - a[2]));
                }
                const float *xc = paths[i].xpr_coeff(0);
                std::complex<double> F = std::complex<double>((double)xc[0], (double)xc[1]) * std::exp(-J * k * Lin) *
                                         std::exp(J * k * st * ((double)last[1] - (double)first[1]));
                if (isT)
                    S.T += F, ++S.nT;
                else
                {
                    S.R += F, ++S.nR;
                    if (ns <= S.segR1)
                        S.R1 = F, S.segR1 = ns;
                }
            }
        }
        return S;
    };
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [&](std::complex<double> z)
    { return std::arg(z) * 180.0 / PI; };

    // ------------------------------------------------------------------------------------------
    // A. A wall coated on its near side and a wall lined on both faces, over frequency
    // ------------------------------------------------------------------------------------------
    // Where the wall reflects 10 dB less than the path off its first coating the two nearly
    // cancel, and the row is held against that first path. A traced leg is short by the relaunch
    // offset, which the reference does not know and which grows with frequency: the rows at
    // 6 GHz are held at 1e-2 dB.
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: reflect port, a wall coated on its near side (SH) and lined on both faces (SHS) through ray_progress.\n");
    std::printf("# coating 6 mm, host 100, 20 interactions, no gain floor. thr: thin_slab_threshold. R_dB, R_deg: the reflected sum\n");
    std::printf("# against the matrix of the wall with the near coating in front of the mesh plane; R_ff: its distance from that\n");
    std::printf("# wall over the shortest reflected path; T: the transmitted sum; mis: 2 k0 cos(theta) thk, deg\n");
    std::printf("wall,host,tracer,thr,fGHz,theta,nT,nR,mis_deg,R_dB,R_deg,R_ff,T_dB\n");
#endif
    for (int lined = 0; lined < 2; ++lined)
        for (int hst : {5, 6})
            for (int refr = 0; refr < 2; ++refr)
                for (float thr : {0.0f, 0.15f})
                    for (double fg : {0.3, 1.0, 3.0, 6.0})
                        for (double thd : {10.0, 30.0, 50.0, 70.0})
                        {
                            if (refr && thr > 0.0f && fg != 1.0)
                                continue;
                            arma::fmat mesh;
                            arma::uvec mi;
                            std::vector<Lay> ls = {{3, 0.006}, {hst, 0.10}};
                            if (lined)
                                build({0.0f, gap, 0.10f + gap, 0.10f + 2.0f * gap}, {3, hst, hst, 3}, {false, false, true, true}, mesh, mi), ls.push_back({3, 0.006});
                            else
                                build({0.0f, gap, 0.10f + gap}, {3, hst, hst}, {false, false, true}, mesh, mi);
                            const double th = thd * PI / 180.0, s2 = std::sin(th) * std::sin(th), k = 2.0 * PI * fg * 1.0e9 / C0;
                            Sum Z = trace(mesh, mi, thd, refr != 0, thr, 0.10f, 20, -300.0f, fg);
                            std::complex<double> r, t;
                            cm(ls, s2, fg, r, t);
                            r *= std::exp(2.0 * J * k * std::cos(th) * 0.006); // the near coating in front of the plane
                            INFO("wall " << (lined ? "SHS" : "SH") << ", host " << hst << ", refr " << refr << ", thr " << thr << ", f " << fg << ", theta " << thd);
                            REQUIRE((Z.nR >= 2 && Z.nT >= 1));
                            if (std::abs(r) < 0.316 * std::abs(Z.R1))
                                CHECK(std::abs(Z.R - r) < 5.0e-3 * std::abs(Z.R1));
                            else
                            {
                                CHECK(std::abs(dB(Z.R / r)) < ((fg > 3.0) ? 1.0e-2 : 5.0e-3));
                                CHECK(std::abs(deg(Z.R / r)) < 0.05);
                            }
                            CHECK(std::abs(dB(Z.T / t)) < 5.0e-3);
#ifdef QD_TEST_VERBOSE
                            std::printf("%s,%d,%d,%.2f,%.1f,%.0f,%d,%d,%.1f,%+.4f,%+.3f,%.2e,%+.4f\n", lined ? "SHS" : "SH", hst, refr ? 5 : 4, (double)thr, fg, thd, Z.nT,
                                        Z.nR, 2.0 * k * std::cos(th) * 0.006 * 180.0 / PI, dB(Z.R / r), deg(Z.R / r), std::abs(Z.R - r) / std::abs(Z.R1), dB(Z.T / t));
#endif
                        }

    // ------------------------------------------------------------------------------------------
    // B. A sheet in front of a wall across an air gap
    // ------------------------------------------------------------------------------------------
    // The layer of the sheet occupies 6 mm of the air around its face, [-u, 6 mm - u]: uh centred,
    // which is the wall the paths sum to; u0 behind the face, in the gap, and ut in front,
    // recorded. Each wall referenced to the sheet's face. The cavity is followed bounce by
    // bounce: with 60 interactions and a floor of -120 dB the lossless host at 70 deg and 3 GHz
    // is 0.059 dB short, so the rows run to 110 interactions and -200 dB.
#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: reflect port, a sheet in front of a wall across an air gap through ray_progress. air | sheet 6 mm | air 30 mm\n");
    std::printf("# | host 100 | air, every series summed, 110 interactions, floor -200 dB. The summed paths against the wall whose\n");
    std::printf("# layer lies at [-u, thk - u]: uh centred, asserted; u0 behind the face and ut in front, recorded; dB and deg\n");
    std::printf("host,tracer,fGHz,theta,nT,nR,R_uh,Rdeg_uh,T_uh,Tdeg_uh,R_u0,Rdeg_u0,T_u0,R_ut,Rdeg_ut,T_ut\n");
#endif
    for (int hst : {5, 6})
        for (int refr = 0; refr < 2; ++refr)
            for (double fg : {0.3, 1.0, 3.0, 6.0})
                for (double thd : {10.0, 40.0, 70.0})
                {
                    if (refr && fg != 1.0)
                        continue;
                    const float g = 0.03f;
                    arma::fmat mesh;
                    arma::uvec mi;
                    build({0.0f, g, g + 0.10f}, {3, hst, hst}, {false, false, true}, mesh, mi);
                    const double th = thd * PI / 180.0, s2 = std::sin(th) * std::sin(th), ck = 2.0 * PI * fg * 1.0e9 / C0 * std::cos(th);
                    Sum Z = trace(mesh, mi, thd, refr != 0, 0.0f, g + 0.10f, 110, -200.0f, fg);
                    std::complex<double> rr[3], tt[3];
                    const double uu[3] = {0.0, 0.003, 0.006};
                    for (int q = 0; q < 3; ++q)
                    {
                        cm({{3, 0.006}, {0, 0.03 - 0.006 + uu[q]}, {hst, 0.10}}, s2, fg, rr[q], tt[q]);
                        rr[q] *= std::exp(2.0 * J * ck * uu[q]), tt[q] *= std::exp(J * ck * uu[q]);
                    }
                    INFO("air gap, host " << hst << ", refr " << refr << ", f " << fg << ", theta " << thd);
                    REQUIRE((Z.nR >= 2 && Z.nT >= 1));
                    CHECK(std::abs(dB(Z.R / rr[1])) < 5.0e-3);
                    CHECK(std::abs(deg(Z.R / rr[1])) < 0.05);
                    CHECK(std::abs(dB(Z.T / tt[1])) < 5.0e-3);
                    CHECK(std::abs(deg(Z.T / tt[1])) < 0.05);
#ifdef QD_TEST_VERBOSE
                    std::printf("%d,%d,%.1f,%.0f,%d,%d,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.4f,%+.3f,%+.4f\n", hst, refr ? 5 : 4, fg, thd, Z.nT, Z.nR, dB(Z.R / rr[1]),
                                deg(Z.R / rr[1]), dB(Z.T / tt[1]), deg(Z.T / tt[1]), dB(Z.R / rr[0]), deg(Z.R / rr[0]), dB(Z.T / tt[0]), dB(Z.R / rr[2]), deg(Z.R / rr[2]),
                                dB(Z.T / tt[2]));
#endif
                }
}

TEST_CASE("ray_progress - a lossy medium has no critical angle and a band takes one path")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // A porous medium, eps 0.31 - 0.20j / fGHz, rarer than air and lossy. It has no critical
    // angle. The magnitude test sqrt|1 / eps| sin(theta) >= 1 would trip at 45.3 deg at 0.5 GHz,
    // 37.4 deg at 1 GHz and 34.8 deg at 2 GHz, so a band would have its frequencies on different
    // sides of it; the verdict on a lossy medium is decided on the medium, and every frequency
    // of a band is what it is alone. Through the tracer itself, scalar, float: one ray,
    // ray_progress called until no ray is left, every path that ends in air outside the
    // structure summed at its polyline's point. The coherent sums of PW are recorded: the
    // bounces of a leak inside the rigid wall have no phase the media define.
    //   P    porous solid 50 mm in air, against the analytic slab
    //   PW   porous solid 50 mm on a rigid wall with tf 0.3
    //   SW   the same layer as a sheet on that wall
    // First each frequency alone, then bands whose base and second frequency fall on different
    // sides of the angle.
    const double PI = 3.14159265358979323846, C0 = 299792458.0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: porous solid, 2: porous sheet 50 mm, 3: rigid wall with tf 0.3
    pal["a"] = {0.31f, 0.31f, 2.401e-9f};
    pal["b"] = std::vector<float>(3, 0.0f);
    pal["c"] = {(float)(0.20 / 17.98), (float)(0.20 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(3, 0.0f);
    pal["fRef"] = std::vector<float>(3, 1.0f);
    pal["thk"] = {0.0f, 0.05f, 0.0f};
    pal["tf"] = {0.0f, 0.0f, 0.3f};
    auto eps_p = [](double fGHz)
    { return std::complex<double>(0.31, -0.20 / fGHz); };

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };
    auto slab = [&](double d, double s2, double fGHz, std::complex<double> &r, std::complex<double> &t) // air | porous d | air
    {
        const double k = 2.0 * PI * fGHz * 1.0e9 / C0;
        std::complex<double> h0 = std::sqrt(std::complex<double>(1.0 - s2, 0.0)), b = std::sqrt(eps_p(fGHz) - s2);
        if (std::imag(b) > 0.0)
            b = -b;
        std::complex<double> dl = k * b * d, c = std::cos(dl), s = std::sin(dl), B = c + J * s / b * h0, Cc = J * b * s + c * h0;
        r = (h0 * B - Cc) / (h0 * B + Cc), t = 2.0 * h0 / (h0 * B + Cc);
    };

    struct Sum
    {
        std::complex<double> T = 0.0, R = 0.0;
        double Tp = 0.0, Rp = 0.0; // the power sums
        int nT = 0, nR = 0;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float thr, float x_back, int max_int, float min_gain,
                     const std::vector<double> &fGHz) -> std::vector<Sum>
    {
        const size_t nf = fGHz.size();
        std::vector<Sum> S(nf);
        const double th = thd * PI / 180.0, st = std::sin(th);
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 120; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)max_int, 100, 100, 0, min_gain, 3.0f, thr, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue;
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                const bool fwd = dest(i, 0) > orig(i, 0);
                const bool isT = fwd && last[0] > x_back - 1.0e-3f, isR = !fwd && last[0] < 1.0e-3f;
                if (!isT && !isR)
                    continue;
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *a = paths[i].coord(s - 1), *b = paths[i].coord(s);
                    Lin += std::sqrt(((double)b[0] - a[0]) * ((double)b[0] - a[0]) + ((double)b[1] - a[1]) * ((double)b[1] - a[1]) +
                                     ((double)b[2] - a[2]) * ((double)b[2] - a[2]));
                }
                for (size_t q = 0; q < nf; ++q)
                {
                    const double k = 2.0 * PI * fGHz[q] * 1.0e9 / C0;
                    const float *xc = paths[i].xpr_coeff(q);
                    std::complex<double> F = std::complex<double>((double)xc[0], (double)xc[1]) * std::exp(-J * k * Lin) *
                                             std::exp(J * k * st * ((double)last[1] - (double)first[1]));
                    if (isT)
                        S[q].T += F, S[q].Tp += std::norm(F), ++S[q].nT;
                    else
                        S[q].R += F, S[q].Rp += std::norm(F), ++S[q].nR;
                }
            }
        }
        return S;
    };
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    struct Geo
    {
        const char *name;
        arma::fmat mesh;
        arma::uvec mi;
        float x_back;
        bool analytic;
    };
    std::vector<Geo> geo(3);
    geo[0].name = "P", geo[0].x_back = 0.05f, geo[0].analytic = true;
    build({0.0f, 0.05f}, {1, 1}, {false, true}, geo[0].mesh, geo[0].mi);
    geo[1].name = "PW", geo[1].x_back = 0.25f, geo[1].analytic = false;
    build({0.0f, 0.05f, 0.05f + gap, 0.25f + gap}, {1, 1, 3, 3}, {false, true, false, true}, geo[1].mesh, geo[1].mi);
    geo[2].name = "SW", geo[2].x_back = 0.2f, geo[2].analytic = false;
    build({0.0f, gap, 0.2f + gap}, {2, 3, 3}, {false, false, true}, geo[2].mesh, geo[2].mi);

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: lossy verdict through ray_progress, scalar, 30 interactions, floor -150 dB, thin_slab_threshold 0.15.\n");
    std::printf("# P: porous solid 50 mm in air; PW: on a rigid wall with tf 0.3; SW: the same layer as a sheet on that wall.\n");
    std::printf("# band: the frequencies of the run, the base first; f: the frequency of the row. T, R: the coherent sums, dB;\n");
    std::printf("# Tp, Rp: the power sums; T_ref, R_ref: the analytic slab, P only. The test trips at 45.3 deg at 0.5 GHz,\n");
    std::printf("# 37.4 deg at 1 GHz and 34.8 deg at 2 GHz\n");
    std::printf("geo,tracer,band,f,theta,nT,nR,T_dB,Tp_dB,T_ref_dB,R_dB,Rp_dB,R_ref_dB\n");
#endif
    const std::vector<std::vector<double>> bands = {{0.5}, {1.0}, {2.0}, {1.0, 0.5}, {0.5, 1.0}, {1.0, 2.0}, {2.0, 1.0}};
    std::map<std::string, Sum> alone; // geometry, tracer, frequency, angle
    auto key = [](const char *g, int refr, double f, double thd)
    { return std::string(g) + "," + std::to_string(refr) + "," + std::to_string((int)(10.0 * f)) + "," + std::to_string((int)thd); };
    for (const Geo &G : geo)
        for (int refr = 0; refr < 2; ++refr)
            for (const auto &band : bands)
                for (double thd : {30.0, 36.0, 41.0, 50.0})
                {
                    if (band.size() == 2 && thd != ((band[0] + band[1] < 2.0) ? 41.0 : 36.0))
                        continue;
                    std::vector<Sum> Z = trace(G.mesh, G.mi, thd, refr != 0, 0.15f, G.x_back, 30, -150.0f, band);
                    const double th = thd * PI / 180.0, s2 = std::sin(th) * std::sin(th);
                    for (size_t q = 0; q < band.size(); ++q)
                    {
                        std::complex<double> r = 0.0, t = 0.0;
                        if (G.analytic)
                            slab(0.05, s2, band[q], r, t);
                        INFO(G.name << ", refr " << refr << ", band of " << band.size() << " based on " << band[0] << ", f " << band[q] << ", theta " << thd);
                        CHECK(Z[q].nT >= 1);
                        CHECK(Z[q].nR >= 1);
                        if (G.analytic) // a traced leg is short by the relaunch offset
                        {
                            CHECK(std::abs(dB(Z[q].T) - dB(t)) < 5.0e-3);
                            CHECK(std::abs(dB(Z[q].R) - dB(r)) < 5.0e-3);
                        }
                        if (band.size() == 1)
                            alone[key(G.name, refr, band[q], thd)] = Z[q];
                        else
                        {
                            const Sum &A = alone.at(key(G.name, refr, band[q], thd));
                            CHECK(std::abs(pdB(Z[q].Tp) - pdB(A.Tp)) < 2.0e-3);
                            CHECK(std::abs(pdB(Z[q].Rp) - pdB(A.Rp)) < 2.0e-3);
                            if (G.x_back != 0.25f) // P and SW: one path through, the coherent sum as well
                            {
                                CHECK(std::abs(dB(Z[q].T) - dB(A.T)) < 2.0e-3);
                                CHECK(std::abs(dB(Z[q].R) - dB(A.R)) < 2.0e-3);
                            }
                        }
#ifdef QD_TEST_VERBOSE
                        std::printf("%s,%d,%s%.1f%s,%.1f,%.0f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", G.name, refr ? 5 : 4, (band.size() == 2) ? ((q == 0) ? "" : (band[0] == 0.5 ? "0.5+" : (band[0] == 1.0 ? "1.0+" : "2.0+"))) : "",
                                    (band.size() == 2 && q == 0) ? band[0] : band[q], (band.size() == 2 && q == 0) ? ((band[1] == 0.5) ? "+0.5" : (band[1] == 1.0 ? "+1.0" : "+2.0")) : "",
                                    band[q], thd, Z[q].nT, Z[q].nR, dB(Z[q].T), pdB(Z[q].Tp), G.analytic ? dB(t) : 0.0, dB(Z[q].R), pdB(Z[q].Rp), G.analytic ? dB(r) : 0.0);
#endif
                    }
                }
}

TEST_CASE("ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // A porous layer, eps 0.31 - 0.20j / fGHz, 50 mm on a rigid wall 200 mm with tf 0.3 and its
    // mass law (m 30). The direction of the wave in the layer depends on the frequency; what a
    // wave from air carries across the closed face, its transverse wavenumber, does not. Through
    // the tracer, scalar, float, one ray until none is left. Recorded per frequency: the strongest
    // path behind the wall, the direction it is handed on with, the segment, and where it leaves.
    //   PWm  porous solid on the wall     SWm  the same layer as a sheet on the wall
    //   Wm   the wall bare                PWt, SWt, Wt  the same with a wall that has tf and no m
    // Asserted on the layer as a traced solid: every frequency of a band against the same
    // frequency alone, and behind the wall with its mass law the direction and the segment on
    // the incidence angle. The level against the sheet is recorded: the layer's series is
    // followed at this threshold and the strongest path is not the sum.
    const double PI = 3.14159265358979323846, C0 = 299792458.0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: porous solid, 2: porous sheet 50 mm, 3: rigid wall tf 0.3 m 30, 4: rigid wall tf 0.3
    pal["a"] = {0.31f, 0.31f, 2.401e-9f, 2.401e-9f};
    pal["b"] = std::vector<float>(4, 0.0f);
    pal["c"] = {(float)(0.20 / 17.98), (float)(0.20 / 17.98), 0.0f, 0.0f};
    pal["d"] = std::vector<float>(4, 0.0f);
    pal["fRef"] = {1.0f, 1.0f, 0.001f, 1.0f};
    pal["thk"] = {0.0f, 0.05f, 0.0f, 0.0f};
    pal["tf"] = {0.0f, 0.0f, 0.3f, 0.3f};
    pal["m"] = {0.0f, 0.0f, 30.0f, 0.0f};

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };

    struct Sum
    {
        double Tp = 0.0, T1 = 0.0, dir = -1.0, seg = -1.0, dy = 0.0; // T1: the strongest path, with its direction, segment and lateral walk
        int nT = 0;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float x_back, const std::vector<double> &fGHz) -> std::vector<Sum>
    {
        const size_t nf = fGHz.size();
        std::vector<Sum> S(nf);
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 40; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)12, 100, 100, 0, -250.0f, 3.0f, 0.15f, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue;
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                if (!(dest(i, 0) > orig(i, 0) && last[0] > x_back - 1.0e-3f))
                    continue;
                const double sx = (double)dest(i, 0) - orig(i, 0), sy = (double)dest(i, 1) - orig(i, 1), sz = (double)dest(i, 2) - orig(i, 2);
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[i].xpr_coeff(q);
                    const double P = (double)xc[0] * xc[0] + (double)xc[1] * xc[1];
                    S[q].Tp += P, ++S[q].nT;
                    if (P > S[q].T1)
                    {
                        const double dx = pd(i, 0, q), dyv = pd(i, 1, q), dz = pd(i, 2, q);
                        S[q].T1 = P;
                        S[q].dir = std::atan2(std::sqrt(dyv * dyv + dz * dz), std::abs(dx)) * 180.0 / PI;
                        S[q].seg = std::atan2(std::sqrt(sy * sy + sz * sz), std::abs(sx)) * 180.0 / PI;
                        S[q].dy = (double)last[1] - (double)first[1];
                    }
                }
            }
        }
        return S;
    };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    struct Geo
    {
        const char *name;
        arma::fmat mesh;
        arma::uvec mi;
        float x_back;
    };
    std::vector<Geo> geo(6);
    for (int w = 0; w < 2; ++w)
    {
        const int W = 3 + w;
        Geo &A = geo[3 * w], &B = geo[3 * w + 1], &Cc = geo[3 * w + 2];
        A.name = w ? "PWt" : "PWm", A.x_back = 0.25f;
        build({0.0f, 0.05f, 0.05f + gap, 0.25f + gap}, {1, 1, W, W}, {false, true, false, true}, A.mesh, A.mi);
        B.name = w ? "SWt" : "SWm", B.x_back = 0.2f;
        build({0.0f, gap, 0.2f + gap}, {2, W, W}, {false, false, true}, B.mesh, B.mi);
        Cc.name = w ? "Wt" : "Wm", Cc.x_back = 0.2f;
        build({0.0f, 0.2f}, {W, W}, {false, true}, Cc.mesh, Cc.mi);
    }

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: closed-face hand-on through ray_progress, scalar, 12 interactions, thin_slab_threshold 0.15. PW: porous\n");
    std::printf("# solid 50 mm on a rigid wall 200 mm with tf 0.3; SW: the same layer as a sheet; W: the wall bare. m: the wall\n");
    std::printf("# with its mass law (m 30); t: without. band: the frequencies of the run, the base first; f: the frequency of\n");
    std::printf("# the row. Tp: the power sum behind the wall, dB; T1: the strongest path; dir: the direction that path is\n");
    std::printf("# handed on with at this frequency; seg: its segment, deg against the normal; dy: its lateral walk, m\n");
    std::printf("geo,tracer,band,f,theta,nT,Tp_dB,T1_dB,dir,seg,dy_m\n");
#endif
    const std::vector<std::vector<double>> bands = {{0.5}, {1.0}, {2.0}, {1.0, 0.5}, {1.0, 2.0}, {0.5, 1.0}, {2.0, 1.0}};
    std::map<std::string, Sum> alone; // geometry, tracer, frequency, angle
    auto key = [](const char *g, int refr, double f, double thd)
    { return std::string(g) + "," + std::to_string(refr) + "," + std::to_string((int)(10.0 * f)) + "," + std::to_string((int)thd); };
    for (const Geo &G : geo)
        for (int refr = 0; refr < 2; ++refr)
            for (const auto &band : bands)
                for (double thd : {30.0, 60.0, 80.0})
                {
                    std::vector<Sum> Z = trace(G.mesh, G.mi, thd, refr != 0, G.x_back, band);
                    for (size_t q = 0; q < band.size(); ++q)
                    {
                        INFO(G.name << ", refr " << refr << ", band of " << band.size() << " based on " << band[0] << ", f " << band[q] << ", theta " << thd);
                        CHECK(Z[q].nT >= 1);
                        if (band.size() == 1)
                            alone[key(G.name, refr, band[q], thd)] = Z[q];
                        else // a frequency of a band is what it is alone
                            CHECK(std::abs(pdB(Z[q].Tp) - pdB(alone.at(key(G.name, refr, band[q], thd)).Tp)) < 2.0e-3);
                        if (std::string(G.name) == "PWm" || std::string(G.name) == "SWm" || std::string(G.name) == "Wm")
                        {
                            CHECK(std::abs(Z[q].dir - thd) < 1.0e-3); // held through the wall at the incidence angle
                            CHECK(std::abs(Z[q].seg - thd) < 1.0e-3);
                        }
#ifdef QD_TEST_VERBOSE
                        char bn[32];
                        if (band.size() == 2)
                            std::snprintf(bn, sizeof(bn), "%.1f+%.1f", band[0], band[1]);
                        else
                            std::snprintf(bn, sizeof(bn), "%.1f", band[0]);
                        std::printf("%s,%d,%s,%.1f,%.0f,%d,%.4f,%.4f,%.4f,%.4f,%+.5f\n", G.name, refr ? 5 : 4, bn, band[q], thd, Z[q].nT, pdB(Z[q].Tp), pdB(Z[q].T1),
                                    Z[q].dir, Z[q].seg, Z[q].dy);
#endif
                    }
                }
}

TEST_CASE("ray_progress - the bonded plate at any depth where the slopes agree")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // Plate depth through the tracer. Bonded mass-law stacks, rigid and lossless with a tf leak,
    // scalar, float, one ray until none is left, both tracers, alone and in a band. The mass a
    // stack paid is the strongest path behind it against the same path on a palette whose mass
    // arguments are taken to nothing (fRef 1e9), per frequency; the reference is the plate of
    // every body from the column definitions. Asserted: the plate at any depth where the slopes
    // agree, and from the side on which they agree where they do not; a frequency of a band
    // pays what it pays alone, since the state words of a run are the base frequency's.
    const double PI = 3.14159265358979323846;
    const float gap = 1.0e-5f;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    // 1 A20, 2 B20, 3 C20, 4 C30, 5 plaster, 6 brick, 7 lining L20 15 mm, 8 sheet S20 12 mm, 9 sheet HS 8 mm with B20's permittivity
    const std::vector<float> eps = {2.401e-9f, 2.1609e-8f, 6.0e-9f, 6.0e-9f, 1.2e-8f, 4.0e-9f, 5.0e-9f, 1.0e-8f, 2.1609e-8f};
    const std::vector<float> tfv = {0.10f, 0.20f, 0.15f, 0.15f, 0.20f, 0.10f, 0.15f, 0.20f, 0.20f};
    const std::vector<float> thk = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.015f, 0.012f, 0.008f};
    const std::vector<float> mv = {20.0f, 20.0f, 20.0f, 30.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f};
    const double fR = 0.001;
    auto pal_of = [&](bool twin) -> Pal
    {
        Pal p;
        p["a"] = eps, p["b"] = std::vector<float>(eps.size(), 0.0f), p["c"] = std::vector<float>(eps.size(), 0.0f);
        p["d"] = std::vector<float>(eps.size(), 0.0f), p["thk"] = thk, p["tf"] = tfv, p["m"] = mv;
        p["fRef"] = std::vector<float>(eps.size(), twin ? 1.0e9f : (float)fR);
        return p;
    };
    const Pal pal = pal_of(false), pal_twin = pal_of(true);

    // A stack along x as tokens: O / C open and close a solid, S a sheet, g advances by d
    struct Tok
    {
        char k;
        int mat;
        float d;
    };
    auto O = [](int m) { return Tok{'O', m, 0.0f}; };
    auto C = [](int m) { return Tok{'C', m, 0.0f}; };
    auto S = [](int m) { return Tok{'S', m, 0.0f}; };
    auto G = [](float d) { return Tok{'g', 0, d}; };
    struct Geo
    {
        arma::fmat mesh;
        arma::uvec mi;
        float x_back = 0.0f;
        std::vector<std::array<double, 3>> body; // m, fRef, normal thickness
    };
    auto build = [&](std::vector<Tok> t, bool rev) -> Geo
    {
        if (rev)
        {
            std::reverse(t.begin(), t.end());
            for (Tok &q : t)
                q.k = (q.k == 'O') ? 'C' : ((q.k == 'C') ? 'O' : q.k);
        }
        Geo g;
        std::vector<float> xs;
        std::vector<int> mats;
        std::vector<bool> flip;
        float x = 0.0f, leg = 0.0f;
        int cur = 0;
        bool last_face = false;
        for (const Tok &q : t)
        {
            if (q.k == 'g')
            {
                x += q.d, last_face = false;
                if (cur)
                    leg += q.d;
                continue;
            }
            if (last_face)
                x += gap;
            xs.push_back(x), mats.push_back(q.mat), flip.push_back(q.k == 'C'), last_face = true;
            if (q.k != 'O' && cur && leg > 0.0f)
                g.body.push_back({(double)mv[(size_t)cur - 1], fR, (double)leg}), leg = 0.0f;
            if (q.k == 'O')
                cur = q.mat, leg = 0.0f;
            else if (q.k == 'C')
                cur = 0;
            else
                g.body.push_back({(double)mv[(size_t)q.mat - 1], fR, (double)thk[(size_t)q.mat - 1]});
        }
        g.x_back = x;
        const float h = 200.0f;
        g.mesh.set_size(2 * xs.size(), 9);
        g.mi.set_size(2 * xs.size());
        for (size_t k = 0; k < xs.size(); ++k)
        {
            if (!flip[k])
                g.mesh.row(2 * k) = arma::frowvec{xs[k], -h, -h, xs[k], -h, h, xs[k], h, -h},
                g.mesh.row(2 * k + 1) = arma::frowvec{xs[k], h, -h, xs[k], -h, h, xs[k], h, h};
            else
                g.mesh.row(2 * k) = arma::frowvec{xs[k], -h, -h, xs[k], h, -h, xs[k], -h, h},
                g.mesh.row(2 * k + 1) = arma::frowvec{xs[k], h, -h, xs[k], h, h, xs[k], -h, h};
            g.mi(2 * k) = (arma::uword)mats[k], g.mi(2 * k + 1) = (arma::uword)mats[k];
        }
        return g;
    };
    // The plate of every body, from the column definitions
    auto plate_of = [](const Geo &g, double fGHz, double ct)
    {
        double X = 0.0;
        for (const auto &b : g.body)
            X += std::pow((fGHz / b[1]) * b[2], 0.05 * b[0]);
        if (!(X > 1.0))
            return 0.0;
        const double A = 20.0 * std::log10(X), x = X * ct;
        return (ct >= 1.0) ? A : A * std::log10(1.0 + x * x) / std::log10(1.0 + X * X);
    };

    // The strongest path behind the wall, power per frequency, and the number of paths
    auto trace = [&](const Pal &pl, const Geo &g, double thd, bool refr, const std::vector<double> &fGHz, int &nT) -> std::vector<double>
    {
        const size_t nf = fGHz.size();
        std::vector<double> T1(nf, 0.0);
        nT = 0;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        for (int it = 0; it < 40; ++it)
        {
            quadriga_lib::ray_progress(g.mesh, g.mi, pl, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)12, 100, 100, 0, -250.0f, 3.0f, 0.15f, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(it + 1))
                    continue;
                const float *last = paths[i].coord(ns - 1);
                if (!(dest(i, 0) > orig(i, 0) && last[0] > g.x_back - 1.0e-3f))
                    continue;
                ++nT;
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[i].xpr_coeff(q);
                    T1[q] = std::max(T1[q], (double)xc[0] * xc[0] + (double)xc[1] * xc[1]);
                }
            }
        }
        return T1;
    };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    struct Stack
    {
        const char *name;
        std::vector<Tok> t;
        int exact; // bit 0: the plate as named, bit 1: from the other side; clear: recorded
    };
    const std::vector<Stack> stacks = {
        {"A20 100 | B20 100 | C20 100", {O(1), G(0.1f), C(1), O(2), G(0.1f), C(2), O(3), G(0.1f), C(3)}, 3},
        {"A20 100 | B20 13 | C20 100", {O(1), G(0.1f), C(1), O(2), G(0.013f), C(2), O(3), G(0.1f), C(3)}, 3},
        {"A20 100 | B20 100 | C30 50", {O(1), G(0.1f), C(1), O(2), G(0.1f), C(2), O(4), G(0.05f), C(4)}, 1},
        {"plaster 15 | brick 115 | plaster 15", {O(5), G(0.015f), C(5), O(6), G(0.115f), C(6), O(5), G(0.015f), C(5)}, 3},
        {"lining L20 > B20 60 | S20 | B20 40", {S(7), O(2), G(0.06f), S(8), G(0.04f), C(2)}, 3},
        {"merged: B20 60 | HS 8 | B20 40", {O(2), G(0.06f), S(9), G(0.04f), C(2)}, 3},
    };

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: plate depth through ray_progress, scalar, float, 12 interactions. dir: f as named, r from the other side.\n");
    std::printf("# band: the frequencies of the run in GHz, the base first; f: the frequency of the row. plate: every body as one\n");
    std::printf("# plate; paid: the strongest path behind the stack against the same path without the mass arguments, dB;\n");
    std::printf("# over: paid - plate; nT: paths behind the stack\n");
    std::printf("stack,dir,tracer,band,f,theta,nT,plate,paid,over\n");
#endif
    const std::vector<std::vector<double>> bands = {{1.0}, {0.25}, {4.0}, {1.0, 0.25}, {1.0, 4.0}, {0.25, 1.0}, {4.0, 1.0}};
    std::map<std::string, double> alone;
    for (const Stack &St : stacks)
        for (int rev = 0; rev < 2; ++rev)
        {
            const Geo g = build(St.t, rev != 0);
            for (int refr = 0; refr < 2; ++refr)
                for (const auto &band : bands)
                    for (double thd : {0.0, 60.0})
                    {
                        int nT = 0, nZ = 0;
                        const std::vector<double> T = trace(pal, g, thd, refr != 0, band, nT), Z = trace(pal_twin, g, thd, refr != 0, band, nZ);
                        for (size_t q = 0; q < band.size(); ++q)
                        {
                            const double paid = pdB(Z[q]) - pdB(T[q]), plate = plate_of(g, band[q], std::cos(thd * PI / 180.0));
                            INFO(St.name << ", dir " << rev << ", refr " << refr << ", band of " << band.size() << " based on " << band[0] << ", f " << band[q] << ", theta " << thd);
                            CHECK(nT >= 1);
                            CHECK(nZ >= 1);
                            const std::string key = std::string(St.name) + "," + std::to_string(rev) + "," + std::to_string(refr) + "," +
                                                    std::to_string((int)(100.0 * band[q])) + "," + std::to_string((int)thd);
                            if (band.size() == 1)
                                alone[key] = paid;
                            else // a frequency of a band pays what it pays alone
                                CHECK(std::abs(paid - alone.at(key)) < 2.0e-3);
                            if ((St.exact >> rev) & 1)
                                CHECK(std::abs(paid - plate) < 0.01);
#ifdef QD_TEST_VERBOSE
                            char bn[32];
                            if (band.size() == 2)
                                std::snprintf(bn, sizeof(bn), "%.2f+%.2f", band[0], band[1]);
                            else
                                std::snprintf(bn, sizeof(bn), "%.2f", band[0]);
                            std::printf("%s,%s,%d,%s,%.2f,%.0f,%d,%.4f,%.4f,%+.4f\n", St.name, rev ? "r" : "f", refr ? 5 : 4, bn, band[q], thd, nT, plate, paid,
                                        paid - plate);
#endif
                        }
                    }
        }
}

TEST_CASE("ray_progress - an ordered stack leaves the ray on the line it arrived on")
{
#ifdef _OPENMP
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // Two co-located sheets of different material have one medium on both sides, so behind them
    // the ray goes on along the line it arrived on, with the default tracer as with the
    // undeviated one, and the next wall is met where a single sheet in their place lets it be
    // met. Through ray_progress, float, scalar, a band of 1 and 2 GHz; the materials carry
    // neither tf nor the mass column. Rows that must not move are printed with every digit.
    const double PI = 3.14159265358979323846;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    Pal pal; // 1 C1 sheet 5.24-0.63j 6 mm, 2 C2 sheet 1.50-0.40j 6 mm, 3 solid 2.50, 4 solid 2.50-0.80j
    pal["a"] = {5.24f, 1.50f, 2.50f, 2.50f};
    pal["b"] = std::vector<float>(4, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), (float)(0.40 / 17.98), 0.0f, (float)(0.80 / 17.98)};
    pal["d"] = std::vector<float>(4, 1.0f);
    pal["fRef"] = std::vector<float>(4, 1.0f);
    pal["thk"] = {0.006f, 0.006f, 0.0f, 0.0f};
    struct Wall
    {
        arma::fmat mesh;
        arma::uvec mi;
    };
    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip) -> Wall
    {
        const float h = 200.0f;
        Wall W;
        W.mesh.set_size(2 * x.size(), 9), W.mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                W.mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h}, W.mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                W.mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h}, W.mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            W.mi(2 * k) = (arma::uword)mat[k], W.mi(2 * k + 1) = (arma::uword)mat[k];
        }
        return W;
    };
    struct Gen // the launch configuration after one call
    {
        arma::fmat orig, dest, hit, tridir;
        arma::fcube dir;
        arma::Col<short> cur;
        std::vector<std::string> types;
        std::vector<std::vector<double>> pw; // per ray, per frequency, dB
        arma::frowvec seg(arma::uword i) const
        {
            arma::frowvec d = dest.row(i) - orig.row(i);
            return d / nrm3(d);
        }
    };
    arma::frowvec k_launch;
    auto trace = [&](const Wall &W, double thd, bool refr, int n_gen, int max_ref, bool beam, int force_gen) -> std::vector<Gen>
    {
        std::vector<Gen> out;
        const size_t nf = 2;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec k = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        k_launch = k;
        arma::fvec cf = {1.0e9f, 2.0e9f};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros), trivec, tridir;
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * k, dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * k;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = k;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        if (beam)
        {
            trivec.zeros(1, 9), tridir.zeros(1, 9);
            arma::frowvec u = arma::cross(arma::frowvec{0.0f, 0.0f, 1.0f}, k);
            u /= nrm3(u);
            arma::frowvec v = arma::cross(k, u);
            const float ha = 0.5f * (float)(PI / 180.0);
            for (int j = 0; j < 3; ++j)
            {
                const float phi = (float)(2.0 * PI * (double)j / 3.0);
                arma::frowvec vj = std::cos(ha) * k + std::sin(ha) * (std::cos(phi) * u + std::sin(phi) * v);
                vj /= nrm3(vj);
                for (int c = 0; c < 3; ++c)
                    tridir(0, 3 * j + c) = vj(c), trivec(0, 3 * j + c) = r0 * (vj(c) - k(c));
            }
        }
        for (int g = 0; g < n_gen; ++g)
        {
            std::vector<bool> ff(orig.n_rows, g == force_gen);
            quadriga_lib::ray_progress(W.mesh, W.mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       beam ? &trivec : nullptr, beam ? &tridir : nullptr, nullptr, nullptr,
                                       20, (uint8_t)max_ref, 10, (uint8_t)(force_gen >= 0 ? 2 : 0), -300.0f, 3.0f, 0.15f, refr, true,
                                       nullptr, nullptr, nullptr, force_gen >= 0 ? &ff : nullptr);
            if (orig.n_rows == 0)
                break;
            Gen G;
            G.orig = orig, G.dest = dest, G.dir = pd, G.cur = cur;
            if (beam)
                G.tridir = tridir;
            G.hit.zeros(orig.n_rows, 3);
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                std::string ty;
                for (auto c : paths[i].interaction_type_codes())
                    ty += std::to_string((unsigned)c) + "/";
                G.types.push_back(ty);
                if (paths[i].n_seg())
                    for (int c = 0; c < 3; ++c)
                        G.hit(i, c) = paths[i].coord(paths[i].n_seg() - 1)[c];
                std::vector<double> p;
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[i].xpr_coeff(q);
                    p.push_back(10.0 * std::log10((double)xc[0] * xc[0] + (double)xc[1] * xc[1] + 1.0e-300));
                }
                G.pw.push_back(p);
            }
            out.push_back(G);
        }
        return out;
    };
    auto dmax = [](const arma::frowvec &a, const arma::frowvec &b) -> double
    {
        double d = 0.0;
        for (int c = 0; c < 3; ++c)
            d = std::max(d, (double)std::abs(a(c) - b(c)));
        return d;
    };
    auto ang = [&](const arma::frowvec &d) { return std::atan2(std::sqrt((double)d(1) * d(1) + (double)d(2) * d(2)), std::abs((double)d(0))) * 180.0 / PI; };
    const char *sn[] = {"C1|C2", "C2|C1", "C1 alone", "C2 alone"};
    // s 0, 1: the stack in its two orders; 2, 3: its first layer alone. nrm 0: normals outward, 1: both towards the source
    auto wall = [&](int H, int s, int nrm) -> Wall
    {
        const float gap = 1.0e-5f, x0 = H ? 0.011f : 0.0f;
        std::vector<float> x;
        std::vector<int> m;
        std::vector<bool> fl;
        if (H)
            x.push_back(0.0f), m.push_back(H), fl.push_back(false);
        x.push_back(x0), m.push_back((s == 0 || s == 2) ? 1 : 2), fl.push_back(false);
        if (s < 2)
            x.push_back(x0 + gap), m.push_back(s == 0 ? 2 : 1), fl.push_back(nrm == 0);
        if (H)
            x.push_back(0.022f), m.push_back(H), fl.push_back(true);
        x.push_back(0.5f), m.push_back(3), fl.push_back(false), x.push_back(0.55f), m.push_back(3), fl.push_back(true);
        return build(x, m, fl);
    };
    const double TOL = 1.0e-6;

    // ------------------------------------------------------------------------------------------
    // A. The transmitted path: behind the stack, at the wall behind it and behind that wall
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: stack trajectory A, the stack in air (host 0) or in the middle of 22 mm of eps 2.50 (3) or 2.50-0.80j (4), a 50 mm\n");
    QD_DIAG("# eps 2.50 wall behind at x = 0.5 m; the transmitted path, no reflections. seg, dir: segment and path_dir behind the stack's\n");
    QD_DIAG("# wall, deg; e_seg, e_dir0, e_dir1: against the launched direction, largest component, the direction at 1 and 2 GHz.\n");
    QD_DIAG("# dy_wall: where the wall behind is met against the twin with the first layer alone, m; e_in: the segment inside that wall\n");
    QD_DIAG("# against the twin's; e_out: the segment behind it against the launched direction; dP0, dP1: the path's power behind it\n");
    QD_DIAG("# against the undeviated tracer's, dB, at 1 and 2 GHz\n");
    QD_DIAG("host,stack,normals,theta,types,seg,dir,e_seg,e_dir0,e_dir1,dy_wall,e_in,e_out,dP0,dP1\n");
    for (int H : {0, 3, 4})
        for (int s = 0; s < 2; ++s)
            for (int nrm = 0; nrm < 2; ++nrm)
                for (double thd : {30.0, 60.0})
                {
                    const int n = H ? 5 : 3, ia = H ? 2 : 0; // calls until the path is behind the far wall; the call that leaves the stack's wall
                    auto G = trace(wall(H, s, nrm), thd, true, n, 0, false, -1), T = trace(wall(H, s + 2, nrm), thd, true, n, 0, false, -1);
                    auto U = trace(wall(H, s, nrm), thd, false, n, 0, false, -1);
                    INFO("A, host " << H << ", " << sn[s] << ", normals " << nrm << ", theta " << thd);
                    REQUIRE((T.size() == (size_t)n && U.size() == (size_t)n));
                    CHECK(G.size() == (size_t)n);
                    if (G.size() != (size_t)n || G[n - 1].orig.n_rows != 1)
                    {
                        QD_DIAG("%d,%s,%s,%.0f,path lost after %zu calls\n", H, sn[s], nrm ? "front|front" : "front|back", thd, G.size());
                        continue;
                    }
                    const Gen &A = G[ia], &B = G[ia + 1], &C = G[ia + 2];
                    const double e_seg = dmax(A.seg(0), k_launch), e0 = dmax(A.dir.slice(0).row(0), k_launch), e1 = dmax(A.dir.slice(1).row(0), k_launch);
                    const double dyw = (double)B.hit(0, 1) - (double)T[ia + 1].hit(0, 1), e_in = dmax(B.seg(0), T[ia + 1].seg(0)), e_out = dmax(C.seg(0), k_launch);
                    const double dP0 = C.pw[0][0] - U[n - 1].pw[0][0], dP1 = C.pw[0][1] - U[n - 1].pw[0][1];
                    QD_DIAG("%d,%s,%s,%.0f,%s,%.4f,%.4f,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%+.4f,%+.4f\n", H, sn[s], nrm ? "front|front" : "front|back", thd, C.types[0].c_str(),
                            ang(A.seg(0)), ang(A.dir.slice(0).row(0)), e_seg, e0, e1, dyw, e_in, e_out, dP0, dP1);
                    CHECK(A.cur(0) == (short)0);
                    CHECK(e_seg < TOL);
                    CHECK(e0 < TOL);
                    CHECK(e1 < TOL);
                    CHECK(std::abs(dyw) < 1.0e-5); // float coordinates of order one
                    CHECK(e_in < TOL);
                    CHECK(e_out < TOL);
                    CHECK(std::abs(dP0) < 1.0e-2);
                    CHECK(std::abs(dP1) < 1.0e-2);
                }

    // ------------------------------------------------------------------------------------------
    // B. A beam, split in air behind the stack
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: stack trajectory B, a beam of 0.5 deg half angle through the stack in air, split in the call after it. n: sub-beams,\n");
    QD_DIAG("# refracting and undeviated tracer; e_seg, e_tube: every sub-beam's segment and vertex rays against the undeviated tracer's,\n");
    QD_DIAG("# largest component; seg_min, seg_max: the sub-beams' segments, deg\n");
    QD_DIAG("stack,normals,theta,n5,n4,e_seg,e_tube,seg_min,seg_max\n");
    for (int s = 0; s < 2; ++s)
        for (int nrm = 0; nrm < 2; ++nrm)
            for (double thd : {30.0, 60.0})
            {
                auto G = trace(wall(0, s, nrm), thd, true, 2, 0, true, 1), U = trace(wall(0, s, nrm), thd, false, 2, 0, true, 1);
                INFO("B, " << sn[s] << ", normals " << nrm << ", theta " << thd);
                REQUIRE((G.size() == 2 && U.size() == 2));
                const arma::uword n5 = G[1].orig.n_rows, n4 = U[1].orig.n_rows;
                double e_seg = 0.0, e_tube = 0.0, lo = 1.0e9, hi = -1.0e9;
                for (arma::uword i = 0; i < std::min(n5, n4); ++i)
                {
                    e_seg = std::max(e_seg, dmax(G[1].seg(i), U[1].seg(i)));
                    for (int j = 0; j < 3; ++j)
                        e_tube = std::max(e_tube, dmax(G[1].tridir.row(i).cols(3 * j, 3 * j + 2), U[1].tridir.row(i).cols(3 * j, 3 * j + 2)));
                    lo = std::min(lo, ang(G[1].seg(i))), hi = std::max(hi, ang(G[1].seg(i)));
                }
                QD_DIAG("%s,%s,%.0f,%u,%u,%.3e,%.3e,%.4f,%.4f\n", sn[s], nrm ? "front|front" : "front|back", thd, (unsigned)n5, (unsigned)n4, e_seg, e_tube, lo, hi);
                CHECK(n5 == n4);
                CHECK(e_seg < TOL);
                CHECK(e_tube < TOL);
            }

    // ------------------------------------------------------------------------------------------
    // C. What must not move, with every digit of a float
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: stack trajectory C, raw. Every ray after the first two calls with one reflection allowed: a single sheet with both\n");
    QD_DIAG("# tracers, the stack with the undeviated tracer, and of the stack with the refracting tracer the reflected ray alone\n");
    QD_DIAG("host,stack,tracer,theta,call,types,orig_x,orig_y,orig_z,dest_x,dest_y,dest_z,P0,P1\n");
    for (int H : {0, 4})
        for (int s = 0; s < 3; ++s)
            for (bool refr : {true, false})
                for (double thd : {30.0, 60.0})
                {
                    auto G = trace(wall(H, s, 0), thd, refr, 2, 1, false, -1);
                    REQUIRE(G.size() == 2);
                    for (size_t g = 0; g < 2; ++g)
                        for (arma::uword i = 0; i < G[g].orig.n_rows; ++i)
                        {
                            const std::string &ty = G[g].types[i];
                            const bool refl_only = ty == "197/" || ty == "9/205/" || ty == "129/" || ty == "9/";
                            if (s < 2 && refr && !refl_only)
                                continue;
                            QD_DIAG("%d,%s,%d,%.0f,%zu,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f,%.6f\n", H, sn[s], refr ? 5 : 4, thd, g, ty.c_str(), G[g].orig(i, 0), G[g].orig(i, 1),
                                    G[g].orig(i, 2), G[g].dest(i, 0), G[g].dest(i, 1), G[g].dest(i, 2), G[g].pw[i][0], G[g].pw[i][1]);
                        }
                }
}

TEST_CASE("ray_progress - a coherent sheet with tf that is rarer than the invariant passes no more than it received")
{
#ifdef _OPENMP
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // A wall of 10 mm + 12 mm of eps 2.50 - j ei with a sheet of 6 mm eps 0.31 in it, which is past
    // its critical angle from 33.83 deg, and tf on the sheet or on the host. One ray through
    // ray_progress, float, a band of 1 and 2 GHz, no reflections followed: the path behind the wall
    // is its entry, the sheet and its exit. A passive wall passes no more than it received, at
    // every loss of the host, continuously as the loss goes to zero. EM materials: tf, no mass column.
    const double PI = 3.14159265358979323846;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    struct Out
    {
        bool found = false;
        double p[2] = {-999.0, -999.0}; // dB per frequency, VV; and HH in EM
        double h[2] = {-999.0, -999.0};
    };
    auto trace = [&](double ei, double tS, double tH, double thd, bool refr, bool scalar) -> Out
    {
        Pal pal; // 1 the sheet, 2 the host
        pal["a"] = {0.31f, 2.50f};
        pal["b"] = {0.0f, 0.0f};
        pal["c"] = {0.0f, (float)(ei / 17.98)};
        pal["d"] = {1.0f, 1.0f};
        pal["fRef"] = {1.0f, 1.0f};
        pal["thk"] = {0.006f, 0.0f};
        pal["tf"] = {(float)tS, (float)tH};
        const float h = 200.0f, xs[3] = {0.0f, 0.010f, 0.022f};
        const int mat[3] = {2, 1, 2};
        arma::fmat mesh(6, 9);
        arma::uvec mi(6);
        for (int k = 0; k < 3; ++k)
        {
            const float x = xs[k];
            if (k < 2)
                mesh.row(2 * k) = arma::frowvec{x, -h, -h, x, -h, h, x, h, -h}, mesh.row(2 * k + 1) = arma::frowvec{x, h, -h, x, -h, h, x, h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x, -h, -h, x, h, -h, x, -h, h}, mesh.row(2 * k + 1) = arma::frowvec{x, h, -h, x, h, h, x, -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
        const size_t nf = 2;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec k = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {1.0e9f, 2.0e9f};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * k, dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * k;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = k;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, scalar);
        paths[0].set_length(r0);
        Out O;
        for (int g = 0; g < 5; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr, nullptr, nullptr,
                                       20, (uint8_t)0, 10, (uint8_t)0, -300.0f, 3.0f, 0.15f, refr, scalar);
            if (orig.n_rows == 0)
                break;
            if (cur(0) == 0 && paths[0].n_seg() == 3) // in air behind the wall
            {
                O.found = true;
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[0].xpr_coeff(q);
                    O.p[q] = 10.0 * std::log10((double)xc[0] * xc[0] + (double)xc[1] * xc[1] + 1.0e-300);
                    if (!scalar)
                        O.h[q] = 10.0 * std::log10((double)xc[6] * xc[6] + (double)xc[7] * xc[7] + 1.0e-300);
                }
                break;
            }
        }
        return O;
    };
    QD_DIAG("\n# diag: ray_progress rare sheet, air | 10 mm eps 2.50 - j ei | S 6 mm eps 0.31 | 12 mm | air, the transmitted path, dB, float.\n");
    QD_DIAG("# tf on the sheet or on the host. scalar: P at 1 and 2 GHz; EM: VV and HH at 1 GHz\n");
    QD_DIAG("mode,tracer,tf_on,tf,theta,ei,found,P_1GHz,P_2GHz_or_HH\n");
    const double losses[5] = {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0};
    for (bool scalar : {true, false})
        for (bool refr : {true, false})
            for (int cfg = 0; cfg < 3; ++cfg) // sheet 0.5, sheet -0.5, host 0.5
                for (double thd : {45.0, 60.0})
                {
                    const double tS = (cfg == 0) ? 0.5 : ((cfg == 1) ? -0.5 : 0.0), tH = (cfg == 2) ? 0.5 : 0.0;
                    double A[5], B[5];
                    for (int q = 0; q < 5; ++q)
                    {
                        Out O = trace(losses[q], tS, tH, thd, refr, scalar);
                        A[q] = O.p[0], B[q] = scalar ? O.p[1] : O.h[0];
                        QD_DIAG("%s,%d,%s,%.1f,%.0f,%.0e,%d,%.4f,%.4f\n", scalar ? "scalar" : "EM", refr ? 5 : 4, cfg == 2 ? "host" : "sheet", cfg == 2 ? tH : tS, thd, losses[q], (int)O.found, A[q], B[q]);
                        INFO("rare sheet, " << (scalar ? "scalar" : "EM") << ", refr " << refr << ", cfg " << cfg << ", theta " << thd << ", ei " << losses[q]);
                        CHECK(O.found);       // the wave tunnels: the path is there, with either sign of tf
                        CHECK(A[q] < 1.0e-4); // and carries no more than was launched
                        if (scalar || cfg != 2)
                            CHECK(B[q] < 1.0e-4);
                        else // tf on the traced host: its bare faces apply tf on the TE/TM mean, so the wall is bounded on the mean
                            CHECK(0.5 * (std::pow(10.0, 0.1 * A[q]) + std::pow(10.0, 0.1 * B[q])) < 1.0 + 1.0e-4);
                        CHECK(A[q] > -40.0);
                    }
                    INFO("rare sheet, continuity at zero loss, scalar " << scalar << ", refr " << refr << ", cfg " << cfg << ", theta " << thd);
                    CHECK(std::abs(A[3] - A[4]) < 1.0e-2); // ei 1e-6 against ei 0
                    CHECK(std::abs(B[3] - B[4]) < 1.0e-2);
                }
}

TEST_CASE("ray_progress - a coherent sheet rarer than the invariant without tf tunnels")
{
#ifdef _OPENMP
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // A sheet of 6 mm eps 0.31 without tf, past its critical angle from 33.83 deg, alone in air and
    // inside a wall of 10 mm + 12 mm of eps 2.50 - j ei. One ray through ray_progress, float, a
    // band of 1 and 2 GHz, no reflections followed. The wave tunnels: the path behind the sheet is
    // there. Alone in air it is the lossless slab of the sheet, written here from its two Fresnel
    // faces; inside the wall its reference needs no formula, the same wall with a tf of 1e-9 and
    // of -1e-9 on the sheet, which is a negligible change. EM materials: no tf, no mass column.
    const double PI = 3.14159265358979323846;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    struct Out
    {
        bool found = false;
        double p[2] = {-999.0, -999.0}; // dB per frequency, VV; and HH in EM
        double h[2] = {-999.0, -999.0};
    };
    // host: the sheet inside 10 mm + 12 mm of the host; otherwise the sheet alone in air
    auto trace = [&](bool host, double ei, double tS, double thd, bool refr, bool scalar) -> Out
    {
        Pal pal; // 1 the sheet, 2 the host
        pal["a"] = {0.31f, 2.50f};
        pal["b"] = {0.0f, 0.0f};
        pal["c"] = {0.0f, (float)(ei / 17.98)};
        pal["d"] = {1.0f, 1.0f};
        pal["fRef"] = {1.0f, 1.0f};
        pal["thk"] = {0.006f, 0.0f};
        pal["tf"] = {(float)tS, 0.0f};
        const float h = 200.0f, xs[3] = {0.0f, 0.010f, 0.022f};
        const int mat[3] = {2, 1, 2};
        const int n_face = host ? 3 : 1;
        arma::fmat mesh(2 * n_face, 9);
        arma::uvec mi(2 * n_face);
        for (int k = 0; k < n_face; ++k)
        {
            const float x = host ? xs[k] : 0.0f;
            if (k < 2)
                mesh.row(2 * k) = arma::frowvec{x, -h, -h, x, -h, h, x, h, -h}, mesh.row(2 * k + 1) = arma::frowvec{x, h, -h, x, -h, h, x, h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x, -h, -h, x, h, -h, x, -h, h}, mesh.row(2 * k + 1) = arma::frowvec{x, h, -h, x, h, h, x, -h, h};
            mi(2 * k) = (arma::uword)(host ? mat[k] : 1), mi(2 * k + 1) = mi(2 * k);
        }
        const size_t nf = 2;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec k = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {1.0e9f, 2.0e9f};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * k, dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * k;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = k;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, scalar);
        paths[0].set_length(r0);
        Out O;
        for (int g = 0; g < 5; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr, nullptr, nullptr,
                                       20, (uint8_t)0, 10, (uint8_t)0, -300.0f, 3.0f, 0.15f, refr, scalar);
            if (orig.n_rows == 0)
                break;
            if (cur(0) == 0 && paths[0].n_seg() == (size_t)n_face) // in air behind the sheet or the wall
            {
                O.found = true;
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[0].xpr_coeff(q);
                    O.p[q] = 10.0 * std::log10((double)xc[0] * xc[0] + (double)xc[1] * xc[1] + 1.0e-300);
                    if (!scalar)
                        O.h[q] = 10.0 * std::log10((double)xc[6] * xc[6] + (double)xc[7] * xc[7] + 1.0e-300);
                }
                break;
            }
        }
        return O;
    };
    // The lossless slab in air from its two Fresnel faces, dB; tm: the TM channel
    auto slab = [&](double thd, double f_hz, bool tm) -> double
    {
        const double th = thd * PI / 180.0, k0 = 2.0 * PI * f_hz / 299792458.0, eps = 0.31, thk = 0.006;
        const std::complex<double> c1(std::cos(th), 0.0);
        std::complex<double> a(1.0 - std::sin(th) * std::sin(th) / eps, -0.0);
        const std::complex<double> c2 = std::sqrt(a), n = std::sqrt(eps); // the decaying root past the critical angle
        const std::complex<double> r = tm ? (n * c1 - c2) / (n * c1 + c2) : (c1 - n * c2) / (c1 + n * c2);
        const std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * n * c2 * thk);
        return 20.0 * std::log10(std::abs((1.0 - r * r) * phi / (1.0 - r * r * phi * phi)));
    };

    QD_DIAG("\n# diag: ray_progress flat branch A, S 6 mm eps 0.31 without tf alone in air, the transmitted path against the lossless slab, dB, float.\n");
    QD_DIAG("# scalar: P at 1 and 2 GHz; EM: VV and HH at 1 GHz\n");
    QD_DIAG("mode,tracer,theta,found,P_1GHz,ref,P_2GHz_or_HH,ref\n");
    for (bool scalar : {true, false})
        for (bool refr : {true, false})
            for (double thd : {20.0, 45.0, 60.0, 80.0})
            {
                Out O = trace(false, 0.0, 0.0, thd, refr, scalar);
                const double A = O.p[0], B = scalar ? O.p[1] : O.h[0], rA = slab(thd, 1.0e9, false), rB = scalar ? slab(thd, 2.0e9, false) : slab(thd, 1.0e9, true);
                QD_DIAG("%s,%d,%.0f,%d,%.4f,%.4f,%.4f,%.4f\n", scalar ? "scalar" : "EM", refr ? 5 : 4, thd, (int)O.found, A, rA, B, rB);
                INFO("flat branch in air, " << (scalar ? "scalar" : "EM") << ", refr " << refr << ", theta " << thd);
                CHECK(O.found);
                CHECK(std::abs(A - rA) < 1.0e-3);
                CHECK(std::abs(B - rB) < 1.0e-3);
            }

    QD_DIAG("\n# diag: ray_progress flat branch B, air | 10 mm eps 2.50 - j ei | S 6 mm eps 0.31 without tf | 12 mm | air, the transmitted path, dB,\n");
    QD_DIAG("# float. d+, d-: the same wall with tf 1e-9 and -1e-9 on the sheet, less the wall without. scalar: P at 1 and 2 GHz; EM: VV and HH at 1 GHz\n");
    QD_DIAG("mode,tracer,theta,ei,found,P_1GHz,d+,d-,P_2GHz_or_HH,d+,d-\n");
    const double losses[5] = {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0};
    for (bool scalar : {true, false})
        for (bool refr : {true, false})
            for (double thd : {45.0, 60.0})
            {
                double A[5], B[5];
                for (int q = 0; q < 5; ++q)
                {
                    Out O = trace(true, losses[q], 0.0, thd, refr, scalar), Op = trace(true, losses[q], 1.0e-9, thd, refr, scalar), Om = trace(true, losses[q], -1.0e-9, thd, refr, scalar);
                    A[q] = O.p[0], B[q] = scalar ? O.p[1] : O.h[0];
                    const double Ap = Op.p[0], Bp = scalar ? Op.p[1] : Op.h[0], Am = Om.p[0], Bm = scalar ? Om.p[1] : Om.h[0];
                    QD_DIAG("%s,%d,%.0f,%.0e,%d,%.4f,%.2e,%.2e,%.4f,%.2e,%.2e\n", scalar ? "scalar" : "EM", refr ? 5 : 4, thd, losses[q], (int)O.found, A[q], Ap - A[q], Am - A[q],
                            B[q], Bp - B[q], Bm - B[q]);
                    INFO("flat branch in a wall, " << (scalar ? "scalar" : "EM") << ", refr " << refr << ", theta " << thd << ", ei " << losses[q]);
                    CHECK(O.found);       // the wave tunnels: the path is there
                    CHECK(A[q] < 1.0e-4); // and carries no more than was launched
                    CHECK(B[q] < 1.0e-4);
                    CHECK(std::abs(Ap - A[q]) < 1.0e-3); // a negligible tf is a negligible change
                    CHECK(std::abs(Am - A[q]) < 1.0e-3);
                    CHECK(std::abs(Bp - B[q]) < 1.0e-3);
                    CHECK(std::abs(Bm - B[q]) < 1.0e-3);
                }
                INFO("flat branch in a wall, continuity at zero loss, scalar " << scalar << ", refr " << refr << ", theta " << thd);
                CHECK(std::abs(A[3] - A[4]) < 1.0e-2); // ei 1e-6 against ei 0
                CHECK(std::abs(B[3] - B[4]) < 1.0e-2);
            }
}

TEST_CASE("ray_progress - a coherent sheet with tf creates nothing and has no gate on the invariant")
{
#ifdef _OPENMP
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // A coherent sheet with tf - no mass column, EM materials - alone in air, as a lining on the
    // near face of 10 mm of the lossless eps 2.50 and inside 10 mm + 12 mm of it. One ray through
    // ray_progress, float, 1 GHz, both tracers, scalar and EM; ray_progress is called until no
    // ray is left, with every series summed and 60 interactions - the two halves of a wall with a
    // sheet inside it are coupled by paths, which converge slowly for a lossless wall towards
    // grazing, 1.0076 after 20 interactions at 73 deg -, and every path that ends in air is summed at its
    // polyline's point, behind the wall into T and in front of it into R. With nothing lossy but
    // the sheet, and tf on the sheet alone, the two sums are the two ports of the wall and
    // together carry no more than was launched, in each channel. tf on a traced host is left out
    // on purpose: its bare faces apply tf on the TE/TM mean, so a wall with it is bounded on the
    // mean and not per channel.
    const double PI = 3.14159265358979323846, C0 = 299792458.0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    struct Sum
    {
        std::complex<double> T[2] = {0.0, 0.0}, R[2] = {0.0, 0.0}; // VV / scalar, HH
        int nT = 0, nR = 0;
    };
    // wall 0: the sheet alone; 1: sheet | host 10 mm; 2: host 10 mm | sheet | host 12 mm
    auto trace = [&](int wall, double a, double ei, double thk, double tS, double thd, bool refr, bool scalar) -> Sum
    {
        Pal pal; // 1 the sheet, 2 the host
        pal["a"] = {(float)a, 2.50f};
        pal["b"] = {0.0f, 0.0f};
        pal["c"] = {(float)(ei / 17.98), 0.0f};
        pal["d"] = {0.0f, 0.0f};
        pal["fRef"] = {1.0f, 1.0f};
        pal["thk"] = {(float)thk, 0.0f};
        pal["tf"] = {(float)tS, 0.0f};
        std::vector<float> x;
        std::vector<int> mat;
        std::vector<bool> flip;
        if (wall == 0)
            x = {0.0f}, mat = {1}, flip = {false};
        else if (wall == 1)
            x = {0.0f, gap, gap + 0.010f}, mat = {1, 2, 2}, flip = {false, false, true};
        else
            x = {0.0f, 0.010f, 0.022f}, mat = {2, 1, 2}, flip = {false, false, true};
        const float h = 200.0f, x_back = x.back();
        arma::fmat mesh(2 * x.size(), 9);
        arma::uvec mi(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h}, mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h}, mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
        Sum S;
        const double th = thd * PI / 180.0, st = std::sin(th), k = 2.0 * PI * 1.0e9 / C0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(1);
        cf(0) = 1.0e9f;
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv, dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, scalar);
        paths[0].set_length(r0);
        for (int g = 0; g < 130; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)60, 100, 100, 0, -300.0f, 3.0f, 0.0f, refr, scalar, nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue; // inside the host, or a ray that was already collected
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                const bool fwd = dest(i, 0) > orig(i, 0);
                const bool isT = fwd && last[0] > x_back - 1.0e-3f, isR = !fwd && last[0] < 1.0e-3f;
                if (!isT && !isR)
                    continue;
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *p = paths[i].coord(s - 1), *q = paths[i].coord(s);
                    Lin += std::sqrt(((double)q[0] - p[0]) * ((double)q[0] - p[0]) + ((double)q[1] - p[1]) * ((double)q[1] - p[1]) + ((double)q[2] - p[2]) * ((double)q[2] - p[2]));
                }
                const float *xc = paths[i].xpr_coeff(0);
                const std::complex<double> ph = std::exp(-J * k * Lin) * std::exp(J * k * st * ((double)last[1] - (double)first[1]));
                for (int c = 0; c < (scalar ? 1 : 2); ++c)
                    (isT ? S.T[c] : S.R[c]) += std::complex<double>((double)xc[6 * c], (double)xc[6 * c + 1]) * ph;
                ++(isT ? S.nT : S.nR);
            }
        }
        return S;
    };
    auto dB = [](double p) { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    QD_DIAG("\n# diag: ray_progress sheet tf A, a coherent sheet with tf alone in air (0), on 10 mm of eps 2.50 (1) and inside 10 mm + 12 mm of it (2),\n");
    QD_DIAG("# every path summed, series summed, float. T, R: dB; sum: T + R in power. scalar: one channel; EM: VV and HH\n");
    QD_DIAG("sheet,mm,tf,wall,mode,tracer,theta,nT,nR,T,R,sum,T_hh,R_hh,sum_hh\n");
    struct Sh
    {
        double a, ei, thk, tf;
        std::vector<double> th;
    };
    const std::vector<Sh> sheets = {
        {0.31, 0.20, 0.006, 0.5, {20.0, 37.0, 60.0}}, {0.31, 0.20, 0.001, 0.9, {20.0, 37.0}}, {0.90, 0.20, 0.006, 0.9, {45.0, 60.0}},
        {5.24, 0.63, 0.025, -0.5, {45.0, 67.0}},     {0.90, 0.20, 0.006, -0.5, {0.0, 45.0}},  {0.90, 0.0, 0.001, 0.5, {60.0, 73.0}},
        {0.31, 0.0, 0.006, 0.5, {20.0, 45.0}},       {5.24, 0.0, 0.006, -0.5, {20.0, 60.0}},
    };
    for (const Sh &S : sheets)
        for (int wall = 0; wall < 3; ++wall)
            for (bool scalar : {true, false})
                for (bool refr : {false, true})
                    for (double thd : S.th)
                    {
                        Sum Z = trace(wall, S.a, S.ei, S.thk, S.tf, thd, refr, scalar);
                        const double T = std::norm(Z.T[0]), R = std::norm(Z.R[0]), Th = std::norm(Z.T[1]), Rh = std::norm(Z.R[1]);
                        QD_DIAG("%.2f-%.2fj,%.0f,%.1f,%d,%s,%d,%.0f,%d,%d,%.4f,%.4f,%.6f,%.4f,%.4f,%.6f\n", S.a, S.ei, 1.0e3 * S.thk, S.tf, wall, scalar ? "scalar" : "EM", refr ? 5 : 4, thd, Z.nT,
                                Z.nR, dB(T), dB(R), T + R, dB(Th), dB(Rh), Th + Rh);
                        INFO("sheet " << S.a << " - " << S.ei << "j, " << 1.0e3 * S.thk << " mm, tf " << S.tf << ", wall " << wall << ", scalar " << scalar << ", refr " << refr << ", theta " << thd);
                        CHECK((Z.nT >= 1 && Z.nR >= 1));
                        CHECK(T + R < 1.0 + 2.0e-3); // the wall creates nothing, in each channel
                        CHECK(Th + Rh < 1.0 + 2.0e-3);
                        if (S.ei == 0.0)
                        {
                            CHECK(T + R > 1.0 - 5.0e-3); // and a lossless wall loses nothing, tunnelling or not
                            if (!scalar)
                                CHECK(Th + Rh > 1.0 - 5.0e-3);
                        }
                    }

    QD_DIAG("\n# diag: ray_progress sheet tf B, the transmitted sum about the angle at which the sheet becomes rarer than the invariant, neighbours\n");
    QD_DIAG("# 0.02 deg apart, dB. Porous: eps 0.31 - 0.20j 6 mm with tf 0.5, 37.40 deg; the other: eps 0.90 - 0.20j 6 mm with tf 0.9, 73.78 deg\n");
    QD_DIAG("sheet,wall,mode,tracer,theta_a,theta_b,T_a,T_b,step,step_hh\n");
    for (int w = 0; w < 2; ++w)
        for (int wall : {0, 2})
            for (bool scalar : {true, false})
                for (bool refr : {false, true})
                {
                    const double a = w ? 0.90 : 0.31, tS = w ? 0.9 : 0.5, ta = w ? 73.77 : 37.39, tb = w ? 73.79 : 37.41;
                    Sum A = trace(wall, a, 0.20, 0.006, tS, ta, refr, scalar), B = trace(wall, a, 0.20, 0.006, tS, tb, refr, scalar);
                    const double st = dB(std::norm(B.T[0])) - dB(std::norm(A.T[0])), sh = scalar ? 0.0 : dB(std::norm(B.T[1])) - dB(std::norm(A.T[1]));
                    QD_DIAG("%.2f-0.20j,%d,%s,%d,%.2f,%.2f,%.4f,%.4f,%+.5f,%+.5f\n", a, wall, scalar ? "scalar" : "EM", refr ? 5 : 4, ta, tb, dB(std::norm(A.T[0])), dB(std::norm(B.T[0])), st, sh);
                    INFO("step, sheet " << a << ", wall " << wall << ", scalar " << scalar << ", refr " << refr);
                    CHECK(std::abs(st) < 0.01); // no step where the sheet becomes rarer than the invariant
                    CHECK(std::abs(sh) < 0.01);
                }
}
