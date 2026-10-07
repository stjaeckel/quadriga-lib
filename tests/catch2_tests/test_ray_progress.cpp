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
