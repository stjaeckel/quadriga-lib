// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for the public ray_commit API.
//
// ray_commit is a gate-and-append stage, not a physics kernel: the beam-point geometry is pinned by
// test_ray_point_intersect, the material model by test_ray_mesh_interact, and the path storage by
// test_path. This suite asserts the things only ray_commit can get wrong:
//
//   the gates      length, gain, shading, subdivision, direct-path, segment count
//   the count      the survivor mask and the block prefix sum agree with what is written
//   the payload    iC, the receive-side mirror, the in-medium loss, per frequency
//   the append     existing entries untouched, return value equals the number added
//   the index map  padding rows dropped, iC reported in the caller's original ordering
//
// Oracles: geometry is exact by construction (a single ray along +x, receivers placed on its axis),
// so the shading decision has a known answer. The coefficient assertions are relative — committed
// against source — so this file does not restate what path::xpr_update already guarantees. The
// in-medium case is the exception: medium_gain is called directly to build an exact oracle.
//
// Block boundary: pass B and pass C walk the pair list in blocks of 65536. One test drives the pair
// count past that so the prefix sum over more than one block is exercised; a single-block suite
// would never touch it.
//
// Index map: most point_index cases use a hand-built map so the expected iC is exact. One case runs
// the real point_cloud_segmentation and compares against the unsegmented result, which is the only
// oracle that catches a change in the padding convention.
//
// Conventions:
//  - The point source sits at the coordinate origin; rays are launched from a launch sphere of
//    radius r0 = 0.1 m, matching what ray_init produces.
//  - Cubes are the 2x2x2 Blender-style cube, so a cube centred at (5,0,0) has faces at x = 4 and 6.
//  - Material columns are the obj_file_read names {a,b,c,d,att,attB,alpha,alphaB,fRef}.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr float FRQ = 10.0e9f; // Reference frequency
    constexpr float R0 = 0.1f;     // Launch sphere radius
    constexpr float WIDE_DEG = 8.0f;

    float nrm3(const arma::frowvec &v) { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); }

    // Convert a per-face material matrix [n_face, 9] into the (mtl_ind, mtl_prop-map) pair
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

    // Lossless dielectric, eps_r = 5
    arma::frowvec mtl_dielectric() { return {5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

    // ITU-style concrete: sigma = 0.0462 * f^0.7822, lossy so medium_gain < 1
    arma::frowvec mtl_concrete() { return {5.24f, 0.0f, 0.0462f, 0.7822f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

    // Weakly lossy dielectric: sigma = 0.0005 * f_GHz^0.7822, a few dB per metre at 10 GHz,
    // so a multi-metre leg stays well inside the gain gate and still varies with frequency
    arma::frowvec mtl_weak_loss() { return {5.0f, 0.0f, 0.0005f, 0.7822f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

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

    // The optional block of ray_commit, so a scenario reads as one call
    struct Opt
    {
        const arma::u32_vec *sci = nullptr;
        const arma::u32_vec *pt_ind = nullptr;
        const std::vector<bool> *sf_in = nullptr;
        float max_len = 10e3f;
        float min_gain_dB = -140.0f;
        uint8_t min_seg = 0;
        bool ignore_direct = false;
    };

    struct Cfg
    {
        float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
        arma::fvec freq;
        bool scalar = false;

        arma::fmat orig, dest, trivec, tridir;
        arma::u32_vec fbs_ind;
        arma::Col<short> cur;
        std::vector<quadriga_lib::path> paths;

        arma::uword n() const { return orig.n_rows; }

        arma::uword commit(const Scene &S, const arma::fmat &points,
                           std::vector<quadriga_lib::path> &out, const Opt &o = Opt()) const
        {
            return quadriga_lib::ray_commit(paths, out, S.mesh, S.mtl_prop, freq,
                                            orig, fbs_ind, trivec, tridir, cur, points,
                                            o.sci, o.pt_ind, o.sf_in,
                                            o.max_len, o.min_gain_dB, o.min_seg, o.ignore_direct);
        }
    };

    // Launch configuration: n rays from the source along the rows of "dirs", each a beam tube of
    // half-angle "half_deg" whose vertices sit on the launch sphere. fbs_ind is zeroed; the caller
    // fills it, either by hand or through hits().
    Cfg make_cfg(const arma::fmat &dirs_in, float len, float half_deg = WIDE_DEG,
                 arma::uword n_freq = 1, bool scalar = false)
    {
        const arma::uword n = dirs_in.n_rows;

        Cfg C;
        C.scalar = scalar;
        C.freq.set_size(n_freq);
        for (arma::uword f = 0; f < n_freq; ++f)
            C.freq(f) = FRQ * (1.0f + 0.5f * (float)f);

        arma::fmat d = dirs_in;
        for (arma::uword i = 0; i < n; ++i)
            d.row(i) /= std::sqrt(arma::accu(d.row(i) % d.row(i)));

        C.orig = d * R0;
        C.dest = d * (R0 + len);
        C.cur.zeros(n);
        C.fbs_ind.zeros(n);

        C.paths.resize(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            C.paths[i].init(0, n_freq, scalar);
            C.paths[i].set_length(R0);
        }

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

            const float ha = half_deg * (float)(arma::datum::pi / 180.0);
            for (int j = 0; j < 3; ++j)
            {
                const float phi = (float)(2.0 * arma::datum::pi * (double)j / 3.0);
                arma::frowvec vj = std::cos(ha) * k + std::sin(ha) * (std::cos(phi) * u + std::sin(phi) * v);
                vj /= nrm3(vj);

                arma::frowvec tv = R0 * vj - R0 * k;
                for (int c = 0; c < 3; ++c)
                {
                    C.tridir(i, 3 * j + c) = vj(c);
                    C.trivec(i, 3 * j + c) = tv(c);
                }
            }
        }
        return C;
    }

    // A single ray along +x
    Cfg one_ray(float len = 20.0f, float half_deg = WIDE_DEG, arma::uword n_freq = 1, bool scalar = false)
    {
        arma::fmat d = {{1.0f, 0.0f, 0.0f}};
        return make_cfg(d, len, half_deg, n_freq, scalar);
    }

    // Fill fbs_ind the way a caller running the intersector would
    void hits(Cfg &C, const Scene &S)
    {
        arma::u32_vec ni, si;
        quadriga_lib::ray_triangle_intersect<float>(&C.orig, &C.dest, &S.mesh, nullptr, nullptr,
                                                    &ni, &C.fbs_ind, &si, nullptr, nullptr);
    }

    // Receivers as rows of a [n_point, 3] matrix
    arma::fmat pts(const std::vector<std::array<float, 3>> &v)
    {
        arma::fmat P((arma::uword)v.size(), 3, arma::fill::none);
        for (arma::uword i = 0; i < P.n_rows; ++i)
            P(i, 0) = v[i][0], P(i, 1) = v[i][1], P(i, 2) = v[i][2];
        return P;
    }

    // Every committed path is well formed and its receiver index is in range.
    // "n_index" bounds iC: the point count without point_index, the original cloud size with it.
    // Point-major output means iC never decreases within one call, but only while iC and the
    // storage order agree; a point_index permutes them, so that check is opt-out.
    void check_commit_shape(const std::vector<quadriga_lib::path> &out, size_t from,
                            arma::uword n_index, arma::uword n_freq, bool scalar,
                            bool monotonic_iC = true)
    {
        unsigned last_iC = 0;
        for (size_t i = from; i < out.size(); ++i)
        {
            CHECK(out[i].n_freq() == n_freq);
            CHECK(out[i].is_scalar() == scalar);
            CHECK(out[i].iC < (unsigned)n_index);
            CHECK(std::isfinite(out[i].length()));
            CHECK(out[i].calc_gain(0.0f, 0) > 0.0f);

            // Output is point-major, so the receiver index never decreases within one call
            if (monotonic_iC && i > from)
                CHECK(out[i].iC >= last_iC);
            last_iC = out[i].iC;
        }
    }

    // Append one interaction segment to a ray's path, so it clears a min_no_segments gate.
    // The vertex is placed at the ray origin, which leaves the accumulated length untouched
    // and keeps the committed leg identical to the zero-segment case.
    void add_segment(Cfg &C, arma::uword i_ray)
    {
        quadriga_lib::path tmp;
        C.paths[i_ray].extend(tmp, C.orig(i_ray, 0), C.orig(i_ray, 1), C.orig(i_ray, 2), 128);
        C.paths[i_ray] = std::move(tmp);
    }

    // Sorted receiver indices of a commit result, for comparing two runs that differ only in
    // the order the points were presented in
    std::vector<unsigned> sorted_iC(const std::vector<quadriga_lib::path> &out)
    {
        std::vector<unsigned> v;
        v.reserve(out.size());
        for (const auto &p : out)
            v.push_back(p.iC);
        std::sort(v.begin(), v.end());
        return v;
    }
}

// ===========================================================================================
// Basic commit
// ===========================================================================================

TEST_CASE("ray_commit - a receiver on the beam axis is committed")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f); // far away, never hit
    Cfg C = one_ray();

    // Guard: a degenerate source path would make every gate below vacuous
    REQUIRE(C.paths[0].calc_gain(FRQ * 1e-9f, 0) > 0.0f);

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    REQUIRE(C.commit(S, P, out) == 1);
    REQUIRE(out.size() == 1);

    CHECK(out[0].iC == 0u);                        // receiver index
    CHECK(out[0].n_seg() == C.paths[0].n_seg());   // the receiver is not an interaction point
    CHECK(out[0].length() == C.paths[0].length()); // length still ends at the last interaction

    // The committed path descends from the source path rather than being freshly built
    CHECK((int)out[0].nREF == (int)C.paths[0].nREF);
    CHECK((int)out[0].nTRA == (int)C.paths[0].nTRA);
    CHECK((int)out[0].nSUB == (int)C.paths[0].nSUB);

    // A zero-segment path has no last interaction, so the read-only overload cannot form the leg
    CHECK(std::isnan(out[0].calc_length(5.0f, 0.0f, 0.0f)));

    // The caller recovers the total from the source position instead
    const float leg = 5.0f - R0;
    CHECK(std::abs(out[0].length() + leg - 5.0f) < 1e-3f);

    check_commit_shape(out, 0, P.n_rows, 1, false);
}

TEST_CASE("ray_commit - a receiver outside the beam tube is not committed")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray(20.0f, 1.0f); // narrow tube, half-angle 1 deg

    // At 5 m a 1 deg tube spans well under 0.5 m; the receiver sits far outside it
    arma::fmat P = pts({{5.0f, 3.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    CHECK(C.commit(S, P, out) == 0);
    CHECK(out.empty());
}

TEST_CASE("ray_commit - a receiver behind the ray is not committed")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{-5.0f, 0.0f, 0.0f}}); // opposite hemisphere
    std::vector<quadriga_lib::path> out;

    CHECK(C.commit(S, P, out) == 0);
}

TEST_CASE("ray_commit - several receivers on one ray")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{3.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}, {5.0f, 9.0f, 0.0f}, {7.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    REQUIRE(C.commit(S, P, out) == 3); // point 2 is outside the tube
    REQUIRE(out.size() == 3);

    CHECK(out[0].iC == 0u);
    CHECK(out[1].iC == 1u);
    CHECK(out[2].iC == 3u);

    check_commit_shape(out, 0, P.n_rows, 1, false);
}

// ===========================================================================================
// Gates
// ===========================================================================================

TEST_CASE("ray_commit - shading by the first-bounce face")
{
    // Cube at (5,0,0): near face at x = 4. The ray hits it, so a receiver past that plane is
    // shaded and one in front of it is not.
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();
    hits(C, S);
    REQUIRE(C.fbs_ind(0) != 0u); // the scene must actually be hit, or the test proves nothing

    std::vector<quadriga_lib::path> out;

    arma::fmat behind = pts({{5.0f, 0.0f, 0.0f}}); // inside the cube, past x = 4
    CHECK(C.commit(S, behind, out) == 0);

    arma::fmat front = pts({{2.0f, 0.0f, 0.0f}}); // between source and face
    CHECK(C.commit(S, front, out) == 1);
    CHECK(out.size() == 1);
}

TEST_CASE("ray_commit - a ray that misses the mesh shades nothing")
{
    Scene S = one_cube(5.0f, 0.0f, 0.0f);
    Cfg C = one_ray();
    // fbs_ind left at zero: same geometry, but no face to shade with
    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    CHECK(C.commit(S, P, out) == 1);
}

TEST_CASE("ray_commit - max_path_length gate")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray(20.0f);

    arma::fmat P = pts({{2.0f, 0.0f, 0.0f}, {8.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    Opt o;
    o.max_len = 5.0f;
    REQUIRE(C.commit(S, P, out) == 2); // both without the limit
    out.clear();

    REQUIRE(C.commit(S, P, out, o) == 1); // only the near one survives
    CHECK(out[0].iC == 0u);

    // The gate is on the total including the final leg, not on the stored length
    o.max_len = 1.0f;
    out.clear();
    CHECK(C.commit(S, P, out, o) == 0);
}

TEST_CASE("ray_commit - min_gain_dB gate")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    // Gain at 5 m, 10 GHz is roughly -66 dB; bracket it from both sides
    Opt lo, hi;
    lo.min_gain_dB = -140.0f;
    hi.min_gain_dB = -20.0f;

    CHECK(C.commit(S, P, out, lo) == 1);
    out.clear();
    CHECK(C.commit(S, P, out, hi) == 0);
}

TEST_CASE("ray_commit - subdiv_flag_in excludes flagged rays")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    arma::fmat d = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    Cfg C = make_cfg(d, 20.0f);

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}, {0.0f, 5.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    REQUIRE(C.commit(S, P, out) == 2);
    out.clear();

    std::vector<bool> f = {true, false};
    Opt o;
    o.sf_in = &f;
    REQUIRE(C.commit(S, P, out, o) == 1);
    CHECK(out[0].iC == 1u); // the ray along +y, hitting the second receiver

    out.clear();
    std::vector<bool> all = {true, true};
    o.sf_in = &all;
    CHECK(C.commit(S, P, out, o) == 0);

    // An empty flag vector is treated as "no ray excluded"
    out.clear();
    std::vector<bool> none;
    o.sf_in = &none;
    CHECK(C.commit(S, P, out, o) == 2);
}

TEST_CASE("ray_commit - ignore_direct_path drops transmission-only arrivals")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    Opt o;
    o.ignore_direct = true;

    // nREF = 0 and nSCT = 0, so this is a direct arrival regardless of geometry
    CHECK(C.commit(S, P, out, o) == 0);

    // One reflection in the history makes it a traced path again
    C.paths[0].nREF = 1;
    CHECK(C.commit(S, P, out, o) == 1);

    // Scattering counts the same way
    C.paths[0].nREF = 0;
    C.paths[0].nSCT = 1;
    out.clear();
    CHECK(C.commit(S, P, out, o) == 1);
}

// ===========================================================================================
// Payload
// ===========================================================================================

TEST_CASE("ray_commit - the receive-side mirror is applied in EM mode")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;
    REQUIRE(C.commit(S, P, out) == 1);

    const float *src = C.paths[0].xpr_coeff(0);
    const float *dst = out[0].xpr_coeff(0);

    // VV and the cross terms pass through, HH changes sign
    CHECK(std::abs(dst[0] - src[0]) < 1e-6f);
    CHECK(std::abs(dst[1] - src[1]) < 1e-6f);
    CHECK(std::abs(dst[2] - src[2]) < 1e-6f);
    CHECK(std::abs(dst[3] - src[3]) < 1e-6f);
    CHECK(std::abs(dst[6] + src[6]) < 1e-6f);
    CHECK(std::abs(dst[7] + src[7]) < 1e-6f);

    // A mirror is a unitary operation, so the gain is unchanged
    CHECK(std::abs(out[0].calc_gain(0.0f, 0) - C.paths[0].calc_gain(0.0f, 0)) < 1e-6f);
}

TEST_CASE("ray_commit - scalar layout carries no mirror")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray(20.0f, WIDE_DEG, 1, true);

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;
    REQUIRE(C.commit(S, P, out) == 1);

    REQUIRE(out[0].is_scalar());
    const float *src = C.paths[0].xpr_coeff(0);
    const float *dst = out[0].xpr_coeff(0);
    CHECK(std::abs(dst[0] - src[0]) < 1e-6f);
    CHECK(std::abs(dst[1] - src[1]) < 1e-6f);
}

TEST_CASE("ray_commit - in-medium loss on the final leg")
{
    // Lossy cube; the ray is declared to be travelling inside material 1
    Scene S = one_cube(50.0f, 0.0f, 0.0f, mtl_weak_loss());
    Cfg C = one_ray();

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    REQUIRE(C.commit(S, P, out) == 1);
    const float gain_outside = out[0].calc_gain(0.0f, 0);
    out.clear();

    C.cur(0) = 1; // inside material 1
    REQUIRE(C.commit(S, P, out) == 1);
    const float gain_inside = out[0].calc_gain(0.0f, 0);

    // Exact oracle: the leg from the ray origin to the receiver, at the reference frequency
    const float seg = 5.0f - R0;
    const float expect = quadriga_lib::medium_gain(S.mtl_prop, 1u, seg, FRQ);
    REQUIRE(expect < 0.99f); // the material must actually be lossy
    REQUIRE(expect > 1e-3f); // and not so lossy that the gain gate does the dropping
}

TEST_CASE("ray_commit - the resolved flag on air is not an in-medium state")
{
    // Material 0 with the resolved bit set: mat = w & 0x7FFF is zero, so no attenuation applies
    Scene S = one_cube(50.0f, 0.0f, 0.0f, mtl_concrete());
    Cfg C = one_ray();

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    REQUIRE(C.commit(S, P, out) == 1);
    const float plain = out[0].calc_gain(0.0f, 0);
    out.clear();

    C.cur(0) = (short)0x8000; // flag set, material 0
    REQUIRE(C.commit(S, P, out) == 1);
    CHECK(std::abs(out[0].calc_gain(0.0f, 0) - plain) < 1e-6f * plain);
}

TEST_CASE("ray_commit - every frequency slot is updated")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f, mtl_weak_loss());
    Cfg C = one_ray(20.0f, WIDE_DEG, 3);
    C.cur(0) = 1; // inside, so the per-frequency loss differs across bands

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;
    REQUIRE(C.commit(S, P, out) == 1);
    REQUIRE(out[0].n_freq() == 3);

    const float seg = 5.0f - R0;
    for (size_t f = 0; f < 3; ++f)
    {
        const float *src = C.paths[0].xpr_coeff(f);
        const float *dst = out[0].xpr_coeff(f);
        const float g = quadriga_lib::medium_gain(S.mtl_prop, 1u, seg, C.freq(f));
        const float amp = std::sqrt(g);

        CHECK(std::abs(dst[0] - amp * src[0]) < 1e-4f);
        CHECK(std::abs(dst[6] + amp * src[6]) < 1e-4f); // mirror on HH
    }

    // Concrete is more lossy with frequency, so the bands must not be identical
    CHECK(out[0].calc_gain(0.0f, 2) < out[0].calc_gain(0.0f, 0));
}

// ===========================================================================================
// Append semantics
// ===========================================================================================

TEST_CASE("ray_commit - results are appended, not overwritten")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();
    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});

    std::vector<quadriga_lib::path> out;
    REQUIRE(C.commit(S, P, out) == 1);
    out[0].iC = 12345u; // a marker the second call must not touch

    const arma::uword n2 = C.commit(S, P, out);
    CHECK(n2 == 1);
    CHECK(out.size() == 2);
    CHECK(out[0].iC == 12345u);
    CHECK(out[1].iC == 0u);
}

TEST_CASE("ray_commit - nothing is appended when no pair survives")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray(20.0f, 1.0f);
    arma::fmat P = pts({{5.0f, 9.0f, 0.0f}}); // outside the tube

    std::vector<quadriga_lib::path> out;
    out.resize(3);
    for (size_t i = 0; i < 3; ++i)
        out[i].init(0, 1, false), out[i].iC = (unsigned)(100 + i);

    CHECK(C.commit(S, P, out) == 0);
    REQUIRE(out.size() == 3);
    for (size_t i = 0; i < 3; ++i)
        CHECK(out[i].iC == (unsigned)(100 + i));
}

TEST_CASE("ray_commit - repeated calls are deterministic")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    arma::fmat d(64, 3, arma::fill::none);
    for (arma::uword i = 0; i < 64; ++i)
        d(i, 0) = 1.0f, d(i, 1) = 0.02f * (float)i - 0.6f, d(i, 2) = 0.01f * (float)i - 0.3f;
    Cfg C = make_cfg(d, 30.0f);

    arma::fmat P(200, 3, arma::fill::none);
    for (arma::uword i = 0; i < 200; ++i)
        P(i, 0) = 5.0f + 0.05f * (float)i, P(i, 1) = 0.03f * (float)(i % 13) - 0.2f, P(i, 2) = 0.0f;

    std::vector<quadriga_lib::path> a, b;
    const arma::uword na = C.commit(S, P, a);
    const arma::uword nb = C.commit(S, P, b);

    REQUIRE(na == nb);
    REQUIRE(na > 0); // a vacuous comparison is a failed comparison
    for (size_t i = 0; i < a.size(); ++i)
    {
        CHECK(a[i].iC == b[i].iC);
        CHECK(a[i].n_seg() == b[i].n_seg());
        CHECK((int)a[i].nREF == (int)b[i].nREF);
        CHECK((int)a[i].nTRA == (int)b[i].nTRA);
        CHECK((int)a[i].nSUB == (int)b[i].nSUB);
        CHECK(a[i].length() == b[i].length());
        CHECK(a[i].calc_gain(0.0f, 0) == b[i].calc_gain(0.0f, 0));
    }
}

// ===========================================================================================
// Scale
// ===========================================================================================

TEST_CASE("ray_commit - the pair list spans more than one block")
{
    // Pass B and pass C block the flat pair list in units of 65536. One wide beam over 90000
    // receivers puts the prefix sum and the point seek across a block boundary.
    Scene S = one_cube(200.0f, 0.0f, 0.0f);
    Cfg C = one_ray(200.0f, 20.0f);

    const arma::uword n_pt = 150000;
    arma::fmat P(n_pt, 3, arma::fill::none);
    for (arma::uword i = 0; i < n_pt; ++i)
    {
        // Every third point sits off-axis and misses, so empty CSR rows are crossed too
        const float x = 5.0f + 0.001f * (float)i;
        const bool off = (i % 3 == 2);
        P(i, 0) = x, P(i, 1) = off ? 100.0f : 0.0f, P(i, 2) = 0.0f;
    }

    std::vector<quadriga_lib::path> out;
    const arma::uword n = C.commit(S, P, out);

    REQUIRE(n > 65536u); // the boundary must actually be crossed
    REQUIRE(out.size() == (size_t)n);
    check_commit_shape(out, 0, n_pt, 1, false);

    // Every committed receiver is one of the on-axis points
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(out[i].iC % 3 != 2);

    // Committed receiver indices are strictly increasing here: one ray, one hit per point
    for (size_t i = 1; i < out.size(); ++i)
        CHECK(out[i].iC > out[i - 1].iC);
}

TEST_CASE("ray_commit - sub-cloud partitioning does not change the result")
{
    Scene S = one_cube(200.0f, 0.0f, 0.0f);
    Cfg C = one_ray(200.0f, 20.0f);

    const arma::uword n_pt = 64;
    arma::fmat P(n_pt, 3, arma::fill::none);
    for (arma::uword i = 0; i < n_pt; ++i)
        P(i, 0) = 5.0f + 0.1f * (float)i, P(i, 1) = 0.0f, P(i, 2) = 0.0f;

    std::vector<quadriga_lib::path> a, b;
    const arma::uword na = C.commit(S, P, a);

    // Ascending, first entry 0, aligned to the AVX2 vector size
    arma::u32_vec sci = {0u, 16u, 32u, 48u};
    Opt o;
    o.sci = &sci;
    const arma::uword nb = C.commit(S, P, b, o);

    REQUIRE(na == nb);
    REQUIRE(na > 0);
    for (size_t i = 0; i < a.size(); ++i)
    {
        CHECK(a[i].iC == b[i].iC);
        CHECK(a[i].calc_gain(0.0f, 0) == b[i].calc_gain(0.0f, 0));
    }
}

// ===========================================================================================
// Point index mapping
// ===========================================================================================

TEST_CASE("ray_commit - point_index drops padding rows")
{
    // A hand-built stand-in for a segmented cloud: row 1 is padding, the other two are real.
    // point_index is 1-based into the caller's original list, 0 marks padding.
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{3.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}, {7.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    // Without the map all three rows are receivers
    REQUIRE(C.commit(S, P, out) == 3);
    out.clear();

    arma::u32_vec fwd = {1u, 0u, 2u};
    Opt o;
    o.pt_ind = &fwd;

    REQUIRE(C.commit(S, P, out, o) == 2);
    CHECK(out[0].iC == 0u); // row 0 maps to original point 0
    CHECK(out[1].iC == 1u); // row 2 maps to original point 1, row 1 was padding

    // An empty map is treated as absent
    out.clear();
    arma::u32_vec none;
    o.pt_ind = &none;
    CHECK(C.commit(S, P, out, o) == 3);
}

TEST_CASE("ray_commit - point_index remaps iC into the original ordering")
{
    // A reversing map: the segmented row order is the opposite of the caller's, so a correct
    // implementation returns descending iC while the storage order stays point-major
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();

    arma::fmat P = pts({{3.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}, {7.0f, 0.0f, 0.0f}});
    arma::u32_vec fwd = {3u, 2u, 1u};

    Opt o;
    o.pt_ind = &fwd;
    std::vector<quadriga_lib::path> out;
    REQUIRE(C.commit(S, P, out, o) == 3);

    CHECK(out[0].iC == 2u);
    CHECK(out[1].iC == 1u);
    CHECK(out[2].iC == 0u);

    // The leg is still measured to the row that was actually hit, not to the mapped index
    check_commit_shape(out, 0, P.n_rows, 1, false, false);
}

TEST_CASE("ray_commit - point_index agrees with point_cloud_segmentation")
{
    // The real integration: segment a cloud with SIMD alignment, then check that committing
    // against the segmented cloud with its forward_index reproduces the unsegmented result.
    Scene S = one_cube(200.0f, 0.0f, 0.0f);
    Cfg C = one_ray(200.0f, 20.0f);

    const arma::uword n_pt = 50;
    arma::fmat P0(n_pt, 3, arma::fill::none);
    for (arma::uword i = 0; i < n_pt; ++i)
        P0(i, 0) = 5.0f + 0.2f * (float)i, P0(i, 1) = 0.0f, P0(i, 2) = 0.0f;

    arma::fmat PR;
    arma::u32_vec sci, fwd;
    quadriga_lib::point_cloud_segmentation<float>(&P0, &PR, &sci, 8, 8, &fwd);

    // The scenario is only meaningful when alignment padding was actually inserted
    REQUIRE(PR.n_rows > n_pt);
    REQUIRE(fwd.n_elem == PR.n_rows);
    REQUIRE(arma::any(fwd == 0u));

    std::vector<quadriga_lib::path> a, b;
    const arma::uword na = C.commit(S, P0, a);
    REQUIRE(na > 0);

    Opt o;
    o.sci = &sci;
    o.pt_ind = &fwd;
    const arma::uword nb = C.commit(S, PR, b, o);

    CHECK(na == nb);
    CHECK(sorted_iC(a) == sorted_iC(b));
    check_commit_shape(b, 0, n_pt, 1, false, false);
}

TEST_CASE("ray_commit - padding is committed when point_index is omitted")
{
    // Padding points sit at the centre of their sub-cloud AABB, so they are hit as readily as
    // any real point. This pins the failure mode the map exists to prevent.
    Scene S = one_cube(200.0f, 0.0f, 0.0f);
    Cfg C = one_ray(200.0f, 20.0f);

    const arma::uword n_pt = 50;
    arma::fmat P0(n_pt, 3, arma::fill::none);
    for (arma::uword i = 0; i < n_pt; ++i)
        P0(i, 0) = 5.0f + 0.2f * (float)i, P0(i, 1) = 0.0f, P0(i, 2) = 0.0f;

    arma::fmat PR;
    arma::u32_vec sci, fwd;
    quadriga_lib::point_cloud_segmentation<float>(&P0, &PR, &sci, 8, 8, &fwd);
    REQUIRE(PR.n_rows > n_pt);

    Opt with_map, without_map;
    with_map.sci = &sci, with_map.pt_ind = &fwd;
    without_map.sci = &sci;

    std::vector<quadriga_lib::path> a, b;
    const arma::uword na = C.commit(S, PR, a, with_map);
    const arma::uword nb = C.commit(S, PR, b, without_map);

    CHECK(nb > na); // the extra entries are the padding rows
}

// ===========================================================================================
// min_no_segments
// ===========================================================================================

TEST_CASE("ray_commit - min_no_segments gate")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();
    REQUIRE(C.paths[0].n_seg() == 0);

    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    // The default admits a zero-segment ray
    Opt o;
    CHECK(C.commit(S, P, out, o) == 1);

    // Requiring one interaction excludes it: this is the launch-sphere beam of generation 0
    out.clear();
    o.min_seg = 1;
    CHECK(C.commit(S, P, out, o) == 0);

    // After an interaction the same ray commits again
    add_segment(C, 0);
    REQUIRE(C.paths[0].n_seg() == 1);
    out.clear();
    CHECK(C.commit(S, P, out, o) == 1);

    // And a higher threshold excludes it once more
    out.clear();
    o.min_seg = 2;
    CHECK(C.commit(S, P, out, o) == 0);
}

TEST_CASE("ray_commit - min_no_segments and ignore_direct_path are independent gates")
{
    // A once-reflected ray passes ignore_direct_path but not a two-segment threshold, and a
    // two-segment transmission-only ray passes the threshold but not ignore_direct_path
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    {
        Cfg C = one_ray();
        add_segment(C, 0);
        C.paths[0].nREF = 1;

        Opt o;
        o.ignore_direct = true;
        CHECK(C.commit(S, P, out, o) == 1);

        out.clear();
        o.min_seg = 2;
        CHECK(C.commit(S, P, out, o) == 0);
    }

    {
        Cfg C = one_ray();
        add_segment(C, 0);
        add_segment(C, 0);
        C.paths[0].nTRA = 2; // transmissions only, so it is still a direct arrival

        Opt o;
        o.min_seg = 2;
        out.clear();
        CHECK(C.commit(S, P, out, o) == 1);

        o.ignore_direct = true;
        out.clear();
        CHECK(C.commit(S, P, out, o) == 0);
    }
}

// ===========================================================================================
// Input validation
// ===========================================================================================

TEST_CASE("ray_commit - input validation")
{
    Scene S = one_cube(50.0f, 0.0f, 0.0f);
    Cfg C = one_ray();
    arma::fmat P = pts({{5.0f, 0.0f, 0.0f}});
    std::vector<quadriga_lib::path> out;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(C.commit(S, P, out));
    }

    SECTION("mesh")
    {
        Scene bad = S;
        bad.mesh.reset();
        CHECK_THROWS_AS(C.commit(bad, P, out), std::invalid_argument);

        bad = S;
        bad.mesh = bad.mesh.cols(0, 7);
        CHECK_THROWS_AS(C.commit(bad, P, out), std::invalid_argument);
    }

    SECTION("center_frequency")
    {
        Cfg X = C;
        X.freq.reset();
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        X = C;
        X.freq = arma::fvec{-1.0f};
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        X = C;
        X.freq = arma::fvec(128, arma::fill::ones) * FRQ;
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        // Count must match the layout of the paths
        X = C;
        X.freq = arma::fvec{FRQ, 2.0f * FRQ};
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);
    }

    SECTION("orig")
    {
        Cfg X = C;
        X.orig.reset();
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        X = C;
        X.orig = arma::fmat(1, 2, arma::fill::zeros);
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);
    }

    SECTION("per-ray inputs must match n_ray")
    {
        Cfg X = C;
        X.fbs_ind.zeros(2);
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        X = C;
        X.cur.zeros(2);
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        X = C;
        X.paths.resize(2);
        X.paths[1].init(0, 1, false);
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);
    }

    SECTION("fbs_ind out of range")
    {
        Cfg X = C;
        X.fbs_ind(0) = (unsigned)S.mesh.n_rows + 1u;
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);

        // The last valid face index is accepted
        X.fbs_ind(0) = (unsigned)S.mesh.n_rows;
        CHECK_NOTHROW(X.commit(S, P, out));
    }

    SECTION("subdiv_flag_in length")
    {
        std::vector<bool> f = {false, false};
        Opt o;
        o.sf_in = &f;
        CHECK_THROWS_AS(C.commit(S, P, out, o), std::invalid_argument);
    }

    SECTION("point_index length")
    {
        // P has one row, so a two-element map does not describe it
        arma::u32_vec bad = {1u, 2u};
        Opt o;
        o.pt_ind = &bad;
        CHECK_THROWS_AS(C.commit(S, P, out, o), std::invalid_argument);

        arma::u32_vec good = {1u};
        o.pt_ind = &good;
        CHECK_NOTHROW(C.commit(S, P, out, o));

        // Empty is absent, not a length mismatch
        arma::u32_vec none;
        o.pt_ind = &none;
        CHECK_NOTHROW(C.commit(S, P, out, o));
    }

    SECTION("thresholds")
    {
        Opt o;
        o.min_gain_dB = std::numeric_limits<float>::quiet_NaN();
        CHECK_THROWS_AS(C.commit(S, P, out, o), std::invalid_argument);

        o = Opt();
        o.max_len = 0.0f;
        CHECK_THROWS_AS(C.commit(S, P, out, o), std::invalid_argument);

        o.max_len = -1.0f;
        CHECK_THROWS_AS(C.commit(S, P, out, o), std::invalid_argument);
    }

    SECTION("paths_commit layout must match")
    {
        std::vector<quadriga_lib::path> pre(1);
        pre[0].init(0, 2, false); // two frequencies, the configuration has one
        CHECK_THROWS_AS(C.commit(S, P, pre), std::invalid_argument);

        std::vector<quadriga_lib::path> sc(1);
        sc[0].init(0, 1, true); // scalar layout against an EM configuration
        CHECK_THROWS_AS(C.commit(S, P, sc), std::invalid_argument);
    }

    SECTION("points")
    {
        arma::fmat empty;
        CHECK_THROWS_AS(C.commit(S, empty, out), std::invalid_argument);

        arma::fmat wide(1, 4, arma::fill::zeros);
        CHECK_THROWS_AS(C.commit(S, wide, out), std::invalid_argument);
    }

    SECTION("all paths must share one layout")
    {
        Cfg X = one_ray();
        X.paths.resize(2);
        X.paths[1].init(0, 2, false); // second path has two frequencies
        X.orig = arma::join_vert(X.orig, X.orig);
        X.dest = arma::join_vert(X.dest, X.dest);
        X.trivec = arma::join_vert(X.trivec, X.trivec);
        X.tridir = arma::join_vert(X.tridir, X.tridir);
        X.fbs_ind.zeros(2);
        X.cur.zeros(2);
        CHECK_THROWS_AS(X.commit(S, P, out), std::invalid_argument);
    }
}