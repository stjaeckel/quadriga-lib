// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"

#include <cmath>
#include <vector>

using quadriga_lib::path;

// Layout mirror for expected-value computation in tests. Must match qd_path.cpp.
namespace
{
    constexpr size_t INLINE_TYPES = 2;

    size_t hist_floats(size_t nseg)
    {
        return nseg <= INLINE_TYPES ? 0 : (nseg - INLINE_TYPES + 3) / 4;
    }
    size_t hist_off(size_t nseg, size_t nfrq, bool scalar)
    {
        return scalar ? nseg * 3 + (nfrq < 5 ? 0 : (nfrq - 4) * 2)
                      : nseg * 3 + (nfrq < 2 ? 0 : (nfrq - 1) * 8);
    }
    size_t dsize(size_t nseg, size_t nfrq, bool scalar)
    {
        return hist_off(nseg, nfrq, scalar) + hist_floats(nseg);
    }

    // 2x2 complex Jones matrix -> 8-float column-major [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]
    void set_jones(float *m, float rvv, float ivv, float rhv, float ihv,
                   float rvh, float ivh, float rhh, float ihh)
    {
        m[0] = rvv, m[1] = ivv, m[2] = rhv, m[3] = ihv;
        m[4] = rvh, m[5] = ivh, m[6] = rhh, m[7] = ihh;
    }
}

TEST_CASE("path - default construction")
{
    path p;
    CHECK(p.n_freq() == 1);
    CHECK_FALSE(p.is_scalar());
    CHECK(p.length() == 0.0f);
    CHECK(p.iC == 0);
    CHECK(p.nREF == 0);
    CHECK(p.nTRA == 0);
    CHECK(p.nSUB == 0);
    CHECK(p.nSCT == 0);

    // Default xprmat is the identity Jones matrix; freq-0 gain (no path loss) is 1.
    CHECK(std::abs(p.calc_gain() - 1.0f) < 1e-6f);

    // No segments: last-coordinate accessor returns NaN for every index.
    CHECK(std::isnan(p(0)));
    CHECK(std::isnan(p(1)));
    CHECK(std::isnan(p(2)));
    CHECK(std::isnan(p(3))); // out-of-range index also NaN, not UB
}

TEST_CASE("path - init argument validation")
{
    path p;
    CHECK_THROWS_AS(p.init(256), std::invalid_argument);    // segments > 255
    CHECK_THROWS_AS(p.init(0, 0), std::invalid_argument);   // frequencies == 0
    CHECK_THROWS_AS(p.init(0, 128), std::invalid_argument); // frequencies > 127
    CHECK_NOTHROW(p.init(255, 127));                        // boundary: both max
    CHECK(p.n_freq() == 127);
    CHECK(p.is_scalar() == false);
}

TEST_CASE("path - init EM layout and seeding")
{
    // 3 segments, 4 frequencies, EM. Freq-0 in xprmat, freqs 1..3 in the buffer.
    path p(3, 4, false);
    CHECK(p.n_freq() == 4);
    CHECK_FALSE(p.is_scalar());

    // Freq 0 seeded to identity in xprmat.
    const float *f0 = p.xpr_coeff(0);
    CHECK(f0[0] == 1.0f); // ReVV
    CHECK(f0[6] == 1.0f); // ReHH
    CHECK(f0[1] == 0.0f);

    // Freqs 1..3 seeded to identity in the buffer (ReVV=ReHH=1, rest 0).
    for (size_t f = 1; f < 4; ++f)
    {
        const float *jf = p.xpr_coeff(f);
        INFO("frequency " << f);
        CHECK(jf[0] == 1.0f); // ReVV
        CHECK(jf[6] == 1.0f); // ReHH
        CHECK(jf[2] == 0.0f); // ReHV
        CHECK(jf[4] == 0.0f); // ReVH
    }

    // Buffer freq offsets follow data + 3*nSEG + (f-1)*8.
    const float *base = p.xpr_coeff(1);
    CHECK(p.xpr_coeff(2) == base + 8);
    CHECK(p.xpr_coeff(3) == base + 16);
}

TEST_CASE("path - init scalar layout and seeding")
{
    // Scalar mode: freqs 0..3 live in xprmat (as 4 complex pairs), freqs >=4 in the buffer.
    path p(2, 6, true);
    CHECK(p.n_freq() == 6);
    CHECK(p.is_scalar());

    // xprmat seeded so all four in-header pressure coeffs have Re=1.
    const float *x = p.xpr_coeff(0);
    CHECK(x[0] == 1.0f); // Re(F0)
    CHECK(x[2] == 1.0f); // Re(F1)
    CHECK(x[4] == 1.0f); // Re(F2)
    CHECK(x[6] == 1.0f); // Re(F3)

    // Freqs 1..3 alias into xprmat at stride 2.
    CHECK(p.xpr_coeff(1) == x + 2);
    CHECK(p.xpr_coeff(2) == x + 4);
    CHECK(p.xpr_coeff(3) == x + 6);

    // Freqs 4,5 seeded to Re=1 in the buffer at data + 3*nSEG + (f-4)*2.
    const float *f4 = p.xpr_coeff(4);
    const float *f5 = p.xpr_coeff(5);
    CHECK(f4[0] == 1.0f);
    CHECK(f5[0] == 1.0f);
    CHECK(f5 == f4 + 2);
}

TEST_CASE("path - xpr_coeff bounds")
{
    path p(2, 3, false);
    CHECK_THROWS_AS(p.xpr_coeff(3), std::invalid_argument); // freq == n_freq
    CHECK_THROWS_AS(p.xpr_coeff(99), std::invalid_argument);
    CHECK_NOTHROW(p.xpr_coeff(0));
    CHECK_NOTHROW(p.xpr_coeff(2));
}

TEST_CASE("path - coord access and bounds")
{
    path p(3, 1, false);
    // Write coordinates via the mutable accessor.
    float *c0 = p.coord(0);
    c0[0] = 1.0f, c0[1] = 2.0f, c0[2] = 3.0f;
    float *c2 = p.coord(2);
    c2[0] = 7.0f, c2[1] = 8.0f, c2[2] = 9.0f;

    const path &cp = p;
    CHECK(cp.coord(0)[1] == 2.0f);
    CHECK(cp.coord(2)[0] == 7.0f);

    // operator() reads the last segment's coordinate.
    CHECK(p(0) == 7.0f);
    CHECK(p(1) == 8.0f);
    CHECK(p(2) == 9.0f);

    CHECK_THROWS_AS(p.coord(3), std::invalid_argument);
}

TEST_CASE("path - calc_length read-only (member + last-to-D)")
{
    path p(2, 1, false);
    float *c0 = p.coord(0);
    c0[0] = 0.0f, c0[1] = 0.0f, c0[2] = 0.0f;
    float *c1 = p.coord(1);
    c1[0] = 3.0f, c1[1] = 0.0f, c1[2] = 0.0f;
    p.set_length(10.0f); // pretend accumulated length

    // last point is (3,0,0); D at (3,4,0) is 4 away -> 10 + 4 = 14.
    CHECK(std::abs(p.calc_length(3.0f, 4.0f, 0.0f) - 14.0f) < 1e-5f);

    // Pointer overload, O = nullptr, same result.
    float D[3] = {3.0f, 4.0f, 0.0f};
    CHECK(std::abs(p.calc_length(D) - 14.0f) < 1e-5f);

    // Read-only form does not touch the member.
    CHECK(p.length() == 10.0f);
}

TEST_CASE("path - calc_length read-only on empty path returns NaN")
{
    path p; // nSEG == 0
    CHECK(std::isnan(p.calc_length(1.0f, 2.0f, 3.0f)));
}

TEST_CASE("path - calc_length full recalculation updates member")
{
    path p(2, 1, false);
    float *c0 = p.coord(0);
    c0[0] = 1.0f, c0[1] = 0.0f, c0[2] = 0.0f;
    float *c1 = p.coord(1);
    c1[0] = 4.0f, c1[1] = 0.0f, c1[2] = 0.0f;

    // O(0,0,0) -> seg0(1,0,0): 1 ; seg0 -> seg1(4,0,0): 3 ; seg1 -> D(4,4,0): 4
    // stored member = O..last = 1 + 3 = 4 ; return = 4 + 4 = 8.
    float ret = p.calc_length(4.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    CHECK(std::abs(ret - 8.0f) < 1e-5f);
    CHECK(std::abs(p.length() - 4.0f) < 1e-5f); // member excludes the ->D leg

    // A subsequent read-only call must agree: member(4) + last(4,0,0)->D(4,4,0)=4 -> 8.
    CHECK(std::abs(p.calc_length(4.0f, 4.0f, 0.0f) - 8.0f) < 1e-5f);

    // Pointer overload with O triggers the same recalculation.
    float D[3] = {4.0f, 4.0f, 0.0f}, O[3] = {0.0f, 0.0f, 0.0f};
    CHECK(std::abs(p.calc_length(D, O) - 8.0f) < 1e-5f);
}

TEST_CASE("path - calc_length recalculation on empty path is O->D")
{
    path p; // nSEG == 0, loop body never runs
    float ret = p.calc_length(3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    CHECK(std::abs(ret - 5.0f) < 1e-5f); // straight O->D distance
    CHECK(p.length() == 0.0f);             // no interior length
}

TEST_CASE("path - calc_gain polarization power without path loss")
{
    path p(0, 1, false);
    // Identity Jones -> max column power = 1.
    CHECK(std::abs(p.calc_gain() - 1.0f) < 1e-6f);

    // Scale VV column so |col0|^2 = 4, |col1|^2 = 1; gain = max = 4.
    float *m = p.xpr_coeff(0);
    set_jones(m, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    CHECK(std::abs(p.calc_gain() - 4.0f) < 1e-6f);
}

TEST_CASE("path - calc_gain degenerate matrix returns zero")
{
    path p(0, 1, false);
    float *m = p.xpr_coeff(0);
    set_jones(m, 0, 0, 0, 0, 0, 0, 0, 0); // all zero
    CHECK(p.calc_gain() == 0.0f);

    set_jones(m, NAN, 0, 0, 0, 0, 0, 1.0f, 0); // NaN in col 0 (the larger side)
    CHECK(p.calc_gain() == 0.0f);
}

TEST_CASE("path - calc_gain EM free-space path loss")
{
    path p(0, 1, false);
    p.set_length(100.0f);

    // FSPL linear power = (c / (4 pi f d))^2 with f in GHz, d in m, c = 0.299792458 m*GHz.
    float fGHz = 2.4f;
    float k = 0.299792458f / (12.566370614f * fGHz * p.length());
    float expected = k * k; // identity matrix gain == 1

    CHECK(std::abs(p.calc_gain(fGHz) - expected) < 1e-12f);

    // fGHz == 0 disables path loss.
    CHECK(std::abs(p.calc_gain(0.0f) - 1.0f) < 1e-6f);
}

TEST_CASE("path - calc_gain scalar spherical spreading")
{
    path p(0, 2, true);
    p.set_length(10.0f);
    float *c = p.xpr_coeff(0); // scalar coeff [Re, Im]
    c[0] = 1.0f, c[1] = 0.0f;

    // Scalar: any fGHz > 0 applies 1/d^2, magnitude ignored.
    CHECK(std::abs(p.calc_gain(1.0f) - (1.0f / 100.0f)) < 1e-8f);
    CHECK(std::abs(p.calc_gain(50.0f) - (1.0f / 100.0f)) < 1e-8f); // magnitude irrelevant

    // No path loss when fGHz == 0.
    CHECK(std::abs(p.calc_gain(0.0f) - 1.0f) < 1e-6f);
}

TEST_CASE("path - xpr_update pure gain, EM")
{
    path p(0, 1, false);
    // Identity matrix, apply power gain 4 -> amplitude 2 on each stored value.
    float g = p.xpr_update(nullptr, 4.0f);
    // Resulting matrix diag(2,2): |col0|^2 = 4, |col1|^2 = 4, gain = 4.
    CHECK(std::abs(g - 4.0f) < 1e-5f);

    const float *m = p.xpr_coeff(0);
    CHECK(std::abs(m[0] - 2.0f) < 1e-5f);
    CHECK(std::abs(m[6] - 2.0f) < 1e-5f);
}

TEST_CASE("path - xpr_update left-multiply, EM")
{
    path p(0, 1, false);
    // Start from identity. Left-multiply by U = diag(2, 3) (real).
    float U[8];
    set_jones(U, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f);
    float g = p.xpr_update(U, 1.0f);

    // Result diag(2,3): |col0|^2 = 4, |col1|^2 = 9, gain = max = 9.
    CHECK(std::abs(g - 9.0f) < 1e-5f);
    const float *m = p.xpr_coeff(0);
    CHECK(std::abs(m[0] - 2.0f) < 1e-5f);
    CHECK(std::abs(m[6] - 3.0f) < 1e-5f);
    CHECK(m[2] == 0.0f);
    CHECK(m[4] == 0.0f);
}

TEST_CASE("path - xpr_update complex multiply matches hand calc")
{
    path p(0, 1, false);
    // Set state to a known matrix, multiply by a rotation-like update, verify one entry.
    float *m = p.xpr_coeff(0);
    set_jones(m, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f); // VV = 1+i

    float U[8];
    set_jones(U, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f); // U_VV = i
    p.xpr_update(U, 1.0f);

    // New VV = i * (1+i) = -1 + i  -> Re = -1, Im = 1.
    const float *r = p.xpr_coeff(0);
    CHECK(std::abs(r[0] - (-1.0f)) < 1e-5f);
    CHECK(std::abs(r[1] - 1.0f) < 1e-5f);
}

TEST_CASE("path - xpr_update self-aliasing is safe")
{
    path p(0, 1, false);
    float *m = p.xpr_coeff(0);
    set_jones(m, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    // Passing the state as its own update squares VV: (1+i)^2 = 2i.
    p.xpr_update(p.xpr_coeff(0), 1.0f);
    const float *r = p.xpr_coeff(0);
    CHECK(std::abs(r[0] - 0.0f) < 1e-5f);
    CHECK(std::abs(r[1] - 2.0f) < 1e-5f);
}

TEST_CASE("path - xpr_update scalar mode")
{
    path p(0, 2, true);
    float *c = p.xpr_coeff(0);
    c[0] = 1.0f, c[1] = 0.0f;

    // Multiply by (0 + i): result i, |.|^2 = 1.
    float U[2] = {0.0f, 1.0f};
    float g = p.xpr_update(U, 1.0f);
    CHECK(std::abs(g - 1.0f) < 1e-5f);
    CHECK(std::abs(c[0] - 0.0f) < 1e-5f);
    CHECK(std::abs(c[1] - 1.0f) < 1e-5f);

    // Pure power gain 9 -> amplitude 3, |.|^2 = 9.
    g = p.xpr_update(nullptr, 9.0f);
    CHECK(std::abs(g - 9.0f) < 1e-5f);
}

TEST_CASE("path - xpr_update applies path loss when fGHz given")
{
    path p(0, 1, false);
    p.set_length(100.0f);
    float fGHz = 2.4f;
    float k = 0.299792458f / (12.566370614f * fGHz * p.length());
    float expected = k * k; // identity gain 1, scaled by FSPL

    float g = p.xpr_update(nullptr, 1.0f, 0, fGHz);
    CHECK(std::abs(g - expected) < 1e-12f);
}

TEST_CASE("path - duplicate produces independent deep copy")
{
    path src(3, 2, false);
    float *c = src.coord(1);
    c[0] = 5.0f, c[1] = 6.0f, c[2] = 7.0f;
    src.iC = 11;
    src.nREF = 2, src.nTRA = 3, src.nSUB = 1, src.nSCT = 4;
    src.set_length(99.0f);
    float *m = src.xpr_coeff(1);
    m[0] = 3.0f;

    path dst;
    float ret = src.duplicate(dst);
    CHECK(ret == 99.0f);

    // Metadata copied.
    CHECK(dst.iC == 11);
    CHECK(dst.nREF == 2);
    CHECK(dst.nTRA == 3);
    CHECK(dst.nSUB == 1);
    CHECK(dst.nSCT == 4);
    CHECK(dst.n_freq() == 2);
    CHECK(dst.coord(1)[1] == 6.0f);
    CHECK(dst.xpr_coeff(1)[0] == 3.0f);

    // Deep copy: buffers are distinct allocations.
    CHECK(dst.coord(0) != src.coord(0));
    dst.coord(1)[0] = -1.0f;
    CHECK(src.coord(1)[0] == 5.0f); // src unaffected
}

TEST_CASE("path - extend appends segment, distance, counters")
{
    // Start with one segment at origin.
    path src(1, 1, false);
    float *c = src.coord(0);
    c[0] = 0.0f, c[1] = 0.0f, c[2] = 0.0f;
    src.set_length(0.0f);
    src.nREF = 0, src.nTRA = 0, src.nSCT = 7;

    // Extend to (3,4,0): new segment 3-4-5 triangle -> +5 length. type 100 -> nTRA++.
    path t1;
    float L = src.extend(t1, 3.0f, 4.0f, 0.0f, 100);
    CHECK(std::abs(L - 5.0f) < 1e-5f);
    CHECK(std::abs(t1.length() - 5.0f) < 1e-5f);
    CHECK(t1.n_seg() == 2);
    CHECK(t1.nTRA == 0);
    CHECK(t1.nREF == 0);
    CHECK(t1.nSCT == 7); // propagated unchanged

    // New coordinate landed in the appended slot.
    CHECK(t1(0) == 3.0f);
    CHECK(t1(1) == 4.0f);
    CHECK(t1(2) == 0.0f);
}

TEST_CASE("path - extend across the inline/history boundary")
{
    // Chain extends 1..N so that segment s gets interaction code s+1. The codes for the
    // first INLINE_TYPES segments sit in the header; the rest spill into the buffer.
    const size_t N = 10;
    REQUIRE(N > INLINE_TYPES); // otherwise the boundary is never crossed
    path cur(1, 1, false);
    cur.coord(0)[0] = 0.0f, cur.coord(0)[1] = 0.0f, cur.coord(0)[2] = 0.0f;

    std::vector<uint8_t> expected;
    for (size_t s = 1; s < N; ++s) // N-1 extends -> N segments total
    {
        path nxt;
        uint8_t type = (uint8_t)(s + 1); // 2..N, all in 1..127 range
        cur.extend(nxt, (float)s, 0.0f, 0.0f, type);
        cur = std::move(nxt);
    }

    // The interaction sequence has one code per segment; segment 0's code was never
    // set (initial path), segments 1..N-1 carry codes 2..N.
    std::vector<uint8_t> seq = cur.interaction_type_codes();
    REQUIRE(seq.size() == N);
    CHECK(seq[0] == 0); // segment 0: default, never assigned
    for (size_t s = 1; s < N; ++s)
    {
        INFO("segment " << s);
        CHECK(seq[s] == (uint8_t)(s + 1));
    }
}

TEST_CASE("path - extend at segment cap throws")
{
    path p(255, 1, false);
    path t;
    CHECK_THROWS_AS(p.extend(t, 1.0f, 2.0f, 3.0f, 0), std::runtime_error);
}

TEST_CASE("path - interaction_type_codes short path stays inline")
{
    // A path with exactly INLINE_TYPES segments keeps every code in the header, so the
    // buffer carries no history block at all. Written against the constant rather than a
    // hard-coded count so it keeps testing the inline-only route if INLINE_TYPES changes.
    REQUIRE(hist_floats(INLINE_TYPES) == 0);

    path cur(1, 1, false);
    cur.coord(0)[0] = 0.0f;

    std::vector<uint8_t> expected(1, 0); // segment 0: default, never assigned
    for (size_t s = 1; s < INLINE_TYPES; ++s)
    {
        path nxt;
        uint8_t type = (uint8_t)(10 * s + 1);
        cur.extend(nxt, (float)s, 0.0f, 0.0f, type);
        cur = std::move(nxt);
        expected.push_back(type);
    }

    std::vector<uint8_t> seq = cur.interaction_type_codes();
    REQUIRE(seq.size() == INLINE_TYPES);
    for (size_t s = 0; s < INLINE_TYPES; ++s)
    {
        INFO("segment " << s);
        CHECK(seq[s] == expected[s]);
    }
}

TEST_CASE("path - move constructor transfers ownership")
{
    path src(4, 2, false);
    src.coord(0)[0] = 42.0f;
    src.iC = 5;
    const float *buf = src.coord(0);

    path dst(std::move(src));
    CHECK(dst.coord(0)[0] == 42.0f);
    CHECK(dst.coord(0) == buf); // same buffer, no reallocation
    CHECK(dst.iC == 5);

    // Moved-from path is left in the valid empty state (nSEG == 0, nFRQ == 1, no buffer).
    CHECK(src.n_seg() == 0);
    CHECK(src.n_freq() == 1);
    CHECK(std::isnan(src(0))); // accessor safe on the emptied source
}

TEST_CASE("path - move assignment frees existing and transfers")
{
    path a(3, 1, false);
    a.coord(0)[0] = 1.0f;
    path b(5, 1, false);
    b.coord(0)[0] = 2.0f;
    const float *b_buf = b.coord(0);

    a = std::move(b);
    CHECK(a.coord(0)[0] == 2.0f);
    CHECK(a.coord(0) == b_buf);
    CHECK(b.n_seg() == 0); // b emptied to the valid empty state
    CHECK(b.n_freq() == 1);
    CHECK(std::isnan(b(0)));
}

TEST_CASE("path - copy assignment is deep and self-safe")
{
    path a(3, 2, false);
    a.coord(1)[0] = 9.0f;
    a.iC = 7;

    path b;
    b = a;
    CHECK(b.coord(1)[0] == 9.0f);
    CHECK(b.iC == 7);
    CHECK(b.coord(1) != a.coord(1)); // distinct buffers
    b.coord(1)[0] = -5.0f;
    CHECK(a.coord(1)[0] == 9.0f); // a unaffected

    // Self-assignment must not corrupt or free-then-read.
    a = a;
    CHECK(a.coord(1)[0] == 9.0f);
    CHECK(a.iC == 7);
}

TEST_CASE("path - free resets to a valid empty state")
{
    path p(4, 3, true);
    p.free();

    // Buffer released and metadata reset to defaults (nSEG == 0, nFRQ == 1, EM).
    CHECK(p.n_seg() == 0);
    CHECK(p.n_freq() == 1);
    CHECK_FALSE(p.is_scalar());

    // Accessors must be safe on a freed object, not segfault.
    CHECK(std::isnan(p(0)));                                // nSEG == 0 -> NaN
    CHECK_THROWS_AS(p.coord(0), std::invalid_argument);     // seg >= nSEG(0)
    CHECK_NOTHROW(p.xpr_coeff(0));                          // freq 0 always valid (xprmat)
    CHECK_THROWS_AS(p.xpr_coeff(1), std::invalid_argument); // freq >= n_freq(1)

    // Idempotent: freeing again is safe.
    CHECK_NOTHROW(p.free());
    CHECK(p.n_seg() == 0);
}