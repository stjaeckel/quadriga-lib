// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"
#ifdef BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif
#ifdef BUILD_WITH_CUDA
#include "quadriga_lib_cuda_functions.hpp"
#endif
#include <cmath>
#include <cstring>

// Unit cube (vertices at +/-1), 12 triangles, row format: v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z
// Triangle ordering matches the public API test in test_ray_triangle_intersect.cpp
static const double cube_raw[12][9] = {
    {-1, 1, 1, 1, -1, 1, 1, 1, 1},      //  0 Top NorthEast
    {1, -1, 1, -1, -1, -1, 1, -1, -1},  //  1 South Lower
    {-1, -1, 1, -1, 1, -1, -1, -1, -1}, //  2 West Lower
    {1, 1, -1, -1, -1, -1, -1, 1, -1},  //  3 Bottom NorthWest
    {1, 1, 1, 1, -1, -1, 1, 1, -1},     //  4 East Lower
    {-1, 1, 1, 1, 1, -1, -1, 1, -1},    //  5 North Lower
    {-1, 1, 1, -1, -1, 1, 1, -1, 1},    //  6 Top SouthWest
    {1, -1, 1, -1, -1, 1, -1, -1, -1},  //  7 South Upper
    {-1, -1, 1, -1, 1, 1, -1, 1, -1},   //  8 West Upper
    {1, 1, -1, 1, -1, -1, -1, -1, -1},  //  9 Bottom SouthEast
    {1, 1, 1, 1, -1, 1, 1, -1, -1},     // 10 East Upper
    {-1, 1, 1, 1, 1, 1, 1, 1, -1}};     // 11 North Upper

// Helper: convert row-format triangles to SoA (Tx,Ty,Tz,E1x,...,E2z)
// Fills the first 12 entries; zero-fills indices [12, n_alloc)
template <typename T>
static void fill_cube_soa(T *Tx, T *Ty, T *Tz,
                          T *E1x, T *E1y, T *E1z,
                          T *E2x, T *E2y, T *E2z,
                          size_t n_alloc)
{
    for (size_t i = 0; i < 12; ++i)
    {
        T v1x = (T)cube_raw[i][0], v1y = (T)cube_raw[i][1], v1z = (T)cube_raw[i][2];
        T v2x = (T)cube_raw[i][3], v2y = (T)cube_raw[i][4], v2z = (T)cube_raw[i][5];
        T v3x = (T)cube_raw[i][6], v3y = (T)cube_raw[i][7], v3z = (T)cube_raw[i][8];
        Tx[i] = v1x;
        Ty[i] = v1y;
        Tz[i] = v1z;
        E1x[i] = v2x - v1x;
        E1y[i] = v2y - v1y;
        E1z[i] = v2z - v1z;
        E2x[i] = v3x - v1x;
        E2y[i] = v3y - v1y;
        E2z[i] = v3z - v1z;
    }
    for (size_t i = 12; i < n_alloc; ++i)
    {
        Tx[i] = T(0);
        Ty[i] = T(0);
        Tz[i] = T(0);
        E1x[i] = T(0);
        E1y[i] = T(0);
        E1z[i] = T(0);
        E2x[i] = T(0);
        E2y[i] = T(0);
        E2z[i] = T(0);
    }
}

// --- Test 1: Single ray, west to east, passes through cube (FBS + SBS) ---
TEST_CASE("RTI Internal - Cube west-east hit")
{
    const size_t n_mesh = 12, n_sub = 1, n_ray = 1;
    double Tx[12], Ty[12], Tz[12], E1x[12], E1y[12], E1z[12], E2x[12], E2y[12], E2z[12];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh);

    unsigned SMI[1] = {0};
    double Xmin[1] = {-1.0}, Xmax[1] = {1.0};
    double Ymin[1] = {-1.0}, Ymax[1] = {1.0};
    double Zmin[1] = {-1.0}, Zmax[1] = {1.0};

    // Ray: origin=(-10,0,0.5), dest=(10,0,0.5) → D=(20,0,0)
    double Ox[1] = {-10.0}, Oy[1] = {0.0}, Oz[1] = {0.5};
    double Dx[1] = {20.0}, Dy[1] = {0.0}, Dz[1] = {0.0};

    double Wf[1], Ws[1];
    unsigned If[1], Is[1], hit_cnt[1];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf, Ws, If, Is, hit_cnt);

    // FBS at x=-1 → W = 9/20 = 0.45, triangle 8 (0-based) → If = 9 (1-based)
    CHECK(std::abs(Wf[0] - 0.45) < 1e-12);
    CHECK(If[0] == 9);

    // SBS at x=+1 → W = 11/20 = 0.55, triangle 10 (0-based) → Is = 11 (1-based)
    CHECK(std::abs(Ws[0] - 0.55) < 1e-12);
    CHECK(Is[0] == 11);

    CHECK(hit_cnt[0] == 2);
}

// --- Test 2: Single ray, misses the cube entirely ---
TEST_CASE("RTI Internal - Miss")
{
    const size_t n_mesh = 12, n_sub = 1, n_ray = 1;
    double Tx[12], Ty[12], Tz[12], E1x[12], E1y[12], E1z[12], E2x[12], E2y[12], E2z[12];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh);

    unsigned SMI[1] = {0};
    double Xmin[1] = {-1.0}, Xmax[1] = {1.0};
    double Ymin[1] = {-1.0}, Ymax[1] = {1.0};
    double Zmin[1] = {-1.0}, Zmax[1] = {1.0};

    // Ray passes above and beside the cube
    double Ox[1] = {-10.0}, Oy[1] = {5.0}, Oz[1] = {5.0};
    double Dx[1] = {20.0}, Dy[1] = {0.0}, Dz[1] = {0.0};

    double Wf[1], Ws[1];
    unsigned If[1], Is[1], hit_cnt[1];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf, Ws, If, Is, hit_cnt);

    CHECK(Wf[0] == 1.0);
    CHECK(If[0] == 0);
    CHECK(Ws[0] == 1.0);
    CHECK(Is[0] == 0);
    CHECK(hit_cnt[0] == 0);
}

// --- Test 3: Ray starting inside the cube (single exit hit) ---
TEST_CASE("RTI Internal - Ray starting inside cube")
{
    const size_t n_mesh = 12, n_sub = 1, n_ray = 1;
    double Tx[12], Ty[12], Tz[12], E1x[12], E1y[12], E1z[12], E2x[12], E2y[12], E2z[12];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh);

    unsigned SMI[1] = {0};
    double Xmin[1] = {-1.0}, Xmax[1] = {1.0};
    double Ymin[1] = {-1.0}, Ymax[1] = {1.0};
    double Zmin[1] = {-1.0}, Zmax[1] = {1.0};

    // Origin inside cube, heading east: O=(0,0,0.5), dest=(10,0,0.5), D=(10,0,0)
    double Ox[1] = {0.0}, Oy[1] = {0.0}, Oz[1] = {0.5};
    double Dx[1] = {10.0}, Dy[1] = {0.0}, Dz[1] = {0.0};

    double Wf[1], Ws[1];
    unsigned If[1], Is[1], hit_cnt[1];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf, Ws, If, Is, hit_cnt);

    // Exits through east face at x=1 → W = 1/10 = 0.1, triangle 10 (0-based) → If = 11
    CHECK(std::abs(Wf[0] - 0.1) < 1e-12);
    CHECK(If[0] == 11);

    // No second hit
    CHECK(Ws[0] == 1.0);
    CHECK(Is[0] == 0);
    CHECK(hit_cnt[0] == 1);
}

// --- Test 4: Multiple rays in a single call ---
TEST_CASE("RTI Internal - Multiple rays")
{
    const size_t n_mesh = 12, n_sub = 1, n_ray = 4;
    double Tx[12], Ty[12], Tz[12], E1x[12], E1y[12], E1z[12], E2x[12], E2y[12], E2z[12];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh);

    unsigned SMI[1] = {0};
    double Xmin[1] = {-1.0}, Xmax[1] = {1.0};
    double Ymin[1] = {-1.0}, Ymax[1] = {1.0};
    double Zmin[1] = {-1.0}, Zmax[1] = {1.0};

    // Ray 0: west→east (same as test 1)
    // Ray 1: miss (same as test 2)
    // Ray 2: start inside (same as test 3)
    // Ray 3: south→north at x=1, z=0.5
    double Ox[4] = {-10.0, -10.0, 0.0, 1.0};
    double Oy[4] = {0.0, 5.0, 0.0, -10.0};
    double Oz[4] = {0.5, 5.0, 0.5, 0.5};
    double Dx[4] = {20.0, 20.0, 10.0, 0.0};
    double Dy[4] = {0.0, 0.0, 0.0, 20.0};
    double Dz[4] = {0.0, 0.0, 0.0, 0.0};

    double Wf[4], Ws[4];
    unsigned If[4], Is[4], hit_cnt[4];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf, Ws, If, Is, hit_cnt);

    // Ray 0: west→east hit
    CHECK(std::abs(Wf[0] - 0.45) < 1e-12);
    CHECK(If[0] == 9);
    CHECK(std::abs(Ws[0] - 0.55) < 1e-12);
    CHECK(Is[0] == 11);
    CHECK(hit_cnt[0] == 2);

    // Ray 1: miss
    CHECK(Wf[1] == 1.0);
    CHECK(If[1] == 0);
    CHECK(Ws[1] == 1.0);
    CHECK(Is[1] == 0);
    CHECK(hit_cnt[1] == 0);

    // Ray 2: start inside, exit east
    CHECK(std::abs(Wf[2] - 0.1) < 1e-12);
    CHECK(If[2] == 11);
    CHECK(Ws[2] == 1.0);
    CHECK(Is[2] == 0);
    CHECK(hit_cnt[2] == 1);

    // Ray 3: south→north, FBS = South Lower (tri 1, 1-based=2), SBS = North Upper (tri 11, 1-based=12)
    CHECK(std::abs(Wf[3] - 0.45) < 1e-12);
    CHECK(If[3] == 2);
    CHECK(std::abs(Ws[3] - 0.55) < 1e-12);
    CHECK(Is[3] == 12);
    CHECK(hit_cnt[3] == 2);
}

// --- Test 5: Sub-mesh AABB filtering ---
TEST_CASE("RTI Internal - Sub-mesh AABB filtering")
{
    const size_t n_mesh = 12, n_ray = 1;
    double Tx[12], Ty[12], Tz[12], E1x[12], E1y[12], E1z[12], E2x[12], E2y[12], E2z[12];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh);

    // Two sub-meshes: sub 0 = tris 0-5, sub 1 = tris 6-11
    const size_t n_sub = 2;
    unsigned SMI[2] = {0, 6};

    // West→east ray (hits tris 8 and 10, both in sub 1)
    double Ox[1] = {-10.0}, Oy[1] = {0.0}, Oz[1] = {0.5};
    double Dx[1] = {20.0}, Dy[1] = {0.0}, Dz[1] = {0.0};
    double Wf[1], Ws[1];
    unsigned If[1], Is[1], hit_cnt[1];

    // 5a: Exclude sub 1 by placing its AABB far away
    {
        double Xmin[2] = {-1.0, 100.0}, Xmax[2] = {1.0, 200.0};
        double Ymin[2] = {-1.0, 100.0}, Ymax[2] = {1.0, 200.0};
        double Zmin[2] = {-1.0, 100.0}, Zmax[2] = {1.0, 200.0};

        qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                       SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                       Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                       Wf, Ws, If, Is, hit_cnt);

        // Hit triangles are in sub 1 which is excluded → no hits
        CHECK(Wf[0] == 1.0);
        CHECK(If[0] == 0);
        CHECK(Ws[0] == 1.0);
        CHECK(Is[0] == 0);
        CHECK(hit_cnt[0] == 0);
    }

    // 5b: Both sub-meshes with correct AABB → same result as test 1
    {
        double Xmin[2] = {-1.0, -1.0}, Xmax[2] = {1.0, 1.0};
        double Ymin[2] = {-1.0, -1.0}, Ymax[2] = {1.0, 1.0};
        double Zmin[2] = {-1.0, -1.0}, Zmax[2] = {1.0, 1.0};

        qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                       SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                       Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                       Wf, Ws, If, Is, hit_cnt);

        CHECK(std::abs(Wf[0] - 0.45) < 1e-12);
        CHECK(If[0] == 9);
        CHECK(std::abs(Ws[0] - 0.55) < 1e-12);
        CHECK(Is[0] == 11);
        CHECK(hit_cnt[0] == 2);
    }
}

// --- Test 6: hit_cnt = nullptr (optional output omitted) ---
TEST_CASE("RTI Internal - hit_cnt nullptr")
{
    const size_t n_mesh = 12, n_sub = 1, n_ray = 1;
    double Tx[12], Ty[12], Tz[12], E1x[12], E1y[12], E1z[12], E2x[12], E2y[12], E2z[12];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh);

    unsigned SMI[1] = {0};
    double Xmin[1] = {-1.0}, Xmax[1] = {1.0};
    double Ymin[1] = {-1.0}, Ymax[1] = {1.0};
    double Zmin[1] = {-1.0}, Zmax[1] = {1.0};

    double Ox[1] = {-10.0}, Oy[1] = {0.0}, Oz[1] = {0.5};
    double Dx[1] = {20.0}, Dy[1] = {0.0}, Dz[1] = {0.0};

    double Wf[1], Ws[1];
    unsigned If[1], Is[1];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf, Ws, If, Is, (unsigned *)nullptr);

    CHECK(std::abs(Wf[0] - 0.45) < 1e-12);
    CHECK(If[0] == 9);
    CHECK(std::abs(Ws[0] - 0.55) < 1e-12);
    CHECK(Is[0] == 11);
}

// --- Test 7: Generic float vs. double ---
TEST_CASE("RTI Internal - Float vs Double")
{
    const size_t n_mesh = 12, n_sub = 1, n_ray = 4;

    // Double-precision data
    double Txd[12], Tyd[12], Tzd[12], E1xd[12], E1yd[12], E1zd[12], E2xd[12], E2yd[12], E2zd[12];
    fill_cube_soa(Txd, Tyd, Tzd, E1xd, E1yd, E1zd, E2xd, E2yd, E2zd, n_mesh);

    // Single-precision data
    float Txf[12], Tyf[12], Tzf[12], E1xf[12], E1yf[12], E1zf[12], E2xf[12], E2yf[12], E2zf[12];
    fill_cube_soa(Txf, Tyf, Tzf, E1xf, E1yf, E1zf, E2xf, E2yf, E2zf, n_mesh);

    unsigned SMId[1] = {0}, SMIf[1] = {0};

    double Xmind[1] = {-1.0}, Xmaxd[1] = {1.0}, Ymind[1] = {-1.0}, Ymaxd[1] = {1.0}, Zmind[1] = {-1.0}, Zmaxd[1] = {1.0};
    float Xminf[1] = {-1.0f}, Xmaxf[1] = {1.0f}, Yminf[1] = {-1.0f}, Ymaxf[1] = {1.0f}, Zminf[1] = {-1.0f}, Zmaxf[1] = {1.0f};

    // Same 4 rays as test 4
    double Oxd[4] = {-10.0, -10.0, 0.0, 1.0};
    double Oyd[4] = {0.0, 5.0, 0.0, -10.0};
    double Ozd[4] = {0.5, 5.0, 0.5, 0.5};
    double Dxd[4] = {20.0, 20.0, 10.0, 0.0};
    double Dyd[4] = {0.0, 0.0, 0.0, 20.0};
    double Dzd[4] = {0.0, 0.0, 0.0, 0.0};

    float Oxf[4] = {-10.0f, -10.0f, 0.0f, 1.0f};
    float Oyf[4] = {0.0f, 5.0f, 0.0f, -10.0f};
    float Ozf[4] = {0.5f, 5.0f, 0.5f, 0.5f};
    float Dxf[4] = {20.0f, 20.0f, 10.0f, 0.0f};
    float Dyf[4] = {0.0f, 0.0f, 0.0f, 20.0f};
    float Dzf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    double Wfd[4], Wsd[4];
    unsigned Ifd[4], Isd[4], hit_cntd[4];
    float Wff[4], Wsf[4];
    unsigned Iff[4], Isf[4], hit_cntf[4];

    qd_RTI_GENERIC(Txd, Tyd, Tzd, E1xd, E1yd, E1zd, E2xd, E2yd, E2zd, n_mesh,
                   SMId, Xmind, Xmaxd, Ymind, Ymaxd, Zmind, Zmaxd, n_sub,
                   Oxd, Oyd, Ozd, Dxd, Dyd, Dzd, n_ray,
                   Wfd, Wsd, Ifd, Isd, hit_cntd);

    qd_RTI_GENERIC(Txf, Tyf, Tzf, E1xf, E1yf, E1zf, E2xf, E2yf, E2zf, n_mesh,
                   SMIf, Xminf, Xmaxf, Yminf, Ymaxf, Zminf, Zmaxf, n_sub,
                   Oxf, Oyf, Ozf, Dxf, Dyf, Dzf, n_ray,
                   Wff, Wsf, Iff, Isf, hit_cntf);

    for (size_t i = 0; i < n_ray; ++i)
    {
        CHECK(std::abs((double)Wff[i] - Wfd[i]) < 1e-6);
        CHECK(std::abs((double)Wsf[i] - Wsd[i]) < 1e-6);
        CHECK(Iff[i] == Ifd[i]);
        CHECK(Isf[i] == Isd[i]);
        CHECK(hit_cntf[i] == hit_cntd[i]);
    }
}

// --- Test 8: AVX2 vs. Generic (float, padded to VEC_SIZE=8) ---
#ifdef BUILD_WITH_AVX2
TEST_CASE("RTI Internal - AVX2 comparison")
{
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        return; // Quietly skip

    const size_t n_mesh = 12;
    const size_t n_mesh_padded = 16; // Next multiple of 8 above 12
    const size_t n_sub = 1;
    const size_t n_sub_s = 8; // n_sub rounded up to 8
    const size_t n_ray = 4;

    // Mesh data, padded to 16 (extra entries are degenerate zero-area triangles)
    float Tx[16], Ty[16], Tz[16], E1x[16], E1y[16], E1z[16], E2x[16], E2y[16], E2z[16];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_padded);

    // Sub-mesh index
    unsigned SMI[1] = {0};

    // AABB arrays, padded to n_sub_s=8; slots [1,7] set to miss-guaranteeing values
    float Xmin[8], Xmax[8], Ymin[8], Ymax[8], Zmin[8], Zmax[8];
    Xmin[0] = -1.0f;
    Xmax[0] = 1.0f;
    Ymin[0] = -1.0f;
    Ymax[0] = 1.0f;
    Zmin[0] = -1.0f;
    Zmax[0] = 1.0f;
    for (size_t i = 1; i < n_sub_s; ++i)
    {
        Xmin[i] = 1e20f;
        Xmax[i] = -1e20f;
        Ymin[i] = 1e20f;
        Ymax[i] = -1e20f;
        Zmin[i] = 1e20f;
        Zmax[i] = -1e20f;
    }

    // Same 4 rays as tests 4 and 7
    float Ox[4] = {-10.0f, -10.0f, 0.0f, 1.0f};
    float Oy[4] = {0.0f, 5.0f, 0.0f, -10.0f};
    float Oz[4] = {0.5f, 5.0f, 0.5f, 0.5f};
    float Dx[4] = {20.0f, 20.0f, 10.0f, 0.0f};
    float Dy[4] = {0.0f, 0.0f, 0.0f, 20.0f};
    float Dz[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Generic float reference
    float Wf_ref[4], Ws_ref[4];
    unsigned If_ref[4], Is_ref[4], hit_cnt_ref[4];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_padded,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf_ref, Ws_ref, If_ref, Is_ref, hit_cnt_ref);

    // AVX2
    float Wf_avx[4], Ws_avx[4];
    unsigned If_avx[4], Is_avx[4], hit_cnt_avx[4];

    qd_RTI_AVX2(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_padded,
                SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                Wf_avx, Ws_avx, If_avx, Is_avx, hit_cnt_avx);

    for (size_t i = 0; i < n_ray; ++i)
    {
        CHECK(std::abs(Wf_avx[i] - Wf_ref[i]) < 1e-6f);
        CHECK(std::abs(Ws_avx[i] - Ws_ref[i]) < 1e-6f);
        CHECK(If_avx[i] == If_ref[i]);
        CHECK(Is_avx[i] == Is_ref[i]);
        CHECK(hit_cnt_avx[i] == hit_cnt_ref[i]);
    }
}
#endif

// --- Test 9: CUDA vs. Generic (float, padded to VEC_SIZE=8) ---
#ifdef BUILD_WITH_CUDA
TEST_CASE("RTI Internal - CUDA comparison")
{
    if (!quadriga_lib::quadriga_lib_has_CUDA())
        return; // Quietly skip

    const size_t n_mesh_padded = 16;
    const size_t n_sub = 1;
    const size_t n_sub_s = 8;
    const size_t n_ray = 4;

    float Tx[16], Ty[16], Tz[16], E1x[16], E1y[16], E1z[16], E2x[16], E2y[16], E2z[16];
    fill_cube_soa(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_padded);

    unsigned SMI[1] = {0};
    float Xmin[8], Xmax[8], Ymin[8], Ymax[8], Zmin[8], Zmax[8];
    Xmin[0] = -1.0f;
    Xmax[0] = 1.0f;
    Ymin[0] = -1.0f;
    Ymax[0] = 1.0f;
    Zmin[0] = -1.0f;
    Zmax[0] = 1.0f;
    for (size_t i = 1; i < n_sub_s; ++i)
    {
        Xmin[i] = 1e20f;
        Xmax[i] = -1e20f;
        Ymin[i] = 1e20f;
        Ymax[i] = -1e20f;
        Zmin[i] = 1e20f;
        Zmax[i] = -1e20f;
    }

    float Ox[4] = {-10.0f, -10.0f, 0.0f, 1.0f};
    float Oy[4] = {0.0f, 5.0f, 0.0f, -10.0f};
    float Oz[4] = {0.5f, 5.0f, 0.5f, 0.5f};
    float Dx[4] = {20.0f, 20.0f, 10.0f, 0.0f};
    float Dy[4] = {0.0f, 0.0f, 0.0f, 20.0f};
    float Dz[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Generic float reference
    float Wf_ref[4], Ws_ref[4];
    unsigned If_ref[4], Is_ref[4], hit_cnt_ref[4];

    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_padded,
                   SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                   Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                   Wf_ref, Ws_ref, If_ref, Is_ref, hit_cnt_ref);

    // CUDA
    float Wf_cuda[4], Ws_cuda[4];
    unsigned If_cuda[4], Is_cuda[4], hit_cnt_cuda[4];

    qd_RTI_CUDA(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_padded,
                SMI, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                Ox, Oy, Oz, Dx, Dy, Dz, n_ray,
                Wf_cuda, Ws_cuda, If_cuda, Is_cuda, hit_cnt_cuda);

    for (size_t i = 0; i < n_ray; ++i)
    {
        CHECK(std::abs(Wf_cuda[i] - Wf_ref[i]) < 1e-6f);
        CHECK(std::abs(Ws_cuda[i] - Ws_ref[i]) < 1e-6f);
        CHECK(If_cuda[i] == If_ref[i]);
        CHECK(Is_cuda[i] == Is_ref[i]);
        CHECK(hit_cnt_cuda[i] == hit_cnt_ref[i]);
    }
}
#endif