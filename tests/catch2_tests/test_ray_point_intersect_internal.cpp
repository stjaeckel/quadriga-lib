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
#include "quadriga_tools.hpp"
#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"

#ifdef BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

#ifdef BUILD_WITH_CUDA
#include "quadriga_lib_cuda_functions.hpp"
#endif

// --- Helper: Compute ray tube SoA data from icosphere outputs ---
// Converts the matrix-based icosphere outputs (orig, trivec, tridir) into the
// separate SoA arrays needed by the internal qd_RPI functions.
template <typename dtype>
static void compute_ray_tube_data(const arma::Mat<dtype> &orig,    // [n_ray, 3] center vectors
                                  const arma::Mat<dtype> &trivec,  // [n_ray, 9] vertex offsets from center
                                  const arma::Mat<dtype> &tridir,  // [n_ray, 9] Cartesian ray directions
                                  arma::Col<dtype> &T1x, arma::Col<dtype> &T1y, arma::Col<dtype> &T1z,
                                  arma::Col<dtype> &T2x, arma::Col<dtype> &T2y, arma::Col<dtype> &T2z,
                                  arma::Col<dtype> &T3x, arma::Col<dtype> &T3y, arma::Col<dtype> &T3z,
                                  arma::Col<dtype> &Nx, arma::Col<dtype> &Ny, arma::Col<dtype> &Nz,
                                  arma::Col<dtype> &D1x, arma::Col<dtype> &D1y, arma::Col<dtype> &D1z,
                                  arma::Col<dtype> &D2x, arma::Col<dtype> &D2y, arma::Col<dtype> &D2z,
                                  arma::Col<dtype> &D3x, arma::Col<dtype> &D3y, arma::Col<dtype> &D3z,
                                  arma::Col<dtype> &rD1, arma::Col<dtype> &rD2, arma::Col<dtype> &rD3)
{
    // Vertex positions in GCS = center + vertex offset
    T1x = orig.col(0) + trivec.col(0);
    T1y = orig.col(1) + trivec.col(1);
    T1z = orig.col(2) + trivec.col(2);
    T2x = orig.col(0) + trivec.col(3);
    T2y = orig.col(1) + trivec.col(4);
    T2z = orig.col(2) + trivec.col(5);
    T3x = orig.col(0) + trivec.col(6);
    T3y = orig.col(1) + trivec.col(7);
    T3z = orig.col(2) + trivec.col(8);

    // Ray directions from tridir (Cartesian)
    D1x = tridir.col(0); D1y = tridir.col(1); D1z = tridir.col(2);
    D2x = tridir.col(3); D2y = tridir.col(4); D2z = tridir.col(5);
    D3x = tridir.col(6); D3y = tridir.col(7); D3z = tridir.col(8);

    // Face normal = cross(edge1, edge2) where edge1 = T2-T1, edge2 = T3-T1
    arma::Col<dtype> e1x = T2x - T1x, e1y = T2y - T1y, e1z = T2z - T1z;
    arma::Col<dtype> e2x = T3x - T1x, e2y = T3y - T1y, e2z = T3z - T1z;
    Nx = e1y % e2z - e1z % e2y;
    Ny = e1z % e2x - e1x % e2z;
    Nz = e1x % e2y - e1y % e2x;

    // Inverse dot product of each ray direction with the face normal
    rD1 = dtype(1.0) / (D1x % Nx + D1y % Ny + D1z % Nz);
    rD2 = dtype(1.0) / (D2x % Nx + D2y % Ny + D2z % Nz);
    rD3 = dtype(1.0) / (D3x % Nx + D3y % Ny + D3z % Nz);
}

// --- Helper: Compute AABBs from pointsR and sub_cloud_index ---
// Allocates AABB arrays of size n_sub_s (padded to vec_size) and fills them
// by scanning points within each sub-cloud range.
template <typename dtype>
static void compute_aabb(const arma::Mat<dtype> &pointsR,
                         const arma::u32_vec &sci,
                         size_t n_sub, size_t n_sub_s,
                         arma::Col<dtype> &Xmin, arma::Col<dtype> &Xmax,
                         arma::Col<dtype> &Ymin, arma::Col<dtype> &Ymax,
                         arma::Col<dtype> &Zmin, arma::Col<dtype> &Zmax)
{
    size_t n_point = pointsR.n_rows;
    Xmin.zeros(n_sub_s); Xmax.zeros(n_sub_s);
    Ymin.zeros(n_sub_s); Ymax.zeros(n_sub_s);
    Zmin.zeros(n_sub_s); Zmax.zeros(n_sub_s);

    const dtype *px = pointsR.colptr(0);
    const dtype *py = pointsR.colptr(1);
    const dtype *pz = pointsR.colptr(2);

    for (size_t s = 0; s < n_sub; ++s)
    {
        size_t start = (size_t)sci.at(s);
        size_t end = (s + 1 < n_sub) ? (size_t)sci.at(s + 1) : n_point;

        dtype xmin = px[start], xmax = px[start];
        dtype ymin = py[start], ymax = py[start];
        dtype zmin = pz[start], zmax = pz[start];

        for (size_t i = start + 1; i < end; ++i)
        {
            if (px[i] < xmin) xmin = px[i];
            if (px[i] > xmax) xmax = px[i];
            if (py[i] < ymin) ymin = py[i];
            if (py[i] > ymax) ymax = py[i];
            if (pz[i] < zmin) zmin = pz[i];
            if (pz[i] > zmax) zmax = pz[i];
        }

        Xmin.at(s) = xmin; Xmax.at(s) = xmax;
        Ymin.at(s) = ymin; Ymax.at(s) = ymax;
        Zmin.at(s) = zmin; Zmax.at(s) = zmax;
    }
}

// --- Helper: Count hits per point from the per-ray hit lists ---
// Returns a vector of length n_point where each element counts how many rays hit that point.
static arma::u32_vec count_hits_per_point(const std::vector<unsigned> *p_hit, size_t n_ray, size_t n_point)
{
    arma::u32_vec hit_count(n_point, arma::fill::zeros);
    for (size_t r = 0; r < n_ray; ++r)
        for (unsigned idx : p_hit[r])
            if (idx < (unsigned)n_point)
                hit_count.at(idx)++;
    return hit_count;
}

// ============================================================================
// Test Case 1: Simple Mode - float, GENERIC
// Mirrors the public API "Simple Mode" test but calls qd_RPI_GENERIC directly.
// ============================================================================
TEST_CASE("RPI Internal - Simple Mode (float, GENERIC)")
{
    // Generate set of points (identical to public API test)
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    // Create sub-cloud index with vec_size=8
    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    arma::uword n_sub = quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8);
    size_t n_point = pointsR.n_rows;
    size_t n_sub_s = ((n_sub + 7) / 8) * 8;

    // Compute AABBs
    arma::fvec Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    compute_aabb(pointsR, sub_cloud_index, n_sub, n_sub_s, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);

    // Generate ray beams from icosphere
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;
    size_t n_ray = orig.n_rows;

    // Compute ray tube SoA data
    arma::fvec T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    arma::fvec Nx, Ny, Nz;
    arma::fvec D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    arma::fvec rD1, rD2, rD3;
    compute_ray_tube_data(orig, trivec, tridir,
                          T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                          Nx, Ny, Nz,
                          D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z,
                          rD1, rD2, rD3);

    // Allocate output
    std::vector<std::vector<unsigned>> hit_lists(n_ray);

    // Call GENERIC implementation
    qd_RPI_GENERIC<float>(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                          sub_cloud_index.memptr(),
                          Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                          n_sub,
                          T1x.memptr(), T1y.memptr(), T1z.memptr(),
                          T2x.memptr(), T2y.memptr(), T2z.memptr(),
                          T3x.memptr(), T3y.memptr(), T3z.memptr(),
                          Nx.memptr(), Ny.memptr(), Nz.memptr(),
                          D1x.memptr(), D1y.memptr(), D1z.memptr(),
                          D2x.memptr(), D2y.memptr(), D2z.memptr(),
                          D3x.memptr(), D3y.memptr(), D3z.memptr(),
                          rD1.memptr(), rD2.memptr(), rD3.memptr(),
                          n_ray, hit_lists.data());

    // Every point should be hit exactly once
    arma::u32_vec hit_count = count_hits_per_point(hit_lists.data(), n_ray, n_point);

    // The original 16 points must each be hit; padded points may or may not be hit
    // Use reverse_index to check only the original points
    arma::u32_vec reverse_index;
    quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8, nullptr, &reverse_index);
    for (arma::uword i = 0; i < points.n_rows; ++i)
        CHECK(hit_count.at(reverse_index.at(i)) >= 1);
}

// ============================================================================
// Test Case 2: Simple Mode - double, GENERIC
// Validates the double-precision path.
// ============================================================================
TEST_CASE("RPI Internal - Simple Mode (double, GENERIC)")
{
    // Generate set of points (same layout, double precision)
    // Note: arma::regspace<vec>(0.0, 0.1, 0.3) can produce only 3 elements in double precision
    // because 3*0.1 evaluates to 0.30000000000000004 > 0.3. Use linspace instead.
    arma::mat points(4, 3);
    points.col(0) = arma::linspace<arma::vec>(0.0, 0.3, 4);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0;
    points.submat(8, 1, 15, 1) += 50.0;
    points.col(2) += 1.0;
    points.col(1) *= 0.1;

    // Single sub-cloud (no segmentation needed for 16 points)
    arma::u32_vec sci(1);
    sci.at(0) = 0;
    size_t n_sub = 1;
    size_t n_sub_s = 8; // Padded to VEC_SIZE

    // Compute AABB for the single sub-cloud
    arma::vec Xmin(n_sub_s, arma::fill::zeros), Xmax(n_sub_s, arma::fill::zeros);
    arma::vec Ymin(n_sub_s, arma::fill::zeros), Ymax(n_sub_s, arma::fill::zeros);
    arma::vec Zmin(n_sub_s, arma::fill::zeros), Zmax(n_sub_s, arma::fill::zeros);
    Xmin.at(0) = points.col(0).min(); Xmax.at(0) = points.col(0).max();
    Ymin.at(0) = points.col(1).min(); Ymax.at(0) = points.col(1).max();
    Zmin.at(0) = points.col(2).min(); Zmax.at(0) = points.col(2).max();

    // Generate ray beams
    arma::mat orig, trivec, tridir;
    quadriga_lib::icosphere<double>(2, 1.0, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0;
    orig.col(1) -= 20.0;
    orig.col(2) -= 30.0;
    size_t n_ray = orig.n_rows;
    size_t n_point = points.n_rows;

    // Compute ray tube SoA data
    arma::vec T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    arma::vec Nx, Ny, Nz;
    arma::vec D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    arma::vec rD1, rD2, rD3;
    compute_ray_tube_data(orig, trivec, tridir,
                          T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                          Nx, Ny, Nz,
                          D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z,
                          rD1, rD2, rD3);

    // Allocate output and call
    std::vector<std::vector<unsigned>> hit_lists(n_ray);
    qd_RPI_GENERIC<double>(points.colptr(0), points.colptr(1), points.colptr(2), n_point,
                           sci.memptr(),
                           Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                           n_sub,
                           T1x.memptr(), T1y.memptr(), T1z.memptr(),
                           T2x.memptr(), T2y.memptr(), T2z.memptr(),
                           T3x.memptr(), T3y.memptr(), T3z.memptr(),
                           Nx.memptr(), Ny.memptr(), Nz.memptr(),
                           D1x.memptr(), D1y.memptr(), D1z.memptr(),
                           D2x.memptr(), D2y.memptr(), D2z.memptr(),
                           D3x.memptr(), D3y.memptr(), D3z.memptr(),
                           rD1.memptr(), rD2.memptr(), rD3.memptr(),
                           n_ray, hit_lists.data());

    // Every point should be hit exactly once
    arma::u32_vec hit_count = count_hits_per_point(hit_lists.data(), n_ray, n_point);
    CHECK(arma::all(hit_count == 1));
}

// ============================================================================
// Test Case 3: AVX2 vs GENERIC - Simple Mode
// Compares hit lists element-by-element. Skipped if AVX2 not available.
// ============================================================================
TEST_CASE("RPI Internal - Simple Mode (AVX2 vs GENERIC)")
{
#ifdef BUILD_WITH_AVX2
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        return; // Silently skip

    // Generate points with sub-cloud segmentation (vec_size=8 for AVX2)
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    arma::uword n_sub = quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8);
    size_t n_point = pointsR.n_rows;
    size_t n_sub_s = ((n_sub + 7) / 8) * 8;

    arma::fvec Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    compute_aabb(pointsR, sub_cloud_index, n_sub, n_sub_s, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);

    // Generate ray beams
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;
    size_t n_ray = orig.n_rows;

    arma::fvec T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    arma::fvec Nx, Ny, Nz;
    arma::fvec D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    arma::fvec rD1, rD2, rD3;
    compute_ray_tube_data(orig, trivec, tridir,
                          T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                          Nx, Ny, Nz,
                          D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z,
                          rD1, rD2, rD3);

    // Run GENERIC
    std::vector<std::vector<unsigned>> hit_generic(n_ray);
    qd_RPI_GENERIC<float>(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                          sub_cloud_index.memptr(),
                          Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                          n_sub,
                          T1x.memptr(), T1y.memptr(), T1z.memptr(),
                          T2x.memptr(), T2y.memptr(), T2z.memptr(),
                          T3x.memptr(), T3y.memptr(), T3z.memptr(),
                          Nx.memptr(), Ny.memptr(), Nz.memptr(),
                          D1x.memptr(), D1y.memptr(), D1z.memptr(),
                          D2x.memptr(), D2y.memptr(), D2z.memptr(),
                          D3x.memptr(), D3y.memptr(), D3z.memptr(),
                          rD1.memptr(), rD2.memptr(), rD3.memptr(),
                          n_ray, hit_generic.data());

    // Run AVX2
    std::vector<std::vector<unsigned>> hit_avx2(n_ray);
    qd_RPI_AVX2(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                sub_cloud_index.memptr(),
                Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                n_sub,
                T1x.memptr(), T1y.memptr(), T1z.memptr(),
                T2x.memptr(), T2y.memptr(), T2z.memptr(),
                T3x.memptr(), T3y.memptr(), T3z.memptr(),
                Nx.memptr(), Ny.memptr(), Nz.memptr(),
                D1x.memptr(), D1y.memptr(), D1z.memptr(),
                D2x.memptr(), D2y.memptr(), D2z.memptr(),
                D3x.memptr(), D3y.memptr(), D3z.memptr(),
                rD1.memptr(), rD2.memptr(), rD3.memptr(),
                n_ray, hit_avx2.data());

    // Compare: same hit counts per point
    arma::u32_vec hc_generic = count_hits_per_point(hit_generic.data(), n_ray, n_point);
    arma::u32_vec hc_avx2 = count_hits_per_point(hit_avx2.data(), n_ray, n_point);
    CHECK(arma::all(hc_generic == hc_avx2));

    // Compare: per-ray hit lists (sorted, since order within a ray may differ)
    for (size_t r = 0; r < n_ray; ++r)
    {
        auto g = hit_generic[r];
        auto a = hit_avx2[r];
        std::sort(g.begin(), g.end());
        std::sort(a.begin(), a.end());
        CHECK(g == a);
    }
#endif
}

// ============================================================================
// Test Case 4: CUDA vs GENERIC - Simple Mode
// Same comparison pattern as AVX2. Skipped if CUDA not available.
// ============================================================================
TEST_CASE("RPI Internal - Simple Mode (CUDA vs GENERIC)")
{
#ifdef BUILD_WITH_CUDA
    if (!quadriga_lib::quadriga_lib_has_CUDA())
        return; // Silently skip

    // Generate points with sub-cloud segmentation (vec_size=8 for CUDA)
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    arma::uword n_sub = quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8);
    size_t n_point = pointsR.n_rows;
    size_t n_sub_s = ((n_sub + 7) / 8) * 8;

    arma::fvec Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    compute_aabb(pointsR, sub_cloud_index, n_sub, n_sub_s, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);

    // Generate ray beams
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;
    size_t n_ray = orig.n_rows;

    arma::fvec T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    arma::fvec Nx, Ny, Nz;
    arma::fvec D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    arma::fvec rD1, rD2, rD3;
    compute_ray_tube_data(orig, trivec, tridir,
                          T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                          Nx, Ny, Nz,
                          D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z,
                          rD1, rD2, rD3);

    // Run GENERIC
    std::vector<std::vector<unsigned>> hit_generic(n_ray);
    qd_RPI_GENERIC<float>(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                          sub_cloud_index.memptr(),
                          Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                          n_sub,
                          T1x.memptr(), T1y.memptr(), T1z.memptr(),
                          T2x.memptr(), T2y.memptr(), T2z.memptr(),
                          T3x.memptr(), T3y.memptr(), T3z.memptr(),
                          Nx.memptr(), Ny.memptr(), Nz.memptr(),
                          D1x.memptr(), D1y.memptr(), D1z.memptr(),
                          D2x.memptr(), D2y.memptr(), D2z.memptr(),
                          D3x.memptr(), D3y.memptr(), D3z.memptr(),
                          rD1.memptr(), rD2.memptr(), rD3.memptr(),
                          n_ray, hit_generic.data());

    // Run CUDA
    std::vector<std::vector<unsigned>> hit_cuda(n_ray);
    qd_RPI_CUDA(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                sub_cloud_index.memptr(),
                Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                n_sub,
                T1x.memptr(), T1y.memptr(), T1z.memptr(),
                T2x.memptr(), T2y.memptr(), T2z.memptr(),
                T3x.memptr(), T3y.memptr(), T3z.memptr(),
                Nx.memptr(), Ny.memptr(), Nz.memptr(),
                D1x.memptr(), D1y.memptr(), D1z.memptr(),
                D2x.memptr(), D2y.memptr(), D2z.memptr(),
                D3x.memptr(), D3y.memptr(), D3z.memptr(),
                rD1.memptr(), rD2.memptr(), rD3.memptr(),
                n_ray, hit_cuda.data());

    // Compare per-point hit counts
    arma::u32_vec hc_generic = count_hits_per_point(hit_generic.data(), n_ray, n_point);
    arma::u32_vec hc_cuda = count_hits_per_point(hit_cuda.data(), n_ray, n_point);
    CHECK(arma::all(hc_generic == hc_cuda));

    // Compare per-ray hit lists (sorted)
    for (size_t r = 0; r < n_ray; ++r)
    {
        auto g = hit_generic[r];
        auto c = hit_cuda[r];
        std::sort(g.begin(), g.end());
        std::sort(c.begin(), c.end());
        CHECK(g == c);
    }
#endif
}

// ============================================================================
// Test Case 5: Dense Point Grid - double, GENERIC (no sub-clouds)
// A coarser version of the public "Ray Subdivision" test, using a single
// sub-cloud (no segmentation) to exercise that code path.
// ============================================================================
TEST_CASE("RPI Internal - Dense Grid (double, GENERIC)")
{
    // Generate a 2D grid of points at z=0.1
    double res = 0.5;
    arma::vec x = arma::regspace(-10.0, res, 10.0);
    arma::vec y = arma::regspace(-10.0, res, 10.0);

    arma::mat X(y.n_elem, x.n_elem);
    arma::mat Y(y.n_elem, x.n_elem);
    for (arma::uword i = 0; i < y.n_elem; ++i)
        X.row(i) = x.t();
    for (arma::uword j = 0; j < x.n_elem; ++j)
        Y.col(j) = y;

    arma::mat points(X.n_elem, 3);
    points.col(0) = arma::vectorise(X);
    points.col(1) = arma::vectorise(Y);
    points.col(2).fill(0.1);
    size_t n_point = points.n_rows;

    // Single sub-cloud covering all points
    arma::u32_vec sci(1);
    sci.at(0) = 0;
    size_t n_sub = 1, n_sub_s = 8;

    arma::vec Xmin(n_sub_s, arma::fill::zeros), Xmax(n_sub_s, arma::fill::zeros);
    arma::vec Ymin(n_sub_s, arma::fill::zeros), Ymax(n_sub_s, arma::fill::zeros);
    arma::vec Zmin(n_sub_s, arma::fill::zeros), Zmax(n_sub_s, arma::fill::zeros);
    Xmin.at(0) = points.col(0).min(); Xmax.at(0) = points.col(0).max();
    Ymin.at(0) = points.col(1).min(); Ymax.at(0) = points.col(1).max();
    Zmin.at(0) = points.col(2).min(); Zmax.at(0) = points.col(2).max();

    // Generate ray beams (icosphere with enough resolution to cover the grid)
    arma::mat orig, trivec, tridir;
    quadriga_lib::icosphere<double>(8, 1.0, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0;
    orig.col(1) -= 20.0;
    orig.col(2) -= 30.0;
    size_t n_ray = orig.n_rows;

    arma::vec T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    arma::vec Nx, Ny, Nz;
    arma::vec D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    arma::vec rD1, rD2, rD3;
    compute_ray_tube_data(orig, trivec, tridir,
                          T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                          Nx, Ny, Nz,
                          D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z,
                          rD1, rD2, rD3);

    // Call GENERIC
    std::vector<std::vector<unsigned>> hit_lists(n_ray);
    qd_RPI_GENERIC<double>(points.colptr(0), points.colptr(1), points.colptr(2), n_point,
                           sci.memptr(),
                           Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                           n_sub,
                           T1x.memptr(), T1y.memptr(), T1z.memptr(),
                           T2x.memptr(), T2y.memptr(), T2z.memptr(),
                           T3x.memptr(), T3y.memptr(), T3z.memptr(),
                           Nx.memptr(), Ny.memptr(), Nz.memptr(),
                           D1x.memptr(), D1y.memptr(), D1z.memptr(),
                           D2x.memptr(), D2y.memptr(), D2z.memptr(),
                           D3x.memptr(), D3y.memptr(), D3z.memptr(),
                           rD1.memptr(), rD2.memptr(), rD3.memptr(),
                           n_ray, hit_lists.data());

    // Every point should be hit exactly once
    arma::u32_vec hit_count = count_hits_per_point(hit_lists.data(), n_ray, n_point);
    CHECK(arma::all(hit_count == 1));
}

// ============================================================================
// Test Case 6: Dense Point Grid - AVX2 vs GENERIC
// Uses segmented sub-clouds with vec_size=8.
// ============================================================================
TEST_CASE("RPI Internal - Dense Grid (AVX2 vs GENERIC)")
{
#ifdef BUILD_WITH_AVX2
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        return;

    // Generate a 2D grid of points at z=0.1
    float res = 0.5f;
    arma::fvec x = arma::regspace<arma::fvec>(-10.0f, res, 10.0f);
    arma::fvec y = arma::regspace<arma::fvec>(-10.0f, res, 10.0f);

    arma::fmat X(y.n_elem, x.n_elem);
    arma::fmat Y(y.n_elem, x.n_elem);
    for (arma::uword i = 0; i < y.n_elem; ++i)
        X.row(i) = x.t();
    for (arma::uword j = 0; j < x.n_elem; ++j)
        Y.col(j) = y;

    arma::fmat points(X.n_elem, 3);
    points.col(0) = arma::vectorise(X);
    points.col(1) = arma::vectorise(Y);
    points.col(2).fill(0.1f);

    // Segment with vec_size=8
    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    arma::uword n_sub = quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 128, 8);
    size_t n_point = pointsR.n_rows;
    size_t n_sub_s = ((n_sub + 7) / 8) * 8;

    arma::fvec Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    compute_aabb(pointsR, sub_cloud_index, n_sub, n_sub_s, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);

    // Generate ray beams
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(8, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;
    size_t n_ray = orig.n_rows;

    arma::fvec T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    arma::fvec Nx, Ny, Nz;
    arma::fvec D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    arma::fvec rD1, rD2, rD3;
    compute_ray_tube_data(orig, trivec, tridir,
                          T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                          Nx, Ny, Nz,
                          D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z,
                          rD1, rD2, rD3);

    // Run GENERIC
    std::vector<std::vector<unsigned>> hit_generic(n_ray);
    qd_RPI_GENERIC<float>(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                          sub_cloud_index.memptr(),
                          Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                          n_sub,
                          T1x.memptr(), T1y.memptr(), T1z.memptr(),
                          T2x.memptr(), T2y.memptr(), T2z.memptr(),
                          T3x.memptr(), T3y.memptr(), T3z.memptr(),
                          Nx.memptr(), Ny.memptr(), Nz.memptr(),
                          D1x.memptr(), D1y.memptr(), D1z.memptr(),
                          D2x.memptr(), D2y.memptr(), D2z.memptr(),
                          D3x.memptr(), D3y.memptr(), D3z.memptr(),
                          rD1.memptr(), rD2.memptr(), rD3.memptr(),
                          n_ray, hit_generic.data());

    // Run AVX2
    std::vector<std::vector<unsigned>> hit_avx2(n_ray);
    qd_RPI_AVX2(pointsR.colptr(0), pointsR.colptr(1), pointsR.colptr(2), n_point,
                sub_cloud_index.memptr(),
                Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                n_sub,
                T1x.memptr(), T1y.memptr(), T1z.memptr(),
                T2x.memptr(), T2y.memptr(), T2z.memptr(),
                T3x.memptr(), T3y.memptr(), T3z.memptr(),
                Nx.memptr(), Ny.memptr(), Nz.memptr(),
                D1x.memptr(), D1y.memptr(), D1z.memptr(),
                D2x.memptr(), D2y.memptr(), D2z.memptr(),
                D3x.memptr(), D3y.memptr(), D3z.memptr(),
                rD1.memptr(), rD2.memptr(), rD3.memptr(),
                n_ray, hit_avx2.data());

    // Compare per-point hit counts
    arma::u32_vec hc_generic = count_hits_per_point(hit_generic.data(), n_ray, n_point);
    arma::u32_vec hc_avx2 = count_hits_per_point(hit_avx2.data(), n_ray, n_point);
    CHECK(arma::all(hc_generic == hc_avx2));

    // Compare per-ray hit lists (sorted)
    for (size_t r = 0; r < n_ray; ++r)
    {
        auto g = hit_generic[r];
        auto a = hit_avx2[r];
        std::sort(g.begin(), g.end());
        std::sort(a.begin(), a.end());
        CHECK(g == a);
    }
#endif
}

// ============================================================================
// Test Case 7: No Hits - points far from ray coverage
// A single ray tube pointing in +z has a fixed cross-section (~1 unit wide).
// Points placed far off to the side (x=1000) are outside this tube and the
// AABB pre-filter should reject the sub-cloud entirely.
// NOTE: The original test used a full icosphere, but a full icosphere tiles
// the entire sphere — every external point is inside exactly one ray tube,
// so "no hits" is impossible with omnidirectional coverage.
// ============================================================================
TEST_CASE("RPI Internal - No Hits")
{
    // Points far off to the side (x = 1000..1003, y = 1000, z = 1000)
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(1000.0f, 1.0f, 1003.0f);
    points.col(1).fill(1000.0f);
    points.col(2).fill(1000.0f);
    size_t n_point = points.n_rows;

    // Single sub-cloud
    arma::u32_vec sci(1);
    sci.at(0) = 0;
    size_t n_sub = 1, n_sub_s = 8;

    arma::fvec Xmin(n_sub_s, arma::fill::zeros), Xmax(n_sub_s, arma::fill::zeros);
    arma::fvec Ymin(n_sub_s, arma::fill::zeros), Ymax(n_sub_s, arma::fill::zeros);
    arma::fvec Zmin(n_sub_s, arma::fill::zeros), Zmax(n_sub_s, arma::fill::zeros);
    Xmin.at(0) = points.col(0).min(); Xmax.at(0) = points.col(0).max();
    Ymin.at(0) = points.col(1).min(); Ymax.at(0) = points.col(1).max();
    Zmin.at(0) = points.col(2).min(); Zmax.at(0) = points.col(2).max();

    // Construct a single ray tube: equilateral triangle at z=0, all rays pointing +z.
    // With parallel rays the cross-section stays constant at ~1 unit wide,
    // so points at x=1000 are well outside.
    size_t n_ray = 1;
    float s = 0.5f;
    float s32 = s * std::sqrt(3.0f) / 2.0f; // s * sqrt(3)/2

    arma::fvec T1x = {s},     T1y = {0.0f},  T1z = {0.0f};
    arma::fvec T2x = {-s/2},  T2y = {s32},   T2z = {0.0f};
    arma::fvec T3x = {-s/2},  T3y = {-s32},  T3z = {0.0f};

    arma::fvec D1x = {0.0f},  D1y = {0.0f},  D1z = {1.0f};
    arma::fvec D2x = {0.0f},  D2y = {0.0f},  D2z = {1.0f};
    arma::fvec D3x = {0.0f},  D3y = {0.0f},  D3z = {1.0f};

    // Face normal = cross(T2-T1, T3-T1); only z-component is nonzero for a z=0 triangle
    arma::fvec e1x = T2x - T1x, e1y = T2y - T1y, e1z = T2z - T1z;
    arma::fvec e2x = T3x - T1x, e2y = T3y - T1y, e2z = T3z - T1z;
    arma::fvec Nx = e1y % e2z - e1z % e2y;  // 0
    arma::fvec Ny = e1z % e2x - e1x % e2z;  // 0
    arma::fvec Nz = e1x % e2y - e1y % e2x;  // positive

    arma::fvec rD1 = 1.0f / (D1x % Nx + D1y % Ny + D1z % Nz);
    arma::fvec rD2 = 1.0f / (D2x % Nx + D2y % Ny + D2z % Nz);
    arma::fvec rD3 = 1.0f / (D3x % Nx + D3y % Ny + D3z % Nz);

    std::vector<std::vector<unsigned>> hit_lists(n_ray);
    qd_RPI_GENERIC<float>(points.colptr(0), points.colptr(1), points.colptr(2), n_point,
                          sci.memptr(),
                          Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                          n_sub,
                          T1x.memptr(), T1y.memptr(), T1z.memptr(),
                          T2x.memptr(), T2y.memptr(), T2z.memptr(),
                          T3x.memptr(), T3y.memptr(), T3z.memptr(),
                          Nx.memptr(), Ny.memptr(), Nz.memptr(),
                          D1x.memptr(), D1y.memptr(), D1z.memptr(),
                          D2x.memptr(), D2y.memptr(), D2z.memptr(),
                          D3x.memptr(), D3y.memptr(), D3z.memptr(),
                          rD1.memptr(), rD2.memptr(), rD3.memptr(),
                          n_ray, hit_lists.data());

    // All hit lists should be empty
    for (size_t r = 0; r < n_ray; ++r)
        CHECK(hit_lists[r].empty());
}

// ============================================================================
// Test Case 8: Sub-cloud AABB Filtering
// Two widely separated point clusters. A single ray tube pointing +z covers
// cluster A (near origin) but not cluster B (far off to the side).
// Confirms the AABB pre-filter correctly skips cluster B.
// NOTE: The original test used a full icosphere, but a full icosphere tiles
// the entire sphere — every external point is inside exactly one ray tube,
// so "cluster B not hit" is impossible with omnidirectional coverage.
// ============================================================================
TEST_CASE("RPI Internal - Sub-cloud AABB Filtering")
{
    // Cluster A: centered at (0, 0, 1) — directly ahead of the +z ray tube
    arma::fmat clusterA(8, 3);
    clusterA.col(0) = arma::regspace<arma::fvec>(-0.035f, 0.01f, 0.035f);
    clusterA.col(1) = arma::regspace<arma::fvec>(-0.035f, 0.01f, 0.035f);
    clusterA.col(2).fill(1.0f);

    // Cluster B: centered at (500, 500, 1) — far off to the side, outside the tube
    arma::fmat clusterB(8, 3);
    clusterB.col(0) = arma::regspace<arma::fvec>(499.965f, 0.01f, 500.035f);
    clusterB.col(1) = arma::regspace<arma::fvec>(499.965f, 0.01f, 500.035f);
    clusterB.col(2).fill(1.0f);

    // Combine into one point set, cluster A first, then cluster B
    arma::fmat points = arma::join_cols(clusterA, clusterB);
    size_t n_point = points.n_rows;

    // Two sub-clouds: SCI = {0, 8}
    arma::u32_vec sci(2);
    sci.at(0) = 0;
    sci.at(1) = 8;
    size_t n_sub = 2, n_sub_s = 8;

    // Compute AABBs
    arma::fvec Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    compute_aabb(points, sci, n_sub, n_sub_s, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax);

    // Sanity check: AABBs are widely separated
    CHECK(Xmax.at(0) < 1.0f);
    CHECK(Xmin.at(1) > 499.0f);

    // Construct a single ray tube: equilateral triangle at z=0, all rays pointing +z.
    // Cross-section is ~1 unit wide (constant for parallel rays), so cluster A
    // at (0,0,1) is inside, and cluster B at (500,500,1) is far outside.
    size_t n_ray = 1;
    float s = 0.5f;
    float s32 = s * std::sqrt(3.0f) / 2.0f;

    arma::fvec T1x = {s},     T1y = {0.0f},  T1z = {0.0f};
    arma::fvec T2x = {-s/2},  T2y = {s32},   T2z = {0.0f};
    arma::fvec T3x = {-s/2},  T3y = {-s32},  T3z = {0.0f};

    arma::fvec D1x = {0.0f},  D1y = {0.0f},  D1z = {1.0f};
    arma::fvec D2x = {0.0f},  D2y = {0.0f},  D2z = {1.0f};
    arma::fvec D3x = {0.0f},  D3y = {0.0f},  D3z = {1.0f};

    arma::fvec e1x = T2x - T1x, e1y = T2y - T1y, e1z = T2z - T1z;
    arma::fvec e2x = T3x - T1x, e2y = T3y - T1y, e2z = T3z - T1z;
    arma::fvec Nx = e1y % e2z - e1z % e2y;
    arma::fvec Ny = e1z % e2x - e1x % e2z;
    arma::fvec Nz = e1x % e2y - e1y % e2x;

    arma::fvec rD1 = 1.0f / (D1x % Nx + D1y % Ny + D1z % Nz);
    arma::fvec rD2 = 1.0f / (D2x % Nx + D2y % Ny + D2z % Nz);
    arma::fvec rD3 = 1.0f / (D3x % Nx + D3y % Ny + D3z % Nz);

    std::vector<std::vector<unsigned>> hit_lists(n_ray);
    qd_RPI_GENERIC<float>(points.colptr(0), points.colptr(1), points.colptr(2), n_point,
                          sci.memptr(),
                          Xmin.memptr(), Xmax.memptr(), Ymin.memptr(), Ymax.memptr(), Zmin.memptr(), Zmax.memptr(),
                          n_sub,
                          T1x.memptr(), T1y.memptr(), T1z.memptr(),
                          T2x.memptr(), T2y.memptr(), T2z.memptr(),
                          T3x.memptr(), T3y.memptr(), T3z.memptr(),
                          Nx.memptr(), Ny.memptr(), Nz.memptr(),
                          D1x.memptr(), D1y.memptr(), D1z.memptr(),
                          D2x.memptr(), D2y.memptr(), D2z.memptr(),
                          D3x.memptr(), D3y.memptr(), D3z.memptr(),
                          rD1.memptr(), rD2.memptr(), rD3.memptr(),
                          n_ray, hit_lists.data());

    // No hits should reference cluster B points (indices 8..15)
    for (size_t r = 0; r < n_ray; ++r)
        for (unsigned idx : hit_lists[r])
            CHECK(idx < 8);

    // Cluster A points should have some hits (they're inside the tube)
    arma::u32_vec hit_count = count_hits_per_point(hit_lists.data(), n_ray, n_point);
    unsigned total_A = arma::accu(hit_count.subvec(0, 7));
    CHECK(total_A > 0);

    // Cluster B should have zero hits
    unsigned total_B = arma::accu(hit_count.subvec(8, 15));
    CHECK(total_B == 0);
}