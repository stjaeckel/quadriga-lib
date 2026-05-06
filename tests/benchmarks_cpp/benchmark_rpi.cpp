// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_rpi benchmark_rpi.cpp /sjc/quadriga-lib/lib/libquadriga.a -L/usr/local/cuda-12.4/targets/x86_64-linux/lib -lcudart

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"
#include "quadriga_lib_cuda_functions.hpp"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ============================================================================
// PointCloud: holds SoA point data + sub-cloud info + AABBs
// ============================================================================
struct PointCloud
{
    size_t n_point;     // total points (padded to vec_size from segmentation)
    size_t n_point_raw; // original point count before padding
    size_t n_sub;       // number of sub-clouds
    size_t n_sub_s;     // n_sub rounded up to multiple of 8

    // SoA point arrays (float), length n_point
    std::vector<float> Px, Py, Pz;

    // Double-precision copies for GENERIC<double>
    std::vector<double> dPx, dPy, dPz;

    // Sub-cloud index, length n_sub
    std::vector<unsigned> SCI;

    // AABB arrays, length n_sub_s (padded)
    std::vector<float> Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    std::vector<double> dXmin, dXmax, dYmin, dYmax, dZmin, dZmax;
};

// Build a point cloud: random points in a spherical shell around the origin
// n_points:      number of points to generate
// r_inner:       inner shell radius
// r_outer:       outer shell radius
// target_size:   target sub-cloud size for segmentation
// vec_size:      SIMD alignment (8 for AVX2)
static PointCloud build_point_cloud(size_t n_points, float r_inner, float r_outer,
                                    arma::uword target_size, arma::uword vec_size,
                                    unsigned seed = 42)
{
    PointCloud pc;
    pc.n_point_raw = n_points;

    // Generate random points in a spherical shell
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    arma::fmat points(n_points, 3);
    for (size_t i = 0; i < n_points; ++i)
    {
        // Uniform random point on sphere
        float theta = 2.0f * (float)M_PI * u01(rng);
        float phi = std::acos(1.0f - 2.0f * u01(rng));

        // Uniform random radius in shell [r_inner, r_outer]
        float r3_min = r_inner * r_inner * r_inner;
        float r3_max = r_outer * r_outer * r_outer;
        float r = std::cbrt(r3_min + u01(rng) * (r3_max - r3_min));

        points(i, 0) = r * std::sin(phi) * std::cos(theta);
        points(i, 1) = r * std::sin(phi) * std::sin(theta);
        points(i, 2) = r * std::cos(phi);
    }

    // Segment the point cloud
    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    arma::uword n_sub = quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index,
                                                                target_size, vec_size);

    pc.n_point = pointsR.n_rows;
    pc.n_sub = n_sub;
    pc.n_sub_s = ((n_sub + 7) / 8) * 8;

    printf("  %zu points (raw %zu), n_sub=%zu, n_sub_s=%zu\n",
           pc.n_point, pc.n_point_raw, pc.n_sub, pc.n_sub_s);

    // Extract SoA point data
    pc.Px.resize(pc.n_point);
    pc.Py.resize(pc.n_point);
    pc.Pz.resize(pc.n_point);
    std::memcpy(pc.Px.data(), pointsR.colptr(0), pc.n_point * sizeof(float));
    std::memcpy(pc.Py.data(), pointsR.colptr(1), pc.n_point * sizeof(float));
    std::memcpy(pc.Pz.data(), pointsR.colptr(2), pc.n_point * sizeof(float));

    // Double-precision copies
    pc.dPx.resize(pc.n_point);
    pc.dPy.resize(pc.n_point);
    pc.dPz.resize(pc.n_point);
    for (size_t i = 0; i < pc.n_point; ++i)
    {
        pc.dPx[i] = pc.Px[i];
        pc.dPy[i] = pc.Py[i];
        pc.dPz[i] = pc.Pz[i];
    }

    // Sub-cloud index
    pc.SCI.resize(pc.n_sub);
    for (size_t i = 0; i < pc.n_sub; ++i)
        pc.SCI[i] = sub_cloud_index(i);

    // Compute AABBs per sub-cloud
    pc.Xmin.resize(pc.n_sub_s, 1e20f);  pc.Xmax.resize(pc.n_sub_s, -1e20f);
    pc.Ymin.resize(pc.n_sub_s, 1e20f);  pc.Ymax.resize(pc.n_sub_s, -1e20f);
    pc.Zmin.resize(pc.n_sub_s, 1e20f);  pc.Zmax.resize(pc.n_sub_s, -1e20f);

    for (size_t s = 0; s < pc.n_sub; ++s)
    {
        size_t i_start = pc.SCI[s];
        size_t i_end = (s + 1 < pc.n_sub) ? pc.SCI[s + 1] : pc.n_point;

        float xmin = 1e20f, xmax = -1e20f;
        float ymin = 1e20f, ymax = -1e20f;
        float zmin = 1e20f, zmax = -1e20f;

        for (size_t i = i_start; i < i_end; ++i)
        {
            float x = pc.Px[i], y = pc.Py[i], z = pc.Pz[i];
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }

        pc.Xmin[s] = xmin; pc.Xmax[s] = xmax;
        pc.Ymin[s] = ymin; pc.Ymax[s] = ymax;
        pc.Zmin[s] = zmin; pc.Zmax[s] = zmax;
    }

    // Double-precision AABB copies
    pc.dXmin.resize(pc.n_sub_s); pc.dXmax.resize(pc.n_sub_s);
    pc.dYmin.resize(pc.n_sub_s); pc.dYmax.resize(pc.n_sub_s);
    pc.dZmin.resize(pc.n_sub_s); pc.dZmax.resize(pc.n_sub_s);
    for (size_t i = 0; i < pc.n_sub_s; ++i)
    {
        pc.dXmin[i] = pc.Xmin[i]; pc.dXmax[i] = pc.Xmax[i];
        pc.dYmin[i] = pc.Ymin[i]; pc.dYmax[i] = pc.Ymax[i];
        pc.dZmin[i] = pc.Zmin[i]; pc.dZmax[i] = pc.Zmax[i];
    }

    return pc;
}

// ============================================================================
// RayTubes: holds SoA ray tube data (30 float arrays + double copies)
// ============================================================================
struct RayTubes
{
    size_t n_ray;

    // Float arrays, length n_ray each
    std::vector<float> T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z;
    std::vector<float> Nx, Ny, Nz;
    std::vector<float> D1x, D1y, D1z, D2x, D2y, D2z, D3x, D3y, D3z;
    std::vector<float> rD1, rD2, rD3;

    // Double-precision copies
    std::vector<double> dT1x, dT1y, dT1z, dT2x, dT2y, dT2z, dT3x, dT3y, dT3z;
    std::vector<double> dNx, dNy, dNz;
    std::vector<double> dD1x, dD1y, dD1z, dD2x, dD2y, dD2z, dD3x, dD3y, dD3z;
    std::vector<double> drD1, drD2, drD3;
};

// Generate ray tubes from icosphere
// n_div: subdivision level (n_ray = 20 * n_div^2)
// radius: icosphere radius (determines tube vertex offset scale)
static RayTubes generate_ray_tubes(arma::uword n_div, float radius)
{
    RayTubes rt;

    // Generate icosphere: center, vert offsets, and Cartesian directions
    arma::fmat center, vert, tridir;
    arma::uword n_faces = quadriga_lib::icosphere(n_div, radius, &center,
                                                   (arma::fvec *)nullptr, &vert, &tridir, true);
    rt.n_ray = n_faces;

    // Allocate all arrays
    rt.T1x.resize(n_faces); rt.T1y.resize(n_faces); rt.T1z.resize(n_faces);
    rt.T2x.resize(n_faces); rt.T2y.resize(n_faces); rt.T2z.resize(n_faces);
    rt.T3x.resize(n_faces); rt.T3y.resize(n_faces); rt.T3z.resize(n_faces);
    rt.Nx.resize(n_faces);  rt.Ny.resize(n_faces);  rt.Nz.resize(n_faces);
    rt.D1x.resize(n_faces); rt.D1y.resize(n_faces); rt.D1z.resize(n_faces);
    rt.D2x.resize(n_faces); rt.D2y.resize(n_faces); rt.D2z.resize(n_faces);
    rt.D3x.resize(n_faces); rt.D3y.resize(n_faces); rt.D3z.resize(n_faces);
    rt.rD1.resize(n_faces); rt.rD2.resize(n_faces); rt.rD3.resize(n_faces);

    // Compute ray tube data (matching the boilerplate logic)
    for (size_t i = 0; i < n_faces; ++i)
    {
        float cx = center(i, 0), cy = center(i, 1), cz = center(i, 2);

        // Vertex positions in GCS = center + vertex offset
        float t1x = cx + vert(i, 0), t1y = cy + vert(i, 1), t1z = cz + vert(i, 2);
        float t2x = cx + vert(i, 3), t2y = cy + vert(i, 4), t2z = cz + vert(i, 5);
        float t3x = cx + vert(i, 6), t3y = cy + vert(i, 7), t3z = cz + vert(i, 8);

        rt.T1x[i] = t1x; rt.T1y[i] = t1y; rt.T1z[i] = t1z;
        rt.T2x[i] = t2x; rt.T2y[i] = t2y; rt.T2z[i] = t2z;
        rt.T3x[i] = t3x; rt.T3y[i] = t3y; rt.T3z[i] = t3z;

        // Edge vectors for normal computation
        float e1x = t2x - t1x, e1y = t2y - t1y, e1z = t2z - t1z;
        float e2x = t3x - t1x, e2y = t3y - t1y, e2z = t3z - t1z;

        // Face normal = cross(edge1, edge2)
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        rt.Nx[i] = nx; rt.Ny[i] = ny; rt.Nz[i] = nz;

        // Ray directions from tridir (Cartesian), normalize if needed
        float d1x = tridir(i, 0), d1y = tridir(i, 1), d1z = tridir(i, 2);
        float d2x = tridir(i, 3), d2y = tridir(i, 4), d2z = tridir(i, 5);
        float d3x = tridir(i, 6), d3y = tridir(i, 7), d3z = tridir(i, 8);

        auto normalize = [](float &x, float &y, float &z)
        {
            float len2 = x * x + y * y + z * z;
            if (std::abs(len2 - 1.0f) > 2e-7f)
            {
                float inv = std::sqrt(1.0f / len2);
                x *= inv; y *= inv; z *= inv;
            }
        };

        normalize(d1x, d1y, d1z);
        normalize(d2x, d2y, d2z);
        normalize(d3x, d3y, d3z);

        rt.D1x[i] = d1x; rt.D1y[i] = d1y; rt.D1z[i] = d1z;
        rt.D2x[i] = d2x; rt.D2y[i] = d2y; rt.D2z[i] = d2z;
        rt.D3x[i] = d3x; rt.D3y[i] = d3y; rt.D3z[i] = d3z;

        // Inverse dot products
        rt.rD1[i] = 1.0f / (d1x * nx + d1y * ny + d1z * nz);
        rt.rD2[i] = 1.0f / (d2x * nx + d2y * ny + d2z * nz);
        rt.rD3[i] = 1.0f / (d3x * nx + d3y * ny + d3z * nz);
    }

    // Double-precision copies
    rt.dT1x.resize(n_faces); rt.dT1y.resize(n_faces); rt.dT1z.resize(n_faces);
    rt.dT2x.resize(n_faces); rt.dT2y.resize(n_faces); rt.dT2z.resize(n_faces);
    rt.dT3x.resize(n_faces); rt.dT3y.resize(n_faces); rt.dT3z.resize(n_faces);
    rt.dNx.resize(n_faces);  rt.dNy.resize(n_faces);  rt.dNz.resize(n_faces);
    rt.dD1x.resize(n_faces); rt.dD1y.resize(n_faces); rt.dD1z.resize(n_faces);
    rt.dD2x.resize(n_faces); rt.dD2y.resize(n_faces); rt.dD2z.resize(n_faces);
    rt.dD3x.resize(n_faces); rt.dD3y.resize(n_faces); rt.dD3z.resize(n_faces);
    rt.drD1.resize(n_faces); rt.drD2.resize(n_faces); rt.drD3.resize(n_faces);

    for (size_t i = 0; i < n_faces; ++i)
    {
        rt.dT1x[i] = rt.T1x[i]; rt.dT1y[i] = rt.T1y[i]; rt.dT1z[i] = rt.T1z[i];
        rt.dT2x[i] = rt.T2x[i]; rt.dT2y[i] = rt.T2y[i]; rt.dT2z[i] = rt.T2z[i];
        rt.dT3x[i] = rt.T3x[i]; rt.dT3y[i] = rt.T3y[i]; rt.dT3z[i] = rt.T3z[i];
        rt.dNx[i] = rt.Nx[i];   rt.dNy[i] = rt.Ny[i];   rt.dNz[i] = rt.Nz[i];
        rt.dD1x[i] = rt.D1x[i]; rt.dD1y[i] = rt.D1y[i]; rt.dD1z[i] = rt.D1z[i];
        rt.dD2x[i] = rt.D2x[i]; rt.dD2y[i] = rt.D2y[i]; rt.dD2z[i] = rt.D2z[i];
        rt.dD3x[i] = rt.D3x[i]; rt.dD3y[i] = rt.D3y[i]; rt.dD3z[i] = rt.D3z[i];
        rt.drD1[i] = rt.rD1[i]; rt.drD2[i] = rt.rD2[i]; rt.drD3[i] = rt.rD3[i];
    }

    return rt;
}

// ============================================================================
// Helper: count total hits across all rays
// ============================================================================
static size_t count_total_hits(const std::vector<std::vector<unsigned>> &hit_lists, size_t n_ray)
{
    size_t total = 0;
    for (size_t r = 0; r < n_ray; ++r)
        total += hit_lists[r].size();
    return total;
}

// ============================================================================
// PERFORMANCE
// ============================================================================
static void run_performance(const PointCloud &pc, const RayTubes &rt, const char *label,
                            bool run_generic, int n_reps)
{
    const size_t n = rt.n_ray;

    printf("%-12s | n_pt=%8zu, n_sub=%4zu | n_ray=%10zu | ", label, pc.n_point, pc.n_sub, n);

    // --- Generic float ---
    double dt_gen_f = 0.0;
    if (run_generic)
    {
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
        {
            std::vector<std::vector<unsigned>> hit_lists(n);
            qd_RPI_GENERIC(pc.Px.data(), pc.Py.data(), pc.Pz.data(), pc.n_point,
                           pc.SCI.data(),
                           pc.Xmin.data(), pc.Xmax.data(), pc.Ymin.data(), pc.Ymax.data(),
                           pc.Zmin.data(), pc.Zmax.data(), pc.n_sub,
                           rt.T1x.data(), rt.T1y.data(), rt.T1z.data(),
                           rt.T2x.data(), rt.T2y.data(), rt.T2z.data(),
                           rt.T3x.data(), rt.T3y.data(), rt.T3z.data(),
                           rt.Nx.data(), rt.Ny.data(), rt.Nz.data(),
                           rt.D1x.data(), rt.D1y.data(), rt.D1z.data(),
                           rt.D2x.data(), rt.D2y.data(), rt.D2z.data(),
                           rt.D3x.data(), rt.D3y.data(), rt.D3z.data(),
                           rt.rD1.data(), rt.rD2.data(), rt.rD3.data(),
                           n, hit_lists.data());
        }
        dt_gen_f = (now_ms() - t0) / n_reps;
        printf("Gen/f %9.1f ms | ", dt_gen_f);
    }
    else
        printf("Gen/f  %9s   | ", "skip");

    // --- Generic double ---
    double dt_gen_d = 0.0;
    if (run_generic)
    {
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
        {
            std::vector<std::vector<unsigned>> hit_lists(n);
            qd_RPI_GENERIC(pc.dPx.data(), pc.dPy.data(), pc.dPz.data(), pc.n_point,
                           pc.SCI.data(),
                           pc.dXmin.data(), pc.dXmax.data(), pc.dYmin.data(), pc.dYmax.data(),
                           pc.dZmin.data(), pc.dZmax.data(), pc.n_sub,
                           rt.dT1x.data(), rt.dT1y.data(), rt.dT1z.data(),
                           rt.dT2x.data(), rt.dT2y.data(), rt.dT2z.data(),
                           rt.dT3x.data(), rt.dT3y.data(), rt.dT3z.data(),
                           rt.dNx.data(), rt.dNy.data(), rt.dNz.data(),
                           rt.dD1x.data(), rt.dD1y.data(), rt.dD1z.data(),
                           rt.dD2x.data(), rt.dD2y.data(), rt.dD2z.data(),
                           rt.dD3x.data(), rt.dD3y.data(), rt.dD3z.data(),
                           rt.drD1.data(), rt.drD2.data(), rt.drD3.data(),
                           n, hit_lists.data());
        }
        dt_gen_d = (now_ms() - t0) / n_reps;
        printf("Gen/d %9.1f ms | ", dt_gen_d);
    }
    else
        printf("Gen/d  %9s   | ", "skip");

    // --- AVX2 ---
    double dt_avx = 0.0;
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
        {
            std::vector<std::vector<unsigned>> hit_lists(n);
            qd_RPI_AVX2(pc.Px.data(), pc.Py.data(), pc.Pz.data(), pc.n_point,
                         pc.SCI.data(),
                         pc.Xmin.data(), pc.Xmax.data(), pc.Ymin.data(), pc.Ymax.data(),
                         pc.Zmin.data(), pc.Zmax.data(), pc.n_sub,
                         rt.T1x.data(), rt.T1y.data(), rt.T1z.data(),
                         rt.T2x.data(), rt.T2y.data(), rt.T2z.data(),
                         rt.T3x.data(), rt.T3y.data(), rt.T3z.data(),
                         rt.Nx.data(), rt.Ny.data(), rt.Nz.data(),
                         rt.D1x.data(), rt.D1y.data(), rt.D1z.data(),
                         rt.D2x.data(), rt.D2y.data(), rt.D2z.data(),
                         rt.D3x.data(), rt.D3y.data(), rt.D3z.data(),
                         rt.rD1.data(), rt.rD2.data(), rt.rD3.data(),
                         n, hit_lists.data());
        }
        dt_avx = (now_ms() - t0) / n_reps;
        printf("AVX2 %9.1f ms | ", dt_avx);
    }
    else
        printf("AVX2 %9s   | ", "N/A");

    // --- CUDA ---
    double dt_cuda = 0.0;
    if (quadriga_lib::quadriga_lib_has_CUDA())
    {
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
        {
            std::vector<std::vector<unsigned>> hit_lists(n);
            qd_RPI_CUDA(pc.Px.data(), pc.Py.data(), pc.Pz.data(), pc.n_point,
                         pc.SCI.data(),
                         pc.Xmin.data(), pc.Xmax.data(), pc.Ymin.data(), pc.Ymax.data(),
                         pc.Zmin.data(), pc.Zmax.data(), pc.n_sub,
                         rt.T1x.data(), rt.T1y.data(), rt.T1z.data(),
                         rt.T2x.data(), rt.T2y.data(), rt.T2z.data(),
                         rt.T3x.data(), rt.T3y.data(), rt.T3z.data(),
                         rt.Nx.data(), rt.Ny.data(), rt.Nz.data(),
                         rt.D1x.data(), rt.D1y.data(), rt.D1z.data(),
                         rt.D2x.data(), rt.D2y.data(), rt.D2z.data(),
                         rt.D3x.data(), rt.D3y.data(), rt.D3z.data(),
                         rt.rD1.data(), rt.rD2.data(), rt.rD3.data(),
                         n, hit_lists.data());
        }
        dt_cuda = (now_ms() - t0) / n_reps;
        printf("CUDA %9.1f ms | ", dt_cuda);
    }
    else
        printf("CUDA %9s   | ", "N/A");

    // --- Speedups vs Generic float (if available) ---
    if (run_generic && dt_gen_f > 0.0)
    {
        if (quadriga_lib::quadriga_lib_has_AVX2() && dt_avx > 0.0)
            printf("AVX2 %5.1fx | ", dt_gen_f / dt_avx);
        if (quadriga_lib::quadriga_lib_has_CUDA() && dt_cuda > 0.0)
            printf("CUDA %5.1fx", dt_gen_f / dt_cuda);
    }

    printf("\n");
}

// ============================================================================
// ACCURACY: AVX2 & CUDA vs Generic-double reference
// ============================================================================
static void run_accuracy(const PointCloud &pc, const RayTubes &rt, const char *label)
{
    const size_t n = rt.n_ray;

    // --- Reference: Generic double ---
    std::vector<std::vector<unsigned>> ref(n);
    qd_RPI_GENERIC(pc.dPx.data(), pc.dPy.data(), pc.dPz.data(), pc.n_point,
                   pc.SCI.data(),
                   pc.dXmin.data(), pc.dXmax.data(), pc.dYmin.data(), pc.dYmax.data(),
                   pc.dZmin.data(), pc.dZmax.data(), pc.n_sub,
                   rt.dT1x.data(), rt.dT1y.data(), rt.dT1z.data(),
                   rt.dT2x.data(), rt.dT2y.data(), rt.dT2z.data(),
                   rt.dT3x.data(), rt.dT3y.data(), rt.dT3z.data(),
                   rt.dNx.data(), rt.dNy.data(), rt.dNz.data(),
                   rt.dD1x.data(), rt.dD1y.data(), rt.dD1z.data(),
                   rt.dD2x.data(), rt.dD2y.data(), rt.dD2z.data(),
                   rt.dD3x.data(), rt.dD3y.data(), rt.dD3z.data(),
                   rt.drD1.data(), rt.drD2.data(), rt.drD3.data(),
                   n, ref.data());

    // Sort reference hit lists for comparison
    for (size_t r = 0; r < n; ++r)
        std::sort(ref[r].begin(), ref[r].end());

    // Count reference hits
    size_t ref_total_hits = count_total_hits(ref, n);
    size_t ref_rays_with_hits = 0;
    for (size_t r = 0; r < n; ++r)
        if (!ref[r].empty()) ++ref_rays_with_hits;

    printf("%-12s | n_pt=%8zu, n_sub=%4zu | n_ray=%10zu | ref: %zu total hits, %zu rays w/ hits (%.1f%%)\n",
           label, pc.n_point, pc.n_sub, n,
           ref_total_hits, ref_rays_with_hits, 100.0 * ref_rays_with_hits / n);

    // --- Helper lambda to compare a result against the reference ---
    auto compare = [&](const char *name, const std::vector<std::vector<unsigned>> &res)
    {
        size_t exact_match = 0;     // rays where sorted hit list == ref
        size_t count_match = 0;     // rays where hit_list.size() == ref.size()
        size_t total_hits_res = 0;
        double jaccard_sum = 0.0;
        size_t jaccard_count = 0;   // rays where at least one of ref/res is non-empty

        for (size_t r = 0; r < n; ++r)
        {
            auto a = res[r];
            std::sort(a.begin(), a.end());
            const auto &b = ref[r]; // already sorted

            total_hits_res += a.size();

            if (a.size() == b.size()) ++count_match;
            if (a == b) ++exact_match;

            // Jaccard similarity (only for rays where at least one has hits)
            if (!a.empty() || !b.empty())
            {
                // Compute intersection and union sizes (both sorted)
                size_t isect = 0;
                size_t ia = 0, ib = 0;
                while (ia < a.size() && ib < b.size())
                {
                    if (a[ia] == b[ib]) { ++isect; ++ia; ++ib; }
                    else if (a[ia] < b[ib]) ++ia;
                    else ++ib;
                }
                size_t union_sz = a.size() + b.size() - isect;
                jaccard_sum += (double)isect / (double)union_sz;
                ++jaccard_count;
            }
        }

        double avg_jaccard = jaccard_count > 0 ? jaccard_sum / jaccard_count : 1.0;

        printf("  %-6s | exact match %6.2f%% | count match %6.2f%% | total hits %zu (ref %zu) | avg Jaccard %.4f\n",
               name,
               100.0 * exact_match / n,
               100.0 * count_match / n,
               total_hits_res, ref_total_hits,
               avg_jaccard);
    };

    // --- AVX2 ---
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        std::vector<std::vector<unsigned>> hit_avx(n);
        qd_RPI_AVX2(pc.Px.data(), pc.Py.data(), pc.Pz.data(), pc.n_point,
                     pc.SCI.data(),
                     pc.Xmin.data(), pc.Xmax.data(), pc.Ymin.data(), pc.Ymax.data(),
                     pc.Zmin.data(), pc.Zmax.data(), pc.n_sub,
                     rt.T1x.data(), rt.T1y.data(), rt.T1z.data(),
                     rt.T2x.data(), rt.T2y.data(), rt.T2z.data(),
                     rt.T3x.data(), rt.T3y.data(), rt.T3z.data(),
                     rt.Nx.data(), rt.Ny.data(), rt.Nz.data(),
                     rt.D1x.data(), rt.D1y.data(), rt.D1z.data(),
                     rt.D2x.data(), rt.D2y.data(), rt.D2z.data(),
                     rt.D3x.data(), rt.D3y.data(), rt.D3z.data(),
                     rt.rD1.data(), rt.rD2.data(), rt.rD3.data(),
                     n, hit_avx.data());
        compare("AVX2", hit_avx);
    }

    // --- CUDA ---
    if (quadriga_lib::quadriga_lib_has_CUDA())
    {
        std::vector<std::vector<unsigned>> hit_cuda(n);
        qd_RPI_CUDA(pc.Px.data(), pc.Py.data(), pc.Pz.data(), pc.n_point,
                     pc.SCI.data(),
                     pc.Xmin.data(), pc.Xmax.data(), pc.Ymin.data(), pc.Ymax.data(),
                     pc.Zmin.data(), pc.Zmax.data(), pc.n_sub,
                     rt.T1x.data(), rt.T1y.data(), rt.T1z.data(),
                     rt.T2x.data(), rt.T2y.data(), rt.T2z.data(),
                     rt.T3x.data(), rt.T3y.data(), rt.T3z.data(),
                     rt.Nx.data(), rt.Ny.data(), rt.Nz.data(),
                     rt.D1x.data(), rt.D1y.data(), rt.D1z.data(),
                     rt.D2x.data(), rt.D2y.data(), rt.D2z.data(),
                     rt.D3x.data(), rt.D3y.data(), rt.D3z.data(),
                     rt.rD1.data(), rt.rD2.data(), rt.rD3.data(),
                     n, hit_cuda.data());
        compare("CUDA", hit_cuda);
    }

    // --- Generic float (for float-vs-double comparison) ---
    {
        std::vector<std::vector<unsigned>> hit_genf(n);
        qd_RPI_GENERIC(pc.Px.data(), pc.Py.data(), pc.Pz.data(), pc.n_point,
                       pc.SCI.data(),
                       pc.Xmin.data(), pc.Xmax.data(), pc.Ymin.data(), pc.Ymax.data(),
                       pc.Zmin.data(), pc.Zmax.data(), pc.n_sub,
                       rt.T1x.data(), rt.T1y.data(), rt.T1z.data(),
                       rt.T2x.data(), rt.T2y.data(), rt.T2z.data(),
                       rt.T3x.data(), rt.T3y.data(), rt.T3z.data(),
                       rt.Nx.data(), rt.Ny.data(), rt.Nz.data(),
                       rt.D1x.data(), rt.D1y.data(), rt.D1z.data(),
                       rt.D2x.data(), rt.D2y.data(), rt.D2z.data(),
                       rt.D3x.data(), rt.D3y.data(), rt.D3z.data(),
                       rt.rD1.data(), rt.rD2.data(), rt.rD3.data(),
                       n, hit_genf.data());
        compare("Gen/f", hit_genf);
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main()
{
    printf("quadriga-lib RPI Benchmark\n");
    printf("AVX2: %s, CUDA: %s\n\n",
           quadriga_lib::quadriga_lib_has_AVX2() ? "YES" : "NO",
           quadriga_lib::quadriga_lib_has_CUDA() ? "YES" : "NO");

    // ---- Build point clouds ----
    // Small: ~30k points in spherical shell [15, 25]
    // Large: ~1M points in spherical shell [15, 25]
    printf("Building point clouds...\n");
    PointCloud pc_small = build_point_cloud(30000,   15.0f, 25.0f, 1024, 8, 42);
    PointCloud pc_large = build_point_cloud(1000000, 15.0f, 25.0f, 1024, 8, 123);
    printf("\n");

    // ---- Generate ray tube sets ----
    // Icosphere at origin with radius 1.0 (small tube cross-section)
    // n_div=23  → 10,580 rays
    // n_div=224 → 1,003,520 rays
    // n_div=1581 → 49,968,420 rays
    printf("Generating ray tubes...\n");

    printf("  10k rays: ");
    double t0 = now_ms();
    RayTubes rt_10k = generate_ray_tubes(23, 1.0f);
    printf("%zu rays, %.1f ms\n", rt_10k.n_ray, now_ms() - t0);

    printf("  1M rays:  ");
    t0 = now_ms();
    RayTubes rt_1M = generate_ray_tubes(224, 1.0f);
    printf("%zu rays, %.1f ms\n", rt_1M.n_ray, now_ms() - t0);

    printf("  50M rays: ");
    t0 = now_ms();
    RayTubes rt_50M = generate_ray_tubes(1581, 1.0f);
    printf("%zu rays, %.1f ms\n", rt_50M.n_ray, now_ms() - t0);

    printf("\n");

    // ---- Performance ----
    const int n_reps = 3;
    printf("============================================================\n");
    printf("=== PERFORMANCE (%d reps, times are per-call averages) ===\n", n_reps);
    printf("============================================================\n\n");

    run_performance(pc_small, rt_10k,  "small/10k",  true,  n_reps);
    run_performance(pc_small, rt_1M,   "small/1M",   true,  n_reps);
    run_performance(pc_small, rt_50M,  "small/50M",  false, n_reps);
    run_performance(pc_large, rt_10k,  "large/10k",  true,  n_reps);
    run_performance(pc_large, rt_1M,   "large/1M",   true,  n_reps);
    run_performance(pc_large, rt_50M,  "large/50M",  false, n_reps);

    printf("\n");

    // ---- Accuracy (smaller ray sets only) ----
    printf("============================================================\n");
    printf("=== ACCURACY (vs Generic-double reference) ===\n");
    printf("============================================================\n\n");

    run_accuracy(pc_small, rt_10k, "small/10k");
    printf("\n");
    run_accuracy(pc_small, rt_1M,  "small/1M");
    printf("\n");
    run_accuracy(pc_large, rt_10k, "large/10k");
    printf("\n");
    run_accuracy(pc_large, rt_1M,  "large/1M");

    printf("\nDone.\n");
    return 0;
}
