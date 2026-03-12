// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_acos benchmark_acos.cpp  /sjc/quadriga-lib/lib/libquadriga.a

#include "quadriga_tools.hpp"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ULP distance between two floats (sign-magnitude aware)
static int32_t ulp_dist(float a, float b)
{
    if (std::isnan(a) || std::isnan(b))
        return INT32_MAX;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    int32_t d = ia - ib;
    return d < 0 ? -d : d;
}

int main()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    printf("%-9s | %-40s | %-40s | %-40s | %-40s | %-13s | %-13s | %-22s | %-22s\n",
           "", "AVX2 / float", "AVX2 / double", "Scalar / float", "Scalar / double",
           "Speedup f", "Speedup d", "ULP vs std::acos (f)", "ULP vs std::acos (d)");
    printf("%.*s\n", 260, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // --- Generate test data ---
        // Sweep uniformly over [-1, 1] with varying density near the boundaries
        arma::fvec xf(n);
        arma::vec xd(n);

        for (size_t i = 0; i < n; ++i)
        {
            // Linear sweep from -1 to 1
            double t = -1.0 + 2.0 * (double)i / (double)(n - 1);
            // Mix in some values near +-1 to stress boundary behavior
            if (i % 7 == 0)
                t = std::copysign(1.0 - 1e-6 * ((i * 13 + 5) % 1000), t);
            xf(i) = (float)t;
            xd(i) = (double)xf(i);
        }

        arma::fvec cf, cd_out;

        // --- 1. AVX2 path via fast_acos (float input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_acos(xf, cf);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_acos(xf, cf);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 path via fast_acos (double input, converted internally) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_acos(xd, cd_out);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_acos(xd, cd_out);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar reference (float, single-threaded) ---
        arma::fvec c_scalar_f(n);
        const float *pxf = xf.memptr();
        float *pcsf = c_scalar_f.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
                pcsf[j] = std::acos(pxf[j]);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
                pcsf[j] = std::acos(pxf[j]);
        double dt_scalar_f = (now_ms() - t0) / n_reps;

        // --- 4. Scalar reference (double → float, single-threaded) ---
        arma::fvec c_scalar_d(n);
        const double *pxd = xd.memptr();
        float *pcsd = c_scalar_d.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
                pcsd[j] = (float)std::acos(pxd[j]);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
                pcsd[j] = (float)std::acos(pxd[j]);
        double dt_scalar_d = (now_ms() - t0) / n_reps;

        // --- ULP accuracy: AVX2 output vs double-precision std::acos reference ---

        // Float input path
        int32_t max_ulp_f = 0;
        int64_t sum_ulp_f = 0;
        const float *pcf = cf.memptr();
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::acos((double)pxf[j]);
            int32_t d = ulp_dist(pcf[j], ref);
            if (d > max_ulp_f) max_ulp_f = d;
            sum_ulp_f += d;
        }
        double mean_ulp_f = (double)sum_ulp_f / (double)n;

        // Double input path
        int32_t max_ulp_d = 0;
        int64_t sum_ulp_d = 0;
        const float *pcd = cd_out.memptr();
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::acos(pxd[j]);
            int32_t d = ulp_dist(pcd[j], ref);
            if (d > max_ulp_d) max_ulp_d = d;
            sum_ulp_d += d;
        }
        double mean_ulp_d = (double)sum_ulp_d / (double)n;

        // --- Report ---
        double elems_M = (double)n / 1e6;
        printf("n = %7zu | %7.3f ms (%7.1f Melem/s) | %7.3f ms (%7.1f Melem/s) "
               "| %7.3f ms (%7.1f Melem/s) | %7.3f ms (%7.1f Melem/s) "
               "| %10.1fx | %10.1fx "
               "| avg %5.2f  max %3d   | avg %5.2f  max %3d\n",
               n,
               dt_avx_f, elems_M / dt_avx_f * 1e3,
               dt_avx_d, elems_M / dt_avx_d * 1e3,
               dt_scalar_f, elems_M / dt_scalar_f * 1e3,
               dt_scalar_d, elems_M / dt_scalar_d * 1e3,
               dt_scalar_f / dt_avx_f,
               dt_scalar_d / dt_avx_d,
               mean_ulp_f, max_ulp_f,
               mean_ulp_d, max_ulp_d);
    }

    return 0;
}
