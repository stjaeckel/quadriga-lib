// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_atan2 benchmark_atan2.cpp  /sjc/quadriga-lib/lib/libquadriga.a

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
           "Speedup f", "Speedup d", "ULP vs std::atan2 (f)", "ULP vs std::atan2 (d)");
    printf("%.*s\n", 260, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // --- Generate test data ---
        // Sweep angles uniformly, with varying magnitudes to exercise all octants
        arma::fvec yf(n), xf(n);
        arma::vec yd(n), xd(n);

        for (size_t i = 0; i < n; ++i)
        {
            double angle = -M_PI + 2.0 * M_PI * (double)i / (double)n;
            double radius = 0.01 + 100.0 * (double)((i * 7 + 13) % n) / (double)n;
            yf(i) = (float)(std::sin(angle) * radius);
            xf(i) = (float)(std::cos(angle) * radius);
            yd(i) = (double)yf(i);
            xd(i) = (double)xf(i);
        }

        arma::fvec af, ad_out;

        // --- 1. AVX2 path via fast_atan2 (float input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_atan2(yf, xf, af);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_atan2(yf, xf, af);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 path via fast_atan2 (double input, converted internally) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_atan2(yd, xd, ad_out);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_atan2(yd, xd, ad_out);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar reference (float, single-threaded) ---
        arma::fvec a_scalar_f(n);
        const float *pyf = yf.memptr(), *pxf = xf.memptr();
        float *pasf = a_scalar_f.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
                pasf[j] = std::atan2(pyf[j], pxf[j]);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
                pasf[j] = std::atan2(pyf[j], pxf[j]);
        double dt_scalar_f = (now_ms() - t0) / n_reps;

        // --- 4. Scalar reference (double → float, single-threaded) ---
        arma::fvec a_scalar_d(n);
        const double *pyd = yd.memptr(), *pxd = xd.memptr();
        float *pasd = a_scalar_d.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
                pasd[j] = (float)std::atan2(pyd[j], pxd[j]);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
                pasd[j] = (float)std::atan2(pyd[j], pxd[j]);
        double dt_scalar_d = (now_ms() - t0) / n_reps;

        // --- ULP accuracy: AVX2 output vs double-precision std::atan2 reference ---
        // Reference: compute atan2 in double precision, then cast to float
        // This is the best possible single-precision answer for a given (y, x) pair

        // Float input path
        int32_t max_ulp_f = 0;
        int64_t sum_ulp_f = 0;
        const float *paf = af.memptr();
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::atan2((double)pyf[j], (double)pxf[j]);
            int32_t d = ulp_dist(paf[j], ref);
            if (d > max_ulp_f) max_ulp_f = d;
            sum_ulp_f += d;
        }
        double mean_ulp_f = (double)sum_ulp_f / (double)n;

        // Double input path
        int32_t max_ulp_d = 0;
        int64_t sum_ulp_d = 0;
        const float *pad = ad_out.memptr();
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::atan2(pyd[j], pxd[j]);
            int32_t d = ulp_dist(pad[j], ref);
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
