// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_sincos benchmark_sincos.cpp  /sjc/quadriga-lib/lib/libquadriga.a

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

// ============================================================================
// PART 1: Performance benchmark (size sweep, input range [-4pi, 4pi])
// ============================================================================
static void run_performance()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    printf("=== PERFORMANCE (sincos combined, input range [-4pi, 4pi]) ===\n\n");
    printf("%-9s | %-40s | %-40s | %-40s | %-40s | %-13s | %-13s\n",
           "", "AVX2 / float", "AVX2 / double", "Scalar / float", "Scalar / double",
           "Speedup f", "Speedup d");
    printf("%.*s\n", 220, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // --- Generate test data: uniform sweep over [-4pi, 4pi] with varying magnitudes ---
        arma::fvec xf(n);
        arma::vec xd(n);

        for (size_t i = 0; i < n; ++i)
        {
            double t = -4.0 * M_PI + 8.0 * M_PI * (double)i / (double)(n > 1 ? n - 1 : 1);
            // Add some jitter to avoid perfectly uniform spacing
            double jitter = 0.01 * std::sin((double)(i * 17 + 3));
            xf(i) = (float)(t + jitter);
            xd(i) = (double)xf(i);
        }

        arma::fvec sf, cf, sd_out, cd_out;

        // --- 1. AVX2 sincos (float input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_sincos(xf, &sf, &cf);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_sincos(xf, &sf, &cf);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 sincos (double input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_sincos(xd, &sd_out, &cd_out);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_sincos(xd, &sd_out, &cd_out);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar sin+cos (float, single-threaded) ---
        arma::fvec s_scalar_f(n), c_scalar_f(n);
        const float *pxf = xf.memptr();
        float *pssf = s_scalar_f.memptr(), *pcsf = c_scalar_f.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                pssf[j] = std::sin(pxf[j]);
                pcsf[j] = std::cos(pxf[j]);
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                pssf[j] = std::sin(pxf[j]);
                pcsf[j] = std::cos(pxf[j]);
            }
        double dt_scalar_f = (now_ms() - t0) / n_reps;

        // --- 4. Scalar sin+cos (double → float, single-threaded) ---
        arma::fvec s_scalar_d(n), c_scalar_d(n);
        const double *pxd = xd.memptr();
        float *pssd = s_scalar_d.memptr(), *pcsd = c_scalar_d.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                pssd[j] = (float)std::sin(pxd[j]);
                pcsd[j] = (float)std::cos(pxd[j]);
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                pssd[j] = (float)std::sin(pxd[j]);
                pcsd[j] = (float)std::cos(pxd[j]);
            }
        double dt_scalar_d = (now_ms() - t0) / n_reps;

        // --- Report ---
        double elems_M = (double)n / 1e6;
        printf("n = %7zu | %7.3f ms (%7.1f Melem/s) | %7.3f ms (%7.1f Melem/s) "
               "| %7.3f ms (%7.1f Melem/s) | %7.3f ms (%7.1f Melem/s) "
               "| %10.1fx | %10.1fx\n",
               n,
               dt_avx_f, elems_M / dt_avx_f * 1e3,
               dt_avx_d, elems_M / dt_avx_d * 1e3,
               dt_scalar_f, elems_M / dt_scalar_f * 1e3,
               dt_scalar_d, elems_M / dt_scalar_d * 1e3,
               dt_scalar_f / dt_avx_f,
               dt_scalar_d / dt_avx_d);
    }
}

// ============================================================================
// PART 2: Accuracy benchmark (range sweep at fixed large n)
// ============================================================================
static void run_accuracy()
{
    const size_t n = 1048576;

    struct Range
    {
        const char *label;
        double lo, hi;
    };
    const Range ranges[] = {
        {"[-pi, pi]", -M_PI, M_PI},
        {"[-10pi, 10pi]", -10.0 * M_PI, 10.0 * M_PI},
        {"[-1e3, 1e3]", -1e3, 1e3},
        {"[-1e5, 1e5]", -1e5, 1e5},
        {"[-1e7, 1e7]", -1e7, 1e7},
    };
    const int n_ranges = sizeof(ranges) / sizeof(ranges[0]);

    printf("\n=== ACCURACY (n = %zu, ULP vs best achievable float result) ===\n\n", n);
    printf("%-18s | %-28s | %-28s | %-28s | %-28s\n",
           "Range", "sin (float in)", "sin (double in)", "cos (float in)", "cos (double in)");
    printf("%.*s\n", 140, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (int r = 0; r < n_ranges; ++r)
    {
        double lo = ranges[r].lo, hi = ranges[r].hi;

        // --- Generate test data ---
        // Both xf and xd contain the same mathematical values (float precision).
        // The double path's advantage is better range reduction, not extra input precision.
        arma::fvec xf(n);
        arma::vec xd(n);

        for (size_t i = 0; i < n; ++i)
        {
            double t = lo + (hi - lo) * (double)i / (double)(n > 1 ? n - 1 : 1);
            double jitter = (hi - lo) * 1e-7 * std::sin((double)(i * 17 + 3));
            xf(i) = (float)(t + jitter);
            xd(i) = (double)xf(i); // Exactly the float value promoted to double
        }

        arma::fvec sf, cf, sd_out, cd_out;

        // --- AVX2: float input ---
        quadriga_lib::fast_sincos(xf, &sf, &cf);

        // --- AVX2: double input ---
        quadriga_lib::fast_sincos(xd, &sd_out, &cd_out);

        // --- ULP: sin, float input ---
        const float *pxf = xf.memptr();
        const float *psf = sf.memptr();
        int32_t max_ulp_sf = 0;
        int64_t sum_ulp_sf = 0;
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::sin((double)pxf[j]);
            int32_t d = ulp_dist(psf[j], ref);
            if (d > max_ulp_sf) max_ulp_sf = d;
            sum_ulp_sf += d;
        }

        // --- ULP: sin, double input ---
        // Since xd = (double)xf, the reference is the same gold standard.
        // The double path should show equal or better ULP thanks to double range reduction.
        const double *pxd = xd.memptr();
        const float *psd = sd_out.memptr();
        int32_t max_ulp_sd = 0;
        int64_t sum_ulp_sd = 0;
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::sin(pxd[j]);
            int32_t d = ulp_dist(psd[j], ref);
            if (d > max_ulp_sd) max_ulp_sd = d;
            sum_ulp_sd += d;
        }

        // --- ULP: cos, float input ---
        const float *pcf = cf.memptr();
        int32_t max_ulp_cf = 0;
        int64_t sum_ulp_cf = 0;
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::cos((double)pxf[j]);
            int32_t d = ulp_dist(pcf[j], ref);
            if (d > max_ulp_cf) max_ulp_cf = d;
            sum_ulp_cf += d;
        }

        // --- ULP: cos, double input ---
        const float *pcd = cd_out.memptr();
        int32_t max_ulp_cd = 0;
        int64_t sum_ulp_cd = 0;
        for (size_t j = 0; j < n; ++j)
        {
            float ref = (float)std::cos(pxd[j]);
            int32_t d = ulp_dist(pcd[j], ref);
            if (d > max_ulp_cd) max_ulp_cd = d;
            sum_ulp_cd += d;
        }

        printf("%-18s | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %5.2f  max %7d\n",
               ranges[r].label,
               (double)sum_ulp_sf / (double)n, max_ulp_sf,
               (double)sum_ulp_sd / (double)n, max_ulp_sd,
               (double)sum_ulp_cf / (double)n, max_ulp_cf,
               (double)sum_ulp_cd / (double)n, max_ulp_cd);
    }
}

int main()
{
    run_performance();
    run_accuracy();
    return 0;
}