// Compile: 
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_slerp benchmark_slerp.cpp  /sjc/quadriga-lib/lib/libquadriga.a

#include "quadriga_tools.hpp"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

// Scalar reference for comparison (bypasses AVX2 entirely)
#include "slerp.h"

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

// Wrap-aware signed phase difference in [-pi, pi]
static inline double phase_diff(double a, double b)
{
    double d = a - b;
    if (d > M_PI) d -= 2.0 * M_PI;
    if (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

// ============================================================================
// PART 1: Performance benchmark (size sweep)
// ============================================================================
static void run_performance()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    printf("=== PERFORMANCE (slerp, angular separation ~1 rad) ===\n\n");
    printf("%-9s | %-40s | %-40s | %-40s | %-40s | %-13s | %-13s\n",
           "", "AVX2 / float", "AVX2 / double", "Scalar / float", "Scalar / double",
           "Speedup f", "Speedup d");
    printf("%.*s\n", 220, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // --- Generate test data: angular separation ~1 rad, varying weight ---
        arma::fvec Ar(n), Ai(n), Br(n), Bi(n), w(n);
        arma::vec dAr(n), dAi(n), dBr(n), dBi(n), dw(n);

        for (size_t i = 0; i < n; ++i)
        {
            float t = (float)i / (float)n;
            Ar(i) = std::cos(t * 5.0f);
            Ai(i) = std::sin(t * 5.0f);
            Br(i) = std::cos(t * 5.0f + 1.0f);
            Bi(i) = std::sin(t * 5.0f + 1.0f);
            w(i) = t;
            dAr(i) = (double)Ar(i);  dAi(i) = (double)Ai(i);
            dBr(i) = (double)Br(i);  dBi(i) = (double)Bi(i);
            dw(i) = (double)w(i);
        }

        arma::fvec Xr_f, Xi_f;

        // --- 1. AVX2 slerp (float input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr_f, Xi_f);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr_f, Xi_f);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 slerp (double input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, dw, Xr_f, Xi_f);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, dw, Xr_f, Xi_f);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar slerp (float, single-threaded) ---
        arma::fvec Xr_s(n), Xi_s(n);
        const float *pAr = Ar.memptr(), *pAi = Ai.memptr();
        const float *pBr = Br.memptr(), *pBi = Bi.memptr();
        const float *pw = w.memptr();
        float *pXr = Xr_s.memptr(), *pXi = Xi_s.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
                slerp_complex_mf<float>(pAr[j], pAi[j], pBr[j], pBi[j], pw[j], pXr[j], pXi[j]);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
                slerp_complex_mf<float>(pAr[j], pAi[j], pBr[j], pBi[j], pw[j], pXr[j], pXi[j]);
        double dt_scalar_f = (now_ms() - t0) / n_reps;

        // --- 4. Scalar slerp (double, single-threaded) ---
        const double *pdAr = dAr.memptr(), *pdAi = dAi.memptr();
        const double *pdBr = dBr.memptr(), *pdBi = dBi.memptr();
        const double *pdw = dw.memptr();
        arma::fvec Xr_sd(n), Xi_sd(n);
        float *pXrd = Xr_sd.memptr(), *pXid = Xi_sd.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                double xr, xi;
                slerp_complex_mf<double>(pdAr[j], pdAi[j], pdBr[j], pdBi[j], pdw[j], xr, xi);
                pXrd[j] = (float)xr;
                pXid[j] = (float)xi;
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                double xr, xi;
                slerp_complex_mf<double>(pdAr[j], pdAi[j], pdBr[j], pdBi[j], pdw[j], xr, xi);
                pXrd[j] = (float)xr;
                pXid[j] = (float)xi;
            }
        double dt_scalar_d = (now_ms() - t0) / n_reps;

        // --- Report ---
        double pairs_M = (double)n / 1e6;
        printf("n = %7zu | %7.3f ms (%7.1f Mpairs/s) | %7.3f ms (%7.1f Mpairs/s) "
               "| %7.3f ms (%7.1f Mpairs/s) | %7.3f ms (%7.1f Mpairs/s) "
               "| %10.1fx | %10.1fx\n",
               n,
               dt_avx_f, pairs_M / dt_avx_f * 1e3,
               dt_avx_d, pairs_M / dt_avx_d * 1e3,
               dt_scalar_f, pairs_M / dt_scalar_f * 1e3,
               dt_scalar_d, pairs_M / dt_scalar_d * 1e3,
               dt_scalar_f / dt_avx_f,
               dt_scalar_d / dt_avx_d);
    }
}

// ============================================================================
// PART 2: Accuracy benchmark (angular-separation sweep at fixed large n)
//
// Metrics (vs double-precision scalar reference):
//   Amplitude error in ULP:  |X| vs |X_ref|, where |X| = sqrt(Xr² + Xi²)
//   Phase error in radians:  atan2(Xi, Xr) vs atan2(Xi_ref, Xr_ref)
//     Phase is only evaluated when reference amplitude > 1e-10 (otherwise
//     the phase is meaningless and would add noise to the statistics).
// ============================================================================
static void run_accuracy()
{
    const size_t n = 1048576;

    struct Scenario
    {
        const char *label;
        double delta; // Angular separation between A and B (radians)
    };
    const Scenario scenarios[] = {
        {"delta = 0.001",    0.001},
        {"delta = 0.01",     0.01},
        {"delta = 0.1",      0.1},
        {"delta = 1.0",      1.0},
        {"delta = pi/2",     M_PI / 2.0},
        {"delta = pi-0.01",  M_PI - 0.01},
    };
    const int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    printf("\n=== ACCURACY (n = %zu, vs double-precision scalar reference) ===\n\n", n);
    printf("%-18s | %-28s | %-28s | %-30s | %-30s\n",
           "Scenario", "Amp ULP (float in)", "Amp ULP (double in)",
           "Phase rad (float in)", "Phase rad (double in)");
    printf("%.*s\n", 148, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (int s = 0; s < n_scenarios; ++s)
    {
        double delta = scenarios[s].delta;

        // --- Generate test data ---
        arma::fvec Ar(n), Ai(n), Br(n), Bi(n), w(n);
        arma::vec dAr(n), dAi(n), dBr(n), dBi(n), dw(n);

        for (size_t i = 0; i < n; ++i)
        {
            double theta = -M_PI + 2.0 * M_PI * (double)i / (double)(n > 1 ? n - 1 : 1);
            double jitter = 1e-6 * std::sin((double)(i * 17 + 3));
            theta += jitter;

            Ar(i) = (float)std::cos(theta);
            Ai(i) = (float)std::sin(theta);
            Br(i) = (float)std::cos(theta + delta);
            Bi(i) = (float)std::sin(theta + delta);
            w(i) = (float)((double)i / (double)(n > 1 ? n - 1 : 1)); // weight in [0, 1]

            dAr(i) = (double)Ar(i);  dAi(i) = (double)Ai(i);
            dBr(i) = (double)Br(i);  dBi(i) = (double)Bi(i);
            dw(i) = (double)w(i);
        }

        arma::fvec Xr_f, Xi_f, Xr_d, Xi_d;

        // --- AVX2: float input ---
        quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr_f, Xi_f);

        // --- AVX2: double input ---
        quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, dw, Xr_d, Xi_d);

        // --- Measure amplitude (ULP) and phase (radians) error ---
        const float *pXr_f = Xr_f.memptr(), *pXi_f = Xi_f.memptr();
        const float *pXr_d = Xr_d.memptr(), *pXi_d = Xi_d.memptr();
        const double *pdAr = dAr.memptr(), *pdAi = dAi.memptr();
        const double *pdBr = dBr.memptr(), *pdBi = dBi.memptr();
        const double *pdw = dw.memptr();

        // Amplitude ULP
        int32_t max_amp_ulp_f = 0, max_amp_ulp_d = 0;
        int64_t sum_amp_ulp_f = 0, sum_amp_ulp_d = 0;

        // Phase error in radians
        double max_phase_f = 0.0, max_phase_d = 0.0;
        double sum_phase_f = 0.0, sum_phase_d = 0.0;
        size_t n_phase = 0; // count of elements with non-negligible amplitude

        const double amp_threshold = 1e-10; // skip phase when reference amplitude is tiny

        for (size_t j = 0; j < n; ++j)
        {
            // Double-precision scalar reference
            double ref_xr, ref_xi;
            slerp_complex_mf<double>(pdAr[j], pdAi[j], pdBr[j], pdBi[j], pdw[j], ref_xr, ref_xi);

            double ref_amp = std::sqrt(ref_xr * ref_xr + ref_xi * ref_xi);

            // --- Float input path ---
            float avx_amp_f = std::sqrt(pXr_f[j] * pXr_f[j] + pXi_f[j] * pXi_f[j]);
            int32_t da_f = ulp_dist(avx_amp_f, (float)ref_amp);
            if (da_f > max_amp_ulp_f) max_amp_ulp_f = da_f;
            sum_amp_ulp_f += da_f;

            // --- Double input path ---
            float avx_amp_d = std::sqrt(pXr_d[j] * pXr_d[j] + pXi_d[j] * pXi_d[j]);
            int32_t da_d = ulp_dist(avx_amp_d, (float)ref_amp);
            if (da_d > max_amp_ulp_d) max_amp_ulp_d = da_d;
            sum_amp_ulp_d += da_d;

            // --- Phase error (only when amplitude is meaningful) ---
            if (ref_amp > amp_threshold)
            {
                double ref_phase = std::atan2(ref_xi, ref_xr);

                double avx_phase_f = std::atan2((double)pXi_f[j], (double)pXr_f[j]);
                double pe_f = std::abs(phase_diff(avx_phase_f, ref_phase));
                if (pe_f > max_phase_f) max_phase_f = pe_f;
                sum_phase_f += pe_f;

                double avx_phase_d = std::atan2((double)pXi_d[j], (double)pXr_d[j]);
                double pe_d = std::abs(phase_diff(avx_phase_d, ref_phase));
                if (pe_d > max_phase_d) max_phase_d = pe_d;
                sum_phase_d += pe_d;

                n_phase++;
            }
        }

        double avg_phase_f = (n_phase > 0) ? sum_phase_f / (double)n_phase : 0.0;
        double avg_phase_d = (n_phase > 0) ? sum_phase_d / (double)n_phase : 0.0;

        printf("%-18s | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %.1e  max %.1e | avg %.1e  max %.1e\n",
               scenarios[s].label,
               (double)sum_amp_ulp_f / (double)n, max_amp_ulp_f,
               (double)sum_amp_ulp_d / (double)n, max_amp_ulp_d,
               avg_phase_f, max_phase_f,
               avg_phase_d, max_phase_d);
    }
}

int main()
{
    run_performance();
    run_accuracy();
    return 0;
}