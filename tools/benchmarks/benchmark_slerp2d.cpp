// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_slerp2d benchmark_slerp2d.cpp  /sjc/quadriga-lib/lib/libquadriga.a

#include "quadriga_math.hpp"
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
// Scalar 2D slerp reference (double precision): chain 3x 1D slerp
//   E = slerp(A, B, u)
//   F = slerp(C, D, u)
//   X = slerp(E, F, v)
// ============================================================================
template <typename dtype>
static inline void slerp2d_scalar(dtype Ar, dtype Ai, dtype Br, dtype Bi,
                                  dtype Cr, dtype Ci, dtype Dr, dtype Di,
                                  dtype u, dtype v,
                                  dtype &Xr, dtype &Xi)
{
    dtype Er, Ei, Fr, Fi;
    slerp_complex_mf<dtype>(Ar, Ai, Br, Bi, u, Er, Ei);
    slerp_complex_mf<dtype>(Cr, Ci, Dr, Di, u, Fr, Fi);
    slerp_complex_mf<dtype>(Er, Ei, Fr, Fi, v, Xr, Xi);
}

// ============================================================================
// PART 1: Performance benchmark (size sweep)
// ============================================================================
static void run_performance()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    printf("=== PERFORMANCE: 2D SLERP (chained 3x1D) ===\n");
    printf("    E = slerp(A,B,u), F = slerp(C,D,u), X = slerp(E,F,v)\n\n");
    printf("%-9s | %-40s | %-40s | %-40s | %-13s | %-13s\n",
           "", "AVX2 / float", "AVX2 / double", "Scalar / double",
           "Speedup f/d", "Speedup d/d");
    printf("%.*s\n", 190, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // --- Generate test data ---
        // 4 corners A,B,C,D with angular separation delta_u (~1 rad) along azimuth
        // and delta_v (~0.5 rad) along elevation
        arma::fvec Ar(n), Ai(n), Br(n), Bi(n), Cr(n), Ci(n), Dr(n), Di(n), u(n), v(n);
        arma::vec dAr(n), dAi(n), dBr(n), dBi(n), dCr(n), dCi(n), dDr(n), dDi(n), du(n), dv(n);

        const double delta_u = 1.0;  // azimuth separation
        const double delta_v = 0.5;  // elevation separation

        for (size_t i = 0; i < n; ++i)
        {
            double t = (double)i / (double)(n > 1 ? n - 1 : 1);
            double theta = -M_PI + 2.0 * M_PI * t;
            double amp = 0.5 + 0.5 * std::cos(theta * 3.0); // amplitude modulation

            // A = base direction
            Ar(i) = (float)(amp * std::cos(theta));
            Ai(i) = (float)(amp * std::sin(theta));
            // B = A rotated by delta_u
            Br(i) = (float)(amp * std::cos(theta + delta_u));
            Bi(i) = (float)(amp * std::sin(theta + delta_u));
            // C = A rotated by delta_v (orthogonal-ish separation)
            Cr(i) = (float)(amp * std::cos(theta + delta_v));
            Ci(i) = (float)(amp * std::sin(theta + delta_v));
            // D = A rotated by delta_u + delta_v
            Dr(i) = (float)(amp * std::cos(theta + delta_u + delta_v));
            Di(i) = (float)(amp * std::sin(theta + delta_u + delta_v));

            u(i) = (float)t;
            v(i) = (float)(1.0 - t); // vary v inversely for diversity

            dAr(i) = (double)Ar(i); dAi(i) = (double)Ai(i);
            dBr(i) = (double)Br(i); dBi(i) = (double)Bi(i);
            dCr(i) = (double)Cr(i); dCi(i) = (double)Ci(i);
            dDr(i) = (double)Dr(i); dDi(i) = (double)Di(i);
            du(i) = (double)u(i);   dv(i) = (double)v(i);
        }

        arma::fvec Er_f, Ei_f, Fr_f, Fi_f, Xr_f, Xi_f;

        // --- 1. AVX2 chained 3x1D slerp (float input) ---
        for (int i = 0; i < n_warmup; ++i)
        {
            quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, u, Er_f, Ei_f);
            quadriga_lib::fast_slerp(Cr, Ci, Dr, Di, u, Fr_f, Fi_f);
            quadriga_lib::fast_slerp(Er_f, Ei_f, Fr_f, Fi_f, v, Xr_f, Xi_f);
        }

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
        {
            quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, u, Er_f, Ei_f);
            quadriga_lib::fast_slerp(Cr, Ci, Dr, Di, u, Fr_f, Fi_f);
            quadriga_lib::fast_slerp(Er_f, Ei_f, Fr_f, Fi_f, v, Xr_f, Xi_f);
        }
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 chained 3x1D slerp (double input) ---
        arma::fvec Er_d, Ei_d, Fr_d, Fi_d, Xr_d, Xi_d;
        for (int i = 0; i < n_warmup; ++i)
        {
            quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, du, Er_d, Ei_d);
            quadriga_lib::fast_slerp(dCr, dCi, dDr, dDi, du, Fr_d, Fi_d);
            quadriga_lib::fast_slerp(Er_d, Ei_d, Fr_d, Fi_d, v, Xr_d, Xi_d); // E,F are fvec → use float weight
        }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
        {
            quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, du, Er_d, Ei_d);
            quadriga_lib::fast_slerp(dCr, dCi, dDr, dDi, du, Fr_d, Fi_d);
            quadriga_lib::fast_slerp(Er_d, Ei_d, Fr_d, Fi_d, v, Xr_d, Xi_d); // E,F are fvec → use float weight
        }
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar 2D slerp (double, single-threaded) ---
        arma::fvec Xr_s(n), Xi_s(n);
        const double *pdAr = dAr.memptr(), *pdAi = dAi.memptr();
        const double *pdBr = dBr.memptr(), *pdBi = dBi.memptr();
        const double *pdCr = dCr.memptr(), *pdCi = dCi.memptr();
        const double *pdDr = dDr.memptr(), *pdDi = dDi.memptr();
        const double *pdu = du.memptr(), *pdv = dv.memptr();
        float *pXrs = Xr_s.memptr(), *pXis = Xi_s.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                double xr, xi;
                slerp2d_scalar<double>(pdAr[j], pdAi[j], pdBr[j], pdBi[j],
                                       pdCr[j], pdCi[j], pdDr[j], pdDi[j],
                                       pdu[j], pdv[j], xr, xi);
                pXrs[j] = (float)xr;
                pXis[j] = (float)xi;
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                double xr, xi;
                slerp2d_scalar<double>(pdAr[j], pdAi[j], pdBr[j], pdBi[j],
                                       pdCr[j], pdCi[j], pdDr[j], pdDi[j],
                                       pdu[j], pdv[j], xr, xi);
                pXrs[j] = (float)xr;
                pXis[j] = (float)xi;
            }
        double dt_scalar_d = (now_ms() - t0) / n_reps;

        // --- Report ---
        double quads_M = (double)n / 1e6;
        printf("n = %7zu | %7.3f ms (%7.1f Mquads/s) | %7.3f ms (%7.1f Mquads/s) "
               "| %7.3f ms (%7.1f Mquads/s) "
               "| %10.1fx | %10.1fx\n",
               n,
               dt_avx_f, quads_M / dt_avx_f * 1e3,
               dt_avx_d, quads_M / dt_avx_d * 1e3,
               dt_scalar_d, quads_M / dt_scalar_d * 1e3,
               dt_scalar_d / dt_avx_f,
               dt_scalar_d / dt_avx_d);
    }
}

// ============================================================================
// PART 2: Accuracy benchmark (angular-separation sweep)
//
// Varies the angular separation between A↔B (delta_u) and C↔D/A↔C (delta_v)
// Reference: double-precision scalar chained 3x1D slerp
//
// Metrics:
//   Amplitude error in ULP:  |X| vs |X_ref|
//   Phase error in radians:  angle(X) vs angle(X_ref)
//     Phase only evaluated when reference amplitude > 1e-10
// ============================================================================
static void run_accuracy()
{
    const size_t n = 1048576;

    struct Scenario
    {
        const char *label;
        double delta_u; // A↔B and C↔D angular separation
        double delta_v; // A↔C and B↔D angular separation
    };
    const Scenario scenarios[] = {
        {"du=0.001 dv=0.001",  0.001,  0.001},
        {"du=0.01  dv=0.01",   0.01,   0.01},
        {"du=0.1   dv=0.1",    0.1,    0.1},
        {"du=1.0   dv=0.5",    1.0,    0.5},
        {"du=pi/2  dv=pi/4",   M_PI/2, M_PI/4},
        {"du=pi-0.01 dv=0.5",  M_PI-0.01, 0.5},
        {"du=1.0   dv=pi-0.01",1.0,    M_PI-0.01},
        {"du=0.1   dv=0.001",  0.1,    0.001},  // anisotropic: tight v, moderate u
        {"du=0.001 dv=0.1",    0.001,  0.1},    // anisotropic: tight u, moderate v
    };
    const int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    printf("\n=== ACCURACY: 2D SLERP (n = %zu, vs double-precision scalar reference) ===\n\n", n);
    printf("%-22s | %-28s | %-28s | %-30s | %-30s\n",
           "Scenario", "Amp ULP (float in)", "Amp ULP (double in)",
           "Phase rad (float in)", "Phase rad (double in)");
    printf("%.*s\n", 152, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (int s = 0; s < n_scenarios; ++s)
    {
        double delta_u = scenarios[s].delta_u;
        double delta_v = scenarios[s].delta_v;

        // --- Generate test data ---
        arma::fvec Ar(n), Ai(n), Br(n), Bi(n), Cr(n), Ci(n), Dr(n), Di(n), u_f(n), v_f(n);
        arma::vec dAr(n), dAi(n), dBr(n), dBi(n), dCr(n), dCi(n), dDr(n), dDi(n), du(n), dv(n);

        for (size_t i = 0; i < n; ++i)
        {
            double t = (double)i / (double)(n > 1 ? n - 1 : 1);
            double theta = -M_PI + 2.0 * M_PI * t;
            double jitter = 1e-6 * std::sin((double)(i * 17 + 3));
            theta += jitter;

            // Vary amplitude: includes near-zero to stress the fallback paths
            double amp = 0.3 + 0.7 * std::abs(std::sin(theta * 2.7));

            // A = base
            double aA = theta;
            // B = A + delta_u along the unit circle
            double aB = theta + delta_u;
            // C = A + delta_v
            double aC = theta + delta_v;
            // D = A + delta_u + delta_v
            double aD = theta + delta_u + delta_v;

            Ar(i) = (float)(amp * std::cos(aA));
            Ai(i) = (float)(amp * std::sin(aA));
            Br(i) = (float)(amp * std::cos(aB));
            Bi(i) = (float)(amp * std::sin(aB));
            Cr(i) = (float)(amp * std::cos(aC));
            Ci(i) = (float)(amp * std::sin(aC));
            Dr(i) = (float)(amp * std::cos(aD));
            Di(i) = (float)(amp * std::sin(aD));

            u_f(i) = (float)t;
            v_f(i) = (float)(1.0 - t);

            dAr(i) = (double)Ar(i); dAi(i) = (double)Ai(i);
            dBr(i) = (double)Br(i); dBi(i) = (double)Bi(i);
            dCr(i) = (double)Cr(i); dCi(i) = (double)Ci(i);
            dDr(i) = (double)Dr(i); dDi(i) = (double)Di(i);
            du(i) = (double)u_f(i); dv(i) = (double)v_f(i);
        }

        // --- AVX2: float input (chained 3x1D) ---
        arma::fvec Er_f, Ei_f, Fr_f, Fi_f, Xr_f, Xi_f;
        quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, u_f, Er_f, Ei_f);
        quadriga_lib::fast_slerp(Cr, Ci, Dr, Di, u_f, Fr_f, Fi_f);
        quadriga_lib::fast_slerp(Er_f, Ei_f, Fr_f, Fi_f, v_f, Xr_f, Xi_f);

        // --- AVX2: double input (chained 3x1D) ---
        arma::fvec Er_d, Ei_d, Fr_d, Fi_d, Xr_d, Xi_d;

        quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, du, Er_d, Ei_d);
        quadriga_lib::fast_slerp(dCr, dCi, dDr, dDi, du, Fr_d, Fi_d);
        // 3rd stage: E,F are fvec outputs from stage 1&2 → use float weight
        quadriga_lib::fast_slerp(Er_d, Ei_d, Fr_d, Fi_d, v_f, Xr_d, Xi_d);

        // --- Measure errors ---
        const float *pXr_f = Xr_f.memptr(), *pXi_f = Xi_f.memptr();
        const float *pXr_d = Xr_d.memptr(), *pXi_d = Xi_d.memptr();
        const double *pdAr = dAr.memptr(), *pdAi = dAi.memptr();
        const double *pdBr = dBr.memptr(), *pdBi = dBi.memptr();
        const double *pdCr = dCr.memptr(), *pdCi = dCi.memptr();
        const double *pdDr = dDr.memptr(), *pdDi = dDi.memptr();
        const double *pdu = du.memptr(), *pdv_p = dv.memptr();

        int32_t max_amp_ulp_f = 0, max_amp_ulp_d = 0;
        int64_t sum_amp_ulp_f = 0, sum_amp_ulp_d = 0;

        double max_phase_f = 0.0, max_phase_d = 0.0;
        double sum_phase_f = 0.0, sum_phase_d = 0.0;
        size_t n_phase = 0;

        const double amp_threshold = 1e-10;

        for (size_t j = 0; j < n; ++j)
        {
            // Double-precision scalar reference (chained 3x1D)
            double ref_xr, ref_xi;
            slerp2d_scalar<double>(pdAr[j], pdAi[j], pdBr[j], pdBi[j],
                                   pdCr[j], pdCi[j], pdDr[j], pdDi[j],
                                   pdu[j], pdv_p[j], ref_xr, ref_xi);

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

            // --- Phase error ---
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

        printf("%-22s | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %.1e  max %.1e | avg %.1e  max %.1e\n",
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