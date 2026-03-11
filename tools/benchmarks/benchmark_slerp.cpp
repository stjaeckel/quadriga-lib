// Compile: 
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_slerp benchmark_slerp.cpp  /sjc/quadriga-lib/lib/libquadriga.a

#include "quadriga_tools.hpp"
#include <chrono>
#include <cstdio>
#include <cmath>

// Scalar reference for comparison (bypasses AVX2 entirely)
#include "slerp.h"

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

int main()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    for (size_t n : n_sizes)
    {
        // --- Generate test data ---
        arma::fvec Ar(n), Ai(n), Br(n), Bi(n), w(n);
        arma::fvec Xr_f, Xi_f;

        arma::vec dAr(n), dAi(n), dBr(n), dBi(n), dw(n);

        for (size_t i = 0; i < n; ++i)
        {
            float t = (float)i / (float)n;
            Ar(i) = std::cos(t * 5.0f);
            Ai(i) = std::sin(t * 5.0f);
            Br(i) = std::cos(t * 5.0f + 1.0f);
            Bi(i) = std::sin(t * 5.0f + 1.0f);
            w(i) = t;
            dAr(i) = Ar(i);  dAi(i) = Ai(i);
            dBr(i) = Br(i);  dBi(i) = Bi(i);
            dw(i) = w(i);
        }

        // --- 1. AVX2 path via fast_slerp (float input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr_f, Xi_f);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr_f, Xi_f);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 path via fast_slerp (double input, converted internally) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, dw, Xr_f, Xi_f);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_slerp(dAr, dAi, dBr, dBi, dw, Xr_f, Xi_f);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar reference (float, single-threaded) ---
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

        // --- 4. Scalar reference (double, single-threaded) ---
        arma::fvec Xr_sd(n), Xi_sd(n);
        float *pXrd = Xr_sd.memptr(), *pXid = Xi_sd.memptr();
        const double *pdAr = dAr.memptr(), *pdAi = dAi.memptr();
        const double *pdBr = dBr.memptr(), *pdBi = dBi.memptr();
        const double *pdw = dw.memptr();

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
        printf("n = %7zu | AVX2/f: %7.3f ms (%7.1f Mpairs/s) | AVX2/d: %7.3f ms (%7.1f Mpairs/s) "
               "| Scalar/f: %7.3f ms (%7.1f Mpairs/s) | Scalar/d: %7.3f ms (%7.1f Mpairs/s) "
               "| Speedup f: %5.1fx | Speedup d: %5.1fx\n",
               n,
               dt_avx_f, pairs_M / dt_avx_f * 1e3,
               dt_avx_d, pairs_M / dt_avx_d * 1e3,
               dt_scalar_f, pairs_M / dt_scalar_f * 1e3,
               dt_scalar_d, pairs_M / dt_scalar_d * 1e3,
               dt_scalar_f / dt_avx_f,
               dt_scalar_d / dt_avx_d);
    }

    return 0;
}