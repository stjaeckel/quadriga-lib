// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_geo benchmark_geo.cpp  /sjc/quadriga-lib/lib/libquadriga.a

#include "quadriga_math.hpp"
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

// Wrap-aware signed angle difference in [-pi, pi]
static inline double angle_diff(double a, double b)
{
    double d = a - b;
    if (d > M_PI) d -= 2.0 * M_PI;
    if (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

// ============================================================================
// PART 1: Performance benchmark (size sweep)
// ============================================================================
static void run_performance_geo2cart()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    printf("=== PERFORMANCE: geo2cart ===\n\n");
    printf("%-9s | %-40s | %-40s | %-40s | %-40s | %-13s | %-13s\n",
           "", "AVX2 / float", "AVX2 / double", "Scalar / float", "Scalar / double",
           "Speedup f", "Speedup d");
    printf("%.*s\n", 220, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // Generate test data: angles spanning full range
        arma::fvec faz(n), fel(n);
        arma::vec daz(n), del_(n);

        for (size_t i = 0; i < n; ++i)
        {
            double t = -M_PI + 2.0 * M_PI * (double)i / (double)(n > 1 ? n - 1 : 1);
            faz(i) = (float)t;
            fel(i) = (float)(t * 0.5); // elevation in [-pi/2, pi/2] range
            daz(i) = (double)faz(i);
            del_(i) = (double)fel(i);
        }

        arma::fvec x, y, z, sAZ, cAZ, sEL, cEL;

        // --- 1. AVX2 geo2cart (float input, with intermediates) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_geo2cart(faz, fel, x, y, z, &sAZ, &cAZ, &sEL, &cEL);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_geo2cart(faz, fel, x, y, z, &sAZ, &cAZ, &sEL, &cEL);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 geo2cart (double input, with intermediates) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_geo2cart(daz, del_, x, y, z, &sAZ, &cAZ, &sEL, &cEL);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_geo2cart(daz, del_, x, y, z, &sAZ, &cAZ, &sEL, &cEL);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar geo2cart (float, single-threaded) ---
        arma::fvec sx(n), sy(n), sz(n);
        const float *paz = faz.memptr(), *pel = fel.memptr();
        float *psx = sx.memptr(), *psy = sy.memptr(), *psz = sz.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                float sa = std::sin(paz[j]), ca = std::cos(paz[j]);
                float se = std::sin(pel[j]), ce = std::cos(pel[j]);
                psx[j] = ce * ca; psy[j] = ce * sa; psz[j] = se;
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                float sa = std::sin(paz[j]), ca = std::cos(paz[j]);
                float se = std::sin(pel[j]), ce = std::cos(pel[j]);
                psx[j] = ce * ca; psy[j] = ce * sa; psz[j] = se;
            }
        double dt_scalar_f = (now_ms() - t0) / n_reps;

        // --- 4. Scalar geo2cart (double, single-threaded, cast to float) ---
        const double *pdaz = daz.memptr(), *pdel = del_.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                double sa = std::sin(pdaz[j]), ca = std::cos(pdaz[j]);
                double se = std::sin(pdel[j]), ce = std::cos(pdel[j]);
                psx[j] = (float)(ce * ca); psy[j] = (float)(ce * sa); psz[j] = (float)se;
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                double sa = std::sin(pdaz[j]), ca = std::cos(pdaz[j]);
                double se = std::sin(pdel[j]), ce = std::cos(pdel[j]);
                psx[j] = (float)(ce * ca); psy[j] = (float)(ce * sa); psz[j] = (float)se;
            }
        double dt_scalar_d = (now_ms() - t0) / n_reps;

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

static void run_performance_cart2geo()
{
    const size_t n_sizes[] = {64, 1024, 16384, 131072, 1048576};
    const int n_reps = 50;
    const int n_warmup = 5;

    printf("\n=== PERFORMANCE: cart2geo ===\n\n");
    printf("%-9s | %-40s | %-40s | %-40s | %-40s | %-13s | %-13s\n",
           "", "AVX2 / float", "AVX2 / double", "Scalar / float", "Scalar / double",
           "Speedup f", "Speedup d");
    printf("%.*s\n", 220, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (size_t n : n_sizes)
    {
        // Generate test data: points on the unit sphere
        arma::fvec fx(n), fy(n), fz(n);
        arma::vec dx(n), dy(n), dz(n);

        for (size_t i = 0; i < n; ++i)
        {
            double t = -M_PI + 2.0 * M_PI * (double)i / (double)(n > 1 ? n - 1 : 1);
            double e = std::sin(t * 0.7);
            double ce = std::cos(std::asin(e));
            fx(i) = (float)(ce * std::cos(t)); fy(i) = (float)(ce * std::sin(t)); fz(i) = (float)e;
            dx(i) = (double)fx(i); dy(i) = (double)fy(i); dz(i) = (double)fz(i);
        }

        arma::fvec az, el;

        // --- 1. AVX2 cart2geo (float input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_cart2geo(fx, fy, fz, az, el);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_cart2geo(fx, fy, fz, az, el);
        double dt_avx_f = (now_ms() - t0) / n_reps;

        // --- 2. AVX2 cart2geo (double input) ---
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::fast_cart2geo(dx, dy, dz, az, el);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::fast_cart2geo(dx, dy, dz, az, el);
        double dt_avx_d = (now_ms() - t0) / n_reps;

        // --- 3. Scalar cart2geo (float, single-threaded) ---
        arma::fvec saz(n), sel(n);
        const float *pfx = fx.memptr(), *pfy = fy.memptr(), *pfz = fz.memptr();
        float *psaz = saz.memptr(), *psel = sel.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                psaz[j] = std::atan2(pfy[j], pfx[j]);
                float zc = pfz[j] > 1.0f ? 1.0f : (pfz[j] < -1.0f ? -1.0f : pfz[j]);
                psel[j] = std::asin(zc);
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                psaz[j] = std::atan2(pfy[j], pfx[j]);
                float zc = pfz[j] > 1.0f ? 1.0f : (pfz[j] < -1.0f ? -1.0f : pfz[j]);
                psel[j] = std::asin(zc);
            }
        double dt_scalar_f = (now_ms() - t0) / n_reps;

        // --- 4. Scalar cart2geo (double, single-threaded, cast to float) ---
        const double *pdx = dx.memptr(), *pdy = dy.memptr(), *pdz = dz.memptr();

        for (int i = 0; i < n_warmup; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                psaz[j] = (float)std::atan2(pdy[j], pdx[j]);
                double zc = pdz[j] > 1.0 ? 1.0 : (pdz[j] < -1.0 ? -1.0 : pdz[j]);
                psel[j] = (float)std::asin(zc);
            }

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                psaz[j] = (float)std::atan2(pdy[j], pdx[j]);
                double zc = pdz[j] > 1.0 ? 1.0 : (pdz[j] < -1.0 ? -1.0 : pdz[j]);
                psel[j] = (float)std::asin(zc);
            }
        double dt_scalar_d = (now_ms() - t0) / n_reps;

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
// PART 2: Accuracy benchmark
//
// geo2cart: compare AVX2 x,y,z (and intermediates) against double-precision
//           std::sin / std::cos reference. Report ULP on each output.
//
// cart2geo: compare AVX2 az,el against double-precision std::atan2 / std::asin
//           reference. Report ULP for el, wrap-aware radians for az.
// ============================================================================
static void run_accuracy_geo2cart()
{
    const size_t n = 1048576;

    printf("\n=== ACCURACY: geo2cart (n = %zu, vs double-precision scalar reference) ===\n\n", n);
    printf("%-10s | %-28s | %-28s | %-28s | %-28s | %-28s\n",
           "Input", "x  (ULP)", "y  (ULP)", "z  (ULP)", "sinAZ (ULP)", "cosEL (ULP)");
    printf("%.*s\n", 158, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    // Generate test data: full angular range with jitter
    arma::fvec faz(n), fel(n);
    arma::vec daz(n), del_(n);

    for (size_t i = 0; i < n; ++i)
    {
        double az = -M_PI + 2.0 * M_PI * (double)i / (double)(n - 1);
        double el = -M_PI / 2.0 + M_PI * (double)i / (double)(n - 1);
        double jitter = 1e-6 * std::sin((double)(i * 17 + 3));
        az += jitter;
        el += jitter * 0.5;
        faz(i) = (float)az;
        fel(i) = (float)el;
        daz(i) = (double)faz(i);
        del_(i) = (double)fel(i);
    }

    // Run AVX2 versions
    arma::fvec x_f, y_f, z_f, sAZ_f, cAZ_f, sEL_f, cEL_f;
    arma::fvec x_d, y_d, z_d, sAZ_d, cAZ_d, sEL_d, cEL_d;
    quadriga_lib::fast_geo2cart(faz, fel, x_f, y_f, z_f, &sAZ_f, &cAZ_f, &sEL_f, &cEL_f);
    quadriga_lib::fast_geo2cart(daz, del_, x_d, y_d, z_d, &sAZ_d, &cAZ_d, &sEL_d, &cEL_d);

    // Measure ULP error on 5 representative outputs: x, y, z, sinAZ, cosEL
    for (int pass = 0; pass < 2; ++pass)
    {
        const char *label = (pass == 0) ? "float" : "double";
        const float *px = (pass == 0) ? x_f.memptr() : x_d.memptr();
        const float *py = (pass == 0) ? y_f.memptr() : y_d.memptr();
        const float *pz = (pass == 0) ? z_f.memptr() : z_d.memptr();
        const float *psAZ = (pass == 0) ? sAZ_f.memptr() : sAZ_d.memptr();
        const float *pcEL = (pass == 0) ? cEL_f.memptr() : cEL_d.memptr();

        int32_t max_x = 0, max_y = 0, max_z = 0, max_sAZ = 0, max_cEL = 0;
        int64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_sAZ = 0, sum_cEL = 0;

        for (size_t j = 0; j < n; ++j)
        {
            // Double-precision reference
            double a = (pass == 0) ? (double)faz(j) : daz(j);
            double e = (pass == 0) ? (double)fel(j) : del_(j);
            double ref_sa = std::sin(a), ref_ca = std::cos(a);
            double ref_se = std::sin(e), ref_ce = std::cos(e);
            double ref_x = ref_ce * ref_ca;
            double ref_y = ref_ce * ref_sa;
            double ref_z = ref_se;

            int32_t dx = ulp_dist(px[j], (float)ref_x);
            int32_t dy = ulp_dist(py[j], (float)ref_y);
            int32_t dz = ulp_dist(pz[j], (float)ref_z);
            int32_t dsAZ = ulp_dist(psAZ[j], (float)ref_sa);
            int32_t dcEL = ulp_dist(pcEL[j], (float)ref_ce);

            if (dx > max_x) max_x = dx; sum_x += dx;
            if (dy > max_y) max_y = dy; sum_y += dy;
            if (dz > max_z) max_z = dz; sum_z += dz;
            if (dsAZ > max_sAZ) max_sAZ = dsAZ; sum_sAZ += dsAZ;
            if (dcEL > max_cEL) max_cEL = dcEL; sum_cEL += dcEL;
        }

        double dn = (double)n;
        printf("%-10s | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %5.2f  max %7d   | avg %5.2f  max %7d\n",
               label,
               (double)sum_x / dn, max_x,
               (double)sum_y / dn, max_y,
               (double)sum_z / dn, max_z,
               (double)sum_sAZ / dn, max_sAZ,
               (double)sum_cEL / dn, max_cEL);
    }
}

static void run_accuracy_cart2geo()
{
    const size_t n = 1048576;

    printf("\n=== ACCURACY: cart2geo (n = %zu, vs double-precision scalar reference) ===\n\n", n);
    printf("%-10s | %-30s | %-30s | %-30s\n",
           "Input", "az  (ULP)", "el  (ULP)", "az  (rad err)");
    printf("%.*s\n", 110, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    // Generate test data: points on the unit sphere (full coverage)
    arma::fvec fx(n), fy(n), fz(n);
    arma::vec dx(n), dy(n), dz(n);

    for (size_t i = 0; i < n; ++i)
    {
        double t = -M_PI + 2.0 * M_PI * (double)i / (double)(n - 1);
        double e = std::sin(t * 0.7);               // elevation mapped into [-1, 1]
        double ce = std::sqrt(1.0 - e * e);
        double jitter = 1e-7 * std::sin((double)(i * 13 + 7));
        fx(i) = (float)(ce * std::cos(t) + jitter);
        fy(i) = (float)(ce * std::sin(t) + jitter);
        fz(i) = (float)e;
        dx(i) = (double)fx(i); dy(i) = (double)fy(i); dz(i) = (double)fz(i);
    }

    // Run AVX2 versions
    arma::fvec az_f, el_f, az_d, el_d;
    quadriga_lib::fast_cart2geo(fx, fy, fz, az_f, el_f);
    quadriga_lib::fast_cart2geo(dx, dy, dz, az_d, el_d);

    for (int pass = 0; pass < 2; ++pass)
    {
        const char *label = (pass == 0) ? "float" : "double";
        const float *paz = (pass == 0) ? az_f.memptr() : az_d.memptr();
        const float *pel = (pass == 0) ? el_f.memptr() : el_d.memptr();

        int32_t max_az_ulp = 0, max_el_ulp = 0;
        int64_t sum_az_ulp = 0, sum_el_ulp = 0;
        double max_az_rad = 0.0, sum_az_rad = 0.0;

        for (size_t j = 0; j < n; ++j)
        {
            double xi = (pass == 0) ? (double)fx(j) : dx(j);
            double yi = (pass == 0) ? (double)fy(j) : dy(j);
            double zi = (pass == 0) ? (double)fz(j) : dz(j);
            double zc = zi > 1.0 ? 1.0 : (zi < -1.0 ? -1.0 : zi);

            double ref_az = std::atan2(yi, xi);
            double ref_el = std::asin(zc);

            // ULP for az and el
            int32_t da = ulp_dist(paz[j], (float)ref_az);
            int32_t de = ulp_dist(pel[j], (float)ref_el);
            if (da > max_az_ulp) max_az_ulp = da; sum_az_ulp += da;
            if (de > max_el_ulp) max_el_ulp = de; sum_el_ulp += de;

            // Wrap-aware radian error for az (more meaningful near ±pi)
            double rad_err = std::abs(angle_diff((double)paz[j], ref_az));
            if (rad_err > max_az_rad) max_az_rad = rad_err;
            sum_az_rad += rad_err;
        }

        double dn = (double)n;
        printf("%-10s | avg %5.2f  max %7d     | avg %5.2f  max %7d     | avg %.1e  max %.1e\n",
               label,
               (double)sum_az_ulp / dn, max_az_ulp,
               (double)sum_el_ulp / dn, max_el_ulp,
               sum_az_rad / dn, max_az_rad);
    }
}

// ============================================================================
// PART 3: Round-trip accuracy (geo2cart → cart2geo and cart2geo → geo2cart)
//
// This tests the composition of the two functions, which is the usage pattern
// in the interpolation kernel: input angles → Cartesian → rotate → back to angles.
// ============================================================================
static void run_roundtrip()
{
    const size_t n = 1048576;

    printf("\n=== ROUND-TRIP ACCURACY (n = %zu) ===\n\n", n);
    printf("%-26s | %-30s | %-30s\n",
           "Path", "az error (rad)", "el error (rad)");
    printf("%.*s\n", 92, "----------------------------------------------------------------------------------------------------------------------------");

    // Generate test data: well-conditioned angles away from poles
    arma::fvec faz(n), fel(n);
    arma::vec daz(n), del_(n);

    for (size_t i = 0; i < n; ++i)
    {
        double t = -M_PI + 2.0 * M_PI * (double)i / (double)(n - 1);
        double e = 0.8 * std::sin(t * 1.3); // stay away from ±pi/2 poles
        faz(i) = (float)t;
        fel(i) = (float)e;
        daz(i) = (double)faz(i);
        del_(i) = (double)fel(i);
    }

    // --- geo2cart → cart2geo (float) ---
    {
        arma::fvec x, y, z, az_out, el_out;
        quadriga_lib::fast_geo2cart(faz, fel, x, y, z);
        quadriga_lib::fast_cart2geo(x, y, z, az_out, el_out);

        double max_az = 0.0, max_el = 0.0, sum_az = 0.0, sum_el = 0.0;
        for (size_t j = 0; j < n; ++j)
        {
            double da = std::abs(angle_diff((double)az_out(j), (double)faz(j)));
            double de = std::abs((double)el_out(j) - (double)fel(j));
            if (da > max_az) max_az = da;
            if (de > max_el) max_el = de;
            sum_az += da; sum_el += de;
        }
        printf("geo2cart→cart2geo (float)  | avg %.1e  max %.1e | avg %.1e  max %.1e\n",
               sum_az / (double)n, max_az, sum_el / (double)n, max_el);
    }

    // --- geo2cart → cart2geo (double) ---
    {
        arma::fvec x, y, z, az_out, el_out;
        quadriga_lib::fast_geo2cart(daz, del_, x, y, z);
        quadriga_lib::fast_cart2geo(x, y, z, az_out, el_out);

        double max_az = 0.0, max_el = 0.0, sum_az = 0.0, sum_el = 0.0;
        for (size_t j = 0; j < n; ++j)
        {
            double da = std::abs(angle_diff((double)az_out(j), daz(j)));
            double de = std::abs((double)el_out(j) - del_(j));
            if (da > max_az) max_az = da;
            if (de > max_el) max_el = de;
            sum_az += da; sum_el += de;
        }
        printf("geo2cart→cart2geo (double) | avg %.1e  max %.1e | avg %.1e  max %.1e\n",
               sum_az / (double)n, max_az, sum_el / (double)n, max_el);
    }

    // --- cart2geo → geo2cart (float) ---
    {
        arma::fvec fx(n), fy(n), fz(n);
        for (size_t i = 0; i < n; ++i)
        {
            double t = -M_PI + 2.0 * M_PI * (double)i / (double)(n - 1);
            double e = 0.8 * std::sin(t * 1.3);
            double ce = std::cos(e);
            fx(i) = (float)(ce * std::cos(t));
            fy(i) = (float)(ce * std::sin(t));
            fz(i) = (float)std::sin(e);
        }

        arma::fvec az, el, x_out, y_out, z_out;
        quadriga_lib::fast_cart2geo(fx, fy, fz, az, el);
        quadriga_lib::fast_geo2cart(az, el, x_out, y_out, z_out);

        double max_err = 0.0, sum_err = 0.0;
        for (size_t j = 0; j < n; ++j)
        {
            // Euclidean distance between original and round-tripped point
            double dx = (double)x_out(j) - (double)fx(j);
            double dy = (double)y_out(j) - (double)fy(j);
            double dz = (double)z_out(j) - (double)fz(j);
            double err = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (err > max_err) max_err = err;
            sum_err += err;
        }
        printf("cart2geo→geo2cart (float)  | avg %.1e  max %.1e (Euclidean dist on unit sphere)\n",
               sum_err / (double)n, max_err);
    }
}

int main()
{
    run_performance_geo2cart();
    run_performance_cart2geo();
    run_accuracy_geo2cart();
    run_accuracy_cart2geo();
    run_roundtrip();
    return 0;
}
