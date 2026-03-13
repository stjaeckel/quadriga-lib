// Compile (adjust paths to match your tree):
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_interp benchmark_interp.cpp /sjc/quadriga-lib/lib/libquadriga.a
//
// Note: qd_arrayant_interpolate_avx2.cpp must be compiled into the library
//       (or linked separately) before building this benchmark.

#include "qd_arrayant_functions.hpp"
#include "quadriga_tools.hpp"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

// ============================================================================
//  Utilities
// ============================================================================

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
    if (ia < 0)
        ia = (int32_t)0x80000000 - ia;
    if (ib < 0)
        ib = (int32_t)0x80000000 - ib;
    int32_t d = ia - ib;
    return d < 0 ? -d : d;
}

// Wrap-aware signed phase difference in [-pi, pi]
static inline double phase_diff(double a, double b)
{
    double d = a - b;
    if (d > M_PI)
        d -= 2.0 * M_PI;
    if (d < -M_PI)
        d += 2.0 * M_PI;
    return d;
}

// ============================================================================
//  Synthetic antenna pattern generator
//  Creates a cross-dipole-like pattern on a uniform spherical grid.
//  V-pol ~ cos(elevation), H-pol ~ sin(azimuth)*sin(elevation)
//  with a smooth phase ramp to exercise the SLERP properly.
// ============================================================================
template <typename dtype>
static void generate_pattern(
    size_t n_azimuth, size_t n_elevation, size_t n_elements,
    arma::Cube<dtype> &e_theta_re, arma::Cube<dtype> &e_theta_im,
    arma::Cube<dtype> &e_phi_re, arma::Cube<dtype> &e_phi_im,
    arma::Col<dtype> &azimuth_grid, arma::Col<dtype> &elevation_grid)
{
    azimuth_grid.set_size(n_azimuth);
    elevation_grid.set_size(n_elevation);
    for (size_t i = 0; i < n_azimuth; ++i)
        azimuth_grid(i) = (dtype)(-M_PI + 2.0 * M_PI * (double)i / (double)n_azimuth);
    for (size_t i = 0; i < n_elevation; ++i)
        elevation_grid(i) = (dtype)(-M_PI / 2.0 + M_PI * (double)i / (double)(n_elevation - 1));

    e_theta_re.set_size(n_elevation, n_azimuth, n_elements);
    e_theta_im.set_size(n_elevation, n_azimuth, n_elements);
    e_phi_re.set_size(n_elevation, n_azimuth, n_elements);
    e_phi_im.set_size(n_elevation, n_azimuth, n_elements);

    for (size_t el = 0; el < n_elements; ++el)
    {
        double el_offset = 0.3 * (double)el; // per-element phase offset
        for (size_t a = 0; a < n_azimuth; ++a)
        {
            double az = (double)azimuth_grid(a);
            for (size_t e = 0; e < n_elevation; ++e)
            {
                double ev = (double)elevation_grid(e);
                // V-pol: amplitude ~ cos(elevation), phase ramp in azimuth
                double amp_v = std::cos(ev) * 0.8 + 0.2;
                double phase_v = az * 1.5 + ev * 0.5 + el_offset;
                e_theta_re(e, a, el) = (dtype)(amp_v * std::cos(phase_v));
                e_theta_im(e, a, el) = (dtype)(amp_v * std::sin(phase_v));

                // H-pol: amplitude ~ sin(az)*sin(el), different phase
                double amp_h = std::abs(std::sin(az) * std::sin(ev)) * 0.6 + 0.1;
                double phase_h = -az * 0.8 + ev * 2.0 + el_offset + 0.7;
                e_phi_re(e, a, el) = (dtype)(amp_h * std::cos(phase_h));
                e_phi_im(e, a, el) = (dtype)(amp_h * std::sin(phase_h));
            }
        }
    }
}

// ============================================================================
//  Accuracy statistics for one output matrix
// ============================================================================
struct AccStats
{
    const char *name;
    double max_abs;         // max |new - ref|
    double avg_abs;         // mean |new - ref|
    int32_t max_ulp;        // max ULP distance (float only)
    double avg_ulp;         // mean ULP distance
    double max_rel;         // max |new - ref| / max(|ref|, tiny)
    double avg_rel;         // mean relative error
    size_t n;
};

template <typename dtype>
static AccStats compute_accuracy(const char *name,
                                 const arma::Mat<dtype> &test,
                                 const arma::Mat<dtype> &ref)
{
    AccStats s;
    s.name = name;
    s.n = ref.n_elem;
    s.max_abs = 0.0;
    s.avg_abs = 0.0;
    s.max_ulp = 0;
    s.avg_ulp = 0.0;
    s.max_rel = 0.0;
    s.avg_rel = 0.0;

    if (s.n == 0)
        return s;

    const dtype *pt = test.memptr();
    const dtype *pr = ref.memptr();
    int64_t sum_ulp = 0;
    double sum_abs = 0.0, sum_rel = 0.0;
    const double tiny = 1e-30;

    for (size_t i = 0; i < s.n; ++i)
    {
        double diff = std::abs((double)pt[i] - (double)pr[i]);
        double rval = std::abs((double)pr[i]);
        double rel = diff / std::max(rval, tiny);

        if (diff > s.max_abs)
            s.max_abs = diff;
        sum_abs += diff;

        if (rval > 1e-15) // only count relative error for non-negligible values
        {
            if (rel > s.max_rel)
                s.max_rel = rel;
            sum_rel += rel;
        }

        int32_t u = ulp_dist((float)pt[i], (float)pr[i]);
        if (u > s.max_ulp)
            s.max_ulp = u;
        sum_ulp += u;
    }

    s.avg_abs = sum_abs / (double)s.n;
    s.avg_ulp = (double)sum_ulp / (double)s.n;
    s.avg_rel = sum_rel / (double)s.n;
    return s;
}

static void print_acc_header()
{
    printf("    %-14s | %12s %12s | %8s %10s | %12s %12s\n",
           "Output", "avg |err|", "max |err|",
           "avg ULP", "max ULP",
           "avg rel", "max rel");
    printf("    %.*s\n", 98,
           "--------------------------------------------------------------------------------------------------------------");
}

static void print_acc(const AccStats &s)
{
    printf("    %-14s | %12.3e %12.3e | %8.1f %10d | %12.3e %12.3e\n",
           s.name, s.avg_abs, s.max_abs,
           s.avg_ulp, s.max_ulp,
           s.avg_rel, s.max_rel);
}

// ============================================================================
//  Single benchmark scenario
// ============================================================================
template <typename dtype>
struct Scenario
{
    const char *label;
    size_t n_azimuth, n_elevation, n_elements;
    size_t n_out, n_ang;
    bool per_element_angles;
    bool per_element_rotation;
    bool per_angle_rotation;
    double bank, tilt, heading; // orientation in radians (base)
};

template <typename dtype>
static void run_scenario(const Scenario<dtype> &sc, int n_warmup, int n_reps)
{
    printf("\n========================================================================\n");
    printf("  %s\n", sc.label);
    printf("  Pattern: %zu x %zu x %zu   n_out: %zu   n_ang: %zu\n",
           sc.n_azimuth, sc.n_elevation, sc.n_elements, sc.n_out, sc.n_ang);
    printf("  per_element_angles: %s   per_element_rotation: %s   per_angle_rotation: %s\n",
           sc.per_element_angles ? "YES" : "no",
           sc.per_element_rotation ? "YES" : "no",
           sc.per_angle_rotation ? "YES" : "no");
    printf("  dtype: %s   n_warmup: %d   n_reps: %d\n",
           sizeof(dtype) == 4 ? "float" : "double", n_warmup, n_reps);
    printf("========================================================================\n");

    // --- Generate pattern ---
    arma::Cube<dtype> e_theta_re, e_theta_im, e_phi_re, e_phi_im;
    arma::Col<dtype> azimuth_grid, elevation_grid;
    generate_pattern<dtype>(sc.n_azimuth, sc.n_elevation, sc.n_elements,
                            e_theta_re, e_theta_im, e_phi_re, e_phi_im,
                            azimuth_grid, elevation_grid);

    // --- Element indices (1-based, cycling through available elements) ---
    arma::Col<unsigned> i_element(sc.n_out);
    for (size_t i = 0; i < sc.n_out; ++i)
        i_element(i) = (unsigned)(i % sc.n_elements) + 1;

    // --- Element positions (spread along x-axis, half-wavelength spacing) ---
    arma::Mat<dtype> element_pos(3, sc.n_out, arma::fill::zeros);
    for (size_t i = 0; i < sc.n_out; ++i)
        element_pos(0, i) = (dtype)(0.5 * (double)i); // x = 0, 0.5, 1.0, ...

    // --- Orientation ---
    size_t n_orient_cols = sc.per_element_rotation ? sc.n_out : 1;
    size_t n_orient_slices = sc.per_angle_rotation ? sc.n_ang : 1;
    arma::Cube<dtype> orientation(3, n_orient_cols, n_orient_slices);
    for (size_t s = 0; s < n_orient_slices; ++s)
        for (size_t c = 0; c < n_orient_cols; ++c)
        {
            double jitter_c = sc.per_element_rotation ? 0.05 * (double)c : 0.0;
            double jitter_s = sc.per_angle_rotation ? 0.001 * std::sin(0.01 * (double)s) : 0.0;
            orientation(0, c, s) = (dtype)(sc.bank + jitter_c + jitter_s);
            orientation(1, c, s) = (dtype)(sc.tilt + jitter_c * 0.5);
            orientation(2, c, s) = (dtype)(sc.heading + jitter_s * 0.3);
        }

    // --- Azimuth / elevation angles for interpolation ---
    size_t az_rows = sc.per_element_angles ? sc.n_out : 1;
    arma::Mat<dtype> azimuth(az_rows, sc.n_ang);
    arma::Mat<dtype> elevation(az_rows, sc.n_ang);
    for (size_t a = 0; a < sc.n_ang; ++a)
    {
        double base_az = -M_PI + 2.0 * M_PI * (double)a / (double)sc.n_ang;
        double base_el = -M_PI / 2.0 + M_PI * ((double)(a % 997) / 997.0); // non-uniform elevation
        for (size_t r = 0; r < az_rows; ++r)
        {
            double per_el_jitter = sc.per_element_angles ? 0.02 * std::sin(0.1 * (double)(a + r * 37)) : 0.0;
            azimuth(r, a) = (dtype)(base_az + per_el_jitter);
            elevation(r, a) = (dtype)(base_el + per_el_jitter * 0.5);
        }
    }

    // --- Allocate outputs (original) ---
    arma::Mat<dtype> V_re_ref(sc.n_out, sc.n_ang), V_im_ref(sc.n_out, sc.n_ang);
    arma::Mat<dtype> H_re_ref(sc.n_out, sc.n_ang), H_im_ref(sc.n_out, sc.n_ang);
    arma::Mat<dtype> dist_ref(sc.n_out, sc.n_ang);
    arma::Mat<dtype> az_loc_ref(sc.n_out, sc.n_ang), el_loc_ref(sc.n_out, sc.n_ang);
    arma::Mat<dtype> gamma_ref(sc.n_out, sc.n_ang);

    // --- Allocate outputs (AVX2) ---
    arma::Mat<dtype> V_re_avx(sc.n_out, sc.n_ang), V_im_avx(sc.n_out, sc.n_ang);
    arma::Mat<dtype> H_re_avx(sc.n_out, sc.n_ang), H_im_avx(sc.n_out, sc.n_ang);
    arma::Mat<dtype> dist_avx(sc.n_out, sc.n_ang);
    arma::Mat<dtype> az_loc_avx(sc.n_out, sc.n_ang), el_loc_avx(sc.n_out, sc.n_ang);
    arma::Mat<dtype> gamma_avx(sc.n_out, sc.n_ang);

    // ================================================================
    //  PERFORMANCE
    // ================================================================
    double dt_ref = 0.0, dt_avx = 0.0;

    // Warmup original
    for (int i = 0; i < n_warmup; ++i)
        qd_arrayant_interpolate<dtype>(
            &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
            &i_element, &orientation, &element_pos,
            &V_re_ref, &V_im_ref, &H_re_ref, &H_im_ref,
            &dist_ref, &az_loc_ref, &el_loc_ref, &gamma_ref);

    // Benchmark original
    double t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        qd_arrayant_interpolate<dtype>(
            &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
            &i_element, &orientation, &element_pos,
            &V_re_ref, &V_im_ref, &H_re_ref, &H_im_ref,
            &dist_ref, &az_loc_ref, &el_loc_ref, &gamma_ref);
    dt_ref = (now_ms() - t0) / n_reps;

    // Warmup AVX2
    for (int i = 0; i < n_warmup; ++i)
        qd_arrayant_interpolate_avx2<dtype>(
            &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
            &i_element, &orientation, &element_pos,
            &V_re_avx, &V_im_avx, &H_re_avx, &H_im_avx,
            &dist_avx, &az_loc_avx, &el_loc_avx, &gamma_avx);

    // Benchmark AVX2
    t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        qd_arrayant_interpolate_avx2<dtype>(
            &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
            &i_element, &orientation, &element_pos,
            &V_re_avx, &V_im_avx, &H_re_avx, &H_im_avx,
            &dist_avx, &az_loc_avx, &el_loc_avx, &gamma_avx);
    dt_avx = (now_ms() - t0) / n_reps;

    double samples = (double)(sc.n_out * sc.n_ang);
    printf("\n  PERFORMANCE:\n");
    printf("    Original : %9.3f ms  (%7.1f Msamples/s)\n", dt_ref, samples / dt_ref / 1e3);
    printf("    AVX2     : %9.3f ms  (%7.1f Msamples/s)\n", dt_avx, samples / dt_avx / 1e3);
    printf("    Speedup  : %9.2fx\n", dt_ref / dt_avx);

    // ================================================================
    //  ACCURACY (AVX2 vs original as reference)
    // ================================================================
    printf("\n  ACCURACY (AVX2 vs original):\n");
    print_acc_header();
    print_acc(compute_accuracy("V_re", V_re_avx, V_re_ref));
    print_acc(compute_accuracy("V_im", V_im_avx, V_im_ref));
    print_acc(compute_accuracy("H_re", H_re_avx, H_re_ref));
    print_acc(compute_accuracy("H_im", H_im_avx, H_im_ref));
    print_acc(compute_accuracy("dist", dist_avx, dist_ref));
    print_acc(compute_accuracy("azimuth_loc", az_loc_avx, az_loc_ref));
    print_acc(compute_accuracy("elevation_loc", el_loc_avx, el_loc_ref));
    print_acc(compute_accuracy("gamma", gamma_avx, gamma_ref));

    // ================================================================
    //  SPOT CHECK: print a few sample values for manual inspection
    // ================================================================
    printf("\n  SPOT CHECK (first element, 5 equally-spaced angles):\n");
    printf("    %5s | %12s %12s | %12s %12s | %12s %12s | %12s %12s\n",
           "ang", "V_re ref", "V_re avx", "V_im ref", "V_im avx",
           "H_re ref", "H_re avx", "dist ref", "dist avx");
    printf("    %.*s\n", 115,
           "--------------------------------------------------------------------------------------------------------------"
           "--------------------------------------------------------------------------------------------------------------");
    for (int i = 0; i < 5; ++i)
    {
        size_t a = (size_t)((double)i / 4.0 * (double)(sc.n_ang - 1));
        printf("    %5zu | %12.6f %12.6f | %12.6f %12.6f | %12.6f %12.6f | %12.6f %12.6f\n",
               a,
               (double)V_re_ref(0, a), (double)V_re_avx(0, a),
               (double)V_im_ref(0, a), (double)V_im_avx(0, a),
               (double)H_re_ref(0, a), (double)H_re_avx(0, a),
               (double)dist_ref(0, a), (double)dist_avx(0, a));
    }

    // ================================================================
    //  AMPLITUDE + PHASE accuracy (like the SLERP benchmark)
    //  Computes amplitude ULP and phase-difference for the complex
    //  field outputs V = V_re + j*V_im and H = H_re + j*H_im.
    // ================================================================
    printf("\n  COMPLEX FIELD ACCURACY:\n");
    printf("    %-8s | %10s %10s | %12s %12s\n",
           "Field", "avg ampULP", "max ampULP", "avg |dPhase|", "max |dPhase|");
    printf("    %.*s\n", 66,
           "--------------------------------------------------------------------------------------------------------------");

    for (int pol = 0; pol < 2; ++pol)
    {
        const char *name = pol == 0 ? "V-pol" : "H-pol";
        const dtype *re_ref = pol == 0 ? V_re_ref.memptr() : H_re_ref.memptr();
        const dtype *im_ref = pol == 0 ? V_im_ref.memptr() : H_im_ref.memptr();
        const dtype *re_avx = pol == 0 ? V_re_avx.memptr() : H_re_avx.memptr();
        const dtype *im_avx = pol == 0 ? V_im_avx.memptr() : H_im_avx.memptr();

        int32_t max_amp_ulp = 0;
        int64_t sum_amp_ulp = 0;
        double max_phase = 0.0, sum_phase = 0.0;
        size_t n_phase = 0;

        for (size_t i = 0; i < samples; ++i)
        {
            float amp_ref = std::sqrt((float)((double)re_ref[i] * (double)re_ref[i] +
                                              (double)im_ref[i] * (double)im_ref[i]));
            float amp_avx = std::sqrt((float)((double)re_avx[i] * (double)re_avx[i] +
                                              (double)im_avx[i] * (double)im_avx[i]));

            int32_t u = ulp_dist(amp_avx, amp_ref);
            if (u > max_amp_ulp)
                max_amp_ulp = u;
            sum_amp_ulp += u;

            if ((double)amp_ref > 1e-10)
            {
                double ph_ref = std::atan2((double)im_ref[i], (double)re_ref[i]);
                double ph_avx = std::atan2((double)im_avx[i], (double)re_avx[i]);
                double pe = std::abs(phase_diff(ph_avx, ph_ref));
                if (pe > max_phase)
                    max_phase = pe;
                sum_phase += pe;
                n_phase++;
            }
        }

        double avg_phase = n_phase > 0 ? sum_phase / (double)n_phase : 0.0;
        printf("    %-8s | %10.1f %10d | %12.3e %12.3e\n",
               name,
               (double)sum_amp_ulp / samples, max_amp_ulp,
               avg_phase, max_phase);
    }
}

// ============================================================================
//  MAIN
// ============================================================================
int main()
{
    printf("###############################################################################\n");
    printf("#  Benchmark: qd_arrayant_interpolate  vs  qd_arrayant_interpolate_avx2       #\n");
    printf("###############################################################################\n");

    const int n_warmup = 3;
    const int n_reps = 20;

    // ---- Scenario 1: Small pattern, large angle set (SLERP-dominated) ----
    {
        Scenario<float> sc;
        sc.label = "S1: Small pattern (10-deg), large n_ang, 4 elements [float]";
        sc.n_azimuth = 37;
        sc.n_elevation = 19;
        sc.n_elements = 4;
        sc.n_out = 4;
        sc.n_ang = 100000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.1;
        sc.tilt = 0.2;
        sc.heading = 0.3;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 2: Large pattern, small angle set (gather-dominated) ----
    {
        Scenario<float> sc;
        sc.label = "S2: Large pattern (1-deg), small n_ang, 4 elements [float]";
        sc.n_azimuth = 361;
        sc.n_elevation = 181;
        sc.n_elements = 4;
        sc.n_out = 4;
        sc.n_ang = 1024;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.0;
        sc.tilt = 0.15;
        sc.heading = 0.0;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 3: Many elements (MIMO array), shared angles ----
    {
        Scenario<float> sc;
        sc.label = "S3: 5-deg pattern, 64 elements, shared angles [float]";
        sc.n_azimuth = 73;
        sc.n_elevation = 37;
        sc.n_elements = 64;
        sc.n_out = 64;
        sc.n_ang = 10000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.05;
        sc.tilt = 0.1;
        sc.heading = -0.2;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 4: Per-element angles + per-element rotation ----
    {
        Scenario<float> sc;
        sc.label = "S4: 5-deg pattern, per-element angles + rotation [float]";
        sc.n_azimuth = 73;
        sc.n_elevation = 37;
        sc.n_elements = 4;
        sc.n_out = 4;
        sc.n_ang = 50000;
        sc.per_element_angles = true;
        sc.per_element_rotation = true;
        sc.per_angle_rotation = false;
        sc.bank = 0.1;
        sc.tilt = -0.1;
        sc.heading = 0.5;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 5: Per-angle rotation (time-varying orientation) ----
    {
        Scenario<float> sc;
        sc.label = "S5: 5-deg pattern, per-angle rotation [float]";
        sc.n_azimuth = 73;
        sc.n_elevation = 37;
        sc.n_elements = 4;
        sc.n_out = 4;
        sc.n_ang = 50000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = true;
        sc.bank = 0.0;
        sc.tilt = 0.0;
        sc.heading = 0.0;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 6: Single element, huge angle set (no shared precompute) ----
    {
        Scenario<float> sc;
        sc.label = "S6: 10-deg pattern, 1 element, 1M angles [float]";
        sc.n_azimuth = 37;
        sc.n_elevation = 19;
        sc.n_elements = 1;
        sc.n_out = 1;
        sc.n_ang = 1000000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.3;
        sc.tilt = -0.2;
        sc.heading = 0.7;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 7: Double precision, large angles (check dtype=double path) ----
    {
        Scenario<double> sc;
        sc.label = "S7: 5-deg pattern, 4 elements, 100k angles [double]";
        sc.n_azimuth = 73;
        sc.n_elevation = 37;
        sc.n_elements = 4;
        sc.n_out = 4;
        sc.n_ang = 100000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.1;
        sc.tilt = 0.2;
        sc.heading = 0.3;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 8: Stress test — 256 elements (MMIMO) ----
    {
        Scenario<float> sc;
        sc.label = "S8: 5-deg pattern, 256 elements, shared angles [float]";
        sc.n_azimuth = 73;
        sc.n_elevation = 37;
        sc.n_elements = 8;
        sc.n_out = 256;
        sc.n_ang = 5000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.0;
        sc.tilt = 0.0;
        sc.heading = 0.0;
        run_scenario(sc, n_warmup, n_reps);
    }

    // ---- Scenario 9: Worst case for accuracy — near-pole angles ----
    {
        printf("\n========================================================================\n");
        printf("  S9: Pole stress test — elevation near ±pi/2 [float]\n");
        printf("========================================================================\n");

        Scenario<float> sc;
        sc.label = "S9: Pole stress, elevation near +-pi/2 [float]";
        sc.n_azimuth = 73;
        sc.n_elevation = 37;
        sc.n_elements = 2;
        sc.n_out = 2;
        sc.n_ang = 10000;
        sc.per_element_angles = false;
        sc.per_element_rotation = false;
        sc.per_angle_rotation = false;
        sc.bank = 0.3;
        sc.tilt = 0.5;
        sc.heading = 0.1;

        // Generate pattern
        arma::Cube<float> etr, eti, epr, epi;
        arma::Col<float> azg, elg;
        generate_pattern<float>(sc.n_azimuth, sc.n_elevation, sc.n_elements,
                                etr, eti, epr, epi, azg, elg);

        arma::Col<unsigned> iel(sc.n_out);
        for (size_t i = 0; i < sc.n_out; ++i)
            iel(i) = (unsigned)(i % sc.n_elements) + 1;

        arma::Mat<float> epos(3, sc.n_out, arma::fill::zeros);
        for (size_t i = 0; i < sc.n_out; ++i)
            epos(0, i) = 0.5f * (float)i;

        arma::Cube<float> orient(3, 1, 1);
        orient(0, 0, 0) = (float)sc.bank;
        orient(1, 0, 0) = (float)sc.tilt;
        orient(2, 0, 0) = (float)sc.heading;

        // Angles clustered near poles
        arma::Mat<float> az(1, sc.n_ang), el(1, sc.n_ang);
        for (size_t a = 0; a < sc.n_ang; ++a)
        {
            az(0, a) = (float)(-M_PI + 2.0 * M_PI * (double)a / (double)sc.n_ang);
            // Alternate between near north and south pole
            double base_el = (a % 2 == 0) ? M_PI / 2.0 - 0.01 : -M_PI / 2.0 + 0.01;
            double jitter = 0.005 * std::sin(0.1 * (double)a);
            el(0, a) = (float)(base_el + jitter);
        }

        arma::Mat<float> Vr_r(sc.n_out, sc.n_ang), Vi_r(sc.n_out, sc.n_ang);
        arma::Mat<float> Hr_r(sc.n_out, sc.n_ang), Hi_r(sc.n_out, sc.n_ang);
        arma::Mat<float> d_r(sc.n_out, sc.n_ang), azl_r(sc.n_out, sc.n_ang);
        arma::Mat<float> ell_r(sc.n_out, sc.n_ang), g_r(sc.n_out, sc.n_ang);

        arma::Mat<float> Vr_a(sc.n_out, sc.n_ang), Vi_a(sc.n_out, sc.n_ang);
        arma::Mat<float> Hr_a(sc.n_out, sc.n_ang), Hi_a(sc.n_out, sc.n_ang);
        arma::Mat<float> d_a(sc.n_out, sc.n_ang), azl_a(sc.n_out, sc.n_ang);
        arma::Mat<float> ell_a(sc.n_out, sc.n_ang), g_a(sc.n_out, sc.n_ang);

        qd_arrayant_interpolate<float>(
            &etr, &eti, &epr, &epi, &azg, &elg, &az, &el,
            &iel, &orient, &epos,
            &Vr_r, &Vi_r, &Hr_r, &Hi_r, &d_r, &azl_r, &ell_r, &g_r);

        qd_arrayant_interpolate_avx2<float>(
            &etr, &eti, &epr, &epi, &azg, &elg, &az, &el,
            &iel, &orient, &epos,
            &Vr_a, &Vi_a, &Hr_a, &Hi_a, &d_a, &azl_a, &ell_a, &g_a);

        printf("\n  ACCURACY (near-pole, AVX2 vs original):\n");
        print_acc_header();
        print_acc(compute_accuracy("V_re", Vr_a, Vr_r));
        print_acc(compute_accuracy("V_im", Vi_a, Vi_r));
        print_acc(compute_accuracy("H_re", Hr_a, Hr_r));
        print_acc(compute_accuracy("H_im", Hi_a, Hi_r));
        print_acc(compute_accuracy("dist", d_a, d_r));
        print_acc(compute_accuracy("azimuth_loc", azl_a, azl_r));
        print_acc(compute_accuracy("elevation_loc", ell_a, ell_r));
        print_acc(compute_accuracy("gamma", g_a, g_r));
    }

    printf("\n\nDone.\n");
    return 0;
}
