// Benchmark: qd_arrayant_interpolate vs qd_arrayant_interpolate_avx2
// Realistic calling pattern: n_ang=1, n_out=1/8/64, per-element rotation.
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_interp benchmark_interp.cpp /sjc/quadriga-lib/lib/libquadriga.a

#include "qd_arrayant_interpolate.hpp"
#include "qd_arrayant_interpolate_avx2.hpp"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static int32_t ulp_dist(float a, float b)
{
    if (std::isnan(a) || std::isnan(b)) return INT32_MAX;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    int32_t d = ia - ib;
    return d < 0 ? -d : d;
}

static inline double phase_diff(double a, double b)
{
    double d = a - b;
    if (d >  M_PI) d -= 2.0 * M_PI;
    if (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

template <typename dtype>
static void generate_pattern(
    size_t n_azimuth, size_t n_elevation, size_t n_elements,
    arma::Cube<dtype> &e_theta_re, arma::Cube<dtype> &e_theta_im,
    arma::Cube<dtype> &e_phi_re,   arma::Cube<dtype> &e_phi_im,
    arma::Col<dtype>  &azimuth_grid, arma::Col<dtype> &elevation_grid)
{
    azimuth_grid.set_size(n_azimuth);
    elevation_grid.set_size(n_elevation);
    for (size_t i = 0; i < n_azimuth; ++i)
        azimuth_grid(i) = (dtype)(-M_PI + 2.0*M_PI*(double)i/(double)n_azimuth);
    for (size_t i = 0; i < n_elevation; ++i)
        elevation_grid(i) = (dtype)(-M_PI/2.0 + M_PI*(double)i/(double)(n_elevation-1));

    e_theta_re.set_size(n_elevation, n_azimuth, n_elements);
    e_theta_im.set_size(n_elevation, n_azimuth, n_elements);
    e_phi_re.set_size(n_elevation, n_azimuth, n_elements);
    e_phi_im.set_size(n_elevation, n_azimuth, n_elements);

    for (size_t el = 0; el < n_elements; ++el) {
        double el_offset = 0.3*(double)el;
        for (size_t a = 0; a < n_azimuth; ++a) {
            double az = (double)azimuth_grid(a);
            for (size_t e = 0; e < n_elevation; ++e) {
                double ev = (double)elevation_grid(e);
                double amp_v   = std::cos(ev)*0.8 + 0.2;
                double phase_v = az*1.5 + ev*0.5 + el_offset;
                e_theta_re(e,a,el) = (dtype)(amp_v*std::cos(phase_v));
                e_theta_im(e,a,el) = (dtype)(amp_v*std::sin(phase_v));
                double amp_h   = std::abs(std::sin(az)*std::sin(ev))*0.6 + 0.1;
                double phase_h = -az*0.8 + ev*2.0 + el_offset + 0.7;
                e_phi_re(e,a,el) = (dtype)(amp_h*std::cos(phase_h));
                e_phi_im(e,a,el) = (dtype)(amp_h*std::sin(phase_h));
            }
        }
    }
}

struct AccStats {
    const char *name;
    double max_abs, avg_abs;
    int32_t max_ulp;
    double avg_ulp, max_rel, avg_rel;
    size_t n;
};

template <typename dtype>
static AccStats compute_accuracy(const char *name,
                                 const arma::Mat<dtype> &test,
                                 const arma::Mat<dtype> &ref)
{
    AccStats s;
    s.name = name; s.n = ref.n_elem;
    s.max_abs = s.avg_abs = s.max_ulp = s.avg_ulp = s.max_rel = s.avg_rel = 0;
    if (s.n == 0) return s;
    const dtype *pt = test.memptr(), *pr = ref.memptr();
    int64_t sum_ulp = 0;
    double sum_abs = 0.0, sum_rel = 0.0;
    const double tiny = 1e-30;
    for (size_t i = 0; i < s.n; ++i) {
        double diff = std::abs((double)pt[i] - (double)pr[i]);
        double rval = std::abs((double)pr[i]);
        if (diff > s.max_abs) s.max_abs = diff;
        sum_abs += diff;
        if (rval > 1e-15) {
            double rel = diff / std::max(rval, tiny);
            if (rel > s.max_rel) s.max_rel = rel;
            sum_rel += rel;
        }
        int32_t u = ulp_dist((float)pt[i], (float)pr[i]);
        if (u > s.max_ulp) s.max_ulp = u;
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
           "Output", "avg |err|", "max |err|", "avg ULP", "max ULP", "avg rel", "max rel");
    printf("    %.*s\n", 98,
           "----------------------------------------------------------------------------------------------------"
           "----------------");
}

static void print_acc(const AccStats &s)
{
    printf("    %-14s | %12.3e %12.3e | %8.1f %10d | %12.3e %12.3e\n",
           s.name, s.avg_abs, s.max_abs, s.avg_ulp, s.max_ulp, s.avg_rel, s.max_rel);
}

// ============================================================================
//  Single benchmark scenario
//  All scenarios use n_ang=1 and per-element rotation (default calling pattern).
// ============================================================================
template <typename dtype>
struct Scenario {
    const char *label;
    size_t n_azimuth, n_elevation;  // pattern grid
    size_t n_elements;              // distinct patterns in the array
    size_t n_out;                   // output ports (the vectorization dimension)
    size_t n_ang;                   // angles (1 = typical real-world call)
    int    n_warmup, n_reps;
    double bank, tilt, heading;     // base orientation (radians)
};

template <typename dtype>
static void run_scenario(const Scenario<dtype> &sc)
{
    printf("\n========================================================================\n");
    printf("  %s\n", sc.label);
    printf("  Pattern: %zu x %zu x %zu   n_out: %zu   n_ang: %zu\n",
           sc.n_azimuth, sc.n_elevation, sc.n_elements, sc.n_out, sc.n_ang);
    printf("  per_element_rotation: YES   dtype: %s   n_warmup: %d   n_reps: %d\n",
           sizeof(dtype)==4 ? "float" : "double", sc.n_warmup, sc.n_reps);
    printf("========================================================================\n");

    // Pattern
    arma::Cube<dtype> e_theta_re, e_theta_im, e_phi_re, e_phi_im;
    arma::Col<dtype>  azimuth_grid, elevation_grid;
    generate_pattern<dtype>(sc.n_azimuth, sc.n_elevation, sc.n_elements,
                            e_theta_re, e_theta_im, e_phi_re, e_phi_im,
                            azimuth_grid, elevation_grid);

    // Element indices (1-based)
    arma::Col<unsigned> i_element(sc.n_out);
    for (size_t i = 0; i < sc.n_out; ++i)
        i_element(i) = (unsigned)(i % sc.n_elements) + 1;

    // Element positions: half-wavelength spacing along x
    arma::Mat<dtype> element_pos(3, sc.n_out, arma::fill::zeros);
    for (size_t i = 0; i < sc.n_out; ++i)
        element_pos(0, i) = (dtype)(0.5 * (double)i);

    // Per-element rotation (one orientation per output element, one slice)
    arma::Cube<dtype> orientation(3, sc.n_out, 1);
    for (size_t c = 0; c < sc.n_out; ++c) {
        double jitter = 0.05 * (double)c;
        orientation(0, c, 0) = (dtype)(sc.bank    + jitter);
        orientation(1, c, 0) = (dtype)(sc.tilt    + jitter * 0.5);
        orientation(2, c, 0) = (dtype)(sc.heading + jitter * 0.3);
    }

    // Angles: n_ang values, away from poles and grid boundaries
    arma::Mat<dtype> azimuth(1, sc.n_ang), elevation(1, sc.n_ang);
    for (size_t a = 0; a < sc.n_ang; ++a) {
        azimuth(0, a)   = (dtype)(0.4  + 0.001  * (double)a);
        elevation(0, a) = (dtype)(0.2  + 0.0005 * (double)a);
    }

    // Output buffers
    arma::Mat<dtype> V_re_ref(sc.n_out,sc.n_ang), V_im_ref(sc.n_out,sc.n_ang);
    arma::Mat<dtype> H_re_ref(sc.n_out,sc.n_ang), H_im_ref(sc.n_out,sc.n_ang);
    arma::Mat<dtype> dist_ref(sc.n_out,sc.n_ang);
    arma::Mat<dtype> az_loc_ref(sc.n_out,sc.n_ang), el_loc_ref(sc.n_out,sc.n_ang);
    arma::Mat<dtype> gamma_ref(sc.n_out,sc.n_ang);

    arma::Mat<dtype> V_re_avx(sc.n_out,sc.n_ang), V_im_avx(sc.n_out,sc.n_ang);
    arma::Mat<dtype> H_re_avx(sc.n_out,sc.n_ang), H_im_avx(sc.n_out,sc.n_ang);
    arma::Mat<dtype> dist_avx(sc.n_out,sc.n_ang);
    arma::Mat<dtype> az_loc_avx(sc.n_out,sc.n_ang), el_loc_avx(sc.n_out,sc.n_ang);
    arma::Mat<dtype> gamma_avx(sc.n_out,sc.n_ang);

    // ----------------------------------------------------------------
    //  PERFORMANCE
    // ----------------------------------------------------------------
    auto call_ref = [&]() {
        qd_arrayant_interpolate<dtype>(
            e_theta_re, e_theta_im, e_phi_re, e_phi_im,
            azimuth_grid, elevation_grid, azimuth, elevation,
            i_element, orientation, element_pos,
            V_re_ref, V_im_ref, H_re_ref, H_im_ref,
            &dist_ref, &az_loc_ref, &el_loc_ref, &gamma_ref);
    };
    auto call_avx = [&]() {
        qd_arrayant_interpolate_avx2<dtype>(
            e_theta_re, e_theta_im, e_phi_re, e_phi_im,
            azimuth_grid, elevation_grid, azimuth, elevation,
            i_element, orientation, element_pos,
            V_re_avx, V_im_avx, H_re_avx, H_im_avx,
            &dist_avx, &az_loc_avx, &el_loc_avx, &gamma_avx);
    };

    for (int i = 0; i < sc.n_warmup; ++i) call_ref();
    double t0 = now_ms();
    for (int i = 0; i < sc.n_reps; ++i) call_ref();
    double dt_ref = (now_ms() - t0) / sc.n_reps;

    for (int i = 0; i < sc.n_warmup; ++i) call_avx();
    t0 = now_ms();
    for (int i = 0; i < sc.n_reps; ++i) call_avx();
    double dt_avx = (now_ms() - t0) / sc.n_reps;

    double samples = (double)(sc.n_out * sc.n_ang);
    printf("\n  PERFORMANCE (per call):\n");
    printf("    Original : %11.4f ms   (%8.2f Msamples/s)\n", dt_ref, samples/dt_ref/1e3);
    printf("    AVX2     : %11.4f ms   (%8.2f Msamples/s)\n", dt_avx, samples/dt_avx/1e3);
    printf("    Speedup  : %11.2fx\n", dt_ref / dt_avx);

    // ----------------------------------------------------------------
    //  ACCURACY  (use last AVX2 and REF outputs from the timing loops)
    // ----------------------------------------------------------------
    printf("\n  ACCURACY (AVX2 vs scalar reference):\n");
    print_acc_header();
    print_acc(compute_accuracy("V_re",    V_re_avx,  V_re_ref));
    print_acc(compute_accuracy("V_im",    V_im_avx,  V_im_ref));
    print_acc(compute_accuracy("H_re",    H_re_avx,  H_re_ref));
    print_acc(compute_accuracy("H_im",    H_im_avx,  H_im_ref));
    print_acc(compute_accuracy("dist",    dist_avx,  dist_ref));
    print_acc(compute_accuracy("az_loc",  az_loc_avx,az_loc_ref));
    print_acc(compute_accuracy("el_loc",  el_loc_avx,el_loc_ref));
    print_acc(compute_accuracy("gamma",   gamma_avx, gamma_ref));

    // ----------------------------------------------------------------
    //  SPOT CHECK: show first min(5, n_out) elements for angle 0
    // ----------------------------------------------------------------
    // size_t show = std::min((size_t)5, sc.n_out);
    // printf("\n  SPOT CHECK (first %zu element(s), angle 0):\n", show);
    // printf("    %-6s  %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
    //        "elem", "V_re", "V_im", "H_re", "H_im", "dist", "az_loc", "el_loc", "gamma");
    // printf("    %-6s  %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
    //        "------", "------------", "------------", "------------",
    //        "------------", "------------", "------------", "------------", "------------");
    // for (size_t i = 0; i < show; ++i) {
    //     printf("    ref%-3zu  %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //            i,
    //            (double)V_re_ref(i,0), (double)V_im_ref(i,0),
    //            (double)H_re_ref(i,0), (double)H_im_ref(i,0),
    //            (double)dist_ref(i,0), (double)az_loc_ref(i,0),
    //            (double)el_loc_ref(i,0),(double)gamma_ref(i,0));
    //     printf("    avx%-3zu  %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //            i,
    //            (double)V_re_avx(i,0), (double)V_im_avx(i,0),
    //            (double)H_re_avx(i,0), (double)H_im_avx(i,0),
    //            (double)dist_avx(i,0), (double)az_loc_avx(i,0),
    //            (double)el_loc_avx(i,0),(double)gamma_avx(i,0));
    // }

    // ----------------------------------------------------------------
    //  COMPLEX AMPLITUDE ACCURACY
    //  Compare amplitude and phase of complex V and H fields.
    // ----------------------------------------------------------------
    printf("\n  COMPLEX FIELD ACCURACY:\n");
    printf("    %-8s  %12s %12s %12s %12s %12s %12s\n",
           "field", "avg amp err", "max amp err", "avg ph err", "max ph err",
           "avg ULP(re)", "max ULP(re)");
    printf("    %.*s\n", 88,
           "----------------------------------------------------------------------------------------------------");

    auto complex_stats = [&](const char *fname,
                              const arma::Mat<dtype> &re_avx, const arma::Mat<dtype> &im_avx,
                              const arma::Mat<dtype> &re_ref, const arma::Mat<dtype> &im_ref)
    {
        double sum_amp_err=0, max_amp_err=0, sum_ph_err=0, max_ph_err=0;
        double sum_ulp=0; int32_t max_ulp_re=0;
        size_t N = re_ref.n_elem;
        const dtype *ra=re_avx.memptr(), *ia=im_avx.memptr();
        const dtype *rr=re_ref.memptr(),  *ir=im_ref.memptr();
        for (size_t i=0; i<N; ++i) {
            double amp_a = std::sqrt((double)ra[i]*(double)ra[i]+(double)ia[i]*(double)ia[i]);
            double amp_r = std::sqrt((double)rr[i]*(double)rr[i]+(double)ir[i]*(double)ir[i]);
            double ae = std::abs(amp_a-amp_r);
            if (ae>max_amp_err) max_amp_err=ae;
            sum_amp_err+=ae;
            if (amp_r>1e-10) {
                double ph_a = std::atan2((double)ia[i],(double)ra[i]);
                double ph_r = std::atan2((double)ir[i],(double)rr[i]);
                double pe = std::abs(phase_diff(ph_a,ph_r));
                if (pe>max_ph_err) max_ph_err=pe;
                sum_ph_err+=pe;
            }
            int32_t u = ulp_dist((float)ra[i],(float)rr[i]);
            if (u>max_ulp_re) max_ulp_re=u;
            sum_ulp+=u;
        }
        printf("    %-8s  %12.3e %12.3e %12.3e %12.3e %12.1f %12d\n",
               fname,
               sum_amp_err/(double)N, max_amp_err,
               sum_ph_err/(double)N,  max_ph_err,
               sum_ulp/(double)N, max_ulp_re);
    };
    complex_stats("V (theta)", V_re_avx, V_im_avx, V_re_ref, V_im_ref);
    complex_stats("H (phi)  ",  H_re_avx, H_im_avx, H_re_ref, H_im_ref);
}

int main()
{
    printf("\n");
    printf("================================================================\n");
    printf("  Antenna Interpolation Benchmark\n");
    printf("  Typical calling pattern: n_ang=1, per-element rotation\n");
    printf("  Comparing scalar vs AVX2 implementation\n");
    printf("================================================================\n");

    // ------------------------------------------------------------------
    //  S1: Single element  (n_out=1)  --  scalar baseline, no SIMD benefit expected
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S1: float  n_out=1   n_ang=1  grid=5deg (73x37)",
        73, 37, 1,   // n_azimuth, n_elevation, n_elements
        1, 1,        // n_out, n_ang
        500000, 1000,// n_warmup, n_reps  (very fast call -> many reps)
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S2: 8 elements (n_out=8)  --  exactly one AVX2 register wide
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S2: float  n_out=8   n_ang=1  grid=5deg (73x37)",
        73, 37, 8,
        8, 1,
        100000, 1000,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S3: 64 elements (n_out=64)  --  typical MIMO base station
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S3: float  n_out=64  n_ang=1  grid=5deg (73x37)",
        73, 37, 64,
        64, 1,
        20000, 500,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S4: 8 elements, large 1-degree pattern  --  gather-stressed
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S4: float  n_out=8   n_ang=1  grid=1deg (361x181)",
        361, 181, 8,
        8, 1,
        50000, 1000,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S5: 64 elements, large 1-degree pattern
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S5: float  n_out=64  n_ang=1  grid=1deg (361x181)",
        361, 181, 64,
        64, 1,
        10000, 500,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S6: Double precision path  (n_out=8)
    // ------------------------------------------------------------------
    run_scenario(Scenario<double>{
        "S6: double n_out=8   n_ang=1  grid=5deg (73x37)",
        73, 37, 8,
        8, 1,
        100000, 1000,
        0.1, 0.05, 0.2});

    // ========================================================================
    //  LARGE n_ang SCENARIOS (production real-world: ~500-2000 angles)
    // ========================================================================

    // ------------------------------------------------------------------
    //  S7: n_out=1, n_ang=500  --  single element, many angles
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S7: float  n_out=1   n_ang=500  grid=5deg (73x37)",
        73, 37, 1,
        1, 500,
        1000, 500,  // fewer reps, single element is fast
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S8: n_out=1, n_ang=2000  --  single element, extreme angle count
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S8: float  n_out=1   n_ang=2000 grid=5deg (73x37)",
        73, 37, 1,
        1, 2000,
        1000, 500,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S9: n_out=64, n_ang=500  --  full array, many angles
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S9: float  n_out=64  n_ang=500  grid=5deg (73x37)",
        73, 37, 64,
        64, 500,
        2000, 200,  // reduce reps for larger workload
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S10: n_out=64, n_ang=2000  --  full array, extreme angles
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S10: float  n_out=64  n_ang=2000 grid=5deg (73x37)",
        73, 37, 64,
        64, 2000,
        1000, 100,  // further reduce reps due to large workload
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S11: n_out=1, n_ang=500, 1° grid  --  single element, fine grid
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S11: float  n_out=1   n_ang=500  grid=1deg (361x181)",
        361, 181, 1,
        1, 500,
        500, 200,  // fine grid = slower grid search, fewer reps
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S12: n_out=64, n_ang=500, 1° grid  --  full array, fine grid, many angles
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S12: float  n_out=64  n_ang=500  grid=1deg (361x181)",
        361, 181, 64,
        64, 500,
        500, 100,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S13: n_out=64, n_ang=2000, 1° grid  --  extreme load
    // ------------------------------------------------------------------
    run_scenario(Scenario<float>{
        "S13: float  n_out=64  n_ang=2000 grid=1deg (361x181)",
        361, 181, 64,
        64, 2000,
        500, 50,   // severe reduction: 64*2000 = 128k samples per call
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S14: n_out=64, n_ang=500, double precision
    // ------------------------------------------------------------------
    run_scenario(Scenario<double>{
        "S14: double n_out=64  n_ang=500  grid=5deg (73x37)",
        73, 37, 64,
        64, 500,
        1000, 100,
        0.1, 0.05, 0.2});

    // ------------------------------------------------------------------
    //  S15: n_out=64, n_ang=2000, double precision
    // ------------------------------------------------------------------
    run_scenario(Scenario<double>{
        "S15: double n_out=64  n_ang=2000 grid=5deg (73x37)",
        73, 37, 64,
        64, 2000,
        500, 50,
        0.1, 0.05, 0.2});

    printf("\n================================================================\n");
    printf("  Benchmark complete.\n");
    printf("================================================================\n\n");
    return 0;
}