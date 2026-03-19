// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o bench_gcs benchmark_get_channels_spherical.cpp /sjc/quadriga-lib/lib/libquadriga.a

#include "quadriga_lib.hpp"
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

// Wrap-aware signed phase difference in [-pi, pi]
static inline double phase_diff(double a, double b)
{
    double d = a - b;
    if (d > M_PI) d -= 2.0 * M_PI;
    if (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

// ============================================================================
// Helper: generate a deterministic 4-element array antenna
// ============================================================================
template <typename dtype>
static quadriga_lib::arrayant<dtype> make_4x_array(dtype spacing_m)
{
    auto ant = quadriga_lib::generate_arrayant_omni<dtype>();
    ant.copy_element(0, 3); // Creates 4 elements
    // ULA along y-axis
    ant.element_pos(1, 0) = -dtype(1.5) * spacing_m;
    ant.element_pos(1, 1) = -dtype(0.5) * spacing_m;
    ant.element_pos(1, 2) = dtype(0.5) * spacing_m;
    ant.element_pos(1, 3) = dtype(1.5) * spacing_m;
    return ant;
}

// ============================================================================
// Helper: generate deterministic scenario data (scatterers, gains, M matrix)
//
// TX at origin, RX at (dist, 0, 0).
// Scatterers are distributed in a half-dome around the midpoint.
// Path 0 is the LOS path. Remaining n_path-1 are NLOS.
// ============================================================================
template <typename dtype>
struct ScenarioData
{
    arma::Mat<dtype> fbs_pos, lbs_pos;
    arma::Col<dtype> path_gain, path_length;
    arma::Mat<dtype> M;
};

template <typename dtype>
static ScenarioData<dtype> make_scenario(size_t n_path, dtype dist)
{
    ScenarioData<dtype> s;
    s.fbs_pos.zeros(3, n_path);
    s.lbs_pos.zeros(3, n_path);
    s.path_gain.zeros(n_path);
    s.path_length.zeros(n_path);
    s.M.zeros(8, n_path);

    // Path 0: LOS
    s.fbs_pos(0, 0) = dist / dtype(2.0);
    s.lbs_pos(0, 0) = dist / dtype(2.0);
    s.path_gain(0) = dtype(1.0);
    s.path_length(0) = dist;
    s.M(0, 0) = dtype(1.0);   // VV = 1
    s.M(6, 0) = dtype(-1.0);  // HH = -1

    // Remaining paths: NLOS scatterers distributed around the link
    for (size_t p = 1; p < n_path; ++p)
    {
        // Deterministic pseudo-random distribution
        double phi = 2.0 * M_PI * (double)p / (double)(n_path - 1);           // Azimuth [0, 2pi]
        double el = 0.5 * std::sin(3.7 * (double)p / (double)(n_path - 1));   // Elevation [-0.5, 0.5] rad
        double r_fbs = 20.0 + 30.0 * std::sin(2.1 * (double)p);              // Distance from TX [20..50] m
        double r_lbs = 20.0 + 30.0 * std::cos(1.7 * (double)p);              // Distance from RX [20..50] m

        // FBS position (near TX)
        s.fbs_pos(0, p) = (dtype)(r_fbs * std::cos(el) * std::cos(phi));
        s.fbs_pos(1, p) = (dtype)(r_fbs * std::cos(el) * std::sin(phi));
        s.fbs_pos(2, p) = (dtype)(r_fbs * std::sin(el));

        // LBS position (near RX)
        double phi2 = phi + 0.5; // Slight angular offset
        s.lbs_pos(0, p) = (dtype)((double)dist + r_lbs * std::cos(el) * std::cos(phi2));
        s.lbs_pos(1, p) = (dtype)(r_lbs * std::cos(el) * std::sin(phi2));
        s.lbs_pos(2, p) = (dtype)(r_lbs * std::sin(el));

        // Path gain: decays with path index (simulating weaker paths)
        s.path_gain(p) = (dtype)(1.0 / (1.0 + 0.1 * (double)p));

        // Path length: 0 = auto-calculate from geometry
        s.path_length(p) = dtype(0.0);

        // Polarization transfer matrix: deterministic rotation
        double rot = 0.3 * (double)p;
        s.M(0, p) = (dtype)std::cos(rot);    // Re(VV)
        s.M(1, p) = (dtype)std::sin(rot);    // Im(VV)
        s.M(2, p) = (dtype)(0.1 * std::sin(2.0 * rot)); // Re(VH)
        s.M(4, p) = (dtype)(0.1 * std::cos(2.0 * rot)); // Re(HV)
        s.M(6, p) = (dtype)(-std::cos(rot)); // Re(HH)
        s.M(7, p) = (dtype)(-std::sin(rot)); // Im(HH)
    }

    return s;
}

// ============================================================================
// PART 1: Performance benchmark
// ============================================================================
static void run_performance()
{
    const int n_reps = 200;
    const int n_warmup = 20;
    const size_t n_path = 500;
    const double dist = 200.0;
    const double fc = 3.5e9; // 3.5 GHz (5G band)

    printf("=== PERFORMANCE: get_channels_spherical ===\n");
    printf("    4x4 MIMO, %zu paths, fc = %.1f GHz, %d repetitions\n\n", n_path, fc / 1e9, n_reps);

    // --- Create antennas ---
    float lambda_f = (float)(299792458.0 / fc);
    double lambda_d = 299792458.0 / fc;
    auto tx_f = make_4x_array<float>(lambda_f * 0.5f);
    auto rx_f = make_4x_array<float>(lambda_f * 0.5f);
    auto tx_d = make_4x_array<double>(lambda_d * 0.5);
    auto rx_d = make_4x_array<double>(lambda_d * 0.5);

    // --- Create scenario data ---
    auto sf = make_scenario<float>(n_path, (float)dist);
    auto sd = make_scenario<double>(n_path, dist);

    // --- Output containers ---
    arma::fcube cr_f, ci_f, dl_f, aod_f, eod_f, aoa_f, eoa_f;
    arma::cube cr_d, ci_d, dl_d, aod_d, eod_d, aoa_d, eoa_d;

    // Struct to hold results
    struct Result { const char *label; double ms; double mpairs; };
    Result results[4];

    // --- 1. float, AVX2 = false (scalar) ---
    for (int i = 0; i < n_warmup; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            &aod_f, &eod_f, &aoa_f, &eoa_f, false);

    double t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            &aod_f, &eod_f, &aoa_f, &eoa_f, false);
    results[0] = {"float / scalar", (now_ms() - t0) / n_reps, 0.0};

    // --- 2. float, AVX2 = true ---
    for (int i = 0; i < n_warmup; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            &aod_f, &eod_f, &aoa_f, &eoa_f, true);

    t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            &aod_f, &eod_f, &aoa_f, &eoa_f, true);
    results[1] = {"float / AVX2", (now_ms() - t0) / n_reps, 0.0};

    // --- 3. double, AVX2 = false (scalar) ---
    for (int i = 0; i < n_warmup; ++i)
        quadriga_lib::get_channels_spherical<double>(
            &tx_d, &rx_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            dist, 0.0, 0.0, 0.0, 0.0, 0.0,
            &sd.fbs_pos, &sd.lbs_pos, &sd.path_gain, &sd.path_length, &sd.M,
            &cr_d, &ci_d, &dl_d, fc, true, false,
            &aod_d, &eod_d, &aoa_d, &eoa_d, false);

    t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        quadriga_lib::get_channels_spherical<double>(
            &tx_d, &rx_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            dist, 0.0, 0.0, 0.0, 0.0, 0.0,
            &sd.fbs_pos, &sd.lbs_pos, &sd.path_gain, &sd.path_length, &sd.M,
            &cr_d, &ci_d, &dl_d, fc, true, false,
            &aod_d, &eod_d, &aoa_d, &eoa_d, false);
    results[2] = {"double / scalar", (now_ms() - t0) / n_reps, 0.0};

    // --- 4. double, AVX2 = true ---
    for (int i = 0; i < n_warmup; ++i)
        quadriga_lib::get_channels_spherical<double>(
            &tx_d, &rx_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            dist, 0.0, 0.0, 0.0, 0.0, 0.0,
            &sd.fbs_pos, &sd.lbs_pos, &sd.path_gain, &sd.path_length, &sd.M,
            &cr_d, &ci_d, &dl_d, fc, true, false,
            &aod_d, &eod_d, &aoa_d, &eoa_d, true);

    t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        quadriga_lib::get_channels_spherical<double>(
            &tx_d, &rx_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            dist, 0.0, 0.0, 0.0, 0.0, 0.0,
            &sd.fbs_pos, &sd.lbs_pos, &sd.path_gain, &sd.path_length, &sd.M,
            &cr_d, &ci_d, &dl_d, fc, true, false,
            &aod_d, &eod_d, &aoa_d, &eoa_d, true);
    results[3] = {"double / AVX2", (now_ms() - t0) / n_reps, 0.0};

    // Total MIMO coefficients per call: n_rx * n_tx * n_path = 4 * 4 * 500 = 8000
    double coeff_per_call = 4.0 * 4.0 * (double)n_path;
    for (int i = 0; i < 4; ++i)
        results[i].mpairs = coeff_per_call / results[i].ms / 1e3; // Mcoeff/s

    // --- Report ---
    printf("%-18s | %10s | %14s | %12s\n", "Config", "Time [ms]", "Mcoeff/s", "Speedup");
    printf("-------------------+------------+----------------+--------------\n");
    for (int i = 0; i < 4; ++i)
    {
        double speedup = results[0].ms / results[i].ms; // relative to float/scalar
        printf("%-18s | %10.3f | %14.2f | %10.2fx\n",
               results[i].label, results[i].ms, results[i].mpairs, speedup);
    }
    printf("\n");

    // Also report with no angle outputs (common use case)
    printf("--- Same scenario, no angle outputs ---\n\n");

    for (int i = 0; i < n_warmup; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr, false);

    t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr, false);
    double dt_noang_scalar = (now_ms() - t0) / n_reps;

    for (int i = 0; i < n_warmup; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr, true);

    t0 = now_ms();
    for (int i = 0; i < n_reps; ++i)
        quadriga_lib::get_channels_spherical<float>(
            &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
            &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
            &cr_f, &ci_f, &dl_f, (float)fc, true, false,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr,
            (arma::fcube *)nullptr, (arma::fcube *)nullptr, true);
    double dt_noang_avx2 = (now_ms() - t0) / n_reps;

    printf("%-18s | %10.3f | %14.2f | %10.2fx\n",
           "float / scalar", dt_noang_scalar, coeff_per_call / dt_noang_scalar / 1e3,
           dt_noang_scalar / dt_noang_scalar);
    printf("%-18s | %10.3f | %14.2f | %10.2fx\n",
           "float / AVX2", dt_noang_avx2, coeff_per_call / dt_noang_avx2 / 1e3,
           dt_noang_scalar / dt_noang_avx2);
}

// ============================================================================
// PART 2: Accuracy benchmark
//
// Reference: double precision, AVX2 = false (full double-precision scalar path)
//
// For each test configuration, we compute:
//   - Coefficient amplitude error in ULP (float)
//   - Coefficient phase error in radians
//   - Delay error (absolute difference)
//   - Angle errors (AOD, EOD, AOA, EOA) in radians
//
// All errors are measured against the double/scalar reference, with the
// reference values cast to float before ULP comparison.
// ============================================================================
static void run_accuracy()
{
    const size_t n_path = 500;
    const double dist = 200.0;
    const double fc = 3.5e9;
    const size_t n_tx = 4, n_rx = 4;
    const size_t n_coeff = n_rx * n_tx * n_path; // 8000

    printf("\n=== ACCURACY: get_channels_spherical ===\n");
    printf("    4x4 MIMO, %zu paths, fc = %.1f GHz\n", n_path, fc / 1e9);
    printf("    Reference: double / scalar (use_avx2 = false)\n\n");

    // --- Create antennas ---
    float lambda_f = (float)(299792458.0 / fc);
    double lambda_d = 299792458.0 / fc;
    auto tx_f = make_4x_array<float>(lambda_f * 0.5f);
    auto rx_f = make_4x_array<float>(lambda_f * 0.5f);
    auto tx_d = make_4x_array<double>(lambda_d * 0.5);
    auto rx_d = make_4x_array<double>(lambda_d * 0.5);

    // --- Create scenario data ---
    auto sf = make_scenario<float>(n_path, (float)dist);
    auto sd = make_scenario<double>(n_path, dist);

    // --- Compute reference: double / scalar ---
    arma::cube cr_ref, ci_ref, dl_ref, aod_ref, eod_ref, aoa_ref, eoa_ref;
    quadriga_lib::get_channels_spherical<double>(
        &tx_d, &rx_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &sd.fbs_pos, &sd.lbs_pos, &sd.path_gain, &sd.path_length, &sd.M,
        &cr_ref, &ci_ref, &dl_ref, fc, true, false,
        &aod_ref, &eod_ref, &aoa_ref, &eoa_ref, false);

    // --- Test configurations ---
    struct Config
    {
        const char *label;
        bool is_float;
        bool avx2;
    };
    const Config configs[] = {
        {"float / scalar",  true,  false},
        {"float / AVX2",    true,  true},
        {"double / scalar", false, false},
        {"double / AVX2",   false, true},
    };
    const int n_configs = sizeof(configs) / sizeof(configs[0]);

    // --- Header ---
    printf("%-18s | %-28s | %-30s | %-18s | %-18s | %-18s | %-18s | %-18s\n",
           "Config",
           "Coeff Amp ULP",
           "Coeff Phase [rad]",
           "Delay err [ps]",
           "AOD err [mrad]",
           "EOD err [mrad]",
           "AOA err [mrad]",
           "EOA err [mrad]");
    printf("%.*s\n", 190, "----------------------------------------------------------------------------------------------------------------------------"
                          "----------------------------------------------------------------------------------------------------------------------------");

    for (int c = 0; c < n_configs; ++c)
    {
        // Compute test output
        arma::fcube cr_f, ci_f, dl_f, aod_f, eod_f, aoa_f, eoa_f;
        arma::cube cr_dd, ci_dd, dl_dd, aod_dd, eod_dd, aoa_dd, eoa_dd;

        if (configs[c].is_float)
        {
            quadriga_lib::get_channels_spherical<float>(
                &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
                &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
                &cr_f, &ci_f, &dl_f, (float)fc, true, false,
                &aod_f, &eod_f, &aoa_f, &eoa_f, configs[c].avx2);
        }
        else
        {
            quadriga_lib::get_channels_spherical<double>(
                &tx_d, &rx_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                dist, 0.0, 0.0, 0.0, 0.0, 0.0,
                &sd.fbs_pos, &sd.lbs_pos, &sd.path_gain, &sd.path_length, &sd.M,
                &cr_dd, &ci_dd, &dl_dd, fc, true, false,
                &aod_dd, &eod_dd, &aoa_dd, &eoa_dd, configs[c].avx2);
        }

        // --- Measure errors ---
        int32_t max_amp_ulp = 0;
        int64_t sum_amp_ulp = 0;
        double max_phase_err = 0.0, sum_phase_err = 0.0;
        double max_delay_err = 0.0, sum_delay_err = 0.0;
        double max_aod_err = 0.0, sum_aod_err = 0.0;
        double max_eod_err = 0.0, sum_eod_err = 0.0;
        double max_aoa_err = 0.0, sum_aoa_err = 0.0;
        double max_eoa_err = 0.0, sum_eoa_err = 0.0;
        size_t n_phase = 0;

        const double amp_threshold = 1e-10;

        for (size_t j = 0; j < n_coeff; ++j)
        {
            // Reference values (double)
            double ref_re = cr_ref.at(j);
            double ref_im = ci_ref.at(j);
            double ref_amp = std::sqrt(ref_re * ref_re + ref_im * ref_im);
            double ref_dl = dl_ref.at(j);
            double ref_aod = aod_ref.at(j);
            double ref_eod = eod_ref.at(j);
            double ref_aoa = aoa_ref.at(j);
            double ref_eoa = eoa_ref.at(j);

            // Test values
            double test_re, test_im, test_dl, test_aod, test_eod, test_aoa, test_eoa;
            if (configs[c].is_float)
            {
                test_re = (double)cr_f.at(j);   test_im = (double)ci_f.at(j);
                test_dl = (double)dl_f.at(j);
                test_aod = (double)aod_f.at(j);  test_eod = (double)eod_f.at(j);
                test_aoa = (double)aoa_f.at(j);  test_eoa = (double)eoa_f.at(j);
            }
            else
            {
                test_re = cr_dd.at(j);   test_im = ci_dd.at(j);
                test_dl = dl_dd.at(j);
                test_aod = aod_dd.at(j);  test_eod = eod_dd.at(j);
                test_aoa = aoa_dd.at(j);  test_eoa = eoa_dd.at(j);
            }

            float test_amp = (float)std::sqrt(test_re * test_re + test_im * test_im);

            // Amplitude ULP
            int32_t da = ulp_dist(test_amp, (float)ref_amp);
            if (da > max_amp_ulp) max_amp_ulp = da;
            sum_amp_ulp += da;

            // Phase error
            if (ref_amp > amp_threshold)
            {
                double ref_phase = std::atan2(ref_im, ref_re);
                double test_phase = std::atan2(test_im, test_re);
                double pe = std::abs(phase_diff(test_phase, ref_phase));
                if (pe > max_phase_err) max_phase_err = pe;
                sum_phase_err += pe;
                n_phase++;
            }

            // Delay error
            double de = std::abs(test_dl - ref_dl);
            if (de > max_delay_err) max_delay_err = de;
            sum_delay_err += de;

            // Angle errors (wrap-aware for azimuth)
            double ae;
            ae = std::abs(phase_diff(test_aod, ref_aod));
            if (ae > max_aod_err) max_aod_err = ae;
            sum_aod_err += ae;

            ae = std::abs(test_eod - ref_eod);
            if (ae > max_eod_err) max_eod_err = ae;
            sum_eod_err += ae;

            ae = std::abs(phase_diff(test_aoa, ref_aoa));
            if (ae > max_aoa_err) max_aoa_err = ae;
            sum_aoa_err += ae;

            ae = std::abs(test_eoa - ref_eoa);
            if (ae > max_eoa_err) max_eoa_err = ae;
            sum_eoa_err += ae;
        }

        double avg_phase = (n_phase > 0) ? sum_phase_err / (double)n_phase : 0.0;

        printf("%-18s | avg %5.1f  max %7d   | avg %.1e  max %.1e | avg %6.2f max %6.2f | avg %6.2f max %6.2f | avg %6.2f max %6.2f | avg %6.2f max %6.2f | avg %6.2f max %6.2f\n",
               configs[c].label,
               (double)sum_amp_ulp / (double)n_coeff, max_amp_ulp,
               avg_phase, max_phase_err,
               sum_delay_err / (double)n_coeff * 1e12, max_delay_err * 1e12,    // Convert to picoseconds
               sum_aod_err / (double)n_coeff * 1e3, max_aod_err * 1e3,          // Convert to milliradians
               sum_eod_err / (double)n_coeff * 1e3, max_eod_err * 1e3,
               sum_aoa_err / (double)n_coeff * 1e3, max_aoa_err * 1e3,
               sum_eoa_err / (double)n_coeff * 1e3, max_eoa_err * 1e3);
    }
}

// ============================================================================
// PART 3: Scaling benchmark (path count sweep)
// ============================================================================
static void run_scaling()
{
    const int n_reps = 100;
    const int n_warmup = 10;
    const double dist = 200.0;
    const double fc = 3.5e9;

    printf("\n=== SCALING: float/AVX2 vs float/scalar, varying n_path ===\n\n");
    printf("%-10s | %-22s | %-22s | %s\n", "n_path", "float / scalar", "float / AVX2", "Speedup");
    printf("-----------+------------------------+------------------------+----------\n");

    float lambda_f = (float)(299792458.0 / fc);
    auto tx_f = make_4x_array<float>(lambda_f * 0.5f);
    auto rx_f = make_4x_array<float>(lambda_f * 0.5f);

    const size_t path_counts[] = {1, 10, 50, 100, 200, 500, 1000, 2000};
    const int n_counts = sizeof(path_counts) / sizeof(path_counts[0]);

    for (int pi = 0; pi < n_counts; ++pi)
    {
        size_t np = path_counts[pi];
        auto sf = make_scenario<float>(np, (float)dist);

        arma::fcube cr, ci, dl;

        // Scalar
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::get_channels_spherical<float>(
                &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
                &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
                &cr, &ci, &dl, (float)fc, true, false,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr, false);

        double t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::get_channels_spherical<float>(
                &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
                &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
                &cr, &ci, &dl, (float)fc, true, false,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr, false);
        double dt_scalar = (now_ms() - t0) / n_reps;

        // AVX2
        for (int i = 0; i < n_warmup; ++i)
            quadriga_lib::get_channels_spherical<float>(
                &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
                &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
                &cr, &ci, &dl, (float)fc, true, false,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr, true);

        t0 = now_ms();
        for (int i = 0; i < n_reps; ++i)
            quadriga_lib::get_channels_spherical<float>(
                &tx_f, &rx_f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                (float)dist, 0.f, 0.f, 0.f, 0.f, 0.f,
                &sf.fbs_pos, &sf.lbs_pos, &sf.path_gain, &sf.path_length, &sf.M,
                &cr, &ci, &dl, (float)fc, true, false,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr,
                (arma::fcube *)nullptr, (arma::fcube *)nullptr, true);
        double dt_avx2 = (now_ms() - t0) / n_reps;

        double n_coeff = 4.0 * 4.0 * (double)np;
        printf("%-10zu | %7.3f ms (%7.1f Mc/s) | %7.3f ms (%7.1f Mc/s) | %6.2fx\n",
               np,
               dt_scalar, n_coeff / dt_scalar / 1e3,
               dt_avx2, n_coeff / dt_avx2 / 1e3,
               dt_scalar / dt_avx2);
    }
}

int main()
{
    run_performance();
    run_accuracy();
    run_scaling();
    return 0;
}
