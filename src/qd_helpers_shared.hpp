// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers used by the sources of several sections: errors in parallel regions, output aliasing checks, kernel
// selection, small vector math, portable random numbers, table interpolation, SLERP of complex values, physical
// constants and the ISO 9613-1 air absorption

#ifndef qd_helpers_shared_hpp
#define qd_helpers_shared_hpp

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

// OpenMP-safe exception handling
// Declare `bool` + `std::string` before the parallel region, pass them in,
// and rethrow a runtime_error after the region if the flag was set.
#define OMP_SAFE_CALL(error_flag, error_msg, ...)                 \
    try                                                           \
    {                                                             \
        __VA_ARGS__;                                              \
    }                                                             \
    catch (const std::exception &e)                               \
    {                                                             \
        _Pragma("omp critical") if (!(error_flag))                \
        {                                                         \
            (error_msg) = e.what();                               \
            (error_flag) = true;                                  \
        }                                                         \
    }                                                             \
    catch (...)                                                   \
    {                                                             \
        _Pragma("omp critical") if (!(error_flag))                \
        {                                                         \
            (error_msg) = "Unknown exception in parallel region"; \
            (error_flag) = true;                                  \
        }                                                         \
    }

// Runs f(i) for the items i = 0 ... n - 1 of a batch, in parallel if 'parallel' is set. An error of an item is
// rethrown after the loop with the prefix "<prefix> <id>: " of the lowest failing item (id = i, or id[i] if given),
// as std::invalid_argument if it was one, else as std::runtime_error. std::exception_ptr is not used: it needs a libstdc++
// symbol version that the libstdc++ bundled with older MATLAB releases lacks, so the MEX files would not load there.
template <typename F>
inline void qd_batch_loop(size_t n, bool parallel, const char *prefix, F &&f, const arma::uvec *id = nullptr)
{
    long long n_i = (long long)n, i_err = n_i;
    bool invalid_arg = false;
    std::string msg;
#pragma omp parallel for if (parallel)
    for (long long i = 0; i < n_i; ++i)
        try
        {
            f((size_t)i);
        }
        catch (const std::exception &e)
        {
#pragma omp critical
            if (i < i_err)
                i_err = i, invalid_arg = dynamic_cast<const std::invalid_argument *>(&e) != nullptr, msg = e.what();
        }
    if (i_err == n_i)
        return;
    msg = std::string(prefix) + " " + std::to_string(id ? (*id)[(arma::uword)i_err] : (arma::uword)i_err) + ": " + msg;
    if (invalid_arg)
        throw std::invalid_argument(msg);
    throw std::runtime_error(msg);
}

// --- Outputs that are the same object as an input, or share memory with one ---
// Outputs are resized and written while inputs are still read, so an output must not alias an input or another
// output. These helpers detect this before anything is written.

template <typename T, typename = void>
struct qd_has_memptr : std::false_type
{
};

template <typename T>
struct qd_has_memptr<T, std::void_t<decltype(std::declval<const T &>().memptr()), decltype(std::declval<const T &>().n_elem)>>
    : std::true_type
{
};

template <typename T, typename = void>
struct qd_has_data : std::false_type
{
};

template <typename T>
struct qd_has_data<T, std::void_t<decltype(std::declval<const T &>().data()), decltype(std::declval<const T &>().size())>>
    : std::true_type
{
};

// Byte range [first, first + n_bytes) of the elements of an Armadillo object (Mat, Col, Row, Cube) or a
// std::vector; n_bytes = 0 for empty objects and for other types
template <typename T>
static inline void qd_mem_range(const T &x, std::uintptr_t &first, std::size_t &n_bytes)
{
    first = 0, n_bytes = 0;
    if constexpr (qd_has_memptr<T>::value)
        first = (std::uintptr_t)x.memptr(), n_bytes = (std::size_t)x.n_elem * sizeof(*x.memptr());
    else if constexpr (qd_has_data<T>::value)
        first = (std::uintptr_t)x.data(), n_bytes = (std::size_t)x.size() * sizeof(*x.data());
}

// True if 'a' and 'b' are the same object (compared only when one type converts to the other, e.g. Col and Mat)
// or if their elements share memory; false if one of them is nullptr
template <typename A, typename B>
static inline bool qd_alias(const A *a, const B *b)
{
    if (a == nullptr || b == nullptr)
        return false;
    if constexpr (std::is_convertible<const A *, const B *>::value)
    {
        if (static_cast<const B *>(a) == b)
            return true;
    }
    else if constexpr (std::is_convertible<const B *, const A *>::value)
    {
        if (static_cast<const A *>(b) == a)
            return true;
    }
    std::uintptr_t a0, b0;
    std::size_t na, nb;
    qd_mem_range(*a, a0, na);
    qd_mem_range(*b, b0, nb);
    return na != 0 && nb != 0 && a0 < b0 + nb && b0 < a0 + na;
}

// Throws std::invalid_argument if output 'out' aliases one of the inputs; nullptr arguments are skipped
template <typename O, typename... I>
static inline void qd_check_alias(const O *out, const char *name, const I *...inputs)
{
    (void)out; // unused when there are no inputs
    if ((qd_alias(out, inputs) || ...))
        throw std::invalid_argument(std::string("Output '") + name + "' cannot be an input or share memory with one.");
}

// Throws std::invalid_argument if two outputs alias each other; nullptr arguments are skipped
template <typename A, typename B>
static inline void qd_check_alias_outputs(const A *a, const char *name_a, const B *b, const char *name_b)
{
    if (qd_alias(a, b))
        throw std::invalid_argument(std::string("Outputs '") + name_a + "' and '" + name_b +
                                    "' cannot be the same object or share memory.");
}

// Throws std::invalid_argument if one of the outputs out[0 .. N-1] aliases one of the inputs or another output;
// nullptr arguments are skipped. Example: qd_check_alias_list<arma::mat>({x, y}, {"x", "y"}, in_a, in_b);
template <typename T, std::size_t N, typename... I>
static inline void qd_check_alias_list(const T *const (&out)[N], const char *const (&name)[N], const I *...inputs)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        qd_check_alias(out[i], name[i], inputs...);
        for (std::size_t j = i + 1; j < N; ++j)
            qd_check_alias_outputs(out[i], name[i], out[j], name[j]);
    }
}

// Single position or orientation: checks that it has 3 finite elements ([3, 1] or [1, 3]) and returns its data pointer
template <typename dtype>
static inline const dtype *qd_xyz(const arma::Mat<dtype> &xyz, const char *name)
{
    if (xyz.n_elem != 3ULL)
        throw std::invalid_argument(std::string("Input '") + name + "' must have 3 elements.");
    const dtype *p = xyz.memptr();
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
        throw std::invalid_argument(std::string("Input '") + name + "' must be finite.");
    return p;
}

// Resolves the kernel selector 'use_kernel' of a function with a GENERIC and an AVX2 kernel whose auto (0) selects AVX2
// when it is available: returns 1 (GENERIC) or 2 (AVX2). Throws std::invalid_argument for values other than 0, 1, 2
// and for 2 when the AVX2 kernel is not available (BUILD_WITH_AVX2 and the CPU check)
static inline int qd_kernel_auto_avx2(int use_kernel)
{
    if (use_kernel < 0 || use_kernel > 2)
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1 or 2.");
#if BUILD_WITH_AVX2
    const bool avx2_available = runtime_AVX2_Check();
#else
    const bool avx2_available = false;
#endif
    if (use_kernel == 2 && !avx2_available)
        throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
    return (use_kernel == 1 || !avx2_available) ? 1 : 2;
}

// Calculate length
template <typename dtype>
static inline dtype qd_calc_length(dtype Ox, dtype Oy, dtype Oz, dtype Dx, dtype Dy, dtype Dz)
{
    dtype a = Dx - Ox;
    dtype b = a * a;
    a = Dy - Oy, b += a * a;
    a = Dz - Oz, b += a * a;
    return std::sqrt(b);
}

// Euler rotations to rotation matrix
template <typename dtype>
static inline void qd_rotation_matrix(const dtype *euler_angles_3xN, // Orientation vectors (Euler rotations)
                                      dtype *rotation_matrix_9xN,    // Rotation matrix in column-major ordering
                                      size_t N = 1,                  // Number of elements
                                      bool transposeR = false)       // Returns the transpose of R
{
    for (size_t i = 0; i < 3 * N; i += 3)
    {
        const dtype *O = &euler_angles_3xN[i];
        dtype *R = &rotation_matrix_9xN[3 * i];

        double cc = (double)O[0], sc = std::sin(cc);
        double cb = (double)O[1], sb = std::sin(cb);
        double ca = (double)O[2], sa = std::sin(ca);
        ca = std::cos(ca), cb = std::cos(cb), cc = std::cos(cc);

        if (transposeR)
        {
            R[0] = dtype(ca * cb);
            R[1] = dtype(ca * sb * sc - sa * cc);
            R[2] = dtype(ca * sb * cc + sa * sc);
            R[3] = dtype(sa * cb);
            R[4] = dtype(sa * sb * sc + ca * cc);
            R[5] = dtype(sa * sb * cc - ca * sc);
            R[6] = dtype(-sb);
            R[7] = dtype(cb * sc);
            R[8] = dtype(cb * cc);
        }
        else
        {
            R[0] = dtype(ca * cb);
            R[1] = dtype(sa * cb);
            R[2] = dtype(-sb);
            R[3] = dtype(ca * sb * sc - sa * cc);
            R[4] = dtype(sa * sb * sc + ca * cc);
            R[5] = dtype(cb * sc);
            R[6] = dtype(ca * sb * cc + sa * sc);
            R[7] = dtype(sa * sb * cc - ca * sc);
            R[8] = dtype(cb * cc);
        }
    }
}

// Apply Euler rotations inplace
template <typename dtype>
static inline void qd_rotate_inplace(dtype bank, dtype tilt, dtype heading, dtype *data3xN, size_t N, bool transpose = false)
{

    double O[3] = {bank, tilt, heading};
    double R[9];
    qd_rotation_matrix(O, R);

    for (size_t i = 0; i < N; ++i)
    {
        size_t ix = transpose ? i : 3 * i;
        size_t iy = transpose ? N + i : 3 * i + 1;
        size_t iz = transpose ? 2 * N + i : 3 * i + 2;

        double xx = (double)data3xN[ix];
        double yy = (double)data3xN[iy];
        double zz = (double)data3xN[iz];

        double a = R[0] * xx + R[3] * yy + R[6] * zz;
        double b = R[1] * xx + R[4] * yy + R[7] * zz;
        double c = R[2] * xx + R[5] * yy + R[8] * zz;

        data3xN[ix] = (dtype)a;
        data3xN[iy] = (dtype)b;
        data3xN[iz] = (dtype)c;
    }
}

// Multiply with scalar
template <typename dtype>
static inline void qd_multiply_scalar(dtype scalar, dtype *data, size_t N)
{
    for (size_t i = 0; i < N; ++i)
        data[i] *= scalar;
}

// ---- Portable random numbers ----
// The draws are computed from the raw output of std::mt19937_64, whose sequence the C++ standard fixes. No
// distribution class of the standard library is used, as their algorithms differ between platforms, so a seed gives
// the same values on all platforms (up to the rounding of log and cos).

// Random number generator: a seed >= 0 gives the same stream on all platforms, -1 seeds from the system random device
static inline std::mt19937_64 qd_rng(arma::sword seed)
{
    std::mt19937_64 rng;
    if (seed >= 0)
        rng.seed((unsigned long long)seed);
    else
    {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        rng.seed(seq);
    }
    return rng;
}

// Uniform random number in [0, 1) from the upper 53 bits of the generator output
static inline double qd_rng_uniform(std::mt19937_64 &rng)
{
    return (double)(rng() >> 11) * (1.0 / 9007199254740992.0);
}

// Standard normal random number from two uniform draws (Box-Muller, cosine branch)
static inline double qd_rng_normal(std::mt19937_64 &rng)
{
    const double u1 = 1.0 - qd_rng_uniform(rng); // (0, 1], avoids log(0)
    const double u2 = qd_rng_uniform(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// Linear interpolation of a table y(x) at x0 (e.g. an ACF table y = acf at the distances x = dist), the last value
// is used beyond the last sample point and the first value below the first sample point; 'x' must increase strictly
static inline double qd_interp_linear(const double *y, const double *x, size_t n, double x0)
{
    if (x0 >= x[n - 1])
        return y[n - 1];
    if (x0 <= x[0])
        return y[0];
    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1)
    {
        size_t mid = (lo + hi) / 2;
        if (x[mid] <= x0)
            lo = mid;
        else
            hi = mid;
    }
    double w = (x0 - x[lo]) / (x[hi] - x[lo]);
    return y[lo] + w * (y[hi] - y[lo]);
}

// Spherical interpolation with linear fallback for a single complex value pair
template <typename dtype>
inline void qd_slerp_complex(dtype Ar, dtype Ai, dtype Br, dtype Bi, dtype w, dtype &Xr, dtype &Xi)
{
    constexpr dtype one = dtype(1.0), zero = dtype(0.0), neg_one = dtype(-1.0);
    const dtype R0 = std::numeric_limits<dtype>::epsilon() * std::numeric_limits<dtype>::epsilon() * std::numeric_limits<dtype>::epsilon();
    const dtype R1 = std::numeric_limits<dtype>::epsilon();
    constexpr dtype tL = dtype(-0.999), tS = dtype(-0.99), dT = one / (tS - tL);

    dtype wB = w, wA = one - w;
    dtype ampA = std::sqrt(Ar * Ar + Ai * Ai);
    dtype ampB = std::sqrt(Br * Br + Bi * Bi);

    const bool tinyA = ampA < R1;
    const bool tinyB = ampB < R1;

    if (tinyA && tinyB)
    {
        Xr = zero;
        Xi = zero;
        return;
    }

    dtype gAr = tinyA ? zero : Ar / ampA;
    dtype gAi = tinyA ? zero : Ai / ampA;
    dtype gBr = tinyB ? zero : Br / ampB;
    dtype gBi = tinyB ? zero : Bi / ampB;
    dtype cPhase = (tinyA || tinyB) ? neg_one : gAr * gBr + gAi * gBi;
    bool linear_int = cPhase < tS;

    dtype fXr = zero, fXi = zero;
    if (linear_int)
        fXr = wA * Ar + wB * Br, fXi = wA * Ai + wB * Bi;

    if (cPhase > tL)
    {
        dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0;
        dtype sPhase = one / std::sin(Phase);
        dtype wp = std::sin(wB * Phase) * sPhase;
        dtype wn = std::sin(wA * Phase) * sPhase;
        dtype gXr = wn * gAr + wp * gBr;
        dtype gXi = wn * gAi + wp * gBi;
        dtype ampX = wA * ampA + wB * ampB;

        if (linear_int) // Transition zone: blend spherical and linear
        {
            dtype m = (tS - cPhase) * dT, n = one - m;
            fXr = n * gXr * ampX + m * fXr;
            fXi = n * gXi * ampX + m * fXi;
        }
        else
            fXr = gXr * ampX, fXi = gXi * ampX;
    }
    Xr = fXr;
    Xi = fXi;
}

// ---- Physical constants and air absorption ----

// Speed of sound in air [m/s], the one value used for acoustics throughout the library
static constexpr double qd_speed_of_sound = 343.0;

// Speed of light in vacuum [m/s]
static constexpr double qd_speed_of_light = 299792458.0;

// Scalar (acoustic) mode maps an acoustic frequency onto an EM-equivalent frequency such that
// c_light / f_eq equals the acoustic wavelength. The public functions take the acoustic frequency
// and convert once at their entry; the material model works in EM-equivalent units.
//   f_eq = f_acoustic * qd_acoustic_freq_scale
static constexpr double qd_acoustic_freq_scale = qd_speed_of_light / qd_speed_of_sound;

// Material-model frequency in GHz of a public frequency argument in Hz (acoustic Hz in scalar mode)
static inline double qd_material_fGHz(double f_Hz, bool scalar_mode)
{
    return (scalar_mode ? f_Hz * qd_acoustic_freq_scale : f_Hz) * 1.0e-9;
}

// Atmospheric absorption coefficient alpha(f) in dB/m according to ISO 9613-1.
// Defaults: T = 20 degrees Celsius, RH = 50 %, p = 1 atm (indoor sea-level).
// Input frequency is given in Hertz.
static inline double qd_iso9613_alpha(double f_Hz,
                                      double T_celsius = 20.0,
                                      double RH_percent = 50.0,
                                      double p_kPa = 101.325)
{
    constexpr double T0 = 293.15;   // Reference temperature [K]
    constexpr double T01 = 273.16;  // Triple-point isotherm [K]
    constexpr double p_r = 101.325; // Reference pressure [kPa]

    const double T = T_celsius + 273.15; // Absolute temperature [K]

    // Saturation vapor pressure ratio and molar concentration of water vapor [%]
    const double C = -6.8346 * std::pow(T01 / T, 1.261) + 4.6151;
    const double psat = std::pow(10.0, C);
    const double h = RH_percent * psat / (p_kPa / p_r);

    // Relaxation frequencies for O2 and N2 [Hz]
    const double pa_pr = p_kPa / p_r;
    const double T_T0 = T / T0;
    const double f_rO = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    const double f_rN = pa_pr * std::pow(T_T0, -0.5) * (9.0 + 280.0 * h * std::exp(-4.170 * (std::pow(T_T0, -1.0 / 3.0) - 1.0)));

    // Absorption coefficient [dB/m]
    const double f2 = f_Hz * f_Hz;
    const double classical = 1.84e-11 / pa_pr * std::sqrt(T_T0);
    const double rot_O = 0.01275 * std::exp(-2239.1 / T) / (f_rO + f2 / f_rO);
    const double rot_N = 0.1068 * std::exp(-3352.0 / T) / (f_rN + f2 / f_rN);

    return 8.686 * f2 * (classical + std::pow(T_T0, -2.5) * (rot_O + rot_N));
}

// Gain at 1 m distance in dB, i.e. the path-loss reference excluding the 20*log10(d) spreading term.
// EM mode: FSPL reference, -32.45 - 20*log10(f_GHz). Scalar (acoustic) mode: pure 1/r spreading, 0 dB.
// Input frequency is given in Hertz.
static inline double qd_gain_at_1m(double f_Hz, bool scalar_mode)
{
    return scalar_mode ? 0.0 : (-32.45 - 20.0 * std::log10(f_Hz * 1.0e-9));
}

#endif
