// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// A collection of small reusable helper functions to reduce copy and pasting code

#ifndef quadriga_lib_helper_H
#define quadriga_lib_helper_H

#include <cstring>
#include <complex>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

// Validate a named material-property map and return the material count (n_mtl).
// - Every present (non-empty) column must have the same length; throws otherwise.
// - Empty columns are treated as absent (consumers apply per-column defaults).
// - An empty map returns 0.
// Call once per public entry point; internal mtl_col / mtl_val accesses are then safe
// for any material index < the returned n_mtl.
template <typename dtype>
static inline arma::uword mtl_validate(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop)
{
    arma::uword n_mtl = 0;
    bool seen = false;
    for (const auto &kv : mtl_prop)
    {
        if (kv.second.empty())
            continue;
        arma::uword len = (arma::uword)kv.second.size();
        if (!seen)
        {
            n_mtl = len;
            seen = true;
        }
        else if (len != n_mtl)
            throw std::invalid_argument("Material property column '" + kv.first + "' has length " +
                                        std::to_string(len) + ", expected " + std::to_string(n_mtl) +
                                        " (all columns must have the same number of materials).");
    }
    return n_mtl;
}

// Resolve a named material-property column to a raw pointer (map-based material model).
// Returns nullptr if the column is absent; the consumer then applies its own default.
template <typename dtype>
static inline const dtype *mtl_col(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, const std::string &key)
{
    auto it = mtl_prop->find(key);
    return (it == mtl_prop->end() || it->second.empty()) ? nullptr : it->second.data();
}

// Read a material value for material index iM from a resolved column pointer.
// Falls back to 'def' when the column is absent.
template <typename dtype>
static inline double mtl_val(const dtype *col, arma::uword iM, double def)
{
    return (col == nullptr) ? def : (double)col[iM];
}

// Assemble complex-valued eta from mtl coefficients
static inline std::complex<double> eta_from_coeffs(double a, double b, double c, double d, double fRef, double fGHz)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double f_rel = fGHz / fRef;
    double eta_r = a * std::pow(f_rel, b);
    double sigma = c * std::pow(f_rel, d);
    double eta_i = -17.98 * sigma / fGHz;
    return std::complex<double>(eta_r, eta_i);
}

// Permittivity resonance (acoustic): complex Lorentz pole added to the interface (Fresnel)
// permittivity only. Inactive unless resF > 0, resQ > 0 and resS != 0, so the EM path is
// unchanged. The +i denominator makes resS > 0 add loss (negative imaginary part), consistent
// with the conductivity term. Deliberately NOT applied to the in-medium loss: a strong pole can
// push Re(eta) < 0, and medium_loss_dB uses a real sqrt(Re eta).
static inline std::complex<double> eta_resonance(double resF, double resQ, double resS, double fGHz)
{
    if (resF <= 0.0 || resQ <= 0.0 || resS == 0.0)
        return std::complex<double>(0.0, 0.0);
    double resF2 = resF * resF;
    std::complex<double> denom(resF2 - fGHz * fGHz, (resF / resQ) * fGHz);
    return (resS * resF2) / denom;
}

// Calculate in-medium loss
static inline double medium_loss_dB(std::complex<double> eta, double alpha, double alphaB,
                                    double fRef, double fGHz, double dist, double mass = 0.0)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double er = std::real(eta);
    double tan_delta = std::imag(eta) / er;
    double cos_delta = 1.0 / std::sqrt(1.0 + tan_delta * tan_delta);
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (fGHz * std::sqrt(er));
    double loss = dist * 8.686 / Delta;
    loss += dist * alpha * std::pow(fGHz / fRef, alphaB);

    // Mass is a bulk-propagation term: never apply it over the ~1 mm co-location
    // epsilon (ray_offset). Real traversals are at least the panel thickness (cm),
    // so a small path floor removes the spurious slope without touching them.
    constexpr double mass_min_path = 0.0015; // m, above ray_offset (0.001)
    if (mass > 0.0 && dist > mass_min_path)
    {
        double m_dB = mass * std::log10((fGHz / fRef) * dist);
        if (m_dB > 0.0)
            loss += m_dB;
    }
    return loss;
}

// In-medium gain for material index iM. No validation: the caller guarantees a
// column-consistent map (via mtl_validate / obj_file_read) and iM < n_mtl.
template <typename dtype>
static inline dtype medium_gain_impl(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, arma::uword iM,
                                     dtype dist, dtype center_frequency)
{
    if (!mtl_prop)
        return (dtype)1.0;
    double fGHz = (double)center_frequency * 1e-9;
    const dtype *m_a = mtl_col(mtl_prop, "a");
    const dtype *m_b = mtl_col(mtl_prop, "b");
    const dtype *m_c = mtl_col(mtl_prop, "c");
    const dtype *m_d = mtl_col(mtl_prop, "d");
    const dtype *m_alpha = mtl_col(mtl_prop, "alpha");
    const dtype *m_alphaB = mtl_col(mtl_prop, "alphaB");
    const dtype *m_mass = mtl_col(mtl_prop, "m");
    const dtype *m_fRef = mtl_col(mtl_prop, "fRef");

    std::complex<double> eta = eta_from_coeffs(mtl_val(m_a, iM, 1.0), mtl_val(m_b, iM, 0.0),
                                               mtl_val(m_c, iM, 0.0), mtl_val(m_d, iM, 0.0),
                                               mtl_val(m_fRef, iM, 1.0), fGHz);

    double A = medium_loss_dB(eta, mtl_val(m_alpha, iM, 0.0), mtl_val(m_alphaB, iM, 0.0),
                              mtl_val(m_fRef, iM, 1.0), fGHz, (double)dist, mtl_val(m_mass, iM, 0.0));

    return (dtype)std::pow(10.0, -0.1 * A);
}

// Lumped per-entry interface attenuation in dB: power-law penetration loss plus an optional
// Lorentzian coincidence feature. Coincidence is active only when coiF > 0 and coiA != 0.
// Total is clamped to >= 0 (a coincidence dip cannot create transmission gain).
static inline double interface_loss_dB(double att, double attB,
                                       double coiF, double coiQ, double coiA,
                                       double fRef, double fGHz)
{
    if (fRef <= 0.0)
        fRef = 1.0;
    double loss = att * std::pow(fGHz / fRef, attB);
    if (coiF > 0.0 && coiA != 0.0)
    {
        double x = coiQ * (fGHz - coiF) / coiF;
        loss += coiA / (1.0 + x * x);
    }
    return loss;
}

// Lumped interface transmission gain for material index iM (the material being entered).
// No validation: the caller guarantees a column-consistent map (via mtl_validate /
// obj_file_read) and iM < n_mtl. Path-independent; applied once on entry.
template <typename dtype>
static inline dtype interface_gain_impl(const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop, arma::uword iM, dtype center_frequency)
{
    if (!mtl_prop)
        return (dtype)1.0;

    double fGHz = (double)center_frequency * 1e-9;
    const dtype *m_att = mtl_col(mtl_prop, "att");
    const dtype *m_attB = mtl_col(mtl_prop, "attB");
    const dtype *m_coiF = mtl_col(mtl_prop, "coiF");
    const dtype *m_coiQ = mtl_col(mtl_prop, "coiQ");
    const dtype *m_coiA = mtl_col(mtl_prop, "coiA");
    const dtype *m_fRef = mtl_col(mtl_prop, "fRef");

    double A = interface_loss_dB(mtl_val(m_att, iM, 0.0), mtl_val(m_attB, iM, 0.0),
                                 mtl_val(m_coiF, iM, 0.0), mtl_val(m_coiQ, iM, 0.0),
                                 mtl_val(m_coiA, iM, 0.0), mtl_val(m_fRef, iM, 1.0), fGHz);
    return (dtype)std::pow(10.0, -0.1 * A);
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

// Cross product
template <typename dtype>
static inline void crossp(dtype vx, dtype vy, dtype vz,    // Vector V
                          dtype kx, dtype ky, dtype kz,    // Vector K
                          dtype *rx, dtype *ry, dtype *rz, // Result V x K
                          bool normalize_output = false)
{
    *rx = vy * kz - vz * ky; // x-component
    *ry = vz * kx - vx * kz; // y-component
    *rz = vx * ky - vy * kx; // z-component

    if (normalize_output)
    {
        dtype scl = (dtype)1.0 / std::sqrt(*rx * *rx + *ry * *ry + *rz * *rz);
        *rx *= scl, *ry *= scl, *rz *= scl;
    }
}

// Dot product
template <typename dtype>
static inline dtype dotp(dtype vx, dtype vy, dtype vz, // Vector V
                         dtype kx, dtype ky, dtype kz, // Vector K
                         bool normalize = false)       // Option to normalize
{
    dtype dot = vx * kx + vy * ky + vz * kz; // Dot product calculation

    if (normalize)
    {
        dtype v_mag = std::sqrt(vx * vx + vy * vy + vz * vz); // Magnitude of V
        dtype k_mag = std::sqrt(kx * kx + ky * ky + kz * kz); // Magnitude of K
        if (v_mag > 0 && k_mag > 0)                           // Normalize the dot product
            dot /= (v_mag * k_mag);
    }

    return dot;
}

// Flip LR inplace
template <typename dtype>
static inline void qd_flip_lr_inplace(dtype *data_NxM, size_t M, size_t N = 3)
{
    if (M < 2)
        return;

    size_t half = M >> 1; // floor(M/2)
    for (size_t i = 0; i < half; ++i)
    {
        dtype *left = data_NxM + i * N;
        dtype *right = data_NxM + (M - 1 - i) * N;

        for (size_t c = 0; c < N; ++c)
        {
            dtype tmp = left[c];
            left[c] = right[c];
            right[c] = tmp;
        }
    }
}

// Repeat sequence of values + Optional typecast
template <typename dtypeIn, typename dtypeOut>
static void qd_repeat_sequence(const dtypeIn *sequence, size_t sequence_length, size_t repeat_value, size_t repeat_sequence, dtypeOut *output)
{
    size_t pos = 0;                                  // Position in output
    for (size_t rs = 0; rs < repeat_sequence; ++rs)  // Repeat sequence of values
        for (size_t v = 0; v < sequence_length; ++v) // Iterate through all values of the sequence
        {
            dtypeOut val = (dtypeOut)sequence[v];        // Type conversion
            for (size_t rv = 0; rv < repeat_value; ++rv) // Repeat each value
                output[pos++] = val;
        }
}

// Euler rotations to rotation matrix
template <typename dtype>
static inline void qd_rotation_matrix(const dtype *euler_angles_3xN, // Orientation vectors (Euler rotations)
                                      dtype *rotation_matrix_9xN,    // Rotation matrix in column-major ordering
                                      size_t N = 1,                  // Number of elements
                                      bool invert_y_axis = false,    // Inverts the y-axis
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
        sb = invert_y_axis ? -sb : sb;

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

// Check if elements of a vector are in a specific range
template <typename dtype>
static bool qd_in_range(const dtype *data, size_t n_elem, dtype min, dtype max, bool inclusive = true, bool check_sorted = false)
{
    if (inclusive)
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            if (data[i] < min || data[i] > max)
                return false;
            if (check_sorted && i != 0 && data[i] <= data[i - 1])
                return false;
        }
    }
    else
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            if (data[i] <= min || data[i] >= max)
                return false;
            if (check_sorted && i != 0 && data[i] <= data[i - 1])
                return false;
        }
    }
    return true;
}

// Calculate Geographic coordinates from Cartesian coordinates
template <typename dtype>
static inline void qd_geo2cart(size_t N,                  // Number of values
                               const dtype *az,           // Input azimuth angles
                               dtype *x, dtype *y,        // 2D Output coordinates (x, y)
                               const dtype *el = nullptr, // Input elevation angles (optional)
                               dtype *z = nullptr,        // Output z-coordinate (optional)
                               const dtype *r = nullptr)  // Input vector length (optional)
{
    bool use_3D = (el != nullptr);
    bool has_length = (r != nullptr);
    bool has_z = (z != nullptr);

    for (size_t i = 0; i < N; ++i)
    {
        double ca = (double)az[i];
        double sa = std::sin(ca);
        ca = std::cos(ca);

        double ce = 1.0, se = 0.0;
        if (use_3D)
        {
            ce = (double)el[i];
            se = std::sin(ce);
            ce = std::cos(ce);
        }

        double le = has_length ? (double)r[i] : 1.0;

        x[i] = dtype(le * ce * ca);
        y[i] = dtype(le * ce * sa);

        if (has_z)
            z[i] = dtype(le * se);
    }
}

template <typename dtype>
static inline void qd_geo2cart_interleaved(size_t N,                 // Number of values
                                           dtype *cart_3xN,          // Interleaved Cartesian coordinates (output)
                                           const dtype *az,          // Input azimuth angles, length N
                                           const dtype *el,          // Input elevation angles, length N
                                           const dtype *r = nullptr) // Input vector length (optional), length N
{
    bool has_length = (r != nullptr);

    for (size_t i = 0; i < N; ++i)
    {
        double ca = (double)az[i];
        double sa = std::sin(ca);
        ca = std::cos(ca);

        double ce = (double)el[i];
        double se = std::sin(ce);
        ce = std::cos(ce);

        double le = has_length ? (double)r[i] : 1.0;

        dtype *C = &cart_3xN[3 * i];
        C[0] = dtype(le * ce * ca);
        C[1] = dtype(le * ce * sa);
        C[2] = dtype(le * se);
    }
}

// Calculate Cartesian coordinates from Geographic coordinates
template <typename dtype>
static inline void qd_cart2geo(size_t n,                       // Number of values
                               dtype *az,                      // Output azimuth angles
                               const dtype *x, const dtype *y, // 2D Input coordinates (x, y)
                               dtype *el = nullptr,            // Output elevation angles (optional)
                               const dtype *z = nullptr,       // Input z-coordinate (optional)
                               dtype *r = nullptr)             // Output vector length (optional)
{
    bool use_3D = (el != nullptr);
    bool has_length = (r != nullptr);
    bool has_z = (z != nullptr);
    bool has_az = (az != nullptr);

    for (size_t in = 0; in < n; ++in)
    {
        double xx = (double)x[in];
        double yy = (double)y[in];
        double zz = has_z ? (double)z[in] : 0.0;

        double le = std::sqrt(xx * xx + yy * yy + zz * zz);

        if (has_length)
            r[in] = (dtype)le;

        le = 1.0 / le;
        xx *= le, yy *= le, zz *= le;
        xx = (xx > 1.0) ? 1.0 : xx;
        yy = (yy > 1.0) ? 1.0 : yy;
        zz = (zz > 1.0) ? 1.0 : zz;

        if (has_az)
            az[in] = (dtype)std::atan2(yy, xx);

        if (use_3D)
            el[in] = (dtype)std::asin(zz);
    }
}

template <typename dtype>
static inline void qd_cart2geo_interleaved(size_t N,              // Number of values
                                           const dtype *cart_3xN, // Interleaved Cartesian coordinates
                                           dtype *az,             // Output azimuth angles
                                           dtype *el = nullptr,   // Output elevation angles (optional)
                                           dtype *r = nullptr)    // Output vector length (optional)
{
    bool has_az = (az != nullptr);
    bool has_el = (el != nullptr);
    bool has_length = (r != nullptr);

    for (size_t i = 0; i < N; ++i)
    {
        const dtype *C = &cart_3xN[3 * i];
        double xx = (double)C[0];
        double yy = (double)C[1];
        double zz = (double)C[2];

        double le = std::sqrt(xx * xx + yy * yy + zz * zz);

        if (has_length)
            r[i] = (dtype)le;

        le = 1.0 / le;
        xx *= le, yy *= le, zz *= le;
        xx = (xx > 1.0) ? 1.0 : xx;
        yy = (yy > 1.0) ? 1.0 : yy;
        zz = (zz > 1.0) ? 1.0 : zz;

        if (has_az)
            az[i] = (dtype)std::atan2(yy, xx);

        if (has_el)
            el[i] = (dtype)std::asin(zz);
    }
}

// Calculate the matrix product X = A^T * B * C
// - A, C can be NULL to calculate X = A^T * B or X = B * C
// - Output does not need to be initialized to 0
template <typename dtype>
static inline void qd_multiply_3_mat(const dtype *A, // n rows, m columns
                                     const dtype *B, // n rows, o columns
                                     const dtype *C, // o rows, p columns
                                     dtype *X,       // m rows, p columns
                                     size_t n, size_t m, size_t o, size_t p)
{
    bool null_A = A == nullptr;
    bool null_C = C == nullptr;

    // Avoid expensive typecasts
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);

    // Calculate the output row by row
    for (size_t im = 0; im < m; ++im)
    {
        for (size_t ip = 0; ip < p; ip++) // Initialize output to zero
            X[ip * m + im] = zero;

        // Process temporary matrix T = A^H * B column-wise
        for (size_t io = 0; io < o; ++io)
        {
            // Calculate one value of the temporary matrix T
            dtype t = zero;
            for (size_t in = 0; in < n; ++in)
            {
                dtype a = (null_A) ? (im == in ? one : zero) : A[im * n + in];
                t += a * B[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (size_t ip = 0; ip < p; ++ip)
            {
                dtype c = (null_C) ? (io == ip ? one : zero) : C[ip * o + io];
                X[ip * m + im] += t * c;
            }
        }
    }
}

// Calculates the matrix product X = A^T * B * C
// - Ar, Ai, Cr, Ci can be N
// - Output does not need to be initialized to 0
template <typename dtype>
static inline void qd_multiply_3_complex_mat(const dtype *Ar, const dtype *Ai, // n rows, m columns
                                             const dtype *Br, const dtype *Bi, // n rows, o columns
                                             const dtype *Cr, const dtype *Ci, // o rows, p columns
                                             dtype *Xr, dtype *Xi,             // m rows, p columns
                                             size_t n, size_t m, size_t o, size_t p)
{
    // Avoid expensive typecasts
    constexpr dtype zero = (dtype)0.0, one = (dtype)1.0;

    // Calculate the output row by row
    for (size_t im = 0; im < m; ++im)
    {
        // Initialize output to zero
        for (size_t ip = 0; ip < p; ++ip)
            Xr[ip * m + im] = zero, Xi[ip * m + im] = zero;

        // Process temporary matrix T = A^H * B column-wise
        for (size_t io = 0; io < o; ++io)
        {
            // Calculate one value of the temporary matrix T
            dtype tR = zero, tI = zero;
            for (size_t in = 0; in < n; ++in)
            {
                dtype a_real = (Ar == nullptr) ? (im == in ? one : zero) : Ar[im * n + in];
                dtype a_imag = (Ai == nullptr) ? zero : Ai[im * n + in];
                tR += a_real * Br[io * n + in] - a_imag * Bi[io * n + in];
                tI += a_real * Bi[io * n + in] + a_imag * Br[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (size_t ip = 0; ip < p; ++ip)
            {
                dtype c_real = (Cr == nullptr) ? (io == ip ? one : zero) : Cr[ip * o + io];
                dtype c_imag = (Ci == nullptr) ? zero : Ci[ip * o + io];
                Xr[ip * m + im] += tR * c_real - tI * c_imag;
                Xi[ip * m + im] += tR * c_imag + tI * c_real;
            }
        }
    }
}

// Multiply with scalar
template <typename dtype>
static inline void qd_multiply_scalar(dtype scalar, dtype *data, size_t N)
{
    for (size_t i = 0; i < N; ++i)
        data[i] *= scalar;
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

// Calculates X = abs( A ).^2 + abs ( B ).^2
// - Optional normalization of the columns by their sum-power
// - Returns identity matrix if normalization is true and inputs A/B are NULL
template <typename dtype>
static inline void qd_power_mat(size_t n, size_t m,                                   // Matrix dimensions (n=rows, m=columns)
                                dtype *X,                                             // Output X with n rows, m columns
                                bool normalize_columns = false,                       // Optional normalization
                                const dtype *Ar = nullptr, const dtype *Ai = nullptr, // Input A with n rows, m columns
                                const dtype *Br = nullptr, const dtype *Bi = nullptr) // Input B with n rows, m columns
{
    constexpr dtype zero = (dtype)0.0, one = (dtype)1.0, limit = (dtype)1.0e-10;
    dtype avg = one / (dtype)n;

    for (size_t im = 0; im < n * m; im += n)
    {
        dtype sum = zero;
        for (size_t in = im; in < im + n; ++in)
        {
            X[in] = zero;
            X[in] += (Ar == nullptr) ? zero : Ar[in] * Ar[in];
            X[in] += (Ai == nullptr) ? zero : Ai[in] * Ai[in];
            X[in] += (Br == nullptr) ? zero : Br[in] * Br[in];
            X[in] += (Bi == nullptr) ? zero : Bi[in] * Bi[in];
            sum += X[in];
        }
        if (normalize_columns)
        {
            if (Ar == nullptr && Ai == nullptr && Br == nullptr && Bi == nullptr) // Return identity matrix
                X[im + im / n] = one;
            else if (sum > limit) // Scale values by sum
            {
                sum = one / sum;
                for (size_t in = im; in < im + n; ++in)
                    X[in] *= sum;
            }
            else
                for (size_t in = im; in < im + n; ++in)
                    X[in] = avg;
        }
    }
}

// Combine interpolated TX/RX antenna patterns with the polarization transfer
// matrix and applies the propagation phase to produce one slice of MIMO
// channel coefficients (one path, all TX-RX element pairs).
//
// For each link index R = 0 .. n_links-1 the operation is:
//
//   Jones product (complex multiply-accumulate across 4 polarization components):
//     c = Vr * M_VV * Vt  +  Hr * M_HV * Vt  +  Vr * M_VH * Ht  +  Hr * M_HH * Ht
//   where Vr, Hr, Vt, Ht and M_xx are complex, giving complex c = re + j*im.
//
//   Phase rotation from element-specific path delay:
//     phase = wave_number * fmod(p_delays[R], wavelength)
//     p_coeff_re[R] = ( re*cos(phase) + im*sin(phase)) * path_amplitude
//     p_coeff_im[R] = (-re*sin(phase) + im*cos(phase)) * path_amplitude
template <typename dtype>
inline void qd_coeff_combine(const dtype *pVrr, const dtype *pVri, // RX V-pol pattern (real, imag), length n_links
                             const dtype *pHrr, const dtype *pHri, // RX H-pol pattern (real, imag), length n_links
                             const dtype *pVtr, const dtype *pVti, // TX V-pol pattern (real, imag), length n_links
                             const dtype *pHtr, const dtype *pHti, // TX H-pol pattern (real, imag), length n_links
                             const dtype *pM,                      // Polarization transfer matrix, 8 values broadcast over all links: {Re_VV, Im_VV, Re_HV, Im_HV, Re_VH, Im_VH, Re_HH, Im_HH}
                             const dtype *p_delays,                // Per-link path lengths in [m], length n_links  (read-only)
                             dtype wave_number,                    // 2*pi*f/c
                             dtype wavelength,                     // wavelength  : c/f
                             dtype path_amplitude,                 // sqrt(path_gain) scaling factor (set 0 for fake LOS path)
                             dtype *p_coeff_re,                    // Output real part of coefficients, length n_links
                             dtype *p_coeff_im,                    // Output imag part of coefficients, length n_links
                             size_t n_links)                       // Number of TX-RX element pairs (n_tx * n_rx)
{
    for (size_t R = 0; R < n_links; ++R)
    {
        // Jones product – real part
        dtype re = pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
        re += pHrr[R] * pM[2] * pVtr[R] - pHri[R] * pM[3] * pVtr[R] - pHrr[R] * pM[3] * pVti[R] - pHri[R] * pM[2] * pVti[R];
        re += pVrr[R] * pM[4] * pHtr[R] - pVri[R] * pM[5] * pHtr[R] - pVrr[R] * pM[5] * pHti[R] - pVri[R] * pM[4] * pHti[R];
        re += pHrr[R] * pM[6] * pHtr[R] - pHri[R] * pM[7] * pHtr[R] - pHrr[R] * pM[7] * pHti[R] - pHri[R] * pM[6] * pHti[R];

        // Jones product – imaginary part
        dtype im = pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
        im += pHrr[R] * pM[3] * pVtr[R] + pHri[R] * pM[2] * pVtr[R] + pHrr[R] * pM[2] * pVti[R] - pHri[R] * pM[3] * pVti[R];
        im += pVrr[R] * pM[5] * pHtr[R] + pVri[R] * pM[4] * pHtr[R] + pVrr[R] * pM[4] * pHti[R] - pVri[R] * pM[5] * pHti[R];
        im += pHrr[R] * pM[7] * pHtr[R] + pHri[R] * pM[6] * pHtr[R] + pHrr[R] * pM[6] * pHti[R] - pHri[R] * pM[7] * pHti[R];

        // Phase rotation from element-specific propagation delay
        dtype phase = wave_number * std::fmod(p_delays[R], wavelength);
        dtype cp = std::cos(phase), sp = std::sin(phase);

        p_coeff_re[R] = (re * cp + im * sp) * path_amplitude;
        p_coeff_im[R] = (-re * sp + im * cp) * path_amplitude;
    }
}

#endif
