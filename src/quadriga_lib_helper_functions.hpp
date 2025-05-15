// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

// A collection of small reusable helper functions to reduce copy and pasting code

#ifndef quadriga_lib_helper_H
#define quadriga_lib_helper_H

#include <cstring>

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
                                      size_t N = 1,               // Number of elements
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

// Apply Euler rotations inplace
template <typename dtype>
static inline void qd_rotate_inplace(dtype bank, dtype tilt, dtype heading, dtype *data3xN, size_t N)
{

    double O[3] = {bank, tilt, heading};
    double R[9];
    qd_rotation_matrix(O, R);

    for (size_t ix = 0; ix < 3 * N; ix += 3)
    {
        size_t iy = ix + 1, iz = ix + 2;

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

#endif
