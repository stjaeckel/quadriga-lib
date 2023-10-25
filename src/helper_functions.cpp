// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

// A collection of small helper functions to reduce copy and pasting code

// Helper-function "quick_rotate"
template <typename dtype>
inline void quick_rotate_inplace(dtype bank, dtype tilt, dtype heading, dtype *data3xN, unsigned long long N)
{
    dtype cc = std::cos(bank), sc = std::sin(bank);
    dtype cb = std::cos(tilt), sb = std::sin(tilt);
    dtype ca = std::cos(heading), sa = std::sin(heading);

    dtype R[9]; // Rotation Matrix
    R[0] = ca * cb;
    R[1] = sa * cb;
    R[2] = -sb;
    R[3] = ca * sb * sc - sa * cc;
    R[4] = sa * sb * sc + ca * cc;
    R[5] = cb * sc;
    R[6] = ca * sb * cc + sa * sc;
    R[7] = sa * sb * cc - ca * sc;
    R[8] = cb * cc;

    for (auto i = 0ULL; i < N; ++i)
    {
        auto ix = 3ULL * i, iy = ix + 1ULL, iz = ix + 2ULL;
        dtype a = R[0] * data3xN[ix] + R[3] * data3xN[iy] + R[6] * data3xN[iz];
        dtype b = R[1] * data3xN[ix] + R[4] * data3xN[iy] + R[7] * data3xN[iz];
        dtype c = R[2] * data3xN[ix] + R[5] * data3xN[iy] + R[8] * data3xN[iz];
        data3xN[ix] = a, data3xN[iy] = b, data3xN[iz] = c;
    }
}

// Helper function "quick_geo2cart"
template <typename dtype>
inline void quick_geo2cart(unsigned long long n,      // Number of values
                           const dtype *az,           // Input azimuth angles
                           dtype *x, dtype *y,        // 2D Output coordinates (x, y)
                           const dtype *el = nullptr, // Input elevation angles (optional)
                           dtype *z = nullptr,        // Output z-coordinate (optional)
                           const dtype *r = nullptr)  // Input vector length (optional)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);
    for (auto in = 0ULL; in < n; ++in)
    {
        dtype ca = az[in], sa = std::sin(ca);
        ca = std::cos(ca);

        dtype ce = (el == nullptr) ? one : std::cos(el[in]);
        dtype se = (el == nullptr) ? zero : std::sin(el[in]);

        dtype le = (r == nullptr) ? one : r[in];

        x[in] = le * ce * ca;
        y[in] = le * ce * sa;

        if (z != nullptr)
            z[in] = le * se;
    }
}

// Helper function "quick_cart2geo"
template <typename dtype>
inline void quick_cart2geo(unsigned long long n,           // Number of values
                           dtype *az,                      // Output azimuth angles
                           const dtype *x, const dtype *y, // 2D Input coordinates (x, y)
                           dtype *el = nullptr,            // Output elevation angles (optional)
                           const dtype *z = nullptr,       // Input z-coordinate (optional)
                           dtype *r = nullptr)             // Output vector length (optional)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);
    for (auto in = 0ULL; in < n; ++in)
    {
        dtype xx = x[in], yy = y[in], zz = (z == nullptr) ? zero : z[in];
        dtype le = std::sqrt(xx * xx + yy * yy + zz * zz);

        if (r != nullptr)
            r[in] = le;

        le = one / le;
        xx *= le, yy *= le, zz *= le;
        xx = xx > one ? one : xx, yy = yy > one ? one : yy, zz = zz > one ? one : zz;

        if (az != nullptr)
            az[in] = std::atan2(yy, xx);

        if (el != nullptr)
            el[in] = std::asin(zz);
    }
}

// Helper function "quick_multiply_3_mat"
// Calculates the matrix product X = A^T * B * C
// A, C can be NULL
template <typename dtype>
inline void quick_multiply_3_mat(const dtype *A, // n rows, m columns
                                 const dtype *B, // n rows, o columns
                                 const dtype *C, // o rows, p columns
                                 dtype *X,       // m rows, p columns
                                 unsigned long long n,
                                 unsigned long long m,
                                 unsigned long long o,
                                 unsigned long long p)
{
    // Avoid expensive typecasts
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);

    // Calculate the output row by row
    for (auto im = 0ULL; im < m; ++im)
    {
        for (auto ip = 0ULL; ip < p; ip++) // Initialize output to zero
            X[ip * m + im] = zero;

        // Process temporary matrix T = A^H * B column-wise
        for (auto io = 0ULL; io < o; ++io)
        {
            // Calculate one value of the temporary matrix T
            dtype t = zero;
            for (auto in = 0ULL; in < n; ++in)
            {
                dtype a = (A == nullptr) ? (im == in ? one : zero) : A[im * n + in];
                t += a * B[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (auto ip = 0ULL; ip < p; ++ip)
            {
                dtype c = (C == nullptr) ? (io == ip ? one : zero) : C[ip * o + io];
                X[ip * m + im] += t * c;
            }
        }
    }
}

// Helper function "quick_multiply_3_complex_mat"
// Calculates the matrix product X = A^T * B * C
// Ar, Ai, Cr, Ci can be NULL
template <typename dtype>
inline void quick_multiply_3_complex_mat(const dtype *Ar, const dtype *Ai, // n rows, m columns
                                         const dtype *Br, const dtype *Bi, // n rows, o columns
                                         const dtype *Cr, const dtype *Ci, // o rows, p columns
                                         dtype *Xr, dtype *Xi,             // m rows, p columns
                                         unsigned long long n,
                                         unsigned long long m,
                                         unsigned long long o,
                                         unsigned long long p)
{
    // Avoid expensive typecasts
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);

    // Calculate the output row by row
    for (auto im = 0ULL; im < m; ++im)
    {
        // Initialize output to zero
        for (auto ip = 0ULL; ip < p; ++ip)
            Xr[ip * m + im] = zero, Xi[ip * m + im] = zero;

        // Process temporary matrix T = A^H * B column-wise
        for (auto io = 0ULL; io < o; ++io)
        {
            // Calculate one value of the temporary matrix T
            dtype tR = zero, tI = zero;
            for (auto in = 0ULL; in < n; ++in)
            {
                dtype a_real = (Ar == nullptr) ? (im == in ? one : zero) : Ar[im * n + in];
                dtype a_imag = (Ai == nullptr) ? zero : Ai[im * n + in];
                tR += a_real * Br[io * n + in] - a_imag * Bi[io * n + in];
                tI += a_real * Bi[io * n + in] + a_imag * Br[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (auto ip = 0ULL; ip < p; ++ip)
            {
                dtype c_real = (Cr == nullptr) ? (io == ip ? one : zero) : Cr[ip * o + io];
                dtype c_imag = (Ci == nullptr) ? zero : Ci[ip * o + io];
                Xr[ip * m + im] += tR * c_real - tI * c_imag;
                Xi[ip * m + im] += tR * c_imag + tI * c_real;
            }
        }
    }
}

// Helper function "quick_power_mat"
// - Calculates X = abs( A ).^2 + abs ( B ).^2
// - Optional normalization of the columns by their sum-power
// - Returns identity matrix normalization is true and inputs A/B are NULL
template <typename dtype>
inline void quick_power_mat(unsigned long long n, unsigned long long m,           // Matrix dimensions (n=rows, m=columns)
                            dtype *X,                                             // Output X with n rows, m columns
                            bool normalize_columns = false,                       // Optional normalization
                            const dtype *Ar = nullptr, const dtype *Ai = nullptr, // Input A with n rows, m columns
                            const dtype *Br = nullptr, const dtype *Bi = nullptr) // Input B with n rows, m columns
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0), limit = dtype(1.0e-10);
    dtype avg = one / dtype(n);

    for (auto im = 0ULL; im < n * m; im += n)
    {
        dtype sum = zero;
        for (auto in = im; in < im + n; ++in)
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
                for (auto in = im; in < im + n; ++in)
                    X[in] *= sum;
            }
            else
                for (auto in = im; in < im + n; ++in)
                    X[in] = avg;
        }
    }
}
