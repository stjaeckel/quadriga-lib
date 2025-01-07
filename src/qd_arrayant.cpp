// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <stdexcept>
#include <cstring>   // std::memcopy
#include <iomanip>   // std::setprecision
#include <algorithm> // std::replace
#include <iostream>

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "qd_arrayant_functions.hpp"

// Helper-function "quick_rotate"
template <typename dtype>
static inline void quick_rotate_inplace(dtype bank, dtype tilt, dtype heading, dtype *data3xN, size_t N)
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

    for (size_t i = 0; i < N; ++i)
    {
        size_t ix = 3 * i, iy = ix + 1, iz = ix + 2;
        dtype a = R[0] * data3xN[ix] + R[3] * data3xN[iy] + R[6] * data3xN[iz];
        dtype b = R[1] * data3xN[ix] + R[4] * data3xN[iy] + R[7] * data3xN[iz];
        dtype c = R[2] * data3xN[ix] + R[5] * data3xN[iy] + R[8] * data3xN[iz];
        data3xN[ix] = a, data3xN[iy] = b, data3xN[iz] = c;
    }
}

// Helper function "quick_geo2cart"
template <typename dtype>
static inline void quick_geo2cart(size_t n,                  // Number of values
                                  const dtype *az,           // Input azimuth angles
                                  dtype *x, dtype *y,        // 2D Output coordinates (x, y)
                                  const dtype *el = nullptr, // Input elevation angles (optional)
                                  dtype *z = nullptr,        // Output z-coordinate (optional)
                                  const dtype *r = nullptr)  // Input vector length (optional)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);
    for (size_t in = 0; in < n; ++in)
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
static inline void quick_cart2geo(size_t n,                       // Number of values
                                  dtype *az,                      // Output azimuth angles
                                  const dtype *x, const dtype *y, // 2D Input coordinates (x, y)
                                  dtype *el = nullptr,            // Output elevation angles (optional)
                                  const dtype *z = nullptr,       // Input z-coordinate (optional)
                                  dtype *r = nullptr)             // Output vector length (optional)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);
    for (size_t in = 0; in < n; ++in)
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
static inline void quick_multiply_3_mat(const dtype *A, // n rows, m columns
                                        const dtype *B, // n rows, o columns
                                        const dtype *C, // o rows, p columns
                                        dtype *X,       // m rows, p columns
                                        size_t n, size_t m, size_t o, size_t p)
{
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
                dtype a = (A == nullptr) ? (im == in ? one : zero) : A[im * n + in];
                t += a * B[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (size_t ip = 0; ip < p; ++ip)
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
static inline void quick_multiply_3_complex_mat(const dtype *Ar, const dtype *Ai, // n rows, m columns
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

// Helper function "quick_power_mat"
// - Calculates X = abs( A ).^2 + abs ( B ).^2
// - Optional normalization of the columns by their sum-power
// - Returns identity matrix normalization is true and inputs A/B are NULL
template <typename dtype>
static inline void quick_power_mat(size_t n, size_t m,                                   // Matrix dimensions (n=rows, m=columns)
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

/*!SECTION
Array antenna class
SECTION!*/

/*!MD
# arrayant <++>
Class for storing and manipulating array antenna models

## Description:
- An array antenna consists of multiple individual elements.
- Each element occupies a specific position relative to the array's phase-center, its local origin.
- Elements can also be inter-coupled, represented by a coupling matrix.

## Attributes:
`arma::Cube<dtype> e_theta_re`    | Vertical component of the electric field, real part
`arma::Cube<dtype> e_theta_im`    | Vertical component of the electric field, imaginary part
`arma::Cube<dtype> e_phi_re`      | Horizontal component of the electric field, real part
`arma::Cube<dtype> e_phi_im`      | Horizontal component of the electric field, imaginary part
`arma::Col<dtype> azimuth_grid`   | Azimuth angles in pattern (theta) in [rad], between -pi and pi, sorted
`arma::Col<dtype> elevation_grid` | Elevation angles in pattern (phi) in [rad], between -pi/2 and pi/2, sorted
`arma::Mat<dtype> element_pos`    | Element positions (optional), Size: Empty or [3, n_elements]
`arma::Mat<dtype> coupling_re`    | Coupling matrix, real part (optional), Size: [n_elements, n_ports]
`arma::Mat<dtype> coupling_im`    | Coupling matrix, imaginary part (optional), Size: [n_elements, n_ports]
`dtype center_frequency`          | Center frequency in [Hz]

- Allowed datatypes (`dtype`): `float` and `double`
- `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` must have size `[n_elevation, n_azimuth, n_elements]`

## Example:
```
float pi = arma::datum::pi;

quadriga_lib::arrayant<float> ant;
ant.azimuth_grid = {-0.75f * pi, 0.0f, 0.75f * pi, pi};
ant.elevation_grid = {-0.45f * pi, 0.0f, 0.45f * pi};

arma::mat A = arma::linspace(1.0, 12.0, 12);
A.reshape(3, 4);

arma::fcube B;
B.zeros(3, 4, 1);
B.slice(0) = arma::conv_to<arma::fmat>::from(A);

ant.e_theta_re = B * 0.5f;
ant.e_theta_im = B * 0.002f;
ant.e_phi_re = -B;
ant.e_phi_im = -B * 0.001f;

arma::fmat C = {1.0f, 2.0f, 4.0f};
ant.element_pos = C.t();

ant.coupling_re = {1.0f};
ant.coupling_im = {0.1f};
ant.center_frequency = 2.0e9f;
ant.name = "name";
```

## Simple member functions:
`.n_elevation()` | Returns number of elevation angles as 64bit integer
`.n_azimuth()`   | Returns number of azimuth angles as 64bit integer
`.n_elements()`  | Returns number of antenna elements as 64bit integer
`.n_ports()`     | Returns number of ports (after coupling) as 64bit integer
`.copy()`        | Creates a copy of the array antenna object
`.reset()`       | Reset the size to zero (the arrayant object will contain no data)
`.is_valid()`    | Returns an empty string if arrayant object is valid or an error message otherwise

## Complex member fuctions:
- <a href="#.calc_directivity_dbi">.calc_directivity_dBi</a>
- <a href="#.combine_pattern">.combine_pattern</a>

MD!*/

template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_elevation() const
{
    return e_theta_re.n_rows;
}
template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_azimuth() const
{
    return e_theta_re.n_cols;
}
template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_elements() const
{
    return e_theta_re.n_slices;
}
template <typename dtype>
arma::uword quadriga_lib::arrayant<dtype>::n_ports() const
{
    if (coupling_re.empty() && coupling_im.empty())
        return e_theta_re.n_slices;
    else if (coupling_re.empty())
        return coupling_im.n_cols;
    else
        return coupling_re.n_cols;
}

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::append(const quadriga_lib::arrayant<dtype> *new_arrayant) const
{
    // Check if arrayant objects are valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());
    error_message = new_arrayant->is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Check if grids are identical
    bool eq = arma::approx_equal(azimuth_grid, new_arrayant->azimuth_grid, "absdiff", (dtype)1.0e-6);
    if (!eq)
    {
        error_message = "Azimuth sampling grids don't match.";
        throw std::invalid_argument(error_message.c_str());
    }
    eq = arma::approx_equal(elevation_grid, new_arrayant->elevation_grid, "absdiff", (dtype)1.0e-6);
    if (!eq)
    {
        error_message = "Elevation sampling grids don't match.";
        throw std::invalid_argument(error_message.c_str());
    }

    arma::uword n_elevation = this->n_elevation();
    arma::uword n_azimuth = this->n_azimuth();
    arma::uword n_elements_1 = this->n_elements();
    arma::uword n_elements_2 = new_arrayant->n_elements();
    arma::uword n_ports_1 = this->n_ports();
    arma::uword n_ports_2 = new_arrayant->n_ports();

    quadriga_lib::arrayant<dtype> output;
    output.set_size(n_elevation, n_azimuth, n_elements_1 + n_elements_2, n_ports_1 + n_ports_2);

    // Copy data from first antenna
    std::memcpy(output.azimuth_grid.memptr(), this->azimuth_grid.memptr(), n_azimuth * sizeof(dtype));
    std::memcpy(output.elevation_grid.memptr(), this->elevation_grid.memptr(), n_elevation * sizeof(dtype));
    std::memcpy(output.e_theta_re.slice_memptr(0), this->e_theta_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(0), this->e_theta_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(0), this->e_phi_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(0), this->e_phi_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_1 * sizeof(dtype));
    std::memcpy(output.element_pos.colptr(0), this->element_pos.colptr(0), 3 * n_elements_1 * sizeof(dtype));
    for (arma::uword n = 0; n < n_ports_1; ++n)
    {
        std::memcpy(output.coupling_re.colptr(n), this->coupling_re.colptr(n), n_elements_1 * sizeof(dtype));
        std::memcpy(output.coupling_im.colptr(n), this->coupling_im.colptr(n), n_elements_1 * sizeof(dtype));
    }

    // Copy data from second antenna
    std::memcpy(output.e_theta_re.slice_memptr(n_elements_1), new_arrayant->e_theta_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_theta_im.slice_memptr(n_elements_1), new_arrayant->e_theta_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_re.slice_memptr(n_elements_1), new_arrayant->e_phi_re.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.e_phi_im.slice_memptr(n_elements_1), new_arrayant->e_phi_im.slice_memptr(0), n_azimuth * n_elevation * n_elements_2 * sizeof(dtype));
    std::memcpy(output.element_pos.colptr(n_elements_1), new_arrayant->element_pos.colptr(0), 3 * n_elements_2 * sizeof(dtype));
    for (arma::uword n = 0; n < n_ports_2; ++n)
    {
        std::memcpy(output.coupling_re.colptr(n + n_ports_1) + n_elements_1, new_arrayant->coupling_re.colptr(n), n_elements_2 * sizeof(dtype));
        std::memcpy(output.coupling_im.colptr(n + n_ports_1) + n_elements_1, new_arrayant->coupling_im.colptr(n), n_elements_2 * sizeof(dtype));
    }

    return output;
}

/*!MD
# .calc_directivity_dBi
Calculate the directivity (in dBi) of array antenna elements

## Description:
- Member function of <a href="#arrayant">arrayant</a>
- Directivity is a parameter of an antenna or which measures the degree to which the radiation emitted
  is concentrated in a single direction. It is the ratio of the radiation intensity in a given direction
  from the antenna to the radiation intensity averaged over all directions. Therefore, the directivity
  of a hypothetical isotropic radiator is 1, or 0 dBi.
- Allowed datatypes (`dtype`): `float` and `double`

## Declaration:
```
dtype calc_directivity_dBi(unsigned element) const;
```
## Arguments:
`unsigned element` | Element index

## Example:
```
auto ant = quadriga_lib::generate_arrayant_dipole<float>();
float directivity = ant.calc_directivity_dBi( 0 );
```
MD!*/

template <typename dtype>
dtype quadriga_lib::arrayant<dtype>::calc_directivity_dBi(arma::uword i_element) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() == 0 && i_element >= n_elements())
        error_message = "Element index out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Constants
    double pi2 = 2.0 * arma::datum::pi, pi_half = 0.5 * arma::datum::pi;
    arma::uword naz = azimuth_grid.n_elem, nel = elevation_grid.n_elem;

    // Az and El grid as double
    arma::vec az = arma::conv_to<arma::vec>::from(azimuth_grid);
    arma::vec el = arma::conv_to<arma::vec>::from(elevation_grid);

    // Calculate the azimuth weights
    arma::vec waz(naz, arma::fill::none);
    double *ptr = waz.memptr();
    for (arma::uword i = 0; i < naz; ++i)
    {
        double x = i == 0 ? az.at(naz - 1) - pi2 : az.at(i - 1);
        double y = i == naz - 1 ? az.at(0) + pi2 : az.at(i + 1);
        ptr[i] = y - x;
    }

    // Calculate the elevation weights
    arma::vec wel(nel, arma::fill::none);
    ptr = wel.memptr();
    for (arma::uword i = 0; i < nel; ++i)
    {
        double x = i == 0 ? -pi_half : 0.5 * el.at(i - 1) + 0.5 * el.at(i);
        double y = i == nel - 1 ? pi_half : 0.5 * el.at(i) + 0.5 * elevation_grid.at(i + 1);
        arma::vec tmp = arma::linspace<arma::vec>(x, y, 21);
        double val = arma::accu(arma::cos(tmp));
        ptr[i] = val * (y - x);
    }

    // Combine weights
    arma::mat W(nel, naz, arma::fill::none);
    ptr = W.memptr();
    double norm = 0.0;
    for (double *col = waz.begin(); col != waz.end(); ++col)
        for (double *row = wel.begin(); row != wel.end(); ++row)
            *ptr = *row * *col, norm += *ptr++;
    ptr = W.memptr();
    norm = 1.0 / norm;
    for (arma::uword i = 0; i < naz * nel; ++i)
        ptr[i] *= norm;

    // Calculate the directivity
    double p_sum = 0.0, p_max = 0.0;
    const dtype *p_theta_re = e_theta_re.slice_memptr(i_element), *p_theta_im = e_theta_im.slice_memptr(i_element);
    const dtype *p_phi_re = e_phi_re.slice_memptr(i_element), *p_phi_im = e_phi_im.slice_memptr(i_element);
    for (arma::uword i = 0; i < naz * nel; ++i)
    {
        double a = double(p_theta_re[i]), b = double(p_theta_im[i]), c = double(p_phi_re[i]), d = double(p_phi_im[i]);
        double pow = a * a + b * b + c * c + d * d;
        p_max = pow > p_max ? pow : p_max;
        p_sum += pow * ptr[i];
    }

    double directivity = p_max < 1.0e-14 ? 0.0 : 10.0 * std::log10(p_max / p_sum);
    return dtype(directivity);
}

/*!MD
# .combine_pattern
Calculate effective radiation patterns for array antennas

## Description:
- Member function of <a href="#arrayant">arrayant</a>
- By integrating element radiation patterns, element positions, and the coupling weights, one can
  determine an effective radiation pattern observable by a receiver in the antenna's far field.
- Leveraging these effective patterns is especially beneficial in antenna design, beamforming
  applications such as in 5G systems, and in planning wireless communication networks in complex
  environments like urban areas. This streamlined approach offers a significant boost in computation
  speed when calculating MIMO channel coefficients, as it reduces the number of necessary operations.
- Allowed datatypes (`dtype`): `float` and `double`

## Declaration:
```
void combine_pattern(arrayant<dtype> *output);
```

## Arguments:
- `arrayant<dtype> *output`<br>
  Pointer to an arrayant object the the results should be written to. Calling this function without
  an argument or passing `nullptr` updates the arrayant properties of `this` arrayant inplace.
MD!*/

template <typename dtype>
void quadriga_lib::arrayant<dtype>::combine_pattern(quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    dtype pi = dtype(arma::datum::pi);
    dtype lambda = dtype(299792448.0) / center_frequency;
    dtype wave_no = dtype(2.0) * pi / lambda;

    unsigned long long n_el = e_theta_re.n_rows;
    unsigned long long n_az = e_theta_re.n_cols;
    unsigned long long n_out = e_theta_re.n_slices;
    unsigned long long n_ang = n_el * n_az;
    unsigned long long n_prt = coupling_re.n_cols;
    unsigned n32_out = unsigned(n_out);

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr(),
          *p_phi = azimuth_grid.memptr(), *p_theta = elevation_grid.memptr();

    for (auto ia = 0ULL; ia < n_az; ++ia)
        for (auto ie = 0ULL; ie < n_el; ++ie)
            *p_azimuth++ = p_phi[ia], *p_elevation++ = p_theta[ie];

    // Interpolate the pattern data
    arma::Col<unsigned> i_element = arma::linspace<arma::Col<unsigned>>(1, n32_out, n32_out);
    arma::Cube<dtype> orientation(3, 1, 1);
    arma::Mat<dtype> V_re(n_out, n_ang), V_im(n_out, n_ang), H_re(n_out, n_ang), H_im(n_out, n_ang), dist(n_out, n_ang);
    arma::Mat<dtype> EMPTY;

    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos,
                            &V_re, &V_im, &H_re, &H_im, &dist, &EMPTY, &EMPTY, &EMPTY);

    // Apply phase shift caused by element positions
    arma::Mat<std::complex<dtype>> phase(arma::cos(wave_no * dist), arma::sin(-wave_no * dist));
    arma::Mat<std::complex<dtype>> Vi(V_re, V_im), Hi(H_re, H_im);
    Vi = Vi % phase, Hi = Hi % phase;

    // Apply coupling
    arma::Mat<std::complex<dtype>> coupling(coupling_re, coupling_im);
    arma::Mat<std::complex<dtype>> Vo(n_ang, n_prt), Ho(n_ang, n_prt);
    for (auto i = 0ULL; i < n_out; ++i)
    {
        arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col(), hi = Hi.row(i).as_col();
        for (auto o = 0ULL; o < n_prt; ++o)
        {
            std::complex<dtype> cpl = coupling.at(i, o);
            Vo.col(o) += vi * cpl, Ho.col(o) += hi * cpl;
        }
    }

    if (output != nullptr) // Write output data
    {
        output->set_size(n_el, n_az, n_prt, n_prt);

        std::memcpy(output->azimuth_grid.memptr(), azimuth_grid.memptr(), n_az * sizeof(dtype));
        std::memcpy(output->elevation_grid.memptr(), elevation_grid.memptr(), n_el * sizeof(dtype));

        arma::Mat<dtype> cpy = arma::real(Vo);
        std::memcpy(output->e_theta_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = arma::imag(Vo);
        std::memcpy(output->e_theta_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = arma::real(Ho);
        std::memcpy(output->e_phi_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = arma::imag(Ho);
        std::memcpy(output->e_phi_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        output->element_pos.zeros();
        output->coupling_re.eye();
        output->coupling_im.zeros();
        output->center_frequency = center_frequency;

        // Set the data pointers for the quick check.
        output->check_ptr[0] = output->e_theta_re.memptr();
        output->check_ptr[1] = output->e_theta_im.memptr();
        output->check_ptr[2] = output->e_phi_re.memptr();
        output->check_ptr[3] = output->e_phi_im.memptr();
        output->check_ptr[4] = output->azimuth_grid.memptr();
        output->check_ptr[5] = output->elevation_grid.memptr();
        output->check_ptr[6] = output->element_pos.memptr();
        output->check_ptr[7] = output->coupling_re.memptr();
        output->check_ptr[8] = output->coupling_im.memptr();
    }
    else if (read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }
    else // Update the properties of current array antenna object
    {
        arma::Mat<dtype> cpy = real(Vo);
        e_theta_re.set_size(n_el, n_az, n_prt);
        std::memcpy(e_theta_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = imag(Vo);
        e_theta_im.set_size(n_el, n_az, n_prt);
        std::memcpy(e_theta_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = real(Ho);
        e_phi_re.set_size(n_el, n_az, n_prt);
        std::memcpy(e_phi_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = imag(Ho);
        e_phi_im.set_size(n_el, n_az, n_prt);
        std::memcpy(e_phi_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        element_pos.zeros(3, n_prt);
        coupling_re.eye(n_prt, n_prt);
        coupling_im.zeros(n_prt, n_prt);

        // Set the data pointers for the quick check.
        check_ptr[0] = e_theta_re.memptr();
        check_ptr[1] = e_theta_im.memptr();
        check_ptr[2] = e_phi_re.memptr();
        check_ptr[3] = e_phi_im.memptr();
        check_ptr[4] = azimuth_grid.memptr();
        check_ptr[5] = elevation_grid.memptr();
        check_ptr[6] = element_pos.memptr();
        check_ptr[7] = coupling_re.memptr();
        check_ptr[8] = coupling_im.memptr();
    }
}

// Creates a copy of the array antenna object
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::arrayant<dtype>::copy() const
{
    quadriga_lib::arrayant<dtype> ant;

    ant.name = name;
    ant.e_theta_re = e_theta_re;
    ant.e_theta_im = e_theta_im;
    ant.e_phi_re = e_phi_re;
    ant.e_phi_im = e_phi_im;
    ant.azimuth_grid = azimuth_grid;
    ant.elevation_grid = elevation_grid;
    ant.element_pos = element_pos;
    ant.coupling_re = coupling_re;
    ant.coupling_im = coupling_im;
    ant.center_frequency = center_frequency;
    ant.read_only = false;

    return ant;
}

// Copy antenna elements, enlarge array size if needed
template <typename dtype>
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uvec destination)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_el = e_theta_re.n_rows;
    arma::uword n_az = e_theta_re.n_cols;
    arma::uword n_ang = n_el * n_az;
    arma::uword n_elements = e_theta_re.n_slices;
    arma::uword n_element_max = destination.max() + 1;
    arma::uword n_ports = coupling_re.n_cols;

    if (source >= n_elements)
        error_message = "Source index out of bound";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Enlarge existing arrayant object
    if (n_element_max > n_elements)
    {
        arma::uword added_elements = n_element_max - n_elements;
        e_theta_re.resize(n_el, n_az, n_element_max);
        e_theta_im.resize(n_el, n_az, n_element_max);
        e_phi_re.resize(n_el, n_az, n_element_max);
        e_phi_im.resize(n_el, n_az, n_element_max);
        element_pos.resize(3, n_element_max);
        coupling_re.resize(n_element_max, n_ports + added_elements);
        coupling_im.resize(n_element_max, n_ports + added_elements);
        for (arma::uword i = 0; i < added_elements; ++i)
            coupling_re.at(n_elements + i, n_ports + i) = (dtype)1.0;
    }

    // Copy data from source to destination
    for (auto dest = destination.begin(); dest != destination.end(); ++dest)
        if (source != *dest)
        {
            std::memcpy(e_theta_re.slice_memptr(*dest), e_theta_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_theta_im.slice_memptr(*dest), e_theta_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_phi_re.slice_memptr(*dest), e_phi_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_phi_im.slice_memptr(*dest), e_phi_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(element_pos.colptr(*dest), element_pos.colptr(source), 3 * sizeof(dtype));
        }

    // Set the data pointers for the quick check.
    check_ptr[0] = e_theta_re.memptr();
    check_ptr[1] = e_theta_im.memptr();
    check_ptr[2] = e_phi_re.memptr();
    check_ptr[3] = e_phi_im.memptr();
    check_ptr[4] = azimuth_grid.memptr();
    check_ptr[5] = elevation_grid.memptr();
    check_ptr[6] = element_pos.memptr();
    check_ptr[7] = coupling_re.memptr();
    check_ptr[8] = coupling_im.memptr();
}

template <typename dtype>
void quadriga_lib::arrayant<dtype>::copy_element(arma::uword source, arma::uword destination)
{
    arma::uvec dest(1);
    dest.at(0) = destination;
    copy_element(source, dest);
}

// ARRAYANT METHOD : OBJ Export
template <typename dtype>
void quadriga_lib::arrayant<dtype>::export_obj_file(std::string fn, dtype directivity_range, std::string colormap,
                                                    dtype object_radius, arma::uword icosphere_n_div,
                                                    arma::uvec i_element) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Input validation
    std::string fn_suffix = ".obj";
    std::string fn_mtl;

    if (fn.size() >= fn_suffix.size() &&
        fn.compare(fn.size() - fn_suffix.size(), fn_suffix.size(), fn_suffix) == 0)
    {
        fn_mtl = fn.substr(0, fn.size() - fn_suffix.size()) + ".mtl";
    }
    else
        throw std::invalid_argument("OBJ-File name must end with .obj");

    // Replace spaces in ant_name
    std::string ant_name = name;
    std::replace(ant_name.begin(), ant_name.end(), ' ', '_');

    // Extract the file name from the path
    std::string fn_mtl_base;
    size_t pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1);
    else
        fn_mtl_base = fn_mtl;

    double range = -std::abs((double)directivity_range);
    if (range == 0.0)
        throw std::invalid_argument("Directivity range cannot be 0.");

    double radius = (double)object_radius;
    if (radius <= 0.0)
        throw std::invalid_argument("Object radius must be larger than 0.");

    arma::uword no_elements = n_elements();

    if (i_element.n_elem == 0)
        i_element = arma::regspace<arma::uvec>(0, no_elements - 1);

    if (arma::any(i_element >= no_elements))
        throw std::invalid_argument("Element indices 'i_element' cannot exceed the array antenna size.");

    size_t no_elements_t = (size_t)i_element.n_elem;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    size_t n_cmap = (size_t)cmap.n_rows;

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (outFile.is_open())
    {
        // Write some text to the file
        outFile << "# QuaDRiGa " << "antenna pattern colormap\n\n";
        for (size_t i = 0; i < n_cmap; ++i)
        {
            double R = (double)cmap(i, 0) / 255.0;
            double G = (double)cmap(i, 1) / 255.0;
            double B = (double)cmap(i, 2) / 255.0;
            outFile << "newmtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i << "\n";
            outFile << std::fixed << std::setprecision(6) << "Kd " << R << " " << G << " " << B << "\n\n";
        }
        outFile.close();
    }

    // Generate vertices and faces
    arma::mat vertices_icosphere; // double precision
    arma::u32_mat faces_icosphere;
    size_t no_faces, no_vert;
    {
        arma::mat center, vert;
        no_faces = quadriga_lib::icosphere<double>(icosphere_n_div, 1.0, &center, nullptr, &vert);

        vertices_icosphere.set_size(3 * no_faces, 3);
        faces_icosphere.set_size(no_faces, 3);

        double *pc = center.memptr(), *pv = vert.memptr(), *v = vertices_icosphere.memptr();
        unsigned *f_ico = faces_icosphere.memptr();

        for (size_t i_face = 0; i_face < no_faces; ++i_face)
        {
            double x, y, z;

            x = pc[i_face] + pv[i_face];
            y = pc[i_face + no_faces] + pv[i_face + no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 2 * no_faces];

            v[3 * i_face] = x;
            v[3 * i_face + 3 * no_faces] = y;
            v[3 * i_face + 6 * no_faces] = z;
            f_ico[i_face] = 3 * (unsigned)i_face + 1;

            x = pc[i_face] + pv[i_face + 3 * no_faces];
            y = pc[i_face + no_faces] + pv[i_face + 4 * no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 5 * no_faces];

            v[3 * i_face + 1] = x;
            v[3 * i_face + 3 * no_faces + 1] = y;
            v[3 * i_face + 6 * no_faces + 1] = z;
            f_ico[i_face + no_faces] = 3 * (unsigned)i_face + 2;

            x = pc[i_face] + pv[i_face + 6 * no_faces];
            y = pc[i_face + no_faces] + pv[i_face + 7 * no_faces];
            z = pc[i_face + 2 * no_faces] + pv[i_face + 8 * no_faces];

            v[3 * i_face + 2] = x;
            v[3 * i_face + 3 * no_faces + 2] = y;
            v[3 * i_face + 6 * no_faces + 2] = z;
            f_ico[i_face + 2 * no_faces] = 3 * (unsigned)i_face + 3;
        }
    }
    no_vert = 3 * no_faces;

    // Convert vertices to angles
    arma::Mat<dtype> azimuth, elevation;
    azimuth.set_size(1, no_vert);
    elevation.set_size(1, no_vert);
    {
        double *v = vertices_icosphere.memptr();
        dtype *az = azimuth.memptr(), *el = elevation.memptr();
        for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
        {
            double x = v[i_vert];
            double y = v[i_vert + no_vert];
            double z = v[i_vert + 2 * no_vert];
            x = x > 1.0 ? 1.0 : x, y = y > 1.0 ? 1.0 : y, z = z > 1.0 ? 1.0 : z;
            x = x < -1.0 ? -1.0 : x, y = y < -1.0 ? -1.0 : y, z = z < -1.0 ? -1.0 : z;
            az[i_vert] = (dtype)std::atan2(y, x);
            el[i_vert] = (dtype)std::asin(z);
        }
    }

    // Interpolate antenna patterns (for all elements in the array)
    arma::Mat<dtype> V_re, V_im, H_re, H_im;
    interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im, i_element);
    dtype *vr = V_re.memptr(), *vi = V_im.memptr(), *hr = H_re.memptr(), *hi = H_im.memptr();

    // Calculate maximum Power in the Pattern
    double max_pow = 0.0;
    for (size_t i_el = 0; i_el < no_elements_t; ++i_el)
        for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
        {
            size_t i = i_vert * no_elements_t + i_el;
            double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
            double p = a * a + b * b + c * c + d * d;
            max_pow = (p > max_pow) ? p : max_pow;
        }
    max_pow = 1.0 / max_pow;

    // Write OBJ File
    outFile = std::ofstream(fn);
    if (outFile.is_open())
    {
        outFile << "# QuaDRiGa " << "antenna pattern\n";
        outFile << "mtllib " << fn_mtl_base << "\n";

        // Face offset for previous element in the OBJ file
        unsigned face_offset = 1; // OBJ starts counting at 1

        // Export each element to the OBJ file
        double scale = 1.0 / (-range);
        for (size_t i_el = 0; i_el < no_elements_t; ++i_el)
        {
            arma::uword i_array_element = i_element(i_el);
            outFile << "\no " << "QuaDRiGa_ANT_" << ant_name << "_" << std::setfill('0') << std::setw(3) << i_array_element << "\n";

            // Calculate Shape
            arma::vec pow(no_vert);
            double *pp = pow.memptr();
            for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
            {
                size_t i = i_vert * no_elements_t + i_el;
                double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
                double p = a * a + b * b + c * c + d * d;
                p *= max_pow;                // Scale by maximum power
                p = 10.0 * std::log10(p);    // Convert to dB
                p = (p < range) ? range : p; // Set minimum
                p = (p - range) * scale;     // Set to range 0 ... 1
                pp[i_vert] = p;
            }

            // Scale vertices
            arma::mat scaled_vertices(no_vert, 3);
            double *v_scale = scaled_vertices.memptr();  // Scaled vertices
            double *v_ico = vertices_icosphere.memptr(); // Icosphere vertices
            for (size_t i_vert = 0; i_vert < no_vert; ++i_vert)
            {
                v_scale[i_vert] = v_ico[i_vert] * pp[i_vert];                             // x
                v_scale[i_vert + no_vert] = v_ico[i_vert + no_vert] * pp[i_vert];         // y
                v_scale[i_vert + 2 * no_vert] = v_ico[i_vert + 2 * no_vert] * pp[i_vert]; // z
            }

            // Process faces
            double eps = 0.001; // Co-location tolerance in [m], relative to 1 m radius
            eps *= eps;         // Squared value

            arma::mat final_vertices(no_vert, 3);
            arma::u32_mat vert_index(no_faces, 3);
            arma::u32_vec cmap_index(no_faces);

            double *v_final = final_vertices.memptr(); // Final vertices
            unsigned *f_final = vert_index.memptr();   // Final face indices
            unsigned *i_cmap = cmap_index.memptr();    // Colormap index

            size_t no_final_vert = 0; // Number of final vertices
            size_t no_final_face = 0; // Number of final faces

            for (size_t i_face = 0; i_face < no_faces; ++i_face)
            {
                // Load the 3 scaled vertices of the face
                double x0 = v_scale[3 * i_face];
                double y0 = v_scale[3 * i_face + no_vert];
                double z0 = v_scale[3 * i_face + 2 * no_vert];

                double x1 = v_scale[3 * i_face + 1];
                double y1 = v_scale[3 * i_face + no_vert + 1];
                double z1 = v_scale[3 * i_face + 2 * no_vert + 1];

                double x2 = v_scale[3 * i_face + 2];
                double y2 = v_scale[3 * i_face + no_vert + 2];
                double z2 = v_scale[3 * i_face + 2 * no_vert + 2];

                // Check if face vertices collapse to a point or line
                double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                double dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x0, dy = y2 - y0, dz = z2 - z0;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                // Do for each of the 3 vertices
                for (size_t i3 = 0; i3 < 3; ++i3)
                {
                    double x = x0, y = y0, z = z0;
                    if (i3 == 1)
                        x = x1, y = y1, z = z1;
                    else if (i3 == 2)
                        x = x2, y = y2, z = z2;

                    // Add the vertices to the final list
                    size_t i_vert_final = 0; // Face index

                    // Get vertex ID
                    for (size_t i = 0; i < no_final_vert; ++i) // Search exisiting vertex
                    {
                        dx = v_final[i] - x;
                        dy = v_final[i + no_vert] - y;
                        dz = v_final[i + 2 * no_vert] - z;
                        dd = dx * dx + dy * dy + dz * dz;
                        if (dd < eps)
                        {
                            i_vert_final = i;
                            break;
                        }
                    }
                    if (i_vert_final == 0) // Add new vertex to ist
                    {
                        i_vert_final = no_final_vert;
                        v_final[i_vert_final] = x;
                        v_final[i_vert_final + no_vert] = y;
                        v_final[i_vert_final + 2 * no_vert] = z;
                        ++no_final_vert;
                    }

                    // Write face ID
                    f_final[no_final_face + i3 * no_faces] = (unsigned)i_vert_final;
                }

                // Get the colormap index
                double pf = pp[3 * i_face] + pp[3 * i_face + 1] + pp[3 * i_face + 2];
                pf *= 21.0; // 63 * 0.33
                pf = std::round(pf);
                unsigned c = (unsigned)pf;
                c = (c > 63) ? 63 : c;
                i_cmap[no_final_face] = c;

                ++no_final_face;
            }

            // Write vertices to file
            for (size_t i_vert = 0; i_vert < no_final_vert; ++i_vert)
            {
                double x = v_final[i_vert];
                double y = v_final[i_vert + no_vert];
                double z = v_final[i_vert + 2 * no_vert];

                x *= radius;
                y *= radius;
                z *= radius;

                if (element_pos.n_elem != 0)
                {
                    x += element_pos(0, i_array_element);
                    y += element_pos(1, i_array_element);
                    z += element_pos(2, i_array_element);
                }

                outFile << "v " << x << " " << y << " " << z << "\n";
            }

            // Write faces to file, ordered by color
            for (unsigned i_color = 0; i_color < 64; ++i_color)
            {
                bool color_not_used = true;
                for (size_t i_face = 0; i_face < no_final_face; ++i_face)
                {
                    if (i_cmap[i_face] == i_color)
                    {
                        if (color_not_used)
                        {
                            outFile << "usemtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i_color << "\n";
                            color_not_used = false;
                        }
                        outFile << "f " << f_final[i_face] + face_offset << " " << f_final[i_face + no_faces] + face_offset << " " << f_final[i_face + 2 * no_faces] + face_offset << "\n";
                    }
                }
            }
            face_offset += (unsigned)no_final_vert;
        }

        outFile.close();
    }
}

// ARRAYANT METHOD : Interpolation
template <typename dtype>
void quadriga_lib::arrayant<dtype>::interpolate(const arma::Mat<dtype> *azimuth,
                                                const arma::Mat<dtype> *elevation,
                                                arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                                                arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                                                arma::uvec i_element,
                                                const arma::Cube<dtype> *orientation,
                                                const arma::Mat<dtype> *element_pos_i,
                                                arma::Mat<dtype> *dist,
                                                arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc,
                                                arma::Mat<dtype> *gamma) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (azimuth == nullptr)
        throw std::invalid_argument("Input 'azimuth' cannot be NULL");

    if (elevation == nullptr)
        throw std::invalid_argument("Input 'elevation' cannot be NULL");

    if (V_re == nullptr || V_im == nullptr || H_re == nullptr || H_im == nullptr)
        throw std::invalid_argument("Outputs 'V_re', 'V_im', 'H_re', 'H_im' cannot be NULL");

    arma::uword n_out = azimuth->n_rows;
    arma::uword n_ang = azimuth->n_cols;

    if (elevation->n_rows != n_out || elevation->n_cols != n_ang)
        throw std::invalid_argument("Sizes of 'azimuth' and 'elevation' do not match.");

    arma::uword n_elements = e_theta_re.n_slices;
    if (i_element.n_elem == 0)
        i_element = arma::regspace<arma::uvec>(0, n_elements - 1);
    else if (arma::any(i_element >= n_elements))
        throw std::invalid_argument("Element indices 'i_element' cannot exceed the array antenna size.");

    // Process orientation
    n_out = i_element.n_elem;
    arma::Cube<dtype> orientation_empty = arma::Cube<dtype>(3, 1, 1);
    const arma::Cube<dtype> *orientation_local = (orientation == nullptr || orientation->n_elem == 0) ? &orientation_empty : orientation;
    if (orientation != nullptr)
    {
        arma::uword o1 = orientation_local->n_rows,
                    o2 = orientation_local->n_cols,
                    o3 = orientation_local->n_slices;

        if (o1 != 3)
            throw std::invalid_argument("Input 'orientation' must have 3 elements on the first dimension.");
        else if (o2 != 1 && o2 != n_out)
            throw std::invalid_argument("Input 'orientation' must have 1 or 'n_elements' elements on the second dimension.");
        else if (o3 != 1 && o3 != n_ang)
            throw std::invalid_argument("Input 'orientation' must have 1 or 'n_ang' elements on the third dimension.");
    }

    // Process alternative element positions
    arma::Mat<dtype> element_pos_local(3, n_out);
    if (element_pos_i != nullptr && element_pos_i->n_elem != 0)
    {
        if (element_pos_i->n_rows != 3 || element_pos_i->n_cols != n_out)
            throw std::invalid_argument("Alternative element positions 'element_pos_i' must have 3 rows and 'n_out' columns.");

        const dtype *ptrI = element_pos_i->memptr();
        dtype *ptrO = element_pos_local.memptr();
        std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));
    }
    else if (element_pos.n_elem != 0)
    {
        const dtype *ptrI = element_pos.memptr();
        dtype *ptrO = element_pos_local.memptr();
        for (arma::uword i = 0; i < n_out; ++i)
            std::memcpy(&ptrO[3 * i], &ptrI[3 * i_element[i]], 3 * sizeof(dtype));
    }

    // Resize output variables
    if (V_re->n_rows != n_out || V_re->n_cols != n_ang)
        V_re->set_size(n_out, n_ang);
    if (V_im->n_rows != n_out || V_im->n_cols != n_ang)
        V_im->set_size(n_out, n_ang);
    if (H_re->n_rows != n_out || H_re->n_cols != n_ang)
        H_re->set_size(n_out, n_ang);
    if (H_im->n_rows != n_out || H_im->n_cols != n_ang)
        H_im->set_size(n_out, n_ang);

    if (dist != nullptr && (dist->n_rows != n_out || dist->n_cols != n_ang))
        dist->set_size(n_out, n_ang);
    if (azimuth_loc != nullptr && (azimuth_loc->n_rows != n_out || azimuth_loc->n_cols != n_ang))
        azimuth_loc->set_size(n_out, n_ang);
    if (elevation_loc != nullptr && (elevation_loc->n_rows != n_out || elevation_loc->n_cols != n_ang))
        elevation_loc->set_size(n_out, n_ang);
    if (gamma != nullptr && (gamma->n_rows != n_out || gamma->n_cols != n_ang))
        gamma->set_size(n_out, n_ang);

    // Convert element index format
    arma::u32_vec i_element_int(n_out); // 1-based
    {
        unsigned *pi = i_element_int.memptr();
        arma::uword *pu = i_element.memptr();
        for (arma::uword i = 0; i < n_out; ++i)
            pi[i] = (unsigned)pu[i] + 1;
    }

    // Call private library function
    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, azimuth, elevation,
                            &i_element_int, orientation_local, &element_pos_local,
                            V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma);
}

// ARRAYANT : Write to QDANT file
template <typename dtype>
unsigned quadriga_lib::arrayant<dtype>::qdant_write(std::string fn, unsigned id, arma::Mat<unsigned> layout) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    unsigned id_in_file = 0;
    error_message = qd_arrayant_qdant_write(fn, id, &name, &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                            &azimuth_grid, &elevation_grid, &element_pos,
                                            &coupling_re, &coupling_im, &center_frequency, &layout,
                                            &id_in_file);

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return id_in_file;
}

// Remove zeros from the pattern
template <typename dtype>
void quadriga_lib::arrayant<dtype>::remove_zeros(quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (output == nullptr && read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }

    unsigned long long n_el = e_theta_re.n_rows;
    unsigned long long n_az = e_theta_re.n_cols;
    unsigned long long n_slices = e_theta_re.n_slices;

    // Calculate the power pattern
    arma::Mat<dtype> pow(n_el, n_az);
    dtype *pp = pow.memptr();
    dtype limit = dtype(1.0e-12), pi_half = dtype(arma::datum::pi / 2.0);

    for (auto is = 0ULL; is < n_slices; ++is)
    {
        dtype *pa = e_theta_re.slice_memptr(is), *pb = e_theta_im.slice_memptr(is),
              *pc = e_phi_re.slice_memptr(is), *pd = e_phi_im.slice_memptr(is);
        for (auto j = 0ULL; j < n_el * n_az; ++j)
            pp[j] += pa[j] * pa[j] + pb[j] * pb[j] + pc[j] * pc[j] + pd[j] * pd[j];
    }

    arma::Row<dtype> az_sum = arma::sum(pow, 0);
    arma::Col<dtype> el_sum = arma::sum(pow, 1);
    pow.reset();

    // Determine which parts of the pattern should be kept
    arma::uvec keep_az(n_az), keep_el(n_el);
    unsigned long long *keep_az_ptr = keep_az.memptr();
    unsigned long long *keep_el_ptr = keep_el.memptr();

    pp = az_sum.memptr();
    dtype *grid_ptr = azimuth_grid.memptr();
    for (auto ia = 0ULL; ia < n_az; ++ia)
    {
        dtype p = ia == 0 ? *grid_ptr + grid_ptr[n_az - 1] : grid_ptr[ia] - grid_ptr[ia - 1];
        dtype n = ia == n_az - 1 ? *grid_ptr + grid_ptr[ia] : grid_ptr[ia + 1] - grid_ptr[ia];

        unsigned long long ip = ia == 0ULL ? n_az - 1ULL : ia - 1ULL;
        unsigned long long in = ia == n_az - 1ULL ? 0ULL : ia + 1ULL;

        if (pp[ia] > limit)
            keep_az_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_az_ptr[ip] : 1,
            keep_az_ptr[ia] = 1,
            keep_az_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_az_ptr[in] : 1;
    }
    az_sum.reset();

    pp = el_sum.memptr(), grid_ptr = elevation_grid.memptr();

    for (auto ie = 0ULL; ie < n_el; ++ie)
    {
        dtype p = ie == 0 ? *grid_ptr + pi_half : grid_ptr[ie] - grid_ptr[ie - 1];
        dtype n = ie == n_el - 1 ? pi_half - grid_ptr[ie] : grid_ptr[ie + 1] - grid_ptr[ie];

        unsigned long long ip = ie == 0ULL ? 0ULL : ie - 1ULL;
        unsigned long long in = ie == n_el - 1ULL ? n_el - 1ULL : ie + 1ULL;

        if (pp[ie] > limit)
            keep_el_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_el_ptr[ip] : 1ULL,
            keep_el_ptr[ie] = 1,
            keep_el_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_el_ptr[in] : 1ULL;
    }
    el_sum.reset();

    // Copy the relevant pattern data
    unsigned long long n_az_new = arma::sum(keep_az), n_el_new = arma::sum(keep_el);

    for (int p = 0; p < 4; ++p)
    {
        dtype *data = nullptr, *ptrO = nullptr, *ptrI = nullptr;

        if (output == nullptr)
            data = new dtype[n_el_new * n_az_new * n_slices];
        else if (p == 0)
        {
            if (output->e_theta_re.n_rows != n_el_new || output->e_theta_re.n_cols != n_az_new || output->e_theta_re.n_slices != n_slices)
                output->e_theta_re.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_theta_re.memptr();
        }
        else if (p == 1)
        {
            if (output->e_theta_im.n_rows != n_el_new || output->e_theta_im.n_cols != n_az_new || output->e_theta_im.n_slices != n_slices)
                output->e_theta_im.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_theta_im.memptr();
        }
        else if (p == 2)
        {
            if (output->e_phi_re.n_rows != n_el_new || output->e_phi_re.n_cols != n_az_new || output->e_phi_re.n_slices != n_slices)
                output->e_phi_re.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_phi_re.memptr();
        }
        else if (p == 3)
        {
            if (output->e_phi_im.n_rows != n_el_new || output->e_phi_im.n_cols != n_az_new || output->e_phi_im.n_slices != n_slices)
                output->e_phi_im.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_phi_im.memptr();
        }

        ptrO = data;
        for (auto is = 0ULL; is < n_slices; ++is)
        {
            if (p == 0)
                ptrI = e_theta_re.slice_memptr(is);
            else if (p == 1)
                ptrI = e_theta_im.slice_memptr(is);
            else if (p == 2)
                ptrI = e_phi_re.slice_memptr(is);
            else if (p == 3)
                ptrI = e_phi_im.slice_memptr(is);

            for (auto ia = 0ULL; ia < n_az; ++ia)
            {
                bool keep_az = keep_az_ptr[ia] == 1;
                for (auto ie = 0ULL; ie < n_el; ++ie)
                    if (keep_az && keep_el_ptr[ie] == 1ULL)
                        *ptrO++ = ptrI[ia * n_el + ie];
            }
        }

        if (output == nullptr)
        {
            if (p == 0)
                e_theta_re.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_theta_re.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 1)
                e_theta_im.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_theta_im.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 2)
                e_phi_re.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_phi_re.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 3)
                e_phi_im.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_phi_im.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));

            delete[] data;
        }
    }

    // Copy the new azimuth grid
    dtype *azimuth_grid_new = new dtype[n_az_new];
    pp = azimuth_grid_new;
    for (auto ia = 0ULL; ia < n_az; ++ia)
        if (keep_az_ptr[ia] == 1)
            *pp++ = azimuth_grid[ia];
    if (output == nullptr)
        azimuth_grid.set_size(n_az_new), pp = azimuth_grid.memptr();
    else if (output->azimuth_grid.n_elem != n_az_new)
        output->azimuth_grid.set_size(n_az_new), pp = output->azimuth_grid.memptr();
    else
        pp = output->azimuth_grid.memptr();
    std::memcpy(pp, azimuth_grid_new, n_az_new * sizeof(dtype));
    delete[] azimuth_grid_new;

    // Copy the new elevation grid
    dtype *elevation_grid_new = new dtype[n_el_new];
    pp = elevation_grid_new;
    for (auto ie = 0ULL; ie < n_el; ++ie)
        if (keep_el_ptr[ie] == 1)
            *pp++ = elevation_grid[ie];
    if (output == nullptr)
        elevation_grid.set_size(n_el_new), pp = elevation_grid.memptr();
    else if (output->elevation_grid.n_elem != n_el_new)
        output->elevation_grid.set_size(n_el_new), pp = output->elevation_grid.memptr();
    else
        pp = output->elevation_grid.memptr();
    std::memcpy(pp, elevation_grid_new, n_el_new * sizeof(dtype));
    delete[] elevation_grid_new;

    // Copy the unchanged data
    if (output != nullptr)
    {
        if (!element_pos.empty())
        {
            if (output->element_pos.n_rows != 3 || output->element_pos.n_cols != n_slices)
                output->element_pos.set_size(3, n_slices);
            std::memcpy(output->element_pos.memptr(), element_pos.memptr(), 3 * n_slices * sizeof(dtype));
        }

        if (!coupling_re.empty())
        {
            if (output->coupling_re.n_rows != n_slices || output->coupling_re.n_cols != coupling_re.n_cols)
                output->coupling_re.set_size(n_slices, coupling_re.n_cols);
            std::memcpy(output->coupling_re.memptr(), coupling_re.memptr(), n_slices * coupling_re.n_cols * sizeof(dtype));
        }

        if (!coupling_im.empty())
        {
            if (output->coupling_im.n_rows != n_slices || output->coupling_im.n_cols != coupling_im.n_cols)
                output->coupling_im.set_size(n_slices, coupling_im.n_cols);
            std::memcpy(output->coupling_im.memptr(), coupling_im.memptr(), n_slices * coupling_im.n_cols * sizeof(dtype));
        }

        output->center_frequency = center_frequency;
        output->validate();
    }
}

// ARRAYANT METHOD : Change the size of an arrayant, without explicitly preserving data
template <typename dtype>
void quadriga_lib::arrayant<dtype>::reset()
{
    if (read_only)
    {
        std::string error_message = "Cannot change size of read-only array antenna object.";
        throw std::invalid_argument(error_message.c_str());
    }

    if (azimuth_grid.n_elem != 0)
        azimuth_grid.reset();

    if (elevation_grid.n_elem != 0)
        elevation_grid.reset();

    if (e_theta_re.n_elem != 0)
        e_theta_re.reset();

    if (e_theta_im.n_elem != 0)
        e_theta_im.reset();

    if (e_phi_re.n_elem != 0)
        e_phi_re.reset();

    if (e_phi_im.n_elem != 0)
        e_phi_im.reset();

    if (element_pos.n_elem != 0)
        element_pos.reset();

    if (coupling_re.n_elem != 0)
        coupling_re.reset();

    if (coupling_im.n_elem != 0)
        coupling_im.reset();

    center_frequency = dtype(299792458.0);
}

// Rotating antenna patterns (adjusts sampling grid if needed, e.g. for parabolic antennas)
template <typename dtype>
void quadriga_lib::arrayant<dtype>::rotate_pattern(dtype x_deg, dtype y_deg, dtype z_deg,
                                                   unsigned usage, unsigned element,
                                                   quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    bool use_all_elements = false;
    if (usage > 3)
        error_message = "Input parameter 'usage' must be 0, 1, 2 or 3.";
    if (element == unsigned(-1) || e_theta_re.n_slices == 1)
        use_all_elements = true;
    else if (element >= e_theta_re.n_slices)
        error_message = "Input parameter 'element' out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (output == nullptr && read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set element indices
    arma::Col<unsigned> i_element(1);
    if (use_all_elements)
        i_element = arma::regspace<arma::Col<unsigned>>(1, unsigned(e_theta_re.n_slices));
    else
        i_element.at(0) = element + 1;

    // Extract the element positions
    arma::Mat<dtype> element_pos_update;
    if (element_pos.n_elem == 0)
        element_pos_update.zeros(3, e_theta_re.n_slices);
    else
        element_pos_update = element_pos;

    unsigned long long n_el = e_theta_re.n_rows;
    unsigned long long n_az = e_theta_re.n_cols;
    unsigned long long n_out = i_element.n_elem;
    unsigned long long n_ang = n_el * n_az;

    dtype tau = dtype(arma::datum::tau), pi_half = dtype(arma::datum::pi / 2.0), pi = dtype(arma::datum::pi),
          deg2rad = dtype(arma::datum::pi / 180.0), limit = dtype(1.0e-6);

    // Calculate the coverage range for the angle sampling grid
    dtype zero = dtype(0.0), az_step_min = dtype(1e38), az_step_max = zero, el_step_min = az_step_min,
          el_step_max = zero, step = zero;
    dtype *ptr = azimuth_grid.memptr();
    for (auto i = 0ULL; i < n_az; ++i)
    {
        step = i == 0 ? *ptr - ptr[n_az - 1] + tau : ptr[i] - ptr[i - 1];
        az_step_min = step < az_step_min && step > limit ? step : az_step_min;
        az_step_max = step > az_step_max ? step : az_step_max;
    }
    ptr = elevation_grid.memptr();
    for (auto i = 0ULL; i <= n_el; ++i)
    {
        if (i == 0)
            step = *ptr + pi_half;
        else if (i == n_el)
            step = pi_half - ptr[i - 1];
        else
            step = ptr[i] - ptr[i - 1];

        el_step_min = step < el_step_min && step > limit ? step : el_step_min;
        el_step_max = step > el_step_max ? step : el_step_max;
    }
    step = pi_half - ptr[n_el - 1];
    el_step_min = step < el_step_min && step > limit ? step : el_step_min;
    el_step_max = step > el_step_max ? step : el_step_max;
    step = el_step_min < az_step_min ? el_step_min : az_step_min;

    bool update_grid = (usage == 0 || usage == 1) && (az_step_max - az_step_min > limit ||
                                                      el_step_max - el_step_min > limit ||
                                                      az_step_min - el_step_min > limit ||
                                                      n_az == 1 || n_el == 1)
                           ? true
                           : false;

    // Obtain interpolation angles
    arma::Col<dtype> azimuth_grid_update, elevation_grid_update;
    if (update_grid)
    {
        if (output == nullptr && !use_all_elements)
        {
            error_message = "Update of sampling grid cannot be done for single elements of an array antenna!";
            throw std::invalid_argument(error_message.c_str());
        }

        if (step > dtype(0.0017)) // >= 0.1 degree --> Use entire sphere
        {
            unsigned long long N = (unsigned long long)std::round(pi / step);
            N = N < 4ULL ? 4ULL : N;

            azimuth_grid_update = arma::linspace<arma::Col<dtype>>(-pi, pi, 2ULL * N + 1ULL);

            if (std::abs(x_deg) < limit && std::abs(y_deg) < limit)
                elevation_grid_update = elevation_grid;
            else
                elevation_grid_update = arma::linspace<arma::Col<dtype>>(-pi_half + step, pi_half - step, N - 1ULL);
        }
        else // step < 0.1 degree --> Use subsampled sphere
        {
            auto ant = quadriga_lib::generate_arrayant_omni<dtype>();
            ant.e_theta_re.zeros();
            arma::Col<unsigned> has_az(361), has_el(181);

            // Select azimuth angles
            dtype *p = azimuth_grid.memptr(), *q = ant.azimuth_grid.memptr();
            unsigned *ra = has_az.memptr();
            for (unsigned i = 0; i < 360; ++i)
            {
                while (p != azimuth_grid.end() && *p <= q[i + 1])
                {
                    ra[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : ra[i];
                    p++;
                }
                ra[i + 1] = ra[i];
                if (p == azimuth_grid.end())
                    break;
            }

            // Select elevation angles
            p = elevation_grid.memptr(), q = ant.elevation_grid.memptr();
            unsigned *re = has_el.memptr();
            for (unsigned i = 0; i < 180; ++i)
            {
                while (p != elevation_grid.end() && *p <= q[i + 1])
                {
                    re[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : re[i];
                    p++;
                }
                re[i + 1] = re[i];
                if (p == elevation_grid.end())
                    break;
            }

            // Set interpolation target
            p = ant.e_theta_re.memptr();
            for (auto i = 0ULL; i < ant.e_theta_re.n_elem; ++i)
                p[i] = ra[i / 181] == 1 && re[i % 181] == 1 ? 1.0 : 0.0;

            // Find target area for interpolation
            ant.rotate_pattern(x_deg, y_deg, z_deg, 1);
            ant.remove_zeros();

            // Subdivide target grid
            dtype sdiv = std::fmod(deg2rad, step) < 0.1 * step ? std::floor(deg2rad / step) : std::ceil(deg2rad / step);
            dtype stp = deg2rad / sdiv;
            unsigned long long ndiv = (unsigned long long)sdiv;

            bool wrap = ant.azimuth_grid.at(ant.azimuth_grid.n_elem - 1) + stp > pi;
            unsigned long long N = wrap ? ndiv * (ant.azimuth_grid.n_elem - 1) + 1 : ndiv * ant.azimuth_grid.n_elem;
            azimuth_grid_update.set_size(N);
            p = azimuth_grid_update.memptr(), q = ant.azimuth_grid.memptr();
            for (auto i = 0ULL; i < ant.azimuth_grid.n_elem; ++i)
                if (wrap && i == ant.azimuth_grid.n_elem - 1ULL)
                    *p = q[i];
                else
                    for (auto j = 0ULL; j < ndiv; ++j)
                        *p++ = q[i] + dtype(j) * stp;

            wrap = ant.elevation_grid.at(ant.elevation_grid.n_elem - 1) + stp > pi_half;
            N = wrap ? ndiv * (ant.elevation_grid.n_elem - 1) + 1 : ndiv * ant.elevation_grid.n_elem;
            elevation_grid_update.set_size(N);
            p = elevation_grid_update.memptr(), q = ant.elevation_grid.memptr();
            for (auto i = 0ULL; i < ant.elevation_grid.n_elem; ++i)
                if (wrap && i == ant.elevation_grid.n_elem - 1)
                    *p = q[i];
                else
                    for (auto j = 0ULL; j < ndiv; ++j)
                        *p++ = q[i] + dtype(j) * stp;
        }

        n_az = azimuth_grid_update.n_elem;
        n_el = elevation_grid_update.n_elem;
        n_ang = n_az * n_el;
    }
    else
    {
        azimuth_grid_update = azimuth_grid;
        elevation_grid_update = elevation_grid;
    }

    // Initiate output variables
    arma::Cube<dtype> orientation(3, 1, 1);
    arma::Mat<dtype> V_re(n_out, n_ang, arma::fill::none), V_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> H_re(n_out, n_ang, arma::fill::none), H_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> azimuth_loc, elevation_loc, gamma, EMPTY;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr(),
          *p_phi = azimuth_grid_update.memptr(), *p_theta = elevation_grid_update.memptr();

    for (auto ia = 0ULL; ia < n_az; ++ia)
        for (auto ie = 0ULL; ie < n_el; ++ie)
            *p_azimuth++ = p_phi[ia], *p_elevation++ = p_theta[ie];

    // Set antenna orientation
    orientation.at(0) = x_deg * deg2rad;
    orientation.at(1) = -y_deg * deg2rad;
    orientation.at(2) = z_deg * deg2rad;

    // Calculate rotation matrix (double precision)
    arma::cube R = quadriga_lib::calc_rotation_matrix(orientation, true);

    if (usage == 1)
        azimuth_loc.set_size(n_out, n_ang), elevation_loc.set_size(n_out, n_ang);
    else if (usage == 2)
        gamma.set_size(n_out, n_ang);

    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos_update,
                            &V_re, &V_im, &H_re, &H_im, &EMPTY, &azimuth_loc, &elevation_loc, &gamma);
    azimuth.reset(), elevation.reset();

    orientation.zeros();
    if (usage == 1) // Only return interpolated pattern (ignore polarization)
        qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                &azimuth_grid, &elevation_grid, &azimuth_loc, &elevation_loc,
                                &i_element, &orientation, &element_pos_update,
                                &V_re, &V_im, &H_re, &H_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);
    azimuth_loc.reset(), elevation_loc.reset();

    // Adjust size of output, if needed
    if (output != nullptr)
        output->set_size(n_el, n_az, n_out, n_out);

    if (usage == 2) // Only adjust the polarization
    {
        gamma = gamma.t();
        for (auto i = 0ULL; i < n_out; ++i)
        {
            dtype *p_gamma = gamma.colptr(i);
            dtype *p_theta_re = use_all_elements ? e_theta_re.slice_memptr(i) : e_theta_re.slice_memptr(element);
            dtype *p_theta_im = use_all_elements ? e_theta_im.slice_memptr(i) : e_theta_im.slice_memptr(element);
            dtype *p_phi_re = use_all_elements ? e_phi_re.slice_memptr(i) : e_phi_re.slice_memptr(element);
            dtype *p_phi_im = use_all_elements ? e_phi_im.slice_memptr(i) : e_phi_im.slice_memptr(element);

            dtype *q_theta_re = p_theta_re, *q_theta_im = p_theta_im, *q_phi_re = p_phi_re, *q_phi_im = p_phi_im;
            if (output != nullptr)
                q_theta_re = use_all_elements ? output->e_theta_re.slice_memptr(i) : output->e_theta_re.memptr(),
                q_theta_im = use_all_elements ? output->e_theta_im.slice_memptr(i) : output->e_theta_im.memptr(),
                q_phi_re = use_all_elements ? output->e_phi_re.slice_memptr(i) : output->e_phi_re.memptr(),
                q_phi_im = use_all_elements ? output->e_phi_im.slice_memptr(i) : output->e_phi_im.memptr();

            for (auto j = 0ULL; j < n_ang; ++j)
            {
                dtype sin_gamma = std::sin(p_gamma[j]), cos_gamma = std::cos(p_gamma[j]);
                dtype tmp = sin_gamma * p_theta_re[j];
                q_theta_re[j] = cos_gamma * p_theta_re[j] - sin_gamma * p_phi_re[j];
                q_phi_re[j] = tmp + cos_gamma * p_phi_re[j];
                tmp = sin_gamma * p_theta_im[j];
                q_theta_im[j] = cos_gamma * p_theta_im[j] - sin_gamma * p_phi_im[j];
                q_phi_im[j] = tmp + cos_gamma * p_phi_im[j];
            }
        }
        gamma.reset();

        // Copy unchanged data
        if (output != nullptr)
        {
            dtype *ptrI = use_all_elements ? element_pos_update.memptr() : element_pos_update.colptr(element);
            dtype *ptrO = output->element_pos.memptr();
            std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));

            ptrI = azimuth_grid_update.memptr();
            ptrO = output->azimuth_grid.memptr();
            std::memcpy(ptrO, ptrI, n_az * sizeof(dtype));

            ptrI = elevation_grid_update.memptr();
            ptrO = output->elevation_grid.memptr();
            std::memcpy(ptrO, ptrI, n_el * sizeof(dtype));
        }
    }
    else // Usage 0, 1 or 3
    {
        gamma.reset();
        if (n_out > 1) // Transpose the interpolated data
            V_re = V_re.t(), V_im = V_im.t(), H_re = H_re.t(), H_im = H_im.t();

        if (output == nullptr && update_grid) // Set new array size
            set_size(n_el, n_az, n_out, n_out);

        dtype *ptrI, *ptrO;

        ptrI = V_re.memptr();
        ptrO = use_all_elements ? e_theta_re.memptr() : e_theta_re.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_theta_re.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        V_re.reset();

        ptrI = V_im.memptr();
        ptrO = use_all_elements ? e_theta_im.memptr() : e_theta_im.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_theta_im.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        V_im.reset();

        ptrI = H_re.memptr();
        ptrO = use_all_elements ? e_phi_re.memptr() : e_phi_re.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_phi_re.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        H_re.reset();

        ptrI = H_im.memptr();
        ptrO = use_all_elements ? e_phi_im.memptr() : e_phi_im.slice_memptr(element);
        ptrO = output == nullptr ? ptrO : output->e_phi_im.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        H_im.reset();

        ptrI = azimuth_grid_update.memptr();
        ptrO = output == nullptr ? azimuth_grid.memptr() : output->azimuth_grid.memptr();
        std::memcpy(ptrO, ptrI, n_az * sizeof(dtype));

        ptrI = elevation_grid_update.memptr();
        ptrO = output == nullptr ? elevation_grid.memptr() : output->elevation_grid.memptr();
        std::memcpy(ptrO, ptrI, n_el * sizeof(dtype));

        ptrI = element_pos_update.memptr();
        ptrO = use_all_elements ? element_pos.memptr() : element_pos.colptr(element);
        ptrO = output == nullptr ? ptrO : output->element_pos.memptr();
        double *R_ptr = R.memptr();
        for (auto i = 0ULL; i < i_element.n_elem; ++i)
        {
            unsigned j = 3 * (i_element.at(i) - 1);
            unsigned k = use_all_elements ? j : 0;
            dtype a = dtype(R_ptr[0]) * ptrI[j] + dtype(R_ptr[3]) * ptrI[j + 1] + dtype(R_ptr[6]) * ptrI[j + 2];
            dtype b = dtype(R_ptr[1]) * ptrI[j] + dtype(R_ptr[4]) * ptrI[j + 1] + dtype(R_ptr[7]) * ptrI[j + 2];
            dtype c = dtype(R_ptr[2]) * ptrI[j] + dtype(R_ptr[5]) * ptrI[j + 1] + dtype(R_ptr[8]) * ptrI[j + 2];
            ptrO[k] = a, ptrO[k + 1] = b, ptrO[k + 2] = c;
        }

        if (output == nullptr && update_grid)
            remove_zeros();
        else if (output != nullptr && update_grid)
            output->remove_zeros();
    }
}

// ARRAYANT METHOD : Change the size of an arrayant, without explicitly preserving data
template <typename dtype>
void quadriga_lib::arrayant<dtype>::set_size(arma::uword n_elevation, arma::uword n_azimuth,
                                             arma::uword n_elements, arma::uword n_ports)
{
    if (read_only)
    {
        std::string error_message = "Cannot change size of read-only array antenna object.";
        throw std::invalid_argument(error_message.c_str());
    }

    if (azimuth_grid.n_elem != n_azimuth)
        azimuth_grid.set_size(n_azimuth);

    if (elevation_grid.n_elem != n_elevation)
        elevation_grid.set_size(n_elevation);

    if (e_theta_re.n_rows != n_elevation || e_theta_re.n_cols != n_azimuth || e_theta_re.n_slices != n_elements)
        e_theta_re.set_size(n_elevation, n_azimuth, n_elements);

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        e_theta_im.set_size(n_elevation, n_azimuth, n_elements);

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        e_phi_re.set_size(n_elevation, n_azimuth, n_elements);

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        e_phi_im.set_size(n_elevation, n_azimuth, n_elements);

    if (element_pos.n_rows != 3 || element_pos.n_cols != n_elements)
        element_pos.zeros(3, n_elements);

    if (coupling_re.n_rows != n_elements || coupling_re.n_cols != n_ports)
        coupling_re.eye(n_elements, n_ports);

    if (coupling_im.n_rows != n_elements || coupling_im.n_cols != n_ports)
        coupling_im.zeros(n_elements, n_ports);
}

// ARRAYANT METHOD : Validates correctness of the member functions
template <typename dtype>
std::string quadriga_lib::arrayant<dtype>::is_valid(bool quick_check) const
{

    // Assuming that the data has been validated before, we can quickly check if the
    // data pointers were updated. If not, we can assume that the data is still valid.
    if (quick_check)
        quick_check = check_ptr[0] != nullptr && check_ptr[1] != nullptr && check_ptr[2] != nullptr &&
                      check_ptr[3] != nullptr && check_ptr[4] != nullptr && check_ptr[5] != nullptr &&
                      check_ptr[0] == e_theta_re.memptr() &&
                      check_ptr[1] == e_theta_im.memptr() &&
                      check_ptr[2] == e_phi_re.memptr() &&
                      check_ptr[3] == e_phi_im.memptr() &&
                      check_ptr[4] == azimuth_grid.memptr() &&
                      check_ptr[5] == elevation_grid.memptr() &&
                      check_ptr[6] == element_pos.memptr() &&
                      check_ptr[7] == coupling_re.memptr() &&
                      check_ptr[8] == coupling_im.memptr();
    if (quick_check)
        return std::string("");

    // Perform a deep check
    if (e_theta_re.n_elem == 0 || e_theta_im.n_elem == 0 || e_phi_re.n_elem == 0 || e_phi_im.n_elem == 0 || azimuth_grid.n_elem == 0 || elevation_grid.n_elem == 0)
        return std::string("Missing data for any of: e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid");

    unsigned long long n_elevation = e_theta_re.n_rows;
    unsigned long long n_azimuth = e_theta_re.n_cols;
    unsigned long long n_elements = e_theta_re.n_slices;

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        return std::string("Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.");

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        return std::string("Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.");

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        return std::string("Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.");

    if (azimuth_grid.n_elem != n_azimuth)
        return std::string("Number of elements in 'azimuth_grid' does not match number of columns in pattern data.");

    if (elevation_grid.n_elem != n_elevation)
        return std::string("Number of elements in 'elevation_grid' does not match number of rows in pattern data.");

    bool error_code = false;
    for (const dtype *val = azimuth_grid.begin(); val < azimuth_grid.end(); ++val)
        error_code = *val < dtype(-3.1415930) || *val > dtype(3.1415930) ? true : error_code;
    if (error_code)
        return std::string("Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).");

    for (const dtype *val = elevation_grid.begin(); val < elevation_grid.end(); ++val)
        error_code = *val < dtype(-1.5707965) || *val > dtype(1.5707965) ? true : error_code;
    if (error_code)
        return std::string("Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).");

    if (!azimuth_grid.is_sorted())
        return std::string("Values of 'azimuth_grid' must be sorted in ascending order.");

    if (!elevation_grid.is_sorted())
        return std::string("Values of 'elevation_grid' must be sorted in ascending order.");

    if (!element_pos.empty() && (element_pos.n_rows != 3 || element_pos.n_cols != n_elements))
        return std::string("Size of 'element_pos' must be either empty or match [3, n_elements]");

    if (!coupling_re.empty() && coupling_re.n_rows != n_elements)
        return std::string("'Coupling' must be a matrix with rows equal to number of elements");

    if (coupling_re.empty() && !coupling_im.empty())
        return std::string("Imaginary part of coupling matrix (phase component) defined without real part (absolute component)");

    if (!coupling_im.empty() && (coupling_im.n_rows != n_elements || coupling_im.n_cols != coupling_re.n_cols))
        return std::string("'coupling_im' must be empty or its size must match 'coupling_re'");

    return std::string("");
}

// ARRAYANT METHOD : Validates correctness of the member functions and initializes element positions and coupling matrix
template <typename dtype>
std::string quadriga_lib::arrayant<dtype>::validate()
{
    std::string error_message = is_valid(false); // Deep check
    if (error_message.length() != 0)
        return error_message;

    unsigned long long n_elements = e_theta_re.n_slices;
    unsigned long long n_prt = coupling_re.empty() ? n_elements : coupling_re.n_cols;

    if (element_pos.empty())
        element_pos.zeros(3, n_elements);

    if (coupling_re.empty())
        coupling_re.eye(n_elements, n_elements), coupling_im.zeros(n_elements, n_elements);

    if (coupling_im.empty())
        coupling_im.zeros(n_elements, n_prt);

    // Set the data pointers for the quick check.
    check_ptr[0] = e_theta_re.memptr();
    check_ptr[1] = e_theta_im.memptr();
    check_ptr[2] = e_phi_re.memptr();
    check_ptr[3] = e_phi_im.memptr();
    check_ptr[4] = azimuth_grid.memptr();
    check_ptr[5] = elevation_grid.memptr();
    check_ptr[6] = element_pos.memptr();
    check_ptr[7] = coupling_re.memptr();
    check_ptr[8] = coupling_im.memptr();

    return std::string("");
}

// Instantiate templates
template class quadriga_lib::arrayant<float>;
template class quadriga_lib::arrayant<double>;

// Read array antenna object and layout from QDANT file
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout)
{
    quadriga_lib::arrayant<dtype> ant;
    std::string error_message;

    if (layout == nullptr)
    {
        arma::Mat<unsigned> tmp_layout;
        error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                               &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                               &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                               &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                               &tmp_layout);
        tmp_layout.reset();
    }
    else
        error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                               &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                               &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                               &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                               layout);

    // Throw parsing errors
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Throw validation errors
    error_message = ant.validate();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout);
template quadriga_lib::arrayant<double> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout);

// Generate : Isotropic radiator, vertical polarization, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "omni";
    ant.e_theta_re.ones(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_omni();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_omni();

// Generate : Cross-polarized isotropic radiator, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "xpol";
    ant.e_theta_re.ones(181, 361, 2);
    ant.e_theta_im.zeros(181, 361, 2);
    ant.e_phi_re.ones(181, 361, 2);
    ant.e_phi_im.zeros(181, 361, 2);
    ant.e_phi_re.slice(0).zeros();
    ant.e_theta_re.slice(1).zeros();
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.element_pos.zeros(3, 2);
    ant.coupling_re.eye(2, 2);
    ant.coupling_im.zeros(2, 2);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_xpol();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_xpol();

// Generate : Short dipole radiating with vertical polarization, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.e_theta_re.zeros(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.e_theta_re.slice(0) = arma::repmat(ant.elevation_grid, 1, 361);
    ant.e_theta_re = arma::cos(dtype(0.999999) * ant.e_theta_re) * dtype(std::sqrt(1.499961));
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_dipole();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_dipole();

// Generate : Half-wave dipole radiating with vertical polarization
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    constexpr dtype scale = dtype(0.999999);

    ant.name = "half-wave-dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.e_theta_re.zeros(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.e_theta_re.slice(0) = arma::repmat(ant.elevation_grid, 1, 361);
    ant.e_theta_re = arma::cos(pih * arma::sin(scale * ant.e_theta_re)) / arma::cos(scale * ant.e_theta_re);
    ant.e_theta_re = ant.e_theta_re * dtype(1.280968208215292);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_half_wave_dipole();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_half_wave_dipole();

// Generate : An antenna with a custom gain in elevation and azimuth
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(dtype az_3dB, dtype el_3db, dtype rear_gain_lin)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0), half = dtype(0.5),
                    limit = dtype(1e-7), step = dtype(-0.382), limit_inf = dtype(1e38);
    const dtype pi = dtype(arma::datum::pi), deg2rad = dtype(arma::datum::pi / 360.0);

    quadriga_lib::arrayant<dtype> ant;

    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi * half, pi * half, 181);
    arma::Col<dtype> phi_sq = ant.azimuth_grid % ant.azimuth_grid;
    arma::Col<dtype> cos_theta = arma::cos(ant.elevation_grid);
    cos_theta.at(0) = zero, cos_theta.at(180) = zero;
    arma::Col<dtype> az_3dB_rad(1), el_3db_rad(1);
    az_3dB_rad.at(0) = az_3dB * deg2rad;
    el_3db_rad.at(0) = el_3db * deg2rad;

    // Calculate azimuth pattern cut
    dtype a = one, d = half, x = limit_inf, delta = limit_inf;
    arma::Col<dtype> xn(1), C(361), D(181);
    for (unsigned lp = 0; lp < 5000; ++lp)
    {
        dtype an = lp == 0 ? a : a + d;
        delta = lp == 0 ? limit_inf : std::abs(a - an);
        C = rear_gain_lin + (one - rear_gain_lin) * arma::exp(-an * phi_sq);
        quadriga_lib::interp(&C, &ant.azimuth_grid, &az_3dB_rad, &xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    C = arma::exp(-a * phi_sq);

    // Calculate elevation pattern cut
    a = one, d = half, x = limit_inf, delta = limit_inf;
    for (unsigned lp = 0; lp < 5000; ++lp)
    {
        dtype an = lp == 0 ? a : a + d;
        delta = lp == 0 ? limit_inf : std::abs(a - an);
        D = arma::pow(cos_theta, an);
        quadriga_lib::interp(&D, &ant.elevation_grid, &el_3db_rad, &xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    D = arma::pow(cos_theta, a);

    // Combined pattern
    ant.e_theta_re.zeros(181, 361, 1);
    dtype *ptr = ant.e_theta_re.memptr();
    for (dtype *col = C.begin(); col != C.end(); ++col)
        for (dtype *row = D.begin(); row != D.end(); ++row)
            *ptr++ = std::sqrt(rear_gain_lin + (one - rear_gain_lin) * *row * *col);

    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.name = "custom";

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    // Normalize to Gain
    dtype directivity = ant.calc_directivity_dBi(0);
    directivity = dtype(std::pow(10.0, 0.1 * double(directivity)));
    dtype p_max = ant.e_theta_re.max();
    p_max *= p_max;
    ant.e_theta_re *= std::sqrt(directivity / p_max);

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_custom(float az_3dB, float el_3db, float rear_gain_lin);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_custom(double az_3dB, double el_3db, double rear_gain_lin);

// Generate : Antenna model for the 3GPP-NR channel model
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(unsigned long long M, unsigned long long N, dtype center_freq,
                                                                   unsigned pol, dtype tilt, dtype spacing,
                                                                   unsigned long long Mg, unsigned long long Ng, dtype dgv, dtype dgh,
                                                                   const arrayant<dtype> *pattern)
{
    double pi = arma::datum::pi, rad2deg = 180.0 / pi, deg2rad = pi / 180.0;
    double wavelength = 299792458.0 / double(center_freq);
    constexpr dtype zero = dtype(0.0);

    quadriga_lib::arrayant<dtype> ant = pattern == nullptr ? quadriga_lib::generate_arrayant_omni<dtype>() : pattern->copy();

    if (pattern != nullptr)
    {
        std::string error_message = ant.validate(); // Deep check
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());
    }

    ant.center_frequency = center_freq;
    unsigned long long n_az = ant.n_azimuth(), n_el = ant.n_elevation();

    if (pattern == nullptr) // Generate 3GPP default radiation pattern
    {
        // Single antenna element vertical radiation pattern cut in dB
        arma::Col<dtype> Y = ant.elevation_grid;
        for (dtype *py = Y.begin(); py < Y.end(); ++py)
        {
            double y = double(*py) * rad2deg / 65.0;
            y = 12.0 * y * y;
            *py = y > 30.0 ? dtype(30.0) : dtype(y);
        }

        // Full pattern (normalized to 8 dBi gain using factor 2.51..)
        dtype *ptr = ant.e_theta_re.memptr(), *py = Y.memptr(), *px = ant.azimuth_grid.memptr();
        for (auto ia = 0ULL; ia < n_az; ++ia)
        {
            double x = double(*px++) * rad2deg / 65.0;
            x = 12.0 * x * x;
            x = x > 30.0 ? 30.0 : x;

            for (auto ie = 0ULL; ie < n_el; ++ie)
            {
                double z = double(py[ie]) + x;
                z = z > 30.0 ? -30.0 : -z;
                *ptr++ = dtype(2.511886431509580 * std::sqrt(std::pow(10.0, 0.1 * z))); // 8dBi gain
            }
        }
        Y.reset();

        // Adjust polarization
        if (pol == 2 || pol == 5)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(90.0, zero, zero, 2, 1);
        }
        else if (pol == 3 || pol == 6)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(dtype(45.0), zero, zero, 2, 0);
            ant.rotate_pattern(dtype(-45.0), zero, zero, 2, 1);
        }
    }

    // Duplicate the existing elements in z-direction (vertical stacking)
    unsigned long long n_elements = ant.n_elements();
    if (M > 1ULL)
        for (unsigned long long source = n_elements; source > 0ULL; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = M * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

    // Calculate the element z-position
    arma::Col<dtype> z_position(M);
    if (M > 1ULL)
    {
        z_position = arma::linspace<arma::Col<dtype>>(zero, dtype(M - 1ULL) * spacing * dtype(wavelength), M);
        z_position = z_position - arma::mean(z_position);

        for (auto m = 0ULL; m < M; ++m)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(2ULL, m * n_elements + n) = z_position.at(m);
    }

    // Apply element coupling for polarization indicators 4, 5, and 6
    if (pol > 3 && M > 1ULL)
    {
        double tmp = 2.0 * pi * std::sin(double(tilt) * deg2rad) / wavelength;
        arma::Col<dtype> cpl_re = z_position * dtype(tmp);
        tmp = 1.0 / std::sqrt(double(M));
        arma::Col<dtype> cpl_im = arma::sin(cpl_re) * dtype(tmp);
        cpl_re = arma::cos(cpl_re) * dtype(tmp);

        ant.coupling_re.zeros(n_elements * M, n_elements);
        ant.coupling_im.zeros(n_elements * M, n_elements);

        for (auto m = 0ULL; m < M; ++m)
            for (auto n = 0ULL; n < n_elements; ++n)
            {
                ant.coupling_re.at(m * n_elements + n, n) = cpl_re.at(m);
                ant.coupling_im.at(m * n_elements + n, n) = cpl_im.at(m);
            }

        ant.combine_pattern();
        M = 1ULL;
    }

    // Duplicate the existing elements in y-direction (horizontal stacking)
    n_elements = ant.n_elements();
    if (N > 1ULL)
    {
        for (unsigned long long source = n_elements; source > 0ULL; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = N * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> y_position = arma::linspace<arma::Col<dtype>>(zero, dtype(N - 1ULL) * spacing * dtype(wavelength), N);
        y_position = y_position - arma::mean(y_position);

        for (auto m = 0ULL; m < N; ++m)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, m * n_elements + n) = y_position.at(m);
    }

    // Duplicate panels in z-direction (vertical panel stacking)
    n_elements = ant.n_elements();
    if (Mg > 1ULL)
    {
        for (unsigned long long source = n_elements; source > 0ULL; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = Mg * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> zg_position = arma::linspace<arma::Col<dtype>>(zero, dtype(Mg - 1ULL) * dgv * dtype(wavelength), Mg);
        zg_position = zg_position - arma::mean(zg_position);

        for (auto mg = 0ULL; mg < Mg; ++mg)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(2ULL, mg * n_elements + n) += zg_position.at(mg);
    }

    // Duplicate panels in y-direction (horizontal panel stacking)
    n_elements = ant.n_elements();
    if (Ng > 1)
    {
        for (unsigned long long source = n_elements; source > 0; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = Ng * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> yg_position = arma::linspace<arma::Col<dtype>>(zero, dtype(Ng - 1ULL) * dgh * dtype(wavelength), Ng);
        yg_position = yg_position - arma::mean(yg_position);

        for (auto mg = 0ULL; mg < Ng; ++mg)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, mg * n_elements + n) += yg_position.at(mg);
    }

    ant.name = "3gpp";
    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_3GPP(unsigned long long M, unsigned long long N, float center_freq,
                                                                            unsigned pol, float tilt, float spacing,
                                                                            unsigned long long Mg, unsigned long long Ng, float dgv, float dgh,
                                                                            const quadriga_lib::arrayant<float> *pattern);

template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_3GPP(unsigned long long M, unsigned long long N, double center_freq,
                                                                             unsigned pol, double tilt, double spacing,
                                                                             unsigned long long Mg, unsigned long long Ng, double dgv, double dgh,
                                                                             const quadriga_lib::arrayant<double> *pattern);

// Calculate channel coefficients for spherical waves
template <typename dtype>
void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<dtype> *tx_array, const quadriga_lib::arrayant<dtype> *rx_array,
                                          dtype Tx, dtype Ty, dtype Tz, dtype Tb, dtype Tt, dtype Th,
                                          dtype Rx, dtype Ry, dtype Rz, dtype Rb, dtype Rt, dtype Rh,
                                          const arma::Mat<dtype> *fbs_pos, const arma::Mat<dtype> *lbs_pos,
                                          const arma::Col<dtype> *path_gain, const arma::Col<dtype> *path_length, const arma::Mat<dtype> *M,
                                          arma::Cube<dtype> *coeff_re, arma::Cube<dtype> *coeff_im, arma::Cube<dtype> *delay,
                                          dtype center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                          arma::Cube<dtype> *aod, arma::Cube<dtype> *eod, arma::Cube<dtype> *aoa, arma::Cube<dtype> *eoa)
{
    // Constants
    constexpr dtype los_limit = (dtype)1.0e-4;
    constexpr dtype zero = (dtype)0.0;
    constexpr dtype rC = dtype(1.0 / 299792458.0); // 1 / (Speed of light)
    dtype wavelength = (center_frequency > zero) ? (dtype)299792458.0 / center_frequency : (dtype)1.0;
    dtype wave_number = (dtype)2.095845021951682e-08 * center_frequency; // 2 * pi / C

    // Catch NULL-Pointers
    std::string error_message;
    if (tx_array == nullptr || rx_array == nullptr ||
        fbs_pos == nullptr || lbs_pos == nullptr || path_gain == nullptr || path_length == nullptr || M == nullptr ||
        coeff_re == nullptr || coeff_im == nullptr || delay == nullptr)
    {
        error_message = "Mandatory inputs and outputs cannot be NULL";
        throw std::invalid_argument(error_message.c_str());
    }

    // Check if the antennas are valid
    error_message = tx_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Transmit antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }
    error_message = rx_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Receive antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    // Check if the number of paths is consistent in all inputs
    if (fbs_pos->n_elem == 0 || lbs_pos->n_elem == 0 || path_gain->n_elem == 0 || path_length->n_elem == 0 || M->n_elem == 0)
    {
        error_message = "Missing data for any of: fbs_pos, lbs_pos, path_gain, path_length, M";
        throw std::invalid_argument(error_message.c_str());
    }
    if (fbs_pos->n_rows != 3 || lbs_pos->n_rows != 3 || M->n_rows != 8)
    {
        error_message = "Inputs 'fbs_pos' and 'lbs_pos' must have 3 rows; 'M' must have 8 rows.";
        throw std::invalid_argument(error_message.c_str());
    }

    // 64-bit integers used by Armadillo
    const arma::uword n_path = fbs_pos->n_cols;                        // Number of paths
    const arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path; // Number of output paths
    const arma::uword n_tx = tx_array->e_theta_re.n_slices;            // Number of TX antenna elements before coupling
    const arma::uword n_rx = rx_array->e_theta_re.n_slices;            // Number of RX antenna elements before coupling
    const arma::uword n_links = n_rx * n_tx;                           // Number of MIMO channel coefficients per path (n_rx * n_tx)
    const arma::uword n_tx_ports = tx_array->n_ports();                // Number of TX antenna elements after coupling
    const arma::uword n_rx_ports = rx_array->n_ports();                // Number of RX antenna elements after coupling
    const arma::uword n_ports = n_tx_ports * n_rx_ports;               // Total number of ports

    // Size types for loops and indexing
    // const size_t n_out_t = (size_t)n_out;
    const size_t n_tx_t = (size_t)n_tx;
    const size_t n_rx_t = (size_t)n_rx;
    const size_t n_links_t = (size_t)n_links;
    const size_t n_tx_ports_t = (size_t)n_tx_ports;
    const size_t n_rx_ports_t = (size_t)n_rx_ports;
    const size_t n_ports_t = (size_t)n_ports;

    // Integer types
    const int n_out_i = (int)n_out;

    if (lbs_pos->n_cols != n_path || path_gain->n_elem != n_path || path_length->n_elem != n_path || M->n_cols != n_path)
    {
        error_message = "Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set the output size
    if (coeff_re->n_rows != n_rx_ports || coeff_re->n_cols != n_tx_ports || coeff_re->n_slices != n_out)
        coeff_re->set_size(n_rx_ports, n_tx_ports, n_out);
    if (coeff_im->n_rows != n_rx_ports || coeff_im->n_cols != n_tx_ports || coeff_im->n_slices != n_out)
        coeff_im->set_size(n_rx_ports, n_tx_ports, n_out);
    if (delay->n_rows != n_rx_ports || delay->n_cols != n_tx_ports || delay->n_slices != n_out)
        delay->set_size(n_rx_ports, n_tx_ports, n_out);
    if (aod != nullptr && (aod->n_rows != n_rx_ports || aod->n_cols != n_tx_ports || aod->n_slices != n_out))
        aod->set_size(n_rx_ports, n_tx_ports, n_out);
    if (eod != nullptr && (eod->n_rows != n_rx_ports || eod->n_cols != n_tx_ports || eod->n_slices != n_out))
        eod->set_size(n_rx_ports, n_tx_ports, n_out);
    if (aoa != nullptr && (aoa->n_rows != n_rx_ports || aoa->n_cols != n_tx_ports || aoa->n_slices != n_out))
        aoa->set_size(n_rx_ports, n_tx_ports, n_out);
    if (eoa != nullptr && (eoa->n_rows != n_rx_ports || eoa->n_cols != n_tx_ports || eoa->n_slices != n_out))
        eoa->set_size(n_rx_ports, n_tx_ports, n_out);

    // Map output memory to internal representation
    arma::Cube<dtype> CR, CI, DL;        // Internal map for delays and coefficients
    arma::Mat<dtype> AOD, EOD, AOA, EOA; // Antenna interpolation requires matrix of size [n_out, n_ang]
    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;

    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
        AOD.set_size(n_links, n_out);
        EOD.set_size(n_links, n_out);
        AOA.set_size(n_links, n_out);
        EOA.set_size(n_links, n_out);
    }
    else // Direct mapping of external memory
    {
        CR = arma::Cube<dtype>(coeff_re->memptr(), n_rx, n_tx, n_out, false, true);
        CI = arma::Cube<dtype>(coeff_im->memptr(), n_rx, n_tx, n_out, false, true);
        DL = arma::Cube<dtype>(delay->memptr(), n_rx, n_tx, n_out, false, true);

        if (aod == nullptr)
            AOD.set_size(n_links, n_out);
        else
            AOD = arma::Mat<dtype>(aod->memptr(), n_links, n_out, false, true);

        if (eod == nullptr)
            EOD.set_size(n_links, n_out);
        else
            EOD = arma::Mat<dtype>(eod->memptr(), n_links, n_out, false, true);

        if (aoa == nullptr)
            AOA.set_size(n_links, n_out);
        else
            AOA = arma::Mat<dtype>(aoa->memptr(), n_links, n_out, false, true);

        if (eoa == nullptr)
            EOA.set_size(n_links, n_out);
        else
            EOA = arma::Mat<dtype>(eoa->memptr(), n_links, n_out, false, true);
    }

    // Get pointers
    const dtype *p_fbs = fbs_pos->memptr();
    const dtype *p_lbs = lbs_pos->memptr();
    const dtype *p_gain = path_gain->memptr();
    const dtype *p_length = path_length->memptr();

    dtype *p_coeff_re = CR.memptr();
    dtype *p_coeff_im = CI.memptr();
    dtype *p_delays = DL.memptr();
    dtype *p_aod = AOD.memptr();
    dtype *p_eod = EOD.memptr();
    dtype *p_aoa = AOA.memptr();
    dtype *p_eoa = EOA.memptr();
    dtype *ptr;

    // Convert inputs to orientation vector
    arma::Cube<dtype> tx_orientation(3, 1, 1);
    arma::Cube<dtype> rx_orientation(3, 1, 1);
    bool tx_orientation_not_zero = (Tb != zero || Tt != zero || Th != zero);
    bool rx_orientation_not_zero = (Rb != zero || Rt != zero || Rh != zero);

    if (tx_orientation_not_zero)
        ptr = tx_orientation.memptr(), ptr[0] = Tb, ptr[1] = Tt, ptr[2] = Th;

    if (rx_orientation_not_zero)
        ptr = rx_orientation.memptr(), ptr[0] = Rb, ptr[1] = Rt, ptr[2] = Rh;

    // Calculate the antenna element positions in GCS
    arma::Mat<dtype> tx_element_pos(3, n_tx), rx_element_pos(3, n_rx);
    dtype *p_tx = tx_element_pos.memptr(), *p_rx = rx_element_pos.memptr();

    if (tx_array->element_pos.n_elem != 0)
        std::memcpy(p_tx, tx_array->element_pos.memptr(), 3 * n_tx * sizeof(dtype));
    if (tx_orientation_not_zero) // Apply TX antenna orientation
        quick_rotate_inplace(Tb, -Tt, Th, p_tx, n_tx);
    for (size_t t = 0; t < 3 * n_tx_t; t += 3) // Add TX position
        p_tx[t] += Tx, p_tx[t + 1] += Ty, p_tx[t + 2] += Tz;

    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    if (rx_orientation_not_zero) // Apply RX antenna orientation
        quick_rotate_inplace(Rb, -Rt, Rh, p_rx, n_rx);
    for (size_t r = 0; r < 3 * n_rx_t; r += 3) // Add RX position
        p_rx[r] += Rx, p_rx[r + 1] += Ry, p_rx[r + 2] += Rz;

    // Calculate the Freespace distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // There may be multiple LOS paths. We need to find the real one
    // Detection is done by sing the shortest length difference to the TX-RX line
    arma::uword true_los_path = 0;
    dtype shortest_path = los_limit;

    // Calculate angles and delays
    // Cannot be parallelized due to "true_los_path" and "shortest_path"
    for (int i_out = 0; i_out < n_out_i; ++i_out) // Loop over paths
    {
        const size_t j = (size_t)i_out;
        const size_t i = add_fake_los_path ? j - 1 : j;
        const size_t ix = 3 * i, iy = ix + 1, iz = ix + 2;
        const size_t o = j * n_links_t; // Slice offset

        // Calculate the shortest possible path length (TX > FBS > LBS > RX)
        dtype d_shortest = dist_rx_tx, d_length = dist_rx_tx, d_fbs_lbs = zero;
        if (!add_fake_los_path || j != 0)
        {
            x = p_fbs[ix] - Tx, y = p_fbs[iy] - Ty, z = p_fbs[iz] - Tz;
            d_shortest = std::sqrt(x * x + y * y + z * z);
            x = p_lbs[ix] - p_fbs[ix], y = p_lbs[iy] - p_fbs[iy], z = p_lbs[iz] - p_fbs[iz];
            d_fbs_lbs = std::sqrt(x * x + y * y + z * z);
            d_shortest += d_fbs_lbs;
            x = Rx - p_lbs[ix], y = Ry - p_lbs[iy], z = Rz - p_lbs[iz];
            d_shortest += std::sqrt(x * x + y * y + z * z);
            d_length = (p_length[i] < d_shortest) ? d_shortest : p_length[i];
        }

        // Calculate path delays, departure angles and arrival angles
        // size_t o = j * n_links, o0 = o;
        if (std::abs(d_length - dist_rx_tx) < shortest_path) // LOS path
        {
            if (!add_fake_los_path || j != 0)
                true_los_path = j, shortest_path = std::abs(d_length - dist_rx_tx);

            for (size_t t = 0; t < n_tx_t; ++t)
                for (size_t r = 0; r < n_rx_t; ++r)
                {
                    x = p_rx[3 * r] - p_tx[3 * t];
                    y = p_rx[3 * r + 1] - p_tx[3 * t + 1];
                    z = p_rx[3 * r + 2] - p_tx[3 * t + 2];
                    dtype d = std::sqrt(x * x + y * y + z * z);

                    size_t io = o + t * n_rx_t + r;
                    p_aod[io] = std::atan2(y, x);
                    p_eod[io] = (d < los_limit) ? zero : std::asin(z / d);
                    p_aoa[io] = std::atan2(-y, -x);
                    p_eoa[io] = -p_eod[io];
                    p_delays[io] = d;
                }
        }
        else // NLOS path
        {
            dtype *dr = new dtype[n_rx];
            for (size_t r = 0; r < n_rx_t; ++r)
                x = p_lbs[ix] - p_rx[3 * r],
                y = p_lbs[iy] - p_rx[3 * r + 1],
                z = p_lbs[iz] - p_rx[3 * r + 2],
                dr[r] = std::sqrt(x * x + y * y + z * z),
                p_aoa[o + r] = std::atan2(y, x),
                p_eoa[o + r] = (dr[r] < los_limit) ? zero : std::asin(z / dr[r]);

            for (size_t t = 0; t < n_tx_t; ++t)
            {
                x = p_fbs[ix] - p_tx[3 * t],
                y = p_fbs[iy] - p_tx[3 * t + 1],
                z = p_fbs[iz] - p_tx[3 * t + 2];

                dtype dt = std::sqrt(x * x + y * y + z * z),
                      at = std::atan2(y, x),
                      et = (dt < los_limit) ? zero : std::asin(z / dt);

                for (size_t r = 0; r < n_rx_t; ++r)
                {
                    size_t io = o + t * n_rx_t + r;
                    p_aod[io] = at, p_eod[io] = et;
                    p_aoa[io] = p_aoa[o + r], p_eoa[io] = p_eoa[o + r];
                    p_delays[io] = dt + d_fbs_lbs + dr[r];
                }
            }
            delete[] dr;
        }
    }

    // Calculate TX element indices and positions
    arma::Col<unsigned> i_tx_element(n_links, arma::fill::none);
    unsigned *p_element = i_tx_element.memptr();
    arma::Mat<dtype> tx_element_pos_interp(3, n_links, arma::fill::none);
    ptr = tx_element_pos_interp.memptr();
    if (tx_array->element_pos.n_elem != 0)
        std::memcpy(p_tx, tx_array->element_pos.memptr(), 3 * n_tx * sizeof(dtype));
    else
        tx_element_pos.zeros();
    for (unsigned t = 0; t < (unsigned)n_tx; ++t)
        for (unsigned r = 0; r < (unsigned)n_rx; ++r)
            *p_element++ = t + 1, *ptr++ = p_tx[3 * t], *ptr++ = p_tx[3 * t + 1], *ptr++ = p_tx[3 * t + 2];

    // Calculate RX element indices and positions
    arma::Col<unsigned> i_rx_element(n_links, arma::fill::none);
    p_element = i_rx_element.memptr();
    arma::Mat<dtype> rx_element_pos_interp(3, n_links, arma::fill::none);
    ptr = rx_element_pos_interp.memptr();
    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    else
        rx_element_pos.zeros();
    for (unsigned t = 0; t < unsigned(n_tx); ++t)
        for (unsigned r = 0; r < unsigned(n_rx); ++r)
            *p_element++ = r + 1, *ptr++ = p_rx[3 * r], *ptr++ = p_rx[3 * r + 1], *ptr++ = p_rx[3 * r + 2];

    // Determine if we need to apply antenna-element coupling
    bool apply_element_coupling = false;
    if (rx_array->coupling_re.n_elem != 0 || rx_array->coupling_im.n_elem != 0 ||
        tx_array->coupling_re.n_elem != 0 || tx_array->coupling_im.n_elem != 0)
    {
        if (n_rx_ports != n_rx || n_tx_ports != n_tx)
            apply_element_coupling = true;

        // Check if TX coupling real part is identity matrix
        if (!apply_element_coupling && tx_array->coupling_re.n_rows == n_tx && tx_array->coupling_re.n_cols == n_tx)
        {
            const dtype *p = tx_array->coupling_re.memptr();
            for (size_t r = 0; r < n_tx_t; ++r)
                for (size_t c = 0; c < n_tx_t; ++c)
                {
                    if (r == c && std::abs(p[c * n_tx + r] - (dtype)1.0) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                    else if (r != c && std::abs(p[c * n_tx + r]) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                }
        }

        // If TX coupling has imaginary part, check if there is any element not 0
        if (!apply_element_coupling && tx_array->coupling_im.n_elem != 0)
        {
            for (const dtype *p = tx_array->coupling_im.begin(); p < tx_array->coupling_im.end(); ++p)
                if (std::abs(*p) > (dtype)1.0e-6)
                    apply_element_coupling = true;
        }

        // Check if RX coupling real part is identity matrix
        if (!apply_element_coupling && rx_array->coupling_re.n_rows == n_rx && rx_array->coupling_re.n_cols == n_rx)
        {
            const dtype *p = rx_array->coupling_re.memptr();
            for (size_t r = 0; r < n_rx_t; ++r)
                for (size_t c = 0; c < n_rx_t; ++c)
                {
                    if (r == c && std::abs(p[c * n_rx + r] - (dtype)1.0) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                    else if (r != c && std::abs(p[c * n_rx + r]) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                }
        }

        // If RX coupling has imaginary part, check if there is any element not 0
        if (!apply_element_coupling && rx_array->coupling_im.n_elem != 0)
        {
            for (const dtype *p = rx_array->coupling_im.begin(); p < rx_array->coupling_im.end(); ++p)
                if (std::abs(*p) > (dtype)1.0e-6)
                    apply_element_coupling = true;
        }
    }

    // Calculate abs( cpl )^2 and normalize the row-sum to 1
    arma::Mat<dtype> tx_coupling_pwr(n_tx, n_tx_ports, arma::fill::none);
    arma::Mat<dtype> rx_coupling_pwr(n_rx, n_rx_ports, arma::fill::none);
    if (apply_element_coupling)
    {
        quick_power_mat(n_tx, n_tx_ports, tx_coupling_pwr.memptr(), true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());
        quick_power_mat(n_rx, n_rx_ports, rx_coupling_pwr.memptr(), true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
    }

    // Calculate the MIMO channel coefficients for each path
#pragma omp parallel for
    for (int i_out = 0; i_out < n_out_i; ++i_out) // Loop over paths
    {
        const size_t j = (size_t)i_out;
        const size_t i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;
        const size_t O = j * n_links_t; // Slice offset

        // Allocate memory for temporary data
        arma::Mat<dtype> Vt_re(n_links, 1, arma::fill::none), Vt_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Ht_re(n_links, 1, arma::fill::none), Ht_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Vr_re(n_links, 1, arma::fill::none), Vr_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Hr_re(n_links, 1, arma::fill::none), Hr_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> EMPTY;

        // Calculate TX antenna response
        const arma::Mat<dtype> AOD_j(AOD.colptr(j), n_links, 1, false, true);
        const arma::Mat<dtype> EOD_j(EOD.colptr(j), n_links, 1, false, true);
        qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                                &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD_j, &EOD_j,
                                &i_tx_element, &tx_orientation, &tx_element_pos_interp,
                                &Vt_re, &Vt_im, &Ht_re, &Ht_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

        // Calculate RX antenna response
        const arma::Mat<dtype> AOA_j(AOA.colptr(j), n_links, 1, false, true);
        const arma::Mat<dtype> EOA_j(EOA.colptr(j), n_links, 1, false, true);
        qd_arrayant_interpolate(&rx_array->e_theta_re, &rx_array->e_theta_im, &rx_array->e_phi_re, &rx_array->e_phi_im,
                                &rx_array->azimuth_grid, &rx_array->elevation_grid, &AOA_j, &EOA_j,
                                &i_rx_element, &rx_orientation, &rx_element_pos_interp,
                                &Vr_re, &Vr_im, &Hr_re, &Hr_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

        // Calculate the MIMO channel coefficients
        const dtype *pM = M->colptr(i);
        const dtype *pVrr = Vr_re.memptr(), *pVri = Vr_im.memptr(),
                    *pHrr = Hr_re.memptr(), *pHri = Hr_im.memptr(),
                    *pVtr = Vt_re.memptr(), *pVti = Vt_im.memptr(),
                    *pHtr = Ht_re.memptr(), *pHti = Ht_im.memptr();
        const dtype path_amplitude = (add_fake_los_path && j == 0) ? zero : std::sqrt(p_gain[i]);

        for (size_t t = 0; t < n_tx_t; ++t)
            for (size_t r = 0; r < n_rx_t; ++r)
            {
                size_t R = t * n_rx_t + r;

                dtype re = zero, im = zero;
                re += pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
                re += pHrr[R] * pM[2] * pVtr[R] - pHri[R] * pM[3] * pVtr[R] - pHrr[R] * pM[3] * pVti[R] - pHri[R] * pM[2] * pVti[R];
                re += pVrr[R] * pM[4] * pHtr[R] - pVri[R] * pM[5] * pHtr[R] - pVrr[R] * pM[5] * pHti[R] - pVri[R] * pM[4] * pHti[R];
                re += pHrr[R] * pM[6] * pHtr[R] - pHri[R] * pM[7] * pHtr[R] - pHrr[R] * pM[7] * pHti[R] - pHri[R] * pM[6] * pHti[R];

                im += pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
                im += pHrr[R] * pM[3] * pVtr[R] + pHri[R] * pM[2] * pVtr[R] + pHrr[R] * pM[2] * pVti[R] - pHri[R] * pM[3] * pVti[R];
                im += pVrr[R] * pM[5] * pHtr[R] + pVri[R] * pM[4] * pHtr[R] + pVrr[R] * pM[4] * pHti[R] - pVri[R] * pM[5] * pHti[R];
                im += pHrr[R] * pM[7] * pHtr[R] + pHri[R] * pM[6] * pHtr[R] + pHrr[R] * pM[6] * pHti[R] - pHri[R] * pM[7] * pHti[R];

                dtype dl = p_delays[O + R]; // path length from previous calculation
                dtype phase = wave_number * std::fmod(dl, wavelength);
                dtype cp = std::cos(phase), sp = std::sin(phase);

                p_coeff_re[O + R] = (re * cp + im * sp) * path_amplitude;
                p_coeff_im[O + R] = (-re * sp + im * cp) * path_amplitude;

                dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                p_delays[O + R] = dl * rC;
            }

        // Apply antenna element coupling
        if (apply_element_coupling)
        {
            // Allocate memory for temporary data
            size_t N = (n_ports_t > n_links_t) ? n_ports_t : n_links_t;
            dtype *tempX = new dtype[N];
            dtype *tempY = new dtype[N];
            dtype *tempZ = new dtype[N];
            dtype *tempT = new dtype[N];

            // Process coefficients and delays
            if (different_output_size) // Data is stored in internal memory, we can write directly to the output
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[O], &p_coeff_im[O],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                // Apply coupling to delays
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), &p_delays[O], tx_coupling_pwr.memptr(), delay->slice_memptr(j),
                                     n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
            }
            else // Data has been written to external memory
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[O], &p_coeff_im[O],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                std::memcpy(&p_coeff_re[O], tempX, n_ports_t * sizeof(dtype));
                std::memcpy(&p_coeff_im[O], tempY, n_ports_t * sizeof(dtype));

                // Apply coupling to delays
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), &p_delays[O], tx_coupling_pwr.memptr(), tempT, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                std::memcpy(&p_delays[O], tempT, n_ports_t * sizeof(dtype));
            }

            // Process departure angles
            if (aod != nullptr || eod != nullptr)
            {
                // Convert AOD and EOD to Cartesian coordinates
                quick_geo2cart(n_links_t, &p_aod[O], tempX, tempY, &p_eod[O], tempZ);

                // Apply coupling to the x, y, z, component independently
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempX, tx_coupling_pwr.memptr(), tempT, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempY, tx_coupling_pwr.memptr(), tempX, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempZ, tx_coupling_pwr.memptr(), tempY, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aod" and "p_eod"
                    quick_cart2geo(n_ports_t, &p_aod[O], tempT, tempX, &p_eod[O], tempY);
                else if (aod == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, NULL, tempT, tempX, eod->slice_memptr(j), tempY);
                else if (eod == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, aod->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    quick_cart2geo<dtype>(n_ports_t, aod->slice_memptr(j), tempT, tempX, eod->slice_memptr(j), tempY);
            }

            // Process arrival angles
            if (aoa != nullptr || eoa != nullptr)
            {
                // Convert AOD and EOD to Cartesian coordinates
                quick_geo2cart(n_links_t, &p_aoa[O], tempX, tempY, &p_eoa[O], tempZ);

                // Apply coupling to the x, y, z, component independently
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempX, tx_coupling_pwr.memptr(), tempT, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempY, tx_coupling_pwr.memptr(), tempX, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempZ, tx_coupling_pwr.memptr(), tempY, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aoa" and "p_eoa"
                    quick_cart2geo(n_ports_t, &p_aoa[O], tempT, tempX, &p_eoa[O], tempY);
                else if (aoa == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, NULL, tempT, tempX, eoa->slice_memptr(j), tempY);
                else if (eoa == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, aoa->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    quick_cart2geo<dtype>(n_ports_t, aoa->slice_memptr(j), tempT, tempX, eoa->slice_memptr(j), tempY);
            }

            // Free temporary memory
            delete[] tempX;
            delete[] tempY;
            delete[] tempZ;
            delete[] tempT;
        }
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0)
    {
        std::memcpy(p_coeff_re, CR.slice_memptr(true_los_path), n_links_t * sizeof(dtype));
        std::memcpy(p_coeff_im, CI.slice_memptr(true_los_path), n_links_t * sizeof(dtype));
        CR.slice(true_los_path).zeros();
        CI.slice(true_los_path).zeros();
    }
}

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<float> *tx_array, const quadriga_lib::arrayant<float> *rx_array,
                                                   float Tx, float Ty, float Tz, float Tb, float Tt, float Th,
                                                   float Rx, float Ry, float Rz, float Rb, float Rt, float Rh,
                                                   const arma::Mat<float> *fbs_pos, const arma::Mat<float> *lbs_pos,
                                                   const arma::Col<float> *path_gain, const arma::Col<float> *path_length, const arma::Mat<float> *M,
                                                   arma::Cube<float> *coeff_re, arma::Cube<float> *coeff_im, arma::Cube<float> *delay,
                                                   float center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                   arma::Cube<float> *aod, arma::Cube<float> *eod, arma::Cube<float> *aoa, arma::Cube<float> *eoa);

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<double> *tx_array, const quadriga_lib::arrayant<double> *rx_array,
                                                   double Tx, double Ty, double Tz, double Tb, double Tt, double Th,
                                                   double Rx, double Ry, double Rz, double Rb, double Rt, double Rh,
                                                   const arma::Mat<double> *fbs_pos, const arma::Mat<double> *lbs_pos,
                                                   const arma::Col<double> *path_gain, const arma::Col<double> *path_length, const arma::Mat<double> *M,
                                                   arma::Cube<double> *coeff_re, arma::Cube<double> *coeff_im, arma::Cube<double> *delay,
                                                   double center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                   arma::Cube<double> *aod, arma::Cube<double> *eod, arma::Cube<double> *aoa, arma::Cube<double> *eoa);

// Calculate channel coefficients for planar waves
template <typename dtype>
void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<dtype> *tx_array, const quadriga_lib::arrayant<dtype> *rx_array,
                                       dtype Tx, dtype Ty, dtype Tz, dtype Tb, dtype Tt, dtype Th,
                                       dtype Rx, dtype Ry, dtype Rz, dtype Rb, dtype Rt, dtype Rh,
                                       const arma::Col<dtype> *aod, const arma::Col<dtype> *eod, const arma::Col<dtype> *aoa, const arma::Col<dtype> *eoa,
                                       const arma::Col<dtype> *path_gain, const arma::Col<dtype> *path_length, const arma::Mat<dtype> *M,
                                       arma::Cube<dtype> *coeff_re, arma::Cube<dtype> *coeff_im, arma::Cube<dtype> *delay,
                                       dtype center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                       arma::Col<dtype> *rx_Doppler)
{
    // Constants
    constexpr dtype los_limit = dtype(1.0e-4);
    constexpr dtype zero = dtype(0.0);
    constexpr dtype one = dtype(1.0);
    constexpr dtype rC = dtype(1.0 / 299792458.0); // 1 / (Speed of light)
    dtype wavelength = center_frequency > zero ? dtype(299792458.0) / center_frequency : one;
    dtype wave_number = dtype(2.095845021951682e-08) * center_frequency; // 2 * pi / C

    // Catch NULL-Pointers
    std::string error_message;
    if (tx_array == nullptr || rx_array == nullptr ||
        aod == nullptr || eod == nullptr || aoa == nullptr || eoa == nullptr || path_gain == nullptr || path_length == nullptr || M == nullptr ||
        coeff_re == nullptr || coeff_im == nullptr || delay == nullptr)
    {
        error_message = "Mandatory inputs and outputs cannot be NULL";
        throw std::invalid_argument(error_message.c_str());
    }

    // Check if the antennas are valid
    error_message = tx_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Transmit antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }
    error_message = rx_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Receive antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    unsigned long long n_path = aod->n_elem;                            // Number of paths
    unsigned long long n_out = add_fake_los_path ? n_path + 1 : n_path; // Number of output paths
    unsigned long long n_tx = tx_array->e_theta_re.n_slices;            // Number of TX antenna elements before coupling
    unsigned long long n_rx = rx_array->e_theta_re.n_slices;            // Number of RX antenna elements before coupling
    unsigned long long n_links = n_rx * n_tx;                           // Number of MIMO channel coefficients per path (n_rx * n_tx)
    unsigned long long n_tx_ports = tx_array->n_ports();                // Number of TX antenna elements after coupling
    unsigned long long n_rx_ports = rx_array->n_ports();                // Number of RX antenna elements after coupling
    unsigned long long n_ports = n_tx_ports * n_rx_ports;               // Total number of ports

    // Check if the number of paths is consistent in all inputs
    if (n_path == 0ULL || eod->n_elem != n_path || aoa->n_elem != n_path || eoa->n_elem != n_path || path_gain->n_elem != n_path || path_length->n_elem != n_path || M->n_cols != n_path)
    {
        error_message = "Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).";
        throw std::invalid_argument(error_message.c_str());
    }
    if (M->n_rows != 8ULL)
    {
        error_message = "Polarization transfer matrix 'M' must have 8 rows.";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set the output size
    if (coeff_re->n_rows != n_rx_ports || coeff_re->n_cols != n_tx_ports || coeff_re->n_slices != n_out)
        coeff_re->set_size(n_rx_ports, n_tx_ports, n_out);
    if (coeff_im->n_rows != n_rx_ports || coeff_im->n_cols != n_tx_ports || coeff_im->n_slices != n_out)
        coeff_im->set_size(n_rx_ports, n_tx_ports, n_out);
    if (delay->n_rows != n_rx_ports || delay->n_cols != n_tx_ports || delay->n_slices != n_out)
        delay->set_size(n_rx_ports, n_tx_ports, n_out);

    // Map output memory to internal representation
    arma::Cube<dtype> CR, CI, DL; // Internal map for coefficients
    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;

    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
    }
    else // Direct mapping of external memory
    {
        CR = arma::Cube<dtype>(coeff_re->memptr(), n_rx, n_tx, n_out, false, true);
        CI = arma::Cube<dtype>(coeff_im->memptr(), n_rx, n_tx, n_out, false, true);
        DL = arma::Cube<dtype>(delay->memptr(), n_rx, n_tx, n_out, false, true);
    }

    // Antenna interpolation requires matrix of size [1, n_ang], 1 angle per path
    // Angles are mapped to correct value ranges during antenna interpolation
    const arma::Mat<dtype> AOD = arma::Mat<dtype>(const_cast<dtype *>(aod->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOD = arma::Mat<dtype>(const_cast<dtype *>(eod->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> AOA = arma::Mat<dtype>(const_cast<dtype *>(aoa->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOA = arma::Mat<dtype>(const_cast<dtype *>(eoa->memptr()), 1, n_path, false, true);

    // Get pointers
    const dtype *p_gain = path_gain->memptr();
    const dtype *p_length = path_length->memptr();

    dtype *p_coeff_re = CR.memptr();
    dtype *p_coeff_im = CI.memptr();
    dtype *p_delays = DL.memptr();
    dtype *ptr;

    // Convert inputs to orientation vector
    arma::Cube<dtype> tx_orientation(3, 1, 1);
    arma::Cube<dtype> rx_orientation(3, 1, 1);
    ptr = tx_orientation.memptr(), ptr[0] = Tb, ptr[1] = Tt, ptr[2] = Th;
    ptr = rx_orientation.memptr(), ptr[0] = Rb, ptr[1] = Rt, ptr[2] = Rh;

    // Calculate the Freespace distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // Interpolate the antenna patterns for all paths
    arma::Mat<dtype> Vt_re(n_tx, n_out, arma::fill::none), Vt_im(n_tx, n_out, arma::fill::none),
        Ht_re(n_tx, n_out, arma::fill::none), Ht_im(n_tx, n_out, arma::fill::none),
        Vr_re(n_rx, n_out, arma::fill::none), Vr_im(n_rx, n_out, arma::fill::none),
        Hr_re(n_rx, n_out, arma::fill::none), Hr_im(n_rx, n_out, arma::fill::none),
        Pt(n_tx, n_out, arma::fill::none), Pr(n_rx, n_out, arma::fill::none);
    arma::Mat<dtype> AOA_loc, EOA_loc, EMPTY;

    // To calculate the Doppler weights, we need the arrival in local antenna-coordinates
    if (rx_Doppler != nullptr)
    {
        AOA_loc.set_size(n_rx, n_path);
        EOA_loc.set_size(n_rx, n_path);
        if (rx_Doppler->n_elem != n_out)
            rx_Doppler->set_size(n_out);
    }

    arma::Col<unsigned> i_element(n_tx, arma::fill::none);
    unsigned *p_element = i_element.memptr();
    for (unsigned t = 0; t < unsigned(n_tx); ++t)
        *p_element++ = t + 1;

    arma::Mat<dtype> element_pos_interp(3ULL, n_tx);
    if (tx_array->element_pos.n_elem != 0ULL)
        std::memcpy(element_pos_interp.memptr(), tx_array->element_pos.memptr(), 3ULL * n_tx * sizeof(dtype));

    qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                            &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD, &EOD,
                            &i_element, &tx_orientation, &element_pos_interp,
                            &Vt_re, &Vt_im, &Ht_re, &Ht_im, &Pt, &EMPTY, &EMPTY, &EMPTY);

    i_element.set_size(n_rx);
    p_element = i_element.memptr();
    for (unsigned r = 0; r < unsigned(n_rx); ++r)
        *p_element++ = r + 1;

    element_pos_interp.zeros(3ULL, n_rx);
    if (rx_array->element_pos.n_elem != 0ULL)
        std::memcpy(element_pos_interp.memptr(), rx_array->element_pos.memptr(), 3ULL * n_rx * sizeof(dtype));

    qd_arrayant_interpolate(&rx_array->e_theta_re, &rx_array->e_theta_im, &rx_array->e_phi_re, &rx_array->e_phi_im,
                            &rx_array->azimuth_grid, &rx_array->elevation_grid, &AOA, &EOA,
                            &i_element, &rx_orientation, &element_pos_interp,
                            &Vr_re, &Vr_im, &Hr_re, &Hr_im, &Pr, &AOA_loc, &EOA_loc, &EMPTY);

    element_pos_interp.reset();

    // Calculate the Doppler weights
    if (rx_Doppler != nullptr)
    {
        dtype *pAz = AOA_loc.memptr(), *pEl = EOA_loc.memptr();
        dtype *pD = add_fake_los_path ? rx_Doppler->memptr() + 1 : rx_Doppler->memptr();
        for (auto i = 0ULL; i < n_path; ++i)
            pD[i] = std::cos(pAz[i * n_rx]) * std::cos(pEl[i * n_rx]);
    }

    // Calculate the MIMO channel coefficients for each path
    auto true_los_path = 0ULL;
    dtype true_los_power = zero;
    for (auto j = 0ULL; j < n_out; ++j) // Loop over paths
    {
        unsigned long long i = add_fake_los_path ? (j == 0ULL ? 0ULL : j - 1ULL) : j;

        const dtype *pM = M->colptr(i);
        dtype *pVrr = Vr_re.colptr(i), *pVri = Vr_im.colptr(i),
              *pHrr = Hr_re.colptr(i), *pHri = Hr_im.colptr(i),
              *pVtr = Vt_re.colptr(i), *pVti = Vt_im.colptr(i),
              *pHtr = Ht_re.colptr(i), *pHti = Ht_im.colptr(i);

        dtype *pPt = Pt.colptr(i), *pPr = Pr.colptr(i);

        dtype path_amplitude = add_fake_los_path && j == 0ULL ? zero : std::sqrt(p_gain[i]);
        dtype path_length = add_fake_los_path && j == 0ULL ? dist_rx_tx : p_length[i];

        // LOS path detection
        if (std::abs(path_length - dist_rx_tx) < los_limit && add_fake_los_path && j != 0ULL && p_gain[i] > true_los_power)
            true_los_path = j, true_los_power = p_gain[i];

        unsigned long long O = j * n_links; // Slice offset
        for (auto t = 0ULL; t < n_tx; ++t)
            for (auto r = 0ULL; r < n_rx; ++r)
            {
                unsigned long long R = t * n_rx + r;

                dtype re = zero, im = zero;
                re += pVrr[r] * pM[0] * pVtr[t] - pVri[r] * pM[1] * pVtr[t] - pVrr[r] * pM[1] * pVti[t] - pVri[r] * pM[0] * pVti[t];
                re += pHrr[r] * pM[2] * pVtr[t] - pHri[r] * pM[3] * pVtr[t] - pHrr[r] * pM[3] * pVti[t] - pHri[r] * pM[2] * pVti[t];
                re += pVrr[r] * pM[4] * pHtr[t] - pVri[r] * pM[5] * pHtr[t] - pVrr[r] * pM[5] * pHti[t] - pVri[r] * pM[4] * pHti[t];
                re += pHrr[r] * pM[6] * pHtr[t] - pHri[r] * pM[7] * pHtr[t] - pHrr[r] * pM[7] * pHti[t] - pHri[r] * pM[6] * pHti[t];

                im += pVrr[r] * pM[1] * pVtr[t] + pVri[r] * pM[0] * pVtr[t] + pVrr[r] * pM[0] * pVti[t] - pVri[r] * pM[1] * pVti[t];
                im += pHrr[r] * pM[3] * pVtr[t] + pHri[r] * pM[2] * pVtr[t] + pHrr[r] * pM[2] * pVti[t] - pHri[r] * pM[3] * pVti[t];
                im += pVrr[r] * pM[5] * pHtr[t] + pVri[r] * pM[4] * pHtr[t] + pVrr[r] * pM[4] * pHti[t] - pVri[r] * pM[5] * pHti[t];
                im += pHrr[r] * pM[7] * pHtr[t] + pHri[r] * pM[6] * pHtr[t] + pHrr[r] * pM[6] * pHti[t] - pHri[r] * pM[7] * pHti[t];

                dtype dl = pPt[t] + path_length + pPr[r];
                dtype phase = wave_number * std::fmod(dl, wavelength);
                dtype cp = std::cos(phase), sp = std::sin(phase);

                p_coeff_re[O + R] = (re * cp + im * sp) * path_amplitude;
                p_coeff_im[O + R] = (-re * sp + im * cp) * path_amplitude;

                dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                p_delays[O + R] = dl * rC;
            }
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0)
    {
        std::memcpy(p_coeff_re, CR.slice_memptr(true_los_path), n_links * sizeof(dtype));
        std::memcpy(p_coeff_im, CI.slice_memptr(true_los_path), n_links * sizeof(dtype));
        CR.slice(true_los_path).zeros();
        CI.slice(true_los_path).zeros();
    }

    // Apply antenna element coupling
    if (rx_array->coupling_re.n_elem != 0ULL || rx_array->coupling_im.n_elem != 0ULL ||
        tx_array->coupling_re.n_elem != 0ULL || tx_array->coupling_im.n_elem != 0ULL)
    {
        // Calculate abs( cpl )^2 and normalize the row-sum to 1
        dtype *p_rx_cpl = new dtype[n_rx * n_rx_ports];
        dtype *p_tx_cpl = new dtype[n_tx * n_tx_ports];
        quick_power_mat(n_rx, n_rx_ports, p_rx_cpl, true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
        quick_power_mat(n_tx, n_tx_ports, p_tx_cpl, true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());

        // Process coefficients and delays
        if (different_output_size) // Data is stored in internal memory, we can write directly to the output
        {
            for (auto j = 0ULL; j < n_out; ++j) // Loop over paths
            {
                unsigned long long o = j * n_links; // Slice offset

                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n_rx, n_rx_ports, n_tx, n_tx_ports);

                // Apply coupling to delays
                quick_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, delay->slice_memptr(j), n_rx, n_rx_ports, n_tx, n_tx_ports);
            }
        }
        else // Data has been written to external memory
        {
            // Allocate memory for temporary data
            dtype *tempX = new dtype[n_ports];
            dtype *tempY = new dtype[n_ports];

            for (auto o = 0ULL; o < n_links * n_out; o += n_links) // Loop over paths
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n_rx, n_rx_ports, n_tx, n_tx_ports);

                std::memcpy(&p_coeff_re[o], tempX, n_ports * sizeof(dtype));
                std::memcpy(&p_coeff_im[o], tempY, n_ports * sizeof(dtype));

                // Apply coupling to delays
                quick_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, tempX, n_rx, n_rx_ports, n_tx, n_tx_ports);
                std::memcpy(&p_delays[o], tempX, n_ports * sizeof(dtype));
            }
            delete[] tempX;
            delete[] tempY;
        }

        // Free temporary memory
        delete[] p_rx_cpl;
        delete[] p_tx_cpl;
    }
}

template void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<float> *tx_array, const quadriga_lib::arrayant<float> *rx_array,
                                                float Tx, float Ty, float Tz, float Tb, float Tt, float Th,
                                                float Rx, float Ry, float Rz, float Rb, float Rt, float Rh,
                                                const arma::Col<float> *aod, const arma::Col<float> *eod, const arma::Col<float> *aoa, const arma::Col<float> *eoa,
                                                const arma::Col<float> *path_gain, const arma::Col<float> *path_length, const arma::Mat<float> *M,
                                                arma::Cube<float> *coeff_re, arma::Cube<float> *coeff_im, arma::Cube<float> *delay,
                                                float center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                arma::Col<float> *rx_Doppler);

template void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<double> *tx_array, const quadriga_lib::arrayant<double> *rx_array,
                                                double Tx, double Ty, double Tz, double Tb, double Tt, double Th,
                                                double Rx, double Ry, double Rz, double Rb, double Rt, double Rh,
                                                const arma::Col<double> *aod, const arma::Col<double> *eod, const arma::Col<double> *aoa, const arma::Col<double> *eoa,
                                                const arma::Col<double> *path_gain, const arma::Col<double> *path_length, const arma::Mat<double> *M,
                                                arma::Cube<double> *coeff_re, arma::Cube<double> *coeff_im, arma::Cube<double> *delay,
                                                double center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                arma::Col<double> *rx_Doppler);
