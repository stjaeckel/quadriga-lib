// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the sources of the channel sections: matrix products of the channel generators, the
// combination of antenna patterns into coefficients, the batch ("_vec") checks and snapshot loop, and the type ID of
// values in std::any

#ifndef qd_helpers_channel_hpp
#define qd_helpers_channel_hpp

#include "quadriga_arrayant.hpp"
#include "qd_helpers_arrayant.hpp"

#include <armadillo>
#include <any>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

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
//   with labels M_<rx><tx> (first letter: RX polarisation, second: TX polarisation), i.e. pM holds the 2x2 Jones
//   matrix column-major: rows 0-1 = (V rx, V tx), 2-3 = (H rx, V tx), 4-5 = (V rx, H tx), 6-7 = (H rx, H tx).
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
                             const dtype *pM,                      // Polarization transfer matrix, 8 values broadcast over all links: {Re_VV, Im_VV, Re_HV, Im_HV, Re_VH, Im_VH, Re_HH, Im_HH}, labels <rx><tx>
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

// Checks of the "_vec" channel functions, before any output is changed: positions and orientations [3, 1] or
// [3, n_snap], antennas and kernel (also without snapshots; the single call checks them again, from a cache).
// 'n_path(s)' gives the number of input paths of snapshot s. Returns whether the snapshot loop runs in parallel: from
// a work of 64 coefficients (element pairs times output paths, plus 8 per call for its fixed cost; measured on one
// thread: 3.2 us per call and 0.35 us per coefficient). 8 SISO snapshots with one path each (work 80) run 2x
// (spherical) and 2.9x (planar) faster than on one thread, 4 snapshots 1.3x (spherical).
template <typename dtype, typename N>
inline bool qd_channels_vec_check(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array,
                                  const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                  const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                  size_t n_snap, N &&n_path, int use_kernel)
{
    if (tx_pos.n_rows != 3ULL || tx_orientation.n_rows != 3ULL || rx_pos.n_rows != 3ULL || rx_orientation.n_rows != 3ULL)
        throw std::invalid_argument("Inputs 'tx_pos', 'tx_orientation', 'rx_pos' and 'rx_orientation' must have 3 rows.");

    for (const arma::Mat<dtype> *m : {&tx_pos, &tx_orientation, &rx_pos, &rx_orientation})
        if (m->n_cols != 1ULL && m->n_cols != n_snap)
            throw std::invalid_argument("Inputs 'tx_pos', 'tx_orientation', 'rx_pos' and 'rx_orientation' must have 1 or 'n_snap' columns.");

    qd_arrayant_interpolate_kernel(use_kernel, std::max(tx_array.e_theta_re.n_elem, rx_array.e_theta_re.n_elem));
    std::string antenna_error = qd_arrayant_is_valid_quick(tx_array);
    if (!antenna_error.empty())
        throw std::invalid_argument("Transmit antenna: " + antenna_error);
    antenna_error = qd_arrayant_is_valid_quick(rx_array);
    if (!antenna_error.empty())
        throw std::invalid_argument("Receive antenna: " + antenna_error);

    const arma::uword n_links = tx_array.e_theta_re.n_slices * rx_array.e_theta_re.n_slices;
    arma::uword n_work = 0ULL;
    for (size_t s = 0; s < n_snap; ++s)
        n_work += ((arma::uword)n_path(s) + 1ULL) * n_links + 8ULL;
    return n_snap > 1 && n_work >= 64;
}

// Column s of a position or orientation with 1 or n_snap columns, aliased without copying
template <typename dtype>
inline arma::Mat<dtype> qd_channels_vec_col(const arma::Mat<dtype> &m, size_t s)
{
    return arma::Mat<dtype>(const_cast<dtype *>(m.colptr(m.n_cols == 1ULL ? 0ULL : s)), 3, 1, false, true);
}

// Calls 'call(s, coeff_re_s, coeff_im_s)' for the snapshots s = 0 ... n_snap - 1, in parallel if 'parallel' is set.
// - Without 'coeff', the split outputs are the entries of 'coeff_re' / 'coeff_im' (resized to n_snap).
// - With 'coeff', each thread computes into its own split cubes and interleaves them into the entry of 'coeff';
//   'coeff_re' and 'coeff_im' are left empty. Entries that already have the output size are written in place.
// - An error throws std::invalid_argument with the message of the call, preceded by "Snapshot <s>: " of the lowest
//   failing snapshot.
// - Inside the parallel region, parallel regions of 'call' run on one thread (one active level by default).
template <typename dtype, typename F>
inline void qd_channels_snapshot_loop(size_t n_snap, bool parallel,
                                      std::vector<arma::Cube<dtype>> &coeff_re, std::vector<arma::Cube<dtype>> &coeff_im,
                                      std::vector<arma::Cube<std::complex<dtype>>> *coeff, F &&call)
{
    if (coeff != nullptr)
    {
        coeff_re.clear(), coeff_im.clear();
        if (coeff->size() != n_snap)
            coeff->resize(n_snap);
    }
    else if (coeff_re.size() != n_snap || coeff_im.size() != n_snap)
        coeff_re.resize(n_snap), coeff_im.resize(n_snap);

    std::string error_message;
    long long error_snap = (long long)n_snap;
    const long long n_snap_i = (long long)n_snap;

    // One snapshot of the loop; 'cr' and 'ci' are the split scratch of the complex output, reused by the thread
    auto run = [&](long long i, arma::Cube<dtype> &cr, arma::Cube<dtype> &ci)
    {
        const size_t s = (size_t)i;
        try
        {
            if (coeff == nullptr)
                call(s, coeff_re[s], coeff_im[s]);
            else
            {
                call(s, cr, ci);
                arma::Cube<std::complex<dtype>> &c = (*coeff)[s];
                if (c.n_rows != cr.n_rows || c.n_cols != cr.n_cols || c.n_slices != cr.n_slices)
                    c.set_size(cr.n_rows, cr.n_cols, cr.n_slices);
                std::complex<dtype> *p = c.memptr();
                const dtype *p_re = cr.memptr(), *p_im = ci.memptr();
                for (arma::uword k = 0; k < cr.n_elem; ++k)
                    p[k] = std::complex<dtype>(p_re[k], p_im[k]);
            }
        }
        catch (const std::exception &e)
        {
#pragma omp critical
            if (i < error_snap)
                error_snap = i, error_message = e.what();
        }
    };

    if (parallel)
    {
#pragma omp parallel
        {
            arma::Cube<dtype> cr, ci;
#pragma omp for schedule(dynamic)
            for (long long i = 0; i < n_snap_i; ++i)
                run(i, cr, ci);
        }
    }
    else // Without a parallel region: its fixed cost (about 1.5 us per call, measured) matters for single snapshots
    {
        arma::Cube<dtype> cr, ci;
        for (long long i = 0; i < n_snap_i && error_snap == n_snap_i; ++i)
            run(i, cr, ci);
    }

    if (error_snap < n_snap_i)
        throw std::invalid_argument("Snapshot " + std::to_string(error_snap) + ": " + error_message);
}

// Type ID, shape and a read-only pointer to the data of the value in a std::any (implementation of any_type_id)

// Codes 10 + k (scalar), 20 + k (arma::Mat), 30 + k (arma::Cube), 40 + k (arma::Col), 50 + k (arma::Row) for the
// element type E with index k; -1 if 'a' holds none of them
template <typename E>
static inline int qd_any_type_id_elem(const std::any *a, int k, unsigned long long *dims, const void **data)
{
    if (const E *x = std::any_cast<E>(a))
        return *data = x, 10 + k;
    if (const arma::Mat<E> *x = std::any_cast<arma::Mat<E>>(a))
        return dims[0] = x->n_rows, dims[1] = x->n_cols, *data = x->memptr(), 20 + k;
    if (const arma::Cube<E> *x = std::any_cast<arma::Cube<E>>(a))
        return dims[0] = x->n_rows, dims[1] = x->n_cols, dims[2] = x->n_slices, *data = x->memptr(), 30 + k;
    if (const arma::Col<E> *x = std::any_cast<arma::Col<E>>(a))
        return dims[0] = x->n_elem, *data = x->memptr(), 40 + k;
    if (const arma::Row<E> *x = std::any_cast<arma::Row<E>>(a))
        return dims[0] = x->n_elem, *data = x->memptr(), 50 + k;
    return -1;
}

// Type ID of the value in a std::any (codes of quadriga_lib::any_type_id), its shape and a pointer to its data
// - dims: optional, 3 values [rows, cols, slices]; vectors give [n_elem, 1, 1], a string [length, 0, 0]
// - data: optional, the data of the contained object (characters of a string); cast it to the type of the code
static inline int qd_any_type_id(const std::any *a, unsigned long long *dims = nullptr, const void **data = nullptr)
{
    if (a == nullptr || !a->has_value())
        return -2;

    unsigned long long d[3];
    const void *p = nullptr;
    dims = dims == nullptr ? d : dims, data = data == nullptr ? &p : data;
    dims[0] = 1ULL, dims[1] = 1ULL, dims[2] = 1ULL;

    if (const std::string *s = std::any_cast<std::string>(a))
        return dims[0] = s->length(), dims[1] = 0ULL, dims[2] = 0ULL, *data = s->data(), 9;

    int id = -1;
    (void)((id = qd_any_type_id_elem<float>(a, 0, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<double>(a, 1, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<unsigned long long>(a, 2, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<long long>(a, 3, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<unsigned>(a, 4, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<int>(a, 5, dims, data)) >= 0);
    return id;
}

#endif
