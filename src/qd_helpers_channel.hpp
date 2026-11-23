// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the sources of the channel sections: matrix products of the channel generators, the
// combination of antenna patterns into coefficients, the spherical-wave geometry, the port weights, the batch
// ("_vec") checks and snapshot loop, and the type ID of values in std::any

#ifndef qd_helpers_channel_hpp
#define qd_helpers_channel_hpp

#include "quadriga_arrayant.hpp"
#include "qd_helpers_arrayant.hpp"
#include "qd_helpers_shared.hpp"

#include <armadillo>
#include <any>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Matrix product X = A^T * B * C
// - A, C can be nullptr to calculate X = A^T * B or X = B * C
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

    constexpr dtype zero = dtype(0.0), one = dtype(1.0);

    // Calculate the output row by row
    for (size_t im = 0; im < m; ++im)
    {
        for (size_t ip = 0; ip < p; ip++) // Initialize output to zero
            X[ip * m + im] = zero;

        // Process temporary matrix T = A^T * B column-wise
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

// Complex matrix product X = A^T * B * C (transpose without conjugation)
// - Ar, Cr can be nullptr (identity), Ai, Ci can be nullptr (zero)
// - Output does not need to be initialized to 0
template <typename dtype>
static inline void qd_multiply_3_complex_mat(const dtype *Ar, const dtype *Ai, // n rows, m columns
                                             const dtype *Br, const dtype *Bi, // n rows, o columns
                                             const dtype *Cr, const dtype *Ci, // o rows, p columns
                                             dtype *Xr, dtype *Xi,             // m rows, p columns
                                             size_t n, size_t m, size_t o, size_t p)
{
    constexpr dtype zero = (dtype)0.0, one = (dtype)1.0;

    // Calculate the output row by row
    for (size_t im = 0; im < m; ++im)
    {
        // Initialize output to zero
        for (size_t ip = 0; ip < p; ++ip)
            Xr[ip * m + im] = zero, Xi[ip * m + im] = zero;

        // Process temporary matrix T = A^T * B column-wise
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

// Combines the interpolated TX/RX antenna patterns with the polarization transfer matrix and applies the
// propagation phase: one slice of MIMO channel coefficients (one path, all TX-RX element pairs).
// For each link index R = 0 .. n_links-1:
//   Jones product (all values complex, c = re + j*im):
//     c = Vr * M_VV * Vt  +  Hr * M_HV * Vt  +  Vr * M_VH * Ht  +  Hr * M_HH * Ht
//   with labels M_<rx><tx> (first letter: RX polarization, second: TX polarization); pM holds the 2x2 Jones
//   matrix column-major: rows 0-1 = (V rx, V tx), 2-3 = (H rx, V tx), 4-5 = (V rx, H tx), 6-7 = (H rx, H tx).
//   Phase rotation from the element-specific path length:
//     phase = wave_number * fmod(p_delays[R], wavelength)
//     p_coeff_re[R] = ( re*cos(phase) + im*sin(phase)) * path_amplitude
//     p_coeff_im[R] = (-re*sin(phase) + im*cos(phase)) * path_amplitude
template <typename dtype>
static inline void qd_coeff_combine(const dtype *pVrr, const dtype *pVri, // RX V-pol pattern (real, imag), length n_links
                             const dtype *pHrr, const dtype *pHri, // RX H-pol pattern (real, imag), length n_links
                             const dtype *pVtr, const dtype *pVti, // TX V-pol pattern (real, imag), length n_links
                             const dtype *pHtr, const dtype *pHti, // TX H-pol pattern (real, imag), length n_links
                             const dtype *pM,                      // Polarization transfer matrix, 8 values broadcast over all links: {Re_VV, Im_VV, Re_HV, Im_HV, Re_VH, Im_VH, Re_HH, Im_HH}, labels <rx><tx>
                             const dtype *p_delays,                // Per-link path lengths in [m], length n_links
                             dtype wave_number,                    // 2*pi*f/c
                             dtype wavelength,                     // c/f
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

// Spherical-wave geometry of get_channels_spherical and get_channels_multifreq (frequency independent)
// - Element positions in GCS: the local positions 'tx_el' [3, n_tx] / 'rx_el' [3, n_rx] (empty = zeros), rotated by
//   the orientation (bank, tilt, heading) and moved to the array position; tx_pos, tx_ori, rx_pos, rx_ori have 3 values
// - Paths: fbs, lbs [3, n_path], path_length [n_path]; n_out = n_path, or n_path + 1 with add_fake_los_path (slot 0 is
//   the LOS path, path i_out - 1 in slot i_out)
// - Writes the departure and arrival angles and the element-pair path length in [m] of output path i_out and link
//   R = t * n_rx + r to aod, eod, aoa, eoa, len at i_out * n_links + R; each [n_links, n_out]
// - NLOS path length between the phase centres: max(path_length, TX-FBS-LBS-RX polyline); the extra length goes into
//   the FBS-LBS segment
// - Element indices and local positions for qd_interpolate_pattern_dispatch: i_tx, i_rx [n_links];
//   tx_pos_interp, rx_pos_interp [3, n_links]
// - dist_rx_tx: distance between the TX and RX phase centres
// - Returns the output slot of the path whose length is closest to dist_rx_tx, within 0.1 mm (the true LOS path);
//   0 if there is none
template <typename dtype>
static inline arma::uword qd_channels_spherical_geometry(const arma::Mat<dtype> &tx_el, const arma::Mat<dtype> &rx_el,
                                                         arma::uword n_tx, arma::uword n_rx,
                                                         const dtype *tx_pos, const dtype *tx_ori,
                                                         const dtype *rx_pos, const dtype *rx_ori,
                                                         const dtype *fbs, const dtype *lbs, const dtype *path_length,
                                                         arma::uword n_out, bool add_fake_los_path,
                                                         dtype *aod, dtype *eod, dtype *aoa, dtype *eoa, dtype *len,
                                                         arma::uvec &i_tx, arma::Mat<dtype> &tx_pos_interp,
                                                         arma::uvec &i_rx, arma::Mat<dtype> &rx_pos_interp,
                                                         dtype &dist_rx_tx)
{
    constexpr dtype los_limit = (dtype)1.0e-4, zero = (dtype)0.0;
    const arma::uword n_links = n_rx * n_tx;
    const dtype Tx = tx_pos[0], Ty = tx_pos[1], Tz = tx_pos[2];
    const dtype Rx = rx_pos[0], Ry = rx_pos[1], Rz = rx_pos[2];

    // Element positions in GCS
    auto element_gcs = [](const arma::Mat<dtype> &el, arma::uword n, const dtype *pos, const dtype *ori)
    {
        arma::Mat<dtype> p(3, n, arma::fill::zeros);
        if (el.n_elem != 0)
            std::memcpy(p.memptr(), el.memptr(), 3 * n * sizeof(dtype));
        if (ori[0] != (dtype)0 || ori[1] != (dtype)0 || ori[2] != (dtype)0)
            qd_rotate_inplace(ori[0], ori[1], ori[2], p.memptr(), n);
        dtype *pp = p.memptr();
        for (arma::uword i = 0; i < 3 * n; i += 3)
            pp[i] += pos[0], pp[i + 1] += pos[1], pp[i + 2] += pos[2];
        return p;
    };
    const arma::Mat<dtype> tx_gcs = element_gcs(tx_el, n_tx, tx_pos, tx_ori), rx_gcs = element_gcs(rx_el, n_rx, rx_pos, rx_ori);
    const dtype *p_tx = tx_gcs.memptr(), *p_rx = rx_gcs.memptr();

    dist_rx_tx = qd_calc_length(Tx, Ty, Tz, Rx, Ry, Rz);

    // There may be several paths within los_limit of the TX-RX distance; the closest one is the true LOS path
    arma::uword true_los_path = 0;
    dtype shortest_path = los_limit;
    std::vector<dtype> dr(n_rx);

    for (arma::uword i_out = 0; i_out < n_out; ++i_out)
    {
        const bool fake_los = add_fake_los_path && i_out == 0;
        const arma::uword i = add_fake_los_path && i_out != 0 ? i_out - 1 : i_out;
        const dtype *pf = &fbs[3 * i], *pl = &lbs[3 * i];
        const arma::uword o_slice = i_out * n_links;

        // Shortest possible path length (TX > FBS > LBS > RX)
        dtype d_length = dist_rx_tx, d_fbs_lbs = zero;
        if (!fake_los)
        {
            d_fbs_lbs = qd_calc_length(pf[0], pf[1], pf[2], pl[0], pl[1], pl[2]);
            const dtype d_shortest = qd_calc_length(Tx, Ty, Tz, pf[0], pf[1], pf[2]) + d_fbs_lbs +
                                     qd_calc_length(pl[0], pl[1], pl[2], Rx, Ry, Rz);
            d_length = (path_length[i] < d_shortest) ? d_shortest : path_length[i];

            // Paths with more interactions than FBS and LBS are longer than the TX-FBS-LBS-RX polyline: the extra
            // length goes into the FBS-LBS segment, so that the phase-centre length equals d_length (= path_length)
            d_fbs_lbs += d_length - d_shortest;
        }

        // Slot 0 of add_fake_los_path is the LOS path by definition, also when the TX-RX distance overflows
        if (fake_los || std::abs(d_length - dist_rx_tx) < shortest_path) // LOS path
        {
            if (!fake_los)
                true_los_path = i_out, shortest_path = std::abs(d_length - dist_rx_tx);

            for (arma::uword t = 0; t < n_tx; ++t)
            {
                const dtype *pt = &p_tx[3 * t];
                for (arma::uword r = 0; r < n_rx; ++r)
                {
                    const dtype *pr = &p_rx[3 * r];
                    const dtype x = pr[0] - pt[0], y = pr[1] - pt[1], z = pr[2] - pt[2];
                    const dtype d = std::sqrt(x * x + y * y + z * z);
                    const arma::uword io = o_slice + t * n_rx + r;
                    aod[io] = std::atan2(y, x);
                    eod[io] = (d < los_limit) ? zero : std::asin(z / d);
                    aoa[io] = std::atan2(-y, -x);
                    eoa[io] = -eod[io];
                    len[io] = d;
                }
            }
        }
        else // NLOS path
        {
            for (arma::uword r = 0; r < n_rx; ++r)
            {
                const dtype *pr = &p_rx[3 * r];
                const dtype x = pl[0] - pr[0], y = pl[1] - pr[1], z = pl[2] - pr[2];
                dr[r] = std::sqrt(x * x + y * y + z * z);
                aoa[o_slice + r] = std::atan2(y, x);
                eoa[o_slice + r] = (dr[r] < los_limit) ? zero : std::asin(z / dr[r]);
            }
            for (arma::uword t = 0; t < n_tx; ++t)
            {
                const dtype *pt = &p_tx[3 * t];
                const dtype x = pf[0] - pt[0], y = pf[1] - pt[1], z = pf[2] - pt[2];
                const dtype dt = std::sqrt(x * x + y * y + z * z), at = std::atan2(y, x),
                            et = (dt < los_limit) ? zero : std::asin(z / dt);
                for (arma::uword r = 0; r < n_rx; ++r)
                {
                    const arma::uword io = o_slice + t * n_rx + r;
                    aod[io] = at;
                    eod[io] = et;
                    aoa[io] = aoa[o_slice + r];
                    eoa[io] = eoa[o_slice + r];
                    len[io] = dt + d_fbs_lbs + dr[r];
                }
            }
        }
    }

    // Element indices and local positions of the links
    i_tx.set_size(n_links), i_rx.set_size(n_links);
    tx_pos_interp.zeros(3, n_links), rx_pos_interp.zeros(3, n_links);
    const dtype *ptl = tx_el.n_elem != 0 ? tx_el.memptr() : nullptr, *prl = rx_el.n_elem != 0 ? rx_el.memptr() : nullptr;
    for (arma::uword t = 0; t < n_tx; ++t)
        for (arma::uword r = 0; r < n_rx; ++r)
        {
            const arma::uword R = t * n_rx + r;
            i_tx[R] = t, i_rx[R] = r;
            if (ptl != nullptr)
                std::memcpy(tx_pos_interp.colptr(R), &ptl[3 * t], 3 * sizeof(dtype));
            if (prl != nullptr)
                std::memcpy(rx_pos_interp.colptr(R), &prl[3 * r], 3 * sizeof(dtype));
        }

    return true_los_path;
}

// True if the port weights (port_weights_re [n_el, n_ports] or empty = identity, port_weights_im of the same size or empty =
// zero) is the identity within 1e-6 per entry, so that the channel generators can skip it
template <typename dtype>
static inline bool qd_port_weights_is_identity(const arma::Mat<dtype> &port_weights_re, const arma::Mat<dtype> &port_weights_im)
{
    constexpr dtype limit = (dtype)1.0e-6, one = (dtype)1.0;
    const arma::uword n = port_weights_re.n_rows;
    if (port_weights_re.n_elem != 0 && port_weights_re.n_cols != n)
        return false;
    const dtype *p = port_weights_re.memptr();
    for (arma::uword c = 0; c < port_weights_re.n_cols; ++c)
        for (arma::uword r = 0; r < n; ++r)
            if (std::abs(p[c * n + r] - (r == c ? one : (dtype)0.0)) > limit)
                return false;
    for (const dtype v : port_weights_im)
        if (std::abs(v) > limit)
            return false;
    return true;
}

// Applies the port weights to one path of the channel generators: coefficients C_out = A_rx^T * C_in * A_tx
// with the complex port weights (rx_re/rx_im [n_rx, n_rx_ports], tx_re/tx_im [n_tx, n_tx_ports], nullptr =
// identity / zero), delays D_out = P_rx^T * D_in * P_tx with the column-normalized port-weight powers (rx_pwr, tx_pwr,
// see qd_power_mat)
// - Inputs cr_in, ci_in, dl_in [n_rx, n_tx]; outputs cr_out, ci_out, dl_out [n_rx_ports, n_tx_ports]
// - The outputs may be the inputs (in place, when the port counts equal the element counts); the result then goes
//   through 'temp', which is resized as needed
template <typename dtype>
static inline void qd_port_weights_apply_slot(const dtype *rx_re, const dtype *rx_im, const dtype *tx_re, const dtype *tx_im,
                                          const dtype *rx_pwr, const dtype *tx_pwr,
                                          size_t n_rx, size_t n_rx_ports, size_t n_tx, size_t n_tx_ports,
                                          const dtype *cr_in, const dtype *ci_in, const dtype *dl_in,
                                          dtype *cr_out, dtype *ci_out, dtype *dl_out, std::vector<dtype> &temp)
{
    const size_t n_ports = n_rx_ports * n_tx_ports;
    if (cr_out != cr_in)
    {
        qd_multiply_3_complex_mat(rx_re, rx_im, cr_in, ci_in, tx_re, tx_im, cr_out, ci_out, n_rx, n_rx_ports, n_tx, n_tx_ports);
        qd_multiply_3_mat(rx_pwr, dl_in, tx_pwr, dl_out, n_rx, n_rx_ports, n_tx, n_tx_ports);
        return;
    }
    if (temp.size() < 2 * n_ports)
        temp.resize(2 * n_ports);
    dtype *tX = temp.data(), *tY = tX + n_ports;
    qd_multiply_3_complex_mat(rx_re, rx_im, cr_in, ci_in, tx_re, tx_im, tX, tY, n_rx, n_rx_ports, n_tx, n_tx_ports);
    std::memcpy(cr_out, tX, n_ports * sizeof(dtype));
    std::memcpy(ci_out, tY, n_ports * sizeof(dtype));
    qd_multiply_3_mat(rx_pwr, dl_in, tx_pwr, tX, n_rx, n_rx_ports, n_tx, n_tx_ports);
    std::memcpy(dl_out, tX, n_ports * sizeof(dtype));
}

// Checks of the "_vec" channel functions, before any output is changed: positions and orientations [3, 1] or
// [3, n_snap], antennas and kernel (also without snapshots; the single call checks them again, from a cache).
// 'n_path(s)' gives the number of input paths of snapshot s. Returns whether the snapshot loop runs in parallel: from
// a work of 64 coefficients (element pairs times output paths, plus 8 per call for its fixed cost; measured on one
// thread: 3.2 us per call and 0.35 us per coefficient). 8 SISO snapshots with one path each (work 80) run 2x
// (spherical) and 2.9x (planar) faster than on one thread, 4 snapshots 1.3x (spherical).
template <typename dtype, typename N>
static inline bool qd_channels_vec_check(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array,
                                  const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                  const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                  size_t n_snap, N &&n_path, int use_kernel)
{
    if (tx_pos.n_rows != 3ULL || tx_orientation.n_rows != 3ULL || rx_pos.n_rows != 3ULL || rx_orientation.n_rows != 3ULL)
        throw std::invalid_argument("Inputs 'tx_pos', 'tx_orientation', 'rx_pos' and 'rx_orientation' must have 3 rows.");

    for (const arma::Mat<dtype> *m : {&tx_pos, &tx_orientation, &rx_pos, &rx_orientation})
        if (m->n_cols != 1ULL && m->n_cols != n_snap)
            throw std::invalid_argument("Inputs 'tx_pos', 'tx_orientation', 'rx_pos' and 'rx_orientation' must have 1 or 'n_snap' columns.");

    qd_interpolate_pattern_kernel(use_kernel, std::max(tx_array.e_theta_re.n_elem, rx_array.e_theta_re.n_elem));
    qd_arrayant_check_tx_rx(tx_array, rx_array);

    const arma::uword n_links = tx_array.e_theta_re.n_slices * rx_array.e_theta_re.n_slices;
    arma::uword n_work = 0ULL;
    for (size_t s = 0; s < n_snap; ++s)
        n_work += ((arma::uword)n_path(s) + 1ULL) * n_links + 8ULL;
    return n_snap > 1 && n_work >= 64;
}

// Column s of a position or orientation with 1 or n_snap columns, aliased without copying
template <typename dtype>
static inline arma::Mat<dtype> qd_channels_vec_col(const arma::Mat<dtype> &m, size_t s)
{
    return arma::Mat<dtype>(const_cast<dtype *>(m.colptr(m.n_cols == 1ULL ? 0ULL : s)), 3, 1, false, true);
}

// Calls 'call(s, coeff_re_s, coeff_im_s)' for the snapshots s = 0 ... n_snap - 1 with qd_batch_loop (prefix "Snapshot").
// - Without 'coeff', the split outputs are the entries of 'coeff_re' / 'coeff_im' (resized to n_snap).
// - With 'coeff', each thread computes into its own split cubes and interleaves them into the entry of 'coeff';
//   'coeff_re' and 'coeff_im' are left empty. Entries that already have the output size are written in place.
// - Inside the parallel region, parallel regions of 'call' run on one thread (one active level by default).
template <typename dtype, typename F>
static inline void qd_channels_snapshot_loop(size_t n_snap, bool parallel,
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

    using split_cubes = std::pair<arma::Cube<dtype>, arma::Cube<dtype>>; // Scratch of the complex output
    qd_batch_loop<split_cubes>(n_snap, parallel, "Snapshot", [&](size_t s, split_cubes &split)
                               {
        if (coeff == nullptr)
        {
            call(s, coeff_re[s], coeff_im[s]);
            return;
        }
        arma::Cube<dtype> &cr = split.first, &ci = split.second;
        call(s, cr, ci);
        arma::Cube<std::complex<dtype>> &c = (*coeff)[s];
        c.set_size(cr.n_rows, cr.n_cols, cr.n_slices);
        std::complex<dtype> *p = c.memptr();
        const dtype *p_re = cr.memptr(), *p_im = ci.memptr();
        for (arma::uword k = 0; k < cr.n_elem; ++k)
            p[k] = std::complex<dtype>(p_re[k], p_im[k]); });
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
