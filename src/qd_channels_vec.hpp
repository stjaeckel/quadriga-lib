// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Checks and snapshot loop of the "_vec" channel functions (internal)

#ifndef qd_channels_vec_H
#define qd_channels_vec_H

#include <armadillo>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"

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

#endif
