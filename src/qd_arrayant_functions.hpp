// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qd_arrayant_qdant_H
#define qd_arrayant_qdant_H

#include <armadillo>
#include <string>

// Returns empty string if there was no error or an error message otherwise
template <typename dtype> // float or double
std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                   std::string *name,
                                   arma::Cube<dtype> *e_theta_re,
                                   arma::Cube<dtype> *e_theta_im,
                                   arma::Cube<dtype> *e_phi_re,
                                   arma::Cube<dtype> *e_phi_im,
                                   arma::Col<dtype> *azimuth_grid,
                                   arma::Col<dtype> *elevation_grid,
                                   arma::Mat<dtype> *element_pos,
                                   arma::Mat<dtype> *coupling_re,
                                   arma::Mat<dtype> *coupling_im,
                                   dtype *center_frequency,
                                   arma::Mat<unsigned> *layout);

// Returns empty string if there was no error or an error message otherwise
template <typename dtype> // float or double
std::string qd_arrayant_qdant_write(const std::string fn, const int id,
                                    const std::string *name,
                                    const arma::Cube<dtype> *e_theta_re,
                                    const arma::Cube<dtype> *e_theta_im,
                                    const arma::Cube<dtype> *e_phi_re,
                                    const arma::Cube<dtype> *e_phi_im,
                                    const arma::Col<dtype> *azimuth_grid,
                                    const arma::Col<dtype> *elevation_grid,
                                    const arma::Mat<dtype> *element_pos,
                                    const arma::Mat<dtype> *coupling_re,
                                    const arma::Mat<dtype> *coupling_im,
                                    const dtype *center_frequency,
                                    const arma::Mat<unsigned> *layout,
                                    unsigned *id_in_file);

// Block-diagonal coupling [C1 0; 0 C2] of two arrays with n_el1 and n_el2 elements (arrayant::append and
// arrayant_concat_multi). An empty coupling_re is an identity block (one port per element); an empty coupling_im is
// zero. Outputs are set to [n_el1 + n_el2, n_ports1 + n_ports2]; they must not alias the inputs.
template <typename dtype>
inline void qd_arrayant_coupling_block_diag(const arma::Mat<dtype> &coupling_re1, const arma::Mat<dtype> &coupling_im1,
                                            arma::uword n_el1,
                                            const arma::Mat<dtype> &coupling_re2, const arma::Mat<dtype> &coupling_im2,
                                            arma::uword n_el2,
                                            arma::Mat<dtype> &coupling_re, arma::Mat<dtype> &coupling_im)
{
    const arma::uword n_ports1 = coupling_re1.is_empty() ? n_el1 : coupling_re1.n_cols;
    const arma::uword n_ports2 = coupling_re2.is_empty() ? n_el2 : coupling_re2.n_cols;
    coupling_re.zeros(n_el1 + n_el2, n_ports1 + n_ports2);
    coupling_im.zeros(n_el1 + n_el2, n_ports1 + n_ports2);

    if (n_el1 != 0 && n_ports1 != 0)
    {
        if (coupling_re1.is_empty())
            coupling_re.submat(0, 0, n_el1 - 1, n_ports1 - 1).eye();
        else
            coupling_re.submat(0, 0, n_el1 - 1, n_ports1 - 1) = coupling_re1;
        if (!coupling_im1.is_empty())
            coupling_im.submat(0, 0, n_el1 - 1, n_ports1 - 1) = coupling_im1;
    }
    if (n_el2 != 0 && n_ports2 != 0)
    {
        const arma::uword r1 = n_el1 + n_el2 - 1, c1 = n_ports1 + n_ports2 - 1;
        if (coupling_re2.is_empty())
            coupling_re.submat(n_el1, n_ports1, r1, c1).eye();
        else
            coupling_re.submat(n_el1, n_ports1, r1, c1) = coupling_re2;
        if (!coupling_im2.is_empty())
            coupling_im.submat(n_el1, n_ports1, r1, c1) = coupling_im2;
    }
}

#endif