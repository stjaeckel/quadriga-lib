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

#endif