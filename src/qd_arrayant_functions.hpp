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

#ifndef qd_arrayant_qdant_H
#define qd_arrayant_qdant_H

#include <armadillo>
#include <string>

template <typename dtype> // float or double
void qd_arrayant_interpolate(const arma::Cube<dtype> *e_theta_re, const arma::Cube<dtype> *e_theta_im,
                             const arma::Cube<dtype> *e_phi_re, const arma::Cube<dtype> *e_phi_im,
                             const arma::Col<dtype> *azimuth_grid, const arma::Col<dtype> *elevation_grid,
                             const arma::Mat<dtype> *azimuth, const arma::Mat<dtype> *elevation,
                             const arma::Col<unsigned> *i_element, const arma::Cube<dtype> *orientation,
                             const arma::Mat<dtype> *element_pos,
                             arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                             arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                             arma::Mat<dtype> *dist,
                             arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc, arma::Mat<dtype> *gamma);

template <typename dtype> // float or double
std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                   std::string *name,
                                   arma::Cube<dtype> *e_theta_re, arma::Cube<dtype> *e_theta_im,
                                   arma::Cube<dtype> *e_phi_re, arma::Cube<dtype> *e_phi_im,
                                   arma::Col<dtype> *azimuth_grid, arma::Col<dtype> *elevation_grid,
                                   arma::Mat<dtype> *element_pos,
                                   arma::Mat<dtype> *coupling_re, arma::Mat<dtype> *coupling_im,
                                   dtype *center_frequency,
                                   arma::Mat<unsigned> *layout);

template <typename dtype> // float or double
std::string qd_arrayant_qdant_write(const std::string fn, const int id,
                                    const std::string *name,
                                    const arma::Cube<dtype> *e_theta_re, const arma::Cube<dtype> *e_theta_im,
                                    const arma::Cube<dtype> *e_phi_re, const arma::Cube<dtype> *e_phi_im,
                                    const arma::Col<dtype> *azimuth_grid, const arma::Col<dtype> *elevation_grid,
                                    const arma::Mat<dtype> *element_pos,
                                    const arma::Mat<dtype> *coupling_re, const arma::Mat<dtype> *coupling_im,
                                    const dtype *center_frequency,
                                    const arma::Mat<unsigned> *layout,
                                    unsigned *id_in_file);

#endif