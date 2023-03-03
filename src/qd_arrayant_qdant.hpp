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

template <typename dataType> // float or double
std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                   std::string *name,
                                   arma::Cube<dataType> *e_theta_re, arma::Cube<dataType> *e_theta_im,
                                   arma::Cube<dataType> *e_phi_re, arma::Cube<dataType> *e_phi_im,
                                   arma::Col<dataType> *azimuth_grid, arma::Col<dataType> *elevation_grid,
                                   arma::Mat<dataType> *element_pos,
                                   arma::Mat<dataType> *coupling_re, arma::Mat<dataType> *coupling_im,
                                   dataType *center_frequency,
                                   arma::Mat<unsigned> *layout);
#endif