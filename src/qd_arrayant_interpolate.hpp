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

#ifndef qd_arrayant_interpolate_H
#define qd_arrayant_interpolate_H

#include "quadriga_lib.hpp"

template <typename dataType> // float or double
void qd_arrayant_interpolate(const quadriga_lib::arrayant<dataType> *ant,
                              const arma::Mat<dataType> *azimuth, const arma::Mat<dataType> *elevation,
                              const arma::Col<unsigned> *i_element, const arma::Cube<dataType> *orientation,
                              const arma::Mat<dataType> *element_pos,
                              arma::Mat<dataType> *V_re, arma::Mat<dataType> *V_im,
                              arma::Mat<dataType> *H_re, arma::Mat<dataType> *H_im,
                              arma::Mat<dataType> *dist,
                              arma::Mat<dataType> *azimuth_loc, arma::Mat<dataType> *elevation_loc);

#endif
