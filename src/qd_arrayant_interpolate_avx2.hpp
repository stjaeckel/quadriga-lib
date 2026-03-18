// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#ifndef qd_arrayant_interpolate_avx2_H
#define qd_arrayant_interpolate_avx2_H

#include <armadillo>

template <typename dtype>
void qd_arrayant_interpolate_avx2(const arma::Cube<dtype> &e_theta_re,       // Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Cube<dtype> &e_theta_im,       // Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Cube<dtype> &e_phi_re,         // Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Cube<dtype> &e_phi_im,         // Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Col<dtype> &azimuth_grid,      // Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
                                  const arma::Col<dtype> &elevation_grid,    // Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
                                  const arma::Mat<dtype> &azimuth,           // Azimuth angles for interpolation in [rad],                      Size [1, n_ang] or [n_out, n_ang]
                                  const arma::Mat<dtype> &elevation,         // Elevation angles for interpolation in [rad],                    Size [1, n_ang] or [n_out, n_ang]
                                  const arma::Col<unsigned> &i_element,      // Element indices, 1-based                                        Vector of length "n_out"
                                  const arma::Cube<dtype> &orientation,      // Orientation of the array antenna (bank, tilt, head) in [rad],   Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
                                  const arma::Mat<dtype> &element_pos,       // Element positions                                               Size [3, n_out]
                                  arma::Mat<dtype> &V_re,                    // Interpolated vertical field, real part,                         Size [n_out, n_ang]
                                  arma::Mat<dtype> &V_im,                    // Interpolated vertical field, imaginary part,                    Size [n_out, n_ang]
                                  arma::Mat<dtype> &H_re,                    // Interpolated horizontal field, real part,                       Size [n_out, n_ang]
                                  arma::Mat<dtype> &H_im,                    // Interpolated horizontal field, imaginary part,                  Size [n_out, n_ang]
                                  arma::Mat<dtype> *dist = nullptr,          // Effective distances, optional                                   Size [n_out, n_ang] or []
                                  arma::Mat<dtype> *azimuth_loc = nullptr,   // Azimuth angles [rad] in local antenna coordinates, optional,    Size [n_out, n_ang] or []
                                  arma::Mat<dtype> *elevation_loc = nullptr, // Elevation angles [rad] in local antenna coordinates, optional,  Size [n_out, n_ang] or []
                                  arma::Mat<dtype> *gamma = nullptr);        // Polarization rotation angles in [rad], optional,                Size [n_out, n_ang] or []

#endif