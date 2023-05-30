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

#ifndef quadriga_tools_H
#define quadriga_tools_H

#include <armadillo>
#include <string>

namespace quadriga_tools
{
    template <typename dtype> // float or double
    arma::cube calc_rotation_matrix(const arma::Cube<dtype> orientation, bool invert_y_axis, bool transposeR);

    template <typename dtype> // float or double
    arma::cube geo2cart(const arma::Mat<dtype> azimuth, const arma::Mat<dtype> elevation, const arma::Mat<dtype> length);

    template <typename dtype> // float or double
    arma::cube cart2geo(const arma::Cube<dtype> cart);

    // 2D linear interpolation (returns error message or empty string in case of no error)
    template <typename dtype>                          // Supported types: float or double
    std::string interp(const arma::Cube<dtype> *input, // Input data; size [ ny, nx, ne ], ne = multiple data sets
                       const arma::Col<dtype> *xi,     // x sample points of input; vector length nx
                       const arma::Col<dtype> *yi,     // y sample points of input; vector length ny
                       const arma::Col<dtype> *xo,     // x sample points of output; vector length mx
                       const arma::Col<dtype> *yo,     // y sample points of output; vector length my
                       arma::Cube<dtype> *output);     // Interpolated data; size [ my, mx, ne ]

    // 1D linear interpolation (returns error message or empty string)
    template <typename dtype>                         // Supported types: float or double
    std::string interp(const arma::Mat<dtype> *input, // Input data; size [ nx, ne ], ne = multiple data sets
                       const arma::Col<dtype> *xi,    // x sample points of input; vector length nx
                       const arma::Col<dtype> *xo,    // x sample points of output; vector length mx
                       arma::Mat<dtype> *output);     // Interpolated data; size [ mx, ne ]

}

#endif
