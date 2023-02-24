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
    template <typename dataType> // float or double
    arma::cube calc_rotation_matrix(const arma::Cube<dataType> orientation, bool invert_y_axis, bool transposeR);

    template <typename dataType> // float or double
    arma::cube geo2cart(const arma::Mat<dataType> azimuth, const arma::Mat<dataType> elevation, const arma::Mat<dataType> length);

    template <typename dataType> // float or double
    arma::cube cart2geo(const arma::Cube<dataType> cart);
}

#endif
