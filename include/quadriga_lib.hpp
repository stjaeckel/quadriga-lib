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

#ifndef quadriga_lib_H
#define quadriga_lib_H

#include <armadillo>
#include <string>
#include <vector>
#include <any>

#include "quadriga_arrayant.hpp"
#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

#define QUADRIGA_LIB_VERSION v0_2_0

namespace quadriga_lib
{
    // Returns the version number as a string in format (x.y.z)
    std::string quadriga_lib_version();

    // Check if AVX2 is supported
    bool quadriga_lib_has_AVX2();
}

#endif
