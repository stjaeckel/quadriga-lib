// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

// A collection of small helper functions to reduce copy and pasting code

#ifndef quadriga_lib_helper_H
#define quadriga_lib_helper_H

#include <cstring>

template <typename dtype>
static bool qd_in_range(const dtype *data, size_t n_elem, dtype min, dtype max, bool inclusive = true, bool check_sorted = false)
{
    if (inclusive)
    {
        for (size_t i = 0ULL; i < n_elem; ++i)
        {
            if (data[i] < min || data[i] > max)
                return false;
            if (check_sorted && i != 0ULL && data[i] <= data[i - 1])
                return false;
        }
    }
    else
    {
        for (size_t i = 0ULL; i < n_elem; ++i)
        {
            if (data[i] <= min || data[i] >= max)
                return false;
            if (check_sorted && i != 0ULL && data[i] <= data[i - 1])
                return false;
        }
    }
    return true;
}

#endif