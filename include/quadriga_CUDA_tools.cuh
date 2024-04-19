// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#ifndef quadriga_cuda_tools_H
#define quadriga_cuda_tools_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

namespace quadriga_lib
{
    // Returns the compute capability of the NVIDIA GPU
    // - If communication with the GPU was unsuccessful, a value of 0.0 is returned
    double get_CUDA_compute_capability(int device = 0);

}

#endif
