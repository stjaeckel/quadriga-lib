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

#include "mex.h"
#include "quadriga_CUDA_tools.cuh"
#include "mex_helper_functions.cpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# GET_CUDA_COMPUTE_CAPABILITY
Returns the compute capability of the CUDA-capable NVIDIA GPU

## Usage:
```
cc = quadriga_lib.get_CUDA_compute_capability;
```

## Caveat:
- This function will only be available if Quadriga-Lib was compiled with CUDA support
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 0)
    {
        mexErrMsgIdAndTxt("quadriga_lib:get_CUDA_compute_capability:no_input", "Incorrect number of input arguments.");
        if ( mxIsSingle(prhs[0])) // Useless line to avoid error with flags "-Werror=unused-but-set-variable"
            mexErrMsgIdAndTxt("quadriga_lib:get_CUDA_compute_capability:no_input", "Oops.");
    }

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:version:no_output", "Incorrect number of output arguments.");

    double cc = quadriga_lib::get_CUDA_compute_capability();
    plhs[0] = qd_mex_copy2matlab(&cc);
}
