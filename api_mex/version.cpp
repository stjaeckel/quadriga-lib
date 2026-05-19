// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# VERSION
Returns the quadriga-lib version number

- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  a suffix `_AVX2` is added after the version number
- If Quadriga-Lib was compiled with CUDA support and a CUDA-capable GPU is available,
  a suffix `_CUDA` is added after the version number

## Usage:
```
version = quadriga_lib.version;
```

## Outputs:
- **`version`** — Version string (e.g. "0.11.5_AVX2_CUDA")
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 0)
    {
        mexErrMsgIdAndTxt("quadriga_lib:version:no_input", "Incorrect number of input arguments.");
        if ( mxIsSingle(prhs[0])) // Useless line to avoid error with flags "-Werror=unused-but-set-variable"
            mexErrMsgIdAndTxt("quadriga_lib:version:no_input", "Oops.");
    }

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:version:no_output", "Incorrect number of output arguments.");

    std::string quadriga_lib_version = quadriga_lib::quadriga_lib_version();

    if (quadriga_lib::quadriga_lib_has_AVX2())
        quadriga_lib_version += "_AVX2";

    if (quadriga_lib::quadriga_lib_has_CUDA())
        quadriga_lib_version += "_CUDA";

    plhs[0] = mxCreateString(quadriga_lib_version.c_str());
}