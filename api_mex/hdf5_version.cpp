// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_VERSION
Return the HDF5 library version string

- Reports the HDF5 C library version that quadriga-lib was compiled against, taken from the
  HDF5 header macros at compile time
- Useful for diagnosing binary/library mismatches when loading or writing channel files

## Usage:
```
version = quadriga_lib.hdf5_version;
```

## Outputs:
- **`version`** — HDF5 version string in the format `"x.y.z"` (e.g. `"1.12.2"`); string
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)prhs;

    // Validate argument counts
    if (nrhs != 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string version;
    CALL_QD(version = quadriga_lib::get_HDF5_version());

    plhs[0] = mxCreateString(version.c_str());
}
