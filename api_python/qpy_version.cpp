// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.


#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# version
Returns the quadriga-lib version number

## Usage:
```
version = quadriga_lib.version();
```

## Caveat:
- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  an suffix `_AVX2` is added after the version number
MD!*/

std::string version()
{
    std::string quadriga_lib_version = quadriga_lib::quadriga_lib_version();

    if (quadriga_lib::quadriga_lib_has_AVX2())
        quadriga_lib_version += "_AVX2";

    return quadriga_lib_version;
}