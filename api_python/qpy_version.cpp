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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# VERSION
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