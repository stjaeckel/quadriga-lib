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

#include <vector>

#if defined(_MSC_VER) // Windows
#include <intrin.h>
#else // Linux
#include <cpuid.h>
#endif

bool runtime_AVX2_Check()
{
    std::vector<int> cpuidInfo(4);

#if defined(_MSC_VER) // Windows
    __cpuidex(cpuidInfo.data(), 7, 0);
#else // Linux
    __cpuid_count(7, 0, cpuidInfo[0], cpuidInfo[1], cpuidInfo[2], cpuidInfo[3]);
#endif

    return (cpuidInfo[1] & (1 << 5)) != 0; // Check the AVX2 bit in EBX
}