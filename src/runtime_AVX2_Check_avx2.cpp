// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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