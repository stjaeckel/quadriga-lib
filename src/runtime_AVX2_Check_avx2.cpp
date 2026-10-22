// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#if defined(_MSC_VER) // Windows
#include <intrin.h>
#include <immintrin.h>
#else // Linux
#include <cpuid.h>
#endif

// The AVX2 kernels also use FMA and need the OS to save the YMM registers. This file is compiled with AVX2 flags, so
// the detection uses only scalar integer code, which runs on any x86-64 CPU.
static bool qd_detect_AVX2_FMA()
{
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;

#if defined(_MSC_VER) // Windows
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7)
        return false;
    __cpuid(info, 1);
    ecx = (unsigned)info[2];
#else // Linux
    if (__get_cpuid_max(0, nullptr) < 7)
        return false;
    __cpuid(1, eax, ebx, ecx, edx);
#endif

    const bool fma = (ecx & (1u << 12)) != 0;     // CPUID.1:ECX bit 12
    const bool osxsave = (ecx & (1u << 27)) != 0; // CPUID.1:ECX bit 27, XGETBV is available
    const bool avx = (ecx & (1u << 28)) != 0;     // CPUID.1:ECX bit 28
    if (!fma || !osxsave || !avx)
        return false;

    // XCR0 bits 1 (SSE) and 2 (YMM): the OS saves the YMM state
#if defined(_MSC_VER) // Windows
    const unsigned long long xcr0 = _xgetbv(0);
#else // Linux
    unsigned xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
#endif
    if ((xcr0 & 0x6ULL) != 0x6ULL)
        return false;

#if defined(_MSC_VER) // Windows
    __cpuidex(info, 7, 0);
    ebx = (unsigned)info[1];
#else // Linux
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif
    return (ebx & (1u << 5)) != 0; // CPUID.(7,0):EBX bit 5 = AVX2
}

bool runtime_AVX2_Check()
{
    static const bool avx2_fma_ok = qd_detect_AVX2_FMA(); // Detected once
    return avx2_fma_ok;
}
