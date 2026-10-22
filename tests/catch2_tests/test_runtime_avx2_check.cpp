// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"

#ifdef BUILD_WITH_AVX2
bool runtime_AVX2_Check(); // src/runtime_AVX2_Check_avx2.cpp
#endif

TEST_CASE("quadriga_lib_has_AVX2 - agrees with the CPU features")
{
    // The AVX2 kernels need AVX2, FMA and an OS that saves the YMM state; the check tests all of them, so it returns
    // false (instead of faulting in the FMA test kernel) on a CPU or VM without one of them
#ifdef BUILD_WITH_AVX2
#if defined(__GNUC__)
    __builtin_cpu_init();
    const bool cpu = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    const bool cpu = runtime_AVX2_Check();
#endif
    CHECK(runtime_AVX2_Check() == cpu);
    CHECK(runtime_AVX2_Check() == cpu); // cached result
    CHECK(quadriga_lib::quadriga_lib_has_AVX2() == cpu);
#else
    CHECK(!quadriga_lib::quadriga_lib_has_AVX2());
#endif
}
