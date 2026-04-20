// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Internal header shared by all quadriga-lib CUDA translation units.
// Do NOT include in public headers.

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " — " +   \
                                     cudaGetErrorString(err));                  \
    } while (0)
