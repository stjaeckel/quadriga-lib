// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#include <cuda_runtime.h>
#include "quadriga_lib_cuda_functions.hpp"

// Simple CUDA kernel: z[i] = X[i] * X[i+8] + 2.0f  (same FMA as the AVX2 test)
__global__ void qd_test_kernel(const float *X, float *Z)
{
    int i = threadIdx.x;
    if (i < 8)
        Z[i] = X[i] * X[i + 8] + 2.0f;
}

// Check if a CUDA-capable GPU is available at runtime
bool runtime_CUDA_Check()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

// Perform a simple test calculation on the GPU
void qd_TEST_CUDA(const float *X, // Host memory, 16 floats
                  float *Z)       // Host memory, 8 floats
{
    float *d_X = nullptr, *d_Z = nullptr;

    // Allocate device memory
    cudaMalloc(&d_X, 16 * sizeof(float));
    cudaMalloc(&d_Z, 8 * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_X, X, 16 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block, 8 threads
    qd_test_kernel<<<1, 8>>>(d_X, d_Z);

    // Copy result back to host
    cudaDeviceSynchronize();
    cudaMemcpy(Z, d_Z, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_Z);
}