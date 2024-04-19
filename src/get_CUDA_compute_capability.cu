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

#include "quadriga_CUDA_tools.cuh"

// KERNEL: Add two numbers
__global__ void Add_A_and_B(float *d_a, float *d_b, float *d_c)
{
    d_c[0] = d_a[0] + d_b[0];
}

// Returns the compute capability of the NVIDIA GPU
double quadriga_lib::get_CUDA_compute_capability(int device)
{
    // Initialize CUDA Error
    cudaError_t error;

    // Connect which GPU to run on
    error = cudaSetDevice(device);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    // Host variables
    float h_a = 3.0, h_b = 7.0;

    // Pointer to device variables
    float *d_a, *d_b, *d_c;
    size_t sz = sizeof(float);

    // Allocate memory on device
    error = cudaMalloc(&d_a, sz);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    error = cudaMalloc(&d_b, sz);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    error = cudaMalloc(&d_c, sz);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    // Copy data to device
    error = cudaMemcpy(d_a, &h_a, sz, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    error = cudaMemcpy(d_b, &h_b, sz, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    // Perform simple computation on the device
    Add_A_and_B<<<1, 1>>>(d_a, d_b, d_c);

    // Check for error
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    // Retrieve data from device
    float *h_c = new float[1];
    error = cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }
    if (h_c[0] != 10)
    {
        cudaDeviceReset();
        return 0.0;
    }
    delete[] h_c;

    int *val = new int[1];
    error = cudaDeviceGetAttribute(val, cudaDevAttrComputeCapabilityMinor, 0);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    double cc = (double)val[0];
    error = cudaDeviceGetAttribute(val, cudaDevAttrComputeCapabilityMajor, 0);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return 0.0;
    }

    cc = cc * 0.1 + (double)val[0];
    delete[] val;

    cudaDeviceReset();
    return cc;
}
