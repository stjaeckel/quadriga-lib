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

// A collection of AVX2 accelerated functions

#ifndef quadriga_lib_avx2fun_H
#define quadriga_lib_avx2fun_H

#include <cstddef>
#include <vector>

// Perform a simple test calculation
void qd_TEST_AVX2(const float *X, // Aligned memory, 16 floats
                  float *Z);      // Aligned memory, 8 floats

// Ray-Triangle Intersection test with BVH acceleration
void qd_RTI_AVX2(const float *Tx, const float *Ty, const float *Tz,    // First vertex coordinate in GCS, aligned to 32 byte, length n_mesh
                 const float *E1x, const float *E1y, const float *E1z, // Edge 1 from first vertex to second vertex, aligned to 32 byte, length n_mesh
                 const float *E2x, const float *E2y, const float *E2z, // Edge 2 from first vertex to third vertex, aligned to 32 byte, length n_mesh
                 const size_t n_mesh,                                  // Number of triangles (multiple of VEC_SIZE)
                 const unsigned *SMI,                                  // List of sub-mesh indices, length n_sub
                 const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, aligned to 32 byte, length n_sub_s
                 const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, aligned to 32 byte, length n_sub_s
                 const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, aligned to 32 byte, length n_sub_s
                 const size_t n_sub,                                   // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                 const float *Ox, const float *Oy, const float *Oz,    // Ray origin in GCS, length n_ray
                 const float *Dx, const float *Dy, const float *Dz,    // Vector from ray origin to ray destination, length n_ray
                 const size_t n_ray,                                   // Number of rays
                 float *Wf,                                            // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                 float *Ws,                                            // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                 unsigned *If,                                         // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                 unsigned *Is,                                         // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                 unsigned *hit_cnt = nullptr);                         // Number of hits between orig and dest, length n_ray, uninitialized, optional

// AVX2 accelerated implementation of RayPointIntersect
void qd_RPI_AVX2(const float *Px, const float *Py, const float *Pz,    // Point coordinates, aligned to 32 byte, length n_point
                 const size_t n_point,                                 // Number of points
                 const unsigned *SCI,                                  // List of sub-cloud indices, length n_sub
                 const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, aligned to 32 byte, length n_sub_s
                 const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, aligned to 32 byte, length n_sub_s
                 const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, aligned to 32 byte, length n_sub_s
                 const size_t n_sub,                                   // Number of sub-clouds (not aligned, i.e. n_sub <= n_sub_s)
                 const float *T1x, const float *T1y, const float *T1z, // First ray vertex coordinate in GCS, length n_ray
                 const float *T2x, const float *T2y, const float *T2z, // Second ray vertex coordinate in GCS, length n_ray
                 const float *T3x, const float *T3y, const float *T3z, // Third ray vertex coordinate in GCS, length n_ray
                 const float *Nx, const float *Ny, const float *Nz,    // Ray tube normal vector, length n_ray
                 const float *D1x, const float *D1y, const float *D1z, // First ray direction in GCS, length n_ray
                 const float *D2x, const float *D2y, const float *D2z, // Second ray direction in GCS, length n_ray
                 const float *D3x, const float *D3y, const float *D3z, // Third ray direction in GCS, length n_ray
                 const float *rD1, const float *rD2, const float *rD3, // Inverse Dot product of ray direction and normal vector
                 const size_t n_ray,                                   // Number of rays
                 std::vector<unsigned> *p_hit);

// AVX2 accelerated implementation of DFT
template <typename dtype>
void qd_DFT_AVX2(const dtype *CFr,       // Channel coefficients, real part, Size [n_ant, n_path]
                 const dtype *CFi,       // Channel coefficients, imaginary part, Size [n_ant, n_path]
                 const dtype *DL,        // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                 const size_t n_ant,     // Number of MIMO sub-links
                 const size_t n_path,    // Number multipath components
                 const bool planar_wave, // Indicator that same delays are used for all antennas
                 const float *phasor,    // Phasor, -pi/2 to pi/2, aligned to 32 byte, Size [ n_carrier ]
                 const size_t n_carrier, // Number of carriers, mutiple of 8
                 float *Hr,              // Channel matrix, real part, Size [ n_carrier, n_ant ]
                 float *Hi);             // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]

// Check for AVX2 availability at runtime
bool runtime_AVX2_Check();

#endif
