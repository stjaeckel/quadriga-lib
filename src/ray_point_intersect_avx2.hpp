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

#ifndef ray_point_intersect_avx2_H
#define ray_point_intersect_avx2_H

#include <cstddef>
#include <vector>

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

#endif