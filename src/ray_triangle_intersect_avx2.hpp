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

#ifndef ray_triangle_intersect_avx2_H
#define ray_triangle_intersect_avx2_H

#include <cstddef>

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

#endif
