// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_lib_cuda_functions_H
#define quadriga_lib_cuda_functions_H

#include <cstddef>
#include <vector>

// Check if a CUDA-capable GPU is available at runtime
bool runtime_CUDA_Check();

// Perform a simple test calculation on the GPU
void qd_TEST_CUDA(const float *X, // Host memory, 16 floats
                  float *Z);      // Host memory, 8 floats

template <typename dtype>
void qd_RTI_CUDA(const dtype *Tx, const dtype *Ty, const dtype *Tz, // First vertex coordinate in GCS, length n_mesh
                 const dtype *Ux, const dtype *Uy, const dtype *Uz, // Second vertex coordinate in GCS, length n_mesh
                 const dtype *Vx, const dtype *Vy, const dtype *Vz, // Third vertex coordinate in GCS, length n_mesh
                 const size_t n_mesh,                               // Number of triangles (multiple of VEC_SIZE)
                 const unsigned *SMI,                               // List of sub-mesh indices, length n_sub
                 const dtype *Xmin, const dtype *Xmax,              // Minimum and maximum x-values of the AABB, length n_sub_s
                 const dtype *Ymin, const dtype *Ymax,              // Minimum and maximum y-values of the AABB, length n_sub_s
                 const dtype *Zmin, const dtype *Zmax,              // Minimum and maximum z-values of the AABB, length n_sub_s
                 const size_t n_sub,                                // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                 const dtype *Ox, const dtype *Oy, const dtype *Oz, // Ray origin in GCS, length n_ray
                 const dtype *Dx, const dtype *Dy, const dtype *Dz, // Ray destination in GCS, length n_ray
                 const size_t n_ray,                                // Number of rays
                 dtype *Wf,                                         // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                 dtype *Ws,                                         // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                 unsigned *If,                                      // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                 unsigned *Is,                                      // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                 unsigned *hit_cnt = nullptr,                       // Number of hits between orig and dest, length n_ray, uninitialized, optional
                 int gpu_id = 0);

void qd_RPI_CUDA(const float *Px, const float *Py, const float *Pz,    // Point coordinates, length n_point
                 const size_t n_point,                                 // Number of points
                 const unsigned *SCI,                                  // List of sub-cloud indices, length n_sub
                 const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, length n_sub_s
                 const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, length n_sub_s
                 const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, length n_sub_s
                 const size_t n_sub,                                   // Number of sub-clouds (n_sub <= n_sub_s)
                 const float *T1x, const float *T1y, const float *T1z, // First ray vertex coordinate in GCS, length n_ray
                 const float *T2x, const float *T2y, const float *T2z, // Second ray vertex coordinate in GCS, length n_ray
                 const float *T3x, const float *T3y, const float *T3z, // Third ray vertex coordinate in GCS, length n_ray
                 const float *Nx, const float *Ny, const float *Nz,    // Ray tube normal vector, length n_ray
                 const float *D1x, const float *D1y, const float *D1z, // First ray direction in GCS, length n_ray
                 const float *D2x, const float *D2y, const float *D2z, // Second ray direction in GCS, length n_ray
                 const float *D3x, const float *D3y, const float *D3z, // Third ray direction in GCS, length n_ray
                 const float *rD1, const float *rD2, const float *rD3, // Inverse Dot product of ray direction and normal vector
                 const size_t n_ray,                                   // Number of rays
                 std::vector<unsigned> *p_hit,                         // Output: Array of std::vector containing list of points that were hit by a ray, length n_ray
                 int gpu_id = 0);

#endif