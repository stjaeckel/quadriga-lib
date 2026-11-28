// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Declarations of the GENERIC kernels

#ifndef quadriga_lib_genericfun_H
#define quadriga_lib_genericfun_H

#include <cstddef>
#include <vector>

// Assemble a point-major CSR hit list from a set of packed hit buffers
// Each hit is a 64 bit record, the upper 32 bit hold the 0-based point index
// and the lower 32 bit the 0-based ray index. The buffers are concatenated in
// the given order, which also fixes the order of the ray indices within each
// point's block. Used by the GENERIC, AVX2 and CUDA implementations of
// RayPointIntersect, defined in ray_point_intersect_generic.cpp
void qd_RPI_assemble(const unsigned long long *const *hit_buffer, // Packed hit records, one pointer per buffer, length n_buffer
                     const size_t *n_hit_buffer,                  // Number of records in each buffer, length n_buffer
                     const size_t n_buffer,                       // Number of buffers, may be 0
                     const size_t n_point,                        // Number of points
                     std::vector<unsigned> *hit_ind,              // Output: flat list of ray indices grouped by point, resized to the total hit count
                     unsigned *hit_offset);                       // Output: 0-based start of each point's block, length n_point + 1

// Per-thread staging of packed hit records for RayPointIntersect, in chunks of fixed size
// Appending a chunk never moves the records already written, so the staged data is written exactly once and the peak
// memory is the payload size. Used by the GENERIC and AVX2 implementations. All members are defined in
// ray_point_intersect_generic.cpp, so the AVX2 file compiles no std::vector code (see quadriga_lib_avx2_functions.hpp).
class qd_RPI_staging
{
public:
    static constexpr size_t chunk_size = 16384; // Number of records per chunk

    explicit qd_RPI_staging(int n_threads);
    ~qd_RPI_staging();
    qd_RPI_staging(const qd_RPI_staging &) = delete;
    qd_RPI_staging &operator=(const qd_RPI_staging &) = delete;

    // Appends a chunk for thread i_thread and returns its first record, the chunk holds chunk_size records
    unsigned long long *new_chunk(size_t i_thread);

    // Marks the end of the records that thread i_thread wrote into its last chunk
    void close(size_t i_thread, const unsigned long long *p_end);

    // Zero-filled scratch space of n values for thread i_thread, kept until the object is destroyed
    int *scratch(size_t i_thread, size_t n);

    // Assembles the point-major CSR hit list from the chunks in thread order, which preserves the ray order
    void assemble(size_t n_point, std::vector<unsigned> *hit_ind, unsigned *hit_offset) const;

private:
    std::vector<std::vector<std::vector<unsigned long long>>> chunks; // [n_threads][n_chunks][chunk_size or less]
    std::vector<std::vector<int>> scratch_space;                      // [n_threads][n]
};

// Generic C++ implementation of RayTriangleIntersect
template <typename dtype>
void qd_RTI_GENERIC(const dtype *Tx, const dtype *Ty, const dtype *Tz, // First vertex coordinate in GCS, length n_mesh
                    const dtype *Ux, const dtype *Uy, const dtype *Uz, // Second vertex coordinate in GCS, length n_mesh
                    const dtype *Vx, const dtype *Vy, const dtype *Vz, // Third vertex coordinate in GCS, length n_mesh
                    const size_t n_mesh,                               // Number of triangles
                    const unsigned *SMI,                               // List of sub-mesh indices, length n_sub
                    const dtype *Xmin, const dtype *Xmax,              // Minimum and maximum x-values of the AABB, length n_sub
                    const dtype *Ymin, const dtype *Ymax,              // Minimum and maximum y-values of the AABB, length n_sub
                    const dtype *Zmin, const dtype *Zmax,              // Minimum and maximum z-values of the AABB, length n_sub
                    const size_t n_sub,                                // Number of sub-meshes
                    const dtype *Ox, const dtype *Oy, const dtype *Oz, // Ray origin in GCS, length n_ray
                    const dtype *Dx, const dtype *Dy, const dtype *Dz, // Ray destination in GCS, length n_ray
                    const size_t n_ray,                                // Number of rays
                    dtype *Wf,                                         // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                    dtype *Ws,                                         // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                    unsigned *If,                                      // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                    unsigned *Is,                                      // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                    unsigned *hit_cnt = nullptr);                      // Number of hits between orig and dest, length n_ray, uninitialized, optional

// Generic C++ implementation of RayPointIntersect
template <typename dtype>
void qd_RPI_GENERIC(const dtype *Px, const dtype *Py, const dtype *Pz,    // Point coordinates, length n_point
                    const size_t n_point,                                 // Number of points
                    const unsigned *SCI,                                  // List of sub-cloud indices, length n_sub
                    const dtype *Xmin, const dtype *Xmax,                 // Minimum and maximum x-values of the AABB, length n_sub
                    const dtype *Ymin, const dtype *Ymax,                 // Minimum and maximum y-values of the AABB, length n_sub
                    const dtype *Zmin, const dtype *Zmax,                 // Minimum and maximum z-values of the AABB, length n_sub
                    const size_t n_sub,                                   // Number of sub-clouds
                    const dtype *T1x, const dtype *T1y, const dtype *T1z, // First ray vertex coordinate in GCS, length n_ray
                    const dtype *T2x, const dtype *T2y, const dtype *T2z, // Second ray vertex coordinate in GCS, length n_ray
                    const dtype *T3x, const dtype *T3y, const dtype *T3z, // Third ray vertex coordinate in GCS, length n_ray
                    const dtype *Nx, const dtype *Ny, const dtype *Nz,    // Ray tube normal vector, length n_ray
                    const dtype *D1x, const dtype *D1y, const dtype *D1z, // First ray direction in GCS, length n_ray
                    const dtype *D2x, const dtype *D2y, const dtype *D2z, // Second ray direction in GCS, length n_ray
                    const dtype *D3x, const dtype *D3y, const dtype *D3z, // Third ray direction in GCS, length n_ray
                    const dtype *rD1, const dtype *rD2, const dtype *rD3, // 1 / dot(D, N) per ray vertex, length n_ray
                    const size_t n_ray,                                   // Number of rays
                    std::vector<unsigned> *hit_ind,                       // Output: flat list of 0-based ray indices grouped by point, resized by the kernel
                    unsigned *hit_offset);                                // Output: 0-based start of each point's block, length n_point + 1, allocated by the caller

// Generic C++ implementation of DFT
// The phase x = freq * delay (in cycles) is computed in double precision and reduced to x - round(x), which is in
// [-0.5, 0.5], before sin / cos are evaluated in single precision; coefficients and sums are single precision.
// Paths are processed one after the other, the carrier loop is innermost and branch-free (vectorizable): the rounding
// uses the magic-number method, exact for |x| < 2^51 and only without -ffast-math (which may fold it away); paths with
// larger or non-finite phases use a scalar loop.
template <typename dtype>
void qd_DFT_GENERIC(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
                    const dtype *__restrict CFi,    // Channel coefficients, imaginary part, Size [n_ant, n_path]
                    const dtype *__restrict DL,     // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                    const size_t n_ant,             // Number of MIMO sub-links
                    const size_t n_path,            // Number of multipath components
                    const bool planar_wave,         // Indicator that same delays are used for all antennas
                    const double *__restrict freq,  // Carrier frequencies -bandwidth * pilot_grid in [Hz], Length [ n_carrier ]
                    const size_t n_carrier,         // Number of carriers
                    float *__restrict Hr,           // Channel matrix, real part, Size [ n_carrier, n_ant ]
                    float *__restrict Hi);          // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]

// Sum-of-sinusoids evaluation, double precision
// val(p, g) = sum_n wc(n, g) · cos( 2·pi · t ) - ws(n, g) · sin( 2·pi · t ), t = fx(n)·px(p) + fy(n)·py(p) + fz(n)·pz(p)
// With a second device (gx != nullptr): t = (f(n)·p(p)) + (g(n)·q(p)), both parts computed alike, so that exchanging
// (f, p) with (g, q) gives the same value bit for bit
// The argument t is reduced to one period before the trigonometric functions are evaluated
void qd_SOS_GENERIC(const double *fx, const double *fy, const double *fz, // Frequencies in [1/m], length n_coeff
                    const double *wc, const double *ws,                   // Weights a·cos(phase), a·sin(phase), Size [n_coeff, n_gen]
                    const size_t n_coeff,                                 // Number of sinusoids
                    const size_t n_gen,                                   // Number of generators
                    const double *px, const double *py, const double *pz, // Positions in [m], length n_pos
                    const size_t n_pos,                                   // Number of positions
                    double *val,                                          // Output: Values, Size [n_pos, n_gen], uninitialized
                    const double *gx = nullptr, const double *gy = nullptr, const double *gz = nullptr,  // Second device: frequencies, length n_coeff
                    const double *qx = nullptr, const double *qy = nullptr, const double *qz = nullptr); // Second device: positions, length n_pos

// Interpolation of antenna array patterns (interpolate_pattern), scalar kernel in the precision of dtype
// Called through qd_interpolate_pattern_dispatch, which checks the preconditions stated below; the kernel does not.
// 'rotation' holds the transposed rotation matrices of the orientations (qd_rotation_matrix with transposeR = true),
// in column-major order.
template <typename dtype>
void qd_interpolate_pattern_GENERIC(const dtype *e_theta_re, const dtype *e_theta_im,       // e-theta field, real / imaginary part, [n_elevation, n_azimuth, n_elements]
                                     const dtype *e_phi_re, const dtype *e_phi_im,           // e-phi field, real / imaginary part, [n_elevation, n_azimuth, n_elements]
                                     const size_t n_elevation, const size_t n_azimuth,       // Number of elevation and azimuth samples, both >= 1
                                     const dtype *azimuth_grid,                              // Azimuth sample angles in rad, strictly increasing, [n_azimuth]
                                     const dtype *elevation_grid,                            // Elevation sample angles in rad, strictly increasing, [n_elevation]
                                     const dtype *azimuth, const dtype *elevation,           // Angles in GCS in rad, [n_angle_rows, n_ang]
                                     const size_t n_angle_rows,                              // 1 (same angles for all outputs) or n_out
                                     const size_t n_ang,                                     // Number of angles, 1 <= n_ang <= INT_MAX
                                     const dtype *rotation,                                  // Rotation matrices, 9 values per orientation column and slice, [9, n_rotation_cols, n_rotation_slices]
                                     const size_t n_rotation_cols,                           // 1 (same orientation for all outputs) or n_out
                                     const size_t n_rotation_slices,                         // 1 (same orientation for all angles) or n_ang
                                     const dtype *element_pos,                               // Element positions in m, [3, n_out]
                                     const unsigned long long *i_element,                    // Element indices, 0-based, < n_elements, [n_out]
                                     const size_t n_out,                                     // Number of outputs, >= 1
                                     dtype *V_re, dtype *V_im, dtype *H_re, dtype *H_im,     // Output: e-theta and e-phi field, [n_out, n_ang], allocated
                                     const size_t V_step, const size_t H_step,               // Output stride of V and H: 1 split, 2 interleaved complex (V_im = V_re + 1)
                                     dtype *dist,                                            // Optional output: projected distance in m, [n_out, n_ang] or nullptr
                                     dtype *azimuth_loc, dtype *elevation_loc,               // Optional output: local angles in rad, [n_out, n_ang] or nullptr
                                     dtype *gamma);                                          // Optional output: polarization rotation angle in rad, [n_out, n_ang] or nullptr

#endif
