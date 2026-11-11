// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// A collection of AVX2 accelerated functions
//
// The files src/*_avx2.cpp are compiled with AVX2 flags, so every function emitted there may contain AVX instructions.
// Templates and inline functions (std::vector, std::unique_ptr, std::numeric_limits, and with MSVC also the exception
// constructors) are emitted as weak symbols, and the linker keeps one copy for all callers in a program. If it keeps
// the AVX2 copy, plain code calls it and stops with "Illegal instruction" on CPUs without AVX2. Only the kernels below
// are protected, by runtime_AVX2_Check(). Therefore the AVX2 files use no standard library templates or inline
// functions: heap buffers come from qd_avx2_buffer, exceptions from qd_avx2_throw_*, constants from <cfloat> and
// <cmath> macros. The test script tests/check_avx2_objects.sh checks the compiled objects (GCC and Clang).

#ifndef quadriga_lib_avx2fun_H
#define quadriga_lib_avx2fun_H

#include <cstddef>
#include <vector>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <cstdint>

// Zero-filled heap buffer of n values of elem_size bytes, released by the destructor; no allocation for n = 0 (ptr is
// then nullptr). The members are defined in the plain file qd_avx2_buffer.cpp, compiled without AVX2 flags.
class qd_avx2_buffer
{
public:
    void *ptr;

    qd_avx2_buffer(size_t n, size_t elem_size);
    ~qd_avx2_buffer();
    qd_avx2_buffer(const qd_avx2_buffer &) = delete;
    qd_avx2_buffer &operator=(const qd_avx2_buffer &) = delete;
};

// Throw std::invalid_argument or std::bad_alloc. Defined in the plain file qd_avx2_buffer.cpp, so the exception
// constructors, which MSVC defines inline, are not compiled with AVX2 flags.
[[noreturn]] void qd_avx2_throw_invalid_argument(const char *message);
[[noreturn]] void qd_avx2_throw_bad_alloc();

// Perform a simple test calculation
void qd_TEST_AVX2(const float *X, // Aligned memory, 16 floats
                  float *Z);      // Aligned memory, 8 floats

// Ray-Triangle Intersection test with BVH acceleration
template <typename dtype>
void qd_RTI_AVX2(const dtype *Tx, const dtype *Ty, const dtype *Tz, // First vertex coordinate in GCS, length n_mesh
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
                 unsigned *hit_cnt = nullptr);                      // Number of hits between orig and dest, length n_ray, uninitialized, optional

// AVX2 accelerated implementation of RayPointIntersect
void qd_RPI_AVX2(const float *Px, const float *Py, const float *Pz,    // Point coordinates, length n_point (allocated n_point + 7)
                 const size_t n_point,                                 // Number of points, the arrays must be allocated up to n_point + 7
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
                 std::vector<unsigned> *hit_index,                     // Output: flat list of 0-based ray indices grouped by point, resized by the kernel
                 unsigned *hit_offset);                                // Output: 0-based start of each point's block, length n_point + 1, allocated by the caller

// AVX2 accelerated implementation of DFT
template <typename dtype>
void qd_DFT_AVX2(const dtype *CFr,       // Channel coefficients, real part, Size [n_ant, n_path]
                 const dtype *CFi,       // Channel coefficients, imaginary part, Size [n_ant, n_path]
                 const dtype *DL,        // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                 const size_t n_ant,     // Number of MIMO sub-links
                 const size_t n_path,    // Number multipath components
                 const bool planar_wave, // Indicator that same delays are used for all antennas
                 const double *freq,     // Carrier frequencies -bandwidth * pilot_grid in [Hz], Size [ n_carrier ] (no alignment needed)
                 const size_t n_carrier, // Number of carriers, mutiple of 8
                 float *Hr,              // Channel matrix, real part, Size [ n_carrier, n_ant ]
                 float *Hi);             // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]

// Combine interpolated TX/RX antenna patterns with the polarization transfer
// matrix and applies the propagation phase to produce one slice of MIMO
// channel coefficients (one path, all TX-RX element pairs).
template <typename dtype>
void qd_coeff_combine_avx2(const dtype *__restrict pVrr, const dtype *__restrict pVri, // RX V-pol pattern (real, imag), length n_links
                           const dtype *__restrict pHrr, const dtype *__restrict pHri, // RX H-pol pattern (real, imag), length n_links
                           const dtype *__restrict pVtr, const dtype *__restrict pVti, // TX V-pol pattern (real, imag), length n_links
                           const dtype *__restrict pHtr, const dtype *__restrict pHti, // TX H-pol pattern (real, imag), length n_links
                           const dtype *__restrict pM,                                 // Polarization transfer matrix, 8 values broadcast over all links: {Re_VV, Im_VV, Re_HV, Im_HV, Re_VH, Im_VH, Re_HH, Im_HH}, labels <rx><tx>
                           const dtype *__restrict p_delays,                           // Per-link path lengths in [m], length n_links  (read-only)
                           dtype wave_number,                                          // 2*pi*f/c
                           dtype wavelength,                                           // wavelength  : c/f
                           dtype path_amplitude,                                       // sqrt(path_gain) scaling factor (set 0 for fake LOS path)
                           dtype *__restrict p_coeff_re,                               // Output real part of coefficients, length n_links
                           dtype *__restrict p_coeff_im,                               // Output imag part of coefficients, length n_links
                           size_t n_links);                                            // Number of TX-RX element pairs (n_tx * n_rx)

// Check for AVX2 availability at runtime
bool runtime_AVX2_Check();

// AVX2 accelerated sum-of-sinusoids evaluation
// val(p, g) = sum_n wc(n, g) · cos( 2·pi · t ) - ws(n, g) · sin( 2·pi · t ), t = fx(n)·px(p) + fy(n)·py(p) + fz(n)·pz(p)
// The argument t is computed and reduced to one period in double precision, sine and cosine are evaluated in
// single precision, the sum is accumulated in double precision; no alignment requirements
void qd_SOS_AVX2(const double *fx, const double *fy, const double *fz, // Frequencies in [1/m], length n_coeff
                 const double *wc, const double *ws,                   // Weights a·cos(phase), a·sin(phase), Size [n_coeff, n_gen]
                 const size_t n_coeff,                                 // Number of sinusoids
                 const size_t n_gen,                                   // Number of generators
                 const double *px, const double *py, const double *pz, // Positions in [m], length n_pos
                 const size_t n_pos,                                   // Number of positions
                 double *val);                                         // Output: Values, Size [n_pos, n_gen], uninitialized

// Interpolation of array antenna patterns (arrayant_interpolate), AVX2 kernel; computes in single precision for
// float and double. Arguments and preconditions as qd_arrayant_interpolate_GENERIC, and the pattern of all elements
// has at most INT_MAX samples (int32 gather offsets); checked by qd_arrayant_interpolate_dispatch, not by the kernel.
template <typename dtype>
void qd_arrayant_interpolate_AVX2(const dtype *e_theta_re, const dtype *e_theta_im,       // e-theta field, real / imaginary part, [n_elevation, n_azimuth, n_elements]
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
                                  dtype *dist,                                            // Optional output: projected distance in m, [n_out, n_ang] or nullptr
                                  dtype *azimuth_loc, dtype *elevation_loc,               // Optional output: local angles in rad, [n_out, n_ang] or nullptr
                                  dtype *gamma);                                          // Optional output: polarization rotation angle in rad, [n_out, n_ang] or nullptr

#endif
