// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the sources of the site-specific simulation tools (ray tracing, meshes, point clouds)

#ifndef qd_helpers_site_hpp
#define qd_helpers_site_hpp

#include "quadriga_lib.hpp"

#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Cross product R = V x K, optionally normalized
template <typename dtype>
static inline void qd_crossp(dtype vx, dtype vy, dtype vz,    // Vector V
                          dtype kx, dtype ky, dtype kz,    // Vector K
                          dtype *rx, dtype *ry, dtype *rz, // Result V x K
                          bool normalize_output = false)
{
    *rx = vy * kz - vz * ky;
    *ry = vz * kx - vx * kz;
    *rz = vx * ky - vy * kx;

    if (normalize_output)
    {
        dtype scl = (dtype)1.0 / std::sqrt(*rx * *rx + *ry * *ry + *rz * *rz);
        *rx *= scl, *ry *= scl, *rz *= scl;
    }
}

// Dot product of V and K, optionally divided by |V| |K| (unchanged if one of them has zero length)
template <typename dtype>
static inline dtype qd_dotp(dtype vx, dtype vy, dtype vz, // Vector V
                         dtype kx, dtype ky, dtype kz, // Vector K
                         bool normalize = false)
{
    dtype dot = vx * kx + vy * ky + vz * kz;

    if (normalize)
    {
        dtype v_mag = std::sqrt(vx * vx + vy * vy + vz * vz);
        dtype k_mag = std::sqrt(kx * kx + ky * ky + kz * kz);
        if (v_mag > 0 && k_mag > 0)
            dot /= (v_mag * k_mag);
    }

    return dot;
}

// Normalizes (x, y, z) with the reciprocal of its length above 2e-7, else sets it to (1, 0, 0); returns the length
static inline double qd_normalize(double &x, double &y, double &z)
{
    double len = std::sqrt(x * x + y * y + z * z);
    if (len > 2e-7)
    {
        double scl = 1.0 / len;
        x *= scl, y *= scl, z *= scl;
    }
    else
        x = 1.0, y = 0.0, z = 0.0;
    return len;
}

// Rotates the vector V by the angle theta (rad) around the unit axis K (Rodrigues' rotation formula); result R
static inline void qd_rotate_around_axis(double vx, double vy, double vz,    // Vector V
                                         double kx, double ky, double kz,    // Rotation axis K, normalized
                                         double theta,                       // Rotation angle
                                         double *rx, double *ry, double *rz) // Result R
{
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double dot = vx * kx + vy * ky + vz * kz;
    *rx = vx * cos_theta + (ky * vz - kz * vy) * sin_theta + kx * dot * (1.0 - cos_theta);
    *ry = vy * cos_theta + (kz * vx - kx * vz) * sin_theta + ky * dot * (1.0 - cos_theta);
    *rz = vz * cos_theta + (kx * vy - ky * vx) * sin_theta + kz * dot * (1.0 - cos_theta);
}

// First vertex P and unit normal N of face i_face (1-based) of a mesh [n_mesh, 9] in column-major order; the vertex
// order sets the direction of the normal
template <typename dtype>
static inline void qd_face_plane(const dtype *p_mesh, size_t n_mesh, size_t i_face,
                                 double &Px, double &Py, double &Pz, double &Nx, double &Ny, double &Nz)
{
    const dtype *p = p_mesh + i_face - 1;
    Px = (double)p[0], Py = (double)p[n_mesh], Pz = (double)p[2 * n_mesh];
    qd_crossp((double)p[3 * n_mesh] - Px, (double)p[4 * n_mesh] - Py, (double)p[5 * n_mesh] - Pz,
           (double)p[6 * n_mesh] - Px, (double)p[7 * n_mesh] - Py, (double)p[8 * n_mesh] - Pz, &Nx, &Ny, &Nz);
    qd_normalize(Nx, Ny, Nz);
}

// Longest edge of the triangle tv (three vertices, x y z each), or e0 where that is longer; e0 is a squared length
// (0 for none, INFINITY marks a vertex without a usable hit)
static inline double qd_tube_edge_length(const double tv[9], double e0)
{
    double e = e0, dx, dy, dz, l;
    dx = tv[3] - tv[0], dy = tv[4] - tv[1], dz = tv[5] - tv[2];
    l = dx * dx + dy * dy + dz * dz;
    e = (l > e) ? l : e;
    dx = tv[6] - tv[0], dy = tv[7] - tv[1], dz = tv[8] - tv[2];
    l = dx * dx + dy * dy + dz * dz;
    e = (l > e) ? l : e;
    dx = tv[6] - tv[3], dy = tv[7] - tv[4], dz = tv[8] - tv[5];
    l = dx * dx + dy * dy + dz * dz;
    e = (l > e) ? l : e;
    return std::sqrt(e);
}

// Resolves the kernel selector 'use_kernel' of a function with GENERIC, AVX2 and CUDA kernels: returns 1 (GENERIC),
// 2 (AVX2) or 3 (CUDA). Auto (0) selects CUDA for at least 500 work items 'n_items' when it is available, else AVX2
// when it is available. Throws std::invalid_argument for values other than 0 to 3 and for a kernel that is not available.
static inline int qd_kernel_auto_cuda(int use_kernel, size_t n_items)
{
    if (use_kernel < 0 || use_kernel > 3)
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1, 2 or 3.");
    if (use_kernel == 2 && !quadriga_lib::quadriga_lib_has_AVX2())
        throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
    if (use_kernel == 3 && !quadriga_lib::quadriga_lib_has_CUDA())
        throw std::invalid_argument("CUDA kernel requested but not available (compile with BUILD_WITH_CUDA and run on CUDA-capable GPU).");
    if (use_kernel != 0)
        return use_kernel;
    if (n_items >= 500 && quadriga_lib::quadriga_lib_has_CUDA())
        return 3;
    return quadriga_lib::quadriga_lib_has_AVX2() ? 2 : 1;
}

// Checks the 0-based start indices 'sub_ind' [n_sub] of the sub-meshes or sub-clouds of an object with 'n_elem' rows:
// the first entry is 0, each entry is larger than the one before and below 'n_elem'. 'name' is the argument name,
// 'elem' names one row in the messages ("mesh face", "point"). Returns n_sub (0 for an empty index).
static inline arma::uword qd_check_sub_ind(const arma::u32_vec &sub_ind, arma::uword n_elem, const char *name, const char *elem)
{
    const arma::uword n_sub = sub_ind.n_elem;
    if (n_sub == 0)
        return 0;
    const unsigned *p_sub = sub_ind.memptr();
    if (p_sub[0] != 0U)
        throw std::invalid_argument(std::string("Input '") + name + "' must start at the first " + elem + ".");
    for (arma::uword i = 1; i < n_sub; ++i)
        if (p_sub[i] <= p_sub[i - 1])
            throw std::invalid_argument(std::string("Entries of '") + name + "' must be strictly increasing.");
    if ((arma::uword)p_sub[n_sub - 1] >= n_elem) // the entries increase, so all are in range
        throw std::invalid_argument(std::string("Entries of '") + name + "' cannot point beyond the last " + elem + ".");
    return n_sub;
}

// Checks the bounding boxes 'aabb' [n_sub, 6] of the 'n_sub' sub-meshes given by 'sub_mesh_ind'; an empty 'aabb' is
// accepted (computed by the caller)
template <typename dtype>
static inline void qd_check_aabb(const arma::Mat<dtype> &aabb, arma::uword n_sub)
{
    if (aabb.n_elem == 0)
        return;
    if (n_sub == 0)
        throw std::invalid_argument("Input 'aabb' requires 'sub_mesh_ind'.");
    if (aabb.n_rows != n_sub || aabb.n_cols != 6)
        throw std::invalid_argument("Input 'aabb' must have size [n_sub, 6].");
}

// Checks that 'mesh' is a non-empty triangle mesh [n_mesh, 9]; returns n_mesh
template <typename dtype>
static inline arma::uword qd_check_mesh(const arma::Mat<dtype> &mesh)
{
    if (mesh.n_rows == 0)
        throw std::invalid_argument("Input 'mesh' cannot be empty.");
    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    return mesh.n_rows;
}

// Number of materials n_mtl in the material property columns 'mtl_prop' (the csv_prop output of obj_file_read);
// empty columns are skipped, all others must have n_mtl entries. With 'state_words', n_mtl is limited to 16383, the
// 14-bit material index of the state words of ray_state_update.
template <typename dtype>
static inline arma::uword qd_material_count(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop, bool state_words = true)
{
    arma::uword n_mtl = 0;
    bool seen = false;
    for (const auto &kv : mtl_prop)
    {
        if (kv.second.empty())
            continue;
        arma::uword len = (arma::uword)kv.second.size();
        if (!seen)
            n_mtl = len, seen = true;
        else if (len != n_mtl)
            throw std::invalid_argument("Material property column '" + kv.first + "' has length " +
                                        std::to_string(len) + ", expected " + std::to_string(n_mtl) +
                                        " (all columns must have the same number of materials).");
    }
    if (state_words && n_mtl > 16383)
        throw std::invalid_argument("Number of materials cannot exceed 16383.");
    return n_mtl;
}

// Checks the frequencies 'center_freq' [n_freq] of the ray tracer: 1 to 127 values (the path storage layout), all > 0;
// returns n_freq
static inline arma::uword qd_check_center_freq(const arma::fvec &center_freq)
{
    const arma::uword n_freq = center_freq.n_elem;
    if (n_freq == 0)
        throw std::invalid_argument("Input 'center_freq' cannot be empty.");
    if (n_freq > 127)
        throw std::invalid_argument("Input 'center_freq' cannot have more than 127 elements.");
    if (center_freq.min() <= 0.0f)
        throw std::invalid_argument("Input 'center_freq' must be > 0.");
    return n_freq;
}

// Checks the ray origins 'orig' [n_ray, 3] of the ray tracer, at most 2^32-1 rays (32-bit ray indices); returns n_ray
static inline arma::uword qd_check_orig(const arma::fmat &orig)
{
    const arma::uword n_ray = orig.n_rows;
    if (n_ray == 0)
        throw std::invalid_argument("Input 'orig' cannot be empty.");
    if (orig.n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns.");
    if (n_ray > 4294967295ull)
        throw std::invalid_argument("Number of rays cannot exceed 2^32-1.");
    return n_ray;
}

// Writes the sequence [sequence_length] repeat_sequence times, each value repeat_value times in a row, converted to
// dtypeOut; output has sequence_length * repeat_value * repeat_sequence values
template <typename dtypeIn, typename dtypeOut>
static inline void qd_repeat_sequence(const dtypeIn *sequence, size_t sequence_length, size_t repeat_value, size_t repeat_sequence, dtypeOut *output)
{
    size_t pos = 0;
    for (size_t rs = 0; rs < repeat_sequence; ++rs)
        for (size_t v = 0; v < sequence_length; ++v)
        {
            dtypeOut val = (dtypeOut)sequence[v];
            for (size_t rv = 0; rv < repeat_value; ++rv)
                output[pos++] = val;
        }
}

#endif
