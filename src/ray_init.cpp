// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_init
Seed a sphere of rays from a point source

- Launches `n_ray` rays from a point source `O` onto an [[icosphere]] tessellation, giving near-uniform
  angular coverage of the full sphere (4π sr).
- `n_ray` is quantized to the icosphere grid: `n_div = round(sqrt(n_ray_target / 20))` (min 1) and
  `n_ray = 20 · n_div²`, so the returned count is the closest tessellation to `n_ray_target`, not exact.
- Ray origins sit on a small launch sphere of radius `r0` centered at `O`, not at `O` itself, so the beam
  triangles (`trivec`) have finite extent from the first segment.
- When `mesh` or `rx_points` is supplied, `r0` is auto-sized to 0.8× the distance to the nearest obstacle
  (found along a coarse probe sphere) or receive point, whichever is closer, clamped to ≥ 0.01 m. A receive
  point inside the launch sphere would never be hit by any ray, so it bounds `r0` the same way an obstacle
  does. If neither is within `max_path_length`, or with both omitted, `r0 = 0.01 m`.
- Emits the per-ray medium-state words and distance accumulators consumed by [[ray_state_update]], all
  initialized to the outside-air / zero-distance start state.
- Beam wavefront (`trivec`) and directions (`tridir`, Cartesian) match the [[ray_mesh_interact]] input format.

## Declaration:
```
arma::uword ray_init(
    arma::uword n_ray_target,
    arma::uword n_freq,
    float Ox, float Oy, float Oz,
    float max_path_length,
    arma::fmat *orig = nullptr,
    arma::fmat *dest = nullptr,
    arma::fmat *trivec = nullptr,
    arma::fmat *tridir = nullptr,
    arma::Col<short> *mtl_ind_prev = nullptr,
    arma::Col<short> *mtl_ind_current = nullptr,
    arma::Col<short> *mtl_ind_buffer = nullptr,
    arma::fmat *path_dir_prev = nullptr,
    arma::fmat *acc_dist = nullptr,
    std::vector<quadriga_lib::path> *paths = nullptr,
    const arma::fmat *mesh = nullptr,
    const arma::u32_vec *sub_mesh_index = nullptr,
    const arma::fmat *rx_points = nullptr,
    bool scalar_mode = false);
```

## Inputs:
- **`n_ray_target`** — Desired ray count; quantized to the nearest icosphere grid (see above)
- **`n_freq`** — Number of frequency bins allocated per path in `paths`; must be ≥ 1 (throws if 0; the
  [[path]] layout supports 1-127)
- **`Ox`**, **`Oy`**, **`Oz`** — Point-source (transmitter) position in GCS [m]
- **`max_path_length`** — Maximum ray length [m]; sets `dest` and bounds the launch-sphere probe (floored at 0.01 m)
- **`mesh`** *(optional)* — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`. Used only to auto-size
  the launch sphere `r0`. NULL → `r0 = 0.01 m`.
- **`sub_mesh_index`** *(optional)* — Sub-mesh partition offsets for the accelerated intersect; passed to
  [[ray_triangle_intersect]]. NULL → no partitioning.
- **`rx_points`** *(optional)* — Receive points in 3D space; `[n_point, 3]`. Used only to bound the launch
  sphere `r0`, so that no receiver starts inside it. NULL → receivers are not considered.
- **`scalar_mode`** — Path storage layout passed to `paths`: `true` = SCALAR (acoustic, one pressure
  coefficient per frequency), `false` = EM (2×2 Jones matrix per frequency)

## Outputs:
- **`orig`** — Ray origins on the launch sphere, `O + r0·d̂`; `[n_ray, 3]`
- **`dest`** — Ray destinations at `max_path_length` from `O`; `[n_ray, 3]`
- **`trivec`** — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, matches [[ray_mesh_interact]]
- **`tridir`** — Per-vertex ray directions, Cartesian; `[n_ray, 9]`
- **`mtl_ind_prev`**, **`mtl_ind_current`**, **`mtl_ind_buffer`** — Initial medium-state words for
  [[ray_state_update]], all zeroed (outside air, no flags); `[n_ray]`
- **`path_dir_prev`** — Initial physical ray direction (unit vectors from `O`); `[n_ray, 3]`
- **`acc_dist`** — Accumulated in-layer distance, zeroed; `[n_ray, 2]`; col 1 = refracted distance, col 2 = geometric distance
- **`paths`** — Per-ray [[path]] objects, one per ray; each reinitialized to 0 segments with `n_freq`
  frequency bins in the `scalar_mode` layout, and `length` seeded to `orig_length` (the `O`-to-origin offset);
  `n_ray` entries

## Returns:
- Number of rays generated, `n_ray = 20 · n_div²`

## See also:
- [[icosphere]] (generates the ray fan, beam wavefront, and directions)
- [[ray_triangle_intersect]] (launch-sphere sizing and per-segment intersection)
- [[ray_mesh_interact]] (consumes `orig` / `dest` / `trivec` / `tridir` / `orig_length`)
- [[ray_state_update]] (consumes the medium-state words, `path_dir_prev`, and `acc_dist`)
- [[path]] (the per-ray storage object populated in `paths`)
- [[ray_commit]] (consumes the same `rx_points` when committing paths that reach a receiver)
MD!*/

arma::uword quadriga_lib::ray_init(arma::uword n_ray_target,
                                   arma::uword n_freq,
                                   float Ox, float Oy, float Oz,
                                   float max_path_length,
                                   arma::fmat *orig,
                                   arma::fmat *dest,
                                   arma::fmat *trivec,
                                   arma::fmat *tridir,
                                   arma::Col<short> *mtl_ind_prev,
                                   arma::Col<short> *mtl_ind_current,
                                   arma::Col<short> *mtl_ind_buffer,
                                   arma::fmat *path_dir_prev,
                                   arma::fmat *acc_dist,
                                   std::vector<quadriga_lib::path> *paths,
                                   const arma::fmat *mesh,
                                   const arma::u32_vec *sub_mesh_index,
                                   const arma::fmat *rx_points,
                                   bool scalar_mode)
{
    // Validate mesh
    bool has_mesh = mesh && mesh->n_rows != 0;
    if (has_mesh && mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    // Validate receive points
    bool has_points = rx_points && rx_points->n_rows != 0;
    if (has_points && rx_points->n_cols != 3)
        throw std::invalid_argument("Input 'rx_points' must have 3 columns containing x,y,z coordinates.");

    // Enforce minimum path length
    max_path_length = max_path_length < 0.01f ? 0.01f : max_path_length;

    // Number of icosphere divisions
    arma::uword n_div = (arma::uword)std::round(std::sqrt((double)n_ray_target / 20.0));
    n_div = n_div == 0 ? 1 : n_div;         // Must be >= 1
    arma::uword n_ray = 20 * n_div * n_div; // Number of rays

    if (n_freq == 0)
        throw std::invalid_argument("Number of frequencies cannot be 0.");

    // Detect launch sphere size
    float r0 = 0.01f; // Launch sphere radius, default = 1 cm
    if (has_mesh || has_points)
    {
        // Squared distance to the nearest obstacle or receive point, seeded with the probe range
        float d2_min = max_path_length * max_path_length;

        if (has_mesh)
        {
            // Make a small icosphere
            arma::fmat orig_tmp, intersect_tmp;
            arma::fvec length_tmp;
            arma::uword n_rays_test = quadriga_lib::icosphere<float>(12, 0.01f, &orig_tmp, &length_tmp);

            // Test origin and destinations
            arma::fmat dest_tmp(n_rays_test, 3, arma::fill::none);
            const float *p_length = length_tmp.memptr();
            float *p_start = orig_tmp.memptr(), *p_end = dest_tmp.memptr();
            for (size_t i_ray = 0; i_ray < n_rays_test; ++i_ray)
            {
                float scl = max_path_length / p_length[i_ray];
                p_end[i_ray] = p_start[i_ray] * scl + Ox;
                p_end[i_ray + n_rays_test] = p_start[i_ray + n_rays_test] * scl + Oy;
                p_end[i_ray + 2 * n_rays_test] = p_start[i_ray + 2 * n_rays_test] * scl + Oz;
                p_start[i_ray] += Ox;
                p_start[i_ray + n_rays_test] += Oy;
                p_start[i_ray + 2 * n_rays_test] += Oz;
            }

            // Get intersect coordinates
            quadriga_lib::ray_triangle_intersect<float>(&orig_tmp, &dest_tmp, mesh, &intersect_tmp, nullptr, nullptr, nullptr, nullptr, sub_mesh_index);

            // Nearest obstacle along the probe sphere
            float *p_intersect = intersect_tmp.memptr();
            for (arma::uword i_ray = 0; i_ray < n_rays_test; ++i_ray)
            {
                float Lx = p_intersect[i_ray] - Ox;
                float Ly = p_intersect[i_ray + n_rays_test] - Oy;
                float Lz = p_intersect[i_ray + 2 * n_rays_test] - Oz;
                float d_squared = Lx * Lx + Ly * Ly + Lz * Lz;
                d2_min = (d_squared < d2_min) ? d_squared : d2_min;
            }
        }

        if (has_points)
        {
            // A receive point inside the launch sphere is never reached by any ray, so the
            // nearest one bounds r0 in the same way an obstacle does
            const arma::uword n_points = rx_points->n_rows;
            const float *p_points = rx_points->memptr();

#pragma omp parallel
            {
                float d2_local = d2_min;

#pragma omp for schedule(static) nowait
                for (long long i_pt = 0; i_pt < (long long)n_points; ++i_pt)
                {
                    const arma::uword i = (arma::uword)i_pt;
                    float Lx = p_points[i] - Ox;
                    float Ly = p_points[i + n_points] - Oy;
                    float Lz = p_points[i + 2 * n_points] - Oz;
                    float d_squared = Lx * Lx + Ly * Ly + Lz * Lz;
                    d2_local = (d_squared < d2_local) ? d_squared : d2_local;
                }

#pragma omp critical
                d2_min = (d2_local < d2_min) ? d2_local : d2_min;
            }
        }

        // Fall back to the default radius when nothing was found within max_path_length
        if (d2_min >= 0.99f * max_path_length * max_path_length)
            r0 = 0.01f;
        else
            r0 = std::sqrt(d2_min) * 0.8f;
        r0 = (r0 < 0.01f) ? 0.01f : r0;
    }

    // Generate rays
    arma::fmat orig_local, dest_local;
    arma::fvec orig_length_local;
    quadriga_lib::icosphere<float>(n_div, r0, &orig_local, &orig_length_local, trivec, tridir, true);
    const float *p_orig_local = orig_local.memptr(), *p_length_local = orig_length_local.memptr();

    bool has_orig = orig != nullptr;
    if (has_orig && (orig->n_rows != n_ray || orig->n_cols != 3))
        orig->set_size(n_ray, 3);
    float *p_orig = has_orig ? orig->memptr() : nullptr;

    bool has_dir = path_dir_prev != nullptr;
    if (has_dir && (path_dir_prev->n_rows != n_ray || path_dir_prev->n_cols != 3))
        path_dir_prev->set_size(n_ray, 3);
    float *p_dir = has_dir ? path_dir_prev->memptr() : nullptr;

    bool has_dest = dest != nullptr;
    if (has_dest && (dest->n_rows != n_ray || dest->n_cols != 3))
        dest->set_size(n_ray, 3);
    float *p_dest = has_dest ? dest->memptr() : nullptr;

    bool has_paths = paths != nullptr;
    if (has_paths && paths->size() != n_ray)
        paths->resize(n_ray);

#pragma omp parallel for schedule(static)
    for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray)
    {
        arma::uword i_ray0 = (arma::uword)i_ray;
        arma::uword i_ray1 = i_ray0 + n_ray;
        arma::uword i_ray2 = i_ray1 + n_ray;

        float scl = 1.0f / p_length_local[i_ray];

        if (has_orig)
        {
            p_orig[i_ray0] = p_orig_local[i_ray0] + Ox;
            p_orig[i_ray1] = p_orig_local[i_ray1] + Oy;
            p_orig[i_ray2] = p_orig_local[i_ray2] + Oz;
        }

        if (has_dir)
        {
            p_dir[i_ray0] = p_orig_local[i_ray0] * scl;
            p_dir[i_ray1] = p_orig_local[i_ray1] * scl;
            p_dir[i_ray2] = p_orig_local[i_ray2] * scl;
        }

        if (has_dest)
        {
            scl *= max_path_length;
            p_dest[i_ray0] = p_orig_local[i_ray0] * scl + Ox;
            p_dest[i_ray1] = p_orig_local[i_ray1] * scl + Oy;
            p_dest[i_ray2] = p_orig_local[i_ray2] * scl + Oz;
        }

        if (has_paths)
        {
            (*paths)[i_ray0].init(0, n_freq, scalar_mode);
            (*paths)[i_ray0].set_length(p_length_local[i_ray0]);
        }
    }

    if (mtl_ind_prev)
        mtl_ind_prev->zeros(n_ray);

    if (mtl_ind_current)
        mtl_ind_current->zeros(n_ray);

    if (mtl_ind_buffer)
        mtl_ind_buffer->zeros(n_ray);

    if (acc_dist)
        acc_dist->zeros(n_ray, 2);

    return n_ray;
}