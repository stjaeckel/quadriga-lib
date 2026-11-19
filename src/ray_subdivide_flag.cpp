// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "qd_helpers_site.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_subdivide_flag
Compute the per-ray subdivision decision from the beam-tube footprint at the first bounce

- Projects the three vertex rays of each beam tube onto the plane of the first-bounce face and
  measures the longest edge of the resulting wavefront triangle
- Flags a ray when that edge exceeds `subdivision_tolerance` and the ray is still eligible to be
  split, i.e. it travels outside a medium and has not reached its subdivision or interaction limit
- The decision is purely geometric: the vertex origins on the face are the same for reflection,
  transmission and refraction, so no material data, second-bounce index or frequency is needed
- This is the single source of truth for the subdivision decision. [[ray_progress]] consumes the
  result when it is passed in and calls this function itself otherwise, so a caller that needs to
  know the outcome in advance — a shading pass that must not commit beams which will reappear as
  sub-beams — gets exactly the set that will actually be split

## Declaration:
```
void quadriga_lib::ray_subdivide_flag(
    const arma::fmat &mesh,
    const arma::fmat &orig,
    const arma::fmat &dest,
    const arma::u32_vec &fbs_ind,
    const arma::fmat &trivec,
    const arma::fmat &tridir,
    const std::vector<quadriga_lib::path> &paths,
    const arma::Col<short> &mtl_ind_current,
    std::vector<bool> &subdiv_flag,
    uint8_t max_interactions = 20,
    uint8_t max_subdivisions = 2,
    float subdivision_tolerance = 3.0f);
```

## Inputs:
- **`mesh`** — Faces of the triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`. Defines `n_ray`; must be non-empty
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`fbs_ind`** — 1-based index of the first intersected mesh element, 0 = no hit; `[n_ray]`. Obtained from [[ray_triangle_intersect]] for the same `orig` / `dest` pair
- **`trivec`** — Beam wavefront triangle vertices relative to the ray origin; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, Cartesian; `[n_ray, 9]`. Need not be unit length
- **`paths`** — Per-ray [[path]] objects; `n_ray` entries. Only the subdivision counter `n_sub` and the segment count are read
- **`mtl_ind_current`** — Current medium state word, 0 = outside; `[n_ray]`. A ray inside a medium is
  never split: the sub-beams restart their in-layer accumulator and recompute their direction
  geometrically, which is only valid outside
- **`max_interactions`** *(optional)* — Total interactions per ray, 0 to 255. A ray that has already reached the limit 
  is not split, so it does not expand into four sub-beams that all terminate in the next generation
- **`max_subdivisions`** *(optional)* — Number of subdivisions per ray, 0 to 255. 0 disables subdivision and 
  all flags are `false`
- **`subdivision_tolerance`** *(optional)* — Maximum beam-tube edge length before a ray is split, in
  metres; must be greater than 0

## Output:
- **`subdiv_flag`** — `true` where the ray must be split; `n_ray` entries. Rays that miss the mesh
  (`fbs_ind = 0`) are always `false`. A beam whose tube only partially covers the face — a vertex ray
  running parallel to it, pointing away from it, or intersecting absurdly far away — is treated as
  having an infinite edge and is always flagged

## See also:
- [[ray_progress]] (advance one generation of a beam-traced ray set)
- [[ray_triangle_intersect]] (compute `fbs_ind`)
- [[subdivide_rays]] (split the flagged beams into sub-beams)
- [[ray_mesh_interact]] (reports the same edge length as `edge_lengthN`)
MD!*/

void quadriga_lib::ray_subdivide_flag(const arma::fmat &mesh,
                                      const arma::fmat &orig,
                                      const arma::fmat &dest,
                                      const arma::u32_vec &fbs_ind,
                                      const arma::fmat &trivec,
                                      const arma::fmat &tridir,
                                      const std::vector<quadriga_lib::path> &paths,
                                      const arma::Col<short> &mtl_ind_current,
                                      std::vector<bool> &subdiv_flag,
                                      uint8_t max_interactions,
                                      uint8_t max_subdivisions,
                                      float subdivision_tolerance)
{
    const arma::uword n_mesh = qd_check_mesh(mesh);

    // Validate the launch configuration
    if (orig.n_elem == 0)
        throw std::invalid_argument("Input 'orig' cannot be empty.");
    if (orig.n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns.");

    const arma::uword n_ray = orig.n_rows;

    if (dest.n_rows != n_ray || dest.n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have size [n_ray, 3].");
    if (fbs_ind.n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'fbs_ind' does not match number of rows in 'orig'.");
    if (trivec.n_rows != n_ray || trivec.n_cols != 9)
        throw std::invalid_argument("Input 'trivec' must have size [n_ray, 9].");
    if (tridir.n_rows != n_ray || tridir.n_cols != 9)
        throw std::invalid_argument("Input 'tridir' must have size [n_ray, 9] (Cartesian).");
    if ((arma::uword)paths.size() != n_ray)
        throw std::invalid_argument("Number of elements in 'paths' does not match number of rows in 'orig'.");
    if (mtl_ind_current.n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'mtl_ind_current' does not match number of rows in 'orig'.");

    // Validate the tolerance
    if (!std::isfinite(subdivision_tolerance) || subdivision_tolerance <= 0.0f)
        throw std::invalid_argument("Input 'subdivision_tolerance' must be larger than 0.");

    // Nothing can be split, skip the geometry entirely
    if (max_subdivisions == 0 || max_interactions == 0)
    {
        subdiv_flag.assign(n_ray, false);
        return;
    }

    // Read pointers, all matrices are [n_ray, k] in column-major order
    const float *p_mesh = mesh.memptr();
    const float *p_orig = orig.memptr();
    const float *p_dest = dest.memptr();
    const float *p_trivec = trivec.memptr();
    const float *p_tridir = tridir.memptr();
    const unsigned *p_fbs_ind = fbs_ind.memptr();
    const short *p_current = mtl_ind_current.memptr();

    const double tolerance = (double)subdivision_tolerance;
    const size_t max_seg = (size_t)max_interactions;

    // One byte per ray so the geometry loop can run in parallel; the bit-packed result is written
    // afterwards. std::vector<bool> shares a word between adjacent elements, so writing it from
    // several threads is a data race even for distinct indices.
    std::vector<uint8_t> tmp(n_ray);
    uint8_t *p_tmp = tmp.data();

    int bad = 0; // Out-of-range face index, reported after the parallel region

#pragma omp parallel for schedule(static) reduction(| : bad) if (n_ray >= 4096)
    for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray) // Ray loop
    {
        const size_t iRx = (size_t)i_ray;
        const size_t iRy = iRx + n_ray, iRz = iRy + n_ray;

        p_tmp[iRx] = 0u; // Default: do not subdivide

        const size_t iFBS = (size_t)p_fbs_ind[iRx]; // Mesh FBS index, 1-based
        if (iFBS == 0)                              // No mesh hit
            continue;
        if (iFBS > n_mesh) // Invalid, must be 1 ... n_mesh
        {
            bad |= 1;
            continue;
        }

        // State gates first: these are scalar reads and let most ineligible rays skip the geometry.
        // The medium word is compared as a whole, not masked, so a ray carrying the resolved flag
        // bit is left alone as well.
        if (p_current[iRx] != 0) // Currently inside a medium
            continue;
        if (paths[iRx].n_sub >= max_subdivisions) // Max. subdiv reached
            continue;
        if (paths[iRx].n_seg() >= max_seg) // Max. interactions reached
            continue;

        // Load origin and destination
        double Ox = (double)p_orig[iRx], Oy = (double)p_orig[iRy], Oz = (double)p_orig[iRz];
        double Dx = (double)p_dest[iRx], Dy = (double)p_dest[iRy], Dz = (double)p_dest[iRz];

        double ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz; // Ray direction O to D

        // Shift D back 2 ULP to drop paths that end on the FBS face
        double odScale = std::max({std::abs(ODx), std::abs(ODy), std::abs(ODz), 1e-30});
        double posScale = std::max({std::abs(Dx), std::abs(Dy), std::abs(Dz), 1.0});
        double offset = 2.0 * posScale * 1.1920929e-7 / odScale;
        Dx -= offset * ODx, Dy -= offset * ODy, Dz -= offset * ODz;

        ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz; // Update direction O to D
        double OD_length = qd_normalize(ODx, ODy, ODz);

        // FBS face plane; its first vertex is the fallback FBS intersect point
        double Fx, Fy, Fz, Nx, Ny, Nz;
        qd_face_plane(p_mesh, n_mesh, iFBS, Fx, Fy, Fz, Nx, Ny, Nz);

        // Ray-plane intersection
        double cos_theta = ODx * Nx + ODy * Ny + ODz * Nz;  // goes to zero as the ray approaches tangency
        if (OD_length > 2e-7 && std::abs(cos_theta) > 2e-7) // guard degenerate ray and grazing/parallel plane
        {
            double OF_length = ((Fx - Ox) * Nx + (Fy - Oy) * Ny + (Fz - Oz) * Nz) / cos_theta;
            if (OF_length <= 0.0) // Origin lies in FBS plane (include)
                Fx = Ox, Fy = Oy, Fz = Oz;
            else if (OF_length < OD_length) // True FBS intersect (include)
                Fx = Ox + OF_length * ODx, Fy = Oy + OF_length * ODy, Fz = Oz + OF_length * ODz;
            else // Destination lies in FBS plane (exclude)
                Fx = Dx, Fy = Dy, Fz = Dz;
        }

        // Project the three vertex rays onto the FBS plane. The resulting vertex origins are the
        // footprint of the tube on the face and are identical for reflection, transmission and
        // refraction: the interaction type only changes the outgoing vertex directions.
        double T[9];
        double edge_length = 0.0; // Accumulates squared edge lengths, INFINITY marks a partial hit

        for (int iTube = 0; iTube < 3; ++iTube) // Vertex loop
        {
            const size_t iT = iRx + 3 * (size_t)iTube * n_ray;

            double Tx = Ox + (double)p_trivec[iT],
                   Ty = Oy + (double)p_trivec[iT + n_ray],
                   Tz = Oz + (double)p_trivec[iT + 2 * n_ray];

            double Vx = (double)p_tridir[iT],
                   Vy = (double)p_tridir[iT + n_ray],
                   Vz = (double)p_tridir[iT + 2 * n_ray];
            qd_normalize(Vx, Vy, Vz);

            // Calculate intersect point of the vertex-ray with the face
            double denom = Vx * Nx + Vy * Ny + Vz * Nz;
            bool no_usable_hit = std::abs(denom) < 1e-6; // true => parallel, no face intersection
            double d = no_usable_hit ? 0.0 : ((Fx - Tx) * Nx + (Fy - Ty) * Ny + (Fz - Tz) * Nz) / denom;
            double Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
            no_usable_hit = no_usable_hit || d > 1.0e5 || d < 0.0;

            if (no_usable_hit) // no usable face intersection
                edge_length = INFINITY;

            T[3 * iTube] = Wx - Fx, T[3 * iTube + 1] = Wy - Fy, T[3 * iTube + 2] = Wz - Fz;
        }

        // Same edge length as ray_mesh_interact reports
        edge_length = qd_tube_edge_length(T, edge_length);

        p_tmp[iRx] = (edge_length > tolerance) ? 1u : 0u;
    }

    if (bad & 1)
        throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");

    // Pack down for the caller, bit-packed writes cannot be parallelized
    subdiv_flag.resize(n_ray);
    for (arma::uword i_ray = 0; i_ray < n_ray; ++i_ray)
        subdiv_flag[i_ray] = (p_tmp[i_ray] != 0u);
}
