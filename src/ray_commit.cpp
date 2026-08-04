// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "bits.hpp"

#include <algorithm>

// This is supposed to run on 64-bit only
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");
static_assert(std::is_nothrow_move_constructible_v<quadriga_lib::path>, "path must have a noexcept move ctor, otherwise vector deep-copies on realloc");

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_commit
Commit the paths of a launch configuration that reach a receiver

- Intersects the current beam set with the receive point cloud ([[ray_point_intersect]]) and appends one
  [[path]] object to `paths_commit` for every ray-point pair that survives the gates below.
- A pair is committed when the receiver is not shaded by the ray's own first-bounce face, the total path
  length stays below `max_path_length`, and the gain at `center_frequency[0]` stays above `min_gain_dB`.
- Shading is a half-space test against the plane of the first-bounce face: the receiver must lie on the
  same side of that plane as the ray origin. The face is treated as an infinite plane, so a receiver just
  past the edge of a small face is conservatively dropped.
- Rays flagged in `subdiv_flag_in` are skipped entirely — they reappear as sub-beams in the next
  generation and would otherwise be committed twice. Subdivision is not detected internally.
- A ray travelling inside a medium is committed with the in-layer attenuation of the final leg folded into
  its coefficients, per frequency, via [[medium_gain]] on `mtl_ind_current & 0x7FFF`.
- In EM mode a receive-side mirror (`VV = 1`, `HH = -1`) is applied per frequency; in SCALAR mode only the gain is applied.
- The receiver index is written to the committed path's `iC`. It indexes `points` as passed in, so a caller
  using a cloud reordered by [[point_cloud_segmentation]] must map it back itself.
- The committed path is not extended by a segment: the receiver is not an interaction point, so `nSEG` and
  `length` are those of the in-flight ray and the caller recovers the total with `path::calc_length`.

## Declaration:
```
arma::uword quadriga_lib::ray_commit(
    const std::vector<quadriga_lib::path> &paths,
    std::vector<quadriga_lib::path> &paths_commit,
    const arma::fmat &mesh,
    const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
    const arma::fvec &center_frequency,
    const arma::fmat &orig,
    const arma::u32_vec &fbs_ind,
    const arma::fmat &trivec,
    const arma::fmat &tridir,
    const arma::Col<short> &mtl_ind_current,
    const arma::fmat &points,
    const arma::u32_vec *sub_cloud_index = nullptr,
    const std::vector<bool> *subdiv_flag_in = nullptr,
    float max_path_length = 10e3,
    float min_gain_dB = -140.0f,
    bool ignore_direct_path = false);
```

## Inputs:
- **`paths`** — In-flight per-ray [[path]] objects; `[n_ray]`. Frequency count and layout define those of the committed paths; not modified
- **`mesh`** — Faces of the triangular mesh; each row `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`. Only the
  faces named by `fbs_ind` are read, for the shading plane
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]);
  each value has length `n_mtl` (max 32767)
- **`center_frequency`** — Center frequencies in [Hz]; `[n_freq]`, 1 to 127 entries, must match the layout of
  `paths`. `center_frequency[0]` is the reference frequency for the gain gate
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`. Defines `n_ray`; must be non-empty
- **`fbs_ind`** — 1-based index of the first intersected mesh element, 0 = no hit; `[n_ray]`. Obtained from
  [[ray_triangle_intersect]] for the same launch configuration
- **`trivec`** — Beam wavefront triangle vertices relative to the ray origin; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, Cartesian; `[n_ray, 9]`. Need not be unit length
- **`mtl_ind_current`** — Current medium state word, 0 = outside (bit-masked: `mat = w & 0x7FFF`,
  `flag = w & 0x8000`); `[n_ray]`
- **`points`** — Receive points in 3D space; `[n_point, 3]`
- **`sub_cloud_index`** *(optional)* — Sub-cloud partition offsets for the point cloud (see
  [[point_cloud_segmentation]]); `[n_sub]`. NULL → no partitioning
- **`subdiv_flag_in`** *(optional)* — Rays that will be split in the next generation and must not be
  committed now; `[n_ray]`, indexed in the full ray set. Pass the output of [[ray_subdivide_flag]] for the
  same launch configuration. NULL / empty → no ray is excluded on these grounds
- **`max_path_length`** *(optional)* — Maximum total path length including the leg to the receiver [m]
- **`min_gain_dB`** *(optional)* — Gain at `center_frequency[0]` below which a path is not committed, in dB;
  evaluated with free-space path loss and the in-medium loss of the final leg included
- **`ignore_direct_path`** *(optional)* — Drop every path that arrives by transmission only
  (`nREF == 0 && nSCT == 0`, which includes pure LOS); these are covered by [[calc_diffraction_gain]].
  The test is unconditional: under refraction the traced path is longer than the straight line, so a
  length-based test would let it through and double-count against the diffraction model

## Output:
- **`paths_commit`** — Committed paths, appended to whatever the vector already holds; extended by
  `n_commit` entries. Each carries the receiver index in `iC`, the interaction history of its ray, and the
  transfer coefficients with the receive-side mirror and any in-medium loss applied. Existing entries are
  not modified; if the vector is non-empty its layout must match `paths`

## Returns:
- Number of newly committed paths, `n_commit`

## See also:
- [[ray_point_intersect]] (ray-point intersection, produces the pair list)
- [[ray_subdivide_flag]] (produces `subdiv_flag_in`)
- [[ray_progress]] (advance the launch configuration to the next generation)
- [[point_cloud_segmentation]] (generate `sub_cloud_index`)
- [[calc_diffraction_gain]] (covers the paths removed by `ignore_direct_path`)
- [[path]] (the per-ray storage object)
MD!*/

arma::uword quadriga_lib::ray_commit(const std::vector<quadriga_lib::path> &paths,
                                     std::vector<quadriga_lib::path> &paths_commit,
                                     const arma::fmat &mesh,
                                     const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
                                     const arma::fvec &center_frequency,
                                     const arma::fmat &orig,
                                     const arma::u32_vec &fbs_ind,
                                     const arma::fmat &trivec,
                                     const arma::fmat &tridir,
                                     const arma::Col<short> &mtl_ind_current,
                                     const arma::fmat &points,
                                     const arma::u32_vec *sub_cloud_index,
                                     const std::vector<bool> *subdiv_flag_in,
                                     float max_path_length, float min_gain_dB, bool ignore_direct_path)
{
    // Mesh validation
    const arma::uword n_mesh = mesh.n_rows;
    if (n_mesh == 0)
        throw std::invalid_argument("Input 'mesh' cannot be empty.");
    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    // Validate mtl_prop, needed for the in-medium attenuation of the last leg
    size_t n_mtl = 0;
    for (const auto &kv : mtl_prop)
        if (!kv.second.empty())
        {
            n_mtl = kv.second.size();
            break;
        }
    if (n_mtl > 32767)
        throw std::invalid_argument("Number of materials cannot exceed 32767.");

    // Frequencies; the path storage layout supports 1 to 127 frequencies
    const arma::uword n_freq = center_frequency.n_elem;
    if (n_freq == 0)
        throw std::invalid_argument("Input 'center_frequency' cannot be empty.");
    if (n_freq > 127)
        throw std::invalid_argument("Input 'center_frequency' cannot have more than 127 elements.");
    if (center_frequency.min() <= 0.0f)
        throw std::invalid_argument("Input 'center_frequency' must be > 0.");
    float fRef_Hz = center_frequency[0];
    float fRef_GHz = fRef_Hz * 1e-9f;

    // Ray count is defined by 'orig'; all other per-ray inputs must match it
    const arma::uword n_ray = orig.n_rows;
    if (n_ray == 0)
        throw std::invalid_argument("Input 'orig' cannot be empty.");
    if (orig.n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns.");
    if (n_ray > 4294967295ull)
        throw std::invalid_argument("Number of rays cannot exceed 2^32-1.");

    if (fbs_ind.n_elem != n_ray)
        throw std::invalid_argument("Input 'fbs_ind' must have n_ray elements.");

    // Medium state words, bit-masked (mat = w & 0x7FFF, flag = w & 0x8000), so negative values are legal
    if (mtl_ind_current.n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_current' must have n_ray elements.");

    // Path storage must match the ray count; the layout is taken from the paths themselves
    if (paths.size() != n_ray)
        throw std::invalid_argument("Input 'paths' must have n_ray elements.");
    if (paths[0].n_freq() != n_freq) // Quick check first path, deep check later
        throw std::invalid_argument("Number of frequencies in 'paths' must match 'center_frequency'.");

    const bool scalar_mode = paths[0].is_scalar();

    // Committed paths are appended, an existing store must use the same layout
    if (!paths_commit.empty() && (paths_commit[0].n_freq() != n_freq || paths_commit[0].is_scalar() != scalar_mode))
        throw std::invalid_argument("Layout of 'paths_commit' must match 'paths'.");

    // Optional subdivision flags, indexed in the full ray set
    const bool has_subdiv = subdiv_flag_in && !subdiv_flag_in->empty();
    if (has_subdiv && (arma::uword)subdiv_flag_in->size() != n_ray)
        throw std::invalid_argument("Input 'subdiv_flag_in' must have n_ray elements.");

    // Termination thresholds
    if (!std::isfinite(min_gain_dB))
        throw std::invalid_argument("Input 'min_gain_dB' must be finite.");
    float min_gain_linear = std::pow(10.0f, 0.1f * min_gain_dB);

    if (!std::isfinite(max_path_length) || max_path_length <= 0.0f)
        throw std::invalid_argument("Input 'max_path_length' must be > 0.");

    // Input validation is delegated to ray_point_intersect for: trivec, tridir, points, sub_cloud_index

    // Compute ray-point intersections
    std::vector<unsigned> hit_index;
    arma::u32_vec hit_offset;
    quadriga_lib::ray_point_intersect(points, orig, trivec, tridir, &hit_index, &hit_offset, nullptr, nullptr, sub_cloud_index);
    const arma::uword n_points = points.n_rows;

    // Number of ray-point pairs found by the intersector
    const size_t n_hit = hit_index.size();
    if (n_hit == 0)
        return 0;

    const unsigned *p_hit = hit_index.data();
    const unsigned *p_off = hit_offset.memptr();
    const unsigned *p_fbs_ind = fbs_ind.memptr();
    const short *p_current = mtl_ind_current.memptr();
    const float *p_orig = orig.memptr();
    const float *p_points = points.memptr();
    const float *p_mesh = mesh.memptr();

    // Per-ray gate. Evaluated once so the pair loop reads a single byte per pair and
    // only touches geometry for rays that can still commit.
    std::vector<qd::bits<uint8_t>> gate(n_ray);
    qd::bits<uint8_t> *p_gate = gate.data();

    // Named bits of the per-ray gate
    constexpr unsigned GATE_ELIGIBLE = 0;  // ray can still commit at all
    constexpr unsigned GATE_IN_MEDIUM = 1; // ray currently travels inside a medium
    constexpr unsigned GATE_HAS_FBS = 2;   // ray has a first-bounce face that can shade

    int bad = 0; // out-of-range flag, reported after the parallel region
#pragma omp parallel for schedule(static) reduction(| : bad)
    for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray)
    {
        const size_t iR = (size_t)i_ray;           // Ray index
        const size_t iFBS = (size_t)p_fbs_ind[iR]; // 1-based, 0 = no hit

        if (iFBS > n_mesh)
        {
            bad |= 1;
            continue;
        }

        if ((size_t(p_current[iR]) & 0x7FFFu) > n_mtl)
        {
            bad |= 2;
            continue;
        }

        if (has_subdiv && (*subdiv_flag_in)[iR]) // reappears as sub-beams next iteration
            continue;

        const quadriga_lib::path &P = paths[iR];

        if (P.length >= max_path_length) // already too long without the final leg
            continue;

        if (ignore_direct_path && P.nREF == 0 && P.nSCT == 0) // covered by the diffraction model
            continue;

        // Optimistic pre-check: the stored length is shorter than the committed one,
        // so this can only over-admit. The exact test happens per pair.
        if (P.calc_gain(fRef_GHz) <= min_gain_linear)
            continue;

        qd::bits<uint8_t> g;
        g.set(GATE_ELIGIBLE);
        g.assign(GATE_IN_MEDIUM, (unsigned(p_current[iR]) & 0x7FFFu) != 0u);
        g.assign(GATE_HAS_FBS, iFBS != 0);
        p_gate[iR] = g;
    }

    if (bad & 1)
        throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");

    if (bad & 2)
        throw std::invalid_argument("Some values in 'mtl_ind_current' exceed the number of materials.");

    // Survivor mask, one bit per ray-point pair. Blocks are a multiple of 8 pairs, so no two threads ever write the same byte.
    // Blocking the flat pair list instead of the point list keeps the load balanced: hit counts per point vary by orders
    // of magnitude, block sizes do not.
    const size_t BLOCK = 1u << 16;
    const size_t n_block = (n_hit + BLOCK - 1) / BLOCK;

    std::vector<qd::bits<uint8_t>> mask((n_hit + 7) / 8);
    std::vector<size_t> block_offset(n_block + 1, 0u);
    qd::bits<uint8_t> *p_mask = mask.data();

#pragma omp parallel for schedule(dynamic)
    for (long long i_block = 0; i_block < (long long)n_block; ++i_block)
    {
        const size_t k0 = (size_t)i_block * BLOCK;
        const size_t k1 = (k0 + BLOCK < n_hit) ? k0 + BLOCK : n_hit;

        // Locate the point that owns the first pair of this block
        size_t i_point = size_t(std::upper_bound(p_off, p_off + n_points + 1, (unsigned)k0) - p_off) - 1;

        // Single-entry face cache: neighboring points are spatially close after
        // segmentation, so consecutive pairs frequently share the same FBS face
        size_t cache_face = (size_t)-1;
        float Nx = 0.0f, Ny = 0.0f, Nz = 0.0f, Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        size_t n_keep = 0;
        for (size_t k = k0; k < k1; ++k)
        {
            while (k >= (size_t)p_off[i_point + 1]) // advance across empty points
                ++i_point;

            const size_t iR = (size_t)p_hit[k]; // Ray index
            const qd::bits<uint8_t> g = p_gate[iR];
            if (!g[GATE_ELIGIBLE])
                continue;

            const float Rx = p_points[i_point], Ry = p_points[i_point + n_points], Rz = p_points[i_point + 2 * n_points];
            const float Ox = p_orig[iR], Oy = p_orig[iR + n_ray], Oz = p_orig[iR + 2 * n_ray];

            // Length of the final leg and the resulting total path length
            const float dx = Rx - Ox, dy = Ry - Oy, dz = Rz - Oz;
            const float seg_length = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float len = paths[iR].length + seg_length;
            if (len >= max_path_length)
                continue;

            // In-medium loss of the final leg, folded in here so the survivor count is exact
            float medium_g = 1.0f;
            if (g[GATE_IN_MEDIUM])
            {
                const unsigned material = unsigned(p_current[iR] & 0x7FFFu);
                medium_g = quadriga_lib::medium_gain(mtl_prop, material, seg_length, fRef_Hz);
            }
            if (medium_g * paths[iR].calc_gain(fRef_GHz, 0, len) <= min_gain_linear)
                continue;

            // Shading: the receiver must lie on the same side of the FBS plane as the ray origin.
            // Only the sign of the dot product matters, so the face normal is left unnormalized.
            if (g[GATE_HAS_FBS])
            {
                const size_t iF = (size_t)p_fbs_ind[iR] - 1;
                if (iF != cache_face)
                {
                    cache_face = iF;
                    Fx = p_mesh[iF], Fy = p_mesh[iF + n_mesh], Fz = p_mesh[iF + 2 * n_mesh];
                    const float E1x = p_mesh[iF + 3 * n_mesh] - Fx,
                                E1y = p_mesh[iF + 4 * n_mesh] - Fy,
                                E1z = p_mesh[iF + 5 * n_mesh] - Fz;
                    const float E2x = p_mesh[iF + 6 * n_mesh] - Fx,
                                E2y = p_mesh[iF + 7 * n_mesh] - Fy,
                                E2z = p_mesh[iF + 8 * n_mesh] - Fz;
                    Nx = E1y * E2z - E1z * E2y;
                    Ny = E1z * E2x - E1x * E2z;
                    Nz = E1x * E2y - E1y * E2x;
                }

                const float sO = Nx * (Ox - Fx) + Ny * (Oy - Fy) + Nz * (Oz - Fz);
                const float sR = Nx * (Rx - Fx) + Ny * (Ry - Fy) + Nz * (Rz - Fz);
                if ((sO < 0.0f) != (sR < 0.0f)) // receiver is behind the face
                    continue;
            }

            p_mask[k >> 3].set(k & 7u); // mark pair k as a survivor
            ++n_keep;
        }

        block_offset[(size_t)i_block + 1] = n_keep;
    }

    // Serial scan over the block counts, gives every block a disjoint output range
    for (size_t i_block = 0; i_block < n_block; ++i_block)
        block_offset[i_block + 1] += block_offset[i_block];

    const size_t n_commit = block_offset[n_block];
    if (n_commit == 0)
        return 0;

    // Committed paths are appended to whatever the caller already holds
    const size_t commit_offset = paths_commit.size();
    paths_commit.resize(commit_offset + n_commit);

    // Receive-side mirror, applied per frequency in EM mode (VV = 1, HH = -1)
    static const float mirror[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f};
    const float *p_mirror = scalar_mode ? nullptr : mirror;

#pragma omp parallel for schedule(dynamic)
    for (long long i_block = 0; i_block < (long long)n_block; ++i_block)
    {
        const size_t k0 = (size_t)i_block * BLOCK;
        const size_t k1 = (k0 + BLOCK < n_hit) ? k0 + BLOCK : n_hit;

        // Same walk as pass B, the mask decides which pairs are written
        size_t i_point = size_t(std::upper_bound(p_off, p_off + n_points + 1, (unsigned)k0) - p_off) - 1;
        size_t i_out = commit_offset + block_offset[(size_t)i_block];

        for (size_t k = k0; k < k1; ++k)
        {
            while (k >= (size_t)p_off[i_point + 1])
                ++i_point;

            if (!p_mask[k / 8][k % 8]) // pair k did not survive
                continue;

            const size_t iR = (size_t)p_hit[k];
            const quadriga_lib::path &P = paths[iR];
            quadriga_lib::path &C = paths_commit[i_out];

            P.duplicate(C);
            C.iC = (unsigned)i_point; // receiver index, the caller owns the mapping
            ++i_out;

            // Length of the final leg, needed for the in-medium attenuation
            const float Rx = p_points[i_point], Ry = p_points[i_point + n_points], Rz = p_points[i_point + 2 * n_points];
            const float dx = Rx - p_orig[iR], dy = Ry - p_orig[iR + n_ray], dz = Rz - p_orig[iR + 2 * n_ray];
            const float seg_length = std::sqrt(dx * dx + dy * dy + dz * dz);
            const unsigned material = unsigned(p_current[iR] & 0x7FFFu);

            // One coefficient update per frequency: mirror and medium loss in a single call
            for (size_t i_freq = 0; i_freq < n_freq; ++i_freq)
            {
                float medium_g = material ? quadriga_lib::medium_gain(mtl_prop, material, seg_length, center_frequency[i_freq]) : 1.0f;
                C.xpr_update(p_mirror, medium_g, i_freq);
            }
        }
    }

    return n_commit;
}
