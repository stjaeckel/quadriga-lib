// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"

// This is supposed to run on 64-bit only
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");
static_assert(std::is_nothrow_move_constructible_v<quadriga_lib::path>, "path must have a noexcept move ctor, otherwise vector deep-copies on realloc");

// HELPER: Calculate length
static inline float calc_length(float Dx, float Dy, float Dz, // Destination
                                float Ox, float Oy, float Oz) // Origin
{
    float a = Dx - Ox;
    float b = a * a;
    a = Dy - Oy, b += a * a;
    a = Dz - Oz, b += a * a;
    return std::sqrt(b);
}

// HELPER: Calculate direction from O to D
static inline float calc_direction(float Dx, float Dy, float Dz,    // Destination
                                   float Ox, float Oy, float Oz,    // Origin
                                   float &Vx, float &Vy, float &Vz) // Out: normalized direction
{
    Vx = Dx - Ox, Vy = Dy - Oy, Vz = Dz - Oz;
    float len = std::sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
    float scl = (len < 1.0e-30f) ? 0.0f : 1.0f / len;
    Vx *= scl, Vy *= scl, Vz *= scl;
    return len;
}

// HELPER: Vector compaction
template <typename dtype>
static inline arma::Col<dtype> compact(const arma::Col<dtype> &data, const arma::u32_vec &ind)
{
    const long long n_ind = (long long)ind.n_elem;
    arma::Col<dtype> out(ind.n_elem, arma::fill::none);

    const dtype *__restrict p_data = data.memptr();
    const arma::u32 *p_ind = ind.memptr();
    dtype *__restrict p_out = out.memptr();

#pragma omp parallel for schedule(static) if (n_ind >= 51200)
    for (long long i = 0; i < n_ind; ++i)
        p_out[i] = p_data[p_ind[i]];

    return out;
}

// HELPER: std::vector compaction
template <typename dtype>
static inline std::vector<dtype> compact(const std::vector<dtype> &data, const arma::u32_vec &ind)
{
    const long long n_ind = (long long)ind.n_elem;
    std::vector<dtype> out((size_t)n_ind);

    const dtype *__restrict p_data = data.data();
    const arma::u32 *p_ind = ind.memptr();
    dtype *__restrict p_out = out.data();

#pragma omp parallel for schedule(static) if (n_ind >= 51200)
    for (long long i = 0; i < n_ind; ++i)
        p_out[i] = p_data[p_ind[i]];

    return out;
}

// Helper: Matrix compaction
template <typename dtype>
static inline arma::Mat<dtype> compact(const arma::Mat<dtype> &data,
                                       const arma::u32_vec &ind, // row or col indices
                                       bool compact_rows = true) // switch for dimension
{
    const long long n_ind = (long long)ind.n_elem;
    const long long n_rows = (long long)data.n_rows;
    const long long n_cols = (long long)data.n_cols;

    arma::Mat<dtype> out(compact_rows ? ind.n_elem : data.n_rows,
                         compact_rows ? data.n_cols : ind.n_elem,
                         arma::fill::none);

    const dtype *__restrict p_data = data.memptr();
    const arma::u32 *p_ind = ind.memptr();
    dtype *__restrict p_out = out.memptr();

    if (compact_rows) // strided gather within each column
    {
#pragma omp parallel for collapse(2) schedule(static) if (n_ind * n_cols >= 51200)
        for (long long c = 0; c < n_cols; ++c)
            for (long long i = 0; i < n_ind; ++i)
                p_out[c * n_ind + i] = p_data[c * n_rows + (long long)p_ind[i]];
    }
    else // whole columns are contiguous, copy them in one block
    {
#pragma omp parallel for schedule(static) if (n_ind * n_rows >= 51200)
        for (long long i = 0; i < n_ind; ++i)
            std::memcpy(p_out + i * n_rows,
                        p_data + (long long)p_ind[i] * n_rows,
                        (size_t)n_rows * sizeof(dtype));
    }

    return out;
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_progress
Advance a ray set by one interaction, spawning reflected, transmitted, and subdivided rays

- Consumes a launch configuration (origins, destinations, per-ray medium state, and [[path]] storage) and
  returns the next iteration: for every ray that hits the mesh, its reflected and/or transmitted
  continuation(s), plus the four sub-beams of any ray flagged for subdivision. Rays that miss, fall below
  `min_gain_dB`, or reach an interaction/reflection/transmission/subdivision limit are terminated.
- The full pipeline per call is: intersect ([[ray_triangle_intersect]]) → interaction ([[ray_mesh_interact]])
  → state resolve ([[ray_state_update]]) for a reflection pass and a transmission/refraction pass →
  subdivision ([[subdivide_rays]]) → assembly of the new launch configuration.
- The function returns per-stage counts (see Returns); the new configuration holds `n_out = 4·n_subdiv + n_reflect + n_transmit`
  rays, which may exceed or fall short of the `n_ray` passed in. When all four counts are zero the arrays come back empty 
  (column counts preserved), and a subsequent call with an empty orig throws — so callers detect end-of-trace from an empty 
  launch configuration (or an all-zero return) rather than a single count.
- Memory is sized for the worst case but committed lazily: the output is built in a reserved-then-resized
  buffer, inputs are compacted in place on the intersect result before the expensive passes, and dead
  intermediates are released as the function proceeds, so peak footprint stays close to one generation.
  Designed for `n_ray` up to ~10^8; the ray index is 32-bit, so `n_ray` is capped at 2^32-1.
- Geometry is traced once, at the reference frequency `center_frequency[0]`. For the remaining frequencies
  only the polarization/gain coefficient is recomputed and folded into each [[path]]; the per-frequency
  refracted direction and in-medium distance are approximated by the reference-frequency values (see the
  note on `center_frequency`).
- Beam subdivision and beam-front updates is active only when `trivec` and `tridir` are supplied;
  otherwise rays are traced as infinitesimal.

## Declaration:
```
std::array<unsigned, 4> quadriga_lib::ray_progress(
    const arma::fmat &mesh,
    const arma::uvec &mtl_ind,
    const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
    const arma::fvec &center_frequency,
    float Ox, float Oy, float Oz,
    arma::fmat &orig,
    arma::fmat &dest,
    arma::Col<short> &mtl_ind_prev,
    arma::Col<short> &mtl_ind_current,
    arma::Col<short> &mtl_ind_buffer,
    arma::fmat &path_dir_prev,
    arma::fmat &acc_dist,
    std::vector<quadriga_lib::path> &paths,
    arma::fmat *trivec = nullptr,
    arma::fmat *tridir = nullptr,
    const arma::u32_vec *sub_mesh_index = nullptr,
    const arma::fmat *aabb = nullptr,
    uint8_t max_no_interactions = 20,
    uint8_t max_no_reflections = 10,
    uint8_t max_no_transmissions = 10,
    uint8_t max_no_subdivisions = 2,
    float min_gain_dB = -140.0f,
    float subdivision_tolerance_m = 3.0f,
    float thin_slab_threshold = 0.15f,
    bool refraction_mode = true,
    bool scalar_mode = false);
```

## Inputs:
- **`mesh`** — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`. 0 = air
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl` (max 32767)
- **`center_frequency`** — Center frequencies in [Hz]; `[n_freq]`, 1 to 127 entries. `center_frequency[0]` is the reference frequency that defines the traced geometry
- **`Ox`**, **`Oy`**, **`Oz`** — Point-source (transmitter) position in GCS [m]; used to recompute path length at new sub-beam origins
- **`sub_mesh_index`** *(optional)* — Sub-mesh partition offsets for the accelerated intersect; 0-based, strictly increasing, 
  first entry 0; passed to [[ray_triangle_intersect]]; `[n_sub]`. NULL → no partitioning
- **`aabb`** *(optional)* — Axis-aligned bounding box per sub-mesh; `[n_sub, 6]`. Requires `sub_mesh_index`. 
  NULL with a partition present → boxes are computed internally via [[triangle_mesh_aabb]]
- **`max_no_interactions`** — Total interactions (segments) per ray before termination, 0 to 255. 0 disables tracing (returns 0 rays)
- **`max_no_reflections`** — Reflections per ray, 0 to 255. 0 skips the reflection pass
- **`max_no_transmissions`** — Transmissions / refractions per ray, 0 to 255. 0 skips the transmission pass
- **`max_no_subdivisions`** — Beam subdivisions per ray, 0 to 255. 0 (or no beam mode) disables subdivision
- **`min_gain_dB`** — Path gain below which a continuation is not launched, in dB (linear-power threshold applied to the accumulated per-path gain × interaction gain)
- **`subdivision_tolerance_m`** — Maximum beam-tube edge length before a ray is subdivided, in [m]; must be > 0
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold forwarded to [[ray_state_update]] as its `eps`; see there. Default 0.15
- **`refraction_mode`** — `true` = refraction (Snell-bent transmission), `false` = straight-path transmission
- **`scalar_mode`** — `true` = scalar (acoustic) layout, `false` = EM (2×2 Jones). Must match the layout of `paths`

## In/out (launch configuration, updated in place; `[n_ray, …]` on entry, `[n_out, …]` on return):
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`. Defines `n_ray`; must be non-empty
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mtl_ind_prev`**, **`mtl_ind_current`**, **`mtl_ind_buffer`** — Medium-state words (bit-masked: `mat = w & 0x7FFF`, `flag = w & 0x8000`); `[n_ray]`
- **`path_dir_prev`** — Physical ray direction entering the current segment (unit vectors); `[n_ray, 3]`
- **`acc_dist`** — Accumulated in-layer distance; `[n_ray, 2]`; col 1 = refracted distance, col 2 = geometric distance
- **`paths`** — Per-ray [[path]] objects; `n_ray` entries. Frequency count and layout must match `center_frequency` and `scalar_mode`. Terminated paths are freed; surviving paths carry the appended interaction segment and the updated polarization product
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`. Must be supplied together with `tridir`; empty / NULL disables beam tracing
- **`tridir`** *(optional)* — Vertex-ray directions, Cartesian; `[n_ray, 9]`

## Returns:
- Per-stage ray counts `{n_interact, n_subdiv, n_reflect, n_transmit}`:
  - **`n_interact`** — rays that hit the mesh (survived compaction against the intersect result)
  - **`n_subdiv`** — rays flagged for subdivision, each expanded into 4 sub-beams
  - **`n_reflect`** — reflected continuations launched
  - **`n_transmit`** — transmitted / refracted continuations launched
- The size of the new launch configuration is `4·n_subdiv + n_reflect + n_transmit`. All four zero means every ray terminated and the arrays come back empty

## See also:
- [[ray_init]] (produces the initial launch configuration this function advances)
- [[ray_triangle_intersect]] (first/second interaction points)
- [[ray_mesh_interact]] (per-interaction Fresnel/Jones result)
- [[ray_state_update]] (inside/outside state machine and thin-slab resolution)
- [[subdivide_rays]] (adaptive beam-tube refinement)
- [[path]] (per-ray storage object accumulated across generations)
MD!*/

std::array<unsigned, 4> quadriga_lib::ray_progress(const arma::fmat &mesh,
                                                   const arma::uvec &mtl_ind,
                                                   const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
                                                   const arma::fvec &center_frequency,
                                                   float Ox, float Oy, float Oz,
                                                   arma::fmat &orig,
                                                   arma::fmat &dest,
                                                   arma::Col<short> &mtl_ind_prev,
                                                   arma::Col<short> &mtl_ind_current,
                                                   arma::Col<short> &mtl_ind_buffer,
                                                   arma::fmat &path_dir_prev,
                                                   arma::fmat &acc_dist,
                                                   std::vector<quadriga_lib::path> &paths,
                                                   arma::fmat *trivec,
                                                   arma::fmat *tridir,
                                                   const arma::u32_vec *sub_mesh_index,
                                                   const arma::fmat *aabb,
                                                   uint8_t max_no_interactions,
                                                   uint8_t max_no_reflections,
                                                   uint8_t max_no_transmissions,
                                                   uint8_t max_no_subdivisions,
                                                   float min_gain_dB,
                                                   float subdivision_tolerance_m,
                                                   float thin_slab_threshold,
                                                   bool refraction_mode,
                                                   bool scalar_mode)
{
    // Mesh validation
    const arma::uword n_mesh = mesh.n_rows;
    if (n_mesh == 0)
        throw std::invalid_argument("Input 'mesh' cannot be empty.");
    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mtl_ind.n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");

    // Validate mtl_prop
    arma::uword n_mtl = 0;
    for (const auto &kv : mtl_prop)
        if (!kv.second.empty())
        {
            n_mtl = (arma::uword)kv.second.size();
            break;
        }
    if (n_mtl > 32767)
        throw std::invalid_argument("Number of materials cannot exceed 32767.");
    if (n_mtl != 0 && mtl_ind.max() > n_mtl)
        throw std::invalid_argument("Entries of 'mtl_ind' cannot exceed the number of materials.");

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

    if (dest.n_rows != n_ray || dest.n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have size [n_ray, 3].");

    // Medium state words, bit-masked (mat = w & 0x7FFF, flag = w & 0x8000), so negative values are legal
    if (mtl_ind_prev.n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_prev' must have n_ray elements.");

    if (mtl_ind_current.n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_current' must have n_ray elements.");

    if (mtl_ind_buffer.n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_buffer' must have n_ray elements.");

    if (path_dir_prev.n_rows != n_ray || path_dir_prev.n_cols != 3)
        throw std::invalid_argument("Input 'path_dir_prev' must have size [n_ray, 3].");

    if (acc_dist.n_rows != n_ray || acc_dist.n_cols != 2)
        throw std::invalid_argument("Input 'acc_dist' must have size [n_ray, 2].");
    if (acc_dist.min() < 0.0f)
        throw std::invalid_argument("Input 'acc_dist' must be >= 0.");

    // Path storage must match the ray count and the requested layout
    if (paths.size() != n_ray)
        throw std::invalid_argument("Input 'paths' must have n_ray elements.");

    if (paths[0].n_freq() != n_freq) // Quick check first path, deep check later
        throw std::invalid_argument("Number of frequencies in 'paths' must match 'center_frequency'.");
    if (paths[0].is_scalar() != scalar_mode)
        throw std::invalid_argument("Layout of 'paths' must match 'scalar_mode'.");
    const arma::uword nXPR = scalar_mode ? 2 : 8;

    // Optional beam tracing; 'trivec' and 'tridir' must be given together
    const bool has_trivec = trivec && trivec->n_elem != 0;
    const bool has_tridir = tridir && tridir->n_elem != 0;
    const bool beam_mode = has_trivec || has_tridir;

    if (beam_mode && (!has_trivec || !has_tridir))
        throw std::invalid_argument("Inputs 'trivec' and 'tridir' must be provided together.");

    if (beam_mode)
    {
        if (trivec->n_rows != n_ray || trivec->n_cols != 9)
            throw std::invalid_argument("Input 'trivec' must have size [n_ray, 9].");

        if (tridir->n_rows != n_ray || tridir->n_cols != 9)
            throw std::invalid_argument("Input 'tridir' must have size [n_ray, 9] (Cartesian).");
    }

    // Optional sub-mesh partitioning, 0-based start indices into the mesh rows
    arma::uword n_sub = 0;
    if (sub_mesh_index && sub_mesh_index->n_elem != 0)
    {
        n_sub = sub_mesh_index->n_elem;
        const unsigned *p_sub = sub_mesh_index->memptr();

        if (p_sub[0] != 0u)
            throw std::invalid_argument("First element of 'sub_mesh_index' must be 0.");

        for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            if ((arma::uword)p_sub[i_sub] >= n_mesh)
                throw std::invalid_argument("Entries of 'sub_mesh_index' cannot exceed the number of mesh faces.");
            if (i_sub != 0 && p_sub[i_sub] <= p_sub[i_sub - 1])
                throw std::invalid_argument("Entries of 'sub_mesh_index' must be strictly increasing.");
        }
    }

    // Optional axis-aligned bounding boxes, one per sub-mesh
    arma::fmat aabb_local;
    const arma::fmat *aabb_ptr = nullptr;
    if (aabb && aabb->n_elem != 0)
    {
        if (n_sub == 0)
            throw std::invalid_argument("Input 'aabb' requires 'sub_mesh_index'.");
        if (aabb->n_rows != n_sub || aabb->n_cols != 6)
            throw std::invalid_argument("Input 'aabb' must have size [n_sub, 6].");
        aabb_ptr = aabb;
    }
    else if (n_sub)
    {
        aabb_local = quadriga_lib::triangle_mesh_aabb(&mesh, sub_mesh_index);
        aabb_ptr = &aabb_local;
    }

    if (!std::isfinite(min_gain_dB))
        throw std::invalid_argument("Input 'min_gain_dB' must be finite.");
    float min_gain_linear = std::pow(10.0f, 0.1f * min_gain_dB);

    if (subdivision_tolerance_m <= 0.0)
        throw std::invalid_argument("Input 'subdivision_tolerance_m' must be > 0.");

    // Launch config data
    struct launch_config
    {
        std::vector<float> orig;
        std::vector<float> dest;
        std::vector<float> trivec;
        std::vector<float> tridir;
        std::vector<short> mtl_ind_prev;
        std::vector<short> mtl_ind_current;
        std::vector<short> mtl_ind_buffer;
        std::vector<float> path_dir_prev;
        std::vector<float> acc_dist;
        std::vector<quadriga_lib::path> paths;
        bool beam_mode_flag;

        size_t size() { return paths.size(); }

        void reserve(size_t sz, bool beam_mode = true) // Reserve address space
        {
            orig.reserve(3 * sz);
            dest.reserve(3 * sz);
            trivec.reserve(beam_mode ? 9 * sz : 0);
            tridir.reserve(beam_mode ? 9 * sz : 0);
            mtl_ind_prev.reserve(sz);
            mtl_ind_current.reserve(sz);
            mtl_ind_buffer.reserve(sz);
            path_dir_prev.reserve(3 * sz);
            acc_dist.reserve(2 * sz);
            paths.reserve(sz);
            beam_mode_flag = beam_mode;
        }
        launch_config(size_t sz, bool beam_mode = true) { reserve(sz, beam_mode); } // Construct

        void resize(size_t sz) // Allocate
        {
            if (sz < size())
                throw std::invalid_argument("Launch buffer size cannot shrink.");

            orig.resize(3 * sz);
            dest.resize(3 * sz);
            trivec.resize(beam_mode_flag ? 9 * sz : 0);
            tridir.resize(beam_mode_flag ? 9 * sz : 0);
            mtl_ind_prev.resize(sz);
            mtl_ind_current.resize(sz);
            mtl_ind_buffer.resize(sz);
            path_dir_prev.resize(3 * sz);
            acc_dist.resize(2 * sz);
            paths.resize(sz);
        }

        // Armadillo wrappers
        arma::fmat origA(bool copy = false) { return arma::fmat(orig.data(), 3, size(), copy, !copy); }
        arma::fmat destA(bool copy = false) { return arma::fmat(dest.data(), 3, size(), copy, !copy); }
        arma::fmat trivecA(bool copy = false) { return beam_mode_flag ? arma::fmat(trivec.data(), 9, size(), copy, !copy) : arma::fmat(); }
        arma::fmat tridirA(bool copy = false) { return beam_mode_flag ? arma::fmat(tridir.data(), 9, size(), copy, !copy) : arma::fmat(); }
        arma::Col<short> prevA(bool copy = false) { return arma::Col<short>(mtl_ind_prev.data(), size(), copy, !copy); }
        arma::Col<short> currentA(bool copy = false) { return arma::Col<short>(mtl_ind_current.data(), size(), copy, !copy); }
        arma::Col<short> bufferA(bool copy = false) { return arma::Col<short>(mtl_ind_buffer.data(), size(), copy, !copy); }
        arma::fmat dirA(bool copy = false) { return arma::fmat(path_dir_prev.data(), 3, size(), copy, !copy); }
        arma::fmat accA(bool copy = false) { return arma::fmat(acc_dist.data(), 2, size(), copy, !copy); }

        // Raw pointers
        float *origP(size_t offset = 0) { return orig.data() + 3 * offset; }
        float *destP(size_t offset = 0) { return dest.data() + 3 * offset; };
        float *trivecP(size_t offset = 0) { return beam_mode_flag ? trivec.data() + 9 * offset : nullptr; };
        float *tridirP(size_t offset = 0) { return beam_mode_flag ? tridir.data() + 9 * offset : nullptr; };
        short *prevP(size_t offset = 0) { return mtl_ind_prev.data() + offset; };
        short *currentP(size_t offset = 0) { return mtl_ind_current.data() + offset; };
        short *bufferP(size_t offset = 0) { return mtl_ind_buffer.data() + offset; };
        float *dirP(size_t offset = 0) { return path_dir_prev.data() + 3 * offset; };
        float *accP(size_t offset = 0) { return acc_dist.data() + 2 * offset; };
        quadriga_lib::path *pathP(size_t offset = 0) { return paths.data() + offset; };
    };

    // Calculate Ray-Mesh interactions (OMP + AVX or CUDA accelerated, very compute-heavy)
    arma::u32_vec no_interact, fbs_ind, sbs_ind; // FBS and SBS indices, 1-based, 0 = no hit
    if (max_no_interactions)
        quadriga_lib::ray_triangle_intersect<float>(&orig, &dest, &mesh, nullptr, nullptr, &no_interact,
                                                    &fbs_ind, &sbs_ind, sub_mesh_index, aabb_ptr);

    // Count number of rays that interact with mesh, build ray_map from compaction, clear discarded path data
    arma::uword n_interact = 0;
    arma::u32_vec ray_map(n_ray, arma::fill::none);
    if (max_no_interactions)
    {
        arma::u32 cnt = 0u;
        const arma::u32 *pi = no_interact.memptr();
        arma::u32 *po = ray_map.memptr();
        for (unsigned i = 0; i < (unsigned)n_ray; ++i)
        {
            if (pi[i])
            {
                if (cnt != i)
                    paths[cnt] = std::move(paths[i]);
                po[cnt] = i, ++cnt;
            }
            else // clear discontinued paths
                paths[i].free();
        }
        n_interact = (arma::uword)cnt;
        ray_map.resize(n_interact);
    }
    paths.resize(n_interact); // Delete tail after compaction

    if (n_interact == 0) // No ray hits the mesh, all paths are terminated
    {
        orig.set_size(0, 3);
        dest.set_size(0, 3);
        mtl_ind_prev.reset();
        mtl_ind_current.reset();
        mtl_ind_buffer.reset();
        path_dir_prev.set_size(0, 3);
        acc_dist.set_size(0, 2);
        if (beam_mode)
            trivec->set_size(0, 9), tridir->set_size(0, 9);
        paths.clear();
        paths.shrink_to_fit(); // Release the 64 byte per ray headers
        return std::array<unsigned, 4>();
    }

    // Compact inputs, discard all rays that do not hit the mesh
    orig = compact(orig, ray_map);
    dest = compact(dest, ray_map);
    if (beam_mode)
        *trivec = compact(*trivec, ray_map), *tridir = compact(*tridir, ray_map);
    mtl_ind_prev = compact(mtl_ind_prev, ray_map);
    mtl_ind_current = compact(mtl_ind_current, ray_map);
    mtl_ind_buffer = compact(mtl_ind_buffer, ray_map);
    path_dir_prev = compact(path_dir_prev, ray_map);
    acc_dist = compact(acc_dist, ray_map);
    no_interact = compact(no_interact, ray_map);
    fbs_ind = compact(fbs_ind, ray_map);
    sbs_ind = compact(sbs_ind, ray_map);
    ray_map.reset(); // No longer needed

    // Reflection @ reference frequency
    arma::fmat origN, destN;            // Origin and destination after mesh interaction
    arma::fmat fbsN, sbsN;              // FBS and SBS recomputed with higher precision
    arma::fvec gainN;                   // Interaction gain
    arma::fmat xprmatN;                 // Polarization transfer function
    arma::fmat trivecN, tridirN;        // Beam front geometry after mesh interaction
    arma::fvec fbs_angleN;              // Incidence angle at FBS in rad
    arma::fvec edge_lengthN;            // Max edge length of ray tube triangle at new origin
    arma::fmat normal_vecN;             // FBS and SBS normal vectors
    std::vector<uint8_t> interact_type; // Interaction type code
    arma::fmat path_dirN;               // Refraction-correct path direction

    quadriga_lib::ray_mesh_interact<float>((scalar_mode ? 3 : 0),
                                           fRef_Hz, &orig, &dest, &mesh, &mtl_ind, &mtl_prop,
                                           &fbs_ind, &sbs_ind,
                                           (beam_mode ? trivec : nullptr),
                                           (beam_mode ? tridir : nullptr),
                                           &origN, &destN, &fbsN, &sbsN, &gainN, &xprmatN,
                                           (beam_mode ? &trivecN : nullptr),
                                           (beam_mode ? &tridirN : nullptr),
                                           &fbs_angleN, nullptr,
                                           (beam_mode ? &edge_lengthN : nullptr),
                                           &normal_vecN, &interact_type, &path_dirN);

    // Flag rays that should be subdivided
    arma::uword n_subdiv = 0;
    arma::u32_vec subdiv_ind, keep_ind;
    std::vector<bool> subdiv_flag(n_interact); // Init to false, bit-packed, cannot parallel write
    if (beam_mode && max_no_subdivisions)
    {
        subdiv_ind.set_size(n_interact);
        keep_ind.set_size(n_interact);

        const float *p_edge = edge_lengthN.memptr();
        const short *p_current = mtl_ind_current.memptr();
        unsigned *p_subdiv_ind = subdiv_ind.memptr();
        unsigned *p_keep_ind = keep_ind.memptr();

        arma::uword n_keep = 0;
        for (unsigned i_int = 0; i_int < n_interact; ++i_int)
        {
            if (p_current[i_int] == 0 &&                   // Currently outside
                paths[i_int].nSUB < max_no_subdivisions && // Max. subdiv not reached
                p_edge[i_int] > subdivision_tolerance_m)   // Tolerance exceeded
            {
                subdiv_flag[i_int] = true;
                p_subdiv_ind[n_subdiv++] = i_int;
            }
            else
                p_keep_ind[n_keep++] = i_int;
        }
        subdiv_ind.resize(n_subdiv);
        keep_ind.resize(n_keep);
    }
    edge_lengthN.reset(); // No longer needed

    // Buffer for the next state, use std::vector ans reserve address space only (lazy allocation on linux)
    arma::uword n_out_max = 4 * n_subdiv + 2 * (n_interact - n_subdiv);
    auto L = launch_config(n_out_max, beam_mode); // Reserve only

    // Subdivision
    if (n_subdiv)
    {
        // Allocate memory
        L.resize(4 * n_subdiv);

        // Subdivision of origs, dest, trivec, tridir
        arma::fmat L_orig = L.origA();
        arma::fmat L_dest = L.destA();
        arma::fmat L_trivec = L.trivecA();
        arma::fmat L_tridir = L.tridirA();
        quadriga_lib::subdivide_rays<float>(orig, *trivec, *tridir, &dest, &L_orig, &L_trivec, &L_tridir, &L_dest, &subdiv_ind, true);

        // Read pointers
        const unsigned *p_subdiv_ind = subdiv_ind.memptr(); // i_subdiv to i_int map
        const float *p_orig = L.origP();                    // Updated origins
        const float *p_dest = L.destP();                    // Updated origins
        const short *cP = mtl_ind_prev.memptr();
        const short *cC = mtl_ind_current.memptr();
        const short *cB = mtl_ind_buffer.memptr();

        // Write pointers
        short *pP = L.prevP();
        short *pC = L.currentP();
        short *pB = L.bufferP();
        float *p_dir = L.dirP();
        float *p_acc = L.accP();

        // Update path storage, state words, directions
#pragma omp parallel for schedule(static)
        for (long long i_subdiv = 0; i_subdiv < (long long)n_subdiv; ++i_subdiv)
        {
            const arma::uword i_int = p_subdiv_ind[i_subdiv]; // Interaction ID
            size_t n_seg = paths[i_int].n_seg();

            // Create the 4 sub-beams
            for (arma::uword i_sub = 0; i_sub < 4; ++i_sub)
            {
                const arma::uword i_out = 4 * (arma::uword)i_subdiv + i_sub; // Index of the ray in the output
                const arma::uword i_out3 = 3 * i_out;

                L.paths[i_out] = paths[i_int]; // Deep copy (copy assignment)
                ++L.paths[i_out].nSUB;         // Increase subdivision counter

                // New origin at subdivision point
                float Sx = p_orig[i_out3];
                float Sy = p_orig[i_out3 + 1];
                float Sz = p_orig[i_out3 + 2];

                if (n_seg) // Update last interaction point
                {
                    float *crd = L.paths[i_out].coord(n_seg - 1);
                    crd[0] = Sx, crd[1] = Sy, crd[2] = Sz;
                }

                // Update length
                float length = (n_seg <= 1) ? calc_length(Sx, Sy, Sz, Ox, Oy, Oz)
                                            : L.paths[i_out].calc_length(Sx, Sy, Sz, Ox, Oy, Oz);
                L.paths[i_out].length = length;

                // Copy interaction state
                pP[i_out] = cP[i_int];
                pC[i_out] = cC[i_int];
                pB[i_out] = cB[i_int];

                // Update path direction
                float Vx, Vy, Vz;
                calc_direction(p_dest[i_out3], p_dest[i_out3 + 1], p_dest[i_out3 + 2], Sx, Sy, Sz, Vx, Vy, Vz);
                p_dir[i_out3] = Vx, p_dir[i_out3 + 1] = Vy, p_dir[i_out3 + 2] = Vz;

                // Initialize in-object accumulator
                p_acc[2 * i_out] = 0.0f, p_acc[2 * i_out + 1] = 0.0f;
            }
        }
    }
    subdiv_ind.reset(); // No longer needed

    // Compact stream, discard all rays that have been subdivided
    if (double(n_subdiv) / double(n_interact) > 0.05) // only if the cost is worth it
    {
        // Previous launch config, still consumed by ray_state_update
        orig = compact(orig, keep_ind);
        dest = compact(dest, keep_ind);
        mtl_ind_prev = compact(mtl_ind_prev, keep_ind);
        mtl_ind_current = compact(mtl_ind_current, keep_ind);
        mtl_ind_buffer = compact(mtl_ind_buffer, keep_ind);
        path_dir_prev = compact(path_dir_prev, keep_ind);
        acc_dist = compact(acc_dist, keep_ind);
        no_interact = compact(no_interact, keep_ind);
        fbs_ind = compact(fbs_ind, keep_ind);
        sbs_ind = compact(sbs_ind, keep_ind);

        // ray_mesh_interact outputs
        origN = compact(origN, keep_ind);
        destN = compact(destN, keep_ind);
        fbsN = compact(fbsN, keep_ind);
        sbsN = compact(sbsN, keep_ind);
        gainN = compact(gainN, keep_ind);
        xprmatN = compact(xprmatN, keep_ind, false); // rays are in the columns
        fbs_angleN = compact(fbs_angleN, keep_ind);
        normal_vecN = compact(normal_vecN, keep_ind);
        path_dirN = compact(path_dirN, keep_ind);
        interact_type = compact(interact_type, keep_ind);
        if (beam_mode)
            trivecN = compact(trivecN, keep_ind), tridirN = compact(tridirN, keep_ind);

        // Path storage, moved in place: keep_ind is strictly increasing, so source >= target
        const unsigned *p_keep = keep_ind.memptr();
        const size_t n_keep = (size_t)keep_ind.n_elem;
        for (size_t i = 0; i < n_keep; ++i)
            if ((size_t)p_keep[i] != i)
                paths[i] = std::move(paths[p_keep[i]]);
        paths.resize(n_keep); // Delete tail after compaction

        n_interact -= n_subdiv;
        subdiv_flag.assign(n_interact, false); // set false
    }
    keep_ind.reset(); // No longer needed

    // Build material indices
    arma::Col<short> mtl_ind_fbs, mtl_ind_sbs;
    if (n_interact) // All rays may be subdivided
        for (int i = 0; i < 2; ++i)
        {
            arma::Col<short> buf = arma::Col<short>(n_interact, arma::fill::none);
            const arma::uword *rL = mtl_ind.memptr();
            const unsigned *rU = i ? sbs_ind.memptr() : fbs_ind.memptr();
            short *wS = buf.memptr();

#pragma omp parallel for schedule(static) if (n_interact >= 51200)
            for (long long i_int = 0; i_int < (long long)n_interact; ++i_int)
            {
                unsigned i_XBS = rU[i_int];                   // FBS or SBS index
                wS[i_int] = i_XBS ? (short)rL[i_XBS - 1] : 0; // corresponding MTL index
            }

            if (i)
                mtl_ind_sbs = std::move(buf);
            else
                mtl_ind_fbs = std::move(buf);
        }

    // Storage for new ray state
    arma::uword n_reflect = 0, n_transmit = 0;
    arma::Col<short> prevN, currentN, bufferN; // New state words
    arma::fmat acc_distN;                      // Updated accumulated VBS distance

    // Lambda to determine which paths to keep based on the gain, used by reflect and transmit pass
    auto check_gains = [&](bool reflect_pass) -> arma::uword
    {
        // Read pointers
        const float *p_orig = orig.memptr();
        const float *p_fbs = fbsN.memptr();
        const float *p_gainN = gainN.memptr();

        // Write indices
        keep_ind.set_size(n_interact);
        unsigned *p_keep_ind = keep_ind.memptr();

        arma::uword cnt = 0;
        for (unsigned i_int = 0; i_int < n_interact; ++i_int)
        {
            if (subdiv_flag[i_int])
                continue;

            size_t iY = (size_t)i_int + n_interact;
            size_t iZ = iY + n_interact;

            float length = calc_length(p_fbs[i_int], p_fbs[iY], p_fbs[iZ], p_orig[i_int], p_orig[iY], p_orig[iZ]) + paths[i_int].length;
            float gain = paths[i_int].calc_gain(fRef_GHz, 0, length) * p_gainN[i_int];

            if (gain > min_gain_linear && paths[i_int].n_seg() < (size_t)max_no_interactions)
            {
                if (reflect_pass)
                {
                    if (paths[i_int].nREF < max_no_reflections)
                        p_keep_ind[cnt++] = i_int;
                }
                else if (paths[i_int].nTRA < max_no_transmissions)
                    p_keep_ind[cnt++] = i_int;
            }
        }
        keep_ind.resize(cnt);
        return cnt;
    };

    // Lambda to attach paths marked by "keep_ind" to the new launch config
    auto update_launch_config = [&](bool reflect_pass)
    {
        arma::uword n_rays = reflect_pass ? n_reflect : n_transmit;
        if (!n_rays)
            return;

        arma::uword offset = L.size(); // Current size
        L.resize(offset + n_rays);     // Make space for new paths

        // Read pointers
        const unsigned *p_keep_ind = keep_ind.memptr();
        const float *p_origN = origN.memptr();
        const float *p_destN = destN.memptr();
        const float *p_trivecN = beam_mode ? trivecN.memptr() : nullptr;
        const float *p_tridirN = beam_mode ? tridirN.memptr() : nullptr;
        const short *cP = prevN.memptr();
        const short *cC = currentN.memptr();
        const short *cB = bufferN.memptr();
        const float *p_path_dirN = path_dirN.memptr();
        const float *p_acc_distN = acc_distN.memptr();
        const float *p_fbsN = fbsN.memptr();
        const float *p_xpr = xprmatN.memptr();
        const uint8_t *p_type = interact_type.data();

        // Write pointers
        float *p_orig = L.origP(offset);
        float *p_dest = L.destP(offset);
        float *p_trivec = L.trivecP(offset);
        float *p_tridir = L.tridirP(offset);
        short *pP = L.prevP(offset);
        short *pC = L.currentP(offset);
        short *pB = L.bufferP(offset);
        float *p_dir = L.dirP(offset);
        float *p_acc = L.accP(offset);
        quadriga_lib::path *p_path = L.pathP(offset);

#pragma omp parallel for schedule(static)
        for (long long i_ray = 0; i_ray < (long long)n_rays; ++i_ray)
        {
            const size_t i_out = (size_t)i_ray;
            const arma::uword i_out3 = 3 * i_out;
            const arma::uword i_out9 = 9 * i_out;
            const size_t iX = (size_t)p_keep_ind[i_ray];
            const size_t iY = iX + n_interact;
            const size_t iZ = iY + n_interact;

            // Update orig and dest
            p_orig[i_out3] = p_origN[iX];
            p_orig[i_out3 + 1] = p_origN[iY];
            p_orig[i_out3 + 2] = p_origN[iZ];

            p_dest[i_out3] = p_destN[iX];
            p_dest[i_out3 + 1] = p_destN[iY];
            p_dest[i_out3 + 2] = p_destN[iZ];

            // Update trivec and tridir
            if (beam_mode)
                for (size_t i = 0; i < 9; ++i)
                {
                    p_trivec[i_out9 + i] = p_trivecN[iX + i * n_interact];
                    p_tridir[i_out9 + i] = p_tridirN[iX + i * n_interact];
                }

            // Update interaction state
            pP[i_out] = cP[iX];
            pC[i_out] = cC[iX];
            pB[i_out] = cB[iX];

            // Update path direction
            p_dir[i_out3] = p_path_dirN[iX];
            p_dir[i_out3 + 1] = p_path_dirN[iY];
            p_dir[i_out3 + 2] = p_path_dirN[iZ];

            // Update in-object accumulator
            p_acc[2 * i_out] = p_acc_distN[iX];
            p_acc[2 * i_out + 1] = p_acc_distN[iY];

            // Add new segment to path storage and left-multiply XPR matrix at base frequency
            paths[iX].extend(p_path[i_out], p_fbsN[iX], p_fbsN[iY], p_fbsN[iZ], p_type[iX]);
            p_path[i_out].xpr_update(&p_xpr[nXPR * iX]);
            if (reflect_pass)
                ++p_path[i_out].nREF;
            else
                ++p_path[i_out].nTRA;
        }

        // Update xprmat for other frequencies
        // ISSUE: For virtual direction tracking, this should always be straight-through (modes 1/4)
        int interaction_type = scalar_mode ? (refraction_mode ? 5 : 4) : (refraction_mode ? 2 : 1);
        if (reflect_pass)
            interaction_type = scalar_mode ? 3 : 0;

        for (arma::uword i_freq = 1; i_freq < n_freq; ++i_freq)
        {
            quadriga_lib::ray_mesh_interact<float>(interaction_type, center_frequency[i_freq], &orig, &dest,
                                                   &mesh, &mtl_ind, &mtl_prop, &fbs_ind, &sbs_ind, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, nullptr, nullptr, &xprmatN, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, nullptr, &interact_type, &path_dirN);

            // ISSUE: "path_dir_prev" and "acc_dist" are for the base frequency. "path_dirN" is computed at the correct frequency
            // Both probably need to be tracked per-frequency, adding 5 floats per frequency.
            // What is the error, if we ignore this here? "path_dir_prev" and "acc_dist" are for f0, path_dirN if for f[i_freq]
            // We probably only need to track the refracted in-medium distance here instead of refracted + geometric.
            // The physical one follows from orig-fbs. This requires a contract change in ray_state_update,
            // but reduced the total cost per-frequency to 4 floats.

            // Questions: In refract mode, the geometric path follows the refracted path @ base frequency.
            // But at other frequencies, the "virtual" path_dir and acc_dist will differ. Do we always pass straight-through
            // mode here?

            quadriga_lib::ray_state_update<float>(interaction_type, center_frequency[i_freq], &orig, &dest,
                                                  &fbsN, &sbsN, &no_interact, &fbs_angleN, &normal_vecN,
                                                  &interact_type, &mtl_prop, &mtl_ind_fbs, &mtl_ind_sbs,
                                                  &mtl_ind_prev, &mtl_ind_current, &mtl_ind_buffer,
                                                  &path_dir_prev, &acc_dist, nullptr, nullptr, nullptr,
                                                  nullptr, &xprmatN, &path_dirN, &acc_distN,
                                                  nullptr, nullptr, (double)thin_slab_threshold);

            const float *p_xpr_freq = xprmatN.memptr();

#pragma omp parallel for schedule(static) if (n_interact >= 51200)
            for (long long i_ray = 0; i_ray < (long long)n_rays; ++i_ray)
            {
                const size_t iX = (size_t)p_keep_ind[i_ray];
                p_path[i_ray].xpr_update(&p_xpr_freq[nXPR * iX], 1.0, i_freq);
            }
        }
    };

    // Add reflected paths to launch config
    if (max_no_reflections && n_interact)
    {
        // Resolve ray state for reflection pass
        quadriga_lib::ray_state_update<float>((scalar_mode ? 3 : 0),
                                              fRef_Hz, &orig, &dest, &fbsN, &sbsN,
                                              &no_interact, &fbs_angleN, &normal_vecN, &interact_type,
                                              &mtl_prop, &mtl_ind_fbs, &mtl_ind_sbs,
                                              &mtl_ind_prev, &mtl_ind_current, &mtl_ind_buffer,
                                              &path_dir_prev, &acc_dist,
                                              &prevN, &currentN, &bufferN,
                                              &gainN, &xprmatN, &path_dirN, &acc_distN,
                                              &interact_type, // Aliasing OK and deliberate
                                              nullptr, (double)thin_slab_threshold);

        n_reflect = check_gains(true); // Parse gainN and build keep_ind
        gainN.reset();                 // No longer needed
        update_launch_config(true);    // Add paths
    }

    // Add transmitted paths to launch config
    if (max_no_transmissions && n_interact)
    {
        // Compute transmission / refraction into medium
        quadriga_lib::ray_mesh_interact<float>((scalar_mode ? (refraction_mode ? 5 : 4) : (refraction_mode ? 2 : 1)),
                                               fRef_Hz, &orig, &dest, &mesh, &mtl_ind, &mtl_prop,
                                               &fbs_ind, &sbs_ind,
                                               (beam_mode ? trivec : nullptr),
                                               (beam_mode ? tridir : nullptr),
                                               &origN, &destN, &fbsN, &sbsN, &gainN, &xprmatN,
                                               (beam_mode ? &trivecN : nullptr),
                                               (beam_mode ? &tridirN : nullptr),
                                               nullptr, nullptr, nullptr, nullptr, &interact_type, &path_dirN);

        // Resolve ray state for transmission pass
        quadriga_lib::ray_state_update<float>((scalar_mode ? (refraction_mode ? 5 : 4) : (refraction_mode ? 2 : 1)),
                                              fRef_Hz, &orig, &dest, &fbsN, &sbsN,
                                              &no_interact, &fbs_angleN, &normal_vecN, &interact_type,
                                              &mtl_prop, &mtl_ind_fbs, &mtl_ind_sbs,
                                              &mtl_ind_prev, &mtl_ind_current, &mtl_ind_buffer,
                                              &path_dir_prev, &acc_dist,
                                              &prevN, &currentN, &bufferN,
                                              &gainN, &xprmatN, &path_dirN, &acc_distN,
                                              &interact_type, // Aliasing OK and deliberate
                                              nullptr, (double)thin_slab_threshold);

        n_transmit = check_gains(false); // Parse gainN and build keep_ind
        gainN.reset();                   // No longer needed
        update_launch_config(false);     // Add paths
    }

    // Free memory
    origN.reset();
    destN.reset();
    fbsN.reset();
    sbsN.reset();
    xprmatN.reset();
    interact_type.clear();
    prevN.reset();
    currentN.reset();
    bufferN.reset();
    path_dirN.reset();
    acc_distN.reset();
    no_interact.reset();

    // ISSUE: The armadillo transpose on an armadillo view may be slow (no OMP, etc.).
    // Should benchmark and hand-roll if needed.

    // Copy paths to new launch config
    orig = L.origA().t();
    L.orig.clear();

    dest = L.destA().t();
    L.dest.clear();

    mtl_ind_prev = L.prevA(true);
    L.mtl_ind_prev.clear();

    mtl_ind_current = L.currentA(true);
    L.mtl_ind_current.clear();

    mtl_ind_buffer = L.bufferA(true);
    L.mtl_ind_buffer.clear();

    path_dir_prev = L.dirA().t();
    L.path_dir_prev.clear();

    acc_dist = L.accA().t();
    L.acc_dist.clear();

    paths = std::move(L.paths);
    if (beam_mode)
    {
        *trivec = L.trivecA().t();
        L.trivec.clear();
        *tridir = L.tridirA().t();
        L.tridir.clear();
    }

    // Statistics
    std::array<unsigned, 4> stats = {(unsigned)n_interact, (unsigned)n_subdiv, (unsigned)n_reflect, (unsigned)n_transmit};
    return stats;
}
