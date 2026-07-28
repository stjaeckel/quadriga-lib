// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# subdivide_rays
Subdivide ray beams into four smaller sub-beams

- Each triangular beam is split into 4 sub-beams, `n_rayN = 4 * n_subdiv` rays are written to the output
- Rays can be selected by `index` (0-based list), all rays are subdivided if it is not given
- `n_subdiv` is the number of selected rays
- `tridir` format is auto-detected: spherical `[n_ray, 6]` or Cartesian `[n_ray, 9]`, output matches the input format
- Pre-allocated outputs that can hold all new rays are reused as they are and must have the same number of rows,
  the new rays are written to the first `n_rayN` rows, leaving the remaining rows untouched. Smaller buffers are
  re-allocated to that size, discarding their content.
- Internal math is done in double precision, the new origins stay within 1 ULP of the wavefront plane
  spanned by `orig` and `trivec`, no offset is applied along the propagation direction
- The direction values of the 3 original vertices are passed through unchanged, only the 3 new edge-midpoint
  directions are calculated, hence repeated subdivision does not accumulate rounding errors at the corners
- If `transposed_output` is true, all outputs are written transposed, i.e. the rays are in the columns
  and the components in the rows, e.g. `origN` becomes `[3, n_rayN]`

## Declaration:
```
arma::uword quadriga_lib::subdivide_rays(
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &trivec,
    const arma::Mat<dtype> &tridir,
    const arma::Mat<dtype> *dest = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    const arma::u32_vec *index = nullptr);
```

## Inputs:
- **`orig`** — Ray origin points in GCS; `[n_ray, 3]`
- **`trivec`** — Vectors from origin to the wavefront vertices, columns `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`; `[n_ray, 9]`
- **`tridir`** — Vertex-ray directions, spherical `[v1az v1el v2az v2el v3az v3el]` or Cartesian
  `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`; `[n_ray, 6]` or `[n_ray, 9]`
- **`dest`** (optional) — Ray destination points, ignored if empty; `[n_ray, 3]`
- **`index`** (optional) — 0-based indices of the rays that should be subdivided, may repeat indices and
  determines the output order; Invalid indices raise an exception after the loop has finished,
  outputs may be partially written by then; `[n_subdiv]`
- **`transposed_output`** (optional) — If true, the outputs are written transposed with the rays in the
  columns, e.g. `origN` becomes `[3, n_rayN]`; default = `false`

## Outputs:
- **`origN`** — Subdivided ray origins, centroids of the sub-beam wavefronts; `[n_rayN, 3]` or `[3, n_rayN]` for `transposed_output`
- **`trivecN`** — Subdivided wavefront vectors, relative to `origN`; `[n_rayN, 9]` or `[9, n_rayN]` for `transposed_output`
- **`tridirN`** — Subdivided vertex-ray directions, same format as `tridir`; `[n_rayN, 6]` or `[n_rayN, 9]` or `[6/9, n_rayN]` for `transposed_output`
- **`destN`** — Subdivided destinations, left untouched if `dest` was `nullptr` or empty; `[n_rayN, 3]` or `[3, n_rayN]` for `transposed_output`

## Returns:
- `n_rayN` — Number of rays written to the output, `4 * n_subdiv`

## See also:
- [[icosphere]] (generate initial beams)
- [[ray_point_intersect]] (beam-sample-point interaction)
- [[ray_triangle_intersect]] (beam-triangle interaction)
MD!*/

template <typename dtype>
arma::uword quadriga_lib::subdivide_rays(const arma::Mat<dtype> &orig,
                                         const arma::Mat<dtype> &trivec,
                                         const arma::Mat<dtype> &tridir,
                                         const arma::Mat<dtype> *dest,
                                         arma::Mat<dtype> *origN,
                                         arma::Mat<dtype> *trivecN,
                                         arma::Mat<dtype> *tridirN,
                                         arma::Mat<dtype> *destN,
                                         const arma::u32_vec *index,
                                         bool transposed_output)
{
    // Check input dimensions
    if (orig.n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");

    const arma::uword n_ray = orig.n_rows; // Number of rays

    if (trivec.n_cols != 9)
        throw std::invalid_argument("Input 'trivec' must have 9 columns.");
    if (tridir.n_cols != 6 && tridir.n_cols != 9)
        throw std::invalid_argument("Input 'tridir' must have 6 or 9 columns.");
    if (trivec.n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
    if (tridir.n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");

    const bool have_dest = (dest && !dest->is_empty());
    if (have_dest && dest->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'dest' does not match number of rows in 'orig'.");
    if (have_dest && dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");

    // Ray selection
    const unsigned *p_ind = nullptr;
    arma::uword n_subdiv = n_ray;
    if (index && index->n_elem != 0)
        p_ind = index->memptr(), n_subdiv = (arma::uword)index->n_elem;
    const arma::uword n_rayN = 4 * n_subdiv; // Number of rays in the output

    if (n_rayN == 0) // Nothing to do, outputs are left untouched
        return 0;

    if (origN == nullptr && trivecN == nullptr && tridirN == nullptr && (!have_dest || destN == nullptr))
        return n_rayN; // No outputs were requested

    // Indicator for Cartesian format
    const bool cartesian_format = (tridir.n_cols == 9);
    const arma::uword n_comp = cartesian_format ? 3 : 2; // Direction components per vertex

    // Output buffers, 'destN' is size-checked but never written if 'dest' is not given
    arma::Mat<dtype> *out_list[4] = {origN, trivecN, tridirN, destN};
    const arma::uword out_cols[4] = {3, 9, tridir.n_cols, 3};
    const char *out_name[4] = {"origN", "trivecN", "tridirN", "destN"};

    // Number of ray slots in the output buffers, rays are in the columns for transposed output.
    // Buffers that can hold all new rays are reused as they are and must have the same size.
    arma::uword n_slotN = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (out_list[i] == nullptr)
            continue;

        const arma::uword n_slot = transposed_output ? out_list[i]->n_cols : out_list[i]->n_rows;

        if (n_slot < n_rayN) // Too small, will be re-allocated
            continue;

        if (n_slotN == 0)
            n_slotN = n_slot;
        else if (n_slot != n_slotN)
            throw std::invalid_argument("Pre-allocated output matrices must have the same number of rays.");
    }
    n_slotN = (n_slotN == 0) ? n_rayN : n_slotN;

    // Resize the outputs, reused buffers keep their content beyond ray 'n_rayN'
    for (int i = 0; i < 4; ++i)
    {
        if (out_list[i] == nullptr || (i == 3 && !have_dest)) // 'destN' is not written if 'dest' is not given
            continue;

        const arma::uword n_slot = transposed_output ? out_list[i]->n_cols : out_list[i]->n_rows;
        const arma::uword n_fix = transposed_output ? out_list[i]->n_rows : out_list[i]->n_cols;

        if (n_slot != n_slotN)
        {
            if (transposed_output)
                out_list[i]->set_size(out_cols[i], n_slotN);
            else
                out_list[i]->set_size(n_slotN, out_cols[i]);
        }
        else if (n_fix != out_cols[i])
            throw std::invalid_argument("Output '" + std::string(out_name[i]) + "' has an unsupported size.");
    }

    // Memory pointers (inputs)
    const dtype *p_orig = orig.memptr();
    const dtype *p_trivec = trivec.memptr();
    const dtype *p_tridir = tridir.memptr();
    const dtype *p_dest = have_dest ? dest->memptr() : nullptr;

    // Memory pointers (outputs)
    dtype *p_origN = origN ? origN->memptr() : nullptr;
    dtype *p_trivecN = trivecN ? trivecN->memptr() : nullptr;
    dtype *p_tridirN = tridirN ? tridirN->memptr() : nullptr;
    dtype *p_destN = (have_dest && destN) ? destN->memptr() : nullptr;

    // Address of component 'c' of output ray 'i_out' is "p[i_out * s_ray + c * s_comp]"
    const arma::uword s_comp = transposed_output ? 1 : n_slotN;        // Step between the components
    const arma::uword s_ray_3 = transposed_output ? 3 : 1;             // Step between the rays, 'origN' and 'destN'
    const arma::uword s_ray_9 = transposed_output ? 9 : 1;             // Step between the rays, 'trivecN'
    const arma::uword s_ray_t = transposed_output ? tridir.n_cols : 1; // Step between the rays, 'tridirN'

    // Vertex numbering: 0 = V1, 1 = V2, 2 = V3, 3 = V12, 4 = V13, 5 = V23
    const int mid_a[3] = {0, 0, 1};                                     // First parent vertex of a midpoint
    const int mid_b[3] = {1, 2, 2};                                     // Second parent vertex of a midpoint
    const int vid[4][3] = {{0, 3, 4}, {4, 3, 5}, {4, 5, 2}, {3, 1, 5}}; // Vertices of the 4 sub-beams

    const long long n_subdiv_l = (long long)n_subdiv; // Signed loop bound for OpenMP 2.0
    int out_of_bound = 0;                             // Set by any thread that hits an invalid index

    // Iterate through all selected rays
#pragma omp parallel for schedule(static) reduction(| : out_of_bound)
    for (long long i_subdiv = 0; i_subdiv < n_subdiv_l; ++i_subdiv)
    {
        const arma::uword i_ray = p_ind ? (arma::uword)p_ind[i_subdiv] : (arma::uword)i_subdiv;

        if (i_ray >= n_ray) // Must be skipped, loading the ray would access invalid memory
        {
            out_of_bound = 1;
            continue;
        }

        // Load beam origin
        const double Ox = (double)p_orig[i_ray],
                     Oy = (double)p_orig[i_ray + n_ray],
                     Oz = (double)p_orig[i_ray + 2 * n_ray];

        // Load destination and calculate the length from orig to dest
        double length = 0.0;
        if (p_destN)
        {
            const double Ux = (double)p_dest[i_ray] - Ox,
                         Uy = (double)p_dest[i_ray + n_ray] - Oy,
                         Uz = (double)p_dest[i_ray + 2 * n_ray] - Oz;
            length = std::sqrt(Ux * Ux + Uy * Uy + Uz * Uz);
        }

        double Wx[6], Wy[6], Wz[6]; // Wavefront vertex positions
        double Dx[6], Dy[6], Dz[6]; // Wavefront vertex directions (normalized)
        dtype To[6][3];             // Vertex directions in output format

        // Load the 3 beam vertices, the sum "O + T" is exact in double precision
        for (int v = 0; v < 3; ++v)
        {
            const arma::uword o = (arma::uword)(3 * v) * n_ray;
            Wx[v] = Ox + (double)p_trivec[i_ray + o];
            Wy[v] = Oy + (double)p_trivec[i_ray + o + n_ray];
            Wz[v] = Oz + (double)p_trivec[i_ray + o + 2 * n_ray];
        }

        // Load the direction vectors at the 3 beam vertices
        for (int v = 0; v < 3; ++v)
        {
            const arma::uword o = (arma::uword)v * n_comp * n_ray;

            if (cartesian_format)
            {
                double dx = (double)p_tridir[i_ray + o],
                       dy = (double)p_tridir[i_ray + o + n_ray],
                       dz = (double)p_tridir[i_ray + o + 2 * n_ray];

                const double scl = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz); // Normalize
                Dx[v] = dx * scl, Dy[v] = dy * scl, Dz[v] = dz * scl;
            }
            else // Spherical format
            {
                const double az = (double)p_tridir[i_ray + o],
                             el = (double)p_tridir[i_ray + o + n_ray];

                const double scl = std::cos(el);
                Dx[v] = std::cos(az) * scl, Dy[v] = std::sin(az) * scl, Dz[v] = std::sin(el);
            }

            // Pass the original vertex directions through unchanged
            if (p_tridirN)
                for (arma::uword c = 0; c < n_comp; ++c)
                    To[v][c] = p_tridir[i_ray + o + c * n_ray];
        }

        // Calculate the 3 additional vertices at the edge midpoints
        for (int m = 0; m < 3; ++m)
        {
            const int a = mid_a[m], b = mid_b[m], v = m + 3;

            Wx[v] = 0.5 * (Wx[a] + Wx[b]);
            Wy[v] = 0.5 * (Wy[a] + Wy[b]);
            Wz[v] = 0.5 * (Wz[a] + Wz[b]);

            // The factor 0.5 of the midpoint direction is irrelevant after normalization
            double dx = Dx[a] + Dx[b], dy = Dy[a] + Dy[b], dz = Dz[a] + Dz[b];

            const double scl = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
            dx *= scl, dy *= scl, dz *= scl;
            Dx[v] = dx, Dy[v] = dy, Dz[v] = dz;

            if (p_tridirN && cartesian_format)
                To[v][0] = (dtype)dx, To[v][1] = (dtype)dy, To[v][2] = (dtype)dz;
            else if (p_tridirN) // Convert to spherical coordinates
            {
                dz = (dz > 1.0) ? 1.0 : ((dz < -1.0) ? -1.0 : dz); // Guard against rounding
                To[v][0] = (dtype)std::atan2(dy, dx), To[v][1] = (dtype)std::asin(dz);
            }
        }

        // Centroids of the 4 sub-beams, written as exact barycentric combinations of the
        // original vertices, e.g. the centroid of sub-beam 0 is (4*W1 + W2 + W3) / 6.
        // This keeps them in the plane spanned by W1, W2, W3 to within 2 ULP (double).
        const double Sx = Wx[0] + Wx[1] + Wx[2],
                     Sy = Wy[0] + Wy[1] + Wy[2],
                     Sz = Wz[0] + Wz[1] + Wz[2];

        const double Cx[4] = {(Sx + 3.0 * Wx[0]) / 6.0, Sx / 3.0, (Sx + 3.0 * Wx[2]) / 6.0, (Sx + 3.0 * Wx[1]) / 6.0};
        const double Cy[4] = {(Sy + 3.0 * Wy[0]) / 6.0, Sy / 3.0, (Sy + 3.0 * Wy[2]) / 6.0, (Sy + 3.0 * Wy[1]) / 6.0};
        const double Cz[4] = {(Sz + 3.0 * Wz[0]) / 6.0, Sz / 3.0, (Sz + 3.0 * Wz[2]) / 6.0, (Sz + 3.0 * Wz[1]) / 6.0};

        // Create the 4 sub-beams
        for (arma::uword i_sub = 0; i_sub < 4; ++i_sub)
        {
            const arma::uword i_out = 4 * (arma::uword)i_subdiv + i_sub; // Index of the ray in the output

            // Round the new origin to the output type before calculating 'trivecN', this way
            // "origN + trivecN" reproduces the wavefront vertices to within 0.5 ULP of 'trivecN'
            const dtype ox = (dtype)Cx[i_sub], oy = (dtype)Cy[i_sub], oz = (dtype)Cz[i_sub];
            const double oxd = (double)ox, oyd = (double)oy, ozd = (double)oz;

            // Write new origin point
            if (p_origN)
            {
                const arma::uword o = i_out * s_ray_3;
                p_origN[o] = ox;
                p_origN[o + s_comp] = oy;
                p_origN[o + 2 * s_comp] = oz;
            }

            // Write new trivec
            if (p_trivecN)
                for (arma::uword v = 0; v < 3; ++v)
                {
                    const int k = vid[i_sub][v];
                    const arma::uword o = i_out * s_ray_9 + 3 * v * s_comp;

                    p_trivecN[o] = (dtype)(Wx[k] - oxd);
                    p_trivecN[o + s_comp] = (dtype)(Wy[k] - oyd);
                    p_trivecN[o + 2 * s_comp] = (dtype)(Wz[k] - ozd);
                }

            // Write new tridir
            if (p_tridirN)
                for (arma::uword v = 0; v < 3; ++v)
                {
                    const int k = vid[i_sub][v];
                    const arma::uword o = i_out * s_ray_t + v * n_comp * s_comp;

                    for (arma::uword c = 0; c < n_comp; ++c)
                        p_tridirN[o + c * s_comp] = To[k][c];
                }

            // Write new destination
            if (p_destN)
            {
                const int k0 = vid[i_sub][0], k1 = vid[i_sub][1], k2 = vid[i_sub][2];

                // Ray direction at the center of the sub-beam
                const double dx = Dx[k0] + Dx[k1] + Dx[k2],
                             dy = Dy[k0] + Dy[k1] + Dy[k2],
                             dz = Dz[k0] + Dz[k1] + Dz[k2];

                const double scl = length / std::sqrt(dx * dx + dy * dy + dz * dz);
                const arma::uword o = i_out * s_ray_3;

                p_destN[o] = (dtype)(oxd + dx * scl);
                p_destN[o + s_comp] = (dtype)(oyd + dy * scl);
                p_destN[o + 2 * s_comp] = (dtype)(ozd + dz * scl);
            }
        }
    }

    if (out_of_bound != 0)
        throw std::invalid_argument("Indices cannot exceed number of rays.");

    return n_rayN;
}

template arma::uword quadriga_lib::subdivide_rays(const arma::Mat<float> &orig, const arma::Mat<float> &trivec, const arma::Mat<float> &tridir, const arma::Mat<float> *dest,
                                                  arma::Mat<float> *origN, arma::Mat<float> *trivecN, arma::Mat<float> *tridirN, arma::Mat<float> *destN,
                                                  const arma::u32_vec *index, bool transposed_output);

template arma::uword quadriga_lib::subdivide_rays(const arma::Mat<double> &orig, const arma::Mat<double> &trivec, const arma::Mat<double> &tridir, const arma::Mat<double> *dest,
                                                  arma::Mat<double> *origN, arma::Mat<double> *trivecN, arma::Mat<double> *tridirN, arma::Mat<double> *destN,
                                                  const arma::u32_vec *index, bool transposed_output);