// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

// FUNCTION: Number formatter
static std::string MioNum(size_t number)
{
    std::string str;
    if (number < 100000)
        str = std::to_string(number);
    else
    {
        double num = std::round(((double)number) / 1.0e4) / 100.0;
        str = std::to_string(num);
        str = num <= 100.0 ? str.substr(0, 5) : str;
        str = num <= 10.0 ? str.substr(0, 4) : str;
        str += " Mio.";
    }
    return str;
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# calc_diffraction_gain
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients, generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [[triangle_mesh_segmentation]]) accelerates computation by skipping triangles whose bounding box does not intersect the TX-RX path

## Declaration:
```
void calc_diffraction_gain(
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *mesh,
    const arma::uvec *mtl_ind,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    dtype center_frequency,
    int lod = 2,
    arma::Col<dtype> *gain = nullptr,
    arma::Cube<dtype> *coord = nullptr,
    int verbose = 0,
    const arma::u32_vec *sub_mesh_index = nullptr,
    int use_kernel = 0,
    int gpu_id = 0,
    bool scalar_mode = false);
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face, 0 = no material (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`center_frequency`** — Center frequency
- **`lod`** *(optional)* — Level of detail (0–6), controls `n_path` and `n_seg`; see [[generate_diffraction_paths]]
- **`verbose`** *(optional)* — Verbosity level
- **`sub_mesh_index`** *(optional)* — 0-based sub-mesh index for acceleration; see [[triangle_mesh_segmentation]]; `[n_mesh]`
- **`use_kernel`** *(optional)* — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable
- **`gpu_id`** *(optional)* — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** *(optional)* — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). Selects
  interaction type passed to [[ray_mesh_interact]] (4 vs. 1).

## Outputs:
- **`gain`** *(optional)* — Diffraction gain per TX-RX pair, linear scale; `[n_pos]`
- **`coord`** *(optional)* — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]`

## See also:
- [[generate_diffraction_paths]] (controls path/segment count via `lod`)
- [[triangle_mesh_segmentation]] (generates `sub_mesh_index`)
- [[obj_file_read]] (defines mtl_prop format)
- [[ray_mesh_interact]] (used for media interactions)
MD!*/

template <typename dtype>
void quadriga_lib::calc_diffraction_gain(const arma::Mat<dtype> *orig,
                                         const arma::Mat<dtype> *dest,
                                         const arma::Mat<dtype> *mesh,
                                         const arma::uvec *mtl_ind,
                                         const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                         dtype center_frequency,
                                         int lod,
                                         arma::Col<dtype> *gain,
                                         arma::Cube<dtype> *coord, int verbose,
                                         const arma::u32_vec *sub_mesh_index,
                                         int use_kernel, int gpu_id, bool scalar_mode)
{
    // Thin-slab resolution threshold for ray_state_update. Values >= 1 disable the Airy
    // resolution entirely and reproduce the legacy calc_diffraction_gain gains.
    const double eps_slab = 1.0;

    // Check for NULL pointers
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mtl_ind == nullptr)
        throw std::invalid_argument("Input 'mtl_ind' cannot be NULL.");
    if (mtl_prop == nullptr)
        throw std::invalid_argument("Input 'mtl_prop' cannot be NULL.");

    // Check for correct number of columns
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    const arma::uword n_pos = orig->n_rows;  // Number of positions
    const arma::uword n_mesh = mesh->n_rows; // Number of mesh elements
    const size_t n_pos_t = (size_t)n_pos;    // Number of positions as size_t
    const int interaction_type = scalar_mode ? 4 : 1;

    // Check for correct number of rows
    if (dest->n_rows != n_pos)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    if (mtl_ind->n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");

    // Frequency in GHz
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");

    // Material indices are carried in signed 16-bit state words by ray_state_update.
    // Range and table validation happens inside ray_mesh_interact / ray_state_update.
    if (!mtl_ind->is_empty() && mtl_ind->max() > 32767)
        throw std::invalid_argument("Material indices must not exceed 32767.");

    // Check range of LOD
    if ((unsigned)lod > 6U)
        throw std::invalid_argument("Input 'lod' must have values in the range 0-6.");

    // Generate diffraction paths
    arma::Cube<dtype> ray_x, ray_y, ray_z, weight;
    if (lod == 0)
        weight.ones(n_pos, 1, 1);
    else
        quadriga_lib::generate_diffraction_paths<dtype>(orig, dest, center_frequency, lod, &ray_x, &ray_y, &ray_z, &weight);

    // Dimensions of the diffraction ellipsoid
    const size_t n_path_t = (size_t)weight.n_cols;
    const size_t n_seg_t = (size_t)weight.n_slices;
    const size_t n_ray_t = n_pos_t * n_path_t;

    if (n_path_t > 61) // Just to be sure for future updates
        throw std::invalid_argument("Max. number of paths is currently fixed to 61.");

    // Track the state of each path: three signed-short words per ray (see ray_state_update),
    // mat = w & 0x7FFF (0 = outside), flag = w & 0x8000. Zero-initialized = outside.
    arma::Col<short> g_prev(n_ray_t, arma::fill::zeros); // previous medium + non-parallel flag
    arma::Col<short> g_cur(n_ray_t, arma::fill::zeros);  // current medium + resolved flag
    arma::Col<short> g_buf(n_ray_t, arma::fill::zeros);  // next-transition buffer

    // Pointer to the path weights
    dtype *p_weight = weight.memptr();

    if (verbose)
        std::cout << "Estimating diffraction gain with " << n_path_t << " paths * "
                  << n_seg_t << " segments for " << MioNum(n_pos_t) << " positions." << std::endl;

    // Pre-compute the AABB of the mesh
    arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb<dtype>(mesh, sub_mesh_index);

    // Test if diffraction paths are blocked - segment by segment
    for (size_t iS = 0; iS < n_seg_t; ++iS)
    {
        // Obtain the origin points of the current segment
        arma::Mat<dtype> s_orig; // Origin of paths for the current segment
        if (iS == 0)
            s_orig = arma::repmat(*orig, n_path_t, 1);
        else
        {
            s_orig.set_size(n_ray_t, 3);
            dtype *p_orig = s_orig.memptr();
            size_t no_bytes = n_ray_t * sizeof(dtype);
            std::memcpy(p_orig, ray_x.slice_memptr(iS - 1), no_bytes);
            std::memcpy(&p_orig[n_ray_t], ray_y.slice_memptr(iS - 1), no_bytes);
            std::memcpy(&p_orig[2 * n_ray_t], ray_z.slice_memptr(iS - 1), no_bytes);
        }

        // Obtain the destination points of the current segment
        arma::Mat<dtype> s_dest; // Destination of paths for the current segment
        if (iS == n_seg_t - 1)
            s_dest = arma::repmat(*dest, n_path_t, 1);
        else
        {
            s_dest.set_size(n_ray_t, 3);
            dtype *p_dest = s_dest.memptr();
            size_t no_bytes = n_ray_t * sizeof(dtype);
            std::memcpy(p_dest, ray_x.slice_memptr(iS), no_bytes);
            std::memcpy(&p_dest[n_ray_t], ray_y.slice_memptr(iS), no_bytes);
            std::memcpy(&p_dest[2 * n_ray_t], ray_z.slice_memptr(iS), no_bytes);
        }

        // Build global ray index for the current segment
        arma::Col<size_t> s_iRAY = arma::regspace<arma::Col<size_t>>(0, n_ray_t - 1);
        size_t n_ray_r = n_ray_t; // Number of rays in reduced set (starts with all rays)

        // Check which rays have been discontinued in a previous segment
        if (iS != 0) // Only for second segment and onwards
        {
            // Allocate memory for continued rays start and end points
            arma::Mat<dtype> c_orig(n_ray_t, 3, arma::fill::none);
            arma::Mat<dtype> c_dest(n_ray_t, 3, arma::fill::none);
            arma::Col<size_t> c_iRAY(n_ray_t); // New ray index

            size_t n_continue = 0;
            size_t previous_segment_ind = (iS - 1) * n_ray_t;
            for (size_t iR = 0; iR < n_ray_r; ++iR) // Iterate through all rays
            {
                size_t iG = s_iRAY.at(iR);
                dtype power = p_weight[previous_segment_ind + iG];
                if (power > (dtype)1.0e-20) // Continue ray
                {
                    size_t iC = n_continue++;
                    c_orig.at(iC, 0) = s_orig.at(iR, 0), c_orig.at(iC, 1) = s_orig.at(iR, 1), c_orig.at(iC, 2) = s_orig.at(iR, 2);
                    c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                    c_iRAY.at(iC) = iG;
                }
                else // Set current segment power to 0 as well
                    p_weight[iS * n_ray_t + iG] = (dtype)0.0;
            }

            // Create reduced set of rays
            if (n_continue < n_ray_t)
            {
                s_orig = arma::resize(c_orig, n_continue, 3);
                s_dest = arma::resize(c_dest, n_continue, 3);
                s_iRAY = arma::resize(c_iRAY, n_continue, 1);
                n_ray_r = n_continue;
            }
        }

        // Trace the rays of the current segment. Find where they are blocked by objects.
        // Calculate losses caused by materials until destination point is reached.

        while (n_ray_r > 0) // Run until there is no ray left to trace
        {
            if (verbose) // Debug output
                std::cout << "  Seg. " << iS << " : " << MioNum(n_ray_r) << " rays" << std::flush;

            // Calculate interaction points of rays and 3D mesh
            arma::Mat<dtype> fbs, sbs;
            arma::Col<unsigned> no_interact, fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect<dtype>(&s_orig, &s_dest, mesh, &fbs, &sbs,
                                                        &no_interact, &fbs_ind, &sbs_ind,
                                                        sub_mesh_index, &aabb, use_kernel, gpu_id);

            // Pointers
            unsigned *p_no_interact = no_interact.memptr(); // Pointer to 'no_interact'
            unsigned *p_fbs_ind = fbs_ind.memptr();         // Pointer to 'fbs_ind'
            unsigned *p_sbs_ind = sbs_ind.memptr();         // Pointer to 'fbs_ind'

            // Create hit index
            size_t no_mesh_hit = 0;                    // Number of mesh-hits
            size_t *p_hit_ind = new size_t[n_ray_r](); // Hit index, initialized to 0
            for (size_t iR = 0; iR < n_ray_r; ++iR)    // Iterate through all rays
                if (p_fbs_ind[iR] != 0U)
                    p_hit_ind[iR] = no_mesh_hit++;

            if (verbose) // Debug output
                std::cout << ", " << MioNum(no_mesh_hit) << " mesh hits" << std::flush;
            if (verbose == 2) // Debug output
                std::cout << std::endl;

            // Calculate transmission gain and in-medium loss
            // - Outputs are only generated when mesh was hit, (origN.n_rows <= s_orig.n_rows)
            arma::Mat<dtype> origN;      // New origin after transmission
            arma::Col<dtype> gainN;      // Transmission gain
            arma::Col<dtype> fbs_angleN; // Incidence angle at FBS
            arma::Col<int> typeN;        // Medium to medium transition indicator

            if (no_mesh_hit != 0)
                quadriga_lib::ray_mesh_interact<dtype>(interaction_type, center_frequency, &s_orig, &s_dest, &fbs, &sbs, mesh, mtl_ind, mtl_prop,
                                                       &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr, &origN, nullptr,
                                                       &gainN, nullptr, nullptr, nullptr, nullptr, &fbs_angleN, nullptr,
                                                       nullptr, nullptr, &typeN);

            // Pointers
            dtype *p_gainN = gainN.memptr(); // Pointer to 'gainN'
            int *p_typeN = typeN.memptr();   // Pointer to 'typeN'

            // Build the compact-set inputs for ray_state_update and patch gainN in place
            arma::Col<short> prev_out, cur_out, buf_out; // new state, compact set [n_rayN]
            if (no_mesh_hit != 0)
            {
                arma::u32_vec ray_ind((arma::uword)no_mesh_hit);    // compact -> reduced-set index
                arma::Col<short> mtl_fbs((arma::uword)no_mesh_hit); // FBS face material, 1-based
                arma::Col<short> mtl_sbs((arma::uword)no_mesh_hit); // SBS face material, 1-based
                arma::Col<short> prev_in((arma::uword)n_ray_r);     // old state, reduced set
                arma::Col<short> cur_in((arma::uword)n_ray_r);
                arma::Col<short> buf_in((arma::uword)n_ray_r);

                for (size_t iR = 0; iR < n_ray_r; ++iR)
                {
                    size_t iG = s_iRAY.at(iR);
                    prev_in.at(iR) = g_prev.at(iG);
                    cur_in.at(iR) = g_cur.at(iG);
                    buf_in.at(iR) = g_buf.at(iG);
                    if (p_fbs_ind[iR] != 0U)
                    {
                        size_t iH = p_hit_ind[iR];
                        ray_ind.at(iH) = (unsigned)iR;
                        mtl_fbs.at(iH) = (short)mtl_ind->at(p_fbs_ind[iR] - 1);
                        mtl_sbs.at(iH) = (p_sbs_ind[iR] == 0U) ? (short)0 : (short)mtl_ind->at(p_sbs_ind[iR] - 1);
                    }
                }

                quadriga_lib::ray_state_update<dtype>(interaction_type, center_frequency,
                                                      &s_orig, &s_dest, &fbs, &sbs, &no_interact,
                                                      &fbs_angleN, &typeN, mtl_prop, &mtl_fbs, &mtl_sbs,
                                                      &prev_in, &cur_in, &buf_in, nullptr,
                                                      &prev_out, &cur_out, &buf_out,
                                                      &gainN, nullptr, &ray_ind, eps_slab);
            }

            // Count double and multi-interactions
            size_t n_continue = 0;                  // Counter for continued rays
            for (size_t iR = 0; iR < n_ray_r; ++iR) // Iterate through all rays
                if (p_no_interact[iR] > 1)
                    ++n_continue;

            // Allocate memory for multi-hit start and end points
            arma::Mat<dtype> c_orig(n_continue, 3, arma::fill::none);
            arma::Mat<dtype> c_dest(n_continue, 3, arma::fill::none);
            arma::Col<size_t> c_iRAY(n_continue); // New ray index

            // Update path weights, taking material effects into account
            n_continue = 0;                         // Reset counter
            for (size_t iR = 0; iR < n_ray_r; ++iR) // Iterate through all rays
            {
                unsigned nH = p_no_interact[iR];           // Number of mesh-hits between "orig" and "dest"
                size_t iG = s_iRAY.at(iR);                 // Ray index in global set
                dtype power = p_weight[iS * n_ray_t + iG]; // Current segment weight

                if (nH == 0) // No interaction: whole-segment in-medium loss is the caller's job
                {
                    arma::uword cur = (arma::uword)(g_cur.at(iG) & (short)0x7FFF);
                    if (cur != 0)
                    {
                        dtype dist = qd_calc_length(s_orig.at(iR, 0), s_orig.at(iR, 1), s_orig.at(iR, 2),
                                                    s_dest.at(iR, 0), s_dest.at(iR, 1), s_dest.at(iR, 2));
                        power *= quadriga_lib::medium_gain<dtype>(*mtl_prop, cur, dist, center_frequency);
                    }
                }
                else // The state machine has patched gainN and produced the new state
                {
                    size_t iH = p_hit_ind[iR];
                    int typeH = p_typeN[iH];
                    power *= p_gainN[iH];

                    g_prev.at(iG) = prev_out.at(iH);
                    g_cur.at(iG) = cur_out.at(iH);
                    g_buf.at(iG) = buf_out.at(iH);

                    // Relaunch when more events remain on this segment: separated double
                    // hits (o-i-o / i-o-i) and all multi-hit rays
                    if (power > (dtype)1.0e-20 &&
                        (nH > 2 || (nH == 2 && (typeH == 1 || typeH == 2))))
                    {
                        size_t iC = n_continue++;
                        c_orig.at(iC, 0) = origN.at(iH, 0), c_orig.at(iC, 1) = origN.at(iH, 1), c_orig.at(iC, 2) = origN.at(iH, 2);
                        c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                        c_iRAY.at(iC) = iG;
                    }

                    // For debugging:
                    if (verbose == 2 && n_pos == 1)
                        std::cout << "nH = " << nH << ", tH = " << typeH
                                  << ", cur = " << (cur_out.at(iH) & (short)0x7FFF)
                                  << ((cur_out.at(iH) & (short)0x8000) ? "*" : "")
                                  << ", prev = " << (prev_out.at(iH) & (short)0x7FFF)
                                  << ", buf = " << buf_out.at(iH)
                                  << ", P = " << power << std::endl;
                }

                // Update segment weight with new power values
                p_weight[iS * n_ray_t + iG] = power;
            }

            if (verbose == 1) // Debug output
                std::cout << " (" << MioNum(n_continue) << " continued)" << std::endl;

            // Add multi-hits to a new launch config
            if (n_continue > 0)
            {
                s_orig = arma::resize(c_orig, n_continue, 3);
                s_dest = arma::resize(c_dest, n_continue, 3);
                s_iRAY = arma::resize(c_iRAY, n_continue, 1);
            }
            n_ray_r = n_continue;

            // Clear hit index
            delete[] p_hit_ind;
        }
    }

    // Adjust size of the output containers, if needed
    const arma::uword n_seg = (arma::uword)n_seg_t - 1;

    if (gain != nullptr && gain->n_elem != n_pos)
        gain->set_size(n_pos);
    if (coord != nullptr && (coord->n_rows != 3 || coord->n_cols != n_seg || coord->n_slices != n_pos))
        coord->set_size(3, n_seg, n_pos);

    // Write output data
    dtype *p_ray_x = ray_x.memptr(), *p_ray_y = ray_y.memptr(), *p_ray_z = ray_z.memptr();
    dtype *p_gain = (gain == nullptr) ? nullptr : gain->memptr();
    dtype *p_coord = (coord == nullptr) ? nullptr : coord->memptr();

    if (p_gain != nullptr || p_coord != nullptr)
        for (size_t iR = 0; iR < n_pos_t; ++iR)
        {
            dtype scl = (dtype)0.0;
            dtype path_gain[61];

            for (size_t iP = 0; iP < n_path_t; ++iP)
            {
                dtype w = (dtype)1.0;
                size_t iG = iP * n_pos_t + iR;
                for (size_t iS = 0; iS < n_seg_t; ++iS)
                    w *= p_weight[iS * n_ray_t + iG];
                path_gain[iP] = w;
                scl += w;
            }

            if (p_gain != nullptr)
                p_gain[iR] = scl;

            if (p_coord != nullptr)
            {
                scl = (dtype)1.0 / scl;
                for (size_t iS = 0; iS < n_seg_t - 1; ++iS)
                {
                    dtype x = (dtype)0.0, y = (dtype)0.0, z = (dtype)0.0;
                    for (size_t iP = 0; iP < n_path_t; ++iP)
                    {
                        size_t iG = iS * n_pos_t * n_path_t + iP * n_pos_t + iR;
                        x += p_ray_x[iG] * path_gain[iP];
                        y += p_ray_y[iG] * path_gain[iP];
                        z += p_ray_z[iG] * path_gain[iP];
                    }
                    x *= scl, y *= scl, z *= scl;
                    *p_coord++ = x;
                    *p_coord++ = y;
                    *p_coord++ = z;
                }
            }
        }
}

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                                  const arma::Mat<float> *mesh, const arma::uvec *mtl_ind,
                                                  const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                                  float center_frequency, int lod,
                                                  arma::Col<float> *gain, arma::Cube<float> *coord, int verbose,
                                                  const arma::u32_vec *sub_mesh_index, int use_kernel, int gpu_id, bool scalar_mode);

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                                  const arma::Mat<double> *mesh, const arma::uvec *mtl_ind,
                                                  const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                                  double center_frequency, int lod,
                                                  arma::Col<double> *gain, arma::Cube<double> *coord, int verbose,
                                                  const arma::u32_vec *sub_mesh_index, int use_kernel, int gpu_id, bool scalar_mode);
