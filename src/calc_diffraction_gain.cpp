// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

// Helper functions
namespace
{
    // Number formatter
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
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &dest,
    const arma::Mat<dtype> &mesh,
    const arma::uvec &mtl_ind,
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    dtype center_frequency,
    int lod = 2,
    arma::Col<dtype> *gain = nullptr,
    arma::Mat<dtype> *xprmat = nullptr,
    arma::Cube<dtype> *coord = nullptr,
    int verbose = 0,
    const arma::u32_vec *sub_mesh_index = nullptr,
    int use_kernel = 0,
    int gpu_id = 0,
    bool scalar_mode = false,
    double thin_slab_threshold = 0.0);
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face, 0 = no material (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail (0–6), controls `n_path` and `n_seg`; see [[generate_diffraction_paths]]
- **`verbose`** — Verbosity level
- **`sub_mesh_index`** — 0-based sub-mesh index for acceleration; see [[triangle_mesh_segmentation]]; `[n_mesh]`
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). Selects
  interaction type passed to [[ray_mesh_interact]] (4 vs. 1).
- **`thin_slab_threshold`** — Thin-slab (Fabry-Pérot) resolve threshold; 0 = resolve always (default), 1 = resolve never,  see [[ray_state_update]]

## Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear scale; `[n_pos]`
- **`xprmat`** — For EM mode: polarization transfer matrix excluding FSPL, interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`; `[8, n_pos]`;
  For scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`.
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]`

## See also:
- [[generate_diffraction_paths]] (controls path/segment count via `lod`)
- [[triangle_mesh_segmentation]] (generates `sub_mesh_index`)
- [[obj_file_read]] (defines mtl_prop format)
- [[ray_mesh_interact]] (used for media interactions)
MD!*/

template <typename dtype>
void quadriga_lib::calc_diffraction_gain(const arma::Mat<dtype> &orig,
                                         const arma::Mat<dtype> &dest,
                                         const arma::Mat<dtype> &mesh,
                                         const arma::uvec &mtl_ind,
                                         const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                         dtype center_frequency,
                                         int lod,
                                         arma::Col<dtype> *gain,
                                         arma::Mat<dtype> *xprmat,
                                         arma::Cube<dtype> *coord,
                                         int verbose,
                                         const arma::u32_vec *sub_mesh_index,
                                         int use_kernel,
                                         int gpu_id,
                                         bool scalar_mode,
                                         double thin_slab_threshold)
{
    // Check for correct number of columns
    if (orig.n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (dest.n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    const arma::uword n_pos = orig.n_rows;            // Number of positions
    const arma::uword n_mesh = mesh.n_rows;           // Number of mesh elements
    const int interaction_type = scalar_mode ? 4 : 1; // Transmission only
    const arma::uword nXPR = scalar_mode ? 2 : 8;     // Number of columns in xprmat (8 for EM, 2 for scalar)

    // Check for correct number of rows
    if (dest.n_rows != n_pos)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    if (mtl_ind.n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");

    // Frequency in GHz
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");

    // Material indices are carried in signed 16-bit state words by ray_state_update.
    // Range and table validation happens inside ray_mesh_interact / ray_state_update.
    if (!mtl_ind.is_empty() && mtl_ind.max() > 32767)
        throw std::invalid_argument("Material indices must not exceed 32767.");

    // Initialize XPRMAT output
    if (xprmat && (xprmat->n_rows != nXPR || xprmat->n_cols != n_pos))
        xprmat->set_size(nXPR, n_pos);
    if (xprmat) // Initialize to unit matrix
        quadriga_lib::xpr_update<dtype>(*xprmat, nullptr, nullptr, true);

    // Generate diffraction paths
    arma::Cube<dtype> ray_x, ray_y, ray_z, weight;
    if (lod == 0)
        weight.ones(n_pos, 1, 1);
    else
        quadriga_lib::generate_diffraction_paths<dtype>(orig, dest, center_frequency, lod, ray_x, ray_y, ray_z, weight);

    // Dimensions of the diffraction ellipsoid
    const arma::uword n_path = weight.n_cols;  // Number of diffraction arcs
    const arma::uword n_seg = weight.n_slices; // Number of segments
    const arma::uword G = n_pos * n_path;      // Global ray set

    if (n_path > 61) // Just to be sure for future updates
        throw std::invalid_argument("Max. number of paths is currently fixed to 61.");

    if (verbose)
        std::cout << "Estimating diffraction gain with " << n_path << " paths * "
                  << n_seg << " segments for " << MioNum(n_pos) << " positions." << std::endl;

    // Track the state of each path: three signed-short words per ray (see ray_state_update),
    // mat = w & 0x7FFF (0 = outside), flag = w & 0x8000. Zero-initialized = outside.
    arma::Col<short> g_prev(G, arma::fill::zeros); // previous medium + non-parallel flag
    arma::Col<short> g_cur(G, arma::fill::zeros);  // current medium + resolved flag
    arma::Col<short> g_buf(G, arma::fill::zeros);  // next-transition buffer

    // Refracted ray direction, 0-initialized falls back to geometric direction in ray_state_update
    arma::Mat<dtype> g_dir(G, 3, arma::fill::zeros);
    arma::Mat<dtype> g_normal(G, 3, arma::fill::zeros); // Entry face normals, init to 0

    // Cosine of the air-medium angle, needed for calculating the in-medium compression factor
    arma::Col<dtype> g_cosair(G, arma::fill::zeros);

    // Accumulated distance: col 0 = dist_refract, col 1 = dist_geo
    arma::Mat<dtype> g_acc(G, 2, arma::fill::zeros);

    // Allocate memory for continued rays start and end points
    arma::Mat<dtype> c_orig(G, 3, arma::fill::none); // Origin of continued ray
    arma::Mat<dtype> c_dest(G, 3, arma::fill::none); // Destination of continued ray
    arma::uvec c_iRAY(G, arma::fill::none);          // New ray index

    // Spine ray index
    arma::uvec c_iSPINE(G, arma::fill::none);
    arma::uword *p_iSPINE = c_iSPINE.memptr();

    // Pointer to the path weights
    dtype *p_weight = weight.memptr();

    // Pre-compute the AABB of the mesh
    arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb<dtype>(&mesh, sub_mesh_index);

    // Lambda for vector normalization
    auto NORMALIZE = [](double &x, double &y, double &z) -> double
    {
        double len = std::sqrt(x * x + y * y + z * z), s = 1.0 / len;
        x = len > 1e-6 ? x * s : 1.0, y = len > 1e-6 ? y * s : 0.0, z = len > 1e-6 ? z * s : 0.0;
        return len;
    };

    // Test if diffraction paths are blocked - segment by segment
    for (arma::uword iS = 0; iS < n_seg; ++iS)
    {
        bool first_segment = iS == 0;
        bool last_segment = iS == n_seg - 1;

        arma::uword R = 0;       // Number of tracked rays (R-set)
        arma::uvec s_iRAY;       // R to G-set mapping
        arma::Mat<dtype> s_orig; // Origin of paths for the current segment
        arma::Mat<dtype> s_dest; // Destination of paths for the current segment

        // Initialize set of tracked rays
        for (arma::uword iG = 0; iG < G; ++iG) // Iterate through all rays in global G-set
        {
            arma::uword iPos = iG % n_pos;      // Position index in repmat(*orig, n_path, 1) layout
            arma::uword iD = iS * G + iG;       // Current destination index
            arma::uword iO = (iS - 1) * G + iG; // Current origin index
            arma::uword iP = (iS - 2) * G + iG; // Previous origin index

            double power = first_segment ? 1.0 : (dtype)p_weight[iO]; // Ray power from previous segment
            if (power > 1e-20)
            {
                // Previous ray origin
                double Px = (iS > 1) ? (double)ray_x[iP] : (double)orig.at(iPos, 0);
                double Py = (iS > 1) ? (double)ray_y[iP] : (double)orig.at(iPos, 1);
                double Pz = (iS > 1) ? (double)ray_z[iP] : (double)orig.at(iPos, 2);

                // Current ray origin
                double Ox = first_segment ? Px : (double)ray_x[iO];
                double Oy = first_segment ? Py : (double)ray_y[iO];
                double Oz = first_segment ? Pz : (double)ray_z[iO];

                // Current ray destination
                double Dx = last_segment ? (double)dest.at(iPos, 0) : (double)ray_x[iD];
                double Dy = last_segment ? (double)dest.at(iPos, 1) : (double)ray_y[iD];
                double Dz = last_segment ? (double)dest.at(iPos, 2) : (double)ray_z[iD];

                if (!first_segment) // Initialize XPR
                {
                    double Ux = Ox - Px, Uy = Oy - Py, Uz = Oz - Pz; // Previous geometric direction
                    if (NORMALIZE(Ux, Uy, Uz) <= 1e-6)
                        continue; // Co-located orig and dest, drop ray

                    double Vx = Dx - Ox, Vy = Dy - Oy, Vz = Dz - Oz; // Current geometric direction
                    if (NORMALIZE(Vx, Vy, Vz) <= 1e-6)
                        continue; // Co-located orig and dest, drop ray

                    // Rotate the carried refracted direction by the compressed arc bend (inside rays only)
                    short mtl_cur = g_cur[iG] & (short)0x7FFF;
                    if (mtl_cur)
                    {
                        double Ax = Uy * Vz - Uz * Vy, Ay = Uz * Vx - Ux * Vz, Az = Ux * Vy - Uy * Vx; // U x V
                        double axis_len = NORMALIZE(Ax, Ay, Az);
                        double dot = Ux * Vx + Uy * Vy + Uz * Vz;
                        dot = dot > 1.0 ? 1.0 : (dot < -1.0 ? -1.0 : dot);
                        double alpha = std::acos(dot);           // air-referenced bend
                        if (axis_len > 1.0e-6 && alpha > 1.0e-6) // co-linear gate
                        {
                            double n = (double)quadriga_lib::refractive_index<dtype>(mtl_prop, (arma::uword)mtl_cur, center_frequency);
                            double c = (double)g_cosair[iG]; // cos(theta_air) at entry
                            double s2 = 1.0 - (1.0 - c * c) / (n * n);
                            double C = (s2 > 1.0e-12) ? c / (n * std::sqrt(s2)) : 0.0; // s2<=0 => exit-side TIR
                            double beta = C * (double)alpha, cb = std::cos(beta), sb = std::sin(beta);
                            double gx = (double)g_dir.at(iG, 0), gy = (double)g_dir.at(iG, 1), gz = (double)g_dir.at(iG, 2);
                            double kg = (double)(Ax * gx + Ay * gy + Az * gz); // Rodrigues about (Ax,Ay,Az)
                            g_dir.at(iG, 0) = (dtype)(gx * cb + (Ay * gz - Az * gy) * sb + Ax * kg * (1.0 - cb));
                            g_dir.at(iG, 1) = (dtype)(gy * cb + (Az * gx - Ax * gz) * sb + Ay * kg * (1.0 - cb));
                            g_dir.at(iG, 2) = (dtype)(gz * cb + (Ax * gy - Ay * gx) * sb + Az * kg * (1.0 - cb));
                        }
                    }
                }

                // Move the origin and dest by 1 ULP along the OD axis to avoid missing a face
                double ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz;
                double odScale = std::max({std::abs(ODx), std::abs(ODy), std::abs(ODz), 1e-30}); // direction normalizer

                double posScale = std::max({std::abs(Ox), std::abs(Oy), std::abs(Oz), 1.0}); // Scale relative to O
                double offset = 1.0 * posScale * 1.1920929e-7 / odScale;                     // 1 float ULP along OD
                Ox -= offset * ODx, Oy -= offset * ODy, Oz -= offset * ODz;                  // Shift O back

                posScale = std::max({std::abs(Dx), std::abs(Dy), std::abs(Dz), 1.0}); // Scale relative to D
                offset = 1.0 * posScale * 1.1920929e-7 / odScale;                     // 1 float ULP along OD
                Dx += offset * ODx, Dy += offset * ODy, Dz += offset * ODz;           // Shift D forward

                arma::uword iR = R++;
                c_orig.at(iR, 0) = (dtype)Ox, c_orig.at(iR, 1) = (dtype)Oy, c_orig.at(iR, 2) = (dtype)Oz;
                c_dest.at(iR, 0) = (dtype)Dx, c_dest.at(iR, 1) = (dtype)Dy, c_dest.at(iR, 2) = (dtype)Dz;
                c_iRAY.at(iR) = iG;
            }
            else // Clear current segment power
                p_weight[iD] = (dtype)0.0;
        }

        // Compacted R-set
        s_orig = arma::resize(c_orig, R, 3);
        s_dest = arma::resize(c_dest, R, 3);
        s_iRAY = arma::resize(c_iRAY, R, 1);

        // Trace the rays of the current segment. Find where they are blocked by objects.
        // Calculate losses caused by materials until destination point is reached.

        arma::uword iT = 0; // Iteration counter
        while (R > 0)       // Run until there is no ray left to trace
        {
            if (verbose) // Debug output
                std::cout << "  Seg. " << iS << "." << iT << " : " << MioNum(R) << " rays" << std::flush;
            if (verbose == 2)
                std::cout << std::endl;

            // Calculate interaction points of rays and 3D mesh
            arma::Col<unsigned> no_interact, fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect<dtype>(&s_orig, &s_dest, &mesh, nullptr, nullptr,
                                                        &no_interact, &fbs_ind, &sbs_ind,
                                                        sub_mesh_index, &aabb, use_kernel, gpu_id);

            unsigned *p_no_interact = no_interact.memptr(); // Pointer to 'no_interact'
            unsigned *p_fbs_ind = fbs_ind.memptr();         // Pointer to 'fbs_ind'
            unsigned *p_sbs_ind = sbs_ind.memptr();         // Pointer to 'fbs_ind'

            // Declare outputs of ray_mesh_interact
            arma::Mat<dtype> fbsN, sbsN;   // FBS and SBS
            arma::Mat<dtype> origN;        // New origin after transmission, [R,3]
            arma::Mat<dtype> dirN;         // Refracted path direction after mesh interaction, [R,3]
            arma::Col<dtype> gainN;        // Path gain, [R]
            arma::Mat<dtype> xprmatN;      // Path gain, [8,R]
            arma::Col<dtype> fbs_angle;    // Incidence angle at FBS, [R]
            arma::Mat<dtype> normal_vec;   // FBS/SBS normals [Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S], [R,6]
            std::vector<uint8_t> hit_type; // Interaction type code from ray_mesh_interact, [R]

            // Calculate the interface interactions; no compaction
            quadriga_lib::ray_mesh_interact<dtype>(interaction_type, center_frequency, &s_orig, &s_dest, &mesh, &mtl_ind, &mtl_prop,
                                                   &fbs_ind, &sbs_ind, nullptr, nullptr, &origN, nullptr, &fbsN, &sbsN,
                                                   &gainN, &xprmatN, nullptr, nullptr, &fbs_angle, nullptr,
                                                   nullptr, &normal_vec, &hit_type, &dirN, false);

            // Declare outputs of ray_state_update
            arma::Col<short> prev_out, cur_out, buf_out; // State words
            arma::Mat<dtype> acc_out;                    // Accumulated distance
            std::vector<uint8_t> res_type;               // Resolved type

            // Only store resolved type for debugging
            auto ptr_res_type = (verbose == 2 && n_pos == 1) ? &res_type : nullptr;

            // Update ray state
            {
                // Declare current state words (to be read from G-set)
                arma::Col<short> prev_in(R, arma::fill::none), cur_in(R, arma::fill::none), buf_in(R, arma::fill::none);
                arma::Col<short> mtl_fbs(R, arma::fill::none);   // FBS face material, 1-based
                arma::Col<short> mtl_sbs(R, arma::fill::none);   // SBS face material, 1-based
                arma::Mat<dtype> dir_in(R, 3, arma::fill::none); // Physical ray direction
                arma::Mat<dtype> acc_in(R, 2, arma::fill::none); // Accumulated distance

                // Direct pointer access
                short *p_prev_in = prev_in.memptr();
                short *p_cur_in = cur_in.memptr();
                short *p_buf_in = buf_in.memptr();
                short *p_mtl_fbs = mtl_fbs.memptr();
                short *p_mtl_sbs = mtl_sbs.memptr();
                dtype *p_dir_in = dir_in.memptr();
                dtype *p_acc_in = acc_in.memptr();

                for (arma::uword iR = 0; iR < R; ++iR) // R-set
                {
                    arma::uword iG = s_iRAY[iR];                                              // Corresponding index in G-set
                    p_prev_in[iR] = g_prev[iG];                                               // Previous material
                    p_cur_in[iR] = g_cur[iG];                                                 // Current material
                    p_buf_in[iR] = g_buf[iG];                                                 // Buffer material
                    p_mtl_fbs[iR] = p_fbs_ind[iR] ? (short)mtl_ind.at(p_fbs_ind[iR] - 1) : 0; // FBS material
                    p_mtl_sbs[iR] = p_sbs_ind[iR] ? (short)mtl_ind.at(p_sbs_ind[iR] - 1) : 0; // SBS material
                    p_dir_in[iR] = g_dir.at(iG, 0);                                           // Refracted direction (x)
                    p_dir_in[iR + R] = g_dir.at(iG, 1);                                       // Refracted direction (y)
                    p_dir_in[iR + 2 * R] = g_dir.at(iG, 2);                                   // Refracted direction (z)
                    p_acc_in[iR] = g_acc.at(iG, 0);                                           // Refracted distance
                    p_acc_in[iR + R] = g_acc.at(iG, 1);                                       // Geometric distance

                    if (no_interact[iR] == 0) // Overwrite zero-hit normal vectors with the buffered ones
                        normal_vec(iR, 0) = g_normal(iG, 0), normal_vec(iR, 1) = g_normal(iG, 1), normal_vec(iR, 2) = g_normal(iG, 2);
                }

                quadriga_lib::ray_state_update<dtype>(interaction_type, center_frequency,
                                                      &s_orig, &s_dest, &fbsN, &sbsN, &no_interact, // RT geometry
                                                      &fbs_angle, &normal_vec,                      // Incidence angle, normal vector
                                                      &hit_type,                                    // Hit type code
                                                      &mtl_prop, &mtl_fbs, &mtl_sbs,                // FBS/SBS materials
                                                      &prev_in, &cur_in, &buf_in,                   // Previous state, reduced-set
                                                      &dir_in, &acc_in,                             // path_dir_prev, acc_dist_in
                                                      &prev_out, &cur_out, &buf_out,                // Next state words
                                                      &gainN, &xprmatN,                             // gain, xprmat of transition
                                                      &dirN, &acc_out,                              // path_dirN, acc_dist_outN
                                                      ptr_res_type,                                 // resolved_typeN
                                                      nullptr,                                      // No compaction n_rayN == n_ray
                                                      thin_slab_threshold);
            }

            // Pointers
            dtype *p_gainN = gainN.memptr();             // Pointer to 'gainN'
            dtype *p_xprmatN = xprmatN.memptr();         // Pointer to 'xprmatN'
            const uint8_t *p_hit_type = hit_type.data(); // pointer to 'hit_type'

            // Update path weights, taking material effects into account
            arma::uword n_continue = 0, n_spine = 0;
            for (arma::uword iR = 0; iR < R; ++iR) // R-set
            {
                arma::uword iG = s_iRAY[iR];  // Index in G-set
                arma::uword iD = iS * G + iG; // Current destination index

                unsigned nH = p_no_interact[iR];        // Number of mesh-hits between "orig" and "dest"
                int typeH = int(p_hit_type[iR] & 0x1F); // geometry bits only, drop TIR (bit 5)
                short old_cur = g_cur[iG] & (short)0x7FFF;
                short new_cur = cur_out[iR] & (short)0x7FFF;
                bool ray_continues = nH > 2 || (nH == 2 && (typeH == 1 || typeH == 3));

                // Update segment weight with resolved gain
                dtype seg_weight = p_weight[iD] * p_gainN[iR];

                // Account for in-medium gain if the last segment ends inside an object
                // Note: the in-medium gain for intermediate segments is resolved by ray_state_update
                if (last_segment && !ray_continues)
                    seg_weight *= new_cur ? quadriga_lib::medium_gain<dtype>(mtl_prop, (arma::uword)new_cur, acc_out.at(iR, 0), center_frequency) : 1.0;

                // Capture cos(theta_air)
                if (old_cur == 0 && new_cur != 0)                     // air -> medium: capture cos(theta_air)
                    g_cosair[iG] = std::abs(std::sin(fbs_angle[iR])); // = |cos(theta+pi/2)| = |OF·N|, matches ray_state_update
                else if (new_cur == 0)                                // exited to air: clear
                    g_cosair[iG] = (dtype)0;

                // Store entry face normal
                {
                    arma::uword o = ((typeH == 5 || typeH == 23 || typeH == 29) && nH != 0) ? 3 : 0;
                    g_normal(iG, 0) = normal_vec(iR, o);
                    g_normal(iG, 1) = normal_vec(iR, o + 1);
                    g_normal(iG, 2) = normal_vec(iR, o + 2);
                }

                // xprmatN is [nXPR, R], indexed by the R-set. iG < n_pos selects the spine arc (iP == 0) and equals
                // the target position. Compact those columns to the front of xprmatN and record their positions;
                // off-spine arcs carry no polarization state and are dropped.
                if (iG < n_pos) // spine ray: iG is the position index
                {
                    if (n_spine != iR) // move column iR down to slot nS (nS <= iR, never clobbers)
                        std::memcpy(&p_xprmatN[nXPR * n_spine], &p_xprmatN[nXPR * iR], nXPR * sizeof(dtype));
                    p_iSPINE[n_spine] = iG;
                    ++n_spine;
                }

                // Debugging output
                if (verbose == 2 && n_pos == 1)
                {
                    std::cout << "iG " << iG
                              << ", nH " << nH
                              << ", tH " << typeH
                              << ", tR " << (int)res_type[iR]
                              << ", cur " << old_cur << ((cur_out[iR] & (short)0x8000) ? "*" : "") << ">" << new_cur << ((cur_out[iR] & (short)0x8000) ? "*" : "")
                              << ", prev " << (g_prev[iG] & (short)0x7FFF) << ">" << (prev_out[iR] & (short)0x7FFF)
                              << ", buf " << g_buf[iG] << ">" << buf_out[iR]
                              << ", P " << p_weight[iD] << " > " << seg_weight
                              << ", acc (" << g_acc.at(iG, 0) << ", " << g_acc.at(iG, 1) << ") > (" << acc_out.at(iR, 0) << ", " << acc_out.at(iR, 1) << ")"
                              << ", O(" << s_orig.at(iG, 0) << "," << s_orig.at(iG, 1) << "," << s_orig.at(iG, 2) << ")"
                              << ", D(" << s_dest.at(iG, 0) << "," << s_dest.at(iG, 1) << "," << s_dest.at(iG, 2) << ")"
                              << ", F" << fbs_ind[iR] << "-(" << fbsN.at(iG, 0) << "," << fbsN.at(iG, 1) << "," << fbsN.at(iG, 2) << ")"
                              << ", S" << sbs_ind[iR] << "-(" << sbsN.at(iG, 0) << "," << sbsN.at(iG, 1) << "," << sbsN.at(iG, 2) << ")"
                              << ", theta = " << fbs_angle[iR] * 57.29577951308232
                              << std::endl;
                    xprmatN.t().print("XPR = ");
                }

                // Update R-set state words based on outputs of ray_state_update
                g_prev[iG] = prev_out[iR];
                g_cur[iG] = cur_out[iR];
                g_buf[iG] = buf_out[iR];
                g_acc.row(iG) = acc_out.row(iR);
                g_dir.row(iG) = dirN.row(iR);
                if (new_cur == 0) // outside: clear stale direction
                    g_dir.row(iG).zeros();

                // Write updated segment weight back to G-set
                p_weight[iD] = seg_weight;

                // Relaunch when more events remain on this segment
                if (ray_continues && seg_weight > (dtype)1.0e-20)
                {
                    arma::uword iC = n_continue++;
                    c_orig.at(iC, 0) = origN.at(iR, 0), c_orig.at(iC, 1) = origN.at(iR, 1), c_orig.at(iC, 2) = origN.at(iR, 2);
                    c_dest.at(iC, 0) = s_dest.at(iR, 0), c_dest.at(iC, 1) = s_dest.at(iR, 1), c_dest.at(iC, 2) = s_dest.at(iR, 2);
                    c_iRAY.at(iC) = iG;
                }
            }

            // Left-multiply the compacted spine transitions into xprmat
            if (xprmat && n_spine != 0)
            {
                arma::uvec ray_index(p_iSPINE, n_spine, false, true);           // Ray index view
                arma::Mat<dtype> update(p_xprmatN, nXPR, n_spine, false, true); // Xprmat update view
                quadriga_lib::xpr_update<dtype>(*xprmat, &update, nullptr, false, false, false, &ray_index);
            }

            // Add multi-hits to a new launch config
            if (n_continue > 0)
            {
                s_orig = arma::resize(c_orig, n_continue, 3);
                s_dest = arma::resize(c_dest, n_continue, 3);
                s_iRAY = arma::resize(c_iRAY, n_continue, 1);
            }
            R = n_continue;
            ++iT;

            if (verbose == 1) // Debug output
                std::cout << " (" << MioNum(n_continue) << " continued)" << std::endl;
        }
    }

    // Adjust size of the output containers, if needed
    if (gain && gain->n_elem != n_pos)
        gain->set_size(n_pos);
    if (coord && (coord->n_rows != 3 || coord->n_cols != n_seg - 1 || coord->n_slices != n_pos))
        coord->set_size(3, n_seg - 1, n_pos);

    // Write output data
    dtype *p_ray_x = ray_x.memptr(), *p_ray_y = ray_y.memptr(), *p_ray_z = ray_z.memptr();
    dtype *p_gain = gain ? gain->memptr() : nullptr;
    dtype *p_coord = coord ? coord->memptr() : nullptr;

    if (verbose == 2) // Debug output
        std::cout << "Gain:" << std::endl;

    // Compute gain and coordinates
    for (arma::uword iR = 0; iR < n_pos; ++iR)
    {
        dtype scl = (dtype)0.0;
        dtype path_gain[61];

        for (arma::uword iP = 0; iP < n_path; ++iP)
        {
            dtype w = (dtype)1.0;
            arma::uword iG = iP * n_pos + iR;

            if (verbose == 2 && n_pos == 1) // Debug
                std::cout << "iG " << iG << " segP( ";

            for (arma::uword iS = 0; iS < n_seg; ++iS)
            {
                if (verbose == 2 && n_pos == 1) // Debug
                    std::cout << p_weight[iS * G + iG] << " ";
                w *= p_weight[iS * G + iG];
            }
            path_gain[iP] = w;

            if (verbose == 2 && n_pos == 1) // Debug
                std::cout << ") = " << w << std::endl;

            scl += w;
        }

        if (p_gain)
            p_gain[iR] = scl;

        if (verbose == 2) // Debug output
            std::cout << "Total gain = " << scl << std::endl;

        if (p_coord)
        {
            scl = (scl <= 1e-6) ? 1.0 : (dtype)1.0 / scl;
            for (arma::uword iS = 0; iS < n_seg - 1; ++iS)
            {
                dtype x = (dtype)0.0, y = (dtype)0.0, z = (dtype)0.0;
                for (arma::uword iP = 0; iP < n_path; ++iP)
                {
                    arma::uword iG = iS * n_pos * n_path + iP * n_pos + iR;
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

    // Finalize XPRMAT: normalize (remove gain)
    if (xprmat)
    {
        // Normalize XPRMAT to remove the gain
        quadriga_lib::xpr_update<dtype>(*xprmat, nullptr, nullptr, false, true);

        if (verbose == 2 && n_pos == 1) // Debug output
            (*xprmat).t().print("Total xprmat = ");
    }
}

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<float> &orig, const arma::Mat<float> &dest,
                                                  const arma::Mat<float> &mesh, const arma::uvec &mtl_ind,
                                                  const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
                                                  float center_frequency, int lod,
                                                  arma::Col<float> *gain, arma::Mat<float> *xprmat, arma::Cube<float> *coord,
                                                  int verbose, const arma::u32_vec *sub_mesh_index,
                                                  int use_kernel, int gpu_id, bool scalar_mode, double thin_slab_threshold);

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<double> &orig, const arma::Mat<double> &dest,
                                                  const arma::Mat<double> &mesh, const arma::uvec &mtl_ind,
                                                  const std::unordered_map<std::string, std::vector<double>> &mtl_prop,
                                                  double center_frequency, int lod,
                                                  arma::Col<double> *gain, arma::Mat<double> *xprmat, arma::Cube<double> *coord,
                                                  int verbose, const arma::u32_vec *sub_mesh_index,
                                                  int use_kernel, int gpu_id, bool scalar_mode, double thin_slab_threshold);
