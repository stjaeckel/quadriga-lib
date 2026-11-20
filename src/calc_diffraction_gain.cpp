// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_site.hpp"

namespace
{
    // Ray counts for the verbose output, e.g. "1.25 Mio."
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
    double center_freq,
    int lod = 2,
    arma::Col<dtype> *gain = nullptr,
    arma::Mat<dtype> *polmat = nullptr,
    arma::Cube<dtype> *coord = nullptr,
    int verbose = 0,
    const arma::u32_vec &sub_mesh_ind = arma::u32_vec(),
    int use_kernel = 0,
    int gpu_id = 0,
    bool scalar_mode = false,
    double thin_slab_threshold = 0.0,
    bool apply_fspl = false,
    bool apply_phase = false);
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face, 0 = no material (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`center_freq`** — Center frequency in [Hz]; in scalar mode the acoustic frequency
- **`lod`** *(optional)* — Level of detail (0–7), controls `n_path` and `n_seg`; 0 = direct ray only (`n_path = 1`, `n_seg = 1`);
  1–7 as in the table of [[generate_diffraction_paths]]
- **`verbose`** *(optional)* — Verbosity level
- **`sub_mesh_ind`** *(optional)* — 0-based sub-mesh index for acceleration; see [[triangle_mesh_segmentation]]; `[n_sub]`
- **`use_kernel`** *(optional)* — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; auto as in [[ray_triangle_intersect]]
- **`gpu_id`** *(optional)* — CUDA device ID; ignored for non-CUDA kernels
- **`scalar_mode`** *(optional)* — If `true`, uses scalar transmission (TE-only reflection coefficient,
  energy-conservation transmission) instead of EM TE/TM averaging. Default `false` (EM mode). Selects
  interaction type passed to [[ray_mesh_interact]] (4 vs. 1).
- **`thin_slab_threshold`** *(optional)* — Thin-slab (Fabry-Pérot) resolve threshold on the round-trip in-slab amplitude `ρ`: a series is summed where `ρ ≥ eps`; 0 = resolve always (default); `eps ≥ 1` re-emits every series whose `ρ` is below one, and a round trip at or above one is always summed; see [[ray_state_update]]
- **`apply_fspl`** *(optional)* — If `true`, folds the geometric spreading loss into `gain`, so that `gain` becomes an
  absolute propagation gain instead of a value relative to FSPL. Default `false`. The reference distance is
  the direct TX-RX distance, matching the `path_length` definition used by [[qrt_file_read]] for a transmission-only 
  path. In EM mode the loss is `32.45 + 20·log10(f_GHz) + 20·log10(d_m)` [dB]; in scalar mode it is 
  `20·log10(d_m) + α(f)·d_m` [dB], with `α` from ISO 9613-1 at T = 20 °C, RH = 50 %, p = 1 atm, evaluated at
  `center_freq`.
- **`apply_phase`** *(optional)* — If `true`, multiplies `polmat` by the propagation phase term `exp(-j·2π·d/λ)`, using the
  same direct TX-RX distance `d`. Default `false`. The wavelength is `λ = c / center_freq` with the speed
  of light (EM) or the speed of sound, 343 m/s (scalar mode). Since the term has unit
  magnitude, `polmat` stays normalized. This is the phase contribution that
  [[get_channels_spherical]] would otherwise add from `path_length`; enabling it here gives the complete
  omni-radiator coefficient without running the channel generator.

## Outputs:
- **`gain`** *(optional)* — Diffraction gain per TX-RX pair, linear power scale; `[n_pos]`. Relative to FSPL by default,
  i.e. bounded by 0 and 1; absolute propagation gain when `apply_fspl = true`.
- **`polmat`** *(optional)* — Normalized polarization transfer matrix: the amplitude carries no path gain (max column
  power is 1), only the relative polarization coupling and phase. The `gain` output holds the power.
  For EM mode: interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` with labels `<rx><tx>`:
  rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx), the row order
  of `polmat` in [[get_channels_spherical]]; `[8, n_pos]`;
  For scalar mode: scalar pressure coefficient `[Re Im]`; `[2, n_pos]`.
  With `apply_phase = true`, the propagation phase `exp(-j·2π·d/λ)` is applied on top; the normalization
  is unaffected because that term has unit magnitude.
- **`coord`** *(optional)* — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]` with `n_seg` from `lod`
  (e.g. `[3, 2, n_pos]` for lod 2, `[3, 9, n_pos]` for lod 7, `[3, 0, n_pos]` for lod 0)

## See also:
- [[generate_diffraction_paths]] (controls path/segment count via `lod`)
- [[triangle_mesh_segmentation]] (generates `sub_mesh_ind`)
- [[obj_file_read]] (defines mtl_prop format)
- [[ray_mesh_interact]] (used for media interactions)
- [[qrt_file_read]] (same path-loss model, used by `apply_fspl`)
- [[get_channels_spherical]] (applies the same propagation phase as `apply_phase`)
MD!*/

template <typename dtype>
void quadriga_lib::calc_diffraction_gain(const arma::Mat<dtype> &orig,
                                         const arma::Mat<dtype> &dest,
                                         const arma::Mat<dtype> &mesh,
                                         const arma::uvec &mtl_ind,
                                         const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                         double center_freq_in,
                                         int lod,
                                         arma::Col<dtype> *gain,
                                         arma::Mat<dtype> *polmat,
                                         arma::Cube<dtype> *coord,
                                         int verbose,
                                         const arma::u32_vec &sub_mesh_ind,
                                         int use_kernel,
                                         int gpu_id,
                                         bool scalar_mode,
                                         double thin_slab_threshold,
                                         bool apply_fspl,
                                         bool apply_phase)
{
    const dtype center_freq = dtype(center_freq_in);
    if (orig.n_elem == 0 || dest.n_elem == 0)
        throw std::invalid_argument("Inputs 'orig' and 'dest' cannot be empty.");
    if (lod < 0 || lod > 7)
        throw std::invalid_argument("Input 'lod' must be 0-7.");

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
    const arma::uword nXPR = scalar_mode ? 2 : 8;     // Number of rows in polmat (8 for EM, 2 for scalar)

    // Check for correct number of rows
    if (dest.n_rows != n_pos)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' do not match.");
    if (mtl_ind.n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");

    // Frequency in GHz
    if (!std::isfinite(center_freq) || center_freq <= (dtype)0.0)
        throw std::invalid_argument("Input 'center_freq' must be finite and > 0.");

    // Material indices are carried as 14 bits of the signed 16-bit state words by ray_state_update.
    // Range and table validation happens inside ray_mesh_interact / ray_state_update.
    if (!mtl_ind.is_empty() && mtl_ind.max() > 16383)
        throw std::invalid_argument("Input 'mtl_ind' must not exceed 16383.");

    qd_check_alias(gain, "gain", &orig, &dest, &mesh);
    qd_check_alias(polmat, "polmat", &orig, &dest, &mesh);
    qd_check_alias_outputs(gain, "gain", polmat, "polmat");

    // Initialize XPRMAT output
    if (polmat)
    {
        polmat->set_size(nXPR, n_pos);
        quadriga_lib::polmat_update<dtype>(*polmat, {}, nullptr, true); // Initialize to unit matrix
    }

    // Generate diffraction paths
    arma::Cube<dtype> ray_x, ray_y, ray_z, weight;
    if (lod == 0)
        weight.ones(n_pos, 1, 1);
    else
        quadriga_lib::generate_diffraction_paths<dtype>(orig, dest, center_freq, lod, ray_x, ray_y, ray_z, weight, scalar_mode);

    // Dimensions of the diffraction ellipsoid
    const arma::uword n_path = weight.n_cols;  // Number of diffraction arcs
    const arma::uword n_seg = weight.n_slices; // Number of segments
    const arma::uword G = n_pos * n_path;      // Global ray set

    if (n_path > 61) // lod 4 gives the most arcs (61)
        throw std::invalid_argument("Max. number of paths is currently fixed to 61.");

    if (verbose)
        std::cout << "Estimating diffraction gain with " << n_path << " paths * "
                  << n_seg << " segments for " << MioNum(n_pos) << " positions." << std::endl;

    // Track the state of each path: three signed-short words per ray (see ray_state_update),
    // mat = w & 0x3FFF (0 = outside), flags in bits 14 and 15. Zero-initialized = outside.
    arma::Col<short> g_prev(G, arma::fill::zeros); // previous medium + non-parallel flag
    arma::Col<short> g_cur(G, arma::fill::zeros);  // current medium + resolved flag
    arma::Col<short> g_buf(G, arma::fill::zeros);  // next-transition buffer

    // Refracted ray direction, 0-initialized falls back to geometric direction in ray_state_update
    arma::Mat<dtype> g_dir(G, 3, arma::fill::zeros);
    arma::Mat<dtype> g_normal(G, 3, arma::fill::zeros); // Entry face normals, init to 0

    // Cosine of the air-medium angle, needed for calculating the in-medium compression factor
    arma::Col<dtype> g_cosair(G, arma::fill::zeros);

    // Accumulated in-medium distance, tracked separately for the geometric and the refracted path
    arma::Col<dtype> g_acc_geo(G, arma::fill::zeros);
    arma::Col<dtype> g_acc_ref(G, arma::fill::zeros);

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
    const arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb<dtype>(mesh, sub_mesh_ind);

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
                    short mtl_cur = g_cur[iG] & (short)0x3FFF;
                    if (mtl_cur)
                    {
                        double Ax, Ay, Az;
                        qd_crossp(Ux, Uy, Uz, Vx, Vy, Vz, &Ax, &Ay, &Az);
                        double axis_len = NORMALIZE(Ax, Ay, Az);
                        double dot = Ux * Vx + Uy * Vy + Uz * Vz;
                        dot = dot > 1.0 ? 1.0 : (dot < -1.0 ? -1.0 : dot);
                        double alpha = std::acos(dot);           // air-referenced bend
                        if (axis_len > 1.0e-6 && alpha > 1.0e-6) // co-linear gate
                        {
                            double n = (double)quadriga_lib::refractive_index<dtype>(mtl_prop, (arma::uword)mtl_cur, center_freq, scalar_mode);
                            double c = (double)g_cosair[iG]; // cos(theta_air) at entry
                            double s2 = 1.0 - (1.0 - c * c) / (n * n);
                            double C = (s2 > 1.0e-12) ? c / (n * std::sqrt(s2)) : 0.0; // s2<=0 => exit-side TIR
                            double rx, ry, rz;
                            qd_rotate_around_axis((double)g_dir.at(iG, 0), (double)g_dir.at(iG, 1), (double)g_dir.at(iG, 2),
                                                  Ax, Ay, Az, C * (double)alpha, &rx, &ry, &rz);
                            g_dir.at(iG, 0) = (dtype)rx, g_dir.at(iG, 1) = (dtype)ry, g_dir.at(iG, 2) = (dtype)rz;
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
            arma::Col<unsigned> n_interact, fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect<dtype>(s_orig, s_dest, mesh, nullptr, nullptr,
                                                        &n_interact, &fbs_ind, &sbs_ind,
                                                        sub_mesh_ind, aabb, use_kernel, gpu_id);

            unsigned *p_no_interact = n_interact.memptr();
            unsigned *p_fbs_ind = fbs_ind.memptr();
            unsigned *p_sbs_ind = sbs_ind.memptr();

            // Declare outputs of ray_mesh_interact
            arma::Mat<dtype> fbsN, sbsN;   // FBS and SBS
            arma::Mat<dtype> origN;        // New origin after transmission, [R,3]
            arma::Mat<dtype> dirN;         // Refracted path direction after mesh interaction, [R,3]
            arma::Col<dtype> gainN;        // Path gain, [R]
            arma::Mat<dtype> polmatN;      // Polarization transfer matrix, [nXPR, R]
            arma::Col<dtype> fbs_angle;    // Incidence angle at FBS, [R]
            arma::Mat<dtype> normal_vec;   // FBS/SBS normals [Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S], [R,6]
            std::vector<uint8_t> hit_type; // Interaction type code from ray_mesh_interact, [R]

            // Calculate the interface interactions; no compaction
            quadriga_lib::ray_mesh_interact<dtype>(interaction_type, center_freq, &s_orig, &s_dest, &mesh, &mtl_ind, &mtl_prop,
                                                   &fbs_ind, &sbs_ind, nullptr, nullptr, &origN, nullptr, &fbsN, &sbsN,
                                                   &gainN, &polmatN, nullptr, nullptr, &fbs_angle, nullptr,
                                                   nullptr, &normal_vec, &hit_type, &dirN, false);

            // Declare outputs of ray_state_update
            arma::Col<short> prev_out, cur_out, buf_out; // State words
            arma::Col<dtype> acc_geo_out, acc_ref_out;   // Accumulated distance
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
                arma::Col<dtype> acc_geo_in(R, arma::fill::none); // Accumulated geometric distance
                arma::Col<dtype> acc_ref_in(R, arma::fill::none); // Accumulated refracted distance

                short *p_prev_in = prev_in.memptr();
                short *p_cur_in = cur_in.memptr();
                short *p_buf_in = buf_in.memptr();
                short *p_mtl_fbs = mtl_fbs.memptr();
                short *p_mtl_sbs = mtl_sbs.memptr();
                dtype *p_dir_in = dir_in.memptr();
                dtype *p_acc_geo_in = acc_geo_in.memptr();
                dtype *p_acc_ref_in = acc_ref_in.memptr();

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
                    p_acc_ref_in[iR] = g_acc_ref[iG];                                         // Refracted distance
                    p_acc_geo_in[iR] = g_acc_geo[iG];                                         // Geometric distance

                    if (n_interact[iR] == 0) // Overwrite zero-hit normal vectors with the buffered ones
                        normal_vec(iR, 0) = g_normal(iG, 0), normal_vec(iR, 1) = g_normal(iG, 1), normal_vec(iR, 2) = g_normal(iG, 2);
                }

                quadriga_lib::ray_state_update<dtype>(interaction_type, center_freq,
                                                      &s_orig, &s_dest, &fbsN, &sbsN, &n_interact,  // RT geometry
                                                      &fbs_angle, &normal_vec,                      // Incidence angle, normal vector
                                                      &hit_type,                                    // Hit type code
                                                      &mtl_prop, &mtl_fbs, &mtl_sbs,                // FBS/SBS materials
                                                      &prev_in, &cur_in, &buf_in,                   // Previous state, reduced-set
                                                      &dir_in, &acc_geo_in, &acc_ref_in,            // path_dir_prev, acc_dist in
                                                      &prev_out, &cur_out, &buf_out,                // Next state words
                                                      &gainN, &polmatN,                             // gain, polmat of transition
                                                      &dirN, &acc_geo_out, &acc_ref_out,            // path_dirN, acc_dist out
                                                      ptr_res_type,                                 // resolved_typeN
                                                      nullptr,                                      // No compaction n_rayN == n_ray
                                                      thin_slab_threshold);
            }

            dtype *p_gainN = gainN.memptr();
            dtype *p_xprmatN = polmatN.memptr();
            const uint8_t *p_hit_type = hit_type.data();

            // Update path weights, taking material effects into account
            arma::uword n_continue = 0, n_spine = 0;
            for (arma::uword iR = 0; iR < R; ++iR) // R-set
            {
                arma::uword iG = s_iRAY[iR];  // Index in G-set
                arma::uword iD = iS * G + iG; // Current destination index

                unsigned nH = p_no_interact[iR];        // Number of mesh-hits between "orig" and "dest"
                int typeH = int(p_hit_type[iR] & 0x1F); // geometry bits only, drop TIR (bit 5)
                short old_cur = g_cur[iG] & (short)0x3FFF;
                short new_cur = cur_out[iR] & (short)0x3FFF;
                bool ray_continues = nH > 2 || (nH == 2 && (typeH == 1 || typeH == 3));

                // Update segment weight with resolved gain
                dtype seg_weight = p_weight[iD] * p_gainN[iR];

                // Account for in-medium gain if the last segment ends inside an object
                // Note: the in-medium gain for intermediate segments is resolved by ray_state_update
                if (last_segment && !ray_continues)
                    seg_weight *= new_cur ? quadriga_lib::medium_gain<dtype>(mtl_prop, (arma::uword)new_cur, acc_ref_out[iR], center_freq, scalar_mode) : dtype(1);

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

                // polmatN is [nXPR, R], indexed by the R-set. iG < n_pos selects the spine arc (iP == 0) and equals
                // the target position. Compact those columns to the front of polmatN and record their positions;
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
                              << ", prev " << (g_prev[iG] & (short)0x3FFF) << ">" << (prev_out[iR] & (short)0x3FFF)
                              << ", buf " << g_buf[iG] << ">" << buf_out[iR]
                              << ", P " << p_weight[iD] << " > " << seg_weight
                              << ", acc (" << g_acc_ref[iG] << ", " << g_acc_geo[iG] << ") > (" << acc_ref_out[iR] << ", " << acc_geo_out[iR] << ")"
                              << ", O(" << s_orig.at(iR, 0) << "," << s_orig.at(iR, 1) << "," << s_orig.at(iR, 2) << ")"
                              << ", D(" << s_dest.at(iR, 0) << "," << s_dest.at(iR, 1) << "," << s_dest.at(iR, 2) << ")"
                              << ", F" << fbs_ind[iR] << "-(" << fbsN.at(iR, 0) << "," << fbsN.at(iR, 1) << "," << fbsN.at(iR, 2) << ")"
                              << ", S" << sbs_ind[iR] << "-(" << sbsN.at(iR, 0) << "," << sbsN.at(iR, 1) << "," << sbsN.at(iR, 2) << ")"
                              << ", theta = " << fbs_angle[iR] * 57.29577951308232
                              << std::endl;
                    polmatN.t().print("polmatN = ");
                }

                // Update R-set state words based on outputs of ray_state_update
                g_prev[iG] = prev_out[iR];
                g_cur[iG] = cur_out[iR];
                g_buf[iG] = buf_out[iR];
                g_acc_geo[iG] = acc_geo_out[iR];
                g_acc_ref[iG] = acc_ref_out[iR];
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

            // Left-multiply the compacted spine transitions into polmat
            if (polmat && n_spine != 0)
            {
                arma::uvec ray_ind(p_iSPINE, n_spine, false, true);             // Ray index view
                arma::Mat<dtype> update(p_xprmatN, nXPR, n_spine, false, true); // Xprmat update view
                quadriga_lib::polmat_update<dtype>(*polmat, update, nullptr, false, false, false, &ray_ind);
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

    // Size the output containers
    if (gain)
        gain->set_size(n_pos);
    if (coord)
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
            scl = (scl <= 1e-6) ? dtype(1) : (dtype)1.0 / scl;
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
    if (polmat)
    {
        quadriga_lib::polmat_update<dtype>(*polmat, {}, nullptr, false, true);

        if (verbose == 2 && n_pos == 1) // Debug output
            (*polmat).t().print("Total polmat = ");
    }

    // Optionally fold the geometric spreading loss into 'gain' and the propagation phase into 'polmat'.
    // Both use the direct TX-RX distance, which is the 'path_length' that qrt_file_read and
    // get_channels_spherical see for a transmission-only path (the spine ray is a straight line).
    if ((apply_fspl && p_gain) || (apply_phase && polmat))
    {
        const double freq = (double)center_freq;

        // Path-loss reference at 1 m and air absorption, shared with the QRT reader
        const double gain_at_1m = qd_gain_at_1m(freq, scalar_mode);

        // Air absorption (ISO 9613-1) in scalar mode
        const double alpha_dB_per_m = scalar_mode ? qd_iso9613_alpha(freq) : 0.0;

        // Wave speed of the medium: light (EM) or sound (scalar mode)
        const double wave_speed = scalar_mode ? qd_speed_of_sound : qd_speed_of_light;
        const double wave_number = 6.283185307179586 * freq / wave_speed;
        const double wavelength = wave_speed / freq;

        dtype *p_xprmat = polmat ? polmat->memptr() : nullptr;

        for (arma::uword iR = 0; iR < n_pos; ++iR)
        {
            double dist = qd_calc_length((double)orig.at(iR, 0), (double)orig.at(iR, 1), (double)orig.at(iR, 2),
                                         (double)dest.at(iR, 0), (double)dest.at(iR, 1), (double)dest.at(iR, 2));

            if (dist < 1.0e-9) // Co-located TX and RX, path loss and phase are undefined
                continue;

            if (apply_fspl && p_gain)
            {
                double gainPL = gain_at_1m - 20.0 * std::log10(dist) - alpha_dB_per_m * dist;
                p_gain[iR] *= (dtype)std::pow(10.0, 0.1 * gainPL);
            }

            if (apply_phase && p_xprmat)
            {
                // exp(-j*k*d); wrap the distance to one wavelength first to keep the argument small
                double phase = -wave_number * std::fmod(dist, wavelength);
                double cp = std::cos(phase), sp = std::sin(phase);

                dtype *col = &p_xprmat[iR * nXPR];
                for (arma::uword k = 0; k < nXPR; k += 2) // Interleaved complex pairs
                {
                    double re = (double)col[k], im = (double)col[k + 1];
                    col[k] = (dtype)(re * cp - im * sp);
                    col[k + 1] = (dtype)(re * sp + im * cp);
                }
            }
        }

        if (verbose == 2 && n_pos == 1 && polmat) // Debug output
            (*polmat).t().print("Total polmat incl. phase = ");
    }
}

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<float> &orig, const arma::Mat<float> &dest,
                                                  const arma::Mat<float> &mesh, const arma::uvec &mtl_ind,
                                                  const std::unordered_map<std::string, std::vector<float>> &mtl_prop,
                                                  double center_freq, int lod,
                                                  arma::Col<float> *gain, arma::Mat<float> *polmat, arma::Cube<float> *coord,
                                                  int verbose, const arma::u32_vec &sub_mesh_ind,
                                                  int use_kernel, int gpu_id, bool scalar_mode, double thin_slab_threshold,
                                                  bool apply_fspl, bool apply_phase);

template void quadriga_lib::calc_diffraction_gain(const arma::Mat<double> &orig, const arma::Mat<double> &dest,
                                                  const arma::Mat<double> &mesh, const arma::uvec &mtl_ind,
                                                  const std::unordered_map<std::string, std::vector<double>> &mtl_prop,
                                                  double center_freq, int lod,
                                                  arma::Col<double> *gain, arma::Mat<double> *polmat, arma::Cube<double> *coord,
                                                  int verbose, const arma::u32_vec &sub_mesh_ind,
                                                  int use_kernel, int gpu_id, bool scalar_mode, double thin_slab_threshold,
                                                  bool apply_fspl, bool apply_phase);
