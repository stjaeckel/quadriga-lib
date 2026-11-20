// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"

#include <iomanip>

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for Blender)
- Tubes are color-coded by path gain using a selected colormap; radius also scales with gain
- Paths below `gain_min_dB` are excluded; `max_paths` limits total count
- Paths with non-finite coordinates are skipped
- Throws with the `is_valid()` message of an invalid channel

## Declaration:
```
void quadriga_lib::channel_export_obj_file(
    const std::string &fn,
    const channel<dtype> &chan,
    arma::uword max_paths = 0,
    dtype gain_max_dB = -60.0,
    dtype gain_min_dB = -140.0,
    const std::string &colormap = "jet",
    const arma::uvec &i_snap = arma::uvec(),
    dtype radius_max = 0.05,
    dtype radius_min = 0.01,
    arma::uword n_edges = 5);
```

## Inputs:
- **`fn`** — Output `.obj` file path
- **`chan`** — Channel with ray tracing data (`n_interact`, `interact_coord`), `center_freq` and coefficients or `polmat`; see [[channel]]
- **`max_paths`** *(optional)* — Max paths to export; `0` includes all paths above `gain_min_dB`
- **`gain_max_dB`** *(optional)* — Upper gain threshold in dB for color/radius mapping; higher values are clipped
- **`gain_min_dB`** *(optional)* — Lower gain threshold in dB; paths below this are excluded; must be smaller than `gain_max_dB`
- **`colormap`** *(optional)* — Colormap name; see [[colormap]] for supported options
- **`i_snap`** *(optional)* — 0-based snapshot indices to include; empty exports all snapshots
- **`radius_max`** *(optional)* — Tube radius at maximum gain
- **`radius_min`** *(optional)* — Tube radius at minimum gain
- **`n_edges`** *(optional)* — Vertices per tube cross-section; must be ≥ 3

## See also:
- [[path_to_tube]] (generates tube geometry from path data)
- [[colormap]] (colormap lookup used for coloring)
MD!*/

template <typename dtype>
void quadriga_lib::channel_export_obj_file(const std::string &fn, const channel<dtype> &chan,
                                           arma::uword max_paths,
                                           dtype gain_max_dB, dtype gain_min_dB,
                                           const std::string &colormap,
                                           const arma::uvec &i_snap_sel,
                                           dtype radius_max, dtype radius_min,
                                           arma::uword n_edges)
{

    std::string error_message = chan.is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (chan.empty())
        throw std::invalid_argument("Input 'chan' cannot be empty.");

    if (chan.center_freq.empty())
        throw std::invalid_argument("Input 'chan' must contain 'center_freq'.");

    if (chan.n_interact.empty() || chan.interact_coord.empty())
        throw std::invalid_argument("Input 'chan' must contain 'n_interact' and 'interact_coord'.");

    if (chan.polmat.empty() && chan.coeff_re.empty())
        throw std::invalid_argument("Input 'chan' must contain 'coeff_re' or 'polmat'.");

    if (n_edges < 3)
        throw std::invalid_argument("Input 'n_edges' must be at least 3.");

    if (!std::isfinite(gain_max_dB) || !std::isfinite(gain_min_dB) || !(gain_max_dB > gain_min_dB))
        throw std::invalid_argument("Inputs 'gain_max_dB' and 'gain_min_dB' must be finite, with 'gain_max_dB' larger than 'gain_min_dB'.");

    std::string fn_suffix = ".obj";
    std::string fn_mtl;

    if (fn.size() >= fn_suffix.size() &&
        fn.compare(fn.size() - fn_suffix.size(), fn_suffix.size(), fn_suffix) == 0)
    {
        fn_mtl = fn.substr(0, fn.size() - fn_suffix.size()) + ".mtl";
    }
    else
        throw std::invalid_argument("Input 'fn' must end with '.obj'.");

    // Extract the file name from the path
    std::string fn_mtl_base;
    arma::uword pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1ULL);
    else
        fn_mtl_base = fn_mtl;

    arma::uword n_snap_in = chan.n_snap();
    arma::uword n_rx = chan.n_rx();
    arma::uword n_tx = chan.n_tx();

    const arma::uvec i_snap = (i_snap_sel.n_elem == 0ULL) ? arma::regspace<arma::uvec>(0ULL, n_snap_in - 1ULL) : i_snap_sel;

    if (arma::any(i_snap >= n_snap_in))
        throw std::invalid_argument("Snapshot indices 'i_snap' cannot exceed the number of snapshots in the channel.");

    arma::uword n_snap_out = i_snap.n_elem;

    if (radius_min < (dtype)0.0 || radius_max < (dtype)0.0)
        throw std::invalid_argument("Inputs 'radius_max' and 'radius_min' cannot be negative.");

    // Make sure that the minimum radius is smaller than the maximum
    radius_min = (radius_min < radius_max) ? radius_min : radius_max;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    arma::uword n_cmap = (arma::uword)cmap.n_rows;

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (outFile.is_open())
    {
        outFile << "# QuaDRiGa " << "path data colormap\n\n";
        for (arma::uword i = 0ULL; i < n_cmap; ++i)
        {
            double R = (double)cmap(i, 0ULL) / 255.0;
            double G = (double)cmap(i, 1ULL) / 255.0;
            double B = (double)cmap(i, 2ULL) / 255.0;
            outFile << "newmtl QuaDRiGa_PATH_" << colormap << "_" << std::setfill('0') << std::setw(2) << i << "\n";
            outFile << std::fixed << std::setprecision(6) << "Kd " << R << " " << G << " " << B << "\n\n";
        }
        outFile.close();
    }
    else
        throw std::runtime_error("Cannot write file '" + fn_mtl + "'.");

    // Export paths to OBJ file
    outFile = std::ofstream(fn);
    if (outFile.is_open())
    {
        outFile << "# QuaDRiGa Path OBJ File\n";
        outFile << "mtllib " << fn_mtl_base << "\n";
    }
    else
        throw std::runtime_error("Cannot write file '" + fn + "'.");

    bool moving_tx = chan.tx_pos.n_cols != 1ULL;
    bool moving_rx = chan.rx_pos.n_cols != 1ULL;

    arma::uword vert_counter = 0ULL;
    for (arma::uword i_snap_out = 0ULL; i_snap_out < n_snap_out; ++i_snap_out)
    {
        arma::uword i_snap_in = i_snap(i_snap_out);

        // Extract path coordinates
        dtype tx = moving_tx ? chan.tx_pos(0ULL, i_snap_in) : chan.tx_pos(0ULL, 0ULL);
        dtype ty = moving_tx ? chan.tx_pos(1ULL, i_snap_in) : chan.tx_pos(1ULL, 0ULL);
        dtype tz = moving_tx ? chan.tx_pos(2ULL, i_snap_in) : chan.tx_pos(2ULL, 0ULL);

        dtype gainF = (chan.center_freq.n_elem > 1ULL) ? chan.center_freq(i_snap_in) : chan.center_freq(0ULL);
        gainF = (dtype)-32.45 - (dtype)20.0 * std::log10(gainF * (dtype)1.0e-9);

        dtype rx = moving_rx ? chan.rx_pos(0ULL, i_snap_in) : chan.rx_pos(0ULL, 0ULL);
        dtype ry = moving_rx ? chan.rx_pos(1ULL, i_snap_in) : chan.rx_pos(1ULL, 0ULL);
        dtype rz = moving_rx ? chan.rx_pos(2ULL, i_snap_in) : chan.rx_pos(2ULL, 0ULL);

        // Calculate path coordinates
        arma::Col<dtype> path_length;
        std::vector<arma::Mat<dtype>> path_coord;
        quadriga_lib::coord2path<dtype>(arma::Col<dtype>{tx, ty, tz}, arma::Col<dtype>{rx, ry, rz}, chan.n_interact[i_snap_in], chan.interact_coord[i_snap_in],
                                        &path_length, nullptr, nullptr, nullptr, &path_coord);

        // Calculate path power
        arma::uword n_path = path_coord.size();
        arma::Col<dtype> path_power_dB(n_path);
        arma::s32_vec color_index(n_path);

        dtype scl = (dtype)63.0 / (gain_max_dB - gain_min_dB); // gain_max_dB > gain_min_dB, checked at the start
        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            // Lambda to calculate the power and check if the pattern has any entry > -200 dB
            auto calc_power = [](const dtype *Re, const dtype *Im, arma::uword n_elem) -> dtype
            {
                dtype p = dtype(0.0);
                for (arma::uword i = 0ULL; i < n_elem; ++i)
                    p += Re[i] * Re[i] + Im[i] * Im[i];
                return (dtype)10.0 * std::log10(p);
                ;
            };

            dtype p = (dtype)0.0;
            if (!chan.coeff_re.empty()) // Use MIMO coefficients
                p = calc_power(chan.coeff_re[i_snap_in].slice_memptr(i_path), chan.coeff_im[i_snap_in].slice_memptr(i_path), n_rx * n_tx);
            else if (!chan.polmat.empty())            // Use polmat
            {
                const dtype *p_xpr = chan.polmat[i_snap_in].colptr(i_path);
                for (arma::uword m = 0ULL; m < 8ULL; ++m) // sum( polmat(:,n).^2 )
                    p += *p_xpr * *p_xpr, ++p_xpr;
                p = (dtype)10.0 * std::log10(p);
                p += gainF - (dtype)20.0 * std::log10(path_length(i_path));
            }
            path_power_dB(i_path) = p;

            dtype x = p;
            x = (x < gain_min_dB) ? gain_min_dB : x;
            x = (x > gain_max_dB) ? gain_max_dB : x;
            x = x - gain_min_dB;
            x = x * scl;
            x = (x > (dtype)63.0) ? (dtype)63.0 : x;
            x = std::round(x);

            color_index(i_path) = (p >= gain_min_dB) ? (int)x : -1; // NaN power is excluded

            // Paths with non-finite coordinates cannot be drawn
            if (!path_coord[i_path].is_finite())
                color_index(i_path) = -1;
        }

        // Sort the path power in decreasing order and return the indices
        arma::uvec pow_indices = arma::sort_index(path_power_dB, "descend");
        arma::uword *i_pow_sorted = pow_indices.memptr();

        // Limit the number of paths that should be shown
        if (max_paths != 0ULL)
            for (arma::uword i = max_paths; i < n_path; ++i)
                color_index(i_pow_sorted[i]) = -1;

        // Write some descriptive information
        outFile << "\n# Snapshot " << i_snap_in << "\n";
        outFile << "#  No.   Length[m]     Gain[dB]   ID" << "\n";

        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            arma::uword i_path_sorted = i_pow_sorted[i_path];
            dtype l = path_length(i_path_sorted);
            dtype p = path_power_dB(i_path_sorted);

            if (p > (dtype)-200.0)
            {
                outFile << "# " << std::setfill('0') << std::setw(4) << i_path << " ";
                if (l < 1000.0f)
                    outFile << " ";
                if (l < 100.0f)
                    outFile << " ";
                if (l < 10.0f)
                    outFile << " ";
                outFile << std::fixed << std::setprecision(6) << l << "  ";
                if (p > -100.0f)
                    outFile << " ";
                if (p > -10.0f)
                    outFile << " ";
                if (p > 10.0f)
                    outFile << " ";
                else if (p > 0.0f)
                    outFile << "  ";

                outFile << std::fixed << std::setprecision(6) << p;

                if (color_index(i_path_sorted) >= 10)
                    outFile << "   " << color_index(i_path_sorted);
                else if (color_index(i_path_sorted) >= 0)
                    outFile << "    " << color_index(i_path_sorted);

                outFile << "\n";
            }
        }

        // Write OBJ elements
        scl = radius_max / (gain_max_dB - gain_min_dB);
        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            arma::uword i_path_sorted = i_pow_sorted[i_path];
            if (color_index(i_path_sorted) >= 0)
            {
                // Calculate radius
                dtype p = path_power_dB(i_path_sorted);
                dtype radius = p - gain_min_dB;
                radius = radius * scl;
                radius = (radius > radius_max) ? radius_max : radius;
                radius = (radius < radius_min) ? radius_min : radius;

                // Write object name to OBJ file
                outFile << "\no QuaDRiGa_path_s" << std::setfill('0') << std::setw(4) << i_snap_in << "_p" << std::setfill('0') << std::setw(4) << i_path << "\n";

                ++vert_counter;
                if (std::abs(radius) > (dtype)1.0e-4)
                {
                    // Calculate vertices and faces
                    arma::Mat<dtype> vert;
                    arma::umat faces;
                    quadriga_lib::path_to_tube(path_coord[i_path_sorted], vert, faces, radius, (arma::uword)n_edges);

                    // Write vertices to file
                    arma::uword n_vert = vert.n_cols;
                    for (arma::uword iV = 0ULL; iV < n_vert; ++iV)
                        outFile << std::defaultfloat << "v " << vert(0ULL, iV) << " " << vert(1ULL, iV) << " " << vert(2ULL, iV) << "\n";

                    outFile << "usemtl QuaDRiGa_PATH_" << colormap << "_" << std::setfill('0') << std::setw(2) << color_index(i_path_sorted) << "\n";

                    // Write faces to file
                    arma::uword n_faces = faces.n_cols;
                    for (arma::uword iF = 0ULL; iF < n_faces; ++iF)
                        outFile << "f " << faces(0ULL, iF) + vert_counter << " " << faces(1ULL, iF) + vert_counter << " " << faces(2ULL, iF) + vert_counter << " " << faces(3ULL, iF) + vert_counter << "\n";

                    vert_counter += vert.n_cols - 1ULL;
                }
                else
                {
                    // Write vertices to file
                    arma::uword n_vert = path_coord[i_path_sorted].n_cols;
                    for (arma::uword iV = 0ULL; iV < n_vert; ++iV)
                        outFile << std::defaultfloat << "v " << path_coord[i_path_sorted](0ULL, iV) << " " << path_coord[i_path_sorted](1ULL, iV) << " " << path_coord[i_path_sorted](2ULL, iV) << "\n";

                    outFile << "usemtl QuaDRiGa_PATH_" << colormap << "_" << std::setfill('0') << std::setw(2) << color_index(i_path_sorted) << "\n";

                    for (arma::uword iV = 0ULL; iV < n_vert - 1ULL; ++iV)
                        outFile << "l " << vert_counter << " " << vert_counter + 1ULL << "\n", ++vert_counter;
                }
            }
        }
    }
    outFile.close();
}

template void quadriga_lib::channel_export_obj_file(const std::string &fn, const channel<float> &chan, arma::uword max_paths,
                                                    float gain_max_dB, float gain_min_dB, const std::string &colormap,
                                                    const arma::uvec &i_snap, float radius_max, float radius_min,
                                                    arma::uword n_edges);
template void quadriga_lib::channel_export_obj_file(const std::string &fn, const channel<double> &chan, arma::uword max_paths,
                                                    double gain_max_dB, double gain_min_dB, const std::string &colormap,
                                                    const arma::uvec &i_snap, double radius_max, double radius_min,
                                                    arma::uword n_edges);
