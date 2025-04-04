// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "quadriga_channel.hpp"
#include "quadriga_tools.hpp"
#include <iomanip> // std::setprecision

template <typename dtype>
void quadriga_lib::export_obj_file(const quadriga_lib::channel<dtype> *ch,
                                   std::string fn,
                                   arma::uword max_no_paths,
                                   dtype gain_max, dtype gain_min,
                                   std::string colormap,
                                   arma::uvec i_snap,
                                   dtype radius_max, dtype radius_min,
                                   arma::uword n_edges)
{

    // Check validity
    std::string error_message = ch->is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (ch->empty())
        throw std::invalid_argument("Channel contains no data.");

    if (ch->center_frequency.empty())
        throw std::invalid_argument("Center frequency is missing.");

    if (ch->no_interact.empty() || ch->interact_coord.empty())
        throw std::invalid_argument("Ray tracing data (no_interact, interact_coord) is missing.");

    if (ch->path_polarization.empty() && ch->coeff_re.empty())
        throw std::invalid_argument("MIMO coefficients or path-metadata is missing.");

    // Input validation
    std::string fn_suffix = ".obj";
    std::string fn_mtl;

    if (fn.size() >= fn_suffix.size() &&
        fn.compare(fn.size() - fn_suffix.size(), fn_suffix.size(), fn_suffix) == 0)
    {
        fn_mtl = fn.substr(0, fn.size() - fn_suffix.size()) + ".mtl";
    }
    else
        throw std::invalid_argument("OBJ-File name must end with .obj");

    // Extract the file name from the path
    std::string fn_mtl_base;
    arma::uword pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1ULL);
    else
        fn_mtl_base = fn_mtl;

    if (max_no_paths == 0ULL)
        max_no_paths = 10000ULL;

    arma::uword n_snap_in = ch->n_snap();
    arma::uword n_rx = ch->n_rx();
    arma::uword n_tx = ch->n_tx();

    if (i_snap.n_elem == 0ULL)
        i_snap = arma::regspace<arma::uvec>(0ULL, n_snap_in - 1ULL);

    if (arma::any(i_snap >= n_snap_in))
        throw std::invalid_argument("Snapshot indices 'i_snap' cannot exceed the number of snapshots in the channel.");

    arma::uword n_snap_out = i_snap.n_elem;

    if (radius_min < (dtype)0.0 || radius_max < (dtype)0.0)
        throw std::invalid_argument("Radius cannot be negative.");

    // Make sure that the minimum radius is smaller than the maximum
    radius_min = (radius_min < radius_max) ? radius_min : radius_max;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    arma::uword n_cmap = (arma::uword)cmap.n_rows;

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (outFile.is_open())
    {
        // Write some text to the file
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
        throw std::invalid_argument("Could not write material file.");

    // Export paths to OBJ file
    outFile = std::ofstream(fn);
    if (outFile.is_open())
    {
        // Write some text to the file
        outFile << "# QuaDRiGa Path OBJ File\n";
        outFile << "mtllib " << fn_mtl_base << "\n";
    }
    else
        throw std::invalid_argument("Could not write OBJ file.");

    bool moving_tx = ch->tx_pos.n_cols != 1ULL;
    bool moving_rx = ch->rx_pos.n_cols != 1ULL;

    // Export each snapshot
    arma::uword vert_counter = 0ULL;
    for (arma::uword i_snap_out = 0ULL; i_snap_out < n_snap_out; ++i_snap_out)
    {
        arma::uword i_snap_in = i_snap(i_snap_out);

        // Extract path coordinates
        dtype tx = moving_tx ? ch->tx_pos(0ULL, i_snap_in) : ch->tx_pos(0ULL, 0ULL);
        dtype ty = moving_tx ? ch->tx_pos(1ULL, i_snap_in) : ch->tx_pos(1ULL, 0ULL);
        dtype tz = moving_tx ? ch->tx_pos(2ULL, i_snap_in) : ch->tx_pos(2ULL, 0ULL);

        dtype gainF = (ch->center_frequency.n_elem > 1ULL) ? ch->center_frequency(i_snap_in) : ch->center_frequency(0ULL);
        gainF = (dtype)-32.45 - (dtype)20.0 * std::log10(gainF * (dtype)1.0e-9);

        dtype rx = moving_rx ? ch->rx_pos(0ULL, i_snap_in) : ch->rx_pos(0ULL, 0ULL);
        dtype ry = moving_rx ? ch->rx_pos(1ULL, i_snap_in) : ch->rx_pos(1ULL, 0ULL);
        dtype rz = moving_rx ? ch->rx_pos(2ULL, i_snap_in) : ch->rx_pos(2ULL, 0ULL);

        // Calculate path coordinates
        arma::Col<dtype> path_length;
        std::vector<arma::Mat<dtype>> path_coord;
        quadriga_lib::coord2path<dtype>(tx, ty, tz, rx, ry, rz, &ch->no_interact[i_snap_in], &ch->interact_coord[i_snap_in],
                                        &path_length, nullptr, nullptr, nullptr, &path_coord);

        // Calculate path power
        arma::uword n_path = path_coord.size();
        arma::Col<dtype> path_power_dB(n_path);
        arma::s32_vec color_index(n_path);
        
        dtype scl = (dtype)63.0 / (gain_max - gain_min);
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
            if (!ch->coeff_re.empty()) // Use MIMO coefficients
                p = calc_power(ch->coeff_re[i_snap_in].slice_memptr(i_path), ch->coeff_im[i_snap_in].slice_memptr(i_path), n_rx * n_tx);
            else if (!ch->path_polarization.empty()) // Use XPR
            {
                const dtype *p_xpr = ch->path_polarization[i_snap_in].colptr(i_path);
                for (arma::uword m = 0ULL; m < 8ULL; ++m) // sum( xprmat(:,n).^2 )
                    p += *p_xpr * *p_xpr, ++p_xpr;
                p = (dtype)10.0 * std::log10(p * (dtype)0.5); // dB, factor 0.5 accounts for XPOL
                p += gainF - (dtype)20.0 * std::log10(path_length(i_path));
            }
            path_power_dB(i_path) = p;

            dtype x = p;
            x = (x < gain_min) ? gain_min : x;
            x = (x > gain_max) ? gain_max : x;
            x = x - gain_min;
            x = x * scl;
            x = (x > (dtype)63.0) ? (dtype)63.0 : x;
            x = std::round(x);

            color_index(i_path) = (p < gain_min) ? -1 : (int)x;
        }

        // Sort the path power in decreasing order and return the indices
        arma::uvec pow_indices = arma::sort_index(path_power_dB, "descend");
        arma::uword *i_pow_sorted = pow_indices.memptr();

        // Limit the number of paths that should be shown
        for (arma::uword i = max_no_paths; i < n_path; ++i)
            color_index(i_pow_sorted[i]) = -1;

        // Write some descriptive information
        outFile << "\n# Snapshot " << i_snap_in << "\n";
        outFile << "#  No.   Lenght[m]     Gain[dB]   ID" << "\n";

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
        scl = radius_max / (gain_max - gain_min);
        for (arma::uword i_path = 0ULL; i_path < n_path; ++i_path)
        {
            arma::uword i_path_sorted = i_pow_sorted[i_path];
            if (color_index(i_path_sorted) >= 0)
            {
                // Calculate radius
                dtype p = path_power_dB(i_path_sorted);
                dtype radius = p - gain_min;
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
                    quadriga_lib::path_to_tube(&path_coord[i_path_sorted], &vert, &faces, radius, (arma::uword)n_edges);

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

template void quadriga_lib::export_obj_file(const quadriga_lib::channel<float> *ch,
                                            std::string fn,
                                            arma::uword max_no_paths,
                                            float gain_max, float gain_min,
                                            std::string colormap,
                                            arma::uvec i_snap,
                                            float radius_max, float radius_min,
                                            arma::uword n_edges);

template void quadriga_lib::export_obj_file(const quadriga_lib::channel<double> *ch,
                                            std::string fn,
                                            arma::uword max_no_paths,
                                            double gain_max, double gain_min,
                                            std::string colormap,
                                            arma::uvec i_snap,
                                            double radius_max, double radius_min,
                                            arma::uword n_edges);