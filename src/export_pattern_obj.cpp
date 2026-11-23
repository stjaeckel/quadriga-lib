// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# export_pattern_obj
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Writes the power pattern of each element as a colored 3D surface, one object per element, placed at its position in
  `element_pos`
- The distance of the surface from the element is linear in the level in dB: `object_radius` at the peak over all
  exported elements, 0 at `directivity_range_dB` below it. The color shows the same level

## Declaration:
```
void quadriga_lib::export_pattern_obj(
    const std::string &fn,
    const arrayant<dtype> &ant,
    double directivity_range_dB = 30.0,
    const std::string &colormap = "jet",
    double object_radius = 1.0,
    arma::uword icosphere_n_div = 4,
    const arma::uvec &i_element = arma::uvec());
```

## Inputs:
- **`fn`** — Output OBJ file name, must end in `.obj`; the colormap is written to a `.mtl` file of the same name;
  existing files are overwritten
- **`ant`** — Valid arrayant; the exported elements must have finite field values and must not be zero in all directions
- **`directivity_range_dB`** *(optional)* — Range of the plot below the peak in dB, finite and not 0; the sign is
  ignored
- **`colormap`** *(optional)* — Colormap name; see [[colormap]] for the supported maps
- **`object_radius`** *(optional)* — Radius of the surface at the peak in m, positive and finite
- **`icosphere_n_div`** *(optional)* — Subdivisions of the [[icosphere]] that carries the surface; higher = finer mesh
  with $20 n^2$ faces
- **`i_element`** *(optional)* — Element indices to export, 0-based; empty: all elements

## See also:
- [[calc_directivity]], [[calc_beamwidth]] (the same pattern as numbers)
- [[colormap]] (the available color scales)
- [[channel_export_obj_file]] (exports propagation paths to an OBJ file)
MD!*/

template <typename dtype>
void quadriga_lib::export_pattern_obj(const std::string &fn, const arrayant<dtype> &ant, double directivity_range_dB_in,
                                            const std::string &colormap, double object_radius_in,
                                            arma::uword icosphere_n_div, const arma::uvec &i_element_in)
{
    const dtype directivity_range_dB = dtype(directivity_range_dB_in);
    const dtype object_radius = dtype(object_radius_in);
    std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    std::string fn_suffix = ".obj";
    std::string fn_mtl;

    if (fn.size() >= fn_suffix.size() &&
        fn.compare(fn.size() - fn_suffix.size(), fn_suffix.size(), fn_suffix) == 0)
    {
        fn_mtl = fn.substr(0, fn.size() - fn_suffix.size()) + ".mtl";
    }
    else
        throw std::invalid_argument("Input 'fn' must end with '.obj'.");

    std::string ant_name = ant.name;
    std::replace(ant_name.begin(), ant_name.end(), ' ', '_');

    // Extract the file name from the path
    std::string fn_mtl_base;
    size_t pos = fn_mtl.find_last_of("/");
    if (pos != std::string::npos)
        fn_mtl_base = fn_mtl.substr(pos + 1);
    else
        fn_mtl_base = fn_mtl;

    double range = -std::abs((double)directivity_range_dB);
    if (range == 0.0 || !std::isfinite(range))
        throw std::invalid_argument("Input 'directivity_range_dB' must be finite and not 0.");

    double radius = (double)object_radius;
    if (!(radius > 0.0) || !std::isfinite(radius))
        throw std::invalid_argument("Input 'object_radius' must be finite and larger than 0.");

    arma::uword no_elements = ant.n_elements();

    const arma::uvec i_element = (i_element_in.n_elem == 0) ? arma::regspace<arma::uvec>(0, no_elements - 1) : i_element_in;

    if (arma::any(i_element >= no_elements))
        throw std::invalid_argument("Input 'i_element' contains an index that is out of range.");

    size_t no_elements_t = (size_t)i_element.n_elem;

    // Colormap
    arma::uchar_mat cmap = quadriga_lib::colormap(colormap);
    size_t n_cmap = (size_t)cmap.n_rows;

    // Generate vertices and faces
    arma::mat vertices_icosphere;
    arma::u32_mat faces_icosphere;
    size_t n_faces, n_vert;
    {
        arma::mat center, vert;
        n_faces = quadriga_lib::icosphere<double>(icosphere_n_div, 1.0, center, nullptr, &vert);

        vertices_icosphere.set_size(3 * n_faces, 3);
        faces_icosphere.set_size(n_faces, 3);

        double *pc = center.memptr(), *pv = vert.memptr(), *v = vertices_icosphere.memptr();
        unsigned *f_ico = faces_icosphere.memptr();

        for (size_t i_face = 0; i_face < n_faces; ++i_face)
        {
            double x, y, z;

            x = pc[i_face] + pv[i_face];
            y = pc[i_face + n_faces] + pv[i_face + n_faces];
            z = pc[i_face + 2 * n_faces] + pv[i_face + 2 * n_faces];

            v[3 * i_face] = x;
            v[3 * i_face + 3 * n_faces] = y;
            v[3 * i_face + 6 * n_faces] = z;
            f_ico[i_face] = 3 * (unsigned)i_face + 1;

            x = pc[i_face] + pv[i_face + 3 * n_faces];
            y = pc[i_face + n_faces] + pv[i_face + 4 * n_faces];
            z = pc[i_face + 2 * n_faces] + pv[i_face + 5 * n_faces];

            v[3 * i_face + 1] = x;
            v[3 * i_face + 3 * n_faces + 1] = y;
            v[3 * i_face + 6 * n_faces + 1] = z;
            f_ico[i_face + n_faces] = 3 * (unsigned)i_face + 2;

            x = pc[i_face] + pv[i_face + 6 * n_faces];
            y = pc[i_face + n_faces] + pv[i_face + 7 * n_faces];
            z = pc[i_face + 2 * n_faces] + pv[i_face + 8 * n_faces];

            v[3 * i_face + 2] = x;
            v[3 * i_face + 3 * n_faces + 2] = y;
            v[3 * i_face + 6 * n_faces + 2] = z;
            f_ico[i_face + 2 * n_faces] = 3 * (unsigned)i_face + 3;
        }
    }
    n_vert = 3 * n_faces;

    // Convert vertices to angles
    arma::Mat<dtype> azimuth, elevation;
    azimuth.set_size(1, n_vert);
    elevation.set_size(1, n_vert);
    {
        double *v = vertices_icosphere.memptr();
        dtype *az = azimuth.memptr(), *el = elevation.memptr();
        for (size_t i_vert = 0; i_vert < n_vert; ++i_vert)
        {
            double x = v[i_vert];
            double y = v[i_vert + n_vert];
            double z = v[i_vert + 2 * n_vert];
            x = x > 1.0 ? 1.0 : x, y = y > 1.0 ? 1.0 : y, z = z > 1.0 ? 1.0 : z;
            x = x < -1.0 ? -1.0 : x, y = y < -1.0 ? -1.0 : y, z = z < -1.0 ? -1.0 : z;
            az[i_vert] = (dtype)std::atan2(y, x);
            el[i_vert] = (dtype)std::asin(z);
        }
    }

    // Interpolate antenna patterns (for all elements in the array)
    arma::Mat<dtype> V_re, V_im, H_re, H_im;
    quadriga_lib::interpolate_pattern<dtype>(ant, azimuth, elevation, V_re, V_im, H_re, H_im, i_element,
                                              {}, {}, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
    dtype *vr = V_re.memptr(), *vi = V_im.memptr(), *hr = H_re.memptr(), *hi = H_im.memptr();

    // Maximum power of the pattern
    double max_pow = 0.0;
    bool has_nan = false;
    auto find_max_pow = [&]()
    {
        max_pow = 0.0, has_nan = false;
        for (size_t i = 0; i < n_vert * no_elements_t; ++i)
        {
            double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
            double p = a * a + b * b + c * c + d * d;
            max_pow = (p > max_pow) ? p : max_pow;
            has_nan = has_nan || std::isnan(p);
        }
    };
    find_max_pow();
    if (std::isinf(max_pow)) // Power above the double range: rescale the interpolated fields
    {
        dtype s = (dtype)qd_inverse_max_field(vr, vi, hr, hi, n_vert * no_elements_t);
        V_re *= s, V_im *= s, H_re *= s, H_im *= s;
        find_max_pow();
    }
    if (has_nan) // NaN field values, or infinite ones (NaN after the rescaling)
        throw std::invalid_argument("Input 'ant' must have finite field values.");
    if (!(max_pow > 0.0))
        throw std::invalid_argument("Pattern is zero in all directions for the exported elements.");

    // Export colormap to material file
    std::ofstream outFile(fn_mtl);
    if (!outFile.is_open())
        throw std::runtime_error("Cannot create file '" + fn_mtl + "'.");
    {
        outFile << "# QuaDRiGa " << "antenna pattern colormap\n\n";
        for (size_t i = 0; i < n_cmap; ++i)
        {
            double R = (double)cmap(i, 0) / 255.0;
            double G = (double)cmap(i, 1) / 255.0;
            double B = (double)cmap(i, 2) / 255.0;
            outFile << "newmtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i << "\n";
            outFile << std::fixed << std::setprecision(6) << "Kd " << R << " " << G << " " << B << "\n\n";
        }
        outFile.close();
    }

    // Write OBJ file
    outFile = std::ofstream(fn);
    if (!outFile.is_open())
        throw std::runtime_error("Cannot create file '" + fn + "'.");
    {
        outFile << "# QuaDRiGa " << "antenna pattern\n";
        outFile << "mtllib " << fn_mtl_base << "\n";

        // Face offset for previous element in the OBJ file
        unsigned face_offset = 1; // OBJ starts counting at 1

        // Export each element to the OBJ file
        for (size_t i_el = 0; i_el < no_elements_t; ++i_el)
        {
            arma::uword i_array_element = i_element(i_el);
            outFile << "\no " << "QuaDRiGa_ANT_" << ant_name << "_" << std::setfill('0') << std::setw(3) << i_array_element << "\n";

            // Shape
            arma::vec pow(n_vert);
            double *pp = pow.memptr();
            for (size_t i_vert = 0; i_vert < n_vert; ++i_vert)
            {
                size_t i = i_vert * no_elements_t + i_el;
                double a = (double)vr[i], b = (double)vi[i], c = (double)hr[i], d = (double)hi[i];
                double p = a * a + b * b + c * c + d * d;
                p /= max_pow;                // Scale by maximum power (division: no overflow for tiny powers)
                p = 10.0 * std::log10(p);    // Convert to dB
                p = (p < range) ? range : p; // Set minimum
                p = (p - range) / (-range);  // Set to range 0 ... 1 (division: no overflow for tiny ranges)
                pp[i_vert] = p;
            }

            // Scale vertices
            arma::mat scaled_vertices(n_vert, 3);
            double *v_scale = scaled_vertices.memptr();
            double *v_ico = vertices_icosphere.memptr();
            for (size_t i_vert = 0; i_vert < n_vert; ++i_vert)
            {
                v_scale[i_vert] = v_ico[i_vert] * pp[i_vert];                             // x
                v_scale[i_vert + n_vert] = v_ico[i_vert + n_vert] * pp[i_vert];           // y
                v_scale[i_vert + 2 * n_vert] = v_ico[i_vert + 2 * n_vert] * pp[i_vert];   // z
            }

            // Process faces
            double eps = 0.001; // Co-location tolerance in [m], relative to 1 m radius
            eps *= eps;         // Squared value

            arma::mat final_vertices(n_vert, 3);
            arma::u32_mat vert_index(n_faces, 3);
            arma::u32_vec cmap_index(n_faces);

            double *v_final = final_vertices.memptr();
            unsigned *f_final = vert_index.memptr();
            unsigned *i_cmap = cmap_index.memptr();

            size_t no_final_vert = 0;
            size_t no_final_face = 0;

            for (size_t i_face = 0; i_face < n_faces; ++i_face)
            {
                // Load the 3 scaled vertices of the face
                double x0 = v_scale[3 * i_face];
                double y0 = v_scale[3 * i_face + n_vert];
                double z0 = v_scale[3 * i_face + 2 * n_vert];

                double x1 = v_scale[3 * i_face + 1];
                double y1 = v_scale[3 * i_face + n_vert + 1];
                double z1 = v_scale[3 * i_face + 2 * n_vert + 1];

                double x2 = v_scale[3 * i_face + 2];
                double y2 = v_scale[3 * i_face + n_vert + 2];
                double z2 = v_scale[3 * i_face + 2 * n_vert + 2];

                // Check if face vertices collapse to a point or line
                double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                double dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x0, dy = y2 - y0, dz = z2 - z0;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
                dd = dx * dx + dy * dy + dz * dz;

                if (dd < eps)
                    continue;

                for (size_t i3 = 0; i3 < 3; ++i3)
                {
                    double x = x0, y = y0, z = z0;
                    if (i3 == 1)
                        x = x1, y = y1, z = z1;
                    else if (i3 == 2)
                        x = x2, y = y2, z = z2;

                    // Add the vertices to the final list
                    size_t i_vert_final = 0; // Index in the final vertex list
                    bool found = false;

                    // Get vertex ID
                    for (size_t i = 0; i < no_final_vert; ++i) // Search existing vertex
                    {
                        dx = v_final[i] - x;
                        dy = v_final[i + n_vert] - y;
                        dz = v_final[i + 2 * n_vert] - z;
                        dd = dx * dx + dy * dy + dz * dz;
                        if (dd < eps)
                        {
                            i_vert_final = i;
                            found = true;
                            break;
                        }
                    }
                    if (!found) // Add new vertex to list
                    {
                        i_vert_final = no_final_vert;
                        v_final[i_vert_final] = x;
                        v_final[i_vert_final + n_vert] = y;
                        v_final[i_vert_final + 2 * n_vert] = z;
                        ++no_final_vert;
                    }

                    // Write face ID
                    f_final[no_final_face + i3 * n_faces] = (unsigned)i_vert_final;
                }

                // Get the colormap index
                double pf = pp[3 * i_face] + pp[3 * i_face + 1] + pp[3 * i_face + 2];
                pf *= 21.0; // 63 * 0.33
                pf = std::round(pf);
                unsigned c = (unsigned)pf;
                c = (c > 63) ? 63 : c;
                i_cmap[no_final_face] = c;

                ++no_final_face;
            }

            // Write vertices to file
            for (size_t i_vert = 0; i_vert < no_final_vert; ++i_vert)
            {
                double x = v_final[i_vert];
                double y = v_final[i_vert + n_vert];
                double z = v_final[i_vert + 2 * n_vert];

                x *= radius;
                y *= radius;
                z *= radius;

                if (ant.element_pos.n_elem != 0)
                {
                    x += ant.element_pos(0, i_array_element);
                    y += ant.element_pos(1, i_array_element);
                    z += ant.element_pos(2, i_array_element);
                }

                outFile << "v " << x << " " << y << " " << z << "\n";
            }

            // Write faces to file, ordered by color
            for (unsigned i_color = 0; i_color < 64; ++i_color)
            {
                bool color_not_used = true;
                for (size_t i_face = 0; i_face < no_final_face; ++i_face)
                {
                    if (i_cmap[i_face] == i_color)
                    {
                        if (color_not_used)
                        {
                            outFile << "usemtl QuaDRiGa_ANT_" << colormap << "_" << std::setfill('0') << std::setw(2) << i_color << "\n";
                            color_not_used = false;
                        }
                        outFile << "f " << f_final[i_face] + face_offset << " " << f_final[i_face + n_faces] + face_offset << " " << f_final[i_face + 2 * n_faces] + face_offset << "\n";
                    }
                }
            }
            face_offset += (unsigned)no_final_vert;
        }

        outFile.close();
    }
}

template void quadriga_lib::export_pattern_obj(const std::string &, const arrayant<float> &, double, const std::string &, double, arma::uword, const arma::uvec &);
template void quadriga_lib::export_pattern_obj(const std::string &, const arrayant<double> &, double, const std::string &, double, arma::uword, const arma::uvec &);