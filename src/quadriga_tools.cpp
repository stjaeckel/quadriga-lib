// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_tools.hpp"
#include <stdexcept>

// FUNCTION: Calculate rotation matrix R from roll, pitch, and yaw angles (given by rows in the input "orientation")
template <typename dtype>
arma::cube quadriga_tools::calc_rotation_matrix(const arma::Cube<dtype> orientation, bool invert_y_axis, bool transposeR)
{
    // Input:       orientation         Orientation vectors (rows = bank (roll), tilt (pitch), heading (yaw)) in [rad], Size [3, n_row, n_col]
    //              invert_y_axis       Inverts the y-axis
    //              transposeR          Returns the transpose of R instead of R
    // Output:      R                   Rotation matrix, column-major order, Size [9, n_row, n_col ]

    if (orientation.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (orientation.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    uword n_row = orientation.n_cols, n_col = orientation.n_slices;
    arma::cube rotation = arma::cube(9, n_row, n_col, arma::fill::zeros); // Always double precision
    const dtype *p_orientation = orientation.memptr();
    double *p_rotation = rotation.memptr();

    for (uword iC = 0; iC < n_col; iC++)
        for (uword iR = 0; iR < n_row; iR++)
        {
            double cc = (double)*p_orientation++, sc = sin(cc);
            double cb = (double)*p_orientation++, sb = sin(cb);
            double ca = (double)*p_orientation++, sa = sin(ca);
            ca = cos(ca), cb = cos(cb), cc = cos(cc), sb = invert_y_axis ? -sb : sb;

            if (transposeR)
            {
                *p_rotation++ = ca * cb;
                *p_rotation++ = ca * sb * sc - sa * cc;
                *p_rotation++ = ca * sb * cc + sa * sc;
                *p_rotation++ = sa * cb;
                *p_rotation++ = sa * sb * sc + ca * cc;
                *p_rotation++ = sa * sb * cc - ca * sc;
                *p_rotation++ = -sb;
                *p_rotation++ = cb * sc;
                *p_rotation++ = cb * cc;
            }
            else
            {
                *p_rotation++ = ca * cb;
                *p_rotation++ = sa * cb;
                *p_rotation++ = -sb;
                *p_rotation++ = ca * sb * sc - sa * cc;
                *p_rotation++ = sa * sb * sc + ca * cc;
                *p_rotation++ = cb * sc;
                *p_rotation++ = ca * sb * cc + sa * sc;
                *p_rotation++ = sa * sb * cc - ca * sc;
                *p_rotation++ = cb * cc;
            }
        }

    return rotation;
}
template arma::cube quadriga_tools::calc_rotation_matrix(const arma::Cube<float>, bool invert_y_axis, bool transposeR);
template arma::cube quadriga_tools::calc_rotation_matrix(const arma::Cube<double>, bool invert_y_axis, bool transposeR);

// FUNCTION: Transform from geographic coordinates to Cartesian coordinates
template <typename dtype>
arma::cube quadriga_tools::geo2cart(const arma::Mat<dtype> azimuth, const arma::Mat<dtype> elevation, const arma::Mat<dtype> length)
{
    // Inputs:          azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector,                   Size [n_row, n_col]
    // Output:          cart            Cartesian coordinates,                  Size [3, n_row, n_col]

    if (azimuth.n_elem == 0 || elevation.n_elem == 0 || length.n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (elevation.n_rows != azimuth.n_rows || length.n_rows != azimuth.n_rows ||
        elevation.n_cols != azimuth.n_cols || length.n_cols != azimuth.n_cols)
        throw std::invalid_argument("Inputs must have the same size.");

    uword n_row = azimuth.n_rows, n_col = azimuth.n_cols;
    arma::cube cart = arma::cube(3, n_row, n_col, arma::fill::zeros); // Always double precision

    for (uword i = 0; i < azimuth.n_elem; i++)
    {
        double ca = (double)azimuth(i), sa = sin(ca), r = (double)length(i);
        double ce = (double)elevation(i), se = sin(ce);
        ca = cos(ca), ce = cos(ce);

        uword rw = i % n_row, co = i / n_row;
        cart(0, rw, co) = r * ce * ca;
        cart(1, rw, co) = r * ce * sa;
        cart(2, rw, co) = r * se;
    }
    return cart;
}
template arma::cube quadriga_tools::geo2cart(const arma::Mat<float> azimuth, const arma::Mat<float> elevation, const arma::Mat<float> length);
template arma::cube quadriga_tools::geo2cart(const arma::Mat<double> azimuth, const arma::Mat<double> elevation, const arma::Mat<double> length);

// FUNCTION: Transform from Cartesian coordinates to geographic coordinates
template <typename dtype>
arma::cube quadriga_tools::cart2geo(const arma::Cube<dtype> cart)
{
    // Input:           cart            Cartesian coordinates,                  Size [3, n_row, n_col]
    // Output:          geo             geographic coordinates (az,el,len)      Size [n_row, n_col, 3]

    if (cart.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (cart.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    uword n_row = cart.n_cols, n_col = cart.n_slices;
    arma::cube geo = arma::cube(n_row, n_col, 3, arma::fill::zeros); // Always double precision

    for (uword r = 0; r < n_row; r++)
        for (uword c = 0; c < n_col; c++)
        {
            double x = (double)cart(0, r, c), y = (double)cart(1, r, c), z = (double)cart(2, r, c);
            double len = sqrt(x * x + y * y + z * z), rlen = 1.0 / len;
            x *= rlen, y *= rlen, z *= rlen;
            x = x > 1.0 ? 1.0 : x, y = y > 1.0 ? 1.0 : y, z = z > 1.0 ? 1.0 : z;
            geo(r, c, 0) = atan2(y, x);
            geo(r, c, 1) = asin(z);
            geo(r, c, 2) = len;
        }
    return geo;
}
template arma::cube quadriga_tools::cart2geo(const arma::Cube<float> cart);
template arma::cube quadriga_tools::cart2geo(const arma::Cube<double> cart);

// Convert path interaction coordinates into FBS/LBS positions, path length and angles
template <typename dtype>
void quadriga_tools::coord2path(dtype Tx, dtype Ty, dtype Tz, dtype Rx, dtype Ry, dtype Rz, const arma::Cube<dtype> *coord,
                                arma::Col<dtype> *path_length, arma::Mat<dtype> *fbs_pos, arma::Mat<dtype> *lbs_pos, arma::Mat<dtype> *path_angles)
{
    if (coord == NULL || coord->n_elem == 0 || coord->n_rows != 3)
        throw std::invalid_argument("Input 'coord' must have 3 rows.");

    arma::uword n_interact = coord->n_cols;
    arma::uword n_path = coord->n_slices;

    constexpr dtype zero = dtype(0.0);
    constexpr dtype half = dtype(0.5);
    constexpr dtype los_limit = dtype(1.0e-4);

    // Set the output size
    if (path_length != NULL && path_length->n_elem != n_path)
        path_length->set_size(n_path);
    if (fbs_pos != NULL && (fbs_pos->n_rows != 3 || fbs_pos->n_cols != n_path))
        fbs_pos->set_size(3, n_path);
    if (lbs_pos != NULL && (lbs_pos->n_rows != 3 || lbs_pos->n_cols != n_path))
        lbs_pos->set_size(3, n_path);
    if (path_angles != NULL && (path_angles->n_rows != n_path || path_angles->n_cols != 4))
        path_angles->set_size(n_path, 4);

    // Get pointers
    const dtype *p_coord = coord->memptr();
    dtype *p_length = path_length == NULL ? NULL : path_length->memptr();
    dtype *p_fbs = fbs_pos == NULL ? NULL : fbs_pos->memptr();
    dtype *p_lbs = lbs_pos == NULL ? NULL : lbs_pos->memptr();
    dtype *p_angles = path_angles == NULL ? NULL : path_angles->memptr();

    // Calculate half way point between TX and RX
    dtype TRx = Rx - Tx, TRy = Ry - Ty, TRz = Rz - Tz;
    TRx = Tx + half * TRx, TRy = Ty + half * TRy, TRz = Tz + half * TRz;

    for (arma::uword ip = 0; ip < n_path; ip++)
    {
        dtype fx = TRx, fy = TRy, fz = TRz;     // Initial FBS-Pos
        dtype lx = TRx, ly = TRy, lz = TRz;     // Initial LBS-Pos
        dtype x = Tx, y = Ty, z = Tz, d = zero; // Initial length

        // Get FBS and LBS positions
        for (arma::uword ii = 0; ii < n_interact; ii++)
        {
            arma::uword ix = 3 * ip * n_interact + 3 * ii, iy = ix + 1, iz = ix + 2;
            if (std::isnan(p_coord[ix]) || std::isnan(p_coord[iy]) || std::isnan(p_coord[iz]))
                break;
            lx = p_coord[ix], ly = p_coord[iy], lz = p_coord[iz];
            x -= lx, y -= ly, z -= lx, d += std::sqrt(x * x + y * y + z * z);
            x = lx, y = ly, z = lz;
            fx = ii == 0 ? lx : fx, fy = ii == 0 ? ly : fy, fz = ii == 0 ? lz : fz;
        }
        x -= Rx, y -= Ry, z -= Rz, d += std::sqrt(x * x + y * y + z * z);

        if (p_length != NULL)
            p_length[ip] = d;
        if (p_fbs != NULL)
            p_fbs[3 * ip] = fx, p_fbs[3 * ip + 1] = fy, p_fbs[3 * ip + 2] = fz;
        if (p_lbs != NULL)
            p_lbs[3 * ip] = lx, p_lbs[3 * ip + 1] = ly, p_lbs[3 * ip + 2] = lz;

        if (p_angles != NULL)
        {
            x = fx - Tx, y = fy - Ty, z = fz - Tz;
            d = std::sqrt(x * x + y * y + z * z);
            p_angles[ip] = std::atan2(y, x);                                 // AOD
            p_angles[n_path + ip] = d < los_limit ? zero : std::asin(z / d); // EOD
            x = lx - Rx, y = ly - Ry, z = lz - Rz;
            d = std::sqrt(x * x + y * y + z * z);
            p_angles[2 * n_path + ip] = std::atan2(y, x);                        // AOA
            p_angles[3 * n_path + ip] = d < los_limit ? zero : std::asin(z / d); // EOA
        }
    }
}
template void quadriga_tools::coord2path(float Tx, float Ty, float Tz, float Rx, float Ry, float Rz, const arma::Cube<float> *coord,
                                         arma::Col<float> *path_length, arma::Mat<float> *fbs_pos, arma::Mat<float> *lbs_pos, arma::Mat<float> *path_angles);
template void quadriga_tools::coord2path(double Tx, double Ty, double Tz, double Rx, double Ry, double Rz, const arma::Cube<double> *coord,
                                         arma::Col<double> *path_length, arma::Mat<double> *fbs_pos, arma::Mat<double> *lbs_pos, arma::Mat<double> *path_angles);

// 2D linear interpolation
template <typename dtype>
std::string quadriga_tools::interp(const arma::Cube<dtype> *input, const arma::Col<dtype> *xi, const arma::Col<dtype> *yi,
                                   const arma::Col<dtype> *xo, const arma::Col<dtype> *yo, arma::Cube<dtype> *output)
{
    constexpr dtype one = dtype(1.0), zero = dtype(0.0);
    const arma::uword nx = input->n_cols, ny = input->n_rows, ne = input->n_slices, nxy = nx * ny;
    const arma::uword mx = xo->n_elem, my = yo->n_elem;

    if (input->n_elem == 0 || xi->n_elem != nx || yi->n_elem != ny || ne == 0)
        return "Data dimensions must match the given number of sample points.";

    if (mx == 0 || my == 0)
        return "Output must have at least one sample point.";

    if (output->n_rows != my || output->n_cols != mx || output->n_slices != ne)
        output->set_size(my, mx, ne);

    arma::uword *i_xp_vec = new arma::uword[mx], *i_xn_vec = new arma::uword[mx];
    dtype *xp_vec = new dtype[mx], *xn_vec = new dtype[mx];

    arma::uword *i_yp_vec = new arma::uword[my], *i_yn_vec = new arma::uword[my];
    dtype *yp_vec = new dtype[my], *yn_vec = new dtype[my];

    {
        // Calculate the x-interpolation parameters
        bool sorted = xi->is_sorted();
        arma::uvec ind = sorted ? arma::regspace<arma::uvec>(0, nx - 1) : arma::sort_index(*xi);
        arma::uword *p_ind = ind.memptr();
        const dtype *pi = xi->memptr();

        dtype *p_grid_srt = new dtype[nx];
        if (!sorted)
            for (arma::uword i = 0; i < nx; i++)
                p_grid_srt[i] = pi[p_ind[i]];
        const dtype *p_grid = sorted ? pi : p_grid_srt;

        dtype *p_diff = new dtype[nx];
        *p_diff = one;
        for (arma::uword i = 1; i < nx; i++)
            p_diff[i] = one / (p_grid[i] - p_grid[i - 1]);

        const dtype *po = xo->memptr();
        for (arma::uword i = 0; i < mx; i++)
        {
            arma::uword ip = 0, in = 0;             // Indices for reading the input data
            dtype val = po[i], wp = one, wn = zero; // Relative weights for interpolation
            while (ip < nx && p_grid[ip] <= val)
                ip++;
            if (ip == nx)
                in = --ip;
            else if (ip != 0)
            {
                in = ip--;
                wp = (p_grid[in] - val) * p_diff[in];
                wp = wp > one ? one : wp;
                wp = wp < zero ? zero : wp;
                wn = one - wp;
            }
            i_xp_vec[i] = p_ind[ip], i_xn_vec[i] = p_ind[in], xp_vec[i] = wp, xn_vec[i] = wn;
        }
        delete[] p_diff;
        delete[] p_grid_srt;
        ind.reset();
    }
    {
        // Calculate the y-interpolation parameters
        bool sorted = yi->is_sorted();
        arma::uvec ind = sorted ? arma::regspace<arma::uvec>(0, ny - 1) : arma::sort_index(*yi);
        arma::uword *p_ind = ind.memptr();
        const dtype *pi = yi->memptr();

        dtype *p_grid_srt = new dtype[ny];
        if (!sorted)
            for (arma::uword i = 0; i < ny; i++)
                p_grid_srt[i] = pi[p_ind[i]];
        const dtype *p_grid = sorted ? pi : p_grid_srt;

        dtype *p_diff = new dtype[ny];
        *p_diff = one;
        for (arma::uword i = 1; i < ny; i++)
            p_diff[i] = one / (p_grid[i] - p_grid[i - 1]);

        const dtype *po = yo->memptr();
        for (arma::uword i = 0; i < my; i++)
        {
            arma::uword ip = 0, in = 0;             // Indices for reading the input data
            dtype val = po[i], wp = one, wn = zero; // Relative weights for interpolation
            while (ip < ny && p_grid[ip] <= val)
                ip++;
            if (ip == ny)
                in = --ip;
            else if (ip != 0)
            {
                in = ip--;
                wp = (p_grid[in] - val) * p_diff[in];
                wp = wp > one ? one : wp;
                wp = wp < zero ? zero : wp;
                wn = one - wp;
            }
            i_yp_vec[i] = p_ind[ip], i_yn_vec[i] = p_ind[in], yp_vec[i] = wp, yn_vec[i] = wn;
        }
        delete[] p_diff;
        delete[] p_grid_srt;
        ind.reset();
    }

    // Interpolate the input data and write to output memory
    const dtype *p_input = input->memptr();
    for (int ie = 0; ie < int(ne); ie++)
    {
        dtype *p_output = output->slice_memptr(ie);
        arma::uword offset = ie * nxy;
        for (arma::uword ix = 0; ix < mx; ix++)
        {
            for (arma::uword iy = 0; iy < my; iy++)
            {
                arma::uword iA = offset + i_xp_vec[ix] * ny + i_yp_vec[iy];
                arma::uword iB = offset + i_xn_vec[ix] * ny + i_yp_vec[iy];
                arma::uword iC = offset + i_xp_vec[ix] * ny + i_yn_vec[iy];
                arma::uword iD = offset + i_xn_vec[ix] * ny + i_yn_vec[iy];

                dtype wA = xp_vec[ix] * yp_vec[iy];
                dtype wB = xn_vec[ix] * yp_vec[iy];
                dtype wC = xp_vec[ix] * yn_vec[iy];
                dtype wD = xn_vec[ix] * yn_vec[iy];

                *p_output++ = wA * p_input[iA] + wB * p_input[iB] + wC * p_input[iC] + wD * p_input[iD];
            }
        }
    }

    delete[] i_xp_vec;
    delete[] i_xn_vec;
    delete[] i_yp_vec;
    delete[] i_yn_vec;
    delete[] xp_vec;
    delete[] xn_vec;
    delete[] yp_vec;
    delete[] yn_vec;

    return "";
}
template std::string quadriga_tools::interp(const arma::Cube<float> *input, const arma::Col<float> *xi, const arma::Col<float> *yi,
                                            const arma::Col<float> *xo, const arma::Col<float> *yo, arma::Cube<float> *output);
template std::string quadriga_tools::interp(const arma::Cube<double> *input, const arma::Col<double> *xi, const arma::Col<double> *yi,
                                            const arma::Col<double> *xo, const arma::Col<double> *yo, arma::Cube<double> *output);

// 1D linear interpolation
template <typename dtype>
std::string quadriga_tools::interp(const arma::Mat<dtype> *input, const arma::Col<dtype> *xi,
                                   const arma::Col<dtype> *xo, arma::Mat<dtype> *output)
{
    const arma::uword nx = input->n_rows, ne = input->n_cols;
    const arma::uword mx = xo->n_elem;

    if (input->n_elem == 0 || xi->n_elem != nx)
        return "Data dimensions must match the given number of sample points.";

    if (mx == 0)
        return "Data dimensions must match the given number of sample points.";

    if (output->n_rows != mx || output->n_cols != ne)
        output->set_size(mx, ne);

    // Reinterpret input matrices as cubes
    const arma::Cube<dtype> input_cube = arma::Cube<dtype>(const_cast<dtype *>(input->memptr()), 1, nx, ne, false, true);
    arma::Cube<dtype> output_cube = arma::Cube<dtype>(output->memptr(), 1, mx, ne, false, true);
    arma::Col<dtype> y(1);

    // Call 2D linear interpolation function
    std::string error_message = quadriga_tools::interp(&input_cube, xi, &y, xo, &y, &output_cube);
    return error_message;
}
template std::string quadriga_tools::interp(const arma::Mat<float> *input, const arma::Col<float> *xi, const arma::Col<float> *xo, arma::Mat<float> *output);
template std::string quadriga_tools::interp(const arma::Mat<double> *input, const arma::Col<double> *xi, const arma::Col<double> *xo, arma::Mat<double> *output);