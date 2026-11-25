// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Fixtures of the layout tests: text files in a work folder, a small hand-built layout, and the comparison of two layouts.
//
// qd_layout_files:   a folder (below the test's current folder) that is removed with the object; 'path' gives a file in
//                    it, 'write' writes a text file and returns its path
// qd_layout_throws:  message of the std::exception that a call throws, "" if it does not throw
// qd_layout_diff:    first difference between two layouts (empty string if equal); numbers within 'tol'
// qd_layout_build:   Tx: site "BS1" with sectors 0 and 1 (one position, orientations 0 and 120 deg), site "BS2" with
//                    one row; Rx: terminal "UT1" with a three-row track, "UT2" a point drop with speed and direction
// qd_layout_diff_array: first difference between two antenna arrays; numbers within 'tol' times the largest magnitude
//                    of the member
// qd_layout_folder:  names of the files in a folder, sorted
// qd_layout_nodes_from: nodes "N0", "N1", ... (sector 0, generic antenna) with one track each, given as position
//                    matrices [3, n]; orientation 0, no other member
// qd_layout_rot:     3x3 rotation matrix of (bank, tilt, heading) in [rad] (library helper calc_rotation_matrix)
// qd_layout_wrap:    angle in [rad] wrapped to [-pi, pi]
// qd_layout_oracle_seed: seed of a layout stream from its base seed and key (64-bit FNV-1a, as documented)
// qd_layout_sites:   the sites of a side (names, first (x, y), node indices)
// qd_layout_images:  the wrap images of a layout in the documented order, [2, n_img]
// qd_layout_nearest: nearest site and wrap image of a point, brute force

#ifndef QD_TEST_LAYOUT_HPP
#define QD_TEST_LAYOUT_HPP

#include "quadriga_layout.hpp"
#include "quadriga_math.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct qd_layout_files
{
    std::string dir;
    explicit qd_layout_files(const std::string &name) : dir("qd_test_layout_" + name)
    {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
    }
    ~qd_layout_files()
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    std::string path(const std::string &name) const { return dir + "/" + name; }
    std::string write(const std::string &name, const std::string &text) const
    {
        std::ofstream f(path(name), std::ios::binary);
        f << text;
        return path(name);
    }
    std::string read(const std::string &name) const
    {
        std::ifstream f(path(name), std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
};

template <class F>
static std::string qd_layout_throws(F f)
{
    try
    {
        f();
    }
    catch (const std::exception &e)
    {
        return e.what();
    }
    return "";
}

static inline bool qd_layout_same_mat(const arma::mat &a, const arma::mat &b, double tol)
{
    if (a.n_elem == 0 && b.n_elem == 0) // Empty is empty, whatever its shape
        return true;
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols)
        return false;
    for (arma::uword i = 0; i < a.n_elem; ++i)
        if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])) || std::abs(a[i] - b[i]) <= tol))
            return false;
    return true;
}

template <class T>
static inline bool qd_layout_same_int(const T &a, const T &b)
{
    return a.n_elem == b.n_elem && (a.n_elem == 0 || arma::all(arma::vectorise(a == b)));
}

// Members of a side that the CSV files hold
// Angles after a round trip through a file: the file holds 15 significant digits of the angle in degrees, so the
// relative error of the value is at most 0.5e-14 (leading digit 1), here twice that as margin, plus 4 ulp for the two
// conversions between radian and degree. The relative bound holds for any angle, also for 359.123456789012345 degrees.
static inline bool qd_layout_same_angle(const arma::mat &a, const arma::mat &b)
{
    if (a.n_elem == 0 && b.n_elem == 0)
        return true;
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols)
        return false;
    for (arma::uword i = 0; i < a.n_elem; ++i)
    {
        const double m = std::max(std::abs(a[i]), std::abs(b[i]));
        if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])) || std::abs(a[i] - b[i]) <= (1.0e-14 + 4.0 * 2.220446049250313e-16) * m))
            return false;
    }
    return true;
}

static inline std::string qd_layout_diff_nodes(const quadriga_lib::layout_nodes &a, const quadriga_lib::layout_nodes &b, double tol = 0.0)
{
    if (a.name != b.name)
        return "name";
    if (!qd_layout_same_int(a.sector, b.sector))
        return "sector";
    if (!qd_layout_same_int(a.i_antenna, b.i_antenna))
        return "i_antenna";
    if (!qd_layout_same_int(a.offset, b.offset))
        return "offset";
    if (!qd_layout_same_mat(a.pos, b.pos, tol))
        return "pos";
    if (!qd_layout_same_angle(a.orientation, b.orientation))
        return "orientation";
    if (!qd_layout_same_angle(a.direction, b.direction))
        return "direction";
    if (!qd_layout_same_mat(a.speed, b.speed, tol))
        return "speed";
    if (!qd_layout_same_mat(a.time, b.time, tol))
        return "time";
    if (a.scenario != b.scenario)
        return "scenario";
    if (!qd_layout_same_int(a.indoor, b.indoor))
        return "indoor";
    if (!qd_layout_same_int(a.floor, b.floor))
        return "floor";
    if (!qd_layout_same_mat(a.d2d_in, b.d2d_in, tol))
        return "d2d_in";
    if (!qd_layout_same_mat(a.clutter, b.clutter, tol))
        return "clutter";
    if (a.scenpar_override.size() != b.scenpar_override.size())
        return "scenpar_override";
    for (const auto &kv : a.scenpar_override)
    {
        const auto it = b.scenpar_override.find(kv.first);
        if (it == b.scenpar_override.end() || !qd_layout_same_mat(kv.second, it->second, tol))
            return "scenpar_override";
    }
    if (!qd_layout_same_int(a.seed, b.seed))
        return "seed";
    if (!qd_layout_same_int(a.relative, b.relative))
        return "relative";
    if (a.extra_names != b.extra_names || a.extra != b.extra)
        return "extra";
    return "";
}

// First difference of the members that the CSV files hold ("" = equal); not compared: name and links
static inline std::string qd_layout_diff(const quadriga_lib::layout &a, const quadriga_lib::layout &b, double tol = 0.0)
{
    std::string s = qd_layout_diff_nodes(a.tx, b.tx, tol);
    if (!s.empty())
        return "tx." + s;
    s = qd_layout_diff_nodes(a.rx, b.rx, tol);
    if (!s.empty())
        return "rx." + s;
    if (a.seed != b.seed)
        return "seed";
    if (!qd_layout_same_mat(a.center_freq, b.center_freq, 0.0))
        return "center_freq";
    if (a.antenna_cols != b.antenna_cols)
        return "antenna_cols";
    if (a.antenna_args != b.antenna_args)
        return "antenna_args";
    if (!qd_layout_same_mat(a.wrap, b.wrap, tol))
        return "wrap";
    if (!qd_layout_same_int(a.pairing, b.pairing))
        return "pairing";
    return "";
}

template <class T>
static inline bool qd_layout_same_data(const T &a, const T &b, double tol)
{
    if (arma::size(a) != arma::size(b))
        return false;
    if (a.n_elem == 0)
        return true;
    return arma::approx_equal(a, b, "absdiff", tol * std::max(arma::abs(a).max(), arma::abs(b).max()));
}

// First difference of two arrays ("" = equal): grids, patterns, element positions, port weights, center frequency
static inline std::string qd_layout_diff_array(const quadriga_lib::arrayant<double> &a, const quadriga_lib::arrayant<double> &b, double tol = 0.0)
{
    if (a.n_elements() == 0 || b.n_elements() == 0)
        return a.n_elements() == b.n_elements() ? "" : "n_elements";
    if (!qd_layout_same_data(a.azimuth_grid, b.azimuth_grid, tol) || !qd_layout_same_data(a.elevation_grid, b.elevation_grid, tol))
        return "grid";
    if (!qd_layout_same_data(a.e_theta_re, b.e_theta_re, tol) || !qd_layout_same_data(a.e_theta_im, b.e_theta_im, tol) ||
        !qd_layout_same_data(a.e_phi_re, b.e_phi_re, tol) || !qd_layout_same_data(a.e_phi_im, b.e_phi_im, tol))
        return "pattern";
    if (!qd_layout_same_data(a.element_pos, b.element_pos, tol))
        return "element_pos";
    if (!qd_layout_same_data(a.port_weights_re, b.port_weights_re, tol) || !qd_layout_same_data(a.port_weights_im, b.port_weights_im, tol))
        return "port_weights";
    if (std::abs(a.center_freq - b.center_freq) > tol * std::abs(a.center_freq))
        return "center_freq";
    return "";
}

static inline std::vector<std::string> qd_layout_folder(const std::string &dir)
{
    std::vector<std::string> f;
    for (const auto &e : std::filesystem::directory_iterator(dir))
        f.push_back(e.path().filename().string());
    std::sort(f.begin(), f.end());
    return f;
}

static inline quadriga_lib::layout_nodes qd_layout_nodes_from(const std::vector<arma::mat> &tracks)
{
    quadriga_lib::layout_nodes N;
    const arma::uword nn = (arma::uword)tracks.size();
    N.sector.zeros(nn);
    N.i_antenna.zeros(nn);
    N.offset.zeros(nn);
    arma::uword n_row = 0;
    for (arma::uword i = 0; i < nn; ++i)
    {
        N.name.push_back("N" + std::to_string(i));
        N.offset[i] = n_row;
        N.pos = arma::join_rows(N.pos, tracks[i]);
        n_row += tracks[i].n_cols;
    }
    N.orientation.zeros(3, n_row);
    return N;
}

// One position as [3, 1] matrix
static inline arma::mat qd_layout_pt(double x, double y, double z)
{
    arma::mat p(3, 1);
    p(0, 0) = x, p(1, 0) = y, p(2, 0) = z;
    return p;
}

// R(bank, tilt, heading) as 3x3 matrix
static inline arma::mat qd_layout_rot(double bank, double tilt, double heading)
{
    arma::mat o = {bank, tilt, heading}; // [3, 1]
    arma::cube O(o.memptr(), 3, 1, 1, true), R;
    quadriga_lib::calc_rotation_matrix(O, R);
    return arma::mat(R.memptr(), 3, 3, true); // column-major
}

static inline double qd_layout_wrap(double a)
{
    return std::remainder(a, 2.0 * 3.14159265358979323846);
}

static inline quadriga_lib::layout qd_layout_build()
{
    const double d = 0.017453292519943295;
    quadriga_lib::layout L;
    quadriga_lib::layout_nodes &T = L.tx;
    T.name = {"BS1", "BS1", "BS2"};
    T.sector = {0, 1, 0};
    T.i_antenna = {0, 0, 0};
    T.offset = {0, 1, 2};
    T.pos = arma::mat{{10.0, 10.0, -300.0}, {10.0, 10.0, 50.0}, {25.0, 25.0, 30.0}}; // Rows x, y, z
    T.orientation = arma::mat{{0.0, 0.0, 0.0}, {0.0, 10.0 * d, 0.0}, {30.0 * d, 150.0 * d, 270.0 * d}};

    quadriga_lib::layout_nodes &R = L.rx;
    R.name = {"UT1", "UT2"};
    R.sector = {0, 0};
    R.i_antenna = {0, 0};
    R.offset = {0, 3};
    R.pos = arma::mat{{100.0, 200.0, 1.5}, {101.0, 200.0, 1.5}, {101.0, 202.0, 1.5}, {-50.0, 70.0, 1.5}}.t();
    R.orientation.zeros(3, 4);
    R.orientation(2, 3) = 45.0 * d;
    R.speed = {0.0, 0.0, 0.0, 1.5};
    R.direction = arma::mat{{arma::datum::nan, 0.0}, {arma::datum::nan, 0.0}, {arma::datum::nan, 0.0}, {120.0 * d, 0.0}}.t();
    return L;
}

// Seed of the layout stream 'key' for a base seed: the documented 64-bit FNV-1a hash of the 8 bytes of the base seed
// (least significant first) and the key, with the sign bit cleared
static inline long long qd_layout_oracle_seed(long long base, const std::string &key)
{
    unsigned long long h = 14695981039346656037ULL;
    for (int i = 0; i < 8; ++i)
        h = (h ^ (((unsigned long long)base >> (8 * i)) & 0xFFULL)) * 1099511628211ULL;
    for (unsigned char c : key)
        h = (h ^ c) * 1099511628211ULL;
    return (long long)(h & 0x7FFFFFFFFFFFFFFFULL);
}

// The sites of one side: names in order of first appearance, (x, y) of the first row of their first node, node indices
struct qd_layout_site_list
{
    std::vector<std::string> name;
    arma::mat xy; // [2, n_site]
    std::vector<std::vector<arma::uword>> nodes;
};

static inline qd_layout_site_list qd_layout_sites(const quadriga_lib::layout_nodes &N)
{
    qd_layout_site_list S;
    for (arma::uword i = 0; i < (arma::uword)N.name.size(); ++i)
    {
        const auto it = std::find(S.name.begin(), S.name.end(), N.name[i]);
        if (it != S.name.end())
        {
            S.nodes[(size_t)(it - S.name.begin())].push_back(i);
            continue;
        }
        S.name.push_back(N.name[i]);
        S.nodes.push_back({i});
        S.xy.resize(2, S.name.size());
        S.xy(0, S.name.size() - 1) = N.pos(0, N.offset[i]);
        S.xy(1, S.name.size() - 1) = N.pos(1, N.offset[i]);
    }
    return S;
}

// The wrap images of a layout: 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2); with one vector 0, +a1, -a1; [2, n_img]
static inline arma::mat qd_layout_images(const arma::mat &wrap)
{
    arma::mat I(2, 1, arma::fill::zeros);
    if (wrap.n_cols >= 1)
        I = arma::join_rows(I, wrap.col(0), -wrap.col(0));
    if (wrap.n_cols == 2)
        I = arma::join_rows(I, arma::join_rows(wrap.col(1), -wrap.col(1), wrap.col(0) - wrap.col(1), wrap.col(1) - wrap.col(0)));
    return I;
}

// Nearest site and image of the point (x, y), brute force over all sites and images; returns the 2D distance
static inline double qd_layout_nearest(const qd_layout_site_list &S, const arma::mat &images, double x, double y, arma::uword &site,
                                       arma::uword &image)
{
    double best = arma::datum::inf;
    for (arma::uword s = 0; s < S.xy.n_cols; ++s)
        for (arma::uword k = 0; k < images.n_cols; ++k)
        {
            const double d = std::hypot(x - S.xy(0, s) - images(0, k), y - S.xy(1, s) - images(1, k));
            if (d < best)
                best = d, site = s, image = k;
        }
    return best;
}

#endif
