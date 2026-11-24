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

#ifndef QD_TEST_LAYOUT_HPP
#define QD_TEST_LAYOUT_HPP

#include "quadriga_layout.hpp"

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

#endif
