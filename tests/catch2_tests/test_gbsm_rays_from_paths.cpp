// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of gbsm_rays_from_paths: delays, angles and powers from spherical path data, LOS detection and default
// taps, GR paths, in-place writes, the round trip through a QRT file, `no_int` of a QRT file as path type,
// validation. The writing side (qrt_paths_from_scatterers) is tested in test_qrt_paths_from_scatterers.cpp.
// Oracles: the hand-made ray data of make_rays (qd_test_rays.hpp) with scatterers, whose delays and angles are built
// from the scatterer positions (delay = (polyline - d3D) / c, angles towards the scatterers, GR angles towards the
// image points); closed-form angles for the special geometries. Tolerances: 1e-12 for double geometry; after the
// float32 QRT file, path length 1e-7 times the sum of the absolute coordinates of the points (delay that over c),
// angles 2e-7·max|coordinate| / b (b = distance from the end to its next point), sqrt(power)·M 1e-6 relative to
// the largest entry of the path.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_channel.hpp"
#include "quadriga_gbsm.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <new>
#include <string>
#include <vector>

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    // Inputs of one call; the optional inputs are passed when their flag is set
    struct in_t
    {
        arma::vec cf, T, R, pl, er, ei;
        arma::mat fbs, lbs, pg;
        arma::cube M;
        std::string method = "quadriga";
        arma::u32_vec type, tap;
        bool has_type = true, has_tap = true, has_eps = true;
    };

    // Polyline length of path p: LOS the direct line, GR via its point in lbs_pos, NLOS T -> FBS -> LBS -> R
    double path_len(const quadriga_lib::gbsm_rays &r, arma::uword p)
    {
        const arma::vec T = r.tx_pos, R = r.rx_pos, F = r.fbs_pos.col(p), L = r.lbs_pos.col(p);
        if (r.path_type(p) == 0)
            return len3(R - T);
        if (r.path_type(p) == 1)
            return len3(L - T) + len3(R - L);
        return polyline(T, F, L, R);
    }

    // Path data of a make_rays object: path gain = power, path length = polyline
    in_t in_of(const quadriga_lib::gbsm_rays &r)
    {
        in_t in;
        in.cf = r.center_frequency, in.T = r.tx_pos, in.R = r.rx_pos, in.fbs = r.fbs_pos, in.lbs = r.lbs_pos;
        in.pg = r.power, in.M = r.M, in.method = r.method, in.type = r.path_type, in.tap = r.tap;
        in.er = r.gr_eps_re, in.ei = r.gr_eps_im, in.has_eps = !r.gr_eps_re.empty();
        in.pl.set_size(r.n_ray());
        for (arma::uword p = 0; p < r.n_ray(); ++p)
            in.pl(p) = path_len(r, p);
        return in;
    }

    // The paths of 'in' in the order 'idx'
    in_t permute(const in_t &in, const arma::uvec &idx)
    {
        in_t o = in;
        o.fbs = in.fbs.cols(idx), o.lbs = in.lbs.cols(idx), o.pg = in.pg.rows(idx), o.pl = in.pl(idx);
        o.type = in.type(idx), o.tap = in.tap(idx);
        for (arma::uword i = 0; i < idx.n_elem; ++i)
            for (arma::uword f = 0; f < in.M.n_slices; ++f)
                o.M.slice(f).col(i) = in.M.slice(f).col(idx(i));
        return o;
    }

    void run(const in_t &in, quadriga_lib::gbsm_rays &out)
    {
        quadriga_lib::gbsm_rays_from_paths(in.cf, in.T, in.R, in.fbs, in.lbs, in.pg, in.pl, in.M, out, in.method,
                                           in.has_type ? &in.type : nullptr, in.has_tap ? &in.tap : nullptr,
                                           in.has_eps ? &in.er : nullptr, in.has_eps ? &in.ei : nullptr);
    }

    // Angle between the directions (az1, el1) and (az2, el2) in [rad]
    double dir_diff(double az1, double el1, double az2, double el2)
    {
        return ang_dist(unit_vec(az1, el1), unit_vec(az2, el2));
    }

    // Azimuth and elevation of the direction from A to B, written out
    void dir_of(const arma::vec &A, const arma::vec &B, double &az, double &el)
    {
        const arma::vec d = B - A;
        az = std::atan2(d(1), d(0)), el = std::atan2(d(2), std::hypot(d(0), d(1)));
    }
}

TEST_CASE("gbsm_rays_from_paths - delays, angles and powers from the path data")
{
    const quadriga_lib::gbsm_rays r = make_rays(31, true, true, 6, 2, "38.901", true);
    const arma::uword n_path = r.n_ray();
    quadriga_lib::gbsm_rays o;
    run(in_of(r), o);

    CHECK(o.is_valid().empty());
    CHECK(o.method == "38.901");
    CHECK(arma::approx_equal(o.center_frequency, r.center_frequency, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.tx_pos, r.tx_pos, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.rx_pos, r.rx_pos, "absdiff", 0.0));
    REQUIRE(o.n_ray() == n_path);
    CHECK(arma::approx_equal(o.delay, r.delay, "absdiff", 1e-12 * arma::max(r.delay)));
    CHECK(arma::approx_equal(o.power, r.power, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.M, r.M, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.fbs_pos, r.fbs_pos, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.lbs_pos, r.lbs_pos, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.gr_eps_re, r.gr_eps_re, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.gr_eps_im, r.gr_eps_im, "absdiff", 0.0));
    CHECK(arma::all(o.path_type == r.path_type));
    CHECK(arma::all(o.tap == r.tap));
    CHECK(o.fallback.empty());

    QD_DIAG("\n# diag: gbsm_rays_from_paths vs make_rays (delay in s, angles in rad)\n");
    QD_DIAG("path,type,d_delay,d_dep,d_arr\n");
    for (arma::uword p = 0; p < n_path; ++p)
    {
        INFO("path " << p << ", type " << r.path_type(p));
        const double d_dep = dir_diff(o.aod(p), o.eod(p), r.aod(p), r.eod(p));
        const double d_arr = dir_diff(o.aoa(p), o.eoa(p), r.aoa(p), r.eoa(p));
        CHECK(d_dep < 1e-12);
        CHECK(d_arr < 1e-12);
        CHECK(std::abs(wrap_rad(o.aod(p) - r.aod(p))) < 1e-12);
        CHECK(std::abs(wrap_rad(o.aoa(p) - r.aoa(p))) < 1e-12);
        QD_DIAG("%d,%d,%.3e,%.3e,%.3e\n", (int)p, (int)r.path_type(p), o.delay(p) - r.delay(p), d_dep, d_arr);
    }
}

TEST_CASE("gbsm_rays_from_paths - LOS detection without path_type and the optional tap")
{
    quadriga_lib::gbsm_rays o;

    SECTION("LOS detected, also when it is not path 0; one tap per path")
    {
        const quadriga_lib::gbsm_rays r = make_rays(33, true, false, 5, 1, "quadriga", true);
        in_t in = permute(in_of(r), {1, 2, 0, 3, 4, 5});
        in.has_type = false, in.has_tap = false;
        run(in, o);
        CHECK(arma::all(o.path_type == arma::u32_vec{2, 2, 0, 2, 2, 2}));
        CHECK(arma::all(o.tap == arma::u32_vec{6, 9, 0, 12, 15, 18}));
        CHECK(dir_diff(o.aod(2), o.eod(2), r.aod(0), r.eod(0)) < 1e-12);
        CHECK(dir_diff(o.aoa(2), o.eoa(2), r.aoa(0), r.eoa(0)) < 1e-12);
    }
    SECTION("path_type given, default taps: LOS 0, GR 3, others 6, 9, ... in path order")
    {
        const quadriga_lib::gbsm_rays r = make_rays(35, true, true, 4, 1, "quadriga", true);
        in_t in = permute(in_of(r), {2, 1, 3, 0, 4, 5});
        in.has_tap = false;
        run(in, o);
        CHECK(arma::all(o.path_type == arma::u32_vec{2, 1, 2, 0, 2, 2}));
        CHECK(arma::all(o.tap == arma::u32_vec{6, 3, 9, 0, 12, 15}));
    }
    SECTION("LOS tolerance 0.1 mm in the scatterer distance and the excess length")
    {
        const quadriga_lib::gbsm_rays r = make_rays(37, true, false, 1, 1, "quadriga", true);
        in_t in = in_of(r); // path 0: LOS at the midpoint, path 1: NLOS
        in.has_type = false;
        in.pl(0) += 0.9e-4;
        run(in, o);
        CHECK(arma::all(o.path_type == arma::u32_vec{0, 2}));
        in.pl(0) += 0.2e-4; // excess 0.11 mm
        run(in, o);
        CHECK(arma::all(o.path_type == arma::u32_vec{2, 2}));
        in = in_of(r), in.has_type = false;
        in.lbs(0, 0) += 1.1e-4; // FBS and LBS 0.11 mm apart
        run(in, o);
        CHECK(arma::all(o.path_type == arma::u32_vec{2, 2}));
        in.lbs(0, 0) -= 0.2e-4; // 0.09 mm
        run(in, o);
        CHECK(arma::all(o.path_type == arma::u32_vec{0, 2}));
    }
    SECTION("GR angles from the image points, also with an end at z = 0")
    {
        // One GR path on the ground at the end that is at z = 0: departure towards (R_x, R_y, -R_z), arrival
        // towards (T_x, T_y, -T_z)
        for (int end : {0, 1})
        {
            INFO("end at z = 0: " << (end == 0 ? "Tx" : "Rx"));
            in_t in;
            in.cf = {3.5e9}, in.T = {3.0, -4.0, end == 0 ? 0.0 : 25.0}, in.R = {120.0, 45.0, end == 1 ? 0.0 : 1.5};
            const arma::vec G = (end == 0) ? in.T : in.R;
            in.fbs = G, in.lbs = G, in.pg = {0.1}, in.pl = {len3(G - in.T) + len3(in.R - G)};
            in.M.zeros(8, 1, 1), in.type = {1}, in.has_tap = false, in.has_eps = false;
            run(in, o);
            double az, el;
            dir_of(in.T, arma::vec{in.R(0), in.R(1), -in.R(2)}, az, el);
            CHECK(dir_diff(o.aod(0), o.eod(0), az, el) < 1e-12);
            CHECK(o.eod(0) < 0.0);
            dir_of(in.R, arma::vec{in.T(0), in.T(1), -in.T(2)}, az, el);
            CHECK(dir_diff(o.aoa(0), o.eoa(0), az, el) < 1e-12);
            CHECK(o.eoa(0) < 0.0);
            CHECK(o.tap(0) == 3);
        }
    }
    SECTION("a LOS path with tx_pos = rx_pos has angles 0")
    {
        in_t in;
        in.cf = {3.5e9}, in.T = {1.0, 2.0, 3.0}, in.R = in.T, in.fbs = in.T, in.lbs = in.T, in.pg = {1.0};
        in.pl = {0.0}, in.M.zeros(8, 1, 1), in.has_type = false, in.has_tap = false, in.has_eps = false;
        run(in, o);
        CHECK(o.path_type(0) == 0);
        CHECK(o.tap(0) == 0);
        CHECK(o.delay(0) == 0.0);
        CHECK(o.aod(0) == 0.0);
        CHECK(o.eod(0) == 0.0);
        CHECK(o.aoa(0) == 0.0);
        CHECK(o.eoa(0) == 0.0);
    }
}

TEST_CASE("gbsm_rays_from_paths - a QRT round trip restores the rays")
{
    const quadriga_lib::gbsm_rays r = make_rays(39, true, true, 6, 3, "quadriga", true);
    const arma::uword n_path = r.n_ray();
    const std::string fn = "test_gbsm_rays_from_paths_round_trip.qrt";

    const arma::vec f_GHz = r.center_frequency * 1.0e-9;
    const arma::mat cir_pos = r.rx_pos.t(), cir_ori(1, 3, arma::fill::zeros);
    quadriga_lib::qrt_file_init<double>(fn, f_GHz, cir_pos, cir_ori);
    quadriga_lib::qrt_file_append<double>(
        fn, quadriga_lib::qrt_paths_from_scatterers(r.center_frequency, r.tx_pos, r.rx_pos, r.fbs_pos, r.lbs_pos,
                                                    r.power, r.M, r.path_type),
        r.tx_pos);

    in_t in;
    arma::u32_vec no_int;
    quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &in.cf, &in.T, nullptr, &in.R, nullptr, &in.fbs, &in.lbs,
                                        &in.pg, &in.pl, &in.M, nullptr, nullptr, nullptr, nullptr, nullptr, 1, &no_int);
    std::remove(fn.c_str());
    REQUIRE(no_int.n_elem == n_path);
    in.type = no_int, in.tap = r.tap, in.er = r.gr_eps_re, in.ei = r.gr_eps_im;

    quadriga_lib::gbsm_rays o;
    run(in, o);
    REQUIRE(o.n_ray() == n_path);
    CHECK(o.is_valid().empty());
    CHECK(arma::all(o.path_type == r.path_type));
    CHECK(arma::all(o.tap == r.tap));

    QD_DIAG("\n# diag: gbsm_rays_from_paths after a QRT file (float32), errors and bounds\n");
    QD_DIAG("path,type,d_delay_s,tol_delay_s,d_dep_rad,d_arr_rad,tol_dep_rad,tol_arr_rad,d_jones_rel\n");
    for (arma::uword p = 0; p < n_path; ++p)
    {
        const arma::uword t = r.path_type(p);
        INFO("path " << p << ", type " << t);
        const arma::vec T = r.tx_pos, R = r.rx_pos, F = r.fbs_pos.col(p), L = r.lbs_pos.col(p);

        // Delay: path length within dL = 1e-7·(sum of the absolute coordinates of the points), over c
        const double dL = 1e-7 * (arma::accu(arma::abs(T)) + arma::accu(arma::abs(R)) + arma::accu(arma::abs(F)) +
                                  arma::accu(arma::abs(L)));
        const double d_delay = std::abs(o.delay(p) - r.delay(p));
        CHECK(d_delay <= dL / c0);

        // Angles: 2e-7·max|coordinate| / b, b = distance from the end to its next point (LOS: the midpoint)
        const double cmax = std::max({arma::abs(T).max(), arma::abs(R).max(), arma::abs(F).max(), arma::abs(L).max()});
        const double b_t = len3(F - T), b_r = len3(L - R);
        const double d_dep = dir_diff(o.aod(p), o.eod(p), r.aod(p), r.eod(p));
        const double d_arr = dir_diff(o.aoa(p), o.eoa(p), r.aoa(p), r.eoa(p));
        CHECK(d_dep <= 2e-7 * cmax / b_t);
        CHECK(d_arr <= 2e-7 * cmax / b_r);

        // sqrt(power)·M per complex entry
        double d_rel = 0.0;
        for (arma::uword f = 0; f < r.n_freq(); ++f)
        {
            const double s_w = std::sqrt(r.power(p, f)), s_r = std::sqrt(o.power(p, f));
            double ref = 0.0, d = 0.0;
            for (arma::uword e = 0; e < 8; ++e)
                ref = std::max(ref, std::abs(s_w * r.M(e, p, f))),
                d = std::max(d, std::abs(s_r * o.M(e, p, f) - s_w * r.M(e, p, f)));
            CHECK(d <= 1e-6 * ref);
            d_rel = std::max(d_rel, (ref > 0.0) ? d / ref : d);
        }
        QD_DIAG("%d,%d,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e\n", (int)p, (int)t, d_delay, dL / c0, d_dep, d_arr,
                2e-7 * cmax / b_t, 2e-7 * cmax / b_r, d_rel);
    }

    // Without path_type, the file LOS path (0 interactions) is found and the GR path counts as NLOS
    in.has_type = false;
    run(in, o);
    CHECK(o.path_type(0) == 0);
    CHECK(arma::accu(o.path_type == 0) == 1);
}

TEST_CASE("gbsm_rays_from_paths - no_int of qrt_file_read passes as path_type")
{
    // Files of qrt_paths_from_scatterers: the interaction count (LOS 0, GR 1, NLOS 2) is the path type, and the
    // u32 vector of qrt_file_read goes into gbsm_rays_from_paths without conversion
    const quadriga_lib::gbsm_rays r = make_rays(41, true, true, 5, 1, "quadriga", true);
    const std::string fn = "test_gbsm_rays_from_paths_no_int.qrt";
    const arma::mat cir_pos = r.rx_pos.t(), cir_ori(1, 3, arma::fill::zeros);
    quadriga_lib::qrt_file_init<double>(fn, r.center_frequency * 1.0e-9, cir_pos, cir_ori);
    quadriga_lib::qrt_file_append<double>(
        fn, quadriga_lib::qrt_paths_from_scatterers(r.center_frequency, r.tx_pos, r.rx_pos, r.fbs_pos, r.lbs_pos,
                                                    r.power, r.M, r.path_type),
        r.tx_pos);
    arma::vec cf, T, R, pl;
    arma::mat fbs, lbs, pg;
    arma::cube M;
    arma::u32_vec no_int;
    quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &cf, &T, nullptr, &R, nullptr, &fbs, &lbs, &pg, &pl, &M,
                                        nullptr, nullptr, nullptr, nullptr, nullptr, 1, &no_int);
    std::remove(fn.c_str());

    quadriga_lib::gbsm_rays o;
    quadriga_lib::gbsm_rays_from_paths(cf, T, R, fbs, lbs, pg, pl, M, o, "quadriga", &no_int);
    CHECK(o.is_valid().empty());
    CHECK(arma::all(o.path_type == r.path_type));
    CHECK(arma::accu(o.path_type == 1) == 1);

    // Ray-traced files count the interactions: a path with 3 throws
    no_int(no_int.n_elem - 1) = 3;
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_rays_from_paths(cf, T, R, fbs, lbs, pg, pl, M, o, "quadriga", &no_int),
                         std::invalid_argument,
                         Message("Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS)."));
}

TEST_CASE("gbsm_rays_from_paths - a GR point off the ground throws; permittivity is optional")
{
    const quadriga_lib::gbsm_rays r = make_rays(41, true, true, 2, 2, "38.901", true);
    REQUIRE(r.path_type(1) == 1);
    const std::string msg = "The interaction point of the GR path must be on the ground (z = 0); shift local "
                            "coordinates in x and y only.";
    quadriga_lib::gbsm_rays o;
    in_t in = in_of(r);
    for (double z : {1.1e-3, -1.1e-3})
    {
        in.lbs(2, 1) = z;
        CHECK_THROWS_MATCHES(run(in, o), std::invalid_argument, Message(msg));
    }
    in.lbs(2, 1) = 0.9e-3;
    CHECK_NOTHROW(run(in, o));
    in.lbs(2, 1) = 1.1e-3, in.has_type = false; // without path_type no path is GR
    CHECK_NOTHROW(run(in, o));

    in = in_of(r);
    run(in, o);
    CHECK(arma::approx_equal(o.gr_eps_re, r.gr_eps_re, "absdiff", 0.0));
    CHECK(arma::approx_equal(o.gr_eps_im, r.gr_eps_im, "absdiff", 0.0));
    in.has_eps = false; // omitted: cleared in the pre-initialised object
    run(in, o);
    CHECK(o.gr_eps_re.empty());
    CHECK(o.gr_eps_im.empty());
    CHECK(o.is_valid().empty());
}

TEST_CASE("gbsm_rays_from_paths - a pre-initialised object is written in place")
{
    const quadriga_lib::gbsm_rays r = make_rays(43, true, true, 20, 2, "quadriga", true);
    const arma::uword n_path = r.n_ray();
    quadriga_lib::gbsm_rays o;
    run(in_of(r), o);
    o.fallback.zeros(n_path), o.fallback(4) = 1;
    o.delay.fill(7.0), o.aod.fill(7.0), o.eod.fill(7.0), o.aoa.fill(7.0), o.eoa.fill(7.0), o.tap.fill(7);

    const double *p_delay = o.delay.memptr(), *p_power = o.power.memptr(), *p_aod = o.aod.memptr();
    const double *p_eod = o.eod.memptr(), *p_aoa = o.aoa.memptr(), *p_eoa = o.eoa.memptr(), *p_M = o.M.memptr();
    const double *p_fbs = o.fbs_pos.memptr(), *p_lbs = o.lbs_pos.memptr();
    const unsigned *p_type = o.path_type.memptr(), *p_tap = o.tap.memptr();

    // New data: doubled gains, method "38.901", NLOS path 2 with FBS = Tx, path 3 with LBS = Rx, path 4 with both
    in_t in = in_of(r);
    in.pg *= 2.0, in.method = "38.901";
    in.fbs.col(2) = in.T, in.lbs.col(3) = in.R, in.fbs.col(4) = in.T, in.lbs.col(4) = in.R;
    for (arma::uword p = 2; p < 5; ++p)
        in.pl(p) = polyline(in.T, in.fbs.col(p), in.lbs.col(p), in.R);
    run(in, o);

    CHECK(o.delay.memptr() == p_delay);
    CHECK(o.power.memptr() == p_power);
    CHECK(o.aod.memptr() == p_aod);
    CHECK(o.eod.memptr() == p_eod);
    CHECK(o.aoa.memptr() == p_aoa);
    CHECK(o.eoa.memptr() == p_eoa);
    CHECK(o.M.memptr() == p_M);
    CHECK(o.fbs_pos.memptr() == p_fbs);
    CHECK(o.lbs_pos.memptr() == p_lbs);
    CHECK(o.path_type.memptr() == p_type);
    CHECK(o.tap.memptr() == p_tap);
    CHECK(o.fallback.empty());
    CHECK(o.method == "38.901");
    CHECK(arma::approx_equal(o.power, 2.0 * r.power, "absdiff", 0.0));
    CHECK(arma::all(o.tap == r.tap));
    CHECK(o.is_valid().empty());

    // Next-point rule: departure towards the first of (FBS, LBS, R) that differs from T, arrival towards the first
    // of (LBS, FBS, T) that differs from R
    const arma::vec F3 = in.fbs.col(3), L2 = in.lbs.col(2);
    const struct
    {
        arma::uword p;
        arma::vec dep, arr; // next points
    } cases[] = {{2, L2, L2}, {3, F3, F3}, {4, in.R, in.T}};
    for (const auto &c : cases)
    {
        INFO("path " << c.p);
        double az, el;
        dir_of(in.T, c.dep, az, el);
        CHECK(dir_diff(o.aod(c.p), o.eod(c.p), az, el) < 1e-12);
        dir_of(in.R, c.arr, az, el);
        CHECK(dir_diff(o.aoa(c.p), o.eoa(c.p), az, el) < 1e-12);
        CHECK(std::abs(o.delay(c.p) - (in.pl(c.p) - len3(in.R - in.T)) / c0) < 1e-20);
    }
    for (arma::uword p = 5; p < n_path; ++p)
        CHECK(std::abs(o.delay(p) - r.delay(p)) <= 1e-12 * arma::max(r.delay));
}

TEST_CASE("gbsm_rays_from_paths - input validation")
{
    const quadriga_lib::gbsm_rays r = make_rays(45, true, true, 4, 2, "quadriga", true);
    const in_t base = in_of(r);
    quadriga_lib::gbsm_rays o;
    CHECK_NOTHROW(run(base, o));
    const double nan = arma::datum::nan;

    auto fails = [&](const std::function<void(in_t &)> &edit, const std::string &msg)
    {
        in_t in = base;
        edit(in);
        quadriga_lib::gbsm_rays out;
        CHECK_THROWS_MATCHES(run(in, out), std::invalid_argument, Message(msg));
    };

    SECTION("center_frequency")
    {
        fails([](in_t &in) { in.cf.reset(); }, "Input 'center_frequency' cannot be empty.");
        fails([](in_t &in) { in.cf(1) = -1.0; }, "Input 'center_frequency' must be finite and > 0.");
        fails([&](in_t &in) { in.cf(0) = nan; }, "Input 'center_frequency' must be finite and > 0.");
    }
    SECTION("tx_pos, rx_pos")
    {
        fails([](in_t &in) { in.T.resize(2); }, "Input 'tx_pos' must have 3 elements.");
        fails([&](in_t &in) { in.T(1) = nan; }, "Input 'tx_pos' must be finite.");
        fails([](in_t &in) { in.R.resize(4); }, "Input 'rx_pos' must have 3 elements.");
        fails([&](in_t &in) { in.R(2) = nan; }, "Input 'rx_pos' must be finite.");
    }
    SECTION("fbs_pos, lbs_pos")
    {
        fails([](in_t &in) { in.lbs.shed_col(0); }, "Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].");
        fails([](in_t &in) { in.lbs.shed_row(2); }, "Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].");
        fails([&](in_t &in) { in.fbs(0, 4) = nan; }, "Inputs 'fbs_pos' and 'lbs_pos' must be finite.");
    }
    SECTION("path_gain")
    {
        fails([](in_t &in) { in.pg.shed_row(0); }, "Input 'path_gain' must have size [n_path, n_freq].");
        fails([](in_t &in) { in.pg(3, 1) = -1.0; }, "Input 'path_gain' must be finite and >= 0.");
        fails([&](in_t &in) { in.pg(3, 0) = nan; }, "Input 'path_gain' must be finite and >= 0.");
    }
    SECTION("path_length")
    {
        fails([](in_t &in) { in.pl.resize(2); }, "Input 'path_length' must have n_path elements.");
        fails([](in_t &in) { in.pl(2) = -1.0; }, "Input 'path_length' must be finite and >= 0.");
        fails([&](in_t &in) { in.pl(2) = nan; }, "Input 'path_length' must be finite and >= 0.");
    }
    SECTION("M")
    {
        fails([](in_t &in) { in.M.shed_slice(0); }, "Input 'M' must have size [8, n_path, n_freq].");
        fails([](in_t &in) { in.M.shed_col(0); }, "Input 'M' must have size [8, n_path, n_freq].");
        fails([&](in_t &in) { in.M(0, 1, 1) = nan; }, "Input 'M' must be finite.");
    }
    SECTION("method")
    {
        fails([](in_t &in) { in.method = "38.901-MF"; }, "Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
    }
    SECTION("path_type")
    {
        fails([](in_t &in) { in.type.resize(5); }, "Input 'path_type' must have n_path elements.");
        fails([](in_t &in) { in.type(2) = 3; }, "Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).");
        fails([](in_t &in) { in.type(2) = 0; }, "Input 'path_type' can hold at most one LOS and one GR path.");
        fails([](in_t &in) { in.type(3) = 1; }, "Input 'path_type' can hold at most one LOS and one GR path.");
        // Without path_type: a second path on the direct line
        fails([](in_t &in)
              { in.has_type = false, in.fbs.col(3) = in.fbs.col(0), in.lbs.col(3) = in.lbs.col(0), in.pl(3) = in.pl(0); },
              "More than one path matches the LOS geometry; pass 'path_type'.");
    }
    SECTION("tap")
    {
        fails([](in_t &in) { in.tap.resize(7); }, "Input 'tap' must have n_path elements.");
    }
    SECTION("permittivity")
    {
        fails([](in_t &in) { in.ei.reset(); }, "Inputs 'gr_eps_re' and 'gr_eps_im' must have n_freq elements.");
        fails([](in_t &in) { in.er.resize(3), in.ei.resize(3); },
              "Inputs 'gr_eps_re' and 'gr_eps_im' must have n_freq elements.");
        fails([](in_t &in) { in.er(1) = 0.9; }, "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.");
        fails([](in_t &in) { in.ei(0) = -0.1; }, "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.");
        fails([&](in_t &in) { in.ei(0) = nan; }, "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.");

        // Only one of the two given
        in_t in = base;
        quadriga_lib::gbsm_rays out;
        const std::string msg = "Inputs 'gr_eps_re' and 'gr_eps_im' must both be given or both be omitted.";
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_rays_from_paths(in.cf, in.T, in.R, in.fbs, in.lbs, in.pg, in.pl, in.M,
                                                                out, in.method, &in.type, &in.tap, &in.er),
                             std::invalid_argument, Message(msg));
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_rays_from_paths(in.cf, in.T, in.R, in.fbs, in.lbs, in.pg, in.pl, in.M,
                                                                out, in.method, &in.type, &in.tap, nullptr, &in.ei),
                             std::invalid_argument, Message(msg));
    }
    SECTION("outputs untouched on error")
    {
        quadriga_lib::gbsm_rays out = r;
        in_t in = base;
        in.method = "x";
        CHECK_THROWS_AS(run(in, out), std::invalid_argument);
        CHECK(arma::approx_equal(out.delay, r.delay, "absdiff", 0.0));
        CHECK(out.method == r.method);
    }
}

TEST_CASE("gbsm_rays_from_paths - outputs that alias inputs")
{
    // An input that is a member of 'rays', or lies on the memory of one, is read while 'rays' is written
    const quadriga_lib::gbsm_rays r = make_rays(47, true, true, 3, 2, "quadriga", true);
    in_t in = in_of(r);
    quadriga_lib::gbsm_rays o = r;
    CHECK_NOTHROW(run(in, o));
    auto call = [&](const arma::vec &cf, const arma::mat &pg, const arma::vec &pl, const arma::u32_vec &type)
    { quadriga_lib::gbsm_rays_from_paths(cf, in.T, in.R, in.fbs, in.lbs, pg, pl, in.M, o, "quadriga", &type); };

    CHECK_THROWS_AS(call(o.center_frequency, in.pg, in.pl, in.type), std::invalid_argument); // center_frequency is 'rays.center_frequency'
    o = r;
    CHECK_THROWS_AS(call(in.cf, o.power, in.pl, in.type), std::invalid_argument); // path_gain is 'rays.power'
    o = r;
    CHECK_THROWS_AS(call(in.cf, in.pg, o.delay, in.type), std::invalid_argument); // path_length is 'rays.delay'
    CHECK(arma::approx_equal(o.delay, r.delay, "absdiff", 0.0));
    o = r;
    CHECK_THROWS_AS(call(in.cf, in.pg, in.pl, o.path_type), std::invalid_argument); // path_type is 'rays.path_type'
    o = r;
    const arma::vec pl = in.pl;
    o.aod.~Col(), ::new (&o.aod) arma::vec(in.pl.memptr(), in.pl.n_elem, false, true);
    CHECK_THROWS_AS(run(in, o), std::invalid_argument); // 'rays.aod' on the memory of 'path_length'
    CHECK(arma::approx_equal(in.pl, pl, "absdiff", 0.0));
    o.aod.~Col(), ::new (&o.aod) arma::vec();
}
