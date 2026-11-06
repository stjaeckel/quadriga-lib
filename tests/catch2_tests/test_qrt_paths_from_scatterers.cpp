// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of qrt_paths_from_scatterers: path objects with folded path gain and free-space loss, interaction codes,
// round trip through qrt_file_init, qrt_file_append and qrt_file_read, sizes, validation. The ray-level reading
// side (gbsm_rays_from_paths) is tested in test_gbsm_rays_from_paths.cpp.
// Oracles: the hand-made ray data of make_rays (qd_test_rays.hpp) with scatterers; polyline lengths and the stored
// Jones matrix J = M·sqrt(path_gain)·L·10^((32.45 + 20·lg f_GHz) / 20), the inverse of the free-space loss that
// qrt_file_read applies (FSPL = 32.45 + 20·lg f_GHz + 20·lg d), written out in the test.
// Tolerances: the path objects and the file store float32; Jones entries 1e-6 relative to the largest entry of the
// path, path lengths 1e-7 times the sum of the absolute coordinates of the points.
// The tests write files with distinct names into the working directory and remove them.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_channel.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    // Polyline length of path p in [m]: LOS the direct line, GR via its point in lbs_pos, NLOS T -> FBS -> LBS -> R
    double path_len(const quadriga_lib::gbsm_rays &r, arma::uword p)
    {
        const arma::vec T = r.tx_pos, R = r.rx_pos, F = r.fbs_pos.col(p), L = r.lbs_pos.col(p);
        if (r.path_type(p) == 0)
            return len3(R - T);
        if (r.path_type(p) == 1)
            return len3(L - T) + len3(R - L);
        return polyline(T, F, L, R);
    }

    // Length bound of a path stored as float32: 1e-7 times the sum of the absolute coordinates of its points
    double len_tol(const quadriga_lib::gbsm_rays &r, arma::uword p)
    {
        return 1e-7 * (arma::accu(arma::abs(r.tx_pos)) + arma::accu(arma::abs(r.rx_pos)) +
                       arma::accu(arma::abs(r.fbs_pos.col(p))) + arma::accu(arma::abs(r.lbs_pos.col(p))));
    }

    // Inverse free-space loss at 1 m as an amplitude factor: 10^((32.45 + 20·lg f_GHz) / 20)
    double inv_fspl_1m(double f) { return std::pow(10.0, (32.45 + 20.0 * std::log10(f * 1.0e-9)) / 20.0); }

    std::vector<quadriga_lib::path> paths_of(const quadriga_lib::gbsm_rays &r, arma::uword i_cir = 0)
    {
        return quadriga_lib::qrt_paths_from_scatterers(r.center_frequency, r.tx_pos, r.rx_pos, r.fbs_pos, r.lbs_pos,
                                                       r.power, r.M, r.path_type, i_cir);
    }

    // Largest difference between the stored Jones entries of frequency f and J = M·s; ref = largest entry of J
    void jones_diff(const quadriga_lib::path &q, const arma::cube &M, arma::uword p, arma::uword f, double s,
                    double &d, double &ref)
    {
        d = 0.0, ref = 0.0;
        for (arma::uword e = 0; e < 8; ++e)
            ref = std::max(ref, std::abs(M(e, p, f) * s)),
            d = std::max(d, std::abs((double)q.xpr_coeff(f)[e] - M(e, p, f) * s));
    }
}

TEST_CASE("qrt_paths_from_scatterers - path objects carry the geometry, the codes and the folded Jones matrices")
{
    const quadriga_lib::gbsm_rays r = make_rays(21, true, true, 5, 3, "quadriga", true);
    const arma::uword n_path = r.n_ray(), n_freq = r.n_freq();
    const std::vector<quadriga_lib::path> P = paths_of(r, 7);
    REQUIRE(P.size() == n_path);

    QD_DIAG("\n# diag: qrt_paths_from_scatterers, stored Jones vs oracle (largest relative difference per path)\n");
    QD_DIAG("path,type,length,d_rel\n");
    for (arma::uword p = 0; p < n_path; ++p)
    {
        const quadriga_lib::path &q = P[p];
        const arma::uword t = r.path_type(p);
        INFO("path " << p << ", type " << t);
        CHECK(q.n_freq() == n_freq);
        CHECK(!q.is_scalar());
        CHECK(q.iC == 7);
        REQUIRE(q.n_seg() == (t == 0 ? 0 : t == 1 ? 1 : 2));
        CHECK(q.nREF == (t == 1 ? 1 : 0));
        CHECK(q.nSCT == (t == 2 ? 2 : 0));
        CHECK(q.nTRA == 0);
        CHECK(q.nSUB == 0);
        const std::vector<uint8_t> codes = q.interaction_type_codes();
        if (t == 1)
        {
            CHECK(codes == std::vector<uint8_t>{129});
            for (arma::uword i = 0; i < 3; ++i)
                CHECK(q.coord(0)[i] == (float)r.lbs_pos(i, p));
        }
        if (t == 2)
        {
            CHECK(codes == std::vector<uint8_t>{193, 193});
            for (arma::uword i = 0; i < 3; ++i)
            {
                CHECK(q.coord(0)[i] == (float)r.fbs_pos(i, p));
                CHECK(q.coord(1)[i] == (float)r.lbs_pos(i, p));
            }
        }

        // Member length: polyline Tx -> last interaction (0 for LOS)
        const double L = path_len(r, p);
        const double L_int = (t == 0) ? 0.0 : L - len3(r.rx_pos - r.lbs_pos.col(p));
        CHECK(std::abs((double)q.length() - L_int) <= 1e-6 * L);

        double d_max = 0.0;
        for (arma::uword f = 0; f < n_freq; ++f)
        {
            double d, ref;
            jones_diff(q, r.M, p, f, std::sqrt(r.power(p, f)) * L * inv_fspl_1m(r.center_frequency(f)), d, ref);
            CHECK(d <= 1e-6 * ref);
            d_max = std::max(d_max, (ref > 0.0) ? d / ref : d);
        }
        QD_DIAG("%d,%d,%.6f,%.3e\n", (int)p, (int)t, L, d_max);
    }
}

TEST_CASE("qrt_paths_from_scatterers - qrt_file_read returns path gain times M and the polyline")
{
    const quadriga_lib::gbsm_rays r = make_rays(23, true, true, 6, 3, "quadriga", true);
    const arma::uword n_path = r.n_ray(), n_freq = r.n_freq();
    const std::string fn = "test_qrt_paths_from_scatterers_read.qrt";

    const arma::vec f_GHz = r.center_frequency * 1.0e-9;
    const arma::mat cir_pos = r.rx_pos.t(), cir_ori(1, 3, arma::fill::zeros);
    quadriga_lib::qrt_file_init<double>(fn, f_GHz, cir_pos, cir_ori);
    quadriga_lib::qrt_file_append<double>(fn, paths_of(r), r.tx_pos);

    QD_DIAG("\n# diag: qrt_paths_from_scatterers, file round trip (largest relative difference per path)\n");
    QD_DIAG("normalize_M,path,type,d_len_m,tol_len_m,d_jones_rel\n");
    for (int norm : {0, 1})
    {
        INFO("normalize_M " << norm);
        arma::vec cf, tx, rx, pl;
        arma::mat fbs, lbs, pg;
        arma::cube M;
        arma::u32_vec no_int;
        std::vector<uint8_t> itype;
        quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &cf, &tx, nullptr, &rx, nullptr, &fbs, &lbs, &pg, &pl, &M,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, norm, &no_int, nullptr, &itype);
        REQUIRE(pg.n_rows == n_path);
        REQUIRE(pg.n_cols == n_freq);
        REQUIRE(M.n_rows == 8);
        REQUIRE(M.n_cols == n_path);
        REQUIRE(M.n_slices == n_freq);
        REQUIRE(no_int.n_elem == n_path);
        CHECK(arma::approx_equal(cf, r.center_frequency, "reldiff", 1e-7));

        std::vector<uint8_t> codes;
        for (arma::uword p = 0; p < n_path; ++p)
        {
            const arma::uword t = r.path_type(p);
            INFO("path " << p << ", type " << t);
            CHECK(no_int(p) == t);
            if (t == 1)
                codes.push_back(129);
            if (t == 2)
                codes.push_back(193), codes.push_back(193);
            const double d_len = std::abs(pl(p) - path_len(r, p));
            CHECK(d_len <= len_tol(r, p));

            double d_rel = 0.0;
            for (arma::uword f = 0; f < n_freq; ++f)
            {
                // sqrt(path_gain)·M per complex entry; catches a conjugated or phase-rotated matrix
                const double s_w = std::sqrt(r.power(p, f)), s_r = std::sqrt(pg(p, f));
                double ref = 0.0, d = 0.0, ref2 = 0.0, d2 = 0.0;
                for (arma::uword e = 0; e < 8; ++e)
                    ref = std::max(ref, std::abs(s_w * r.M(e, p, f))),
                    d = std::max(d, std::abs(s_r * M(e, p, f) - s_w * r.M(e, p, f)));
                CHECK(d <= 1e-6 * ref);
                d_rel = std::max(d_rel, (ref > 0.0) ? d / ref : d);

                // Documented property: path_gain·abs(M)² per complex entry
                for (arma::uword e = 0; e < 8; e += 2)
                {
                    const double w = r.power(p, f) * (r.M(e, p, f) * r.M(e, p, f) + r.M(e + 1, p, f) * r.M(e + 1, p, f));
                    const double g = pg(p, f) * (M(e, p, f) * M(e, p, f) + M(e + 1, p, f) * M(e + 1, p, f));
                    ref2 = std::max(ref2, w), d2 = std::max(d2, std::abs(g - w));
                }
                CHECK(d2 <= 1e-6 * ref2);
            }
            QD_DIAG("%d,%d,%d,%.3e,%.3e,%.3e\n", norm, (int)p, (int)t, d_len, len_tol(r, p), d_rel);
        }
        CHECK(itype == codes);
    }
    std::remove(fn.c_str());
}

TEST_CASE("qrt_paths_from_scatterers - n_path = 0 and 127 frequencies")
{
    const quadriga_lib::gbsm_rays r0 = make_rays(25, false, false, 0, 2, "quadriga", true);
    CHECK(paths_of(r0).empty());

    // 127 frequencies: a LOS and an NLOS path, the folded Jones matrix of the last frequency
    const quadriga_lib::gbsm_rays r = make_rays(27, true, false, 1, 1, "quadriga", true);
    const arma::vec cf = arma::linspace(1.0e9, 100.0e9, 127);
    const arma::mat pg = arma::repmat(r.power, 1, 127);
    arma::cube M(8, 2, 127);
    M.each_slice() = r.M.slice(0);
    const std::vector<quadriga_lib::path> P =
        quadriga_lib::qrt_paths_from_scatterers(cf, r.tx_pos, r.rx_pos, r.fbs_pos, r.lbs_pos, pg, M, r.path_type);
    REQUIRE(P.size() == 2);
    for (arma::uword p = 0; p < 2; ++p)
    {
        INFO("path " << p);
        CHECK(P[p].n_freq() == 127);
        double d, ref;
        jones_diff(P[p], M, p, 126, std::sqrt(pg(p, 126)) * path_len(r, p) * inv_fspl_1m(cf(126)), d, ref);
        CHECK(d <= 1e-6 * ref);
    }
}

TEST_CASE("qrt_paths_from_scatterers - input validation")
{
    const quadriga_lib::gbsm_rays base = make_rays(29, true, true, 4, 2, "quadriga", true);
    CHECK_NOTHROW(paths_of(base));
    const double nan = arma::datum::nan;

    using R = quadriga_lib::gbsm_rays;
    using U = arma::uword;
    auto fails = [&](const std::function<void(R &, U &)> &edit, const std::string &msg)
    {
        R r = base;
        U i_cir = 0;
        edit(r, i_cir);
        CHECK_THROWS_MATCHES(paths_of(r, i_cir), std::invalid_argument, Message(msg));
    };

    SECTION("center_frequency")
    {
        fails([](R &r, U &) { r.center_frequency.reset(); }, "Input 'center_frequency' must have 1 to 127 elements.");
        fails([](R &r, U &)
              { r.center_frequency = arma::linspace(1e9, 2e9, 128); }, "Input 'center_frequency' must have 1 to 127 elements.");
        fails([](R &r, U &) { r.center_frequency(1) = 0.0; }, "Input 'center_frequency' must be finite and > 0.");
        fails([&](R &r, U &) { r.center_frequency(0) = nan; }, "Input 'center_frequency' must be finite and > 0.");
    }
    SECTION("tx_pos, rx_pos")
    {
        fails([](R &r, U &) { r.tx_pos.resize(2); }, "Input 'tx_pos' must have 3 elements.");
        fails([&](R &r, U &) { r.tx_pos(2) = nan; }, "Input 'tx_pos' must be finite.");
        fails([](R &r, U &) { r.rx_pos.resize(4); }, "Input 'rx_pos' must have 3 elements.");
        fails([&](R &r, U &) { r.rx_pos(0) = nan; }, "Input 'rx_pos' must be finite.");
    }
    SECTION("fbs_pos, lbs_pos")
    {
        fails([](R &r, U &) { r.lbs_pos.shed_col(0); }, "Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].");
        fails([](R &r, U &) { r.fbs_pos.shed_row(0); }, "Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].");
        fails([&](R &r, U &) { r.lbs_pos(1, 3) = nan; }, "Inputs 'fbs_pos' and 'lbs_pos' must be finite.");
    }
    SECTION("path_gain")
    {
        fails([](R &r, U &) { r.power.shed_col(1); }, "Input 'path_gain' must have size [n_path, n_freq].");
        fails([](R &r, U &) { r.power(2, 1) = -1e-9; }, "Input 'path_gain' must be finite and >= 0.");
        fails([&](R &r, U &) { r.power(0, 0) = nan; }, "Input 'path_gain' must be finite and >= 0.");
    }
    SECTION("M")
    {
        fails([](R &r, U &) { r.M.shed_slice(1); }, "Input 'M' must have size [8, n_path, n_freq].");
        fails([](R &r, U &) { r.M.shed_row(7); }, "Input 'M' must have size [8, n_path, n_freq].");
        fails([&](R &r, U &) { r.M(3, 2, 1) = nan; }, "Input 'M' must be finite.");
    }
    SECTION("path_type")
    {
        fails([](R &r, U &) { r.path_type.resize(3); }, "Input 'path_type' must have n_path elements.");
        fails([](R &r, U &) { r.path_type(3) = 3; }, "Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).");
        fails([](R &r, U &) { r.path_type(3) = 0; }, "Input 'path_type' can hold at most one LOS and one GR path.");
        fails([](R &r, U &) { r.path_type(3) = 1; }, "Input 'path_type' can hold at most one LOS and one GR path.");
    }
    SECTION("i_cir")
    {
        fails([](R &, U &i) { i = 4294967296ULL; }, "Input 'i_cir' must be < 2^32.");
        CHECK_NOTHROW(paths_of(base, 4294967295ULL));
    }
    SECTION("zero length")
    {
        // Tx = Rx: the LOS path has length 0; the same geometry without a LOS path is valid
        fails([](R &r, U &) { r.rx_pos = r.tx_pos; }, "Inputs 'tx_pos' and 'rx_pos' must differ for a path of zero length.");
        R r = base;
        r.rx_pos = r.tx_pos, r.path_type(0) = 2;
        CHECK_NOTHROW(paths_of(r));
        // Ends 1 nm apart: length 1e-9 m in double, 0 in the float32 coordinates that qrt_file_read uses
        fails([](R &r, U &) { r.rx_pos = r.tx_pos + arma::vec{0.0, 0.0, 1e-9}; },
              "Inputs 'tx_pos' and 'rx_pos' must differ for a path of zero length.");
    }
    SECTION("float32 range")
    {
        fails([](R &r, U &) { r.fbs_pos(0, 3) = 1e39; },
              "Inputs 'tx_pos', 'rx_pos', 'fbs_pos' and 'lbs_pos' must be within the float32 range of QRT files.");
        fails([](R &r, U &) { r.power(2, 0) = 1e80; },
              "Inputs 'path_gain' and 'M' give stored Jones matrices beyond the float32 range of QRT files.");
    }
}
