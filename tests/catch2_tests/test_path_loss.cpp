// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "path_loss*"

// Verification suite for quadriga_lib::path_loss.
//
// Pins down: the path loss and shadow-fading std of the 23 v19 files against TR 38.901 v19.5.0 Table 7.4.1-1
// (including the random environment height of note 1 for UMa), the outdoor distance of the legacy O2I mode, the
// formulas of the 10 parametric models with their defaults, the distance clamps at 0.1 m, the default SF std from
// the LSP formula, the empty model, optional and pre-sized outputs and the input validation.
// Oracles, independent of the function:
//  - TR 38.901 v19.5.0 Table 7.4.1-1 and its notes 1 and 5 typed with c = 3e8 m/s (RMa h = 5 m, W = 20 m; SMa
//    h = 10 m, W = 10 m; hE = 1 m for UMa and UMi), the SF std of the table, and the O2I SF std of Table 7.5-6
//  - the model formulas, typed from their definition (dual_slope with the 3D breakpoint distance beyond the
//    breakpoint, tripple_slope, nlos with the maximum of up to three terms, winner_los with the breakpoint from the
//    heights of each link and C2 beyond it, winner_nlos, winner_pathloss, logdist, logdist_simple, constant,
//    satellite with the zenith attenuation table typed below, interpolated linearly by interp_1D)
//  - the legacy O2I distance rule (indoor part of the 3D distance, 10 m minimum outdoor distance)
//  - the LSP std formula for the default SF std, with both limits at 0
// Tolerances: the v19 files write derived constants rounded to 4 significant digits; the TR checks take the bound of
// that rounding over the TR ranges (stated at the check); typed model formulas 1e-10 dB; exact-by-construction
// results 1e-12 (relative for dB values of order 100).
// Conventions: positions are rows {x, y, z} in meters; frequencies in Hz at the interface, f in GHz in formulas;
// TR reference geometry tx = (0, 0, hBS), rx = (d2D, 0, hUT).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_math.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    typedef std::unordered_map<std::string, double> scen_t;
    typedef std::unordered_map<std::string, std::vector<double>> plpar_t;

    const double pi = 3.141592653589793;
    const double c_tr = 3.0e8; // speed of light of the TR breakpoint formulas

    double lg(double x) { return std::log10(x); }

    scen_t conf(const std::string &name, plpar_t *pl = nullptr, std::string *model = nullptr)
    {
        scen_t sp;
        quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", sp, nullptr, model, pl);
        return sp;
    }

    bool close_rel(double a, double b, double tol) { return std::abs(a - b) <= tol * std::max(1.0, std::abs(b)); }

    // ---------------------------------------------------------------------------------------------------------
    // TR 38.901 v19.5.0, Table 7.4.1-1 (f in GHz, distances and heights in m)

    // RMa / SMa LOS with building height h, break point of note 5: dBP = 2 pi hBS hUT fc / c.
    // PL2 = PL1(dBP) + 40 lg(d3D / dBP) as in the TR (form3d = false), or with the 3D breakpoint distance
    // sqrt(dBP^2 + (hBS - hUT)^2) in place of dBP (form3d = true)
    double tr_rma_los(double h, double hBS, double hUT, double d2, double f, bool form3d, bool &beyond)
    {
        const double d3 = std::sqrt(d2 * d2 + (hBS - hUT) * (hBS - hUT));
        auto PL1 = [&](double x)
        {
            return 20.0 * lg(40.0 * pi * x * f / 3.0) + std::min(0.03 * std::pow(h, 1.72), 10.0) * lg(x) -
                   std::min(0.044 * std::pow(h, 1.72), 14.77) + 0.002 * lg(h) * x;
        };
        const double dBP = 2.0 * pi * hBS * hUT * f * 1.0e9 / c_tr;
        beyond = d2 > dBP;
        if (!beyond)
            return PL1(d3);
        const double db = form3d ? std::sqrt(dBP * dBP + (hBS - hUT) * (hBS - hUT)) : dBP;
        return PL1(db) + 40.0 * lg(d3 / db);
    }

    // RMa NLOS' and SMa NLOS with building height h and street width W
    double tr_rma_nlos_prime(double h, double W, double hBS, double hUT, double d3, double f)
    {
        const double g = lg(11.75 * hUT);
        return 161.04 - 7.1 * lg(W) + 7.5 * lg(h) - (24.37 - 3.7 * (h / hBS) * (h / hBS)) * lg(hBS) +
               (43.42 - 3.1 * lg(hBS)) * (lg(d3) - 3.0) + 20.0 * lg(f) - (3.2 * g * g - 4.97);
    }

    // UMa LOS with environment height hE (note 1: d'BP = 4 h'BS h'UT fc / c)
    double tr_uma_los(double hBS, double hUT, double d2, double f, double hE)
    {
        const double d3 = std::sqrt(d2 * d2 + (hBS - hUT) * (hBS - hUT));
        const double dBP = 4.0 * (hBS - hE) * (hUT - hE) * f * 1.0e9 / c_tr;
        if (d2 <= dBP)
            return 28.0 + 22.0 * lg(d3) + 20.0 * lg(f);
        return 28.0 + 40.0 * lg(d3) + 20.0 * lg(f) - 9.0 * lg(dBP * dBP + (hBS - hUT) * (hBS - hUT));
    }

    double tr_uma_nlos_prime(double hUT, double d3, double f)
    {
        return 13.54 + 39.08 * lg(d3) + 20.0 * lg(f) - 0.6 * (hUT - 1.5);
    }

    // UMi LOS, hE = 1 m (note 1)
    double tr_umi_los(double hBS, double hUT, double d2, double f)
    {
        const double d3 = std::sqrt(d2 * d2 + (hBS - hUT) * (hBS - hUT));
        const double dBP = 4.0 * (hBS - 1.0) * (hUT - 1.0) * f * 1.0e9 / c_tr;
        if (d2 <= dBP)
            return 32.4 + 21.0 * lg(d3) + 20.0 * lg(f);
        return 32.4 + 40.0 * lg(d3) + 20.0 * lg(f) - 9.5 * lg(dBP * dBP + (hBS - hUT) * (hBS - hUT));
    }

    // Path loss and SF std of a v19 file at one point. pl3: the same with the 3D breakpoint form for the RMa and
    // SMa LOS term; rs_beyond: the RMa / SMa LOS term enters beyond its breakpoint; sl_wins: InF-DL point where
    // the InF-SL term is the largest
    double tr_file(const std::string &name, double hBS, double hUT, double d2, double f, double &pl3, double &sf,
                   bool &rs_beyond, bool &sl_wins)
    {
        const std::string scen = name.substr(0, name.find('_'));
        const bool nlos = name.find("NLOS") != std::string::npos;
        const bool o2i = name.find("O2I") != std::string::npos;
        const double d3 = std::sqrt(d2 * d2 + (hBS - hUT) * (hBS - hUT));
        rs_beyond = false, sl_wins = false;
        double pl = 0.0;
        if (scen == "RMa" || scen == "SMa")
        {
            const double h = scen == "RMa" ? 5.0 : 10.0, W = scen == "RMa" ? 20.0 : 10.0;
            bool beyond = false;
            const double los = tr_rma_los(h, hBS, hUT, d2, f, false, beyond);
            const double los3 = tr_rma_los(h, hBS, hUT, d2, f, true, beyond);
            if (!nlos)
                pl = los, pl3 = los3, sf = beyond ? 6.0 : 4.0, rs_beyond = beyond;
            else if (scen == "RMa")
            {
                const double np = tr_rma_nlos_prime(h, W, hBS, hUT, d3, f);
                pl = std::max(los, np), pl3 = std::max(los3, np), sf = 8.0, rs_beyond = beyond;
            }
            else // SMa NLOS: no maximum with the LOS term
                pl = pl3 = tr_rma_nlos_prime(h, W, hBS, hUT, d3, f), sf = 8.0;
        }
        else if (scen == "UMa")
        {
            const double los = tr_uma_los(hBS, hUT, d2, f, 1.0);
            pl = nlos ? std::max(los, tr_uma_nlos_prime(hUT, d3, f)) : los;
            sf = nlos ? 6.0 : 4.0;
        }
        else if (scen == "UMi")
        {
            const double los = tr_umi_los(hBS, hUT, d2, f);
            pl = nlos ? std::max(los, 35.3 * lg(d3) + 22.4 + 21.3 * lg(f) - 0.3 * (hUT - 1.5)) : los;
            sf = nlos ? 7.82 : 4.0;
        }
        else if (scen == "Indoor")
        {
            const double los = 32.4 + 17.3 * lg(d3) + 20.0 * lg(f);
            pl = nlos ? std::max(los, 38.3 * lg(d3) + 17.30 + 24.9 * lg(f)) : los;
            sf = nlos ? 8.03 : 3.0;
        }
        else // InF
        {
            const double los = 31.84 + 21.50 * lg(d3) + 19.00 * lg(f);
            const double sl = 33.0 + 25.5 * lg(d3) + 20.0 * lg(f);
            pl = los, sf = 4.3;
            if (name == "InF_NLOS_SL")
                pl = std::max(sl, los), sf = 5.7;
            else if (name == "InF_NLOS_DL")
            {
                const double dl = 18.6 + 35.7 * lg(d3) + 20.0 * lg(f);
                pl = std::max(std::max(dl, los), sl), sf = 7.2;
                sl_wins = sl > dl && sl > los;
            }
            else if (name == "InF_NLOS_SH")
                pl = std::max(32.4 + 23.0 * lg(d3) + 20.0 * lg(f), los), sf = 5.9;
            else if (name == "InF_NLOS_DH")
                pl = std::max(33.63 + 21.9 * lg(d3) + 20.0 * lg(f), los), sf = 4.0;
        }
        if (scen != "RMa" && scen != "SMa")
            pl3 = pl;
        if (o2i) // O2I shadow fading of Table 7.5-6
            sf = (scen == "UMa" || scen == "UMi") ? 7.0 : 8.0;
        return pl;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Model formulas, typed from their definition. [X|v] is the first value of key X if present, else v.

    double pv(const plpar_t &p, const char *k, double def = 0.0)
    {
        auto it = p.find(k);
        return it == p.end() ? def : it->second[0];
    }
    bool ph(const plpar_t &p, const char *k) { return p.count(k) != 0; }

    // dual_slope: hb = max(hBS, 1.1 hE, 0.1), hm the same (random hE: hm = max(hUT, 0.1));
    // dBP = E f (hb - hE)(hm - hE); dBP3 = sqrt(dBP^2 + (hb - hm)^2); PL1(x) = A1 lg x + B + C lg f + D x;
    // d2 <= dBP: PL1(d3), sf = sig1; else PL1(dBP3) + A2 lg(d3 / dBP3), sf = sig2 (each only if present)
    // hE_rnd: NaN = the fixed [hE|0]; region: 1 or 2
    double ref_dual_slope(const plpar_t &p, double f, double d3, double d2, double hBS, double hUT, double hE_rnd,
                          double &sf, int &region)
    {
        const double A1 = ph(p, "A1") ? pv(p, "A1") : pv(p, "A"), A2 = pv(p, "A2", A1);
        const double B = pv(p, "B"), C = pv(p, "C"), D = pv(p, "D"), E = pv(p, "E");
        const bool rnd = !std::isnan(hE_rnd);
        const double hE = rnd ? hE_rnd : pv(p, "hE");
        double hb = hBS;
        if (hb < 1.1 * hE)
            hb = 1.1 * hE;
        if (hb < 0.1)
            hb = 0.1;
        double hm = hUT;
        if (!rnd && hm < 1.1 * hE)
            hm = 1.1 * hE;
        if (hm < 0.1)
            hm = 0.1;
        const double dBP = E * f * (hb - hE) * (hm - hE);
        const double dBP3 = std::sqrt(dBP * dBP + (hb - hm) * (hb - hm));
        auto PL1 = [&](double x)
        { return A1 * lg(x) + B + C * lg(f) + D * x; };
        if (d2 <= dBP)
        {
            region = 1;
            if (ph(p, "sig1"))
                sf = pv(p, "sig1");
            return PL1(d3);
        }
        region = 2;
        if (ph(p, "sig2"))
            sf = pv(p, "sig2");
        return PL1(dBP3) + A2 * lg(d3 / dBP3);
    }

    // nlos: hb = max(hBS, 0.1), hm = max(hUT, 0.1)
    // PLn = An lg d3 + Bn + Cn lg f + Dn lg hb + D1n lg(hb)/hb + D2n lg(hb)/hb^2 + D3n hb + En lg hm + E1n lg(hm)/hm
    //     + E2n lg(hm)/hm^2 + E3n hm + Fn lg(hb) lg(d3) + G1n (lg(G2n hm))^2   (G2n default 1, all others 0)
    // pl = PLn; [LOSmax|1] != 0: max with dual_slope; Ax, Bx or Cx present: max with Ax lg d3 + Bx + Cx lg f
    // sf = sig if present; which: 0 PLn, 1 LOS term, 2 extra term
    double ref_nlos(const plpar_t &p, double f, double d3, double d2, double hBS, double hUT, double hE_rnd, double &sf,
                    int &which)
    {
        const double hb = std::max(hBS, 0.1), hm = std::max(hUT, 0.1);
        const double g = lg(pv(p, "G2n", 1.0) * hm);
        double pl = pv(p, "An") * lg(d3) + pv(p, "Bn") + pv(p, "Cn") * lg(f) + pv(p, "Dn") * lg(hb) +
                    pv(p, "D1n") * lg(hb) / hb + pv(p, "D2n") * lg(hb) / (hb * hb) + pv(p, "D3n") * hb +
                    pv(p, "En") * lg(hm) + pv(p, "E1n") * lg(hm) / hm + pv(p, "E2n") * lg(hm) / (hm * hm) +
                    pv(p, "E3n") * hm + pv(p, "Fn") * lg(hb) * lg(d3) + pv(p, "G1n") * g * g;
        which = 0;
        if (pv(p, "LOSmax", 1.0) != 0.0)
        {
            double sf_unused = 0.0;
            int region = 0;
            const double los = ref_dual_slope(p, f, d3, d2, hBS, hUT, hE_rnd, sf_unused, region);
            if (los > pl)
                pl = los, which = 1;
        }
        if (ph(p, "Ax") || ph(p, "Bx") || ph(p, "Cx"))
        {
            const double x = pv(p, "Ax") * lg(d3) + pv(p, "Bx") + pv(p, "Cx") * lg(f);
            if (x > pl)
                pl = x, which = 2;
        }
        if (ph(p, "sig"))
            sf = pv(p, "sig");
        return pl;
    }

    // Legacy O2I distance rule: indoor part of the 3D distance in proportion of the indoor 2D distance, outdoor part
    // at least 10 m; d2d_in = 0 keeps the geometric distances; both clamped at 0.1 m
    void ref_o2i_dist(double d2_geo, double d3_geo, double d2d_in, double &d2, double &d3)
    {
        d2 = d2_geo, d3 = d3_geo;
        if (d2d_in > 0.0)
        {
            const double din = std::min(d2d_in, d2);
            double d3in = d2 > 0.0 ? d3 * din / d2 : d3;
            d3in = std::min(d3in, d3);
            if (d3 - d3in < 10.0)
                d3in = d3 - 10.0;
            d3in = std::max(d3in, 0.0);
            const double p = d3 > 0.0 ? d3in / d3 : 0.0;
            d3 = d3 - d3in;
            d2 = d2 * (1.0 - p);
        }
        d3 = std::max(d3, 0.1), d2 = std::max(d2, 0.1);
    }

    // Zenith attenuation of the satellite model: frequency in GHz, attenuation in dB (93 values)
    const double zen_f[93] = {
        1, 1.99, 2.97, 3.96, 4.95, 5.92, 6.87, 7.87, 8.83, 9.77, 11.03, 12.32, 13.17, 14.44, 15.65, 16.37, 17.18, 18.03,
        18.8, 19.34, 19.91, 20.42, 20.71, 21.16, 21.49, 21.7, 22.09, 22.52, 22.99, 23.98, 24.93, 26.13, 27.38, 29.18,
        30.44, 32.18, 33.8, 36.09, 38.87, 40.59, 42.33, 44, 44.96, 47.2, 48.42, 49.42, 50.17, 50.96, 51.73, 52.62, 53.32,
        53.6, 53.75, 54.06, 54.38, 55.12, 55.73, 56.54, 56.97, 57.52, 58.01, 58.18, 58.33, 58.62, 58.99, 59.38, 59.57,
        59.84, 60.27, 60.59, 60.93, 61.22, 61.44, 61.93, 62.27, 63.34, 64.27, 65.42, 66.35, 67.19, 68.16, 69.22, 70.33,
        71.66, 73.53, 75.28, 77.48, 80.64, 83.77, 87.07, 92.1, 96.39, 100};
    const double zen_a[93] = {
        0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19,
        0.23, 0.28, 0.35, 0.4, 0.46, 0.51, 0.52, 0.52, 0.49, 0.44, 0.36, 0.3, 0.26, 0.24, 0.23, 0.24, 0.25, 0.27, 0.32,
        0.4, 0.46, 0.56, 0.68, 0.8, 1.16, 1.53, 2.18, 2.99, 4.47, 8.08, 15.49, 30.37, 39.94, 47.33, 60.66, 80.97, 102.6,
        120.55, 137.92, 144.01, 149.57, 153.68, 155.18, 156.63, 157.9, 158.81, 157.57, 156.04, 151.64, 141.33, 130.52,
        119.21, 97.42, 68, 52.74, 39.81, 17.77, 8.61, 4.2, 2.8, 2.19, 1.69, 1.42, 1.22, 1.07, 0.93, 0.85, 0.79, 0.75,
        0.74, 0.75, 0.81, 0.87, 0.97};

    // Linear interpolation of the zenith attenuation (interp_1D), constant beyond 1 and 100 GHz
    double zenith_att(double f)
    {
        return quadriga_lib::interp_1D<double>(arma::vec(zen_a, 93), arma::vec(zen_f, 93), arma::vec{f})(0);
    }

    // Links with per-link Tx used in the model sections: 4 links, heights below 1.5 m included
    const arma::mat mdl_tx = {{0.0, 0.0, 25.0}, {10.0, -5.0, 12.0}, {0.0, 0.0, 1.0}, {5.0, 5.0, 40.0}};
    const arma::mat mdl_rx = {{30.0, 40.0, 1.5}, {-200.0, 300.0, 1.2}, {800.0, -600.0, 2.0}, {3005.0, 5.0, 22.0}};

    void link_geo(const arma::mat &tx, const arma::mat &rx, arma::uword l, double &d2, double &d3, double &hBS,
                  double &hUT)
    {
        const arma::uword t = tx.n_rows == 1 ? 0 : l;
        const double dx = rx(l, 0) - tx(t, 0), dy = rx(l, 1) - tx(t, 1), dz = rx(l, 2) - tx(t, 2);
        d2 = std::sqrt(dx * dx + dy * dy), d3 = std::sqrt(dx * dx + dy * dy + dz * dz);
        hBS = tx(t, 2), hUT = rx(l, 2);
    }
}

TEST_CASE("path_loss - the v19 files reproduce TR 38.901 Table 7.4.1-1")
{
    // Tolerance per file (tol_r): the files write derived constants rounded to 4 significant digits; the bound of
    // the change over the TR ranges (UMa, UMi: hUT 1.5 to 22.5 m, 0.5 to 100 GHz, d2D up to 5 km; RMa: hBS 10 to
    // 150 m, hUT 1 to 10 m, 0.5 to 30 GHz; SMa: hBS 25 to 35 m, hUT 1 to 14 m, 0.5 to 37 GHz; RMa, SMa d2D up to
    // 10 km):
    //  - UMa, UMi: E = 13.33 for 40/3 changes the loss beyond the breakpoint by 19 lg((40/3)/13.33) <= 2.1e-3 dB
    //  - RMa: A1 = 20.48, B = 31.74, D = 0.001398, E = 20.94 of the LOS term at most 8.2e-3 dB; the NLOS term with
    //    Bn = 31.75 differs by 5e-3 dB, so the maximum stays within the bound
    //  - SMa LOS: A1 = 21.574, B = 30.13, E = 20.94 at most 6.6e-3 dB
    //  - InH, InF, SMa NLOS: no rounded constant reaches the result, 1e-9 dB
    //  - O2I files: the bound of their outdoor file
    // Points outside these d2D ranges are skipped: there the bound does not hold (e.g. the breakpoint shift of
    // E = 20.94 times D = 0.002 dB/m grows with dBP).
    // RMa and SMa LOS term beyond the breakpoint: the library evaluates PL2 with the 3D breakpoint distance
    // sqrt(dBP^2 + (hBS - hUT)^2), the TR with dBP. The difference (A1 - A2) lg(dBP3 / dBP) + D (dBP3 - dBP) is at
    // most 0.034 dB over the TR ranges (RMa: hBS 10 to 150 m, hUT 1 to 10 m, 0.5 to 30 GHz, maximum 0.0341 dB at
    // 0.5 GHz and hUT 1 m; SMa: hBS 25 to 35 m, hUT 1 to 14 m, 0.5 to 37 GHz, maximum 0.0313 dB). These points are
    // checked (a) against the TR formula with 0.035 dB + tol_r and (b) against the TR formula with PL2 in the 3D
    // breakpoint form with tol_r, which separates the breakpoint form from the rounding of the constants.

    const std::vector<std::string> names = {
        "InF_LOS", "InF_NLOS_DH", "InF_NLOS_DL", "InF_NLOS_SH", "InF_NLOS_SL", "Indoor_LOS", "Indoor_NLOS",
        "RMa_LOS", "RMa_LOS_O2I", "RMa_NLOS", "RMa_NLOS_O2I", "SMa_LOS", "SMa_LOS_O2I", "SMa_NLOS", "SMa_NLOS_O2I",
        "UMa_LOS", "UMa_LOS_O2I", "UMa_NLOS", "UMa_NLOS_O2I", "UMi_LOS", "UMi_LOS_O2I", "UMi_NLOS", "UMi_NLOS_O2I"};

    QD_DIAG("\n# diag: path_loss against TR 38.901 Table 7.4.1-1, per file\n");
    QD_DIAG("file,n_points,n_rs_beyond,max_err_3d_form,max_err_tr,tol_r,n_sl_wins\n");
    int n_sl_wins_dl = 0;
    for (const std::string &name : names)
    {
        INFO("file " << name);
        plpar_t plpar;
        std::string model;
        const scen_t sp = conf(name, &plpar, &model);
        const std::string scen = name.substr(0, name.find('_'));
        const bool nlos = name.find("NLOS") != std::string::npos;
        const bool los_term_rounded = scen == "RMa" || (scen == "SMa" && !nlos) || scen == "UMa" || scen == "UMi";
        const double tol_r = !los_term_rounded ? 1e-9 : (scen == "RMa" ? 8.2e-3 : (scen == "SMa" ? 6.6e-3 : 2.1e-3));

        // Points (hBS, hUT, f) and the d2D list per scenario; d2D beyond the TR range of the bound is skipped
        std::vector<double> hbs, hut, fgh;
        double d_max = 1e9;
        if (scen == "UMa" || scen == "UMi")
            hbs = {scen == "UMa" ? 25.0 : 10.0}, hut = {1.5, 12.0}, fgh = {3.5, 28.0}, d_max = 5000.0;
        else if (scen == "RMa")
            hbs = {35.0, 25.0}, hut = {1.5, 5.0}, fgh = {0.9, 3.5, 6.0}, d_max = nlos ? 5000.0 : 10000.0;
        else if (scen == "SMa")
            hbs = {25.0, 35.0}, hut = {1.5, 4.5, 13.5}, fgh = {3.5, 20.0}, d_max = 10000.0;
        else if (scen == "Indoor")
            hbs = {3.0}, hut = {1.0}, fgh = {3.5, 28.0, 60.0};
        else
            hbs = {1.5, 8.0}, hut = {1.5}, fgh = {3.5, 28.0};

        int n_pts = 0, n_rs = 0, n_sl = 0;
        double err3 = 0.0, err_tr = 0.0;
        for (double f : fgh)
        {
            std::vector<double> lh_bs, lh_ut, ld2;
            for (double hB : hbs)
                for (double hU : hut)
                {
                    std::vector<double> d2s;
                    if (scen == "UMa" || scen == "UMi")
                    {
                        const double dbp = 4.0 * (hB - 1.0) * (hU - 1.0) * f * 1e9 / c_tr;
                        d2s = {15.0, 0.5 * dbp, 2.0 * dbp, 1000.0, 4500.0};
                    }
                    else if (scen == "RMa" || scen == "SMa")
                    {
                        const double dbp = 2.0 * pi * hB * hU * f * 1e9 / c_tr;
                        d2s = {50.0, 0.5 * dbp, 2.0 * dbp, scen == "RMa" ? 4000.0 : 3000.0};
                    }
                    else if (scen == "Indoor")
                        d2s = {1.0, 5.0, 20.0, 100.0};
                    else
                        d2s = {1.0, 5.0, 10.0, 30.0, 100.0, 500.0};
                    for (double d : d2s)
                        if (d <= d_max)
                            lh_bs.push_back(hB), lh_ut.push_back(hU), ld2.push_back(d);
                }

            const arma::uword n = ld2.size();
            arma::mat tx(n, 3), rx(n, 3);
            for (arma::uword l = 0; l < n; ++l)
            {
                tx(l, 0) = 0.0, tx(l, 1) = 0.0, tx(l, 2) = lh_bs[l];
                rx(l, 0) = ld2[l], rx(l, 1) = 0.0, rx(l, 2) = lh_ut[l];
            }
            arma::mat pl, sf;
            quadriga_lib::path_loss(model, plpar, sp, arma::vec({f * 1e9}), tx, rx, pl, &sf);
            REQUIRE(pl.n_rows == n);
            REQUIRE(pl.n_cols == 1);
            REQUIRE(sf.n_rows == n);

            for (arma::uword l = 0; l < n; ++l)
            {
                double pl3 = 0.0, sf_tr = 0.0;
                bool rs_beyond = false, sl_wins = false;
                const double pl_tr = tr_file(name, lh_bs[l], lh_ut[l], ld2[l], f, pl3, sf_tr, rs_beyond, sl_wins);
                INFO("hBS " << lh_bs[l] << ", hUT " << lh_ut[l] << ", d2D " << ld2[l] << ", f " << f << " GHz, pl "
                            << pl(l, 0) << ", TR " << pl_tr << ", 3D form " << pl3);
                CHECK(std::abs(pl(l, 0) - pl3) <= tol_r);
                if (rs_beyond)
                    CHECK(std::abs(pl(l, 0) - pl_tr) <= 0.035 + tol_r);
                CHECK(sf(l, 0) == sf_tr);
                err3 = std::max(err3, std::abs(pl(l, 0) - pl3));
                err_tr = std::max(err_tr, std::abs(pl(l, 0) - pl_tr));
                ++n_pts, n_rs += rs_beyond ? 1 : 0, n_sl += sl_wins ? 1 : 0;
            }
        }
        if (name == "InF_NLOS_DL")
            n_sl_wins_dl = n_sl;
        QD_DIAG("%s,%d,%d,%.3e,%.3e,%.1e,%d\n", name.c_str(), n_pts, n_rs, err3, err_tr, tol_r, n_sl);
        CHECK(n_pts > 0);
    }
    CHECK(n_sl_wins_dl > 0); // the InF-SL term of InF-DL is the largest at some points
}

TEST_CASE("path_loss - O2I distances: geometric distance without d2d_in, outdoor distance with it")
{
    // UMa NLOS file: the oracle is the nlos model with the file's constants at the distances of the legacy rule
    plpar_t plpar;
    std::string model;
    const scen_t sp = conf("UMa_NLOS", &plpar, &model);
    REQUIRE(model == "nlos");

    // Links: regular; d2d_in > d2D; 10 m minimum outdoor distance (d3D 15 m, d2d_in 12 m); d2D = 0 with
    // d2d_in > 0; d3D = 0; then the first and the fourth link again with d2d_in = 0
    const double z_b = 25.0;
    const double d2_min = std::sqrt(15.0 * 15.0 - 8.5 * 8.5); // d3D = 15 m with hBS 10 m, hUT 1.5 m
    const arma::mat tx = {{0, 0, z_b}, {0, 0, z_b}, {0, 0, 10.0}, {0, 0, z_b}, {7, 3, z_b}, {0, 0, z_b}, {0, 0, z_b}};
    const arma::mat rx = {{120, 50, 1.5}, {12, 0, 1.5}, {d2_min, 0, 1.5}, {0, 0, 1.5}, {7, 3, z_b}, {120, 50, 1.5},
                          {0, 0, 1.5}};
    const arma::vec d2d_in = {8.0, 30.0, 12.0, 5.0, 5.0, 0.0, 0.0};
    const arma::vec fc = {3.5e9, 28.0e9};

    arma::mat pl_geo, pl_o2i, sf;
    quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl_geo);
    quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl_o2i, nullptr, &d2d_in);

    QD_DIAG("\n# diag: path_loss O2I distances (UMa NLOS file)\n");
    QD_DIAG("link,d2d_in,d2_geo,d3_geo,d2_out,d3_out,pl_geo_3.5,pl_o2i_3.5\n");
    for (arma::uword l = 0; l < 7; ++l)
    {
        double d2, d3, hBS, hUT, d2o, d3o, d2g, d3g;
        link_geo(tx, rx, l, d2, d3, hBS, hUT);
        ref_o2i_dist(d2, d3, d2d_in(l), d2o, d3o);
        ref_o2i_dist(d2, d3, 0.0, d2g, d3g);
        QD_DIAG("%d,%g,%g,%g,%g,%g,%.6f,%.6f\n", (int)l, d2d_in(l), d2, d3, d2o, d3o, pl_geo(l, 0), pl_o2i(l, 0));
        for (arma::uword k = 0; k < 2; ++k)
        {
            INFO("link " << l << ", f " << fc(k) / 1e9 << " GHz");
            double s = 0.0;
            int w = 0;
            CHECK(close_rel(pl_geo(l, k), ref_nlos(plpar, fc(k) / 1e9, d3g, d2g, hBS, hUT, NAN, s, w), 1e-12));
            CHECK(close_rel(pl_o2i(l, k), ref_nlos(plpar, fc(k) / 1e9, d3o, d2o, hBS, hUT, NAN, s, w), 1e-12));
            if (d2d_in(l) == 0.0)
                CHECK(pl_o2i(l, k) == pl_geo(l, k));
        }
    }

    // The rule's special cases, written out
    double d2o, d3o;
    ref_o2i_dist(d2_min, 15.0, 12.0, d2o, d3o);
    CHECK(std::abs(d3o - 10.0) < 1e-12); // 10 m minimum outdoor distance
    CHECK(std::abs(d2o - d2_min * 10.0 / 15.0) < 1e-12);
    ref_o2i_dist(0.0, 23.5, 5.0, d2o, d3o);
    CHECK(std::abs(d3o - 10.0) < 1e-12); // d2D = 0: indoor part d3D - 10
    CHECK(d2o == 0.1);
    ref_o2i_dist(0.0, 0.0, 5.0, d2o, d3o);
    CHECK((d3o == 0.1 && d2o == 0.1)); // d3D = 0: indoor fraction 0, clamped
    CHECK(pl_o2i(1, 0) != pl_geo(1, 0));
}

TEST_CASE("path_loss - UMa random hE follows the probability of TR note 1")
{
    plpar_t p_los, p_nlos;
    std::string m_los, m_nlos;
    const scen_t sp_los = conf("UMa_LOS", &p_los, &m_los);
    const scen_t sp_nlos = conf("UMa_NLOS", &p_nlos, &m_nlos);
    REQUIRE(m_los == "dual_slope");
    REQUIRE(m_nlos == "nlos");
    REQUIRE(pv(p_los, "hE_rand") == 1.0);
    REQUIRE(pv(p_nlos, "hE_rand") == 1.0);
    plpar_t p_copy = p_nlos; // the NLOS term never wins: the LOS term of the nlos model is observed
    p_copy["Bn"] = {-1000.0};

    // Tolerance 2.1e-3 dB: rounding bound of PL_E = 13.33 for 40/3 in the UMa files, 19 lg((40/3)/13.33)
    const double tol = 2.1e-3;

    // TR note 1: C(d2D, hUT) = C'(hUT) g(d2D), hE = 1 with probability 1/(1 + C), else uniform on {12, 15, ...,
    // hUT - 1.5}
    auto note1 = [](double d2D, double hUT, double &p1, int &n)
    {
        const double g = d2D <= 18.0 ? 0.0 : 1.25 * std::pow(d2D / 100.0, 3.0) * std::exp(-d2D / 150.0);
        const double Cp = hUT < 13.0 ? 0.0 : std::pow((hUT - 13.0) / 10.0, 1.5);
        p1 = 1.0 / (1.0 + Cp * g);
        n = 0;
        for (double h = 12.0; h <= hUT - 1.5 + 1e-9; h += 3.0)
            ++n;
        return Cp * g;
    };

    const double hBS = 25.0, fgh = 0.05, d2D = 400.0;
    const arma::uword K = 10000;
    const double cand[4] = {1.0, 12.0, 15.0, 18.0};

    for (double hUT : {20.0, 13.2, 10.0})
    {
        INFO("hUT " << hUT);
        arma::mat tx = {{0.0, 0.0, hBS}}, rx(K, 3);
        arma::vec u(K);
        for (arma::uword k = 0; k < K; ++k)
            rx(k, 0) = d2D, rx(k, 1) = 0.0, rx(k, 2) = hUT, u(k) = (k + 0.5) / K;
        const arma::vec fc = {fgh * 1e9};

        double tr_c[4];
        for (int i = 0; i < 4; ++i)
            tr_c[i] = tr_uma_los(hBS, hUT, d2D, fgh, cand[i]);
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j)
                REQUIRE(std::abs(tr_c[i] - tr_c[j]) > 0.01);

        arma::mat pl_los, pl_copy, pl_nlos, pl_fixed;
        quadriga_lib::path_loss(m_los, p_los, sp_los, fc, tx, rx, pl_los, nullptr, nullptr, &u);
        quadriga_lib::path_loss(m_nlos, p_copy, sp_nlos, fc, tx, rx, pl_copy, nullptr, nullptr, &u);
        quadriga_lib::path_loss(m_nlos, p_nlos, sp_nlos, fc, tx, rx, pl_nlos, nullptr, nullptr, &u);
        quadriga_lib::path_loss(m_los, p_los, sp_los, fc, tx, rx, pl_fixed);

        int count[4] = {0, 0, 0, 0}, n_bad = 0, n_bad_copy = 0, n_bad_nlos = 0, n_bad_fixed = 0;
        const double d3 = std::sqrt(d2D * d2D + (hBS - hUT) * (hBS - hUT));
        const double pln = tr_uma_nlos_prime(hUT, d3, fgh);
        for (arma::uword k = 0; k < K; ++k)
        {
            int best = 0;
            for (int i = 1; i < 4; ++i)
                if (std::abs(pl_los(k, 0) - tr_c[i]) < std::abs(pl_los(k, 0) - tr_c[best]))
                    best = i;
            n_bad += std::abs(pl_los(k, 0) - tr_c[best]) > tol ? 1 : 0;
            n_bad_copy += std::abs(pl_copy(k, 0) - tr_c[best]) > tol ? 1 : 0;
            n_bad_nlos += std::abs(pl_nlos(k, 0) - std::max(pln, tr_c[best])) > tol ? 1 : 0;
            n_bad_fixed += std::abs(pl_fixed(k, 0) - tr_c[0]) > tol ? 1 : 0;
            ++count[best];
        }
        double p1 = 1.0;
        int n = 0;
        const double C = note1(d2D, hUT, p1, n);
        QD_DIAG("\n# diag: path_loss random hE, hUT %g m, d2D %g m, C %.5f, p1 %.5f, n %d\n", hUT, d2D, C, p1, n);
        QD_DIAG("hE,count,expected,pl_TR\n");
        for (int i = 0; i < 4; ++i)
            QD_DIAG("%g,%d,%.2f,%.6f\n", cand[i], count[i], i == 0 ? K * p1 : (n > 0 && hUT >= 13.0 ? K * (1.0 - p1) / n : 0.0),
                    tr_c[i]);
        CHECK(n_bad == 0);
        CHECK(n_bad_copy == 0);
        CHECK(n_bad_nlos == 0);
        CHECK(n_bad_fixed == 0); // rnd_he omitted: the fixed hE = 1
        if (hUT == 20.0)
        {
            REQUIRE(n == 3);
            CHECK(std::abs(count[0] - K / (1.0 + C)) <= 1.0);
            for (int i = 1; i < 4; ++i)
                CHECK(std::abs(count[i] - K * (1.0 - p1) / 3.0) <= 2.0);
        }
        else
            CHECK(count[0] == (int)K); // hUT 13.2 (empty set) and hUT 10 (C = 0): all hE = 1
    }

    // TR UT heights of floors 7 and 8: hE = hUT - 1.5 (h'UT = 1.5 m), no 1.1 hE clamp of the UT height
    QD_DIAG("\n# diag: path_loss random hE at hUT - 1.5\n");
    QD_DIAG("hUT,d2D,hE,u,pl_los,TR_los,pl_nlos,TR_nlos\n");
    for (double hUT : {19.5, 22.5})
        for (double d : {200.0, 1000.0})
        {
            double p1 = 1.0;
            int n = 0;
            note1(d, hUT, p1, n);
            REQUIRE(n > 0);
            REQUIRE(p1 < 1.0);
            const double hE = 12.0 + 3.0 * (n - 1);
            REQUIRE(hE == hUT - 1.5);
            const arma::vec u = {p1 + (n - 0.5) / n * (1.0 - p1)};
            const arma::mat tx = {{0.0, 0.0, hBS}}, rx = {{d, 0.0, hUT}};
            const arma::vec fc = {3.5e9};
            arma::mat a, b, c;
            quadriga_lib::path_loss(m_los, p_los, sp_los, fc, tx, rx, a, nullptr, nullptr, &u);
            quadriga_lib::path_loss(m_nlos, p_copy, sp_nlos, fc, tx, rx, b, nullptr, nullptr, &u);
            quadriga_lib::path_loss(m_nlos, p_nlos, sp_nlos, fc, tx, rx, c, nullptr, nullptr, &u);
            const double los = tr_uma_los(hBS, hUT, d, 3.5, hE);
            const double d3 = std::sqrt(d * d + (hBS - hUT) * (hBS - hUT));
            const double nl = std::max(tr_uma_nlos_prime(hUT, d3, 3.5), los);
            QD_DIAG("%g,%g,%g,%.6f,%.6f,%.6f,%.6f,%.6f\n", hUT, d, hE, u(0), a(0, 0), los, c(0, 0), nl);
            INFO("hUT " << hUT << ", d2D " << d);
            CHECK(std::abs(a(0, 0) - los) <= tol);
            CHECK(std::abs(b(0, 0) - los) <= tol);
            CHECK(std::abs(c(0, 0) - nl) <= tol);
        }

    // u = 1 where p1 = 1/(1 + C) rounds to 1 (C < 1.1e-16 at 10 km, hUT 20 m): hE = 1, as for u = 0
    {
        const arma::mat tx = {{0.0, 0.0, hBS}}, rx = {{4000.0, 0.0, 20.0}, {10000.0, 0.0, 20.0}};
        const arma::vec fc = {3.5e9}, u0 = {0.0, 0.0}, u1 = {1.0, 1.0};
        arma::mat a, b;
        quadriga_lib::path_loss(m_los, p_los, sp_los, fc, tx, rx, a, nullptr, nullptr, &u0);
        quadriga_lib::path_loss(m_los, p_los, sp_los, fc, tx, rx, b, nullptr, nullptr, &u1);
        QD_DIAG("\n# diag: path_loss random hE with u = 1: 4 km %.4f dB (u = 0: %.4f), 10 km %.4f dB (u = 0: %.4f)\n",
                b(0, 0), a(0, 0), b(1, 0), a(1, 0));
        CHECK(b.is_finite());
        CHECK(b(0, 0) != a(0, 0)); // 4 km: C = 1.2e-7, u = 1 selects the largest hE (18 m)
        CHECK(b(1, 0) == a(1, 0));
    }
}

TEST_CASE("path_loss - legacy models match the get_pl formulas")
{
    const arma::vec fc = {0.8e9, 2.4e9, 28.0e9};
    const scen_t sp = {{"SF_sigma", 5.5}};
    const double sf_def = 5.5;
    const double tol = 1e-10;

    // Runs the model and compares every link and frequency with 'ref' (pl and sf); returns the number of
    // (link, frequency) pairs per value of the region counter that 'ref' reports
    auto run = [&](const std::string &model, const plpar_t &p, const arma::vec &f,
                   const std::function<double(double, double, double, double, double, double &, int &)> &ref)
    {
        arma::mat pl, sf;
        quadriga_lib::path_loss(model, p, sp, f, mdl_tx, mdl_rx, pl, &sf);
        REQUIRE(pl.n_rows == 4);
        REQUIRE(pl.n_cols == f.n_elem);
        std::vector<int> regions(4, 0);
        for (arma::uword l = 0; l < 4; ++l)
        {
            double d2, d3, hBS, hUT;
            link_geo(mdl_tx, mdl_rx, l, d2, d3, hBS, hUT);
            for (arma::uword k = 0; k < f.n_elem; ++k)
            {
                double s = sf_def;
                int region = 0;
                const double r = ref(f(k) / 1e9, d3, d2, hBS, hUT, s, region);
                INFO("model " << model << ", link " << l << ", f " << f(k) / 1e9 << " GHz");
                CHECK(std::abs(pl(l, k) - r) <= tol);
                CHECK(sf(l, k) == s);
                regions[std::min(region, 3)]++;
            }
        }
        return regions;
    };

    SECTION("constant")
    {
        run("constant", {{"A", {87.5}}}, fc, [](double, double, double, double, double, double &, int &)
            { return 87.5; });
    }
    SECTION("logdist with and without SF")
    {
        const plpar_t p = {{"A", {21.5}}, {"B", {31.84}}, {"C", {19.0}}};
        run("logdist", p, fc, [](double f, double d3, double, double, double, double &, int &)
            { return 21.5 * lg(d3) + 31.84 + 19.0 * lg(f); });
        plpar_t q = p;
        q["SF"] = {4.5};
        run("logdist", q, fc, [](double f, double d3, double, double, double, double &s, int &)
            { s = 4.5; return 21.5 * lg(d3) + 31.84 + 19.0 * lg(f); });
    }
    SECTION("logdist_simple")
    {
        run("logdist_simple", {{"A", {17.3}}, {"B", {32.4}}}, fc,
            [](double, double d3, double, double, double, double &, int &)
            { return 17.3 * lg(d3) + 32.4; });
    }
    SECTION("winner_los")
    {
        // Breakpoint from the heights of each link; C2 beyond it (C1 != C2)
        const plpar_t p = {{"A1", {22.7}}, {"A2", {40.0}}, {"B1", {41.0}}, {"B2", {9.45}}, {"C1", {20.0}},
                           {"C2", {17.3}}, {"D1", {0.5}}, {"D2", {-17.3}}, {"E1", {-0.4}}, {"E2", {-17.3}},
                           {"F1", {0.02}}, {"F2", {0.02}}, {"sig1", {3.0}}, {"sig2", {6.0}}};
        auto ref = [](double f, double, double d2, double hBS, double hUT, double &s, int &region)
        {
            const double hb = std::max(hBS, 1.5), hm = std::max(hUT, 1.5);
            const double G = 41.0 + 20.0 * lg(f) + 0.5 * lg(hb) - 0.4 * lg(hm);
            const double H = 9.45 + 17.3 * lg(f) - 17.3 * lg(hb) - 17.3 * lg(hm);
            const double bp = std::pow(10.0, (H - G) / (22.7 - 40.0));
            if (d2 <= bp)
            {
                region = 1, s = 3.0;
                return 22.7 * lg(d2) + 41.0 + 20.0 * lg(f) + 0.5 * lg(hb) - 0.4 * lg(hm) + 0.02 * hm;
            }
            region = 2, s = 6.0;
            return 40.0 * lg(d2) + 9.45 + 17.3 * lg(f) - 17.3 * lg(hb) - 17.3 * lg(hm) + 0.02 * hm;
        };
        const std::vector<int> r = run("winner_los", p, fc, ref);
        CHECK(r[1] > 0);
        CHECK(r[2] > 0);

        // Continuity at the breakpoint (F1 = F2): links just before and after it
        const double f = 2.4, hb = 25.0, hm = 1.5;
        const double G = 41.0 + 20.0 * lg(f) + 0.5 * lg(hb) - 0.4 * lg(hm);
        const double H = 9.45 + 17.3 * lg(f) - 17.3 * lg(hb) - 17.3 * lg(hm);
        const double bp = std::pow(10.0, (H - G) / (22.7 - 40.0));
        const arma::mat tx = {{0.0, 0.0, hb}}, rx = {{bp * (1.0 - 1e-9), 0.0, hm}, {bp * (1.0 + 1e-9), 0.0, hm}};
        arma::mat pl;
        quadriga_lib::path_loss("winner_los", p, sp, arma::vec({f * 1e9}), tx, rx, pl);
        QD_DIAG("\n# diag: path_loss winner_los at the breakpoint %.3f m: %.9f %.9f\n", bp, pl(0, 0), pl(1, 0));
        CHECK(std::abs(pl(0, 0) - pl(1, 0)) < 1e-6);
    }
    SECTION("winner_nlos")
    {
        plpar_t p;
        const double A[3] = {44.9, 43.9, 42.9}, Ah[3] = {-6.55, -6.45, -6.35}, B[3] = {13.47, 14.47, 15.47},
                     C[3] = {26.0, 23.0, 20.0}, D[3] = {5.83, 5.73, 5.63}, E[3] = {-0.6, -0.5, -0.4},
                     F[3] = {0.01, 0.02, 0.03};
        for (int j = 0; j < 3; ++j)
        {
            const std::string s = std::to_string(j + 1);
            p["A" + s] = {A[j]}, p["Ah" + s] = {Ah[j]}, p["B" + s] = {B[j]}, p["C" + s] = {C[j]};
            p["D" + s] = {D[j]}, p["E" + s] = {E[j]}, p["F" + s] = {F[j]};
        }
        auto ref = [&](double f, double, double d2, double hBS, double hUT, double &, int &region)
        {
            const double hb = std::max(hBS, 1.5), hm = std::max(hUT, 1.5);
            const int j = f < 1.5 ? 0 : (f < 2.0 ? 1 : 2);
            region = j + 1;
            return (A[j] + Ah[j] * lg(hb)) * lg(d2) + B[j] + C[j] * lg(f) + D[j] * lg(hb) + E[j] * lg(hm) + F[j] * hm;
        };
        const std::vector<int> r = run("winner_nlos", p, arma::vec({1.0e9, 1.7e9, 2.5e9}), ref);
        CHECK((r[1] == 4 && r[2] == 4 && r[3] == 4));
    }
    SECTION("winner_pathloss with SF")
    {
        run("winner_pathloss", {{"A", {18.7}}, {"B", {46.8}}, {"C", {20.0}}, {"X", {3.0}}, {"SF", {3.0}}}, fc,
            [](double f, double d3, double, double, double, double &s, int &)
            { s = 3.0; return 18.7 * lg(d3) + 46.8 + 20.0 * lg(f / 5.0) + 3.0; });
    }
    SECTION("dual_slope")
    {
        // Defaults: A1 = A, A2 = A1, C = 0, D = 0, hE = 0, SF std of scenpar
        const plpar_t p = {{"A", {25.0}}, {"B", {30.0}}, {"E", {13.33}}};
        run("dual_slope", p, fc, [&](double f, double d3, double d2, double hBS, double hUT, double &s, int &region)
            {
                const double hb = std::max(hBS, 0.1), hm = std::max(hUT, 0.1);
                const double dBP = 13.33 * f * hb * hm, dBP3 = std::sqrt(dBP * dBP + (hb - hm) * (hb - hm));
                (void)s;
                region = d2 <= dBP ? 1 : 2;
                return d2 <= dBP ? 25.0 * lg(d3) + 30.0 : 25.0 * lg(dBP3) + 30.0 + 25.0 * lg(d3 / dBP3); });

        // All keys, hE clamp 1.1 hE (link 1: hUT 1.2 m, link 2: hBS 1.0 m), both sides with hBS != hUT
        const plpar_t q = {{"A1", {22.0}}, {"A2", {40.0}}, {"B", {28.0}}, {"C", {20.0}}, {"D", {0.001}}, {"E", {13.33}},
                           {"hE", {1.2}}, {"sig1", {4.0}}, {"sig2", {6.0}}};
        auto ref = [](double f, double d3, double d2, double hBS, double hUT, double &s, int &region)
        {
            const double hE = 1.2;
            const double hb = std::max(std::max(hBS, 1.1 * hE), 0.1), hm = std::max(std::max(hUT, 1.1 * hE), 0.1);
            const double dBP = 13.33 * f * (hb - hE) * (hm - hE), dBP3 = std::sqrt(dBP * dBP + (hb - hm) * (hb - hm));
            auto PL1 = [&](double x)
            { return 22.0 * lg(x) + 28.0 + 20.0 * lg(f) + 0.001 * x; };
            if (d2 <= dBP)
                return region = 1, s = 4.0, PL1(d3);
            return region = 2, s = 6.0, PL1(dBP3) + 40.0 * lg(d3 / dBP3);
        };
        const std::vector<int> r = run("dual_slope", q, fc, ref);
        CHECK(r[1] > 0);
        CHECK(r[2] > 0);
    }
    SECTION("tripple_slope")
    {
        // hb = max(hBS, 1.1 hE1, 1.1 hE2, 0.1), hm the same; dBP1 = min(dBP1, dBP2); three regions
        auto make_ref = [](double A1, double A2, double A3, double B, double C, double D, double E1, double E2,
                           double hE1, double hE2, bool sig)
        {
            return [=](double f, double d3, double d2, double hBS, double hUT, double &s, int &region)
            {
                const double hb = std::max(std::max(std::max(hBS, 1.1 * hE1), 1.1 * hE2), 0.1);
                const double hm = std::max(std::max(std::max(hUT, 1.1 * hE1), 1.1 * hE2), 0.1);
                double dBP1 = E1 * f * (hb - hE1) * (hm - hE1);
                const double dBP2 = E2 * f * (hb - hE2) * (hm - hE2);
                dBP1 = std::min(dBP1, dBP2);
                const double dBP13 = std::sqrt(dBP1 * dBP1 + (hb - hm) * (hb - hm));
                const double dBP23 = std::sqrt(dBP2 * dBP2 + (hb - hm) * (hb - hm));
                auto PL1 = [&](double x)
                { return A1 * lg(x) + B + C * lg(f) + D * x; };
                const double LBP1 = PL1(dBP13), LBP2 = LBP1 + A2 * lg(dBP23 / dBP13);
                if (d2 <= dBP1)
                    return region = 1, s = sig ? 3.0 : s, PL1(d3);
                if (d2 <= dBP2)
                    return region = 2, s = sig ? 5.0 : s, LBP1 + A2 * lg(d3 / dBP13);
                return region = 3, s = sig ? 7.0 : s, LBP2 + A3 * lg(d3 / dBP23);
            };
        };
        // 'E' for E1, E2 default = E1; hE1 > hE2 gives dBP1 < dBP2
        const plpar_t p = {{"A1", {20.0}}, {"A2", {35.0}}, {"A3", {50.0}}, {"B", {30.0}}, {"C", {20.0}}, {"D", {0.002}},
                           {"E", {4.0}}, {"hE1", {1.0}}, {"hE2", {0.0}}, {"sig1", {3.0}}, {"sig2", {5.0}}, {"sig3", {7.0}}};
        const std::vector<int> r = run("tripple_slope", p, fc, make_ref(20, 35, 50, 30, 20, 0.002, 4, 4, 1, 0, true));
        CHECK((r[1] > 0 && r[2] > 0 && r[3] > 0));

        // Defaults A2 = A1, A3 = A2, C = D = 0; dBP1 > dBP2 (hE1 = 0, hE2 = 1) for most links
        const plpar_t q = {{"A", {21.0}}, {"B", {31.0}}, {"E1", {4.0}}, {"E2", {5.0}}, {"hE2", {1.0}}};
        run("tripple_slope", q, fc, make_ref(21, 21, 21, 31, 0, 0, 4, 5, 0, 1, false));
        const plpar_t q2 = {{"A1", {21.0}}, {"A2", {33.0}}, {"B", {31.0}}, {"E1", {4.0}}, {"E2", {5.0}}, {"hE2", {1.0}}};
        const std::vector<int> r2 = run("tripple_slope", q2, fc, make_ref(21, 33, 33, 31, 0, 0, 4, 5, 0, 1, false));
        CHECK(r2[3] > 0);
    }
    SECTION("nlos")
    {
        // All 13 terms non-zero, G2n default, LOSmax default 1, sig
        const plpar_t p = {{"An", {39.08}}, {"Bn", {13.54}}, {"Cn", {20.0}}, {"Dn", {-0.5}}, {"D1n", {0.3}},
                           {"D2n", {2.0}}, {"D3n", {0.01}}, {"En", {1.2}}, {"E1n", {0.4}}, {"E2n", {0.2}},
                           {"E3n", {-0.6}}, {"Fn", {-1.1}}, {"G1n", {-3.2}}, {"A1", {22.0}}, {"A2", {40.0}},
                           {"B", {28.0}}, {"C", {20.0}}, {"E", {13.33}}, {"hE", {1.0}}, {"sig", {6.5}}};
        auto ref = [](const plpar_t &q)
        {
            return [q](double f, double d3, double d2, double hBS, double hUT, double &s, int &region)
            { return ref_nlos(q, f, d3, d2, hBS, hUT, NAN, s, region); };
        };
        std::vector<int> r = run("nlos", p, fc, ref(p));
        QD_DIAG("\n# diag: path_loss nlos, winning term counts (PLn, LOS, extra)\n%d,%d,%d\n", r[0], r[1], r[2]);
        CHECK(r[0] > 0);
        CHECK(r[1] > 0);

        // G2n given, LOSmax = 0, extra term Ax, Bx, Cx
        plpar_t q = p;
        q["G2n"] = {11.75}, q["LOSmax"] = {0.0}, q["Ax"] = {25.5}, q["Bx"] = {33.0}, q["Cx"] = {20.0};
        q["Bn"] = {5.0};
        r = run("nlos", q, fc, ref(q));
        QD_DIAG("%d,%d,%d\n", r[0], r[1], r[2]);
        CHECK(r[1] == 0);
        CHECK(r[2] > 0);

        // LOSmax = 1 and only Bx: the extra term is active with Ax = Cx = 0
        plpar_t q3 = p;
        q3["LOSmax"] = {1.0}, q3["Bx"] = {100.0};
        r = run("nlos", q3, fc, ref(q3));
        QD_DIAG("%d,%d,%d\n", r[0], r[1], r[2]);
        CHECK(r[2] > 0);
    }
    SECTION("satellite")
    {
        // 600 km height; the last link has the Tx below the horizon (absolute height < 0): +Inf
        const arma::mat tx = {{0.0, 0.0, 600e3}, {0.0, 0.0, 600e3}, {0.0, 0.0, 600e3}, {0.0, 0.0, -1000.0}};
        const arma::mat rx = {{0.0, 0.0, 0.0}, {500e3, 0.0, 0.0}, {2000e3, 0.0, 10.0}, {100.0, 0.0, 0.0}};
        const arma::vec f = {1.0e9, 20.71e9, 33.0e9, 150.0e9};
        auto check = [&](const plpar_t &p, double A, double B, double C, double D, bool usePLa)
        {
            arma::mat pl, sf;
            quadriga_lib::path_loss("satellite", p, sp, f, tx, rx, pl, &sf);
            for (arma::uword l = 0; l < 4; ++l)
            {
                double d2, d3, hBS, hUT;
                link_geo(tx, rx, l, d2, d3, hBS, hUT);
                for (arma::uword k = 0; k < 4; ++k)
                {
                    INFO("link " << l << ", f " << f(k) / 1e9 << " GHz");
                    const double fg = f(k) / 1e9;
                    const double sa = std::min(hBS / d3, 1.0);
                    CHECK(sf(l, k) == sf_def);
                    if (sa <= 0.0)
                    {
                        CHECK((std::isinf(pl(l, k)) && pl(l, k) > 0.0));
                        continue;
                    }
                    const double PLa = usePLa ? zenith_att(fg) / sa : 0.0;
                    CHECK(std::abs(pl(l, k) - (A * lg(d3) + B + C * lg(fg) + D * lg(std::asin(sa)) + PLa)) <= tol);
                }
            }
        };
        CHECK(std::abs(zenith_att(20.71) - 0.4) < 1e-15);
        CHECK(std::abs(zenith_att(33.0) - (0.25 + (33.0 - 32.18) / (33.8 - 32.18) * 0.02)) < 1e-15);
        check({}, 20.0, 32.45, 20.0, 0.0, true);
        check({{"A", {21.0}}, {"B", {32.0}}, {"C", {19.0}}, {"D", {2.5}}, {"usePLa", {0.0}}}, 21.0, 32.0, 19.0, 2.5, false);
        check({{"D", {-1.5}}, {"usePLa", {1.0}}}, 20.0, 32.45, 20.0, -1.5, true);
    }
}

TEST_CASE("path_loss - distances below 0.1 m are clamped")
{
    // Co-located Tx and Rx (d3D = 0), a 3D distance of 0.05 m and a vertical link (d2D = 0, d3D = 5 m)
    const arma::mat tx = {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 8.0}};
    const arma::mat rx = {{1.0, 2.0, 3.0}, {1.03, 2.04, 3.0}, {1.0, 2.0, 3.0}};
    const arma::vec fc = {2.0e9, 30.0e9};
    arma::mat pl;
    quadriga_lib::path_loss("logdist", {{"A", {20.0}}, {"B", {32.45}}, {"C", {20.0}}}, {}, fc, tx, rx, pl);
    for (arma::uword k = 0; k < 2; ++k)
    {
        const double f = fc(k) / 1e9;
        CHECK(close_rel(pl(0, k), 20.0 * lg(0.1) + 32.45 + 20.0 * lg(f), 1e-12));
        CHECK(close_rel(pl(1, k), 20.0 * lg(0.1) + 32.45 + 20.0 * lg(f), 1e-12));
        CHECK(close_rel(pl(2, k), 20.0 * lg(5.0) + 32.45 + 20.0 * lg(f), 1e-12));
    }
    // winner_nlos uses d2D: the vertical link gives lg(0.1)
    plpar_t p;
    for (int j = 1; j <= 3; ++j)
        for (const char *k : {"A", "Ah", "B", "C", "D", "E", "F"})
            p[std::string(k) + std::to_string(j)] = {std::string(k) == "A" ? 40.0 : (std::string(k) == "B" ? 10.0 : 0.0)};
    quadriga_lib::path_loss("winner_nlos", p, {}, fc, tx, rx, pl);
    CHECK(close_rel(pl(2, 0), 40.0 * lg(0.1) + 10.0, 1e-12));
}

TEST_CASE("path_loss - default sf_sigma follows the LSP formula")
{
    // sigma = max(0, max(0, SF_sigma + SF_delta lg(SF_omega + f')) + SF_kappa lg d + SF_tau lg h + SF_beta lg a),
    // d = max(d2D, 1e-4), h = max(hBS, 1e-4), a = atan(h / d), f' = max(f, LSP_fc_min)
    auto ref = [](const scen_t &s, double f, double d, double h)
    {
        auto g = [&](const char *k)
        { auto it = s.find(k); return it == s.end() ? 0.0 : it->second; };
        const double fp = std::max(f, g("LSP_fc_min"));
        const double a = std::atan(h / d);
        const double s0 = std::max(0.0, g("SF_sigma") + g("SF_delta") * lg(g("SF_omega") + fp));
        return std::max(0.0, s0 + g("SF_kappa") * lg(d) + g("SF_tau") * lg(h) + g("SF_beta") * lg(a));
    };
    const plpar_t p = {{"A", {20.0}}, {"B", {30.0}}, {"C", {20.0}}};

    const scen_t s1 = {{"SF_sigma", 6.0}, {"SF_delta", 1.5}, {"SF_omega", 0.5}, {"SF_kappa", 0.3}, {"SF_tau", -0.2},
                       {"SF_beta", 0.1}, {"LSP_fc_min", 6.0}};
    const arma::mat tx = {{0.0, 0.0, 25.0}, {10.0, 0.0, 3.0}, {0.0, 5.0, 40.0}};
    const arma::mat rx = {{100.0, 0.0, 1.5}, {10.0, 30.0, 8.0}, {1500.0, -200.0, 1.5}};
    const arma::vec fc = {3.0e9, 10.0e9};
    arma::mat pl, sf;
    quadriga_lib::path_loss("logdist", p, s1, fc, tx, rx, pl, &sf);
    QD_DIAG("\n# diag: path_loss default sf_sigma\nlink,f,sf,ref\n");
    for (arma::uword l = 0; l < 3; ++l)
    {
        double d2, d3, hBS, hUT;
        link_geo(tx, rx, l, d2, d3, hBS, hUT);
        for (arma::uword k = 0; k < 2; ++k)
        {
            const double r = ref(s1, fc(k) / 1e9, d2, hBS);
            QD_DIAG("%d,%g,%.12f,%.12f\n", (int)l, fc(k) / 1e9, sf(l, k), r);
            CHECK(std::abs(sf(l, k) - r) < 1e-12);
        }
    }

    // Elevation term: 1.4 + lg(a) is -0.36 at 1 deg (limited at 0) and 0.642 at 10 deg
    const scen_t s2 = {{"SF_sigma", 1.4}, {"SF_beta", 1.0}};
    const double h = 10.0;
    const arma::mat tx2 = {{0.0, 0.0, h}};
    const arma::mat rx2 = {{h / std::tan(pi / 180.0), 0.0, 1.5}, {h / std::tan(10.0 * pi / 180.0), 0.0, 1.5}};
    quadriga_lib::path_loss("logdist", p, s2, fc, tx2, rx2, pl, &sf);
    for (arma::uword k = 0; k < 2; ++k)
    {
        CHECK(sf(0, k) == 0.0);
        CHECK(std::abs(sf(1, k) - (1.4 + lg(10.0 * pi / 180.0))) < 1e-12);
    }
    CHECK(1.4 + lg(pi / 180.0) < -0.35);
}

TEST_CASE("path_loss - an empty model gives zero loss")
{
    const scen_t sp = conf("UMa_NLOS");
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {-300.0, 20.0, 7.5}};
    const arma::vec fc = {3.5e9, 28.0e9};
    arma::mat pl(2, 2, arma::fill::value(-7.0)), sf;
    quadriga_lib::path_loss("", {{"A", {1.0, 2.0}}}, sp, fc, tx, rx, pl, &sf);
    CHECK(arma::all(arma::vectorise(pl) == 0.0));
    CHECK(arma::all(arma::vectorise(sf) == 6.0)); // SF_sigma of the file
}

TEST_CASE("path_loss - optional outputs and pre-sized outputs")
{
    plpar_t plpar;
    std::string model;
    const scen_t sp = conf("UMa_NLOS", &plpar, &model);
    const arma::mat tx1 = {{5.0, -3.0, 25.0}};
    const arma::mat rx = {{100.0, 0.0, 1.5}, {-300.0, 20.0, 19.5}, {40.0, 400.0, 22.5}, {5.0, -3.0, 1.5}};
    const arma::mat txn = arma::repmat(tx1, 4, 1);
    const arma::vec fc = {3.5e9, 28.0e9, 0.8e9};
    const arma::vec d2d_in = {0.0, 10.0, 3.0, 2.0}, u = {0.1, 0.9, 0.6, 0.5};

    arma::mat pl0, sf0;
    quadriga_lib::path_loss(model, plpar, sp, fc, tx1, rx, pl0, &sf0, &d2d_in, &u);
    REQUIRE(pl0.n_rows == 4);
    REQUIRE(pl0.n_cols == 3);
    REQUIRE(sf0.n_rows == 4);
    REQUIRE(sf0.n_cols == 3);

    arma::mat pl1;
    quadriga_lib::path_loss(model, plpar, sp, fc, tx1, rx, pl1, nullptr, &d2d_in, &u);
    CHECK(arma::all(arma::vectorise(pl1 == pl0)));

    arma::mat pl2(4, 3, arma::fill::value(-7.0)), sf2(4, 3, arma::fill::value(-7.0));
    quadriga_lib::path_loss(model, plpar, sp, fc, tx1, rx, pl2, &sf2, &d2d_in, &u);
    CHECK(arma::all(arma::vectorise(pl2 == pl0)));
    CHECK(arma::all(arma::vectorise(sf2 == sf0)));

    arma::mat pl3(2, 7, arma::fill::value(-7.0)), sf3(1, 1);
    quadriga_lib::path_loss(model, plpar, sp, fc, txn, rx, pl3, &sf3, &d2d_in, &u); // repeated Tx rows, wrong sizes
    CHECK(arma::all(arma::vectorise(pl3 == pl0)));
    CHECK(arma::all(arma::vectorise(sf3 == sf0)));

    arma::mat pl4, sf4; // repeated call
    quadriga_lib::path_loss(model, plpar, sp, fc, tx1, rx, pl4, &sf4, &d2d_in, &u);
    CHECK(arma::all(arma::vectorise(pl4 == pl0)));
    CHECK(arma::all(arma::vectorise(sf4 == sf0)));

    // n_link = 0, with 1 or 0 Tx rows
    arma::mat pl5(3, 3, arma::fill::value(1.0)), sf5(3, 3);
    const arma::mat rx0(0, 3);
    const arma::vec e0;
    quadriga_lib::path_loss(model, plpar, sp, fc, tx1, rx0, pl5, &sf5, &e0, &e0);
    CHECK((pl5.n_rows == 0 && pl5.n_cols == 3 && sf5.n_rows == 0 && sf5.n_cols == 3));
    quadriga_lib::path_loss(model, plpar, sp, fc, arma::mat(0, 3), rx0, pl5, &sf5);
    CHECK((pl5.n_rows == 0 && pl5.n_cols == 3));
}

TEST_CASE("path_loss - input validation")
{
    plpar_t plpar;
    std::string model;
    const scen_t sp = conf("UMa_NLOS", &plpar, &model);
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {-150.0, 80.0, 1.5}, {30.0, -400.0, 7.5}};
    const arma::vec fc = {3.5e9, 28.0e9};
    const arma::vec d2d_in = {0.0, 5.0, 10.0}, u = {0.2, 0.5, 0.8};
    arma::mat pl;

    auto call = [&](const std::string &m, const plpar_t &p)
    { quadriga_lib::path_loss(m, p, sp, fc, tx, rx, pl); };

    // Complete parameter sets of each model
    plpar_t winner_nlos;
    for (int j = 1; j <= 3; ++j)
        for (const char *k : {"A", "Ah", "B", "C", "D", "E", "F"})
            winner_nlos[std::string(k) + std::to_string(j)] = {std::string(k) == "A" ? 40.0 : 1.0};
    const std::vector<std::pair<std::string, plpar_t>> full = {
        {"constant", {{"A", {80.0}}}},
        {"logdist", {{"A", {20.0}}, {"B", {30.0}}, {"C", {20.0}}}},
        {"logdist_simple", {{"A", {20.0}}, {"B", {30.0}}}},
        {"winner_los", {{"A1", {22.7}}, {"A2", {40.0}}, {"B1", {41.0}}, {"B2", {9.45}}, {"C1", {20.0}}, {"C2", {17.3}}, {"D1", {0.5}}, {"D2", {-17.3}}, {"E1", {-0.4}}, {"E2", {-17.3}}, {"F1", {0.0}}, {"F2", {0.0}}, {"sig1", {3.0}}, {"sig2", {6.0}}}},
        {"winner_nlos", winner_nlos},
        {"winner_pathloss", {{"A", {18.7}}, {"B", {46.8}}, {"C", {20.0}}, {"X", {3.0}}}},
        {"dual_slope", {{"A1", {22.0}}, {"B", {28.0}}, {"E", {13.33}}}},
        {"tripple_slope", {{"A1", {22.0}}, {"B", {28.0}}, {"E1", {13.33}}}},
        {"nlos", {{"An", {39.08}}, {"Bn", {13.54}}, {"A1", {22.0}}, {"B", {28.0}}, {"E", {13.33}}}}};

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl));
        arma::mat sf;
        CHECK_NOTHROW(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, &sf, &d2d_in, &u));
        CHECK_NOTHROW(quadriga_lib::path_loss(model, plpar, {}, fc, tx, rx, pl)); // empty scenpar: defaults
        for (const auto &m : full)
            CHECK_NOTHROW(call(m.first, m.second));
        CHECK_NOTHROW(call("satellite", {}));
        CHECK_NOTHROW(call("", {}));
        CHECK_NOTHROW(call("nlos", {{"An", {39.08}}, {"Bn", {13.54}}, {"LOSmax", {0.0}}})); // no LOS keys
        CHECK_NOTHROW(call("logdist", {{"A", {20.0}}, {"B", {30.0}}, {"C", {20.0}}, {"Xyz", {1.0, 2.0}}}));
        CHECK_NOTHROW(call("dual_slope", {{"A", {22.0}}, {"B", {28.0}}, {"E", {13.33}}})); // A for A1
        CHECK_NOTHROW(call("tripple_slope", {{"A", {22.0}}, {"B", {28.0}}, {"E", {13.33}}}));
    }
    SECTION("unknown model")
    {
        CHECK_THROWS_MATCHES(call("foo", plpar), std::invalid_argument,
                             Message("Path-loss model 'foo' is not supported."));
        CHECK_THROWS_MATCHES(call("Logdist", plpar), std::invalid_argument,
                             Message("Path-loss model 'Logdist' is not supported."));
    }
    SECTION("missing required key")
    {
        for (const auto &m : full)
            for (const auto &e : m.second)
            {
                plpar_t p = m.second;
                p.erase(e.first);
                INFO("model " << m.first << ", key " << e.first);
                CHECK_THROWS_MATCHES(call(m.first, p), std::invalid_argument,
                                     Message("Path-loss model '" + m.first + "' requires the parameter '" + e.first + "'."));
            }
    }
    SECTION("one value per parameter")
    {
        CHECK_THROWS_MATCHES(call("logdist", {{"A", {20.0}}, {"B", {}}, {"C", {20.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'B' must have one value."));
        CHECK_THROWS_MATCHES(call("logdist", {{"A", {20.0}}, {"B", {30.0, 31.0}}, {"C", {20.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'B' must have one value."));
        CHECK_THROWS_MATCHES(call("logdist", {{"A", {20.0}}, {"B", {30.0}}, {"C", {20.0}}, {"SF", {1.0, 2.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'SF' must have one value."));
        CHECK_THROWS_MATCHES(call("satellite", {{"usePLa", {}}}), std::invalid_argument,
                             Message("Path-loss parameter 'usePLa' must have one value."));
        CHECK_THROWS_MATCHES(call("nlos", {{"An", {39.08}}, {"Bn", {13.54}}, {"LOSmax", {0.0, 1.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'LOSmax' must have one value."));
    }
    SECTION("breakpoint factors")
    {
        CHECK_THROWS_MATCHES(call("dual_slope", {{"A1", {22.0}}, {"B", {28.0}}, {"E", {0.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'E' must be > 0."));
        CHECK_THROWS_MATCHES(call("nlos", {{"An", {39.0}}, {"Bn", {13.5}}, {"A1", {22.0}}, {"B", {28.0}}, {"E", {-1.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'E' must be > 0."));
        CHECK_NOTHROW(call("nlos", {{"An", {39.0}}, {"Bn", {13.5}}, {"LOSmax", {0.0}}, {"E", {-1.0}}}));
        CHECK_THROWS_MATCHES(call("tripple_slope", {{"A1", {22.0}}, {"B", {28.0}}, {"E1", {0.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'E1' must be > 0."));
        CHECK_THROWS_MATCHES(call("tripple_slope", {{"A1", {22.0}}, {"B", {28.0}}, {"E1", {4.0}}, {"E2", {-1.0}}}), std::invalid_argument,
                             Message("Path-loss parameter 'E2' must be > 0."));
        plpar_t p = full[3].second;
        p["A2"] = p["A1"];
        CHECK_THROWS_MATCHES(call("winner_los", p), std::invalid_argument,
                             Message("Path-loss parameters 'A1' and 'A2' must differ for model 'winner_los'."));
    }
    SECTION("d2d_in")
    {
        arma::vec bad = {0.0, -1.0, 2.0};
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, nullptr, &bad), std::invalid_argument,
                             Message("Input 'd2d_in' must be finite and >= 0."));
        bad(1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, nullptr, &bad), std::invalid_argument,
                             Message("Input 'd2d_in' must be finite and >= 0."));
        const arma::vec bad2 = {1.0, 2.0};
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, nullptr, &bad2), std::invalid_argument,
                             Message("Number of elements in 'd2d_in' does not match number of rows in 'rx_pos'."));
    }
    SECTION("rnd_he")
    {
        arma::vec bad = {0.0, 1.5, 1.0};
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, nullptr, nullptr, &bad), std::invalid_argument,
                             Message("Input 'rnd_he' must be in the range from 0 to 1."));
        bad(1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, nullptr, nullptr, &bad), std::invalid_argument,
                             Message("Input 'rnd_he' must be in the range from 0 to 1."));
        const arma::vec bad2 = {0.5, 0.5, 0.5, 0.5};
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, rx, pl, nullptr, nullptr, &bad2), std::invalid_argument,
                             Message("Number of elements in 'rnd_he' does not match number of rows in 'rx_pos'."));
    }
    SECTION("fc")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, arma::vec(), tx, rx, pl), std::invalid_argument,
                             Message("Input 'fc' cannot be empty."));
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, arma::vec({3.5e9, 0.0}), tx, rx, pl), std::invalid_argument,
                             Message("Input 'fc' must be finite and > 0."));
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, arma::vec({NAN}), tx, rx, pl), std::invalid_argument,
                             Message("Input 'fc' must be finite and > 0."));
    }
    SECTION("positions")
    {
        arma::mat bad(3, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, bad, pl), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 columns."));
        bad = rx, bad(1, 2) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, tx, bad, pl), std::invalid_argument,
                             Message("Input 'rx_pos' must be finite."));
        bad.zeros(1, 2);
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, bad, rx, pl), std::invalid_argument,
                             Message("Input 'tx_pos' must have 3 columns."));
        bad.zeros(2, 3);
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, bad, rx, pl), std::invalid_argument,
                             Message("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."));
        bad = tx, bad(0, 0) = INFINITY;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, sp, fc, bad, rx, pl), std::invalid_argument,
                             Message("Input 'tx_pos' must be finite."));
    }
    SECTION("scenpar")
    {
        scen_t bad = sp;
        bad["SF_lambda"] = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, bad, fc, tx, rx, pl), std::invalid_argument,
                             Message("Value of 'SF_lambda' must be finite and >= 0."));
        bad = sp;
        bad["SF_sigmaX"] = 1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, bad, fc, tx, rx, pl), std::invalid_argument,
                             Message("Unknown key 'SF_sigmaX' in 'scenpar'."));
    }
    SECTION("SF frequency term of the default sf_sigma")
    {
        // Checked only when sf_sigma is requested; f' = max(f, LSP_fc_min) = 3.5 GHz gives SF_omega + f' = 0
        scen_t bad = sp;
        bad["SF_omega"] = -3.5, bad["SF_delta"] = 1.0, bad["LSP_fc_min"] = 0.0;
        arma::mat sf;
        CHECK_THROWS_MATCHES(quadriga_lib::path_loss(model, plpar, bad, fc, tx, rx, pl, &sf), std::invalid_argument,
                             Message("Scenario parameter 'SF_omega' plus the frequency in GHz must be > 0 where 'SF_gamma' or 'SF_delta' is not 0."));
        CHECK_NOTHROW(quadriga_lib::path_loss(model, plpar, bad, fc, tx, rx, pl));
    }
    SECTION("outputs are not touched on error")
    {
        arma::mat out(3, 2, arma::fill::value(-7.0)), sf(3, 2, arma::fill::value(-7.0));
        CHECK_THROWS(quadriga_lib::path_loss("foo", plpar, sp, fc, tx, rx, out, &sf));
        scen_t bad = sp;
        bad["SF_lambda"] = -1.0;
        CHECK_THROWS(quadriga_lib::path_loss(model, plpar, bad, fc, tx, rx, out, &sf));
        CHECK(arma::all(arma::vectorise(out) == -7.0));
        CHECK(arma::all(arma::vectorise(sf) == -7.0));
    }
}
