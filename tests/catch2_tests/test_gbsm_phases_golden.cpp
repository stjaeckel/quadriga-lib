// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "*the previous version"

// Golden values of the "lsf" and "cluster" sets of quadriga_lib::gbsm_init_phases and quadriga_lib::gbsm_draw: values
// of quadriga-lib before the merge of the random-set functions, so that stored seeds and phase cubes keep their
// results.
// Values: 8 phases per set (rows, columns and both slices spread) for the seeds 7 and 123456789, sets "lsf" without
// and with bias removal over 5 positions and "cluster" -1, 0, 3, scenario 3GPP_38.901_v19_UMi_NLOS (N = 19), preset
// Comb256; 6 values per set at 3 links for seed 7 in single mobility, dual mobility and with floor_id = {0, 2, 0};
// printed with 17 significant digits.
// Tolerances: phases exact, except the bias-removed "lsf" phases 1e-12 rad (sos_remove_bias uses cos, sin, atan2 and
// sums that depend on libm and FP contraction: up to 1.95e-14 rad between an FMA and a non-FMA build); draws 2e-5
// (the AVX2 kernel of sos_val is within 4e-7·sqrt(2·256) = 9.05e-6 of the double evaluation, so values recorded on
// one kernel and checked on the other differ by at most 9.05e-6 in single and 1.28e-5 in dual mobility).
// Conventions: positions are rows {x, y, z} in meters; the 3 links are the first 3 of the 5 bias-removal positions.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
    typedef std::unordered_map<std::string, double> spmap;

    spmap umi_nlos()
    {
        spmap sp;
        quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_UMi_NLOS.conf", sp);
        return sp;
    }

    const arma::mat RX5 = {{10.0, 20.0, 1.5}, {50.0, -30.0, 1.5}, {-40.0, 70.0, 4.5}, {120.0, 15.0, 1.5}, {-80.0, -60.0, 7.5}};
    const arma::mat TX5 = {{5.0, -8.0, 1.5}, {60.0, 40.0, 1.5}, {-30.0, 90.0, 1.5}, {100.0, -20.0, 1.5}, {-70.0, -90.0, 1.5}};
    const arma::mat TX1 = {{0.0, 0.0, 25.0}};

    // One recorded set: set type, set id, bias removal over RX5 (with TX5 as tx_pos, single mobility)
    struct golden_set
    {
        const char *name, *type;
        arma::sword set_id;
        bool bias;
        double phase[2][8];  // seeds 7 and 123456789
        double draw[3][6];   // seed 7: single, dual, floor
    };

    // Values of quadriga-lib before the merge of the random-set functions
    const golden_set GOLDEN[] = {
        {"lsf_nb", "lsf", -1, false,
         {{1.5983500054156474, -0.98231016785905501, -2.221985480450293, 0.046004503373173286,
           -1.5878197801134506, -2.9094886439789875, -1.3408142557581384, -1.8521286352402024},
          {-0.94956708543161272, 0.60604480818673812, 0.077614149703936056, 2.2891155690234926,
           -1.9309316448140557, 1.5740321138817477, 1.875724882345768, -1.362242925302275}},
         {{-0.35921966470777988, 0.031272325474718747, -0.91455934196710587, 1.3803290305659175,
           -0.17262288741767406, 0.013348627835512161},
          {-0.73288720619230063, 0.27789935051558151, -0.46924174099123084, 2.0625153766189364,
           0.86062610463017175, -0.038628855952838756},
          {-0.35921966470777988, 0.78812760190925635, -0.91455934196710587, 1.3803290305659175,
           0.69287591427564621, 0.013348627835512161}}},
        {"lsf_b", "lsf", -1, true,
         {{2.9015125004060947, -0.98231016785905501, -2.0049562546179236, -0.86109113023604245,
           -2.6883462040039738, 2.3845640072740135, -1.3408142557581384, -0.64948853098925441},
          {-0.49183792840141716, 0.60604480818673812, 0.56915101925373313, 2.8304869093889837,
           -1.9826628701411275, 1.1081141007167272, 1.875724882345768, -2.455133302608639}},
         {{1.0294581274501979, 0.031272325474718747, -0.14501047879457474, 1.1894445475190878,
           0.043483270332217216, 0.95977003872394562},
          {0.97136674456391159, 0.27789935051558151, 0.074356191907700001, 1.0898940326239996,
           0.40514684896917424, 1.1599641560884055},
          {1.0294581274501979, 0.78812760190925635, -0.14501047879457474, 1.1894445475190878,
           0.18335978798131691, 0.95977003872394562}}},
        {"cl_m1", "cluster", -1, false,
         {{-1.2214847790543986, 0.52259673617529623, -2.887697967253442, 1.9515958628054251,
           2.6595441737750063, 1.1215228098321697, 2.9967303788693052, -0.048830323963826},
          {0.85490603472441407, -0.20893521335471607, 1.5623554597159819, -0.16271715394034125,
           -1.0228495032459306, -0.30525963449492016, 1.6060232206980245, -2.7938223983714514}},
         {{0.71098893270509478, 0.16378326807171106, 1.2300995551049709, -1.5366132818162441,
           0.11250755563052793, 1.7891268259845674},
          {0.81848480787102407, 0.72393222688548731, 1.257198072911121, -1.7956255779487051,
           0.097812635181659488, 0.39736868367638306},
          {0.71098893270509478, -0.64889253210276365, 1.2300995551049709, -1.5366132818162441,
           0.19858729978931747, 1.7891268259845674}}},
        {"cl_0", "cluster", 0, false,
         {{-2.7600551523726571, -1.9082887289715809, 0.79438160248888179, 0.040050986235148821,
           -1.8646427984154856, 2.0033249095514867, -1.135608372628762, 1.9922643003250042},
          {1.3002904385944971, 2.0942503788969611, 2.4007845766297069, -1.895296003037648,
           -2.6026840462131635, -1.5914973695796928, 0.92141095437630671, -0.87159094095393763}},
         {{-1.5537299331626855, -0.44474236667156219, 0.013588963076472282, -0.5706611899076961,
           -0.15797752141952515, -1.0599636890692636},
          {-1.436615331259584, -0.43684112266050468, 0.098660623728577659, 0.81987996123962004,
           0.33151902384231396, -1.3687503717743639},
          {-1.5537299331626855, -0.63292966783046722, 0.013588963076472282, -0.5706611899076961,
           -1.8394738342612982, -1.0599636890692636}}},
        {"cl_3", "cluster", 3, false,
         {{0.52234245904311549, 0.51224154365299235, -2.3966172144478075, -1.7757017297473998,
           0.25601450777933993, -0.82711366562408983, -2.3439553575700329, -0.9140807607229281},
          {-1.9512640053452313, 2.603698188285251, 2.0007841689342758, 1.1515029555257392,
           1.4012439171530182, -1.6393987691920593, -0.21604738343532051, -1.1344177039073635}},
         {{-1.125470545142889, -0.65972386300563812, -1.4063736950047314, -0.69420585967600346,
           0.30209166556596756, -0.80981711507774889},
          {-0.89356515010087356, -1.4710885543622785, -0.96168293446075881, -1.0713658004930278,
           1.09327575790006, -0.028611454937830835},
          {-1.125470545142889, 0.55957160051912069, -1.4063736950047314, -0.69420585967600346,
           0.77435305714607239, -0.80981711507774889}}},
    };

    // Sampled positions: phases (row, column, slice) and draws (link, column) from the number of generators n
    void phase_index(arma::uword n, int i, arma::uword &r, arma::uword &c, arma::uword &s)
    {
        const arma::uword pr[8] = {0, 255, 17, 128, 5, 200, 64, 255};
        const arma::uword pc[8] = {0, n - 1, n / 3, n / 2, n / 4, 2 * n / 3, n - 2, 1};
        r = pr[i], c = pc[i], s = (arma::uword)(i / 4);
    }
    void draw_index(arma::uword n, int i, arma::uword &l, arma::uword &c)
    {
        const arma::uword dl[6] = {0, 1, 2, 0, 1, 2};
        const arma::uword dc[6] = {0, n - 1, n / 2, n / 3, 1, 2 * n / 3};
        l = dl[i], c = dc[i];
    }

    void init(const spmap &sp, const golden_set &g, arma::sword seed, arma::cube &ph)
    {
        if (g.bias)
            quadriga_lib::gbsm_init_phases(sp, g.type, ph, seed, g.set_id, "Comb256", &TX5, &RX5);
        else
            quadriga_lib::gbsm_init_phases(sp, g.type, ph, seed, g.set_id);
    }

    void check_phases(bool lsf)
    {
        const spmap sp = umi_nlos();
        const arma::sword seeds[2] = {7, 123456789};
        QD_DIAG("\n# diag: gbsm_init_phases against the previous version, max abs difference\nset,seed,max_diff\n");
        for (const auto &g : GOLDEN)
            for (int is = 0; is < 2; ++is)
            {
                if ((std::string(g.type) == "lsf") != lsf)
                    continue;
                arma::cube ph;
                init(sp, g, seeds[is], ph);
                REQUIRE(ph.n_rows == 256);
                REQUIRE(ph.n_slices == 2);
                const double tol = g.bias ? 1e-12 : 0.0; // bias removal: libm and FP contraction
                double max_diff = 0.0;
                for (int i = 0; i < 8; ++i)
                {
                    arma::uword r, c, s;
                    phase_index(ph.n_cols, i, r, c, s);
                    INFO("set " << g.name << ", seed " << seeds[is] << ", value " << i);
                    CHECK(std::abs(ph(r, c, s) - g.phase[is][i]) <= tol);
                    max_diff = std::max(max_diff, std::abs(ph(r, c, s) - g.phase[is][i]));
                }
                QD_DIAG("%s,%lld,%.3e\n", g.name, (long long)seeds[is], max_diff);
            }
    }

    void check_draws(bool lsf)
    {
        const spmap sp = umi_nlos();
        const arma::mat rx3 = RX5.rows(0, 2), tx3 = TX5.rows(0, 2);
        const arma::uvec fl = {0, 2, 0};
        QD_DIAG("\n# diag: gbsm_draw against the previous version, max abs difference\nset,mode,max_diff\n");
        for (const auto &g : GOLDEN)
        {
            if ((std::string(g.type) == "lsf") != lsf)
                continue;
            arma::cube ph;
            init(sp, g, 7, ph);
            arma::mat rnd[3];
            quadriga_lib::gbsm_draw(sp, g.type, ph, TX1, rx3, rnd[0], g.set_id);
            quadriga_lib::gbsm_draw(sp, g.type, ph, tx3, rx3, rnd[1], g.set_id, "Comb256", nullptr, nullptr, true);
            quadriga_lib::gbsm_draw(sp, g.type, ph, TX1, rx3, rnd[2], g.set_id, "Comb256", &fl);
            const char *mode[3] = {"single", "dual", "floor"};
            for (int m = 0; m < 3; ++m)
            {
                REQUIRE(rnd[m].n_rows == 3);
                REQUIRE(rnd[m].n_cols == ph.n_cols);
                double max_diff = 0.0;
                for (int i = 0; i < 6; ++i)
                {
                    arma::uword l, c;
                    draw_index(rnd[m].n_cols, i, l, c);
                    INFO("set " << g.name << ", " << mode[m] << ", value " << i);
                    CHECK(std::abs(rnd[m](l, c) - g.draw[m][i]) < 2e-5); // AVX2 vs GENERIC kernel of sos_val
                    max_diff = std::max(max_diff, std::abs(rnd[m](l, c) - g.draw[m][i]));
                }
                QD_DIAG("%s,%s,%.3e\n", g.name, mode[m], max_diff);
            }
        }
    }
}

TEST_CASE("gbsm_init_phases - lsf phases equal the previous version")
{
    check_phases(true);
}

TEST_CASE("gbsm_draw - lsf values equal the previous version")
{
    check_draws(true);
}

TEST_CASE("gbsm_init_phases - cluster phases equal the previous version")
{
    check_phases(false);
}

TEST_CASE("gbsm_draw - cluster values equal the previous version")
{
    check_draws(false);
}
