// Timings of the functions around sos_val. Each line: label, best time in ms over repeated calls.
#include "quadriga_lib.hpp"
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
using namespace quadriga_lib;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double T(const char *label, const std::function<void()> &fn, double budget = 0.5, int max_reps = 2000)
{
    fn(); double best = 1e30, t0 = now(); int reps = 0;
    while ((now() - t0 < budget || reps < 2) && reps < max_reps) { double a = now(); fn(); double t = now() - a; best = t < best ? t : best; ++reps; }
    std::printf("%-78s %12.4f ms\n", label, best * 1e3); std::fflush(stdout); return best;
}
int main(int argc, char **argv)
{
    std::string what = argc > 1 ? argv[1] : "all";
    arma::mat *nm = nullptr; arma::vec *nv = nullptr;
    arma::mat freq, phase, val; arma::vec acf, dist;
    sos_preset<double>("Comb256", freq, &acf, &dist, 10.0);
    std::mt19937_64 rng(5); auto rnd = [&](arma::uword r, arma::uword c, double s) { arma::mat m(r, c); for (auto &v : m) v = s * (double)(rng() >> 11) / 9007199254740992.0; return m; };

    if (what == "all" || what == "small") {
        for (const char *n : {"Comb128", "Comb256", "Comb1024"}) { std::string l = std::string("sos_preset ") + n; T(l.c_str(), [&]() { arma::mat f; arma::vec a, d; sos_preset<double>(n, f, &a, &d, 10.0); }); }
        T("sos_init_phases 256 x 2", [&]() { arma::mat p; sos_init_phases<double>(256, 2, p, 1); });
        T("sos_init_phases 256 x 800", [&]() { arma::mat p; sos_init_phases<double>(256, 800, p, 1); });
        arma::mat p1 = rnd(1, 3, 1000), p100 = rnd(100, 3, 1000); arma::mat ph1; sos_init_phases<double>(256, 1, ph1, 1); arma::vec d1 = {10.0};
        T("sos_val    1 pos x 1 gen (Comb256)", [&]() { sos_val<double>(freq, ph1, p1, val); });
        T("sos_randn  1 pos x 1 gen (Comb256), seed 1", [&]() { sos_randn<double>(d1, p1, val, nm, nv, "Comb256", 1); });
        T("sos_val    100 pos x 1 gen", [&]() { sos_val<double>(freq, ph1, p100, val); });
        T("sos_randn  100 pos x 1 gen", [&]() { sos_randn<double>(d1, p100, val, nm, nv, "Comb256", 1); });
    }
    if (what == "all" || what == "rand") {
        arma::mat pos = rnd(100000, 3, 1000), pos_b = rnd(100000, 3, 1000);
        for (arma::uword ng : {8, 400}) {
            arma::vec same(ng); same.fill(10.0); arma::vec diff(ng); for (arma::uword g = 0; g < ng; ++g) diff(g) = 5.0 + (double)g;
            arma::mat ph; sos_init_phases<double>(256, ng, ph, 1); char l[120];
            std::snprintf(l, 120, "sos_val   1e5 pos x %llu gen (shared frequencies)", (unsigned long long)ng); T(l, [&]() { sos_val<double>(freq, ph, pos, val); });
            std::snprintf(l, 120, "sos_randn 1e5 pos x %llu gen, all distances equal", (unsigned long long)ng); T(l, [&]() { sos_randn<double>(same, pos, val, nm, nv, "Comb256", 1); });
            std::snprintf(l, 120, "sos_randn 1e5 pos x %llu gen, all distances different", (unsigned long long)ng); T(l, [&]() { sos_randn<double>(diff, pos, val, nm, nv, "Comb256", 1); }, 0.5, 20);
            if (ng == 8) { std::snprintf(l, 120, "sos_randn 1e5 pos x %llu gen, equal, two devices same map", (unsigned long long)ng); T(l, [&]() { sos_randn<double>(same, pos, val, &pos_b, nv, "Comb256", 1); });
                           std::snprintf(l, 120, "sos_rand  1e5 pos x %llu gen, all distances equal (Uniform)", (unsigned long long)ng); T(l, [&]() { sos_rand<double>(same, pos, val, nm, nv, "Comb256", 1); });
                           std::snprintf(l, 120, "sos_randi 1e5 pos x %llu gen, all distances equal", (unsigned long long)ng); T(l, [&]() { sos_randi<double>(same, pos, 6, val, nm, nv, "Comb256", 1); }); }
        }
    }
    if (what == "all" || what == "map") {
        arma::mat ph; sos_init_phases<double>(256, 20, ph, 1);
        arma::vec x = arma::regspace(0.0, 0.5, 250.0), y = arma::regspace(0.0, 0.5, 100.0); std::vector<arma::cube> map;
        arma::mat pos(x.n_elem * y.n_elem, 2); for (arma::uword ix = 0; ix < x.n_elem; ++ix) for (arma::uword iy = 0; iy < y.n_elem; ++iy) pos(iy + y.n_elem * ix, 0) = x(ix), pos(iy + y.n_elem * ix, 1) = y(iy);
        T("sos_map 501 x 201 grid x 20 gen (Comb256)", [&]() { sos_map<double>(freq, ph, x, map, &y); });
        T("sos_val on the same 100701 positions x 20 gen", [&]() { sos_val<double>(freq, ph, pos, val); });
    }
    if (what == "all" || what == "bias") {
        for (arma::uword np : {1000, 100000}) for (arma::uword ng : {8, 400}) {
            arma::mat pos = rnd(np, 3, 1000), pos_b = rnd(np, 3, 1000), ph0; sos_init_phases<double>(256, ng, ph0, 1); char l[120];
            std::snprintf(l, 120, "sos_remove_bias %llu pos x %llu gen, one device", (unsigned long long)np, (unsigned long long)ng); T(l, [&]() { arma::mat ph = ph0; sos_remove_bias<double>(freq, ph, pos); }, 0.5, 50);
            std::snprintf(l, 120, "sos_remove_bias %llu pos x %llu gen, same map", (unsigned long long)np, (unsigned long long)ng); T(l, [&]() { arma::mat ph = ph0; sos_remove_bias<double>(freq, ph, pos, &pos_b, nm, &acf, &dist); }, 0.5, 50);
            std::snprintf(l, 120, "   for comparison: sos_val %llu pos x %llu gen", (unsigned long long)np, (unsigned long long)ng); T(l, [&]() { sos_val<double>(freq, ph0, pos, val); }, 0.5, 50); }
    }
    if (what == "all" || what == "mse") {
        for (const char *n : {"Comb256", "Comb1024"}) { arma::mat f; arma::vec a, d; sos_preset<double>(n, f, &a, &d, 10.0); double c, al; arma::cube ap; arma::mat a2; char l[120];
            for (arma::uword nd : {10, 300}) { std::snprintf(l, 120, "sos_calc_mse %s, n_dir %llu, directional only", n, (unsigned long long)nd); T(l, [&]() { sos_calc_mse<double>(f, a, d, nullptr, nullptr, nv, nm, nd); }, 0.5, 20); }
            std::snprintf(l, 120, "sos_calc_mse %s, n_dir 10, with mse_core and mse_all", n); T(l, [&]() { sos_calc_mse<double>(f, a, d, &c, &al, nv, nm, 10); }, 0.5, 20);
            std::snprintf(l, 120, "sos_acf_approx %s (3 planes, 797 x 797)", n); T(l, [&]() { sos_acf_approx<double>(f, d, ap); }, 0.5, 20);
            std::snprintf(l, 120, "sos_acf_2d (797 x 797)"); T(l, [&]() { sos_acf_2d<double>(a, d, a2); }, 0.5, 20); }
    }
    if (what == "all" || what == "est") {
        arma::vec bins = arma::regspace(1.0, 2.0, 49.0), est, ph; { arma::mat p; sos_init_phases<double>(256, 1, p, 1); ph = p.col(0); }
        for (arma::uword np : {1000, 5000, 20000}) { char l[120]; std::snprintf(l, 120, "sos_acf_estimate n_pos %llu (pairs %.1e), one device", (unsigned long long)np, 2.0 * np * np); T(l, [&]() { sos_acf_estimate<double>(freq, ph, bins, np, est, nv, nv, nv, nv, 1); }, 0.3, 5);
            if (np <= 5000) { arma::vec dual; std::snprintf(l, 120, "sos_acf_estimate n_pos %llu, with acf_est_dual", (unsigned long long)np); T(l, [&]() { sos_acf_estimate<double>(freq, ph, bins, np, est, nv, &dual, &acf, &dist, 1); }, 0.3, 5); } }
    }
    if (what == "all" || what == "gen") {
        arma::mat f; arma::vec ase; char l[120];
        for (arma::uword n : {128, 256}) { std::snprintf(l, 120, "sos_generate %llu sinusoids, 3-D, 300 directions, preset settings, seed 13", (unsigned long long)n); double t = T(l, [&]() { sos_generate<double>(acf, dist, n, f, &ase, nm, 3, true, 300, true, 0.005, 15.0 / 49.75, 13); }, 0.0, 1); std::printf("      passes: %llu, %.1f ms per pass\n", (unsigned long long)ase.n_elem, t * 1e3 / ase.n_elem); }
        arma::vec d2 = arma::regspace(0.0, 1.0, 40.0), a2 = arma::exp(-d2 / 10.0);
        double t = T("sos_generate 24 sinusoids, 1-D, 41 samples (as in the test suite)", [&]() { sos_generate<double>(a2, d2, 24, f, &ase, nm, 1, true, 7, true, 0.0, 0.0, 3); }, 0.0, 1); std::printf("      passes: %llu, last 3 errors: %.6e %.6e %.6e\n", (unsigned long long)ase.n_elem, ase(ase.n_elem - 3), ase(ase.n_elem - 2), ase(ase.n_elem - 1));
        // unequal spacing: reported error against an independent evaluation
        arma::vec d3 = {0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0}, a3 = arma::exp(-d3 / 10.0);
        sos_generate<double>(a3, d3, 16, f, &ase, nm, 2, true, 7, true, 0.0, 0.0, 3);
        double err = 0; for (int t2 = 0; t2 < 7; ++t2) { double az = 3.141592653589793 * t2 / 7.0; for (arma::uword s = 0; s < d3.n_elem; ++s) { double r = 0; for (arma::uword n = 0; n < 16; ++n) r += std::cos(6.283185307179586 * (f(n, 0) * std::cos(az) + f(n, 1) * std::sin(az)) * d3(s)); double e = a3(s) - r / 16.0; err += e * e; } }
        std::printf("sos_generate unequal spacing (12 samples, 2-D): reported %.12e, independent %.12e, passes %llu\n", ase(ase.n_elem - 1), err / (7.0 * d3.n_elem), (unsigned long long)ase.n_elem);
        // drift of the incremental sums: reported final error against independent evaluation for the 128-sinusoid preset run
        sos_generate<double>(acf, dist, 128, f, &ase, nm, 3, true, 300, true, 0.005, 15.0 / 49.75, 13);
        double mse = sos_calc_mse<double>(f, acf, dist, nullptr, nullptr, nv, nm, 300);
        std::printf("sos_generate drift check (128, 3-D): reported %.15e, sos_calc_mse %.15e, rel diff %.2e\n", ase(ase.n_elem - 1), std::pow(10.0, -mse / 10.0), ase(ase.n_elem - 1) / std::pow(10.0, -mse / 10.0) - 1.0);
        for (bool ri : {true, false}) { sos_generate<double>(acf, dist, 128, f, &ase, nm, 3, true, 100, ri, 0.005, 15.0 / 49.75, 13); std::printf("sos_generate 128, 3-D, 100 dir, random_init=%d: passes %llu, fit %.2f dB\n", (int)ri, (unsigned long long)ase.n_elem, sos_calc_mse<double>(f, acf, dist, nullptr, nullptr, nv, nm, 100)); }
    }
    return 0;
}
