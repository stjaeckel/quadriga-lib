// Where the time of sos_val goes outside the kernel. usage: bench4 <n_gen> <n_coeff> <n_pos>
#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"
#include <chrono>
#include <cstdio>
#include <functional>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double T(const std::function<void()> &fn) { fn(); double best = 1e30, t0 = now(); int reps = 0; while ((now() - t0 < 0.5 || reps < 3) && reps < 100) { double a = now(); fn(); double t = now() - a; best = t < best ? t : best; ++reps; } return best * 1e3; }
int main(int, char **argv)
{
    size_t n_gen = std::atoll(argv[1]), n_coeff = std::atoll(argv[2]), n_pos = std::atoll(argv[3]);
    const char *name = n_coeff == 128 ? "Comb128" : n_coeff == 256 ? "Comb256" : n_coeff == 512 ? "Comb512" : "Comb1024";
    arma::mat freq, phase, pos(n_pos, 3), pos_b(n_pos, 3), val(n_pos, n_gen), val2; arma::vec acf, dist; arma::mat *nm = nullptr; arma::vec *nv = nullptr;
    quadriga_lib::sos_preset<double>(name, freq, &acf, &dist, 10.0); quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, phase, 1);
    std::mt19937_64 rng(5); for (auto &v : pos) v = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0; for (auto &v : pos_b) v = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0;
    std::vector<double> fx(n_coeff), fy(n_coeff), fz(n_coeff), wc, ws, px, py, pz; double a = std::sqrt(2.0 / n_coeff);
    for (size_t n = 0; n < n_coeff; ++n) fx[n] = freq(n, 0), fy[n] = freq(n, 1), fz[n] = freq(n, 2);
    double t_w = T([&]() { wc.resize(n_coeff * n_gen), ws.resize(n_coeff * n_gen); for (size_t i = 0; i < n_coeff * n_gen; ++i) wc[i] = a * std::cos(phase(i)), ws[i] = a * std::sin(phase(i)); });
    double t_p = T([&]() { px.assign(n_pos, 0.0), py.assign(n_pos, 0.0), pz.assign(n_pos, 0.0); for (size_t i = 0; i < n_pos; ++i) px[i] = pos(i, 0), py[i] = pos(i, 1), pz[i] = pos(i, 2); });
    double t_k = T([&]() { qd_SOS_AVX2(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), n_coeff, n_gen, px.data(), py.data(), pz.data(), n_pos, val.memptr()); });
    double t_tmp = T([&]() { arma::mat sa(n_pos, n_gen, arma::fill::none); qd_SOS_AVX2(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), n_coeff, n_gen, px.data(), py.data(), pz.data(), n_pos, sa.memptr()); });
    arma::mat sa = val;
    double t_cp = T([&]() { const double *ps = sa.memptr(); double *pv = val.memptr(); for (size_t i = 0; i < n_pos * n_gen; ++i) pv[i] = ps[i]; });
    double t_erf = T([&]() { const double *ps = sa.memptr(); double *pv = val.memptr(); const double r = std::sqrt(0.5); for (size_t i = 0; i < n_pos * n_gen; ++i) pv[i] = 0.5 * std::erfc(-ps[i] * r); });
    double t_erfp = T([&]() { const double *ps = sa.memptr(); double *pv = val.memptr(); const double r = std::sqrt(0.5); const long long n = (long long)(n_pos * n_gen);
#pragma omp parallel for
        for (long long i = 0; i < n; ++i) pv[i] = 0.5 * std::erfc(-ps[i] * r); });
    double t_api = T([&]() { quadriga_lib::sos_val<double>(freq, phase, pos, val); });
    double t_api_fresh = T([&]() { arma::mat v; quadriga_lib::sos_val<double>(freq, phase, pos, v); });
    double t_uni = T([&]() { quadriga_lib::sos_val<double>(freq, phase, pos, val, nm, nm, nv, nv, 1.0, true); });
    double t_same = T([&]() { quadriga_lib::sos_val<double>(freq, phase, pos, val, &pos_b, nm, &acf, &dist); });
    // normalisation per position as in sos_val (serial binary search)
    arma::vec w(n_pos); double t_norm = T([&]() { for (size_t p = 0; p < n_pos; ++p) { double dx = pos(p, 0) - pos_b(p, 0), dy = pos(p, 1) - pos_b(p, 1), dz = pos(p, 2) - pos_b(p, 2), d = std::sqrt(dx * dx + dy * dy + dz * dz);
        size_t lo = 0, hi = acf.n_elem - 1; double A; if (d >= dist(hi)) A = acf(hi); else { while (hi - lo > 1) { size_t mid = (lo + hi) / 2; if (dist(mid) <= d) lo = mid; else hi = mid; } A = acf(lo) + (d - dist(lo)) / (dist(hi) - dist(lo)) * (acf(hi) - acf(lo)); } w(p) = 1.0 / std::sqrt(2.0 + 2.0 * A); } });
    double t_comb = T([&]() { for (size_t g = 0; g < n_gen; ++g) { double *pa = val.colptr(g); const double *pb = sa.colptr(g), *pw = w.memptr(); for (size_t p = 0; p < n_pos; ++p) pa[p] = (pa[p] + pb[p]) * pw[p]; } });
    std::printf("n_gen %zu, N %zu, n_pos %zu, threads %s [ms]\n  weights (cos, sin of the phases)   %9.3f\n  positions to x/y/z vectors         %9.3f\n  kernel into an existing matrix     %9.3f\n  kernel into a new temporary        %9.3f\n  copy temporary -> val              %9.3f\n  erfc mapping, serial               %9.3f\n  erfc mapping, parallel             %9.3f\n  normalisation weights (same map)   %9.3f\n  combination of two devices         %9.3f\n  sos_val, val pre-sized             %9.3f\n  sos_val, val empty                 %9.3f\n  sos_val, uniform = true            %9.3f\n  sos_val, two devices same map      %9.3f\n",
                n_gen, n_coeff, n_pos, getenv("OMP_NUM_THREADS"), t_w, t_p, t_k, t_tmp, t_cp, t_erf, t_erfp, t_norm, t_comb, t_api, t_api_fresh, t_uni, t_same);
}
