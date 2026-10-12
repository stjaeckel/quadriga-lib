// usage: bench <mode> <n_gen> <n_coeff> <n_pos> [tile_blocks]
// mode 1: qd_SOS_GENERIC, 2: qd_SOS_AVX2, 10/11/12: sos_val<double> use_kernel 0/1/2, 20: sos_val<float> auto,
//      30: sos_val two devices same map (auto), 31: two maps (auto)
#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv)
{
    int mode = std::atoi(argv[1]); size_t n_gen = std::atoll(argv[2]), n_coeff = std::atoll(argv[3]), n_pos = std::atoll(argv[4]);
    size_t tile = argc > 5 ? std::atoll(argv[5]) : 0;
    const char *name = n_coeff == 128 ? "Comb128" : n_coeff == 256 ? "Comb256" : n_coeff == 512 ? "Comb512" : "Comb1024";
    arma::mat freq, phase, phase_b, pos(n_pos, 3), pos_b(n_pos, 3); arma::vec acf, dist;
    quadriga_lib::sos_preset<double>(name, freq, &acf, &dist, 10.0);
    quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, phase, 1); quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, phase_b, 2);
    std::mt19937_64 rng(5); for (auto &v : pos) v = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0; for (auto &v : pos_b) v = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0;
    std::vector<double> fx(n_coeff), fy(n_coeff), fz(n_coeff), wc(n_coeff * n_gen), ws(n_coeff * n_gen), px(n_pos), py(n_pos), pz(n_pos);
    double a = std::sqrt(2.0 / n_coeff);
    for (size_t n = 0; n < n_coeff; ++n) fx[n] = freq(n, 0), fy[n] = freq(n, 1), fz[n] = freq(n, 2);
    for (size_t i = 0; i < n_coeff * n_gen; ++i) wc[i] = a * std::cos(phase(i)), ws[i] = a * std::sin(phase(i));
    for (size_t i = 0; i < n_pos; ++i) px[i] = pos(i, 0), py[i] = pos(i, 1), pz[i] = pos(i, 2);
    arma::mat val(n_pos, n_gen); arma::fmat valf; arma::fmat ff, pf, posf;
    if (mode == 20) ff = arma::conv_to<arma::fmat>::from(freq), pf = arma::conv_to<arma::fmat>::from(phase), posf = arma::conv_to<arma::fmat>::from(pos), valf.set_size(n_pos, n_gen);
    arma::mat *nm = nullptr; arma::vec *nv = nullptr;
    auto run = [&]() {
        switch (mode) {
        case 1: qd_SOS_GENERIC(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), n_coeff, n_gen, px.data(), py.data(), pz.data(), n_pos, val.memptr()); break;
        case 2: qd_SOS_AVX2(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), n_coeff, n_gen, px.data(), py.data(), pz.data(), n_pos, val.memptr()); break;
        case 10: case 11: case 12: quadriga_lib::sos_val<double>(freq, phase, pos, val, nm, nm, nv, nv, 1.0, false, mode - 10); break;
        case 20: quadriga_lib::sos_val<float>(ff, pf, posf, valf); break;
        case 30: quadriga_lib::sos_val<double>(freq, phase, pos, val, &pos_b, nm, &acf, &dist); break;
        case 31: quadriga_lib::sos_val<double>(freq, phase, pos, val, &pos_b, &phase_b, nv, nv, 2.0); break;
        } };
    run(); // warm-up, page faults
    double best = 1e30, t_all = now(); int reps = 0;
    while ((now() - t_all < 0.6 || reps < 3) && reps < 200) { double t0 = now(); run(); double t = now() - t0; best = t < best ? t : best; ++reps; }
    double chk = 0; if (mode == 20) for (size_t i = 0; i < valf.n_elem; i += 97) chk += valf(i); else for (size_t i = 0; i < val.n_elem; i += 97) chk += val(i);
    std::printf("%d,%zu,%zu,%zu,%.4f,%.3f,%d,%.6f\n", mode, n_gen, n_coeff, n_pos, best * 1e3, best * 1e9 / ((double)n_pos * n_gen), reps, chk);
    return 0;
}
