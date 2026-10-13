// usage: bench_variants <variant> <n_gen> <n_coeff> <n_pos>
// variants: a4 (library AVX2), ah4 / ah8 (named accumulators, 4 / 8 generators), at8 (tiled, packed weights),
//           g (library GENERIC), gs (grouped accumulation), gsc (grouped + sincos)
// output: variant, n_gen, n_coeff, n_pos, ms per call, ns per element, bit-identical to the library kernel (1/0), max difference
#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#define DECL(name) void name(const double *, const double *, const double *, const double *, const double *, const size_t, const size_t, const double *, const double *, const double *, const size_t, double *)
DECL(var_SOS_AVX2_H4); DECL(var_SOS_AVX2_H8); DECL(var_SOS_AVX2_T8);
void var_SOS_GENERIC(const double *, const double *, const double *, const double *, const double *, const size_t, const size_t, const double *, const double *, const double *, const size_t, double *, int);
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv)
{
    std::string v = argv[1]; size_t n_gen = std::atoll(argv[2]), n_coeff = std::atoll(argv[3]), n_pos = std::atoll(argv[4]);
    const char *name = n_coeff == 128 ? "Comb128" : n_coeff == 256 ? "Comb256" : n_coeff == 512 ? "Comb512" : "Comb1024";
    arma::mat freq, phase; quadriga_lib::sos_preset<double>(name, freq, (arma::vec *)nullptr, (arma::vec *)nullptr, 10.0); quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, phase, 1);
    std::mt19937_64 rng(5); std::vector<double> fx(n_coeff), fy(n_coeff), fz(n_coeff), wc(n_coeff * n_gen), ws(n_coeff * n_gen), px(n_pos), py(n_pos), pz(n_pos);
    double a = std::sqrt(2.0 / n_coeff);
    for (size_t n = 0; n < n_coeff; ++n) fx[n] = freq(n, 0), fy[n] = freq(n, 1), fz[n] = freq(n, 2);
    for (size_t i = 0; i < n_coeff * n_gen; ++i) wc[i] = a * std::cos(phase(i)), ws[i] = a * std::sin(phase(i));
    for (size_t i = 0; i < n_pos; ++i) px[i] = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0, py[i] = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0, pz[i] = 1000.0 * (double)(rng() >> 11) / 9007199254740992.0;
    arma::mat val(n_pos, n_gen), ref(n_pos, n_gen);
#define ARGS fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), n_coeff, n_gen, px.data(), py.data(), pz.data(), n_pos
    auto run = [&](double *out) {
        if (v == "a4") qd_SOS_AVX2(ARGS, out); else if (v == "ah4") var_SOS_AVX2_H4(ARGS, out); else if (v == "ah8") var_SOS_AVX2_H8(ARGS, out); else if (v == "at8") var_SOS_AVX2_T8(ARGS, out);
        else if (v == "g") qd_SOS_GENERIC(ARGS, out); else if (v == "gs") var_SOS_GENERIC(ARGS, out, 0); else var_SOS_GENERIC(ARGS, out, 1); };
    run(val.memptr());
    double best = 1e30, t_all = now(); int reps = 0;
    while ((now() - t_all < 0.6 || reps < 3) && reps < 200) { double t0 = now(); run(val.memptr()); double t = now() - t0; best = t < best ? t : best; ++reps; }
    if (v[0] == 'a') qd_SOS_AVX2(ARGS, ref.memptr()); else qd_SOS_GENERIC(ARGS, ref.memptr());
    std::printf("%s,%zu,%zu,%zu,%.4f,%.3f,%d,%.3e\n", v.c_str(), n_gen, n_coeff, n_pos, best * 1e3, best * 1e9 / ((double)n_pos * n_gen), (int)(std::memcmp(ref.memptr(), val.memptr(), n_pos * n_gen * 8) == 0), arma::abs(ref - val).max());
}
