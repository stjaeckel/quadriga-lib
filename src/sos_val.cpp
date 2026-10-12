// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib.hpp"
#include "quadriga_lib_helper_functions.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

// Frequencies as separate x, y, z vectors in double, scaled; missing dimensions are 0
template <typename dtype>
static void qd_sos_split_freq(const arma::Mat<dtype> &freq, double scale,
                              std::vector<double> &fx, std::vector<double> &fy, std::vector<double> &fz)
{
    arma::uword n_coeff = freq.n_rows;
    fx.assign(n_coeff, 0.0), fy.assign(n_coeff, 0.0), fz.assign(n_coeff, 0.0);
    for (arma::uword n = 0; n < n_coeff; ++n)
    {
        fx[n] = scale * (double)freq.at(n, 0);
        if (freq.n_cols > 1)
            fy[n] = scale * (double)freq.at(n, 1);
        if (freq.n_cols > 2)
            fz[n] = scale * (double)freq.at(n, 2);
    }
}

// Positions as separate x, y, z vectors in double; missing dimensions are 0
template <typename dtype>
static void qd_sos_split_pos(const arma::Mat<dtype> &pos,
                             std::vector<double> &px, std::vector<double> &py, std::vector<double> &pz)
{
    arma::uword n_pos = pos.n_rows;
    px.assign(n_pos, 0.0), py.assign(n_pos, 0.0), pz.assign(n_pos, 0.0);
    for (arma::uword c = 0; c < pos.n_cols; ++c)
    {
        const dtype *p = pos.colptr(c);
        double *q = (c == 0) ? px.data() : (c == 1 ? py.data() : pz.data());
        for (arma::uword i = 0; i < n_pos; ++i)
            q[i] = (double)p[i];
    }
}

// Weights a·cos(phase) and a·sin(phase) with a = sqrt(2/n_coeff)
template <typename dtype>
static void qd_sos_weights(const arma::Mat<dtype> &phase, std::vector<double> &wc, std::vector<double> &ws)
{
    arma::uword n = phase.n_elem;
    double a = std::sqrt(2.0 / (double)phase.n_rows);
    wc.resize(n), ws.resize(n);
    const dtype *p = phase.memptr();
    for (arma::uword i = 0; i < n; ++i)
        wc[i] = a * std::cos((double)p[i]), ws[i] = a * std::sin((double)p[i]);
}

static void qd_sos_run_kernel(int kernel, const std::vector<double> &fx, const std::vector<double> &fy, const std::vector<double> &fz,
                              const std::vector<double> &wc, const std::vector<double> &ws, size_t n_gen,
                              const std::vector<double> &px, const std::vector<double> &py, const std::vector<double> &pz,
                              double *val)
{
    if (kernel == 2)
    {
#if BUILD_WITH_AVX2
        qd_SOS_AVX2(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), fx.size(), n_gen,
                    px.data(), py.data(), pz.data(), px.size(), val);
#else
        throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
#endif
    }
    else
        qd_SOS_GENERIC(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), fx.size(), n_gen,
                       px.data(), py.data(), pz.data(), px.size(), val);
}

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_val
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of `freq`)
  and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function (ACF)
  `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the frequencies were
  fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The sines and
  cosines of the positions are computed once and reused for all generators
- Positions may have fewer columns than `freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and `pos_b`
  and `A` the target ACF interpolated as in [[sos_acfi]]; `acf` and `dist` are required. The value does
  not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `freq · freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same matrix as
  `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and 1
  as far as `val` is Normal distributed
- Kernels: GENERIC computes in double precision. AVX2 reduces the argument `f_n·x` to one period in
  double precision, evaluates sine and cosine in single precision and accumulates in double precision;
  its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, at any distance from the origin
- The value at a position does not depend on the other positions or generators of the call, nor on the
  number of threads
- `val` must not be the same object as an input

## Declaration:
```
void quadriga_lib::sos_val(
    const arma::Mat<dtype> &freq,
    const arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Mat<dtype> *phase_b = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    dtype freq_scale_b = 1.0,
    bool uniform = false,
    int use_kernel = 0);
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or empty =
  same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0 and increase strictly; `[n_acf]`
- **`freq_scale_b`** *(optional)* — Frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0
- **`uniform`** *(optional)* — Output distribution: `false` = Normal(0,1), `true` = Uniform between 0 and 1
- **`use_kernel`** *(optional)* — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  throws if the requested kernel is unavailable

## Output:
- **`val`** — Values; one row per position, one column per generator; `[n_pos, n_gen]`

## See also:
- [[sos_preset]] (frequencies and ACF of the predefined models)
- [[sos_init_phases]] (random phases)
- [[sos_map]] (evaluation on a grid)
- [[sos_randn]] (generator set-up and evaluation in one call)
MD!*/

template <typename dtype>
void quadriga_lib::sos_val(const arma::Mat<dtype> &freq,
                           const arma::Mat<dtype> &phase,
                           const arma::Mat<dtype> &pos,
                           arma::Mat<dtype> &val,
                           const arma::Mat<dtype> *pos_b,
                           const arma::Mat<dtype> *phase_b,
                           const arma::Col<dtype> *acf,
                           const arma::Col<dtype> *dist,
                           dtype freq_scale_b,
                           bool uniform,
                           int use_kernel)
{
    const bool two_devices = pos_b != nullptr && !pos_b->is_empty();
    const bool two_maps = phase_b != nullptr && !phase_b->is_empty();

    if (freq.n_elem == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (freq.n_cols > 3)
        throw std::invalid_argument("Input 'freq' must have 1, 2 or 3 columns.");
    if (phase.n_elem == 0)
        throw std::invalid_argument("Input 'phase' cannot be empty.");
    if (phase.n_rows != freq.n_rows)
        throw std::invalid_argument("Number of rows in 'phase' does not match number of rows in 'freq'.");
    if (pos.n_elem == 0)
        throw std::invalid_argument("Input 'pos' cannot be empty.");
    if (pos.n_cols > freq.n_cols)
        throw std::invalid_argument("Input 'pos' cannot have more columns than 'freq'.");
    if (two_devices)
    {
        if (pos_b->n_rows != pos.n_rows)
            throw std::invalid_argument("Number of rows in 'pos_b' does not match number of rows in 'pos'.");
        if (pos_b->n_cols > freq.n_cols)
            throw std::invalid_argument("Input 'pos_b' cannot have more columns than 'freq'.");
    }
    if (two_maps)
    {
        if (!two_devices)
            throw std::invalid_argument("Input 'phase_b' requires 'pos_b'.");
        if (phase_b->n_rows != phase.n_rows || phase_b->n_cols != phase.n_cols)
            throw std::invalid_argument("Input 'phase_b' must have the same size as 'phase'.");
    }
    if (!(freq_scale_b > (dtype)0.0) || !std::isfinite((double)freq_scale_b))
        throw std::invalid_argument("Input 'freq_scale_b' must be > 0.");
    if (freq_scale_b != (dtype)1.0 && !two_maps)
        throw std::invalid_argument("Input 'freq_scale_b' other than 1 requires 'phase_b'.");
    if (two_devices && !two_maps)
    {
        if (acf == nullptr || dist == nullptr || acf->is_empty() || dist->is_empty())
            throw std::invalid_argument("Inputs 'acf' and 'dist' are required for two devices on the same map.");
        qd_sos_check_acf(*acf, *dist);
    }

    // kernel: 1 = GENERIC, 2 = AVX2
    int kernel = 1;
    if (use_kernel == 1)
        kernel = 1;
    else if (use_kernel == 2)
    {
        if (!quadriga_lib::quadriga_lib_has_AVX2())
            throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
        kernel = 2;
    }
    else if (use_kernel == 0)
        kernel = quadriga_lib::quadriga_lib_has_AVX2() ? 2 : 1;
    else
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1 or 2.");

    const arma::uword n_pos = pos.n_rows, n_gen = phase.n_cols;

    // First device
    std::vector<double> fx, fy, fz, px, py, pz, wc, ws;
    qd_sos_split_freq(freq, 1.0, fx, fy, fz);
    qd_sos_split_pos(pos, px, py, pz);
    qd_sos_weights(phase, wc, ws);

    arma::mat sa(n_pos, n_gen, arma::fill::none);
    qd_sos_run_kernel(kernel, fx, fy, fz, wc, ws, n_gen, px, py, pz, sa.memptr());

    if (two_devices)
    {
        std::vector<double> qx, qy, qz;
        qd_sos_split_pos(*pos_b, qx, qy, qz);
        if (two_maps)
        {
            qd_sos_split_freq(freq, (double)freq_scale_b, fx, fy, fz);
            qd_sos_weights(*phase_b, wc, ws);
        }

        arma::mat sb(n_pos, n_gen, arma::fill::none);
        qd_sos_run_kernel(kernel, fx, fy, fz, wc, ws, n_gen, qx, qy, qz, sb.memptr());

        // Normalisation per position
        arma::vec w(n_pos);
        if (two_maps)
            w.fill(std::sqrt(0.5));
        else
        {
            const arma::vec acf_d = arma::conv_to<arma::vec>::from(*acf), dist_d = arma::conv_to<arma::vec>::from(*dist);
            for (arma::uword p = 0; p < n_pos; ++p)
            {
                double dx = px[p] - qx[p], dy = py[p] - qy[p], dz = pz[p] - qz[p];
                double A = qd_sos_acf_interp(acf_d.memptr(), dist_d.memptr(), acf_d.n_elem, std::sqrt(dx * dx + dy * dy + dz * dz));
                w.at(p) = 1.0 / std::sqrt(2.0 + 2.0 * A);
            }
        }

        for (arma::uword g = 0; g < n_gen; ++g)
        {
            double *pa = sa.colptr(g);
            const double *pb = sb.colptr(g), *pw = w.memptr();
            for (arma::uword p = 0; p < n_pos; ++p)
                pa[p] = (pa[p] + pb[p]) * pw[p];
        }
    }

    if (val.n_rows != n_pos || val.n_cols != n_gen)
        val.set_size(n_pos, n_gen);

    const double *ps = sa.memptr();
    dtype *pv = val.memptr();
    if (uniform)
    {
        const double rsqrt2 = std::sqrt(0.5);
        for (arma::uword i = 0; i < n_pos * n_gen; ++i)
            pv[i] = (dtype)(0.5 * std::erfc(-ps[i] * rsqrt2));
    }
    else
        for (arma::uword i = 0; i < n_pos * n_gen; ++i)
            pv[i] = (dtype)ps[i];
}

template void quadriga_lib::sos_val(const arma::Mat<float> &freq, const arma::Mat<float> &phase, const arma::Mat<float> &pos, arma::Mat<float> &val,
                                    const arma::Mat<float> *pos_b, const arma::Mat<float> *phase_b, const arma::Col<float> *acf, const arma::Col<float> *dist,
                                    float freq_scale_b, bool uniform, int use_kernel);
template void quadriga_lib::sos_val(const arma::Mat<double> &freq, const arma::Mat<double> &phase, const arma::Mat<double> &pos, arma::Mat<double> &val,
                                    const arma::Mat<double> *pos_b, const arma::Mat<double> *phase_b, const arma::Col<double> *acf, const arma::Col<double> *dist,
                                    double freq_scale_b, bool uniform, int use_kernel);
