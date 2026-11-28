// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_sos.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Positions as separate x, y, z arrays in double; missing dimensions are 0
// For dtype = double the columns of 'pos' are used in place and 'store' only holds the zeros of the missing
// columns; for dtype = float 'store' holds the converted coordinates.
template <typename dtype>
static void qd_sos_split_pos(const arma::Mat<dtype> &pos, std::vector<double> &store,
                             const double *&px, const double *&py, const double *&pz)
{
    const arma::uword n_pos = pos.n_rows, n_cols = pos.n_cols;
    const double *col[3] = {nullptr, nullptr, nullptr};
    if (std::is_same<dtype, double>::value)
    {
        if (n_cols < 3)
            store.assign(n_pos, 0.0);
        for (arma::uword c = 0; c < 3; ++c)
            col[c] = (c < n_cols) ? reinterpret_cast<const double *>(pos.colptr(c)) : store.data();
    }
    else
    {
        store.assign(3 * n_pos, 0.0);
        const long long n_pos_ll = (long long)n_pos;
        for (arma::uword c = 0; c < 3; ++c)
        {
            double *q = store.data() + c * n_pos;
            col[c] = q;
            if (c < n_cols)
            {
                const dtype *p = pos.colptr(c);
#pragma omp parallel for if (n_pos_ll > 65536)
                for (long long i = 0; i < n_pos_ll; ++i)
                    q[i] = (double)p[i];
            }
        }
    }
    px = col[0], py = col[1], pz = col[2];
}

// Weights a·cos(phase) and a·sin(phase) with a = sqrt(2/n_coeff)
template <typename dtype>
static void qd_sos_weights(const arma::Mat<dtype> &phase, std::vector<double> &wc, std::vector<double> &ws)
{
    const long long n = (long long)phase.n_elem;
    const double a = std::sqrt(2.0 / (double)phase.n_rows);
    wc.resize(phase.n_elem), ws.resize(phase.n_elem);
    const dtype *p = phase.memptr();
    double *pc = wc.data(), *ps = ws.data();
#pragma omp parallel for if (n > 16384)
    for (long long i = 0; i < n; ++i)
        pc[i] = a * std::cos((double)p[i]), ps[i] = a * std::sin((double)p[i]);
}

// Second device of a kernel call: frequencies g and positions q, or none (g == nullptr)
struct qd_sos_second
{
    const std::vector<double> *gx = nullptr, *gy = nullptr, *gz = nullptr;
    const double *qx = nullptr, *qy = nullptr, *qz = nullptr;
};

static void qd_sos_run_kernel([[maybe_unused]] int kernel, const std::vector<double> &fx, const std::vector<double> &fy, const std::vector<double> &fz,
                              const std::vector<double> &wc, const std::vector<double> &ws, size_t n_gen,
                              const double *px, const double *py, const double *pz, size_t n_pos,
                              double *val, const qd_sos_second &b = qd_sos_second())
{
    const double *gx = b.gx ? b.gx->data() : nullptr, *gy = b.gy ? b.gy->data() : nullptr, *gz = b.gz ? b.gz->data() : nullptr;
    // Without BUILD_WITH_AVX2, qd_kernel_auto_avx2 never returns 2
#if BUILD_WITH_AVX2
    if (kernel == 2)
    {
        qd_SOS_AVX2(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), fx.size(), n_gen, px, py, pz, n_pos, val,
                    gx, gy, gz, b.qx, b.qy, b.qz);
        return;
    }
#endif
    qd_SOS_GENERIC(fx.data(), fy.data(), fz.data(), wc.data(), ws.data(), fx.size(), n_gen, px, py, pz, n_pos, val,
                   gx, gy, gz, b.qx, b.qy, b.qz);
}

// Fixed permutation of the sinusoids for two devices (declared in qd_helpers_sos.hpp)
// - Sattolo's algorithm with the uniform stream of qd_rng(2021): one cycle through all sinusoids, so that no sinusoid
//   is paired with itself (n_coeff >= 2); the same on all platforms, a function of n_coeff only
std::vector<arma::uword> qd_sos_permutation(arma::uword n_coeff)
{
    std::vector<arma::uword> p(n_coeff);
    for (arma::uword n = 0; n < n_coeff; ++n)
        p[n] = n;
    std::mt19937_64 rng = qd_rng(2021);
    for (arma::uword i = n_coeff; i > 1; --i)
    {
        const arma::uword j = (arma::uword)(qd_rng_uniform(rng) * (double)(i - 1)); // 0 ... i - 2
        std::swap(p[i - 1], p[j]);
    }
    return p;
}

// Calls fn(i0, i1) for consecutive index ranges that cover 0 ... n-1, in parallel for large n.
// The ranges have a fixed size, and fn must only touch the elements of its range.
template <typename Fn>
static void qd_sos_parallel_ranges(arma::uword n, Fn fn)
{
    const arma::uword block = 4096;
    const long long n_block = (long long)((n + block - 1) / block);
#pragma omp parallel for if (n_block > 4)
    for (long long i_block = 0; i_block < n_block; ++i_block)
    {
        const arma::uword i0 = (arma::uword)i_block * block, i1 = (n - i0 < block) ? n : i0 + block;
        fn(i0, i1);
    }
}

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!SECTION_DESC
These functions generate **spatially correlated random numbers**: values that are random, but similar at
nearby positions. Two positions that are much closer than the *decorrelation distance* get almost the same
value, two positions that are far apart get independent values. The decorrelation distance is the distance
at which the correlation has fallen to `exp(-1)`, about 0.37. Typical uses are shadow fading maps and the
spatially consistent parameters of the 3GPP TR 38.901 channel model.<br><br>

The values are a sum of sinusoids, `s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n )`. The frequencies
`f_n` set how the correlation falls with the distance, i.e. the autocorrelation function (ACF). The random
phases `phi_n` select one realisation of the random field. The generator needs no map and no grid:
positions can be anywhere, in 1, 2 or 3 dimensions, and the same frequencies and phases always give the
same field.<br><br>

**Quick start**<br>
[[sos_randn]] sets up the generators, draws the phases and returns the values in one call. [[sos_rand]]
and [[sos_randi]] do the same for Uniform and integer values.

```
#include "quadriga_lib.hpp"

// 1000 positions; one row per position, columns are {x, y, z} in meters
arma::mat pos(1000, 3, arma::fill::randu);
pos *= 100.0;

// One independent generator per decorrelation distance: 10 m and 50 m
arma::vec dist_decorr = {10.0, 50.0};

// Normal distributed values with zero mean and unit variance, size [1000, 2]
arma::mat val;
quadriga_lib::sos_randn<double>(dist_decorr, pos, val, {}, {}, "Comb256", 42);

// Uniform values between 0 and 1, and integers from 1 to 6
arma::mat uni, dice;
quadriga_lib::sos_rand<double>(dist_decorr, pos, uni, {}, {}, "Comb256", 42);
quadriga_lib::sos_randi<double>(dist_decorr, pos, 6, dice, {}, {}, "Comb256", 42);
```

The last argument is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field.<br><br>

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.<br><br>
**Keeping a generator**<br>
A generator is two matrices: the frequencies from [[sos_preset]] and the phases from [[sos_init_phases]].
[[sos_val]] evaluates them. Each column of the phase matrix is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```
// Frequencies of a predefined generator for a decorrelation distance of 10 m
arma::vec acf, dist;
arma::mat spatial_freq = quadriga_lib::sos_preset<double>("Comb256", &acf, &dist, 10.0);

// Phases of 20 generators; each column is one independent random field
arma::mat phase = quadriga_lib::sos_init_phases<double>(spatial_freq.n_rows, 20, 42);

// Values of all 20 generators at the positions, size [1000, 20]
arma::mat val_all;
quadriga_lib::sos_val(spatial_freq, phase, pos, val_all);

// The same fields at other positions, at any later time
arma::mat pos_new = pos + 2.5, val_new;
quadriga_lib::sos_val(spatial_freq, phase, pos_new, val_new);

// Uniform distributed output
arma::mat val_uni;
quadriga_lib::sos_val<double>(spatial_freq, phase, pos, val_uni, {}, 1.0, true, true);

// Map on a regular grid, one cube of size [n_y, n_x, 1] per generator
arma::vec x = arma::regspace(0.0, 0.5, 250.0), y = arma::regspace(0.0, 0.5, 100.0);
std::vector<arma::cube> grid;
quadriga_lib::sos_map<double>(spatial_freq, phase, x, grid, y);
```

- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `"Exp512"`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [[sos_calc_mse]] reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `"Comb256"` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves, keeps the correlation of one device while the
  other stays, and does not change when the two ends swap.

```
// Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
arma::mat pos_tx = pos, pos_rx(1000, 3, arma::fill::randu), val_link;
pos_rx *= 100.0;
quadriga_lib::sos_val<double>(spatial_freq, phase, pos_tx, val_link, pos_rx);
```

- **Mean over an area:** the values at positions that lie within a few decorrelation distances of each
  other are similar, so their average is not zero for a single realisation. This is how a correlated
  field behaves. [[sos_remove_bias]] can force the average to zero if an application needs it. This
  removes variance: for 200 positions in a square with an edge of 20 decorrelation distances 2.5 %, for
  an edge of 5 decorrelation distances 14 %, for 2 already 44 %. It is meant for positions that spread
  over many decorrelation distances.
- **Coordinates:** use local coordinates. The argument `f_n·x` of a sinusoid is computed in double
  precision and is known to about 1e-16 of its size, so the values lose accuracy with the distance from
  the origin counted in wavelengths. Up to 1e9 wavelengths (more than 1e9 m for a decorrelation distance
  of 10 m) the results stay within the tolerance stated in [[sos_val]]. The `float` versions of the
  functions read positions in single precision, i.e. rounded to about 1 mm at a distance of 10 km.
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [[sos_generate]], which runs on one thread. No result depends on the number of threads,
  except the last digits of the errors of [[sos_calc_mse]].
  [[sos_val]] computes in its output: for `double` it needs no further memory of that size for one device
  and one more array of that size for two devices; for `float` one array in double precision is added.

**Own autocorrelation functions**<br>
[[sos_generate]] finds the frequencies for any ACF that is given as a table over the distance, and
[[sos_calc_mse]] reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```
// Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
arma::vec dist_own = arma::regspace(0.0, 0.125, 24.875);
arma::vec acf_own = arma::exp(-arma::square(dist_own / 5.0));

// 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
arma::mat freq_own = quadriga_lib::sos_generate<double>(acf_own, dist_own, 256, nullptr, nullptr, 3, true, 300, true,
                                                        0.01, 0.0, 1);

// Approximation error in dB, larger is better; about 35 dB for this example
double mse = quadriga_lib::sos_calc_mse(freq_own, acf_own, dist_own);
```

The result is used like a preset: draw phases with [[sos_init_phases]] and evaluate with [[sos_val]]. To
change the decorrelation distance afterwards, divide the frequencies and multiply the distances by the
same factor.<br><br>

**Reference**<br>
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
<a target="_blank" rel="noopener noreferrer" href="https://ieeexplore.ieee.org/document/8644265">IEEE Xplore</a> (Original version),
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.pdf">PDF version</a>,
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.md">Markdown version</a><br><br>
SECTION_DESC!*/

/*!MD
# sos_val
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of `spatial_freq`)
  and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function (ACF)
  `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the frequencies were
  fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The sines and
  cosines of the positions are computed once and reused for all generators
- Positions may have fewer columns than `spatial_freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices (`pos_b` given): one generator over the positions of both,
  `s(x, y) = sqrt(2/n_coeff) · sum_n cos( 2·pi · (f_n·x + b·f_p(n)·y) + phi_n )`, with `b = spatial_freq_scale_b` and
  `p` a fixed pseudo-random permutation of the sinusoids that depends only on `n_coeff` and pairs no sinusoid with
  itself. The correlation of two links approximates the product of the ACFs of the two moves; while one device
  stays at its position, it is the ACF of one device (with `reciprocal = true` for devices more than about one
  decorrelation distance apart)
- `reciprocal = true` (requires `b = 1`): `val = ( s(pos, pos_b) + s(pos_b, pos) ) / sqrt( 2 + 2·c )` with
  `c = (1/n_coeff) · sum_n cos( 2·pi · (f_n - f_p(n))·(pos - pos_b) )`. The value does not change when the two
  devices swap their positions; for devices closer than about one decorrelation distance the correlation of two
  links contains the term this symmetry requires
- `reciprocal = false`: `val = s(pos, pos_b)`. With `pos` and `pos_b` exchanged, the same phases give a second
  field with the same statistics, for example the arrival value of a link whose departure value is `val`
- For large moves of both devices the correlation keeps a residual with rms `1/sqrt(2·n_coeff)`
- With `reciprocal = true`, a position where `c = -1` (possible only for tables with few sinusoids) gives NaN
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and 1
  as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the tail is
  rounded to them, for `dtype = float` every value above 5.4 gives 1
- Kernels: GENERIC computes in double precision. AVX2 reduces the argument `f_n·x` to one period in
  double precision, evaluates sine and cosine in single precision and accumulates in double precision;
  its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by less than 1e-6; with two
  devices and `reciprocal = true` the difference is scaled by `2 / sqrt( 2 + 2·c )`, at most about 1.7 for the presets
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The bound
  between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both kernels lose
  accuracy at the same rate, and from 2^53 periods on every position returns the value at the origin
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor on the
  number of threads
- For `dtype = float` the inputs are converted to double, the computation is the same and the result is
  rounded to single precision
- Memory: for `dtype = double` the values are computed in `val`; two devices with `reciprocal = true` need one
  more array of the size of `val`, and `dtype = float` one array of that size in double precision

## Declaration:
```
void quadriga_lib::sos_val(
    const arma::Mat<dtype> &spatial_freq,
    const arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> &pos_b = arma::Mat<dtype>(),
    double spatial_freq_scale_b = 1.0,
    bool reciprocal = true,
    bool uniform = false,
    int use_kernel = 0);
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`spatial_freq_scale_b`** *(optional)* — Spatial-frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0; other than 1 requires `pos_b` and `reciprocal = false`
- **`reciprocal`** *(optional)* — Two devices: `true` = the value does not change when the devices swap positions
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
void quadriga_lib::sos_val(const arma::Mat<dtype> &spatial_freq,
                           const arma::Mat<dtype> &phase,
                           const arma::Mat<dtype> &pos,
                           arma::Mat<dtype> &val,
                           const arma::Mat<dtype> &pos_b,
                           double spatial_freq_scale_b_in,
                           bool reciprocal,
                           bool uniform,
                           int use_kernel)
{
    const dtype spatial_freq_scale_b = dtype(spatial_freq_scale_b_in);
    const bool two_devices = !pos_b.is_empty();

    if (spatial_freq.n_elem == 0)
        throw std::invalid_argument("Input 'spatial_freq' cannot be empty.");
    if (spatial_freq.n_cols > 3)
        throw std::invalid_argument("Input 'spatial_freq' must have 1, 2 or 3 columns.");
    if (phase.n_elem == 0)
        throw std::invalid_argument("Input 'phase' cannot be empty.");
    if (phase.n_rows != spatial_freq.n_rows)
        throw std::invalid_argument("Number of rows in 'phase' does not match number of rows in 'spatial_freq'.");
    if (pos.n_elem == 0)
        throw std::invalid_argument("Input 'pos' cannot be empty.");
    if (pos.n_cols > spatial_freq.n_cols)
        throw std::invalid_argument("Input 'pos' cannot have more columns than 'spatial_freq'.");
    if (two_devices)
    {
        if (pos_b.n_rows != pos.n_rows)
            throw std::invalid_argument("Number of rows in 'pos_b' does not match number of rows in 'pos'.");
        if (pos_b.n_cols > spatial_freq.n_cols)
            throw std::invalid_argument("Input 'pos_b' cannot have more columns than 'spatial_freq'.");
    }
    if (!(spatial_freq_scale_b > (dtype)0.0) || !std::isfinite((double)spatial_freq_scale_b))
        throw std::invalid_argument("Input 'spatial_freq_scale_b' must be finite and > 0.");
    if (spatial_freq_scale_b != (dtype)1.0 && !two_devices)
        throw std::invalid_argument("Input 'spatial_freq_scale_b' other than 1 requires 'pos_b'.");
    if (spatial_freq_scale_b != (dtype)1.0 && reciprocal)
        throw std::invalid_argument("Input 'spatial_freq_scale_b' other than 1 requires 'reciprocal = false'.");

    qd_check_alias(&val, "val", &spatial_freq, &phase, &pos, &pos_b);

    const int kernel = qd_kernel_auto_avx2(use_kernel); // 1 = GENERIC, 2 = AVX2

    const arma::uword n_pos = pos.n_rows, n_gen = phase.n_cols, n_elem = n_pos * n_gen, n_coeff = spatial_freq.n_rows;
    const bool is_double = std::is_same<dtype, double>::value;

    // First device
    std::vector<double> fx, fy, fz, wc, ws, pos_store;
    const double *px = nullptr, *py = nullptr, *pz = nullptr;
    qd_sos_split_freq(spatial_freq, 1.0, fx, fy, fz);
    qd_sos_split_pos(pos, pos_store, px, py, pz);
    qd_sos_weights(phase, wc, ws);

    // The kernels compute in double. For dtype = double they write into the output, for float into a temporary.
    arma::mat tmp;
    double *sa = nullptr;
    if (is_double)
    {
        val.set_size(n_pos, n_gen);
        sa = reinterpret_cast<double *>(val.memptr());
    }
    else
    {
        tmp.set_size(n_pos, n_gen);
        sa = tmp.memptr();
    }

    if (!two_devices)
        qd_sos_run_kernel(kernel, fx, fy, fz, wc, ws, n_gen, px, py, pz, n_pos, sa);
    else
    {
        // Second device: frequencies b·f_p(n) at pos_b; s(pos, pos_b) in sa
        std::vector<double> pos_b_store, gx, gy, gz;
        const double *qx = nullptr, *qy = nullptr, *qz = nullptr;
        qd_sos_split_pos(pos_b, pos_b_store, qx, qy, qz);
        const std::vector<arma::uword> perm = qd_sos_permutation(n_coeff);
        qd_sos_split_freq(spatial_freq, (double)spatial_freq_scale_b, gx, gy, gz, &perm);

        qd_sos_second sec;
        sec.gx = &gx, sec.gy = &gy, sec.gz = &gz, sec.qx = qx, sec.qy = qy, sec.qz = qz;
        qd_sos_run_kernel(kernel, fx, fy, fz, wc, ws, n_gen, px, py, pz, n_pos, sa, sec);

        if (reciprocal)
        {
            // s(pos_b, pos) = f_p(n) at pos and f_n at pos_b (b = 1); the kernel adds the two parts as in the first
            // call with the parts exchanged, so that swapping the devices exchanges the two calls bit for bit
            arma::mat sb_mat(n_pos, n_gen, arma::fill::none);
            const double *sb = sb_mat.memptr();
            qd_sos_second sec2;
            sec2.gx = &fx, sec2.gy = &fy, sec2.gz = &fz, sec2.qx = qx, sec2.qy = qy, sec2.qz = qz;
            qd_sos_run_kernel(kernel, gx, gy, gz, wc, ws, n_gen, px, py, pz, n_pos, sb_mat.memptr(), sec2);

            // Normalization per position, 1 / sqrt( 2 + 2·c ): c is one more generator of the same kernel with the
            // frequencies f_n - f_p(n), weights 1/N and the displacement d = pos - pos_b. c is even in d; the sign of d
            // is fixed (first non-zero coordinate positive), so that swapping the devices gives the same c bit for bit
            std::vector<double> dfx(n_coeff), dfy(n_coeff), dfz(n_coeff), w1(n_coeff, 1.0 / (double)n_coeff), w0(n_coeff, 0.0);
            for (arma::uword n = 0; n < n_coeff; ++n)
                dfx[n] = fx[n] - gx[n], dfy[n] = fy[n] - gy[n], dfz[n] = fz[n] - gz[n];
            std::vector<double> dx(n_pos), dy(n_pos), dz(n_pos), w_vec(n_pos);
            {
                double *pdx = dx.data(), *pdy = dy.data(), *pdz = dz.data();
                qd_sos_parallel_ranges(n_pos, [=](arma::uword i0, arma::uword i1)
                                       {
                    for (arma::uword p = i0; p < i1; ++p)
                    {
                        double a = px[p] - qx[p], b = py[p] - qy[p], c = pz[p] - qz[p];
                        if (a < 0.0 || (a == 0.0 && (b < 0.0 || (b == 0.0 && c < 0.0))))
                            a = -a, b = -b, c = -c;
                        pdx[p] = a, pdy[p] = b, pdz[p] = c;
                    } });
            }
            double *w = w_vec.data();
            qd_sos_run_kernel(kernel, dfx, dfy, dfz, w1, w0, 1, dx.data(), dy.data(), dz.data(), n_pos, w);
            qd_sos_parallel_ranges(n_pos, [=](arma::uword i0, arma::uword i1)
                                   {
                for (arma::uword p = i0; p < i1; ++p)
                {
                    const double v = 2.0 + 2.0 * w[p]; // 0 where c = -1 (tables with few sinusoids): NaN there
                    w[p] = (v > 1.0e-6) ? 1.0 / std::sqrt(v) : std::numeric_limits<double>::quiet_NaN();
                } });
            qd_sos_parallel_ranges(n_elem, [=](arma::uword i0, arma::uword i1)
                                   {
                arma::uword p = i0 % n_pos;
                for (arma::uword i = i0; i < i1; ++i)
                {
                    sa[i] = (sa[i] + sb[i]) * w[p];
                    p = (p + 1 == n_pos) ? 0 : p + 1;
                } });
        }
    }

    // Output: for dtype = double the values are already in place
    if (!is_double)
        val.set_size(n_pos, n_gen);

    dtype *pv = val.memptr();
    if (uniform)
    {
        const double rsqrt2 = std::sqrt(0.5);
        qd_sos_parallel_ranges(n_elem, [=](arma::uword i0, arma::uword i1)
                               {
            for (arma::uword i = i0; i < i1; ++i)
                pv[i] = (dtype)(0.5 * std::erfc(-sa[i] * rsqrt2)); });
    }
    else if (!is_double)
        qd_sos_parallel_ranges(n_elem, [=](arma::uword i0, arma::uword i1)
                               {
            for (arma::uword i = i0; i < i1; ++i)
                pv[i] = (dtype)sa[i]; });
}

template void quadriga_lib::sos_val(const arma::Mat<float> &spatial_freq, const arma::Mat<float> &phase, const arma::Mat<float> &pos, arma::Mat<float> &val,
                                    const arma::Mat<float> &pos_b, double spatial_freq_scale_b, bool reciprocal, bool uniform, int use_kernel);
template void quadriga_lib::sos_val(const arma::Mat<double> &spatial_freq, const arma::Mat<double> &phase, const arma::Mat<double> &pos, arma::Mat<double> &val,
                                    const arma::Mat<double> &pos_b, double spatial_freq_scale_b, bool reciprocal, bool uniform, int use_kernel);
