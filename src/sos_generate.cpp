// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_generate
Find the frequencies of a sum-of-sinusoids generator for a given ACF

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = true` places them regularly (equally spaced azimuth, spiral lattice, as the test directions
  of [[sos_calc_mse]]), `false` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the residual
  between the target and the contribution of all other sinusoids along its own direction: coarse search
  over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The
  new frequency is kept only if the mean squared error over the test directions decreases. The iteration
  stops after a pass that changes nothing or lowers the error by less than a factor of 1e-6, or after
  1000 passes
- `random_init = true` starts from uniformly random frequencies; `false` builds the set up sinusoid by
  sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is not
  controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [[sos_calc_mse]]). To prevent this, set `freq_min`; the tables of [[sos_preset]] use
  `0.05 / d` with the decorrelation distance `d`
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse in
  others; the tables of [[sos_preset]] were made with 300
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each sinusoid;
  0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- The search grid has `( freq_max - freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are stored for
  all sample points. It must not have more than 100000 points (the default has 151), which limits
  `freq_max - freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- `ase` and `test_dir` may be the same objects as `acf` or `dist`
- `test_dir` and the reported error use the same test directions as [[sos_calc_mse]] with the same `n_dir`
- All computations are done in double precision for both instantiations
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject decisions, so a
  difference in the last bit of the input or of the math library can lead to another table of equivalent
  quality

## Declaration:
```
void quadriga_lib::sos_generate(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    arma::uword n_coeff,
    arma::Mat<dtype> &freq,
    arma::Col<dtype> *ase = nullptr,
    arma::Mat<dtype> *test_dir = nullptr,
    arma::uword n_dim = 3,
    bool uniform_smp = true,
    arma::uword n_dir = 10,
    bool random_init = true,
    dtype freq_min = 0.0,
    dtype freq_max = 0.0,
    arma::sword seed = -1);
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_dim`** *(optional)* — Number of dimensions; 1, 2 or 3
- **`uniform_smp`** *(optional)* — Regular (`true`) or random (`false`) directions of the sinusoids
- **`n_dir`** *(optional)* — Number of test directions; must be > 0; ignored for `n_dim = 1`
- **`random_init`** *(optional)* — Start from random frequencies (`true`) or build up sinusoid by sinusoid
  (`false`)
- **`freq_min`** *(optional)* — Lower limit of the frequency magnitude in [1/m]; must be >= 0; 0 = none
- **`freq_max`** *(optional)* — Upper limit of the frequency magnitude in [1/m]; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; limited by the size of the search grid, see above
- **`seed`** *(optional)* — Seed; must be >= -1; `-1` seeds from the system random device

## Outputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, n_dim]`
- **`ase`** *(optional)* — Mean squared error over the test directions after each pass, linear scale;
  `[n_pass]`
- **`test_dir`** *(optional)* — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}`
  for `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

## See also:
- [[sos_calc_mse]] (error of a frequency table)
- [[sos_preset]] (precomputed tables)
MD!*/

template <typename dtype>
void quadriga_lib::sos_generate(const arma::Col<dtype> &acf,
                                const arma::Col<dtype> &dist,
                                arma::uword n_coeff,
                                arma::Mat<dtype> &freq,
                                arma::Col<dtype> *ase,
                                arma::Mat<dtype> *test_dir,
                                arma::uword n_dim,
                                bool uniform_smp,
                                arma::uword n_dir,
                                bool random_init,
                                dtype freq_min,
                                dtype freq_max,
                                arma::sword seed)
{
    qd_sos_check_acf(acf, dist);
    if (n_coeff == 0)
        throw std::invalid_argument("Input 'n_coeff' cannot be 0.");
    if (n_dim < 1 || n_dim > 3)
        throw std::invalid_argument("Input 'n_dim' must be 1, 2 or 3.");
    if (n_dir == 0)
        throw std::invalid_argument("Input 'n_dir' cannot be 0.");
    if (!(freq_min >= (dtype)0.0) || !std::isfinite((double)freq_min))
        throw std::invalid_argument("Input 'freq_min' must be >= 0.");
    if (!(freq_max >= (dtype)0.0) || !std::isfinite((double)freq_max))
        throw std::invalid_argument("Input 'freq_max' must be >= 0.");
    const double d_max = (double)dist.at(dist.n_elem - 1);
    const double f_hi = (freq_max != (dtype)0.0) ? (double)freq_max : 15.0 / d_max, f_lo = (double)freq_min;
    if (!(f_hi > f_lo))
        throw std::invalid_argument("Input 'freq_max' (or its default) must be larger than 'freq_min'.");
    // The cosines of the search grid are stored for all sample points; the presets use 150 grid points
    const double n_grid_d = std::floor((f_hi * d_max - f_lo * d_max) / 0.1 + 1.0e-9) + 1.0;
    if (!(n_grid_d <= 100000.0))
        throw std::invalid_argument("Input 'freq_max' is too large: the search grid of ( freq_max - freq_min ) · max(dist) / 0.1 points cannot exceed 100000.");
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");

    const arma::uword n_acf = acf.n_elem, n_test = (n_dim == 1) ? 1 : n_dir;
    const double two_pi = 6.283185307179586;

    // Internal frequency unit: periods per fitted range, i.e. the model along a direction is
    // (1/n_coeff) · sum_n cos( f_n · dn ) with dn = 2·pi · dist / max(dist)
    const double f_lo_n = f_lo * d_max, f_hi_n = f_hi * d_max;
    std::vector<double> R(n_acf), dn(n_acf);
    for (arma::uword s = 0; s < n_acf; ++s)
        R[s] = (double)acf.at(s), dn[s] = two_pi * (double)dist.at(s) / d_max;

    // out[s] = cos( f · dn[s] ) for all sample points. For equally spaced sample points the cosine is
    // advanced by a plane rotation per sample (error grows by one rounding per sample), which is much
    // faster than one cosine call per sample.
    const bool equal_steps = qd_sos_equally_spaced(dist);
    const double dn_step = two_pi / (double)(n_acf - 1);
    auto cos_series = [&](double f, double *out)
    {
        if (equal_steps)
        {
            const double cr = std::cos(f * dn_step), sr = std::sin(f * dn_step);
            double c = 1.0, sn = 0.0;
            for (arma::uword s = 0; s < n_acf; ++s)
            {
                out[s] = c;
                double c_next = c * cr - sn * sr;
                sn = sn * cr + c * sr, c = c_next;
            }
        }
        else
            for (arma::uword s = 0; s < n_acf; ++s)
                out[s] = std::cos(f * dn[s]);
    };

    std::mt19937_64 rng = qd_sos_rng(seed);

    // Directions of the sinusoids and test directions, unit vectors
    std::vector<double> ux(n_coeff), uy(n_coeff), uz(n_coeff), tx(n_test), ty(n_test), tz(n_test), az(n_test), el(n_test);
    qd_sos_directions(n_test, n_dim, tx.data(), ty.data(), tz.data(), az.data(), el.data());
    if (uniform_smp || n_dim == 1)
        qd_sos_directions(n_coeff, n_dim, ux.data(), uy.data(), uz.data());
    else
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            double a = two_pi * (qd_sos_uniform(rng) - 0.5), z = 0.0;
            if (n_dim == 3)
                z = 2.0 * qd_sos_uniform(rng) - 1.0;
            double r = std::sqrt(1.0 - z * z);
            ux[n] = r * std::cos(a), uy[n] = r * std::sin(a), uz[n] = z;
        }

    // Search grid of the frequency magnitude with a spacing of 0.1 periods, and its cosines, Size [n_acf, n_grid]
    const arma::uword n_grid = (arma::uword)std::floor((f_hi_n - f_lo_n) / 0.1 + 1.0e-9) + 1;
    std::vector<double> grid(n_grid), grid_cos(n_grid * n_acf);
    for (arma::uword k = 0; k < n_grid; ++k)
    {
        grid[k] = f_lo_n + 0.1 * (double)k;
        for (arma::uword s = 0; s < n_acf; ++s)
            grid_cos[k * n_acf + s] = std::cos(grid[k] * dn[s]);
    }

    // Signed frequency of each sinusoid along its direction
    // Without random start values every sinusoid begins at the lowest allowed magnitude
    std::vector<double> fn(n_coeff, f_lo_n);
    if (random_init)
    {
        const double top = std::min(3.141592653589793, f_hi_n);
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            double mag = (top > f_lo_n) ? f_lo_n + qd_sos_uniform(rng) * (top - f_lo_n) : f_lo_n;
            fn[n] = (qd_sos_uniform(rng) < 0.5) ? -mag : mag;
        }
    }

    // Sum over all sinusoids of their ACF along each test direction, Size [n_acf, n_test]
    arma::mat sum_test(n_acf, n_test), term_old(n_acf, n_test), term_new(n_acf, n_test);
    auto test_term = [&](arma::uword n, double f, arma::mat &term)
    {
        for (arma::uword t = 0; t < n_test; ++t)
        {
            cos_series(f * (ux[n] * tx[t] + uy[n] * ty[t] + uz[n] * tz[t]), term.colptr(t));
        }
    };
    auto rebuild_sum = [&]()
    {
        sum_test.zeros();
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            test_term(n, fn[n], term_new);
            sum_test += term_new;
        }
    };
    auto cost_of = [&](const arma::mat &sum) -> double
    {
        double acc = 0.0;
        for (arma::uword t = 0; t < n_test; ++t)
        {
            const double *p = sum.colptr(t);
            for (arma::uword s = 0; s < n_acf; ++s)
            {
                double e = R[s] - p[s] / (double)n_coeff;
                acc += e * e;
            }
        }
        return acc / (double)(n_acf * n_test);
    };

    const double weight = 1.0 / (double)n_coeff;
    std::vector<double> resid(n_acf), series(n_acf), other(n_acf);

    // Squared error of the weighted cosine with frequency f against the residual
    auto fit_error = [&](double f) -> double
    {
        cos_series(f, series.data());
        double acc = 0.0;
        for (arma::uword s = 0; s < n_acf; ++s)
        {
            double e = resid[s] - weight * series[s];
            acc += e * e;
        }
        return acc;
    };

    // The sum is updated incrementally, so that the tracked error never increases
    std::vector<double> ase_list;
    rebuild_sum();
    double cost = random_init ? cost_of(sum_test) : std::numeric_limits<double>::infinity();
    arma::uword n_changed = 1;
    bool converged = false;

    for (arma::uword pass = 0; pass < 1000 && n_changed != 0 && !converged; ++pass)
    {
        n_changed = 0;
        const bool build_up = pass == 0 && !random_init;
        const double cost_start = cost;

        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            // Residual along the direction of sinusoid n. Regular pass: target minus the contribution of all
            // other sinusoids. Build-up pass: target minus the mean of the sinusoids placed so far.
            arma::uword n_other = 0;
            std::fill(other.begin(), other.end(), 0.0);
            for (arma::uword m = 0; m < n_coeff; ++m)
                if (build_up ? (m < n) : (m != n))
                {
                    cos_series(fn[m] * (ux[m] * ux[n] + uy[m] * uy[n] + uz[m] * uz[n]), series.data());
                    for (arma::uword s = 0; s < n_acf; ++s)
                        other[s] += series[s];
                    ++n_other;
                }
            const double other_scale = build_up ? (n_other == 0 ? 0.0 : 1.0 / (double)n_other) : weight;
            for (arma::uword s = 0; s < n_acf; ++s)
                resid[s] = R[s] - other_scale * other[s];

            // Coarse search
            arma::uword k_best = 0;
            double x_best = std::numeric_limits<double>::infinity();
            for (arma::uword k = 0; k < n_grid; ++k)
            {
                const double *gc = &grid_cos[k * n_acf];
                double acc = 0.0;
                for (arma::uword s = 0; s < n_acf; ++s)
                {
                    double e = resid[s] - weight * gc[s];
                    acc += e * e;
                }
                if (acc < x_best)
                    x_best = acc, k_best = k;
            }

            // Step refinement inside the limits
            double f = grid[k_best], x = x_best, step = 0.037;
            while (std::abs(step) > 1.0e-6)
            {
                double f_try = std::min(std::max(f + step, f_lo_n), f_hi_n);
                double x_try = fit_error(f_try);
                if (x_try < x)
                    x = x_try, f = f_try;
                else
                    step *= -0.21;
            }

            // A cosine does not have a sign of its frequency; it is drawn at random
            if (qd_sos_uniform(rng) < 0.5)
                f = -f;

            // Keep the new frequency only if the error over the test directions decreases
            test_term(n, fn[n], term_old);
            test_term(n, f, term_new);
            arma::mat sum_try = sum_test - term_old + term_new;
            double cost_try = cost_of(sum_try);
            if (cost_try < cost)
            {
                cost = cost_try, fn[n] = f, sum_test = sum_try;
                ++n_changed;
            }
        }
        ase_list.push_back(cost);
        converged = cost_start - cost < 1.0e-6 * cost_start;
    }

    if (freq.n_rows != n_coeff || freq.n_cols != n_dim)
        freq.set_size(n_coeff, n_dim);
    for (arma::uword n = 0; n < n_coeff; ++n)
    {
        freq.at(n, 0) = (dtype)(fn[n] * ux[n] / d_max);
        if (n_dim > 1)
            freq.at(n, 1) = (dtype)(fn[n] * uy[n] / d_max);
        if (n_dim > 2)
            freq.at(n, 2) = (dtype)(fn[n] * uz[n] / d_max);
    }

    if (ase != nullptr)
    {
        if (ase->n_elem != ase_list.size())
            ase->set_size(ase_list.size());
        for (arma::uword i = 0; i < ase_list.size(); ++i)
            ase->at(i) = (dtype)ase_list[i];
    }

    if (test_dir != nullptr)
    {
        arma::uword nr = (n_dim == 1) ? 0 : n_test, nc = (n_dim == 1) ? 0 : n_dim - 1;
        if (test_dir->n_rows != nr || test_dir->n_cols != nc)
            test_dir->set_size(nr, nc);
        for (arma::uword t = 0; t < nr; ++t)
        {
            test_dir->at(t, 0) = (dtype)az[t];
            if (nc > 1)
                test_dir->at(t, 1) = (dtype)el[t];
        }
    }
}

template void quadriga_lib::sos_generate(const arma::Col<float> &acf, const arma::Col<float> &dist, arma::uword n_coeff, arma::Mat<float> &freq, arma::Col<float> *ase,
                                         arma::Mat<float> *test_dir, arma::uword n_dim, bool uniform_smp, arma::uword n_dir, bool random_init,
                                         float freq_min, float freq_max, arma::sword seed);
template void quadriga_lib::sos_generate(const arma::Col<double> &acf, const arma::Col<double> &dist, arma::uword n_coeff, arma::Mat<double> &freq, arma::Col<double> *ase,
                                         arma::Mat<double> *test_dir, arma::uword n_dim, bool uniform_smp, arma::uword n_dir, bool random_init,
                                         double freq_min, double freq_max, arma::sword seed);
