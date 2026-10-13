// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// The output as a double matrix if it is one, nullptr otherwise
static inline arma::mat *qd_sos_double_output(arma::Mat<double> &val) { return &val; }
static inline arma::mat *qd_sos_double_output(arma::Mat<float> &) { return nullptr; }

// Common implementation of sos_randn, sos_rand and sos_randi
template <typename dtype>
static void qd_sos_rand_impl(const arma::Col<dtype> &dist_decorr, const arma::Mat<dtype> &pos, arma::Mat<dtype> &val,
                             const arma::Mat<dtype> *pos_b, const arma::Col<dtype> *dist_decorr_b,
                             const std::string &preset, arma::sword seed, bool uniform, arma::uword imax)
{
    const bool two_devices = pos_b != nullptr && !pos_b->is_empty();
    const bool has_decorr_b = dist_decorr_b != nullptr && !dist_decorr_b->is_empty();

    if (dist_decorr.n_elem == 0)
        throw std::invalid_argument("Input 'dist_decorr' cannot be empty.");
    for (arma::uword g = 0; g < dist_decorr.n_elem; ++g)
        if (!(dist_decorr.at(g) >= (dtype)0.0) || !std::isfinite((double)dist_decorr.at(g)))
            throw std::invalid_argument("Input 'dist_decorr' must be >= 0.");
    if (pos.n_elem == 0)
        throw std::invalid_argument("Input 'pos' cannot be empty.");
    if (pos.n_cols > 3)
        throw std::invalid_argument("Input 'pos' must have 1, 2 or 3 columns.");
    if (two_devices)
    {
        if (pos_b->n_rows != pos.n_rows)
            throw std::invalid_argument("Number of rows in 'pos_b' does not match number of rows in 'pos'.");
        if (pos_b->n_cols > 3)
            throw std::invalid_argument("Input 'pos_b' must have 1, 2 or 3 columns.");
    }
    if (has_decorr_b)
    {
        if (!two_devices)
            throw std::invalid_argument("Input 'dist_decorr_b' requires 'pos_b'.");
        if (dist_decorr_b->n_elem != dist_decorr.n_elem)
            throw std::invalid_argument("Number of elements in 'dist_decorr_b' does not match number of elements in 'dist_decorr'.");
        for (arma::uword g = 0; g < dist_decorr_b->n_elem; ++g)
            if (!(dist_decorr_b->at(g) >= (dtype)0.0) || !std::isfinite((double)dist_decorr_b->at(g)))
                throw std::invalid_argument("Input 'dist_decorr_b' must be >= 0.");
    }
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");

    const arma::uword n_gen = dist_decorr.n_elem, n_pos = pos.n_rows;

    // Preset for 10 m (throws for an unknown name) and the phases of the first device
    arma::mat freq10, phase, phase_b;
    arma::vec acf, dist10;
    quadriga_lib::sos_preset<double>(preset, freq10, &acf, &dist10, 10.0);
    quadriga_lib::sos_init_phases<double>(freq10.n_rows, n_gen, phase, seed);

    const arma::mat pos_d = arma::conv_to<arma::mat>::from(pos);
    arma::mat pos_b_d;
    if (two_devices)
        pos_b_d = arma::conv_to<arma::mat>::from(*pos_b);

    // Decorrelation distances of both devices, 0 replaced by 0.1 m
    std::vector<double> da(n_gen), db(n_gen);
    for (arma::uword g = 0; g < n_gen; ++g)
    {
        da[g] = (double)dist_decorr.at(g);
        db[g] = has_decorr_b ? (double)dist_decorr_b->at(g) : da[g];
        da[g] = (da[g] == 0.0) ? 0.1 : da[g], db[g] = (db[g] == 0.0) ? 0.1 : db[g];
    }

    // Phases of a second device on its own map: a second random stream, so that column g does not depend on
    // the number of generators for either device. Its seed is the seed with bit 62 flipped (seed + 2^62 for
    // seeds below 2^62), which is never negative and never equal to the seed.
    bool any_two_maps = false;
    for (arma::uword g = 0; g < n_gen; ++g)
        any_two_maps = any_two_maps || (two_devices && da[g] != db[g]);
    if (any_two_maps)
    {
        const arma::sword seed_b = (seed < 0) ? -1 : (arma::sword)((unsigned long long)seed ^ (1ULL << 62));
        quadriga_lib::sos_init_phases<double>(freq10.n_rows, n_gen, phase_b, seed_b);
    }

    if (val.n_rows != n_pos || val.n_cols != n_gen)
        val.set_size(n_pos, n_gen);

    // Generators with the same pair of decorrelation distances share the frequencies and are evaluated together
    std::vector<bool> done(n_gen, false);
    for (arma::uword g0 = 0; g0 < n_gen; ++g0)
    {
        if (done[g0])
            continue;

        std::vector<arma::uword> group;
        for (arma::uword g = g0; g < n_gen; ++g)
            if (!done[g] && da[g] == da[g0] && db[g] == db[g0])
                group.push_back(g), done[g] = true;

        const arma::uword n_grp = group.size();
        const bool two_maps = two_devices && da[g0] != db[g0];
        arma::mat ph_a(freq10.n_rows, n_grp), ph_b;
        for (arma::uword k = 0; k < n_grp; ++k)
            ph_a.col(k) = phase.col(group[k]);
        if (two_maps)
        {
            ph_b.set_size(freq10.n_rows, n_grp);
            for (arma::uword k = 0; k < n_grp; ++k)
                ph_b.col(k) = phase_b.col(group[k]);
        }

        // A group that holds all generators is evaluated into the output if that is a double matrix
        arma::mat v_local, *v_out = (n_grp == n_gen) ? qd_sos_double_output(val) : nullptr;
        arma::mat &v = (v_out != nullptr) ? *v_out : v_local;

        const arma::mat freq = freq10 * (10.0 / da[g0]);
        if (!two_devices)
            quadriga_lib::sos_val<double>(freq, ph_a, pos_d, v, nullptr, nullptr, nullptr, nullptr, 1.0, uniform, 0);
        else if (!two_maps)
        {
            const arma::vec dist = dist10 * (da[g0] / 10.0);
            quadriga_lib::sos_val<double>(freq, ph_a, pos_d, v, &pos_b_d, nullptr, &acf, &dist, 1.0, uniform, 0);
        }
        else
            quadriga_lib::sos_val<double>(freq, ph_a, pos_d, v, &pos_b_d, &ph_b, nullptr, nullptr, da[g0] / db[g0], uniform, 0);

        if (v_out != nullptr && imax == 0)
            continue;

        const long long n_pos_ll = (long long)n_pos;
        for (arma::uword k = 0; k < n_grp; ++k)
        {
            const double *src = v.colptr(k);
            dtype *dst = val.colptr(group[k]);
            if (imax == 0)
            {
#pragma omp parallel for if (n_pos_ll > 65536)
                for (long long p = 0; p < n_pos_ll; ++p)
                    dst[p] = (dtype)src[p];
            }
            else
            {
#pragma omp parallel for if (n_pos_ll > 65536)
                for (long long p = 0; p < n_pos_ll; ++p)
                {
                    double i = std::ceil(src[p] * (double)imax);
                    dst[p] = (dtype)(i < 1.0 ? 1.0 : (i > (double)imax ? (double)imax : i));
                }
            }
        }
    }
}

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_randn
Spatially correlated Normal distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [[sos_val]] for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [[sos_val]]
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

## Declaration:
```
void quadriga_lib::sos_randn(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

## Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [[sos_preset]]
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

## Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row per
  position, one column per generator; `[n_pos, n_gen]`

## See also:
- [[sos_val]] (evaluation with given frequencies and phases)
- [[sos_preset]], [[sos_init_phases]] (the two steps this function combines)
- [[sos_rand]], [[sos_randi]] (Uniform and integer values)
MD!*/

template <typename dtype>
void quadriga_lib::sos_randn(const arma::Col<dtype> &dist_decorr,
                             const arma::Mat<dtype> &pos,
                             arma::Mat<dtype> &val,
                             const arma::Mat<dtype> *pos_b,
                             const arma::Col<dtype> *dist_decorr_b,
                             const std::string &preset,
                             arma::sword seed)
{
    qd_sos_rand_impl(dist_decorr, pos, val, pos_b, dist_decorr_b, preset, seed, false, 0);
}

/*!MD
# sos_rand
Spatially correlated Uniform distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [[sos_val]] for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [[sos_val]]
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

## Declaration:
```
void quadriga_lib::sos_rand(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

## Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [[sos_preset]]
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

## Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `[n_pos, n_gen]`

## See also:
- [[sos_val]] (evaluation with given frequencies and phases)
- [[sos_preset]], [[sos_init_phases]] (the two steps this function combines)
- [[sos_randn]], [[sos_randi]] (Normal and integer values)
MD!*/

template <typename dtype>
void quadriga_lib::sos_rand(const arma::Col<dtype> &dist_decorr,
                            const arma::Mat<dtype> &pos,
                            arma::Mat<dtype> &val,
                            const arma::Mat<dtype> *pos_b,
                            const arma::Col<dtype> *dist_decorr_b,
                            const std::string &preset,
                            arma::sword seed)
{
    qd_sos_rand_impl(dist_decorr, pos, val, pos_b, dist_decorr_b, preset, seed, true, 0);
}

/*!MD
# sos_randi
Spatially correlated integer random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [[sos_val]] for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- The Uniform value `u` of [[sos_rand]] is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [[sos_val]], and
  an integer can then differ by 1 where the Uniform value lies at a step
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

## Declaration:
```
void quadriga_lib::sos_randi(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::uword imax,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

## Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`imax`** — Largest value; must be > 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [[sos_preset]]
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

## Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability; one row per
  position, one column per generator; `[n_pos, n_gen]`

## See also:
- [[sos_val]] (evaluation with given frequencies and phases)
- [[sos_preset]], [[sos_init_phases]] (the two steps this function combines)
- [[sos_randn]], [[sos_rand]] (Normal and Uniform values)
MD!*/

template <typename dtype>
void quadriga_lib::sos_randi(const arma::Col<dtype> &dist_decorr,
                             const arma::Mat<dtype> &pos,
                             arma::uword imax,
                             arma::Mat<dtype> &val,
                             const arma::Mat<dtype> *pos_b,
                             const arma::Col<dtype> *dist_decorr_b,
                             const std::string &preset,
                             arma::sword seed)
{
    if (imax == 0)
        throw std::invalid_argument("Input 'imax' cannot be 0.");
    qd_sos_rand_impl(dist_decorr, pos, val, pos_b, dist_decorr_b, preset, seed, true, imax);
}

template void quadriga_lib::sos_randn(const arma::Col<float> &dist_decorr, const arma::Mat<float> &pos, arma::Mat<float> &val, const arma::Mat<float> *pos_b,
                                      const arma::Col<float> *dist_decorr_b, const std::string &preset, arma::sword seed);
template void quadriga_lib::sos_randn(const arma::Col<double> &dist_decorr, const arma::Mat<double> &pos, arma::Mat<double> &val, const arma::Mat<double> *pos_b,
                                      const arma::Col<double> *dist_decorr_b, const std::string &preset, arma::sword seed);
template void quadriga_lib::sos_rand(const arma::Col<float> &dist_decorr, const arma::Mat<float> &pos, arma::Mat<float> &val, const arma::Mat<float> *pos_b,
                                     const arma::Col<float> *dist_decorr_b, const std::string &preset, arma::sword seed);
template void quadriga_lib::sos_rand(const arma::Col<double> &dist_decorr, const arma::Mat<double> &pos, arma::Mat<double> &val, const arma::Mat<double> *pos_b,
                                     const arma::Col<double> *dist_decorr_b, const std::string &preset, arma::sword seed);
template void quadriga_lib::sos_randi(const arma::Col<float> &dist_decorr, const arma::Mat<float> &pos, arma::uword imax, arma::Mat<float> &val, const arma::Mat<float> *pos_b,
                                      const arma::Col<float> *dist_decorr_b, const std::string &preset, arma::sword seed);
template void quadriga_lib::sos_randi(const arma::Col<double> &dist_decorr, const arma::Mat<double> &pos, arma::uword imax, arma::Mat<double> &val, const arma::Mat<double> *pos_b,
                                      const arma::Col<double> *dist_decorr_b, const std::string &preset, arma::sword seed);
