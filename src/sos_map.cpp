// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_map
Spatially correlated random values on a regular grid

- Evaluates the generators of [[sos_val]] for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz)` of a cube belongs to the position `{x(ix), y(iy), z(iz)}`
- One cube per generator, i.e. per column of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; an omitted or empty
  `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [[sos_val]] at the grid points, bit for bit; kernels, accuracy and the Uniform
  mapping are described there
- Cubes of the correct size keep their memory. If they lie one after the other in memory (one block of
  `n_gen` cubes, as the MATLAB and Python interfaces allocate it), the values are computed in place;
  otherwise they are computed in a matrix `[ny·nx·nz, n_gen]` and copied

## Declaration:
```
void quadriga_lib::sos_map(
    const arma::Mat<dtype> &freq,
    const arma::Mat<dtype> &phase,
    const arma::Col<dtype> &x,
    std::vector<arma::Cube<dtype>> &map,
    const arma::Col<dtype> *y = nullptr,
    const arma::Col<dtype> *z = nullptr,
    bool uniform = false,
    int use_kernel = 0);
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`x`** — Grid points on the x-axis in [m]; `[nx]`
- **`y`** *(optional)* — Grid points on the y-axis in [m]; requires `n_dim >= 2`; omitted = one sample at 0,
  `ny = 1`; `[ny]`
- **`z`** *(optional)* — Grid points on the z-axis in [m]; requires `n_dim = 3`; omitted = one sample at 0,
  `nz = 1`; `[nz]`
- **`uniform`** *(optional)* — Output distribution: `false` = Normal(0,1), `true` = Uniform between 0 and 1
- **`use_kernel`** *(optional)* — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  throws if the requested kernel is unavailable

## Output:
- **`map`** — Values; vector of length `n_gen` with one cube of size `[ny, nx, nz]` per generator

## See also:
- [[sos_val]] (evaluation at arbitrary positions, two devices)
MD!*/

template <typename dtype>
void quadriga_lib::sos_map(const arma::Mat<dtype> &freq,
                           const arma::Mat<dtype> &phase,
                           const arma::Col<dtype> &x,
                           std::vector<arma::Cube<dtype>> &map,
                           const arma::Col<dtype> *y,
                           const arma::Col<dtype> *z,
                           bool uniform,
                           int use_kernel)
{
    const bool has_y = y != nullptr && !y->is_empty();
    const bool has_z = z != nullptr && !z->is_empty();

    if (freq.n_elem == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (freq.n_cols > 3)
        throw std::invalid_argument("Input 'freq' must have 1, 2 or 3 columns.");
    if (phase.n_elem == 0)
        throw std::invalid_argument("Input 'phase' cannot be empty.");
    if (phase.n_rows != freq.n_rows)
        throw std::invalid_argument("Number of rows in 'phase' does not match number of rows in 'freq'.");
    if (x.n_elem == 0)
        throw std::invalid_argument("Input 'x' cannot be empty.");
    if (has_y && freq.n_cols < 2)
        throw std::invalid_argument("Input 'y' requires 'freq' to have at least 2 columns.");
    if (has_z && freq.n_cols < 3)
        throw std::invalid_argument("Input 'z' requires 'freq' to have 3 columns.");
    if (use_kernel < 0 || use_kernel > 2)
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1 or 2.");
    if (use_kernel == 2 && !quadriga_lib::quadriga_lib_has_AVX2())
        throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");

    const arma::uword nx = x.n_elem, ny = has_y ? y->n_elem : 1, nz = has_z ? z->n_elem : 1, n_gen = phase.n_cols;

    // Positions in meshgrid order: iy runs fastest, then ix, then iz
    const arma::uword n_pos = nx * ny * nz, n_dim_pos = has_z ? 3 : (has_y ? 2 : 1);
    arma::Mat<dtype> pos(n_pos, n_dim_pos, arma::fill::none);
    for (arma::uword iz = 0; iz < nz; ++iz)
        for (arma::uword ix = 0; ix < nx; ++ix)
            for (arma::uword iy = 0; iy < ny; ++iy)
            {
                arma::uword i = iy + ny * (ix + nx * iz);
                pos.at(i, 0) = x.at(ix);
                if (n_dim_pos > 1)
                    pos.at(i, 1) = has_y ? y->at(iy) : (dtype)0.0;
                if (n_dim_pos > 2)
                    pos.at(i, 2) = z->at(iz);
            }

    if (map.size() != n_gen)
        map.resize(n_gen);
    for (arma::uword g = 0; g < n_gen; ++g)
        if (map[g].n_rows != ny || map[g].n_cols != nx || map[g].n_slices != nz)
            map[g].set_size(ny, nx, nz);

    // Cubes that lie one after the other in memory (a single cube, or a 4-D array that a wrapper allocated
    // as one block) are the columns of one matrix [n_pos, n_gen]: the values are computed in place.
    bool one_block = true;
    for (arma::uword g = 1; g < n_gen; ++g)
        one_block = one_block && (std::uintptr_t)map[g].memptr() == (std::uintptr_t)map[0].memptr() + g * n_pos * sizeof(dtype);

    if (one_block)
    {
        arma::Mat<dtype> val(map[0].memptr(), n_pos, n_gen, false, true);
        quadriga_lib::sos_val<dtype>(freq, phase, pos, val, nullptr, nullptr, nullptr, nullptr, (dtype)1.0, uniform, use_kernel);
    }
    else
    {
        arma::Mat<dtype> val;
        quadriga_lib::sos_val<dtype>(freq, phase, pos, val, nullptr, nullptr, nullptr, nullptr, (dtype)1.0, uniform, use_kernel);
        const long long n_gen_ll = (long long)n_gen;
#pragma omp parallel for if (n_pos * n_gen > 65536)
        for (long long g = 0; g < n_gen_ll; ++g)
            std::memcpy(map[(arma::uword)g].memptr(), val.colptr((arma::uword)g), n_pos * sizeof(dtype));
    }
}

template void quadriga_lib::sos_map(const arma::Mat<float> &freq, const arma::Mat<float> &phase, const arma::Col<float> &x, std::vector<arma::Cube<float>> &map,
                                    const arma::Col<float> *y, const arma::Col<float> *z, bool uniform, int use_kernel);
template void quadriga_lib::sos_map(const arma::Mat<double> &freq, const arma::Mat<double> &phase, const arma::Col<double> &x, std::vector<arma::Cube<double>> &map,
                                    const arma::Col<double> *y, const arma::Col<double> *z, bool uniform, int use_kernel);
