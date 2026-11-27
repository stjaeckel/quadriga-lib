// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "qd_helpers_shared.hpp"

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
- The number of grid vectors given must not exceed the number of columns of `spatial_freq`; an omitted or empty
  `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `spatial_freq` has 3 columns
- The values are those of [[sos_val]] at the grid points, bit for bit; kernels, accuracy and the Uniform
  mapping are described there
- Cubes of the correct size keep their memory. If they lie one after the other in memory (one block of
  `n_gen` cubes, as the MATLAB and Python interfaces allocate it), the values are computed in place;
  otherwise they are computed in a matrix `[ny·nx·nz, n_gen]` and copied

## Declaration:
```
void quadriga_lib::sos_map(
    const arma::Mat<dtype> &spatial_freq,
    const arma::Mat<dtype> &phase,
    const double_list &x,
    std::vector<arma::Cube<dtype>> &map,
    const double_list &y = {},
    const double_list &z = {},
    bool uniform = false,
    int use_kernel = 0);
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
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
void quadriga_lib::sos_map(const arma::Mat<dtype> &spatial_freq,
                           const arma::Mat<dtype> &phase,
                           const double_list &x_in,
                           std::vector<arma::Cube<dtype>> &map,
                           const double_list &y_in,
                           const double_list &z_in,
                           bool uniform,
                           int use_kernel)
{
    const arma::Col<dtype> x = x_in.as<dtype>();
    const arma::Col<dtype> y = y_in.as<dtype>();
    const arma::Col<dtype> z = z_in.as<dtype>();
    const bool has_y = !y.is_empty();
    const bool has_z = !z.is_empty();

    if (spatial_freq.n_elem == 0)
        throw std::invalid_argument("Input 'spatial_freq' cannot be empty.");
    if (spatial_freq.n_cols > 3)
        throw std::invalid_argument("Input 'spatial_freq' must have 1, 2 or 3 columns.");
    if (phase.n_elem == 0)
        throw std::invalid_argument("Input 'phase' cannot be empty.");
    if (phase.n_rows != spatial_freq.n_rows)
        throw std::invalid_argument("Number of rows in 'phase' does not match number of rows in 'spatial_freq'.");
    if (x.n_elem == 0)
        throw std::invalid_argument("Input 'x' cannot be empty.");
    if (has_y && spatial_freq.n_cols < 2)
        throw std::invalid_argument("Input 'y' requires 'spatial_freq' to have at least 2 columns.");
    if (has_z && spatial_freq.n_cols < 3)
        throw std::invalid_argument("Input 'z' requires 'spatial_freq' to have 3 columns.");
    qd_kernel_auto_avx2(use_kernel); // Checks the selector here; sos_val selects the kernel

    const arma::uword nx = x.n_elem, ny = has_y ? y.n_elem : 1, nz = has_z ? z.n_elem : 1, n_gen = phase.n_cols;

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
                    pos.at(i, 1) = has_y ? y.at(iy) : (dtype)0.0;
                if (n_dim_pos > 2)
                    pos.at(i, 2) = z.at(iz);
            }

    for (const auto &cube : map)
        qd_check_alias(&cube, "map", &spatial_freq, &phase); // The grids are copies, an output cannot overwrite them

    if (map.size() != n_gen)
        map.resize(n_gen);
    for (arma::uword g = 0; g < n_gen; ++g)
        map[g].set_size(ny, nx, nz);

    // Cubes that lie one after the other in memory (a single cube, or a 4-D array that a wrapper allocated
    // as one block) are the columns of one matrix [n_pos, n_gen]: the values are computed in place.
    bool one_block = true;
    for (arma::uword g = 1; g < n_gen; ++g)
        one_block = one_block && (std::uintptr_t)map[g].memptr() == (std::uintptr_t)map[0].memptr() + g * n_pos * sizeof(dtype);

    if (one_block)
    {
        arma::Mat<dtype> val(map[0].memptr(), n_pos, n_gen, false, true);
        quadriga_lib::sos_val<dtype>(spatial_freq, phase, pos, val, {}, {}, {}, {}, (dtype)1.0, uniform, use_kernel);
    }
    else
    {
        arma::Mat<dtype> val;
        quadriga_lib::sos_val<dtype>(spatial_freq, phase, pos, val, {}, {}, {}, {}, (dtype)1.0, uniform, use_kernel);
        const long long n_gen_ll = (long long)n_gen;
#pragma omp parallel for if (n_pos * n_gen > 65536)
        for (long long g = 0; g < n_gen_ll; ++g)
            std::memcpy(map[(arma::uword)g].memptr(), val.colptr((arma::uword)g), n_pos * sizeof(dtype));
    }
}

template void quadriga_lib::sos_map(const arma::Mat<float> &spatial_freq, const arma::Mat<float> &phase, const double_list &x, std::vector<arma::Cube<float>> &map,
                                    const double_list &y, const double_list &z, bool uniform, int use_kernel);
template void quadriga_lib::sos_map(const arma::Mat<double> &spatial_freq, const arma::Mat<double> &phase, const double_list &x, std::vector<arma::Cube<double>> &map,
                                    const double_list &y, const double_list &z, bool uniform, int use_kernel);
