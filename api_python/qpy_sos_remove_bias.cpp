// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF)
  over all pairs of positions, which is much larger than `1 / n_pos` when the positions lie within
  a few decorrelation distances of each other. This is a property of any correlated field, not an
  error of the generator
- This function adds one offset to all phases of a generator such that the mean of the Normal
  distributed values of [[sos_val]] over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`,
  which vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the
  smaller magnitude is applied, so the offset lies between -pi/2 and pi/2
- One device (`pos_b` omitted): one offset per generator
- Two devices on the same map (`pos_b` given, `phase_b_in` omitted): one offset per generator that
  zeroes the mean of the combined, normalised value; `acf` and `dist` are required
- Two devices on different maps (`pos_b` and `phase_b_in` given): each device is treated on its
  own, `phase_in` with `pos` and `phase_b_in` with `pos_b` and the frequencies
  `freq · freq_scale_b`; two offsets per generator. The mean of the combined value is then zero
  as well
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions.
  The variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the
  sample mean and `c` the mean ACF between that position and all positions; on average over the
  positions the loss is at least `v`. It is small only if the positions spread over many
  decorrelation distances. For positions within about one decorrelation distance most of the
  variance is removed
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are
  used with other frequencies (another decorrelation distance) or positions
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation
  distance `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for
  `L = 5·d`, 5 % for `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose
  `abs(Z)` is below `1e-14 · n_coeff` already has a zero mean for every offset and is returned
  unchanged, also if its phases lie outside that interval
- Two devices on the same map need an ACF above -1 at all sample points, as in [[sos_val]]
- The input arrays `phase_in` and `phase_b_in` are not modified; the corrected phases are new arrays
- An optional array that is `None` or empty counts as not given

## Usage:
```
phase_out, phase_b_out, offset, mean_before = quadriga_lib.sos.remove_bias( freq, phase_in, pos, \
    pos_b, phase_b_in, acf, dist, freq_scale_b )
```

## Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase_in`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; not more columns than `freq`; a 1-D array is one column,
  i.e. one x-coordinate per position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos_b, 1)`, `(n_pos_b, 2)`
  or `(n_pos_b, 3)` as for `pos`; `n_pos_b = n_pos` for the same map, any number for different
  maps; default: `None`
- **`phase_b_in`** — Phases of the second device; requires `pos_b`; `None` = same map;
  `(n_coeff, n_gen)`; default: `None`
- **`acf`** — Target ACF; required for two devices on the same map, must then be > -1; `(n_acf,)`;
  default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`;
  default: `None`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `phase_b_in`;
  default: 1.0

## Outputs:
- **`phase_out`** — Corrected phases; `(n_coeff, n_gen)`
- **`phase_b_out`** — Corrected phases of the second device; `(n_coeff, n_gen)`; `None` if
  `phase_b_in` was not given
- **`offset`** — Offsets added to the phases; `(n_gen, 1)`, or `(n_gen, 2)` with
  `phase_b_in` (column 0: `phase_in`, column 1: `phase_b_in`)
- **`mean_before`** — Sample mean before the correction; `(n_gen, 1)`: mean of the values of
  [[sos_val]] for the given arguments; with `phase_b_in` `(n_gen, 2)`: mean of the field of each
  device on its own positions, the mean of the combined value being their sum divided by `sqrt(2)`

## See also:
- [[sos_val]] (the values whose mean is removed)
MD!*/

py::tuple sos_remove_bias(const py::array_t<double> &freq,
                          const py::array_t<double> &phase_in,
                          const py::array_t<double> &pos,
                          py::handle pos_b,
                          py::handle phase_b_in,
                          py::handle acf,
                          py::handle dist,
                          double freq_scale_b)
{
    // Read inputs; the phases are modified by the library and therefore copied
    const auto freq_a = qd_python_numpy2arma_Mat<double>(freq, true);
    arma::mat phase_out = qd_python_numpy2arma_Mat<double>(phase_in, false);
    const auto pos_a = qd_python_numpy2arma_Mat<double>(pos, true);
    const auto pos_b_a = qd_python_numpy2arma_Mat<double>(pos_b, true);
    arma::mat phase_b_out = qd_python_numpy2arma_Mat<double>(phase_b_in, false);
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Wrap optional pointers
    arma::mat offset, mean_before;
    const arma::mat *p_pos_b = pos_b_a.is_empty() ? nullptr : &pos_b_a;
    arma::mat *p_phase_b = phase_b_out.is_empty() ? nullptr : &phase_b_out;
    const arma::vec *p_acf = acf_a.is_empty() ? nullptr : &acf_a;
    const arma::vec *p_dist = dist_a.is_empty() ? nullptr : &dist_a;

    // Call library function
    quadriga_lib::sos_remove_bias<double>(freq_a, phase_out, pos_a, p_pos_b, p_phase_b, p_acf, p_dist,
                                          &offset, &mean_before, freq_scale_b);

    // Copy to python
    auto phase_out_py = qd_python_copy2numpy(&phase_out);
    py::object phase_b_out_py = py::none();
    if (p_phase_b != nullptr)
        phase_b_out_py = qd_python_copy2numpy(&phase_b_out);
    auto offset_py = qd_python_copy2numpy(&offset);
    auto mean_before_py = qd_python_copy2numpy(&mean_before);

    return py::make_tuple(phase_out_py, phase_b_out_py, offset_py, mean_before_py);
}

// pybind11 declaration:
// m.def("remove_bias", &sos_remove_bias,
//       py::arg("freq").none(false),
//       py::arg("phase_in").none(false),
//       py::arg("pos").none(false),
//       py::arg("pos_b") = py::none(),
//       py::arg("phase_b_in") = py::none(),
//       py::arg("acf") = py::none(),
//       py::arg("dist") = py::none(),
//       py::arg("freq_scale_b") = 1.0);
