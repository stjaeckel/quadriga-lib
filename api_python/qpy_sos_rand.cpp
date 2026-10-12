// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_rand
Spatially correlated Uniform distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [[sos_val]] for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [[sos_val]]. The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

## Usage:
```
val = quadriga_lib.sos.rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed )
```

## Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must be
  >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [[sos_preset]]; `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

## Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `(n_pos, n_gen)`

## See also:
- [[sos_val]] (evaluation with given frequencies and phases)
- [[sos_preset]], [[sos_init_phases]] (the two steps this function combines)
- [[sos_randn]], [[sos_randi]] (Normal and integer values)
MD!*/

py::array_t<double> sos_rand(const py::array_t<double> &dist_decorr,
                             const py::array_t<double> &pos,
                             py::handle pos_b,
                             py::handle dist_decorr_b,
                             const std::string &preset,
                             arma::sword seed)
{
    // Read inputs
    const auto dist_decorr_a = qd_python_numpy2arma_Col<double>(dist_decorr, true);
    const auto pos_a = qd_python_numpy2arma_Mat<double>(pos, true);
    const auto pos_b_a = qd_python_numpy2arma_Mat<double>(pos_b, true);
    const auto dist_decorr_b_a = qd_python_numpy2arma_Col<double>(dist_decorr_b, true);

    // Output allocation
    arma::mat val;
    auto val_py = qd_python_init_output(pos_a.n_rows, dist_decorr_a.n_elem, &val);

    // Wrap optional pointers
    const arma::mat *p_pos_b = pos_b_a.is_empty() ? nullptr : &pos_b_a;
    const arma::vec *p_dist_decorr_b = dist_decorr_b_a.is_empty() ? nullptr : &dist_decorr_b_a;

    // Call library function
    quadriga_lib::sos_rand<double>(dist_decorr_a, pos_a, val, p_pos_b, p_dist_decorr_b, preset, seed);

    return val_py;
}

// pybind11 declaration:
// m.def("rand", &sos_rand,
//       py::arg("dist_decorr").none(false),
//       py::arg("pos").none(false),
//       py::arg("pos_b") = py::none(),
//       py::arg("dist_decorr_b") = py::none(),
//       py::arg("preset") = "Comb256",
//       py::arg("seed") = -1);
