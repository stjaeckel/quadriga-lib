// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α and δ^α,
  where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband
  (incoherent power preservation)
- If all fractional per-tap offsets are below 0.01 or above 0.99, weight computation is skipped
  (nearest-neighbor selection) but tap-selection logic still applies
- Snapshots are passed as Python lists; each list item is one snapshot and `n_path` may vary per item
- Coefficients are given either as one complex list `coeff`, or as split `coeff_re` + `coeff_im`;
  supplying both forms is an error. The input and output forms are independent
- `complex=True` returns one combined complex coefficient output; `complex=False` (default) returns
  separate real `coeff_re_q` and `coeff_im_q`
- Input `delay` items may be per-antenna `(n_rx, n_tx, n_path)` or shared `(1, 1, n_path)`;
  shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`

## Usage:
```
# Split real / imaginary output
coeff_re_q, coeff_im_q, delay_q = quadriga_lib.channel.quantize_delays( coeff_re, coeff_im, delay )

# Complex input and / or output (keyword)
coeff_q, delay_q = quadriga_lib.channel.quantize_delays( coeff=coeff, delay=delay, complex=True )
```

## Inputs:
- **`coeff_re`** — Real part of channel coefficients; list of length `n_snap`; each item
  `(n_rx, n_tx, n_path)`; must be paired with `coeff_im`; mutually exclusive with `coeff`;
  default: `None`
- **`coeff_im`** — Imaginary part of channel coefficients; same layout as `coeff_re`; default: `None`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or
  shared `(1, 1, n_path)`
- **`tap_spacing`** — Delay bin spacing in seconds; 5 ns corresponds to 200 MHz sampling rate;
  default: 5e-9
- **`max_no_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; 0 = per tx-rx pair and snapshot, 1 = single shared grid,
  2 = per snapshot, 3 = per tx-rx pair across all snapshots; default: 0
- **`stack`** — If `True`, stack snapshots into a 4D array; if `False`, return a list of per-snapshot
  arrays; default: `False`
- **`complex`** — If `True`, return combined complex coefficients `coeff_q`; if `False`, return
  separate `coeff_re_q` and `coeff_im_q`; default: `False`
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`;
  mutually exclusive with `coeff_re` / `coeff_im`; default: `None`

## Outputs:
- **`coeff_q`** — Combined complex output coefficients; returned when `complex` is `True`; list of
  `(n_rx, n_tx, n_taps)` when `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)`
- **`coeff_re_q`** — Output coefficients, real part; returned when `complex` is `False`; list of
  `(n_rx, n_tx, n_taps)` when `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)`
- **`coeff_im_q`** — Output coefficients, imaginary part; returned when `complex` is `False`; same
  shape as `coeff_re_q`
- **`delay_q`** — Output delays in seconds; list of `(n_rx, n_tx, n_taps)` or `(1, 1, n_taps)` when
  `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)` or `(1, 1, n_taps, n_snap)` depending on
  `fix_taps`
MD!*/

py::tuple quantize_delays(const py::object &coeff_re,
                          const py::object &coeff_im,
                          const py::object &delay,
                          double tap_spacing,
                          arma::uword max_no_taps,
                          double power_exponent,
                          int fix_taps,
                          bool stack,
                          const bool complex,
                          const py::object &coeff)
{
    // Coefficient input mode: complex 'coeff' or split 'coeff_re'/'coeff_im'
    bool have_coeff = !coeff.is_none();
    bool have_cre = !coeff_re.is_none();
    bool have_cim = !coeff_im.is_none();

    if (have_coeff && (have_cre || have_cim))
        throw std::invalid_argument("Cannot provide both 'coeff' and 'coeff_re'/'coeff_im'.");
    if (have_cre != have_cim)
        throw std::invalid_argument("'coeff_re' and 'coeff_im' must both be provided.");
    if (!have_coeff && !have_cre)
        throw std::invalid_argument("Must provide either 'coeff' or both 'coeff_re' and 'coeff_im'.");
    if (delay.is_none())
        throw std::invalid_argument("'delay' must be provided.");

    bool use_complex = have_coeff;

    // Convert coefficient lists to std::vector<arma::Cube<double>> (length = n_snap)
    std::vector<arma::Cube<double>> coeff_re_a, coeff_im_a;
    if (use_complex)
    {
        if (!py::isinstance<py::list>(coeff))
            throw std::invalid_argument("'coeff' must be a list.");
        qd_python_list2vector_Cube_Cplx<double>(py::cast<py::list>(coeff), coeff_re_a, coeff_im_a);
    }
    else
    {
        if (!py::isinstance<py::list>(coeff_re) || !py::isinstance<py::list>(coeff_im))
            throw std::invalid_argument("'coeff_re' and 'coeff_im' must be lists.");
        coeff_re_a = qd_python_list2vector_Cube<double>(py::cast<py::list>(coeff_re));
        coeff_im_a = qd_python_list2vector_Cube<double>(py::cast<py::list>(coeff_im));
    }

    // Convert delay list (ragged path counts allowed)
    if (!py::isinstance<py::list>(delay))
        throw std::invalid_argument("'delay' must be a list.");
    auto delay_a = qd_python_list2vector_Cube<double>(py::cast<py::list>(delay));

    // Output vectors (size set by the C++ call)
    std::vector<arma::Cube<double>> coeff_re_q, coeff_im_q, delay_q;

    // Call library function
    quadriga_lib::quantize_delays<double>(&coeff_re_a, &coeff_im_a, &delay_a,
                                          &coeff_re_q, &coeff_im_q, &delay_q,
                                          tap_spacing, max_no_taps, power_exponent, fix_taps);

    // Delay output (always real); list (stack=False) or 4D array (stack=True)
    py::object delay_q_py;
    if (stack)
        delay_q_py = qd_python_stack2numpy(&delay_q);
    else
        delay_q_py = qd_python_copy2list(&delay_q);

    // Coefficient output form set by 'complex', independent of the input form
    if (complex)
    {
        py::object coeff_q_py;
        if (stack)
            coeff_q_py = qd_python_stack2numpy<double, std::complex<double>>(&coeff_re_q, &coeff_im_q);
        else
            coeff_q_py = qd_python_copy2list<arma::Cube<double>, std::complex<double>>(&coeff_re_q, &coeff_im_q);
        return py::make_tuple(coeff_q_py, delay_q_py);
    }

    py::object coeff_re_q_py, coeff_im_q_py;
    if (stack)
    {
        coeff_re_q_py = qd_python_stack2numpy(&coeff_re_q);
        coeff_im_q_py = qd_python_stack2numpy(&coeff_im_q);
    }
    else
    {
        coeff_re_q_py = qd_python_copy2list(&coeff_re_q);
        coeff_im_q_py = qd_python_copy2list(&coeff_im_q);
    }
    return py::make_tuple(coeff_re_q_py, coeff_im_q_py, delay_q_py);
}

// pybind11 declaration (channel submodule):
// m.def("quantize_delays", &quantize_delays,
//       py::arg("coeff_re") = py::none(),
//       py::arg("coeff_im") = py::none(),
//       py::arg("delay") = py::none(),
//       py::arg("tap_spacing") = 5e-9,
//       py::arg("max_no_taps") = 48,
//       py::arg("power_exponent") = 1.0,
//       py::arg("fix_taps") = 0,
//       py::arg("stack") = false,
//       py::arg("complex") = false,
//       py::arg("coeff") = py::none());