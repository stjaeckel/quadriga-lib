// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"
#include "quadriga_lib.hpp"

#include <limits>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS and UT heights, and,
  when `rnd_g` and `center_freq` are given, the soft LOS state of TR 38.901 eq. (7.6-18); the LOS state itself (a draw
  against `p_los`) and its spatial correlation are the caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office). Where the table leaves
  cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT > 23` m; InF-SH and InF-DH give 1 for
  `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS <= hUT`; SMa gives 1 for `hBS <= hUT` and uses
  `max(h_x - hUT, 0)` in all three terms (the TR only in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly interpolated; `Suburban` and
  `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`: 0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another `d_clutter` pass a
  matching correlation distance

## Usage:
```
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, \
    h_clutter, rnd_g, center_freq )
```

## Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in m, finite and >= 0: `d2D - d2d_in` for O2I UTs, clamped at 0 by the caller; the
  2D distance in indoor scenarios; `(n_link,)`
- **`h_bs`** — BS height in m, finite; float or `(n_link,)`
- **`h_ut`** — UT height in m, finite; float or `(n_link,)`
- **`d_clutter`** — Clutter size in m of the InF and SMa models, > 0; `None` or nan = TR default (InF-SL and InF-SH 10
  m, InF-DL and InF-DH 2 m, SMa 30 m); default: None
- **`r_clutter`** — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and DH), vegetation density
  of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1); `None` or nan = default; default: None
- **`h_clutter`** — Clutter height in m of InF-SH (default 2 m) and InF-DH (6 m), >= 0; `None` or nan = default. The
  other models ignore the clutter arguments, which are validated for every model; default: None
- **`rnd_g`** — Normal(0,1) value per link, G of eq. (7.6-18); `(n_link,)`; default: None
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0 (messages name it `center_frequency`); `(n_freq,)`;
  default: None

## Outputs:
- **`p_los`** — LOS probability in [0, 1]; `(n_link,)`
- **`los_soft`** — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq. (7.6-18), which has the
  probability `p_los`; `(n_link, n_freq)`; `None` if `rnd_g` or `center_freq` is `None`

## See also:
- [[o2i_loss]] (indoor distance for `d2d_out`)
- [[sos.rand]], [[sos.randn]] (spatially consistent LOS-state draw and `rnd_g`)
MD!*/

py::tuple los_probability(const std::string &model,
                          const py::array_t<double> &d2d_out,
                          const py::array_t<double> &h_bs,
                          const py::array_t<double> &h_ut,
                          py::handle d_clutter,
                          py::handle r_clutter,
                          py::handle h_clutter,
                          py::handle rnd_g,
                          py::handle center_freq)
{
    // Clutter arguments: None = NaN (TR default of the model), else a real scalar
    auto clutter = [](py::handle value, const std::string &name)
    {
        if (value.is_none())
            return std::numeric_limits<double>::quiet_NaN();
        const std::string error_msg = "Input '" + name + "' must be a real scalar or None.";
        const auto a = qd_python_real_array(value, error_msg);
        if (a.size() != 1)
            throw std::invalid_argument(error_msg);
        return *a.data();
    };
    const double d_clutter_d = clutter(d_clutter, "d_clutter"), r_clutter_d = clutter(r_clutter, "r_clutter"),
                 h_clutter_d = clutter(h_clutter, "h_clutter");

    // Read inputs
    const auto d2d_out_a = qd_python_numpy2arma_Col<double>(d2d_out, true);
    const auto h_bs_a = qd_python_numpy2arma_Col<double>(h_bs, true);
    const auto h_ut_a = qd_python_numpy2arma_Col<double>(h_ut, true);
    const auto rnd_g_a = qd_python_numpy2arma_Col<double>(rnd_g, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);

    // Output allocation; the soft LOS state needs both 'rnd_g' and 'center_freq', else it is returned as None
    const bool soft = !rnd_g.is_none() && !center_freq.is_none();
    arma::vec p_los;
    arma::mat los_soft;
    auto p_los_py = qd_python_init_output(d2d_out_a.n_elem, &p_los);
    py::object los_soft_py = py::none();
    if (soft)
        los_soft_py = qd_python_init_output(d2d_out_a.n_elem, center_freq_a.n_elem, &los_soft);

    // Wrap optional pointers
    const arma::vec *p_rnd_g = soft ? &rnd_g_a : nullptr;
    const arma::vec *p_center_freq = soft ? &center_freq_a : nullptr;
    arma::mat *p_los_soft = soft ? &los_soft : nullptr;

    // Call library function
    quadriga_lib::los_probability(model, d2d_out_a, h_bs_a, h_ut_a, p_los, d_clutter_d, r_clutter_d, h_clutter_d,
                                  p_rnd_g, p_center_freq, p_los_soft);

    return py::make_tuple(p_los_py, los_soft_py);
}

// pybind11 declaration:
// m.def("los_probability", &los_probability,
//       py::arg("model"),
//       py::arg("d2d_out").none(false),
//       py::arg("h_bs").none(false),
//       py::arg("h_ut").none(false),
//       py::arg("d_clutter") = py::none(),
//       py::arg("r_clutter") = py::none(),
//       py::arg("h_clutter") = py::none(),
//       py::arg("rnd_g") = py::none(),
//       py::arg("center_freq") = py::none());
