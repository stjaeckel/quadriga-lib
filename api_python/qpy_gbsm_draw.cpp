// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm.draw
Spatially correlated normalized random values of one random set at a set of links

- Evaluates the SOS phases of [[gbsm.init_phases]] at the links as [[sos.val]], one column per generator. The value of a
  link depends neither on the other links of the call nor on the other columns; with `dual_mobility = False` not on
  `tx_pos`
- `dual_mobility = True`: the Tx is a second device on the same fields; its side of each generator uses the mirror
  generator (e.g. departure and arrival quantities exchange), and every column keeps its distribution
- 2D mode (`horizontal = True`, TR): the values are correlated in the x-y plane; links with equal `floor_id` share one
  field, fields of different floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3):
  exactly for IDs in the same group of `n_coeff` consecutive IDs (0 to `n_coeff - 1`, ...), otherwise up to a random
  residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- 3D mode (legacy QuaDRiGa): the height separates the floors, so UTs on different floors are correlated by the ACF of
  their height difference. The per-link statistics are the same in both modes
- An empty set (`NumClusters = 1`) gives `(n_link, 0)` and accepts any empty `phase`

Random sets, with `N = NumClusters - 1` and `M = NumSubPaths`; a set never changes size (new random quantities come in
new sets), and a decorrelation distance of 0 is used as 0.1 m:

| `set_type`, `set_id` | `n_gen` | Columns | Decorrelation distance | Used by |
| --- | --- | --- | --- | --- |
| `"lsf"` | 11 | 0 to 7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR and 8 absolute TOA, Normal(0,1); 9 ground permittivity and 10 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (0 to 7), `absTOA_lambda` (8), `SC_lambda` (9), `SF_lambda` (10) | [[lsp_generate]] `rnd`, [[path_loss]] `rnd_he` (column 10) |
| `"cluster"`, -1 | `9·N` | column `b·N + n` for block `b` and NLOS cluster `n`: block 0 delay, Uniform(0,1); blocks 1 to 8 Normal(0,1): 1 to 4 sign of AOD, AOA, EOD, EOA, 5 to 8 offset of the same | `SC_lambda` | [[ssf_clusters]] `rnd` |
| `"cluster"`, `k` | `N` | cluster shadowing at frequency `k`, Normal(0,1) | `SC_lambda` | [[ssf_clusters]] `rnd_freq`, slice `k` |
| `"ray"`, -1 or `k` | `14·N·M + N` | blocks 0 to 14, below | `SC_lambda`; block 14 of set -1: 1.6·c·c_DS / `PerClusterAS_A` (in rad), within 0.1 m to 10 km | [[ssf_rays]] `rnd_ray`: -1 for `"quadriga"`, `k` as slice `k` for the TR methods |

A `"ray"` set holds block `b` up to 12 for ray `m` of NLOS cluster `n` in column `b·N·M + n·M + m`: 0 to 3 coupling
ranks of AOD, AOA, EOD, EOA; 4 XPR; 5 to 8 initial phases θθ, θφ, φθ, φφ; 9 to 12 polarization power variability of
the same. Block 13, the XPR per cluster, is in column `13·N·M + n`; block 14, the ray delay of `"quadriga"`, in column
`13·N·M + N + n·M + m`. All blocks are Normal(0,1) except block 14, Uniform(0,1).

## Usage:
```
rnd = quadriga_lib.gbsm.draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, center_freq, \
    dual_mobility, horizontal )
```

## Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, `PerClusterDS*` and
  `PerClusterAS_A`
- **`set_type`** — Set type of [[gbsm.init_phases]]; str
- **`phase`** — SOS phases from [[gbsm.init_phases]], finite; `(n_coeff, n_gen, 2)`
- **`tx_pos`** — Tx positions in m; used in dual mobility, always validated; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`, `n_link = 0` allowed
- **`set_id`** — Set of [[gbsm.init_phases]]: `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`;
  another set with the same `n_gen` is not detected; default: -1
- **`preset`** — Preset of [[gbsm.init_phases]]; another preset with the same `n_coeff` is not detected; default:
  `"Comb256"`
- **`floor_id`** — Floor number of the UT per link, ground floor 0 (a label; `n_fl - 1` for the UT heights of TR 38.901
  Table 7.2-1, 0 for outdoor UTs); int or float array; `None` or empty = floor 0; 2D mode only; `(n_link,)`;
  default: None
- **`center_freq`** — Carrier frequencies in Hz; required by the shared `"ray"` set, whose block 14 uses their mean
  c_DS ([[ssf_rays]]): pass the same list as to [[ssf_rays]] (another list gives other `"quadriga"` rays); ignored and
  not checked for all other sets; `(n_freq,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move; default: False
- **`horizontal`** — `True`: 2D mode; `False`: 3D mode; default: True

## Output:
- **`rnd`** — Random values; `(n_link, n_gen)`

## See also:
- [[gbsm.init_phases]] (the SOS phases)
- [[lsp_generate]] (the large-scale parameters from the `"lsf"` set)
- [[path_loss]] (column 10 of the `"lsf"` set as `rnd_he`)
- [[ssf_clusters]] (the clusters from the `"cluster"` sets)
- [[ssf_rays]] (the rays from the `"ray"` sets)
MD!*/

py::array_t<double> gbsm_draw(py::handle scenpar,
                              const std::string &set_type,
                              const py::array_t<double> &phase,
                              const py::array_t<double> &tx_pos,
                              const py::array_t<double> &rx_pos,
                              long long set_id,
                              const std::string &preset,
                              py::handle floor_id,
                              py::handle center_freq,
                              py::handle dual_mobility,
                              py::handle horizontal)
{
    // Read inputs
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto phase_a = qd_python_numpy2arma_Cube<double>(phase, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const arma::u32_vec floor_id_a = qd_python_u32_entries(floor_id, "floor_id");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const bool dual_mobility_b = qd_python_flag(dual_mobility, false, "dual_mobility");
    const bool horizontal_b = qd_python_flag(horizontal, true, "horizontal");

    // Wrap optional pointers
    const arma::u32_vec *p_floor_id = floor_id_a.is_empty() ? nullptr : &floor_id_a;
    const arma::vec *p_center_freq = center_freq.is_none() ? nullptr : &center_freq_a;

    // Output allocation in numpy memory: (n_link, n_gen), n_gen = columns of 'phase' (checked by the library)
    arma::mat rnd;
    auto rnd_py = qd_python_init_output(rx_pos_a.n_rows, phase_a.n_cols, &rnd);

    // Call library function
    quadriga_lib::gbsm_draw(scenpar_a, set_type, phase_a, tx_pos_a, rx_pos_a, rnd, (arma::sword)set_id, preset,
                            p_floor_id, p_center_freq, dual_mobility_b, horizontal_b);

    return rnd_py;
}

// pybind11 declaration:
// m.def("draw", &gbsm_draw,
//       py::arg("scenpar"),
//       py::arg("set_type"),
//       py::arg("phase").none(false),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("set_id") = -1,
//       py::arg("preset") = "Comb256",
//       py::arg("floor_id") = py::none(),
//       py::arg("center_freq") = py::none(),
//       py::arg("dual_mobility") = false,
//       py::arg("horizontal") = true);
