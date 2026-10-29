// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_init_phases
Draw the random phases of one small-scale fading generator set of a BS site and scenario

- The cluster random values of the small-scale fading (TR 38.901 clause 7.6.3.1) come in two kinds of sets, with
  `N = NumClusters - 1` NLOS clusters: one shared cluster set (`set_id = -1`, `9·N` generators) and one set per
  carrier frequency (`set_id = k`, `N` generators, `k` = index in the caller's frequency list, the same order in
  every call). [[ssf_draw]] evaluates the phases at the links and lists the columns
- The phases are a fixed function of `seed`, the seed of the large-scale set of the same site and scenario
  ([[lsf_init_phases]]), and `set_id`, equal on all platforms; the sets do not repeat the large-scale phases and are
  independent of each other. Keep an explicit seed (or the phase cubes) to reproduce the clusters
- Each scenario of a site (LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4 makes the link
  types uncorrelated
- `NumClusters = 1`: both sets are empty

## Usage:
```
phase = quadriga_lib.gbsm.ssf_init_phases( scenpar, seed, set_id, preset )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict from [[conf_file_read]] or [[scenpar_validate]] or any subset of its
  keys (missing keys take their defaults, `SubpathMethod` is ignored); `NumClusters` sets `N`
- **`seed`** — Seed of the large-scale set, int from -1 to 2^63-1; -1 = system random device; default: -1
- **`set_id`** — `-1` = shared cluster set; `k >= 0` = set of frequency `k`; default: -1
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; default: `"Comb256"`

## Output:
- **`phase`** — Phases in rad, in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side (mirror generators);
  `(n_coeff, n_gen, 2)`, `n_gen = 9·N` (shared set) or `N`

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_init_phases]] (the large-scale set and its seed)
MD!*/

py::array_t<double> ssf_init_phases(py::handle scenpar,
                                    long long seed,
                                    long long set_id,
                                    const std::string &preset)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");

    // Call library function; the size of 'phase' depends on 'scenpar' and 'preset', so it is copied afterwards
    arma::cube phase;
    quadriga_lib::ssf_init_phases(scenpar_a, phase, (arma::sword)seed, (arma::sword)set_id, preset);

    return qd_python_copy2numpy(&phase);
}

// pybind11 declaration:
// m.def("ssf_init_phases", &ssf_init_phases,
//       py::arg("scenpar"),
//       py::arg("seed") = -1,
//       py::arg("set_id") = -1,
//       py::arg("preset") = "Comb256");
